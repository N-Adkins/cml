#include <cml/context.h>
#include <cml/linear.h>
#include <cml/tape.h>
#include <cml/tensor.h>
#include "unity.h"

#define CTX_SIZE (4 * 1024 * 1024)
#define DELTA 1e-4f

static cml_context_t *ctx;

void setUp(void) {
    ctx = cml_init(CTX_SIZE);
}

void tearDown(void) {
    cml_deinit(ctx);
}

/* --- init --- */

static void test_linear_init_not_null(void) {
    cml_linear_t *l = cml_linear_init(ctx, 4, 3);
    TEST_ASSERT_NOT_NULL(l);
    TEST_ASSERT_EQUAL(CML_OK, cml_get_status(ctx));
}

static void test_linear_null_ctx_returns_null(void) {
    TEST_ASSERT_NULL(cml_linear_init(NULL, 4, 3));
}

static void test_linear_zero_features_errors(void) {
    TEST_ASSERT_NULL(cml_linear_init(ctx, 0, 3));
    TEST_ASSERT_EQUAL(CML_INVALID_ARG, cml_get_status(ctx));
}

static void test_linear_weight_shape(void) {
    cml_linear_t *l = cml_linear_init(ctx, 4, 3);
    cml_tensor_t *w = cml_linear_weight(l);
    TEST_ASSERT_NOT_NULL(w);
    TEST_ASSERT_EQUAL_size_t(4, cml_tensor_rows(w));
    TEST_ASSERT_EQUAL_size_t(3, cml_tensor_cols(w));
}

static void test_linear_bias_shape(void) {
    cml_linear_t *l = cml_linear_init(ctx, 4, 3);
    cml_tensor_t *b = cml_linear_bias(l);
    TEST_ASSERT_NOT_NULL(b);
    TEST_ASSERT_EQUAL_size_t(1, cml_tensor_rows(b));
    TEST_ASSERT_EQUAL_size_t(3, cml_tensor_cols(b));
}

static void test_linear_weight_requires_grad(void) {
    cml_linear_t *l = cml_linear_init(ctx, 4, 3);
    TEST_ASSERT_TRUE(cml_tensor_requires_grad(cml_linear_weight(l)));
}

static void test_linear_bias_requires_grad(void) {
    cml_linear_t *l = cml_linear_init(ctx, 4, 3);
    TEST_ASSERT_TRUE(cml_tensor_requires_grad(cml_linear_bias(l)));
}

static void test_linear_bias_initialized_zero(void) {
    cml_linear_t *l = cml_linear_init(ctx, 4, 3);
    cml_tensor_t *b = cml_linear_bias(l);
    for (size_t c = 0; c < 3; c++) {
        TEST_ASSERT_EQUAL_FLOAT(0.0f, cml_tensor_get(b, 0, c));
    }
}

static void test_linear_weight_in_xavier_range(void) {
    /* limit = sqrt(6 / (in + out)) = sqrt(6/7) ≈ 0.9258 */
    cml_linear_t *l = cml_linear_init(ctx, 4, 3);
    cml_tensor_t *w = cml_linear_weight(l);
    float limit = 0.9259f;
    for (size_t r = 0; r < 4; r++) {
        for (size_t c = 0; c < 3; c++) {
            float v = cml_tensor_get(w, r, c);
            TEST_ASSERT_GREATER_OR_EQUAL_FLOAT(-limit, v);
            TEST_ASSERT_LESS_OR_EQUAL_FLOAT(limit, v);
        }
    }
}

/* --- accessors --- */

static void test_linear_accessor_null_safety(void) {
    TEST_ASSERT_NULL(cml_linear_weight(NULL));
    TEST_ASSERT_NULL(cml_linear_bias(NULL));
}

/* --- forward --- */

static void test_linear_forward_output_shape(void) {
    cml_linear_t *l = cml_linear_init(ctx, 4, 3);
    cml_tensor_t *x = cml_tensor_init(ctx, 2, 4);
    cml_tensor_fill(x, 1.0f);
    cml_tensor_t *out = cml_linear_forward(ctx, l, x);
    TEST_ASSERT_NOT_NULL(out);
    TEST_ASSERT_EQUAL_size_t(2, cml_tensor_rows(out));
    TEST_ASSERT_EQUAL_size_t(3, cml_tensor_cols(out));
}

static void test_linear_forward_known_values(void) {
    /* w = identity [[1,0],[0,1]], b = [[5, 10]], x = [[3, 4]]
     * out = dot(x, w) + b = [[3, 4]] + [[5, 10]] = [[8, 14]] */
    cml_linear_t *l = cml_linear_init(ctx, 2, 2);
    cml_tensor_t *w = cml_linear_weight(l);
    cml_tensor_t *b = cml_linear_bias(l);

    cml_tensor_zero(w);
    cml_tensor_set(w, 0, 0, 1.0f);
    cml_tensor_set(w, 1, 1, 1.0f);
    cml_tensor_set(b, 0, 0, 5.0f);
    cml_tensor_set(b, 0, 1, 10.0f);

    cml_tensor_t *x = cml_tensor_init(ctx, 1, 2);
    cml_tensor_set(x, 0, 0, 3.0f);
    cml_tensor_set(x, 0, 1, 4.0f);

    cml_tensor_t *out = cml_linear_forward(ctx, l, x);
    TEST_ASSERT_NOT_NULL(out);
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 8.0f,  cml_tensor_get(out, 0, 0));
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 14.0f, cml_tensor_get(out, 0, 1));
}

static void test_linear_forward_null_layer_errors(void) {
    cml_tensor_t *x = cml_tensor_init(ctx, 1, 2);
    TEST_ASSERT_NULL(cml_linear_forward(ctx, NULL, x));
    TEST_ASSERT_EQUAL(CML_INVALID_ARG, cml_get_status(ctx));
}

static void test_linear_forward_shape_mismatch_errors(void) {
    cml_linear_t *l = cml_linear_init(ctx, 4, 3);
    cml_tensor_t *x = cml_tensor_init(ctx, 1, 2); /* wrong: expects 4 cols */
    cml_tensor_fill(x, 1.0f);
    TEST_ASSERT_NULL(cml_linear_forward(ctx, l, x));
    TEST_ASSERT_EQUAL(CML_INVALID_ARG, cml_get_status(ctx));
}

/* --- backward --- */

static void test_linear_backward_weight_grad(void) {
    /* w = [[2],[3]], b = [[0]], x = [[1,1]] (batch=1)
     * out = dot(x,w) + b = [[5]]; loss = sum = 5
     * grad_w = x^T @ grad_z = [[1],[1]]              */
    cml_linear_t *l = cml_linear_init(ctx, 2, 1);
    cml_tensor_t *w = cml_linear_weight(l);
    cml_tensor_set(w, 0, 0, 2.0f);
    cml_tensor_set(w, 1, 0, 3.0f);

    cml_tensor_t *x = cml_tensor_init(ctx, 1, 2);
    cml_tensor_set(x, 0, 0, 1.0f);
    cml_tensor_set(x, 0, 1, 1.0f);

    cml_tensor_t *loss = cml_tensor_sum(ctx, cml_linear_forward(ctx, l, x));
    cml_backward(ctx, loss);

    cml_tensor_t *gw = cml_tensor_grad(w);
    TEST_ASSERT_NOT_NULL(gw);
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 1.0f, cml_tensor_get(gw, 0, 0));
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 1.0f, cml_tensor_get(gw, 1, 0));
}

static void test_linear_backward_bias_grad(void) {
    /* x = [[1,1],[2,2]] (batch=2), w = [[1],[1]], b = [[0]]
     * out = [[2],[4]]; loss = sum = 6
     * grad_b = sum_rows(grad_out) = [[1+1]] = [[2]]          */
    cml_linear_t *l = cml_linear_init(ctx, 2, 1);
    cml_tensor_t *w = cml_linear_weight(l);
    cml_tensor_set(w, 0, 0, 1.0f);
    cml_tensor_set(w, 1, 0, 1.0f);

    cml_tensor_t *x = cml_tensor_init(ctx, 2, 2);
    cml_tensor_set(x, 0, 0, 1.0f); cml_tensor_set(x, 0, 1, 1.0f);
    cml_tensor_set(x, 1, 0, 2.0f); cml_tensor_set(x, 1, 1, 2.0f);

    cml_tensor_t *loss = cml_tensor_sum(ctx, cml_linear_forward(ctx, l, x));
    cml_backward(ctx, loss);

    cml_tensor_t *gb = cml_tensor_grad(cml_linear_bias(l));
    TEST_ASSERT_NOT_NULL(gb);
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 2.0f, cml_tensor_get(gb, 0, 0));
}

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_linear_init_not_null);
    RUN_TEST(test_linear_null_ctx_returns_null);
    RUN_TEST(test_linear_zero_features_errors);
    RUN_TEST(test_linear_weight_shape);
    RUN_TEST(test_linear_bias_shape);
    RUN_TEST(test_linear_weight_requires_grad);
    RUN_TEST(test_linear_bias_requires_grad);
    RUN_TEST(test_linear_bias_initialized_zero);
    RUN_TEST(test_linear_weight_in_xavier_range);

    RUN_TEST(test_linear_accessor_null_safety);

    RUN_TEST(test_linear_forward_output_shape);
    RUN_TEST(test_linear_forward_known_values);
    RUN_TEST(test_linear_forward_null_layer_errors);
    RUN_TEST(test_linear_forward_shape_mismatch_errors);

    RUN_TEST(test_linear_backward_weight_grad);
    RUN_TEST(test_linear_backward_bias_grad);

    return UNITY_END();
}
