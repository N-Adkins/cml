#include <cml/context.h>
#include <cml/nn.h>
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

static void test_nn_linear_module_init_not_null(void) {
    cml_module_t *m = cml_nn_linear(ctx, 3, 2);
    TEST_ASSERT_NOT_NULL(m);
    TEST_ASSERT_EQUAL(CML_OK, cml_get_status(ctx));
}

static void test_nn_relu_module_init_not_null(void) {
    cml_module_t *m = cml_nn_relu(ctx);
    TEST_ASSERT_NOT_NULL(m);
    TEST_ASSERT_EQUAL(CML_OK, cml_get_status(ctx));
}

static void test_nn_sequential_init_not_null(void) {
    cml_module_t *l = cml_nn_linear(ctx, 2, 2);
    cml_module_t *r = cml_nn_relu(ctx);
    cml_module_t *children[] = { l, r };
    cml_module_t *seq = cml_nn_sequential(ctx, children, 2);
    TEST_ASSERT_NOT_NULL(seq);
    TEST_ASSERT_EQUAL(CML_OK, cml_get_status(ctx));
}

static void test_nn_sequential_null_child_errors(void) {
    cml_module_t *children[] = { NULL };
    TEST_ASSERT_NULL(cml_nn_sequential(ctx, children, 1));
    TEST_ASSERT_EQUAL(CML_INVALID_ARG, cml_get_status(ctx));
}

static void test_nn_module_forward_linear_known_values(void) {
    cml_module_t *linear = cml_nn_linear(ctx, 2, 2);
    size_t n_params = cml_module_param_count(linear);
    cml_tensor_t *params[2];
    TEST_ASSERT_EQUAL_size_t(2, n_params);
    TEST_ASSERT_EQUAL_size_t(2, cml_module_collect_params(linear, params, 2, 0));

    cml_tensor_t *w = params[0];
    cml_tensor_t *b = params[1];

    cml_tensor_zero(w);
    cml_tensor_set(w, 0, 0, 1.0f);
    cml_tensor_set(w, 1, 1, 1.0f);
    cml_tensor_set(b, 0, 0, 5.0f);
    cml_tensor_set(b, 0, 1, 10.0f);

    cml_tensor_t *x = cml_tensor_init(ctx, 1, 2);
    cml_tensor_set(x, 0, 0, 3.0f);
    cml_tensor_set(x, 0, 1, 4.0f);

    cml_tensor_t *out = cml_module_forward(ctx, linear, x);
    TEST_ASSERT_NOT_NULL(out);
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 8.0f, cml_tensor_get(out, 0, 0));
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 14.0f, cml_tensor_get(out, 0, 1));
}

static void test_nn_module_forward_relu(void) {
    cml_module_t *relu = cml_nn_relu(ctx);
    cml_tensor_t *x = cml_tensor_init(ctx, 1, 3);
    cml_tensor_set(x, 0, 0, -2.0f);
    cml_tensor_set(x, 0, 1, 0.5f);
    cml_tensor_set(x, 0, 2, 3.0f);

    cml_tensor_t *out = cml_module_forward(ctx, relu, x);
    TEST_ASSERT_NOT_NULL(out);
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 0.0f, cml_tensor_get(out, 0, 0));
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 0.5f, cml_tensor_get(out, 0, 1));
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 3.0f, cml_tensor_get(out, 0, 2));
}

static void test_nn_module_forward_sequential_known_values(void) {
    cml_module_t *linear = cml_nn_linear(ctx, 2, 2);
    cml_module_t *relu = cml_nn_relu(ctx);
    cml_module_t *children[] = { linear, relu };
    cml_module_t *seq = cml_nn_sequential(ctx, children, 2);

    cml_tensor_t *params[2];
    TEST_ASSERT_EQUAL_size_t(2, cml_module_collect_params(linear, params, 2, 0));
    cml_tensor_t *w = params[0];
    cml_tensor_t *b = params[1];

    cml_tensor_zero(w);
    cml_tensor_set(w, 0, 0, 1.0f);
    cml_tensor_set(w, 1, 1, 1.0f);
    cml_tensor_set(b, 0, 0, -1.0f);
    cml_tensor_set(b, 0, 1, 1.0f);

    cml_tensor_t *x = cml_tensor_init(ctx, 1, 2);
    cml_tensor_set(x, 0, 0, 2.0f);
    cml_tensor_set(x, 0, 1, -3.0f);

    cml_tensor_t *out = cml_module_forward(ctx, seq, x);
    TEST_ASSERT_NOT_NULL(out);
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 1.0f, cml_tensor_get(out, 0, 0));
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 0.0f, cml_tensor_get(out, 0, 1));
}

static void test_nn_module_param_count_and_collect_nested(void) {
    cml_module_t *l1 = cml_nn_linear(ctx, 3, 4);
    cml_module_t *r = cml_nn_relu(ctx);
    cml_module_t *l2 = cml_nn_linear(ctx, 4, 1);
    cml_module_t *children[] = { l1, r, l2 };
    cml_module_t *seq = cml_nn_sequential(ctx, children, 3);

    TEST_ASSERT_EQUAL_size_t(4, cml_module_param_count(seq));

    cml_tensor_t *params[4];
    size_t end = cml_module_collect_params(seq, params, 4, 0);
    TEST_ASSERT_EQUAL_size_t(4, end);

    TEST_ASSERT_EQUAL_size_t(3, cml_tensor_rows(params[0]));
    TEST_ASSERT_EQUAL_size_t(4, cml_tensor_cols(params[0]));
    TEST_ASSERT_EQUAL_size_t(1, cml_tensor_rows(params[1]));
    TEST_ASSERT_EQUAL_size_t(4, cml_tensor_cols(params[1]));
    TEST_ASSERT_EQUAL_size_t(4, cml_tensor_rows(params[2]));
    TEST_ASSERT_EQUAL_size_t(1, cml_tensor_cols(params[2]));
    TEST_ASSERT_EQUAL_size_t(1, cml_tensor_rows(params[3]));
    TEST_ASSERT_EQUAL_size_t(1, cml_tensor_cols(params[3]));
}

static void test_nn_module_forward_null_module_errors(void) {
    cml_tensor_t *x = cml_tensor_init(ctx, 1, 1);
    TEST_ASSERT_NULL(cml_module_forward(ctx, NULL, x));
    TEST_ASSERT_EQUAL(CML_INVALID_ARG, cml_get_status(ctx));
}

static void test_nn_backward_through_sequential(void) {
    cml_module_t *linear = cml_nn_linear(ctx, 2, 1);
    cml_module_t *relu = cml_nn_relu(ctx);
    cml_module_t *children[] = { linear, relu };
    cml_module_t *seq = cml_nn_sequential(ctx, children, 2);

    cml_tensor_t *params[2];
    TEST_ASSERT_EQUAL_size_t(2, cml_module_collect_params(linear, params, 2, 0));
    cml_tensor_t *w = params[0];
    cml_tensor_t *b = params[1];

    cml_tensor_set(w, 0, 0, 2.0f);
    cml_tensor_set(w, 1, 0, 3.0f);
    cml_tensor_set(b, 0, 0, 0.0f);

    cml_tensor_t *x = cml_tensor_init(ctx, 1, 2);
    cml_tensor_set(x, 0, 0, 1.0f);
    cml_tensor_set(x, 0, 1, 1.0f);

    cml_tensor_t *loss = cml_tensor_sum(ctx, cml_module_forward(ctx, seq, x));
    cml_backward(ctx, loss);

    cml_tensor_t *gw = cml_tensor_grad(w);
    cml_tensor_t *gb = cml_tensor_grad(b);
    TEST_ASSERT_NOT_NULL(gw);
    TEST_ASSERT_NOT_NULL(gb);
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 1.0f, cml_tensor_get(gw, 0, 0));
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 1.0f, cml_tensor_get(gw, 1, 0));
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 1.0f, cml_tensor_get(gb, 0, 0));
}

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_nn_linear_module_init_not_null);
    RUN_TEST(test_nn_relu_module_init_not_null);
    RUN_TEST(test_nn_sequential_init_not_null);
    RUN_TEST(test_nn_sequential_null_child_errors);
    RUN_TEST(test_nn_module_forward_linear_known_values);
    RUN_TEST(test_nn_module_forward_relu);
    RUN_TEST(test_nn_module_forward_sequential_known_values);
    RUN_TEST(test_nn_module_param_count_and_collect_nested);
    RUN_TEST(test_nn_module_forward_null_module_errors);
    RUN_TEST(test_nn_backward_through_sequential);

    return UNITY_END();
}
