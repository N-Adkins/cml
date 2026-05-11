#include <cml/context.h>
#include <cml/tape.h>
#include <cml/tensor.h>
#include "context.h"
#include "tensor.h"
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

/* --- requires_grad accessors --- */

static void test_requires_grad_defaults_false(void) {
    cml_tensor_t *t = cml_tensor_init(ctx, 2, 2);
    TEST_ASSERT_FALSE(cml_tensor_requires_grad(t));
    TEST_ASSERT_NULL(cml_tensor_grad(t));
}

static void test_set_requires_grad(void) {
    cml_tensor_t *t = cml_tensor_init(ctx, 2, 2);
    cml_tensor_set_requires_grad(t, true);
    TEST_ASSERT_TRUE(cml_tensor_requires_grad(t));
}

static void test_requires_grad_null_safe(void) {
    TEST_ASSERT_FALSE(cml_tensor_requires_grad(NULL));
    TEST_ASSERT_NULL(cml_tensor_grad(NULL));
}

static void test_op_output_inherits_requires_grad(void) {
    cml_tensor_t *a = cml_tensor_init(ctx, 2, 2);
    cml_tensor_t *b = cml_tensor_init(ctx, 2, 2);
    cml_tensor_fill(a, 1.0f);
    cml_tensor_fill(b, 2.0f);
    cml_tensor_set_requires_grad(a, true);
    cml_tensor_t *c = cml_tensor_add(ctx, a, b);
    TEST_ASSERT_TRUE(cml_tensor_requires_grad(c));
    TEST_ASSERT_NOT_NULL(c->creator);
}

static void test_op_output_no_requires_grad_when_neither_input(void) {
    cml_tensor_t *a = cml_tensor_init(ctx, 2, 2);
    cml_tensor_t *b = cml_tensor_init(ctx, 2, 2);
    cml_tensor_fill(a, 1.0f);
    cml_tensor_fill(b, 2.0f);
    cml_tensor_t *c = cml_tensor_add(ctx, a, b);
    TEST_ASSERT_FALSE(cml_tensor_requires_grad(c));
    TEST_ASSERT_NULL(c->creator);
}

/* --- grad_enable / grad_disable --- */

static void test_grad_disable_suppresses_recording(void) {
    cml_tensor_t *a = cml_tensor_init(ctx, 2, 2);
    cml_tensor_fill(a, 1.0f);
    cml_tensor_set_requires_grad(a, true);
    cml_grad_disable(ctx);
    cml_tensor_t *b = cml_tensor_scale(ctx, a, 2.0f);
    TEST_ASSERT_FALSE(cml_tensor_requires_grad(b));
    TEST_ASSERT_NULL(b->creator);
    TEST_ASSERT_NULL(ctx->tape_head);
}

static void test_grad_enable_restores_recording(void) {
    cml_tensor_t *a = cml_tensor_init(ctx, 2, 2);
    cml_tensor_fill(a, 1.0f);
    cml_tensor_set_requires_grad(a, true);
    cml_grad_disable(ctx);
    cml_grad_enable(ctx);
    cml_tensor_t *b = cml_tensor_scale(ctx, a, 2.0f);
    TEST_ASSERT_TRUE(cml_tensor_requires_grad(b));
    TEST_ASSERT_NOT_NULL(b->creator);
}

/* --- backward: single ops --- */

static void test_backward_sum(void) {
    /* loss = sum(a); grad_a = ones */
    cml_tensor_t *a = cml_tensor_init(ctx, 2, 3);
    cml_tensor_fill(a, 1.0f);
    cml_tensor_set_requires_grad(a, true);
    cml_tensor_t *loss = cml_tensor_sum(ctx, a);
    cml_backward(ctx, loss);
    cml_tensor_t *ga = cml_tensor_grad(a);
    TEST_ASSERT_NOT_NULL(ga);
    for (size_t r = 0; r < 2; r++) {
        for (size_t c = 0; c < 3; c++) {
            TEST_ASSERT_FLOAT_WITHIN(DELTA, 1.0f, cml_tensor_get(ga, r, c));
        }
    }
}

static void test_backward_add(void) {
    /* loss = sum(a + b); grad_a = grad_b = ones */
    cml_tensor_t *a = cml_tensor_init(ctx, 2, 2);
    cml_tensor_t *b = cml_tensor_init(ctx, 2, 2);
    cml_tensor_fill(a, 1.0f);
    cml_tensor_fill(b, 2.0f);
    cml_tensor_set_requires_grad(a, true);
    cml_tensor_set_requires_grad(b, true);
    cml_tensor_t *loss = cml_tensor_sum(ctx, cml_tensor_add(ctx, a, b));
    cml_backward(ctx, loss);
    for (size_t r = 0; r < 2; r++) {
        for (size_t c = 0; c < 2; c++) {
            TEST_ASSERT_FLOAT_WITHIN(DELTA, 1.0f, cml_tensor_get(cml_tensor_grad(a), r, c));
            TEST_ASSERT_FLOAT_WITHIN(DELTA, 1.0f, cml_tensor_get(cml_tensor_grad(b), r, c));
        }
    }
}

static void test_backward_sub(void) {
    /* loss = sum(a - b); grad_a = +1, grad_b = -1 */
    cml_tensor_t *a = cml_tensor_init(ctx, 2, 2);
    cml_tensor_t *b = cml_tensor_init(ctx, 2, 2);
    cml_tensor_fill(a, 3.0f);
    cml_tensor_fill(b, 1.0f);
    cml_tensor_set_requires_grad(a, true);
    cml_tensor_set_requires_grad(b, true);
    cml_tensor_t *loss = cml_tensor_sum(ctx, cml_tensor_sub(ctx, a, b));
    cml_backward(ctx, loss);
    for (size_t r = 0; r < 2; r++) {
        for (size_t c = 0; c < 2; c++) {
            TEST_ASSERT_FLOAT_WITHIN(DELTA,  1.0f, cml_tensor_get(cml_tensor_grad(a), r, c));
            TEST_ASSERT_FLOAT_WITHIN(DELTA, -1.0f, cml_tensor_get(cml_tensor_grad(b), r, c));
        }
    }
}

static void test_backward_mul(void) {
    /* loss = sum(a * b); grad_a = b, grad_b = a */
    cml_tensor_t *a = cml_tensor_init(ctx, 1, 2);
    cml_tensor_t *b = cml_tensor_init(ctx, 1, 2);
    cml_tensor_set(a, 0, 0, 2.0f); cml_tensor_set(a, 0, 1, 3.0f);
    cml_tensor_set(b, 0, 0, 4.0f); cml_tensor_set(b, 0, 1, 5.0f);
    cml_tensor_set_requires_grad(a, true);
    cml_tensor_set_requires_grad(b, true);
    cml_tensor_t *loss = cml_tensor_sum(ctx, cml_tensor_mul(ctx, a, b));
    cml_backward(ctx, loss);
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 4.0f, cml_tensor_get(cml_tensor_grad(a), 0, 0));
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 5.0f, cml_tensor_get(cml_tensor_grad(a), 0, 1));
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 2.0f, cml_tensor_get(cml_tensor_grad(b), 0, 0));
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 3.0f, cml_tensor_get(cml_tensor_grad(b), 0, 1));
}

static void test_backward_scale(void) {
    /* loss = sum(3 * a); grad_a = 3 * ones */
    cml_tensor_t *a = cml_tensor_init(ctx, 2, 2);
    cml_tensor_fill(a, 1.0f);
    cml_tensor_set_requires_grad(a, true);
    cml_tensor_t *loss = cml_tensor_sum(ctx, cml_tensor_scale(ctx, a, 3.0f));
    cml_backward(ctx, loss);
    for (size_t r = 0; r < 2; r++)
        for (size_t c = 0; c < 2; c++)
            TEST_ASSERT_FLOAT_WITHIN(DELTA, 3.0f, cml_tensor_get(cml_tensor_grad(a), r, c));
}

static void test_backward_dot(void) {
    /* a = [[1,2],[3,4]], b = [[5,6],[7,8]]
     * loss = sum(a @ b)
     * grad_a = ones @ b^T = [[11,15],[11,15]]
     * grad_b = a^T @ ones = [[4,4],[6,6]]            */
    cml_tensor_t *a = cml_tensor_init(ctx, 2, 2);
    cml_tensor_t *b = cml_tensor_init(ctx, 2, 2);
    cml_tensor_set(a, 0, 0, 1.0f); cml_tensor_set(a, 0, 1, 2.0f);
    cml_tensor_set(a, 1, 0, 3.0f); cml_tensor_set(a, 1, 1, 4.0f);
    cml_tensor_set(b, 0, 0, 5.0f); cml_tensor_set(b, 0, 1, 6.0f);
    cml_tensor_set(b, 1, 0, 7.0f); cml_tensor_set(b, 1, 1, 8.0f);
    cml_tensor_set_requires_grad(a, true);
    cml_tensor_set_requires_grad(b, true);
    cml_tensor_t *loss = cml_tensor_sum(ctx, cml_tensor_dot(ctx, a, b));
    cml_backward(ctx, loss);
    cml_tensor_t *ga = cml_tensor_grad(a);
    cml_tensor_t *gb = cml_tensor_grad(b);
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 11.0f, cml_tensor_get(ga, 0, 0));
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 15.0f, cml_tensor_get(ga, 0, 1));
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 11.0f, cml_tensor_get(ga, 1, 0));
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 15.0f, cml_tensor_get(ga, 1, 1));
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 4.0f,  cml_tensor_get(gb, 0, 0));
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 4.0f,  cml_tensor_get(gb, 0, 1));
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 6.0f,  cml_tensor_get(gb, 1, 0));
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 6.0f,  cml_tensor_get(gb, 1, 1));
}

static void test_backward_transpose(void) {
    /* b = a^T; loss = sum(b); grad_a = ones (same shape as a) */
    cml_tensor_t *a = cml_tensor_init(ctx, 2, 3);
    cml_tensor_fill(a, 1.0f);
    cml_tensor_set_requires_grad(a, true);
    cml_tensor_t *loss = cml_tensor_sum(ctx, cml_tensor_transpose(ctx, a));
    cml_backward(ctx, loss);
    cml_tensor_t *ga = cml_tensor_grad(a);
    TEST_ASSERT_NOT_NULL(ga);
    TEST_ASSERT_EQUAL_size_t(2, cml_tensor_rows(ga));
    TEST_ASSERT_EQUAL_size_t(3, cml_tensor_cols(ga));
    for (size_t r = 0; r < 2; r++) {
        for (size_t c = 0; c < 3; c++) {
            TEST_ASSERT_FLOAT_WITHIN(DELTA, 1.0f, cml_tensor_get(ga, r, c));
        }
    }
}

static void test_backward_sigmoid(void) {
    /* a = [[0.0]]; σ(0) = 0.5; grad_a = 1 * 0.5 * 0.5 = 0.25 */
    cml_tensor_t *a = cml_tensor_init(ctx, 1, 1);
    cml_tensor_set(a, 0, 0, 0.0f);
    cml_tensor_set_requires_grad(a, true);
    cml_tensor_t *loss = cml_tensor_sum(ctx, cml_tensor_sigmoid(ctx, a));
    cml_backward(ctx, loss);
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 0.25f, cml_tensor_get(cml_tensor_grad(a), 0, 0));
}

static void test_backward_relu(void) {
    /* a = [[-1, 2]]; relu = [[0, 2]]; grad_a = [[0, 1]] */
    cml_tensor_t *a = cml_tensor_init(ctx, 1, 2);
    cml_tensor_set(a, 0, 0, -1.0f);
    cml_tensor_set(a, 0, 1,  2.0f);
    cml_tensor_set_requires_grad(a, true);
    cml_tensor_t *loss = cml_tensor_sum(ctx, cml_tensor_relu(ctx, a));
    cml_backward(ctx, loss);
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 0.0f, cml_tensor_get(cml_tensor_grad(a), 0, 0));
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 1.0f, cml_tensor_get(cml_tensor_grad(a), 0, 1));
}

static void test_backward_relu_all_negative(void) {
    cml_tensor_t *a = cml_tensor_init(ctx, 1, 3);
    cml_tensor_fill(a, -5.0f);
    cml_tensor_set_requires_grad(a, true);
    cml_tensor_t *loss = cml_tensor_sum(ctx, cml_tensor_relu(ctx, a));
    cml_backward(ctx, loss);
    for (size_t c = 0; c < 3; c++) {
        TEST_ASSERT_FLOAT_WITHIN(DELTA, 0.0f, cml_tensor_get(cml_tensor_grad(a), 0, c));
    }
}

/* --- chain rule --- */

static void test_chain_rule_scale_then_sum(void) {
    /* loss = sum(2 * a); grad_a = 2 * ones */
    cml_tensor_t *a = cml_tensor_init(ctx, 1, 3);
    cml_tensor_fill(a, 1.0f);
    cml_tensor_set_requires_grad(a, true);
    cml_tensor_t *loss = cml_tensor_sum(ctx, cml_tensor_scale(ctx, a, 2.0f));
    cml_backward(ctx, loss);
    for (size_t c = 0; c < 3; c++) {
        TEST_ASSERT_FLOAT_WITHIN(DELTA, 2.0f, cml_tensor_get(cml_tensor_grad(a), 0, c));
    }
}

static void test_chain_rule_relu_then_scale(void) {
    /* a = [[-1, 2, -3]]; b = relu(a) = [[0, 2, 0]]
     * c = 2 * b = [[0, 4, 0]]; loss = sum(c) = 4
     * grad_a = 2 * (a > 0) = [[0, 2, 0]]         */
    cml_tensor_t *a = cml_tensor_init(ctx, 1, 3);
    cml_tensor_set(a, 0, 0, -1.0f);
    cml_tensor_set(a, 0, 1,  2.0f);
    cml_tensor_set(a, 0, 2, -3.0f);
    cml_tensor_set_requires_grad(a, true);
    cml_tensor_t *b = cml_tensor_relu(ctx, a);
    cml_tensor_t *loss = cml_tensor_sum(ctx, cml_tensor_scale(ctx, b, 2.0f));
    cml_backward(ctx, loss);
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 0.0f, cml_tensor_get(cml_tensor_grad(a), 0, 0));
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 2.0f, cml_tensor_get(cml_tensor_grad(a), 0, 1));
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 0.0f, cml_tensor_get(cml_tensor_grad(a), 0, 2));
}

static void test_chain_rule_two_inputs(void) {
    /* loss = sum(a + b) * 2 via scale; grad_a = grad_b = 2 */
    cml_tensor_t *a = cml_tensor_init(ctx, 2, 2);
    cml_tensor_t *b = cml_tensor_init(ctx, 2, 2);
    cml_tensor_fill(a, 1.0f);
    cml_tensor_fill(b, 1.0f);
    cml_tensor_set_requires_grad(a, true);
    cml_tensor_set_requires_grad(b, true);
    cml_tensor_t *loss = cml_tensor_sum(ctx,
                            cml_tensor_scale(ctx, cml_tensor_add(ctx, a, b), 2.0f));
    cml_backward(ctx, loss);
    for (size_t r = 0; r < 2; r++) {
        for (size_t c = 0; c < 2; c++) {
            TEST_ASSERT_FLOAT_WITHIN(DELTA, 2.0f, cml_tensor_get(cml_tensor_grad(a), r, c));
            TEST_ASSERT_FLOAT_WITHIN(DELTA, 2.0f, cml_tensor_get(cml_tensor_grad(b), r, c));
        }
    }
}

/* --- gradient accumulation --- */

static void test_grad_accumulates_when_tensor_used_twice(void) {
    /* c = a + a; loss = sum(c); grad_a should be 2 * ones */
    cml_tensor_t *a = cml_tensor_init(ctx, 1, 3);
    cml_tensor_fill(a, 1.0f);
    cml_tensor_set_requires_grad(a, true);
    cml_tensor_t *loss = cml_tensor_sum(ctx, cml_tensor_add(ctx, a, a));
    cml_backward(ctx, loss);
    for (size_t c = 0; c < 3; c++) {
        TEST_ASSERT_FLOAT_WITHIN(DELTA, 2.0f, cml_tensor_get(cml_tensor_grad(a), 0, c));
    }
}

/* --- no grad for non-requires_grad tensors --- */

static void test_no_grad_computed_without_requires_grad(void) {
    cml_tensor_t *a = cml_tensor_init(ctx, 2, 2);
    cml_tensor_t *b = cml_tensor_init(ctx, 2, 2);
    cml_tensor_fill(a, 1.0f);
    cml_tensor_fill(b, 2.0f);
    cml_tensor_t *loss = cml_tensor_sum(ctx, cml_tensor_add(ctx, a, b));
    cml_backward(ctx, loss);
    TEST_ASSERT_NULL(cml_tensor_grad(a));
    TEST_ASSERT_NULL(cml_tensor_grad(b));
}

static void test_only_requires_grad_inputs_get_grad(void) {
    cml_tensor_t *a = cml_tensor_init(ctx, 2, 2);
    cml_tensor_t *b = cml_tensor_init(ctx, 2, 2);
    cml_tensor_fill(a, 1.0f);
    cml_tensor_fill(b, 2.0f);
    cml_tensor_set_requires_grad(a, true);
    cml_tensor_t *loss = cml_tensor_sum(ctx, cml_tensor_add(ctx, a, b));
    cml_backward(ctx, loss);
    TEST_ASSERT_NOT_NULL(cml_tensor_grad(a));
    TEST_ASSERT_NULL(cml_tensor_grad(b));
}

/* --- cml_tape_clear --- */

static void test_tape_clear_nulls_leaf_grads(void) {
    cml_tensor_t *a = cml_tensor_init(ctx, 2, 2);
    cml_tensor_fill(a, 1.0f);
    cml_tensor_set_requires_grad(a, true);
    cml_tensor_t *loss = cml_tensor_sum(ctx, cml_tensor_scale(ctx, a, 2.0f));
    cml_backward(ctx, loss);
    TEST_ASSERT_NOT_NULL(cml_tensor_grad(a));
    cml_tape_clear(ctx);
    TEST_ASSERT_NULL(cml_tensor_grad(a));
}

static void test_tape_clear_preserves_leaf_requires_grad(void) {
    cml_tensor_t *a = cml_tensor_init(ctx, 2, 2);
    cml_tensor_fill(a, 1.0f);
    cml_tensor_set_requires_grad(a, true);
    cml_tensor_t *loss = cml_tensor_sum(ctx, cml_tensor_scale(ctx, a, 2.0f));
    cml_backward(ctx, loss);
    cml_tape_clear(ctx);
    TEST_ASSERT_TRUE(cml_tensor_requires_grad(a));
}

static void test_tape_clear_resets_tape_head(void) {
    cml_tensor_t *a = cml_tensor_init(ctx, 2, 2);
    cml_tensor_fill(a, 1.0f);
    cml_tensor_set_requires_grad(a, true);
    cml_tensor_sum(ctx, cml_tensor_scale(ctx, a, 2.0f));
    TEST_ASSERT_NOT_NULL(ctx->tape_head);
    cml_tape_clear(ctx);
    TEST_ASSERT_NULL(ctx->tape_head);
}

static void test_tape_clear_allows_fresh_backward(void) {
    cml_tensor_t *a = cml_tensor_init(ctx, 1, 2);
    cml_tensor_set(a, 0, 0, 1.0f);
    cml_tensor_set(a, 0, 1, 2.0f);
    cml_tensor_set_requires_grad(a, true);

    /* first pass */
    cml_tensor_t *loss = cml_tensor_sum(ctx, cml_tensor_scale(ctx, a, 3.0f));
    cml_backward(ctx, loss);
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 3.0f, cml_tensor_get(cml_tensor_grad(a), 0, 0));
    cml_tape_clear(ctx);

    /* second pass — same parameter, fresh grad */
    loss = cml_tensor_sum(ctx, cml_tensor_scale(ctx, a, 5.0f));
    cml_backward(ctx, loss);
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 5.0f, cml_tensor_get(cml_tensor_grad(a), 0, 0));
    cml_tape_clear(ctx);
}

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_requires_grad_defaults_false);
    RUN_TEST(test_set_requires_grad);
    RUN_TEST(test_requires_grad_null_safe);
    RUN_TEST(test_op_output_inherits_requires_grad);
    RUN_TEST(test_op_output_no_requires_grad_when_neither_input);

    RUN_TEST(test_grad_disable_suppresses_recording);
    RUN_TEST(test_grad_enable_restores_recording);

    RUN_TEST(test_backward_sum);
    RUN_TEST(test_backward_add);
    RUN_TEST(test_backward_sub);
    RUN_TEST(test_backward_mul);
    RUN_TEST(test_backward_scale);
    RUN_TEST(test_backward_dot);
    RUN_TEST(test_backward_transpose);
    RUN_TEST(test_backward_sigmoid);
    RUN_TEST(test_backward_relu);
    RUN_TEST(test_backward_relu_all_negative);

    RUN_TEST(test_chain_rule_scale_then_sum);
    RUN_TEST(test_chain_rule_relu_then_scale);
    RUN_TEST(test_chain_rule_two_inputs);

    RUN_TEST(test_grad_accumulates_when_tensor_used_twice);

    RUN_TEST(test_no_grad_computed_without_requires_grad);
    RUN_TEST(test_only_requires_grad_inputs_get_grad);

    RUN_TEST(test_tape_clear_nulls_leaf_grads);
    RUN_TEST(test_tape_clear_preserves_leaf_requires_grad);
    RUN_TEST(test_tape_clear_resets_tape_head);
    RUN_TEST(test_tape_clear_allows_fresh_backward);

    return UNITY_END();
}
