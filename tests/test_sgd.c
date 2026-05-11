#include <cml/context.h>
#include <cml/loss.h>
#include <cml/sgd.h>
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

static void test_sgd_init_not_null(void) {
    cml_sgd_t *opt = cml_sgd_init(ctx, 0.01f);
    TEST_ASSERT_NOT_NULL(opt);
    TEST_ASSERT_EQUAL(CML_OK, cml_get_status(ctx));
}

static void test_sgd_null_ctx_returns_null(void) {
    TEST_ASSERT_NULL(cml_sgd_init(NULL, 0.01f));
}

/* --- step --- */

static void test_sgd_step_updates_param(void) {
    // param = [[3, 1]], target = [[0, 0]]
    // MSE grad = (2/2) * (param - target) = [[3, 1]]
    // After step with lr=0.1: param = [[3 - 0.3, 1 - 0.1]] = [[2.7, 0.9]]
    cml_tensor_t *param  = cml_tensor_init(ctx, 1, 2);
    cml_tensor_t *target = cml_tensor_init(ctx, 1, 2);
    cml_tensor_set(param,  0, 0, 3.0f); cml_tensor_set(param,  0, 1, 1.0f);
    cml_tensor_set(target, 0, 0, 0.0f); cml_tensor_set(target, 0, 1, 0.0f);
    cml_tensor_set_requires_grad(param, true);

    cml_tensor_t *loss = cml_loss_mse(ctx, param, target);
    cml_backward(ctx, loss);

    cml_sgd_t *opt = cml_sgd_init(ctx, 0.1f);
    cml_tensor_t *params[] = {param};
    cml_sgd_step(opt, params, 1);

    TEST_ASSERT_FLOAT_WITHIN(DELTA, 2.7f, cml_tensor_get(param, 0, 0));
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 0.9f, cml_tensor_get(param, 0, 1));
}

static void test_sgd_step_skips_null_grad(void) {
    cml_tensor_t *param = cml_tensor_init(ctx, 1, 1);
    cml_tensor_set(param, 0, 0, 5.0f);
    // no backward call, so grad is NULL
    cml_sgd_t *opt = cml_sgd_init(ctx, 0.1f);
    cml_tensor_t *params[] = {param};
    cml_sgd_step(opt, params, 1);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, cml_tensor_get(param, 0, 0));
}

static void test_sgd_step_null_opt_no_crash(void) {
    cml_tensor_t *param = cml_tensor_init(ctx, 1, 1);
    cml_tensor_t *params[] = {param};
    cml_sgd_step(NULL, params, 1); // should not crash
    TEST_ASSERT_EQUAL(CML_OK, cml_get_status(ctx));
}

static void test_sgd_step_null_param_entry_no_crash(void) {
    cml_sgd_t *opt = cml_sgd_init(ctx, 0.1f);
    cml_tensor_t *params[] = {NULL};
    cml_sgd_step(opt, params, 1); // should not crash
    TEST_ASSERT_EQUAL(CML_OK, cml_get_status(ctx));
}

static void test_sgd_step_multiple_params(void) {
    // Sum both MSE losses so both gradients flow in one backward pass
    cml_tensor_t *p1 = cml_tensor_init(ctx, 1, 1);
    cml_tensor_t *p2 = cml_tensor_init(ctx, 1, 1);
    cml_tensor_t *t1 = cml_tensor_init(ctx, 1, 1);
    cml_tensor_t *t2 = cml_tensor_init(ctx, 1, 1);
    cml_tensor_set(p1, 0, 0, 4.0f); cml_tensor_set(t1, 0, 0, 0.0f);
    cml_tensor_set(p2, 0, 0, 6.0f); cml_tensor_set(t2, 0, 0, 0.0f);
    cml_tensor_set_requires_grad(p1, true);
    cml_tensor_set_requires_grad(p2, true);

    cml_tensor_t *combined = cml_tensor_add(ctx,
                                 cml_loss_mse(ctx, p1, t1),
                                 cml_loss_mse(ctx, p2, t2));
    cml_backward(ctx, combined);

    cml_sgd_t *opt = cml_sgd_init(ctx, 0.1f);
    cml_tensor_t *params[] = {p1, p2};
    cml_sgd_step(opt, params, 2);
    cml_tape_clear(ctx);

    // grad_p1 = 2*(4-0)/1 = 8  →  p1 = 4 - 0.1*8 = 3.2
    // grad_p2 = 2*(6-0)/1 = 12 →  p2 = 6 - 0.1*12 = 4.8
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 3.2f, cml_tensor_get(p1, 0, 0));
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 4.8f, cml_tensor_get(p2, 0, 0));
}

static void test_sgd_step_moves_toward_target(void) {
    // Verify 10 steps of gradient descent converge a param toward the target
    cml_tensor_t *param  = cml_tensor_init(ctx, 1, 1);
    cml_tensor_t *target = cml_tensor_init(ctx, 1, 1);
    cml_tensor_set(param,  0, 0, 10.0f);
    cml_tensor_set(target, 0, 0,  2.0f);
    cml_tensor_set_requires_grad(param, true);

    cml_sgd_t *opt = cml_sgd_init(ctx, 0.1f);
    cml_tensor_t *params[] = {param};

    for (int i = 0; i < 10; i++) {
        cml_tensor_t *loss = cml_loss_mse(ctx, param, target);
        cml_backward(ctx, loss);
        cml_sgd_step(opt, params, 1);
        cml_tape_clear(ctx);
    }

    // After 10 steps param should be much closer to 2.0 than 10.0
    float val = cml_tensor_get(param, 0, 0);
    TEST_ASSERT_GREATER_THAN_FLOAT(2.0f,  val);
    TEST_ASSERT_LESS_THAN_FLOAT(10.0f, val);
}

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_sgd_init_not_null);
    RUN_TEST(test_sgd_null_ctx_returns_null);

    RUN_TEST(test_sgd_step_updates_param);
    RUN_TEST(test_sgd_step_skips_null_grad);
    RUN_TEST(test_sgd_step_null_opt_no_crash);
    RUN_TEST(test_sgd_step_null_param_entry_no_crash);
    RUN_TEST(test_sgd_step_multiple_params);
    RUN_TEST(test_sgd_step_moves_toward_target);

    return UNITY_END();
}
