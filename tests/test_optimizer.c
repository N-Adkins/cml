#include <cml/context.h>
#include <cml/linear.h>
#include <cml/loss.h>
#include <cml/nn.h>
#include <cml/optimizer.h>
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

/* --- SGD: init --- */

static void test_sgd_init_not_null(void) {
    cml_optimizer_t *opt = cml_optimizer_sgd(ctx, 0.01f);
    TEST_ASSERT_NOT_NULL(opt);
    TEST_ASSERT_EQUAL(CML_OK, cml_get_status(ctx));
}

static void test_sgd_null_ctx_returns_null(void) {
    TEST_ASSERT_NULL(cml_optimizer_sgd(NULL, 0.01f));
}

/* --- SGD: step --- */

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

    cml_optimizer_t *opt = cml_optimizer_sgd(ctx, 0.1f);
    cml_tensor_t *params[] = {param};
    cml_optimizer_step(opt, params, 1);

    TEST_ASSERT_FLOAT_WITHIN(DELTA, 2.7f, cml_tensor_get(param, 0, 0));
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 0.9f, cml_tensor_get(param, 0, 1));
}

static void test_sgd_step_skips_null_grad(void) {
    cml_tensor_t *param = cml_tensor_init(ctx, 1, 1);
    cml_tensor_set(param, 0, 0, 5.0f);
    // no backward call, so grad is NULL
    cml_optimizer_t *opt = cml_optimizer_sgd(ctx, 0.1f);
    cml_tensor_t *params[] = {param};
    cml_optimizer_step(opt, params, 1);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, cml_tensor_get(param, 0, 0));
}

static void test_sgd_step_null_opt_no_crash(void) {
    cml_tensor_t *param = cml_tensor_init(ctx, 1, 1);
    cml_tensor_t *params[] = {param};
    cml_optimizer_step(NULL, params, 1);
    TEST_ASSERT_EQUAL(CML_OK, cml_get_status(ctx));
}

static void test_sgd_step_null_param_entry_no_crash(void) {
    cml_optimizer_t *opt = cml_optimizer_sgd(ctx, 0.1f);
    cml_tensor_t *params[] = {NULL};
    cml_optimizer_step(opt, params, 1);
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

    cml_optimizer_t *opt = cml_optimizer_sgd(ctx, 0.1f);
    cml_tensor_t *params[] = {p1, p2};
    cml_optimizer_step(opt, params, 2);
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

    cml_optimizer_t *opt = cml_optimizer_sgd(ctx, 0.1f);
    cml_tensor_t *params[] = {param};

    for (int i = 0; i < 10; i++) {
        cml_tensor_t *loss = cml_loss_mse(ctx, param, target);
        cml_backward(ctx, loss);
        cml_optimizer_step(opt, params, 1);
        cml_tape_clear(ctx);
    }

    float val = cml_tensor_get(param, 0, 0);
    TEST_ASSERT_GREATER_THAN_FLOAT(2.0f,  val);
    TEST_ASSERT_LESS_THAN_FLOAT(10.0f, val);
}

/* --- ADAM: init --- */

static void test_adam_init_not_null(void) {
    cml_module_t *layer = cml_nn_linear(ctx, 2, 2);
    cml_optimizer_t *opt = cml_optimizer_adam(ctx, layer, 0.01f, 0.9f, 0.999f, 1e-8f);
    TEST_ASSERT_NOT_NULL(opt);
    TEST_ASSERT_EQUAL(CML_OK, cml_get_status(ctx));
}

static void test_adam_null_ctx_returns_null(void) {
    cml_module_t *layer = cml_nn_linear(ctx, 2, 2);
    TEST_ASSERT_NULL(cml_optimizer_adam(NULL, layer, 0.01f, 0.9f, 0.999f, 1e-8f));
}

static void test_adam_null_module_returns_null(void) {
    TEST_ASSERT_NULL(cml_optimizer_adam(ctx, NULL, 0.01f, 0.9f, 0.999f, 1e-8f));
}

/* --- ADAM: step ---
 *
 * Helper: a tiny y = w*x model (1-in, 1-out linear with no bias activity)
 * lets us reason about the first Adam step analytically. With a fresh state
 * (m=v=0, t=0) the first update simplifies to:
 *     m_hat = g / (1 - beta1^1) * beta1's complement -> just g
 *     v_hat = g^2
 *     delta = lr * g / (sqrt(g^2) + eps) ≈ lr * sign(g)
 */

static void test_adam_first_step_is_signed_lr(void) {
    // Build a 1x1 linear, force weights/bias to known values.
    cml_module_t *layer = cml_nn_linear(ctx, 1, 1);
    cml_tensor_t *params[2];
    size_t n = cml_module_collect_params(layer, params, 0);
    TEST_ASSERT_EQUAL_UINT(2, n);

    cml_tensor_set(params[0], 0, 0, 1.0f); // weight
    cml_tensor_set(params[1], 0, 0, 0.0f); // bias

    cml_tensor_t *x = cml_tensor_init(ctx, 1, 1);
    cml_tensor_t *y = cml_tensor_init(ctx, 1, 1);
    cml_tensor_set(x, 0, 0, 1.0f);
    cml_tensor_set(y, 0, 0, 0.0f);

    cml_optimizer_t *opt = cml_optimizer_adam(ctx, layer, 0.1f, 0.9f, 0.999f, 1e-8f);

    cml_tensor_t *pred = cml_module_forward(ctx, layer, x);
    cml_tensor_t *loss = cml_loss_mse(ctx, pred, y);
    cml_backward(ctx, loss);

    cml_optimizer_step(opt, params, n);

    // Pred = 1.0, target = 0.0, grad_w > 0 (pred too high), so update is -lr.
    TEST_ASSERT_FLOAT_WITHIN(1e-3f, 0.9f, cml_tensor_get(params[0], 0, 0));
    // Bias gradient is also positive (constant 1.0 backprop signal), so bias drops by lr.
    TEST_ASSERT_FLOAT_WITHIN(1e-3f, -0.1f, cml_tensor_get(params[1], 0, 0));
}

static void test_adam_step_skips_null_grad(void) {
    cml_module_t *layer = cml_nn_linear(ctx, 1, 1);
    cml_tensor_t *params[2];
    size_t n = cml_module_collect_params(layer, params, 0);

    cml_tensor_set(params[0], 0, 0, 1.0f);
    cml_tensor_set(params[1], 0, 0, 2.0f);

    cml_optimizer_t *opt = cml_optimizer_adam(ctx, layer, 0.1f, 0.9f, 0.999f, 1e-8f);
    // No backward pass: params have no grad attached.
    cml_optimizer_step(opt, params, n);

    TEST_ASSERT_EQUAL_FLOAT(1.0f, cml_tensor_get(params[0], 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(2.0f, cml_tensor_get(params[1], 0, 0));
}

static void test_adam_step_moves_toward_target(void) {
    // 100 steps of Adam should drive a 1x1 linear toward y = 2 for x = 1.
    cml_module_t *layer = cml_nn_linear(ctx, 1, 1);
    cml_tensor_t *params[2];
    size_t n = cml_module_collect_params(layer, params, 0);
    cml_tensor_set(params[0], 0, 0, 10.0f);
    cml_tensor_set(params[1], 0, 0, 0.0f);

    cml_tensor_t *x = cml_tensor_init(ctx, 1, 1);
    cml_tensor_t *y = cml_tensor_init(ctx, 1, 1);
    cml_tensor_set(x, 0, 0, 1.0f);
    cml_tensor_set(y, 0, 0, 2.0f);

    cml_optimizer_t *opt = cml_optimizer_adam(ctx, layer, 0.1f, 0.9f, 0.999f, 1e-8f);

    for (int i = 0; i < 200; i++) {
        cml_tensor_t *pred = cml_module_forward(ctx, layer, x);
        cml_tensor_t *loss = cml_loss_mse(ctx, pred, y);
        cml_backward(ctx, loss);
        cml_optimizer_step(opt, params, n);
        cml_tape_clear(ctx);
    }

    float w = cml_tensor_get(params[0], 0, 0);
    float b = cml_tensor_get(params[1], 0, 0);
    // y_pred = w*1 + b should land near 2.
    TEST_ASSERT_FLOAT_WITHIN(0.1f, 2.0f, w + b);
}

static void test_adam_reset_restarts_bias_correction(void) {
    // After running a few Adam steps, reset() should clear m/v/t so that the
    // NEXT step behaves like the first step on a fresh optimizer.
    cml_module_t *layer = cml_nn_linear(ctx, 1, 1);
    cml_tensor_t *params[2];
    size_t n = cml_module_collect_params(layer, params, 0);
    cml_tensor_set(params[0], 0, 0, 1.0f);
    cml_tensor_set(params[1], 0, 0, 0.0f);

    cml_tensor_t *x = cml_tensor_init(ctx, 1, 1);
    cml_tensor_t *y = cml_tensor_init(ctx, 1, 1);
    cml_tensor_set(x, 0, 0, 1.0f);
    cml_tensor_set(y, 0, 0, 0.0f);

    cml_optimizer_t *opt = cml_optimizer_adam(ctx, layer, 0.1f, 0.9f, 0.999f, 1e-8f);

    // Burn in a couple steps to accumulate m/v/t.
    for (int i = 0; i < 3; i++) {
        cml_tensor_t *pred = cml_module_forward(ctx, layer, x);
        cml_tensor_t *loss = cml_loss_mse(ctx, pred, y);
        cml_backward(ctx, loss);
        cml_optimizer_step(opt, params, n);
        cml_tape_clear(ctx);
    }

    // Reset and force weight back to 1.0; one more step should now match the
    // analytical first-step result (-lr * sign(grad)).
    cml_optimizer_reset(opt);
    cml_tensor_set(params[0], 0, 0, 1.0f);
    cml_tensor_set(params[1], 0, 0, 0.0f);

    cml_tensor_t *pred = cml_module_forward(ctx, layer, x);
    cml_tensor_t *loss = cml_loss_mse(ctx, pred, y);
    cml_backward(ctx, loss);
    cml_optimizer_step(opt, params, n);

    TEST_ASSERT_FLOAT_WITHIN(1e-3f, 0.9f, cml_tensor_get(params[0], 0, 0));
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

    RUN_TEST(test_adam_init_not_null);
    RUN_TEST(test_adam_null_ctx_returns_null);
    RUN_TEST(test_adam_null_module_returns_null);
    RUN_TEST(test_adam_first_step_is_signed_lr);
    RUN_TEST(test_adam_step_skips_null_grad);
    RUN_TEST(test_adam_step_moves_toward_target);
    RUN_TEST(test_adam_reset_restarts_bias_correction);

    return UNITY_END();
}
