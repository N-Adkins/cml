#include <cml/context.h>
#include <cml/loss.h>
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

/* --- mse --- */

static void test_mse_perfect_prediction_is_zero(void) {
    cml_tensor_t *pred = cml_tensor_init(ctx, 2, 3);
    cml_tensor_t *target = cml_tensor_init(ctx, 2, 3);
    cml_tensor_fill(pred, 5.0f);
    cml_tensor_fill(target, 5.0f);
    cml_tensor_t *loss = cml_loss_mse(ctx, pred, target);
    TEST_ASSERT_NOT_NULL(loss);
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 0.0f, cml_tensor_get(loss, 0, 0));
}

static void test_mse_known_value(void) {
    // pred = [[2, 4]], target = [[1, 3]]
    // diff^2 = [[1, 1]], mean = 1.0
    cml_tensor_t *pred = cml_tensor_init(ctx, 1, 2);
    cml_tensor_t *target = cml_tensor_init(ctx, 1, 2);
    cml_tensor_set(pred, 0, 0, 2.0f); cml_tensor_set(pred, 0, 1, 4.0f);
    cml_tensor_set(target, 0, 0, 1.0f); cml_tensor_set(target, 0, 1, 3.0f);
    cml_tensor_t *loss = cml_loss_mse(ctx, pred, target);
    TEST_ASSERT_NOT_NULL(loss);
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 1.0f, cml_tensor_get(loss, 0, 0));
}

static void test_mse_output_is_scalar(void) {
    cml_tensor_t *pred = cml_tensor_init(ctx, 3, 4);
    cml_tensor_t *target = cml_tensor_init(ctx, 3, 4);
    cml_tensor_fill(pred, 1.0f);
    cml_tensor_fill(target, 2.0f);
    cml_tensor_t *loss = cml_loss_mse(ctx, pred, target);
    TEST_ASSERT_NOT_NULL(loss);
    TEST_ASSERT_EQUAL_size_t(1, cml_tensor_rows(loss));
    TEST_ASSERT_EQUAL_size_t(1, cml_tensor_cols(loss));
}

static void test_mse_null_pred_errors(void) {
    cml_tensor_t *target = cml_tensor_init(ctx, 2, 2);
    TEST_ASSERT_NULL(cml_loss_mse(ctx, NULL, target));
    TEST_ASSERT_EQUAL(CML_INVALID_ARG, cml_get_status(ctx));
}

static void test_mse_null_target_errors(void) {
    cml_tensor_t *pred = cml_tensor_init(ctx, 2, 2);
    TEST_ASSERT_NULL(cml_loss_mse(ctx, pred, NULL));
    TEST_ASSERT_EQUAL(CML_INVALID_ARG, cml_get_status(ctx));
}

static void test_mse_shape_mismatch_errors(void) {
    cml_tensor_t *pred = cml_tensor_init(ctx, 2, 3);
    cml_tensor_t *target = cml_tensor_init(ctx, 3, 2);
    TEST_ASSERT_NULL(cml_loss_mse(ctx, pred, target));
    TEST_ASSERT_EQUAL(CML_INVALID_ARG, cml_get_status(ctx));
}

static void test_mse_backward(void) {
    // pred = [[2, 4]], target = [[1, 3]]
    // d(mse)/d(pred_i) = (2/n) * (pred_i - target_i) = 1 * [[1, 1]] = [[1, 1]]
    cml_tensor_t *pred = cml_tensor_init(ctx, 1, 2);
    cml_tensor_t *target = cml_tensor_init(ctx, 1, 2);
    cml_tensor_set(pred, 0, 0, 2.0f); cml_tensor_set(pred, 0, 1, 4.0f);
    cml_tensor_set(target, 0, 0, 1.0f); cml_tensor_set(target, 0, 1, 3.0f);
    cml_tensor_set_requires_grad(pred, true);
    cml_tensor_t *loss = cml_loss_mse(ctx, pred, target);
    cml_backward(ctx, loss);
    cml_tensor_t *gp = cml_tensor_grad(pred);
    TEST_ASSERT_NOT_NULL(gp);
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 1.0f, cml_tensor_get(gp, 0, 0));
    TEST_ASSERT_FLOAT_WITHIN(DELTA, 1.0f, cml_tensor_get(gp, 0, 1));
}

/* --- cross entropy --- */

static void test_cross_entropy_known_value(void) {
    // pred = [[0.9, 0.1]], target = [[1.0, 0.0]]
    // loss = -mean(target * log(pred)) = -log(0.9) / 2 = 0.052682
    cml_tensor_t *pred = cml_tensor_init(ctx, 1, 2);
    cml_tensor_t *target = cml_tensor_init(ctx, 1, 2);
    cml_tensor_set(pred, 0, 0, 0.9f); cml_tensor_set(pred, 0, 1, 0.1f);
    cml_tensor_set(target, 0, 0, 1.0f); cml_tensor_set(target, 0, 1, 0.0f);
    cml_tensor_t *loss = cml_loss_cross_entropy(ctx, pred, target);
    TEST_ASSERT_NOT_NULL(loss);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.052682f, cml_tensor_get(loss, 0, 0));
}

static void test_cross_entropy_output_is_scalar(void) {
    cml_tensor_t *pred = cml_tensor_init(ctx, 2, 4);
    cml_tensor_t *target = cml_tensor_init(ctx, 2, 4);
    cml_tensor_fill(pred, 0.5f);
    cml_tensor_fill(target, 0.25f);
    cml_tensor_t *loss = cml_loss_cross_entropy(ctx, pred, target);
    TEST_ASSERT_NOT_NULL(loss);
    TEST_ASSERT_EQUAL_size_t(1, cml_tensor_rows(loss));
    TEST_ASSERT_EQUAL_size_t(1, cml_tensor_cols(loss));
}

static void test_cross_entropy_null_pred_errors(void) {
    cml_tensor_t *target = cml_tensor_init(ctx, 2, 2);
    TEST_ASSERT_NULL(cml_loss_cross_entropy(ctx, NULL, target));
    TEST_ASSERT_EQUAL(CML_INVALID_ARG, cml_get_status(ctx));
}

static void test_cross_entropy_shape_mismatch_errors(void) {
    cml_tensor_t *pred = cml_tensor_init(ctx, 2, 3);
    cml_tensor_t *target = cml_tensor_init(ctx, 3, 2);
    TEST_ASSERT_NULL(cml_loss_cross_entropy(ctx, pred, target));
    TEST_ASSERT_EQUAL(CML_INVALID_ARG, cml_get_status(ctx));
}

static void test_cross_entropy_zero_pred_does_not_nan(void) {
    /* log(0) used to return -inf and poison gradients with NaN; the backend
     * now floors at FLT_MIN so the loss is finite (very negative) instead. */
    cml_tensor_t *pred = cml_tensor_init(ctx, 1, 2);
    cml_tensor_t *target = cml_tensor_init(ctx, 1, 2);
    cml_tensor_set(pred, 0, 0, 0.0f); cml_tensor_set(pred, 0, 1, 1.0f);
    cml_tensor_set(target, 0, 0, 1.0f); cml_tensor_set(target, 0, 1, 0.0f);
    cml_tensor_t *loss = cml_loss_cross_entropy(ctx, pred, target);
    TEST_ASSERT_NOT_NULL(loss);
    float v = cml_tensor_get(loss, 0, 0);
    TEST_ASSERT_FALSE(v != v); // not NaN
    TEST_ASSERT_TRUE(v > -1e30f);
    TEST_ASSERT_TRUE(v <  1e30f);
}

static void test_mse_empty_tensor_errors(void) {
    cml_tensor_t *pred = cml_tensor_init(ctx, 0, 4);
    cml_tensor_t *target = cml_tensor_init(ctx, 0, 4);
    TEST_ASSERT_NULL(cml_loss_mse(ctx, pred, target));
    TEST_ASSERT_EQUAL(CML_INVALID_ARG, cml_get_status(ctx));
}

static void test_cross_entropy_backward(void) {
    // pred = [[0.9, 0.1]], target = [[1.0, 0.0]]
    // d(loss)/d(pred_i) = -(1/n) * target_i / pred_i
    // grad[0,0] = -0.5 * 1.0 / 0.9 = -0.55556
    // grad[0,1] = -0.5 * 0.0 / 0.1 =  0.0
    cml_tensor_t *pred = cml_tensor_init(ctx, 1, 2);
    cml_tensor_t *target = cml_tensor_init(ctx, 1, 2);
    cml_tensor_set(pred, 0, 0, 0.9f); cml_tensor_set(pred, 0, 1, 0.1f);
    cml_tensor_set(target, 0, 0, 1.0f); cml_tensor_set(target, 0, 1, 0.0f);
    cml_tensor_set_requires_grad(pred, true);
    cml_tensor_t *loss = cml_loss_cross_entropy(ctx, pred, target);
    cml_backward(ctx, loss);
    cml_tensor_t *gp = cml_tensor_grad(pred);
    TEST_ASSERT_NOT_NULL(gp);
    TEST_ASSERT_FLOAT_WITHIN(DELTA, -0.55556f, cml_tensor_get(gp, 0, 0));
    TEST_ASSERT_FLOAT_WITHIN(DELTA,  0.0f, cml_tensor_get(gp, 0, 1));
}

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_mse_perfect_prediction_is_zero);
    RUN_TEST(test_mse_known_value);
    RUN_TEST(test_mse_output_is_scalar);
    RUN_TEST(test_mse_null_pred_errors);
    RUN_TEST(test_mse_null_target_errors);
    RUN_TEST(test_mse_shape_mismatch_errors);
    RUN_TEST(test_mse_backward);

    RUN_TEST(test_cross_entropy_known_value);
    RUN_TEST(test_cross_entropy_output_is_scalar);
    RUN_TEST(test_cross_entropy_null_pred_errors);
    RUN_TEST(test_cross_entropy_shape_mismatch_errors);
    RUN_TEST(test_cross_entropy_backward);
    RUN_TEST(test_cross_entropy_zero_pred_does_not_nan);
    RUN_TEST(test_mse_empty_tensor_errors);

    return UNITY_END();
}
