#include <cml/context.h>
#include <cml/linear.h>
#include <cml/loss.h>
#include <cml/tape.h>
#include <cml/tensor.h>
#include <cml/trainer.h>
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

typedef struct { cml_linear_t *layer; } test_model_t;

static cml_tensor_t *test_forward(cml_context_t *c, void *model,
                                   cml_tensor_t *x, cml_tensor_t *y) {
    test_model_t *m = model;
    return cml_loss_mse(c, cml_linear_forward(c, m->layer, x), y);
}

/* --- init --- */

static void test_trainer_init_not_null(void) {
    test_model_t model = { cml_linear_init(ctx, 1, 1) };
    cml_tensor_t *params[2];
    size_t n = cml_linear_collect_params(model.layer, params, 0);
    cml_trainer_t *trainer = cml_trainer_init(ctx, &model, test_forward, params, n, 0.01f);
    TEST_ASSERT_NOT_NULL(trainer);
    TEST_ASSERT_EQUAL(CML_OK, cml_get_status(ctx));
}

static void test_trainer_null_ctx_returns_null(void) {
    cml_tensor_t *params[2];
    TEST_ASSERT_NULL(cml_trainer_init(NULL, NULL, test_forward, params, 0, 0.01f));
}

static void test_trainer_null_loss_fn_errors(void) {
    cml_tensor_t *params[2];
    TEST_ASSERT_NULL(cml_trainer_init(ctx, NULL, NULL, params, 0, 0.01f));
    TEST_ASSERT_EQUAL(CML_INVALID_ARG, cml_get_status(ctx));
}

/* --- fit --- */

static void test_trainer_fit_decreases_loss(void) {
    /* y = 2, x = 1: single sample, linear(1,1) should learn w ≈ 2 */
    cml_tensor_t *x = cml_tensor_init(ctx, 1, 1);
    cml_tensor_t *y = cml_tensor_init(ctx, 1, 1);
    cml_tensor_set(x, 0, 0, 1.0f);
    cml_tensor_set(y, 0, 0, 2.0f);

    test_model_t model = { cml_linear_init(ctx, 1, 1) };

    /* Force a known-bad starting weight so the test is deterministic */
    cml_tensor_set(cml_linear_weight(model.layer), 0, 0, 10.0f);

    cml_tensor_t *params[2];
    size_t n = cml_linear_collect_params(model.layer, params, 0);

    cml_tensor_t *init_loss = test_forward(ctx, &model, x, y);
    float before = cml_tensor_get(init_loss, 0, 0);
    cml_tape_clear(ctx);

    cml_trainer_t *trainer = cml_trainer_init(ctx, &model, test_forward, params, n, 0.05f);
    cml_trainer_fit(ctx, trainer, x, y, 50, false);

    cml_tensor_t *final_loss = test_forward(ctx, &model, x, y);
    float after = cml_tensor_get(final_loss, 0, 0);
    cml_tape_clear(ctx);

    TEST_ASSERT_LESS_THAN_FLOAT(before, after);
}

static void test_trainer_fit_null_ctx_no_crash(void) {
    cml_tensor_t *x = cml_tensor_init(ctx, 1, 1);
    cml_tensor_t *y = cml_tensor_init(ctx, 1, 1);
    test_model_t model = { cml_linear_init(ctx, 1, 1) };
    cml_tensor_t *params[2];
    size_t n = cml_linear_collect_params(model.layer, params, 0);
    cml_trainer_t *trainer = cml_trainer_init(ctx, &model, test_forward, params, n, 0.01f);
    cml_trainer_fit(NULL, trainer, x, y, 5, false); /* should not crash */
    TEST_ASSERT_EQUAL(CML_OK, cml_get_status(ctx));
}

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_trainer_init_not_null);
    RUN_TEST(test_trainer_null_ctx_returns_null);
    RUN_TEST(test_trainer_null_loss_fn_errors);

    RUN_TEST(test_trainer_fit_decreases_loss);
    RUN_TEST(test_trainer_fit_null_ctx_no_crash);

    return UNITY_END();
}
