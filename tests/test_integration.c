#include <cml/cml.h>
#include "unity.h"

#include <stdlib.h>

// End-to-end training scenarios driven against both CPU and CUDA contexts;
// Basically each test has a _impl and takes a backend type and if cuda is available
// it will run both, otherwise just CPU.

#define CTX_SIZE (8 * 1024 * 1024)

static cml_context_t *active_ctx;

void setUp(void) {
    active_ctx = NULL;
}

void tearDown(void) {
    if (active_ctx != NULL) {
        cml_deinit(active_ctx);
        active_ctx = NULL;
    }
}

static cml_context_t *open_ctx(cml_backend_t backend) {
    active_ctx = cml_init_with_backend(CTX_SIZE, backend);
    if (active_ctx == NULL) {
        if (backend == CML_BACKEND_CUDA) {
            TEST_IGNORE_MESSAGE("CUDA backend/device unavailable at runtime");
        } else {
            TEST_FAIL_MESSAGE("CPU context init failed");
        }
    }
    return active_ctx;
}

/* --- 1. Linear regression: y = 3x + 2 --- */

typedef struct {
    float w;
    float b;
    float final_loss;
} linreg_result_t;

typedef struct {
    cml_linear_t *layer;
} linreg_model_t;

static cml_tensor_t *linreg_forward(cml_context_t *ctx, void *ptr,
                                     cml_tensor_t *x, cml_tensor_t *y) {
    linreg_model_t *m = (linreg_model_t *)ptr;
    cml_tensor_t *pred = cml_linear_forward(ctx, m->layer, x);
    return cml_loss_mse(ctx, pred, y);
}

static cml_status_t train_linreg(cml_context_t *ctx, size_t epochs, float lr,
                                 linreg_result_t *out) {
    static const float x_vals[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    static const float y_vals[] = {2.3f, 2.6f, 2.9f, 3.2f, 3.5f, 3.8f, 4.1f, 4.4f};
    const size_t n_samples = sizeof(x_vals) / sizeof(x_vals[0]);

    cml_tensor_t *x_train = cml_tensor_init(ctx, n_samples, 1);
    cml_tensor_t *y_train = cml_tensor_init(ctx, n_samples, 1);
    if (x_train == NULL || y_train == NULL) return cml_get_status(ctx);
    for (size_t i = 0; i < n_samples; i++) {
        cml_tensor_set(x_train, i, 0, x_vals[i]);
        cml_tensor_set(y_train, i, 0, y_vals[i]);
    }

    linreg_model_t model = { cml_linear_init(ctx, 1, 1) };
    if (model.layer == NULL) return cml_get_status(ctx);

    cml_tensor_t *params[2];
    size_t n_params = cml_linear_collect_params(model.layer, params, 2, 0);

    cml_optimizer_t *opt = cml_optimizer_sgd(ctx, lr);
    if (opt == NULL) return cml_get_status(ctx);

    cml_trainer_t *trainer = cml_trainer_init(ctx, &model, linreg_forward,
                                              params, n_params, opt);
    if (trainer == NULL) return cml_get_status(ctx);

    cml_trainer_fit(ctx, trainer, x_train, y_train, epochs, false);
    if (cml_get_status(ctx) != CML_OK) return cml_get_status(ctx);

    out->w = cml_tensor_get(cml_linear_weight(model.layer), 0, 0);
    out->b = cml_tensor_get(cml_linear_bias(model.layer), 0, 0);

    cml_grad_disable(ctx);
    cml_tensor_t *final_loss = linreg_forward(ctx, &model, x_train, y_train);
    out->final_loss = (final_loss != NULL) ? cml_tensor_get(final_loss, 0, 0) : 0.0f;
    cml_grad_enable(ctx);
    return cml_get_status(ctx);
}

static void linreg_impl(cml_backend_t backend) {
    cml_context_t *ctx = open_ctx(backend);
    srand(42);

    linreg_result_t result;
    cml_status_t status = train_linreg(ctx, 1000, 0.05f, &result);
    TEST_ASSERT_EQUAL(CML_OK, status);

    TEST_ASSERT_FLOAT_WITHIN(0.05f, 3.0f, result.w);
    TEST_ASSERT_FLOAT_WITHIN(0.05f, 2.0f, result.b);
    TEST_ASSERT_LESS_THAN_FLOAT(1e-3f, result.final_loss);
}

static void test_linear_regression_cpu(void)  { linreg_impl(CML_BACKEND_CPU); }
static void test_linear_regression_cuda(void) { linreg_impl(CML_BACKEND_CUDA); }

/* --- 2. XOR MLP: small sequential net learning the XOR truth table --- */

typedef struct {
    cml_module_t *net;
} xor_model_t;

static cml_tensor_t *xor_forward(cml_context_t *ctx, void *ptr,
                                  cml_tensor_t *x, cml_tensor_t *y) {
    xor_model_t *m = (xor_model_t *)ptr;
    cml_tensor_t *pred = cml_module_forward(ctx, m->net, x);
    return cml_loss_mse(ctx, pred, y);
}

static void xor_impl(cml_backend_t backend) {
    cml_context_t *ctx = open_ctx(backend);
    srand(7);

    cml_tensor_t *x = cml_tensor_init(ctx, 4, 2);
    cml_tensor_t *y = cml_tensor_init(ctx, 4, 1);
    static const float xs[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    static const float ys[4]    = { 0,    1,    1,    0  };
    for (size_t i = 0; i < 4; i++) {
        cml_tensor_set(x, i, 0, xs[i][0]);
        cml_tensor_set(x, i, 1, xs[i][1]);
        cml_tensor_set(y, i, 0, ys[i]);
    }

    cml_module_t *layers[] = {
        cml_nn_linear(ctx, 2, 16),
        cml_nn_relu(ctx),
        cml_nn_linear(ctx, 16, 1),
    };
    xor_model_t model = { cml_nn_sequential(ctx, layers, 3) };

    size_t n_params = cml_module_param_count(model.net);
    cml_tensor_t **params = (cml_tensor_t **)malloc(n_params * sizeof(*params));
    TEST_ASSERT_NOT_NULL(params);
    cml_module_collect_params(model.net, params, n_params, 0);

    cml_optimizer_t *opt = cml_optimizer_adam(ctx, model.net, 0.01f, 0.9f, 0.999f, 1e-8f);
    cml_trainer_t *trainer = cml_trainer_init(ctx, &model, xor_forward,
                                              params, n_params, opt);
    cml_trainer_fit(ctx, trainer, x, y, 2000, false);
    TEST_ASSERT_EQUAL(CML_OK, cml_get_status(ctx));

    cml_grad_disable(ctx);
    cml_tensor_t *pred = cml_module_forward(ctx, model.net, x);
    TEST_ASSERT_NOT_NULL(pred);
    float p[4];
    for (size_t i = 0; i < 4; i++) p[i] = cml_tensor_get(pred, i, 0);
    cml_grad_enable(ctx);

    free(params);

    TEST_ASSERT_FLOAT_WITHIN(0.3f, 0.0f, p[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.3f, 1.0f, p[1]);
    TEST_ASSERT_FLOAT_WITHIN(0.3f, 1.0f, p[2]);
    TEST_ASSERT_FLOAT_WITHIN(0.3f, 0.0f, p[3]);
}

static void test_xor_mlp_cpu(void)  { xor_impl(CML_BACKEND_CPU); }
static void test_xor_mlp_cuda(void) { xor_impl(CML_BACKEND_CUDA); }

/* --- 3. Gradient check: autograd vs central finite differences --- */

static float eval_loss(cml_context_t *ctx, cml_tensor_t *x, cml_tensor_t *W, cml_tensor_t *y) {
    cml_tensor_t *pred = cml_tensor_dot(ctx, x, W);
    cml_tensor_t *diff = cml_tensor_sub(ctx, pred, y);
    cml_tensor_t *sq   = cml_tensor_mul(ctx, diff, diff);
    cml_tensor_t *loss = cml_tensor_sum(ctx, sq);
    return cml_tensor_get(loss, 0, 0);
}

static void grad_check_impl(cml_backend_t backend) {
    cml_context_t *ctx = open_ctx(backend);
    srand(123);

    cml_tensor_t *x = cml_tensor_init(ctx, 4, 3);
    cml_tensor_t *y = cml_tensor_init(ctx, 4, 2);
    cml_tensor_t *W = cml_tensor_init(ctx, 3, 2);
    cml_tensor_rand(x, -1.0f, 1.0f);
    cml_tensor_rand(y, -1.0f, 1.0f);
    cml_tensor_rand(W, -0.5f, 0.5f);
    cml_tensor_set_requires_grad(W, true);

    // Autograd pass
    cml_tensor_t *pred = cml_tensor_dot(ctx, x, W);
    cml_tensor_t *diff = cml_tensor_sub(ctx, pred, y);
    cml_tensor_t *sq = cml_tensor_mul(ctx, diff, diff);
    cml_tensor_t *loss = cml_tensor_sum(ctx, sq);
    cml_backward(ctx, loss);

    cml_tensor_t *gW = cml_tensor_grad(W);
    TEST_ASSERT_NOT_NULL(gW);

    cml_tape_clear(ctx);
    cml_grad_disable(ctx);

    // Compare against central finite differences at every element of W
    const float eps = 1e-3f;
    for (size_t r = 0; r < 3; r++) {
        for (size_t c = 0; c < 2; c++) {
            float w0 = cml_tensor_get(W, r, c);

            cml_tensor_set(W, r, c, w0 + eps);
            float lp = eval_loss(ctx, x, W, y);

            cml_tensor_set(W, r, c, w0 - eps);
            float lm = eval_loss(ctx, x, W, y);

            cml_tensor_set(W, r, c, w0); // restore

            float numerical = (lp - lm) / (2.0f * eps);
            float autograd  = cml_tensor_get(gW, r, c);
            // float32 + eps=1e-3 typically agrees to a few parts in 1e3
            TEST_ASSERT_FLOAT_WITHIN(2e-2f, numerical, autograd);
        }
    }

    cml_grad_enable(ctx);
}

static void test_gradient_check_cpu(void)  { grad_check_impl(CML_BACKEND_CPU); }
static void test_gradient_check_cuda(void) { grad_check_impl(CML_BACKEND_CUDA); }

/* --- 4. SGD actually decreases loss --- */

static void sgd_descends_impl(cml_backend_t backend) {
    cml_context_t *ctx = open_ctx(backend);
    srand(11);

    linreg_result_t early, late;
    TEST_ASSERT_EQUAL(CML_OK, train_linreg(ctx, 5, 0.05f, &early));
    cml_deinit(ctx);
    active_ctx = NULL;

    ctx = open_ctx(backend);
    srand(11);
    TEST_ASSERT_EQUAL(CML_OK, train_linreg(ctx, 500, 0.05f, &late));

    TEST_ASSERT_LESS_THAN_FLOAT(early.final_loss, late.final_loss);
}

static void test_sgd_descends_cpu(void)  { sgd_descends_impl(CML_BACKEND_CPU); }
static void test_sgd_descends_cuda(void) { sgd_descends_impl(CML_BACKEND_CUDA); }

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_linear_regression_cpu);
    RUN_TEST(test_xor_mlp_cpu);
    RUN_TEST(test_gradient_check_cpu);
    RUN_TEST(test_sgd_descends_cpu);

    if (cml_cuda_is_available()) {
        RUN_TEST(test_linear_regression_cuda);
        RUN_TEST(test_xor_mlp_cuda);
        RUN_TEST(test_gradient_check_cuda);
        RUN_TEST(test_sgd_descends_cuda);
    }

    return UNITY_END();
}
