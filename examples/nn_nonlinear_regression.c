#include <cml/cml.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * Learns a highly nonlinear scalar function:
 *   f(x) = 0.15*x^3 + 0.60*sin(3x) - 0.35*cos(5x)
 * using a small MLP: Linear(1,32) -> ReLU -> Linear(32,32) -> ReLU -> Linear(32,1).
 */

typedef struct {
    cml_module_t *net;
} model_t;

static float target_fn(float x) {
    return 0.15f * x * x * x + 0.60f * sinf(3.0f * x) - 0.35f * cosf(5.0f * x);
}

static cml_tensor_t *forward(cml_context_t *ctx, void *ptr, cml_tensor_t *x, cml_tensor_t *y) {
    model_t *m = ptr;
    cml_tensor_t *pred = cml_module_forward(ctx, m->net, x);
    return cml_loss_mse(ctx, pred, y);
}

int main(void) {
    cml_context_t *ctx = cml_init_with_backend(1024 * 1024, CML_BACKEND_CUDA);
    if (ctx == NULL) return 1;

    const size_t n_samples = 512;
    const float x_min = -2.0f;
    const float x_max = 2.0f;

    cml_tensor_t *x_train = cml_tensor_init(ctx, n_samples, 1);
    cml_tensor_t *y_train = cml_tensor_init(ctx, n_samples, 1);

    for (size_t i = 0; i < n_samples; i++) {
        float t = (float)i / (float)(n_samples - 1);
        float x = x_min + t * (x_max - x_min);
        cml_tensor_set(x_train, i, 0, x);
        cml_tensor_set(y_train, i, 0, target_fn(x));
    }

    cml_module_t *l1 = cml_nn_linear(ctx, 1, 32);
    cml_module_t *r1 = cml_nn_relu(ctx);
    cml_module_t *l2 = cml_nn_linear(ctx, 32, 64);
    cml_module_t *r2 = cml_nn_relu(ctx);
    cml_module_t *l3 = cml_nn_linear(ctx, 64, 512);
    cml_module_t *r3 = cml_nn_relu(ctx);
    cml_module_t *l4 = cml_nn_linear(ctx, 512, 64);
    cml_module_t *r4 = cml_nn_relu(ctx);
    cml_module_t *l5 = cml_nn_linear(ctx, 64, 32);
    cml_module_t *r5 = cml_nn_relu(ctx);
    cml_module_t *l6 = cml_nn_linear(ctx, 32, 1);
    cml_module_t *layers[] = { l1, r1, l2, r2, l3, r3, l4, r4, l5, r5, l6 };

    model_t model = { cml_nn_sequential(ctx, layers, 11) };

    const size_t n_params = cml_module_param_count(model.net);
    cml_tensor_t *params = malloc(sizeof(cml_tensor_t) * n_params);;
    cml_module_collect_params(model.net, params, 0);

    cml_trainer_t *trainer = cml_trainer_init(ctx, &model, forward, params, n_params, 0.1f);
    cml_trainer_fit(ctx, trainer, x_train, y_train, 5000, true);

    cml_tensor_t *pred = cml_module_forward(ctx, model.net, x_train);

    printf("\nSample predictions after training:\n");
    printf("      x        pred      target\n");
    for (size_t i = 0; i < n_samples; i += 16) {
        float x = cml_tensor_get(x_train, i, 0);
        float p = cml_tensor_get(pred, i, 0);
        float y = cml_tensor_get(y_train, i, 0);
        printf("%8.4f  %8.4f  %8.4f\n", (double)x, (double)p, (double)y);
    }
    
    free(params);

    cml_deinit(ctx);
    return 0;
}
