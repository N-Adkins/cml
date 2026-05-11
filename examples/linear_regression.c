#include <cml/cml.h>

#include <stdio.h>

/*
 * Learns y = 3x + 2 from 8 samples using a single linear layer.
 * Expected result after training: w = 3.0, b = 2.0.
 */

typedef struct {
    cml_linear_t *layer;
} linreg_model_t;

static cml_tensor_t *linreg_forward(cml_context_t *ctx, void *ptr,
                                     cml_tensor_t *x, cml_tensor_t *y) {
    linreg_model_t *m = (linreg_model_t *)ptr;
    cml_tensor_t *pred = cml_linear_forward(ctx, m->layer, x);
    return cml_loss_mse(ctx, pred, y);
}

int main(void) {
    cml_context_t *ctx = cml_init(4 * 1024 * 1024);
    if (ctx == NULL) return 1;

    static const float x_vals[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    static const float y_vals[] = {2.3f, 2.6f, 2.9f, 3.2f, 3.5f, 3.8f, 4.1f, 4.4f};
    const size_t n_samples = sizeof(x_vals) / sizeof(x_vals[0]);

    cml_tensor_t *x_train = cml_tensor_init(ctx, n_samples, 1);
    cml_tensor_t *y_train = cml_tensor_init(ctx, n_samples, 1);
    for (size_t i = 0; i < n_samples; i++) {
        cml_tensor_set(x_train, i, 0, x_vals[i]);
        cml_tensor_set(y_train, i, 0, y_vals[i]);
    }

    linreg_model_t model = { cml_linear_init(ctx, 1, 1) };

    cml_tensor_t *params[2];
    size_t n_params = cml_linear_collect_params(model.layer, params, 0);

    cml_trainer_t *trainer = cml_trainer_init(ctx, &model, linreg_forward,
                                              params, n_params, 0.05f);

    cml_trainer_fit(ctx, trainer, x_train, y_train, 1000, true);
    if (cml_get_status(ctx) != CML_OK) {
        fprintf(stderr, "\ntraining failed: %s\n", cml_get_error_msg(ctx));
        cml_deinit(ctx);
        return 1;
    }

    float w = cml_tensor_get(cml_linear_weight(model.layer), 0, 0);
    float b = cml_tensor_get(cml_linear_bias(model.layer), 0, 0);

    cml_grad_disable(ctx);
    cml_tensor_t *final_loss = linreg_forward(ctx, &model, x_train, y_train);
    float loss = cml_tensor_get(final_loss, 0, 0);
    cml_grad_enable(ctx);

    printf("\nLearned:  w = %6.4f  b = %6.4f\n", (double)w, (double)b);
    printf("Expected: w = 3.0000  b = 2.0000\n");
    printf("Final loss: %.6f\n", (double)loss);

    cml_deinit(ctx);
    return 0;
}
