#include "trainer.h"
#include "context.h"

#include <cml/tape.h>
#include <stdio.h>

cml_trainer_t *cml_trainer_init(cml_context_t *ctx,
                                 void *model,
                                 cml_loss_fn loss_fn,
                                 cml_tensor_t **params,
                                 size_t n_params,
                                 float lr) {
    if (ctx == NULL) return NULL;
    if (ctx->status != CML_OK) return NULL;
    if (loss_fn == NULL || params == NULL) {
        cml_context_error(ctx, CML_INVALID_ARG, "argument is NULL");
        return NULL;
    }

    cml_trainer_t *trainer = cml_arena_alloc(&ctx->arena, sizeof(struct cml_trainer_s));
    if (trainer == NULL) {
        cml_context_error(ctx, CML_OUT_OF_MEMORY, "trainer allocation failed");
        return NULL;
    }

    trainer->model    = model;
    trainer->loss_fn  = loss_fn;
    trainer->params   = params;
    trainer->n_params = n_params;
    trainer->opt      = cml_sgd_init(ctx, lr);
    if (trainer->opt == NULL) return NULL;

    return trainer;
}

void cml_trainer_fit(cml_context_t *ctx, cml_trainer_t *trainer,
                     cml_tensor_t *x, cml_tensor_t *y,
                     size_t epochs, bool verbose) {
    if (ctx == NULL || trainer == NULL || x == NULL || y == NULL) return;

    int width = 1;
    size_t tmp = epochs;
    while (tmp >= 10) { tmp /= 10; width++; }

    for (size_t epoch = 0; epoch < epochs; epoch++) {
        size_t arena_mark = cml_arena_mark(&ctx->arena);
        if (ctx->status != CML_OK) break;

        cml_tensor_t *loss = trainer->loss_fn(ctx, trainer->model, x, y);
        if (loss == NULL) {
            cml_tape_clear(ctx);
            cml_arena_rewind(&ctx->arena, arena_mark);
            break;
        }

        float loss_val = cml_tensor_get(loss, 0, 0);

        if (verbose) {
            printf("Epoch %*zu/%zu  loss: %f\n", width, epoch + 1, epochs, (double)loss_val);
            fflush(stdout);
        }

        cml_backward(ctx, loss);
        cml_sgd_step(trainer->opt, trainer->params, trainer->n_params);
        cml_tape_clear(ctx);
        cml_arena_rewind(&ctx->arena, arena_mark);
    }
}
