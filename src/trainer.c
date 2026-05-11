#include "trainer.h"
#include "context.h"
#include "backend/backend.h"

#include <cml/tape.h>
#include <stdio.h>

#define CML_TRAINER_PROGRESS_BAR_WIDTH 28

// Handles the whole progress bar and everything
static void cml_trainer_print_progress(size_t epoch, size_t epochs, int width, float loss_val) {
    double progress = (epochs > 0) ? (double)epoch / (double)epochs : 1.0;
    size_t filled = (size_t)(progress * CML_TRAINER_PROGRESS_BAR_WIDTH);
    if (filled > CML_TRAINER_PROGRESS_BAR_WIDTH) filled = CML_TRAINER_PROGRESS_BAR_WIDTH;

    printf("\rEpoch %*zu/%zu [", width, epoch, epochs);
    for (size_t i = 0; i < CML_TRAINER_PROGRESS_BAR_WIDTH; i++) {
        putchar(i < filled ? '=' : '-');
    }
    printf("] %6.2f%%  loss: %.6f", progress * 100.0, (double)loss_val);
    fflush(stdout);
}

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
    trainer->last_loss = 0.0f;
    trainer->last_epoch = 0;
    trainer->has_loss = false;

    return trainer;
}

void cml_trainer_fit(cml_context_t *ctx, cml_trainer_t *trainer,
                     cml_tensor_t *x, cml_tensor_t *y,
                     size_t epochs, bool verbose) {
    if (ctx == NULL || trainer == NULL || x == NULL || y == NULL) return;

    // Upload persistent tensors to device before taking the per-epoch device
    // mark. Otherwise their lazy allocations land above the mark and get
    // freed by cml_backend_device_rewind, leaving dangling device_data on
    // the next epoch.
    cml_backend_tensor_to_device(ctx, x);
    cml_backend_tensor_to_device(ctx, y);
    for (size_t i = 0; i < trainer->n_params; i++) {
        cml_backend_tensor_to_device(ctx, trainer->params[i]);
    }

    int width = 1;
    size_t tmp = epochs;
    while (tmp >= 10) { tmp /= 10; width++; }

    for (size_t epoch = 0; epoch < epochs; epoch++) {
        size_t arena_mark = cml_arena_mark(&ctx->arena);
        void *device_mark = cml_backend_device_mark(ctx);
        if (ctx->status != CML_OK) break;

        cml_tensor_t *loss = trainer->loss_fn(ctx, trainer->model, x, y);
        if (loss == NULL) {
            cml_tape_clear(ctx);
            cml_arena_rewind(&ctx->arena, arena_mark);
            cml_backend_device_rewind(ctx, device_mark);
            break;
        }

        float loss_val = cml_tensor_get(loss, 0, 0);
        trainer->last_loss = loss_val;
        trainer->last_epoch = epoch + 1;
        trainer->has_loss = true;

        if (verbose) {
            cml_trainer_print_progress(epoch + 1, epochs, width, loss_val);
        }

        cml_backward(ctx, loss);
        cml_sgd_step(trainer->opt, trainer->params, trainer->n_params);
        cml_tape_clear(ctx);
        cml_arena_rewind(&ctx->arena, arena_mark);
        cml_backend_device_rewind(ctx, device_mark);
    }

    if (verbose && trainer->has_loss) {
        cml_trainer_print_progress(trainer->last_epoch, epochs, width, trainer->last_loss);
        printf("  final\n");
    }
}
