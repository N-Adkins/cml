#include "trainer.h"
#include "context.h"
#include "data.h"
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
                                 cml_optimizer_t *opt) {
    if (ctx == NULL) return NULL;
    if (ctx->status != CML_OK) return NULL;
    if (loss_fn == NULL || params == NULL || opt == NULL) {
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
    trainer->opt      = opt;
    trainer->last_loss = 0.0f;
    trainer->last_epoch = 0;
    trainer->has_loss = false;

    return trainer;
}

static bool cml_trainer_step(cml_context_t *ctx, cml_trainer_t *trainer,
                             cml_tensor_t *x_batch, cml_tensor_t *y_batch,
                             float *loss_out) {
    cml_tensor_t *loss = trainer->loss_fn(ctx, trainer->model, x_batch, y_batch);
    if (loss == NULL) {
        return false;
    }

    float loss_val = cml_tensor_get(loss, 0, 0);
    cml_backward(ctx, loss);
    cml_optimizer_step(trainer->opt, trainer->params, trainer->n_params);
    cml_tape_clear(ctx);
    if (ctx->status != CML_OK) {
        return false;
    }

    *loss_out = loss_val;
    return true;
}

static void cml_trainer_prepare_param_device(cml_context_t *ctx, cml_trainer_t *trainer) {
    for (size_t i = 0; i < trainer->n_params; i++) {
        cml_backend_tensor_to_device(ctx, trainer->params[i]);
    }
}

void cml_trainer_fit_loader(cml_context_t *ctx, cml_trainer_t *trainer,
                            cml_data_loader_t *loader,
                            size_t epochs, bool verbose) {
    if (ctx == NULL || trainer == NULL || loader == NULL) return;

    cml_trainer_prepare_param_device(ctx, trainer);
    cml_data_loader_prepare_device(ctx, loader);
    if (ctx->status != CML_OK) return;

    int width = 1;
    size_t tmp = epochs;
    while (tmp >= 10) { tmp /= 10; width++; }

    for (size_t epoch = 0; epoch < epochs; epoch++) {
        cml_data_loader_reset(ctx, loader);
        if (ctx->status != CML_OK) break;

        float epoch_loss_sum = 0.0f;
        size_t epoch_batch_count = 0;

        while (true) {
            size_t arena_mark = cml_arena_mark(&ctx->arena);
            void *device_mark = cml_backend_device_mark(ctx);
            if (ctx->status != CML_OK) {
                cml_arena_rewind(&ctx->arena, arena_mark);
                cml_backend_device_rewind(ctx, device_mark);
                break;
            }

            cml_tensor_t *x_batch = NULL;
            cml_tensor_t *y_batch = NULL;
            bool has_batch = cml_data_loader_next(ctx, loader, &x_batch, &y_batch);
            if (ctx->status != CML_OK || !has_batch) {
                cml_arena_rewind(&ctx->arena, arena_mark);
                cml_backend_device_rewind(ctx, device_mark);
                break;
            }

            float batch_loss = 0.0f;
            bool step_ok = cml_trainer_step(ctx, trainer, x_batch, y_batch, &batch_loss);
            if (!step_ok) {
                cml_tape_clear(ctx);
                cml_arena_rewind(&ctx->arena, arena_mark);
                cml_backend_device_rewind(ctx, device_mark);
                break;
            }

            epoch_loss_sum += batch_loss;
            epoch_batch_count++;
            cml_arena_rewind(&ctx->arena, arena_mark);
            cml_backend_device_rewind(ctx, device_mark);
        }

        if (ctx->status != CML_OK) break;
        if (epoch_batch_count == 0) break;

        trainer->last_loss = epoch_loss_sum / (float)epoch_batch_count;
        trainer->last_epoch = epoch + 1;
        trainer->has_loss = true;

        if (verbose) {
            cml_trainer_print_progress(epoch + 1, epochs, width, trainer->last_loss);
        }
    }

    if (verbose && trainer->has_loss) {
        cml_trainer_print_progress(trainer->last_epoch, epochs, width, trainer->last_loss);
        printf("  final\n");
    }
}

void cml_trainer_fit(cml_context_t *ctx, cml_trainer_t *trainer,
                     cml_tensor_t *x, cml_tensor_t *y,
                     size_t epochs, bool verbose) {
    if (ctx == NULL || trainer == NULL || x == NULL || y == NULL) return;

    cml_dataset_t *dataset = cml_dataset_from_tensors(ctx, x, y);
    if (dataset == NULL) return;

    cml_data_loader_t *loader = cml_data_loader_init(ctx, dataset, cml_dataset_num_samples(dataset), false);
    if (loader == NULL) return;

    cml_trainer_fit_loader(ctx, trainer, loader, epochs, verbose);
}
