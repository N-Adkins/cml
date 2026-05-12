#include <cml/loss.h>
#include "backend/backend.h"
#include "context.h"
#include "tape.h"
#include "tensor.h"

cml_tensor_t *cml_loss_mse(cml_context_t *ctx, cml_tensor_t *pred, cml_tensor_t *target) {
    if (ctx == NULL) return NULL;
    if (ctx->status != CML_OK) return NULL;
    if (pred == NULL || target == NULL) {
        cml_context_error(ctx, CML_INVALID_ARG, "tensor argument is NULL");
        return NULL;
    }
    if (cml_tensor_rows(pred) != cml_tensor_rows(target) ||
        cml_tensor_cols(pred) != cml_tensor_cols(target)) {
        cml_context_error(ctx, CML_INVALID_ARG, "shape mismatch between pred and target");
        return NULL;
    }
    if (cml_tensor_rows(pred) == 0 || cml_tensor_cols(pred) == 0) {
        cml_context_error(ctx, CML_INVALID_ARG, "loss undefined for empty tensor");
        return NULL;
    }

    cml_tensor_t *diff = cml_tensor_sub(ctx, pred, target);
    if (diff == NULL) return NULL;

    cml_tensor_t *sq = cml_tensor_mul(ctx, diff, diff);
    if (sq == NULL) return NULL;

    cml_tensor_t *s = cml_tensor_sum(ctx, sq);
    if (s == NULL) return NULL;

    float n = (float)(cml_tensor_rows(pred) * cml_tensor_cols(pred));

    return cml_tensor_scale(ctx, s, 1.0f / n);
}

cml_tensor_t *cml_loss_cross_entropy(cml_context_t *ctx, cml_tensor_t *pred, cml_tensor_t *target) {
    if (ctx == NULL) return NULL;
    if (ctx->status != CML_OK) return NULL;
    if (pred == NULL || target == NULL) {
        cml_context_error(ctx, CML_INVALID_ARG, "tensor argument is NULL");
        return NULL;
    }
    if (cml_tensor_rows(pred) != cml_tensor_rows(target) ||
        cml_tensor_cols(pred) != cml_tensor_cols(target)) {
        cml_context_error(ctx, CML_INVALID_ARG, "shape mismatch between pred and target");
        return NULL;
    }
    if (cml_tensor_rows(pred) == 0 || cml_tensor_cols(pred) == 0) {
        cml_context_error(ctx, CML_INVALID_ARG, "loss undefined for empty tensor");
        return NULL;
    }

    cml_tensor_t *log_pred = cml_tensor_log(ctx, pred);
    if (log_pred == NULL) return NULL;

    cml_tensor_t *prod = cml_tensor_mul(ctx, target, log_pred);
    if (prod == NULL) return NULL;

    cml_tensor_t *s = cml_tensor_sum(ctx, prod);
    if (s == NULL) return NULL;

    // Reduce by batch size (rows) to match conventional cross-entropy magnitude
    float n = (float)cml_tensor_rows(pred);

    return cml_tensor_scale(ctx, s, -1.0f / n);
}

cml_tensor_t *cml_loss_softmax_cross_entropy(cml_context_t *ctx, cml_tensor_t *logits, cml_tensor_t *target) {
    if (ctx == NULL) return NULL;
    if (ctx->status != CML_OK) return NULL;
    if (logits == NULL || target == NULL) {
        cml_context_error(ctx, CML_INVALID_ARG, "tensor argument is NULL");
        return NULL;
    }
    if (cml_tensor_rows(logits) != cml_tensor_rows(target) ||
        cml_tensor_cols(logits) != cml_tensor_cols(target)) {
        cml_context_error(ctx, CML_INVALID_ARG, "shape mismatch between logits and target");
        return NULL;
    }
    if (cml_tensor_rows(logits) == 0 || cml_tensor_cols(logits) == 0) {
        cml_context_error(ctx, CML_INVALID_ARG, "loss undefined for empty tensor");
        return NULL;
    }

    cml_tensor_t *softmax_buf = cml_tensor_init(ctx, cml_tensor_rows(logits), cml_tensor_cols(logits));
    if (softmax_buf == NULL) return NULL;

    cml_tensor_t *loss = cml_tensor_init(ctx, 1, 1);
    if (loss == NULL) return NULL;

    float loss_val = 0.0f;
    if (cml_backend_softmax_xent_forward(ctx, softmax_buf, logits, target, &loss_val) != CML_OK) {
        return NULL;
    }
    loss->data[0] = loss_val;
    cml_backend_mark_host_write(loss);

    cml_tape_record_softmax_xent(ctx, loss, logits, target, softmax_buf);
    return loss;
}
