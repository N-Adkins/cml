#include "linear.h"
#include "backend/backend.h"
#include "context.h"
#include "tensor.h"
#include "tape.h"

#include <math.h>

static cml_tensor_t *linear_add_bias(cml_context_t *ctx,
                                      cml_tensor_t *a, cml_tensor_t *b) {
    cml_tensor_t *out = cml_tensor_init(ctx, a->rows, a->cols);
    if (out == NULL) return NULL;
    if (cml_backend_add_bias(ctx, out, a, b) != CML_OK) return NULL;

    cml_tape_record_add_bias(ctx, out, a, b);
    return out;
}

cml_linear_t *cml_linear_init(cml_context_t *ctx, size_t in_features, size_t out_features) {
    if (ctx == NULL) return NULL;
    if (ctx->status != CML_OK) return NULL;
    if (in_features == 0 || out_features == 0) {
        cml_context_error(ctx, CML_INVALID_ARG, "feature counts must be non-zero");
        return NULL;
    }

    cml_linear_t *layer = cml_arena_alloc(&ctx->arena, sizeof(struct cml_linear_s));
    if (layer == NULL) {
        cml_context_error(ctx, CML_OUT_OF_MEMORY, "linear layer allocation failed");
        return NULL;
    }

    layer->w = cml_tensor_init(ctx, in_features, out_features);
    if (layer->w == NULL) return NULL;

    layer->b = cml_tensor_init(ctx, 1, out_features);
    if (layer->b == NULL) return NULL;

    // Xavier/Glorot uniform: keeps activation variance stable across layers
    float limit = sqrtf(6.0f / (float)(in_features + out_features));
    cml_tensor_rand(layer->w, -limit, limit);
    cml_tensor_zero(layer->b);

    cml_tensor_set_requires_grad(layer->w, true);
    cml_tensor_set_requires_grad(layer->b, true);

    return layer;
}

cml_tensor_t *cml_linear_forward(cml_context_t *ctx, cml_linear_t *layer, cml_tensor_t *x) {
    if (ctx == NULL) return NULL;
    if (ctx->status != CML_OK) return NULL;
    if (layer == NULL || x == NULL) {
        cml_context_error(ctx, CML_INVALID_ARG, "argument is NULL");
        return NULL;
    }
    if (x->cols != layer->w->rows) {
        cml_context_error(ctx, CML_INVALID_ARG, "input features do not match layer width");
        return NULL;
    }

    cml_tensor_t *z = cml_tensor_dot(ctx, x, layer->w);
    if (z == NULL) return NULL;
    return linear_add_bias(ctx, z, layer->b);
}

cml_tensor_t *cml_linear_weight(const cml_linear_t *layer) {
    if (layer == NULL) return NULL;
    return layer->w;
}

cml_tensor_t *cml_linear_bias(const cml_linear_t *layer) {
    if (layer == NULL) return NULL;
    return layer->b;
}

size_t cml_linear_collect_params(cml_linear_t *layer, cml_tensor_t **params, size_t offset) {
    if (layer == NULL || params == NULL) return offset;
    params[offset]     = layer->w;
    params[offset + 1] = layer->b;
    return offset + 2;
}
