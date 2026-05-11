#include "tensor.h"
#include "context.h"

#include <cblas.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static cml_tensor_t *tensor_alloc(cml_context_t *ctx, size_t rows, size_t cols) {
    cml_tensor_t *t = cml_arena_alloc(&ctx->arena, sizeof(struct cml_tensor_s));
    if (t == NULL) {
        cml_context_error(ctx, CML_OUT_OF_MEMORY, "tensor struct allocation failed");
        return NULL;
    }
    t->rows = rows;
    t->cols = cols;
    t->stride = cols;
    t->data = NULL;
    return t;
}

cml_tensor_t *cml_tensor_init(cml_context_t *ctx, size_t rows, size_t cols) {
    if (ctx == NULL) return NULL;
    if (ctx->status != CML_OK) return NULL;

    cml_tensor_t *t = tensor_alloc(ctx, rows, cols);
    if (t == NULL) return NULL;

    t->data = cml_arena_alloc(&ctx->arena, rows * cols * sizeof(float));
    if (t->data == NULL) {
        cml_context_error(ctx, CML_OUT_OF_MEMORY, "tensor data allocation failed");
        return NULL;
    }

    return t;
}

size_t cml_tensor_rows(const cml_tensor_t *tensor) {
    if (tensor == NULL) return 0;
    return tensor->rows;
}

size_t cml_tensor_cols(const cml_tensor_t *tensor) {
    if (tensor == NULL) return 0;
    return tensor->cols;
}

float cml_tensor_get(const cml_tensor_t *tensor, size_t row, size_t col) {
    if (tensor == NULL) return 0.0f;
    return tensor->data[row * tensor->stride + col];
}

void cml_tensor_set(cml_tensor_t *tensor, size_t row, size_t col, float value) {
    if (tensor == NULL) return;
    tensor->data[row * tensor->stride + col] = value;
}

float *cml_tensor_data(cml_tensor_t *tensor) {
    if (tensor == NULL) return NULL;
    return tensor->data;
}

void cml_tensor_zero(cml_tensor_t *tensor) {
    if (tensor == NULL) return;

    for (size_t r = 0; r < tensor->rows; r++) {
        memset(tensor->data + r * tensor->stride, 0, tensor->cols * sizeof(float));
    }
}

void cml_tensor_fill(cml_tensor_t *tensor, float value) {
    if (tensor == NULL) return;

    for (size_t r = 0; r < tensor->rows; r++) {
        float *row = tensor->data + r * tensor->stride;
        for (size_t c = 0; c < tensor->cols; c++) {
            row[c] = value;
        }
    }
}

void cml_tensor_rand(cml_tensor_t *tensor, float low, float high) {
    if (tensor == NULL) return;

    float range = high - low;
    for (size_t r = 0; r < tensor->rows; r++) {
        float *row = tensor->data + r * tensor->stride;
        for (size_t c = 0; c < tensor->cols; c++) {
            row[c] = low + range * ((float)rand() / (float)RAND_MAX);
        }
    }
}

void cml_tensor_copy(cml_context_t *ctx, cml_tensor_t *dst, const cml_tensor_t *src) {
    if (ctx == NULL) return;
    if (ctx->status != CML_OK) return;
    if (dst == NULL || src == NULL) {
        cml_context_error(ctx, CML_INVALID_ARG, "tensor argument is NULL");
        return;
    }
    if (dst->rows != src->rows || dst->cols != src->cols) {
        cml_context_error(ctx, CML_INVALID_ARG, "shape mismatch in copy");
        return;
    }

    for (size_t r = 0; r < src->rows; r++) {
        cblas_scopy((int)src->cols,
                    src->data + r * src->stride, 1,
                    dst->data + r * dst->stride, 1);
    }
}

cml_tensor_t *cml_tensor_reshape(cml_context_t *ctx, cml_tensor_t *tensor,
                                  size_t new_rows, size_t new_cols) {
    if (ctx == NULL) return NULL;
    if (ctx->status != CML_OK) return NULL;
    if (tensor == NULL) {
        cml_context_error(ctx, CML_INVALID_ARG, "tensor is NULL");
        return NULL;
    }
    if (tensor->stride != tensor->cols) {
        cml_context_error(ctx, CML_INVALID_ARG, "cannot reshape a non-contiguous view");
        return NULL;
    }
    if (new_rows * new_cols != tensor->rows * tensor->cols) {
        cml_context_error(ctx, CML_INVALID_ARG, "reshape element count mismatch");
        return NULL;
    }

    cml_tensor_t *view = tensor_alloc(ctx, new_rows, new_cols);
    if (view == NULL) return NULL;
    view->data = tensor->data;
    return view;
}

cml_tensor_t *cml_tensor_view(cml_context_t *ctx, cml_tensor_t *tensor,
                               size_t start_row, size_t start_col,
                               size_t rows, size_t cols) {
    if (ctx == NULL) return NULL;
    if (ctx->status != CML_OK) return NULL;
    if (tensor == NULL) {
        cml_context_error(ctx, CML_INVALID_ARG, "tensor is NULL");
        return NULL;
    }
    if (start_row + rows > tensor->rows || start_col + cols > tensor->cols) {
        cml_context_error(ctx, CML_INVALID_ARG, "view dimensions out of bounds");
        return NULL;
    }

    cml_tensor_t *view = tensor_alloc(ctx, rows, cols);
    if (view == NULL) return NULL;
    view->data = tensor->data + start_row * tensor->stride + start_col;
    view->stride = tensor->stride; /* non-contiguous: rows remain parent-spaced */
    return view;
}

cml_tensor_t *cml_tensor_scale(cml_context_t *ctx, cml_tensor_t *tensor, float scalar) {
    if (ctx == NULL) return NULL;
    if (ctx->status != CML_OK) return NULL;
    if (tensor == NULL) {
        cml_context_error(ctx, CML_INVALID_ARG, "tensor is NULL");
        return NULL;
    }

    cml_tensor_t *out = cml_tensor_init(ctx, tensor->rows, tensor->cols);
    if (out == NULL) return NULL;

    for (size_t r = 0; r < tensor->rows; r++) {
        cblas_scopy((int)tensor->cols, tensor->data + r * tensor->stride, 1,
                    out->data + r * out->stride, 1);
        cblas_sscal((int)out->cols, scalar, out->data + r * out->stride, 1);
    }

    return out;
}

cml_tensor_t *cml_tensor_sigmoid(cml_context_t *ctx, cml_tensor_t *tensor) {
    if (ctx == NULL) return NULL;
    if (ctx->status != CML_OK) return NULL;
    if (tensor == NULL) {
        cml_context_error(ctx, CML_INVALID_ARG, "tensor is NULL");
        return NULL;
    }

    cml_tensor_t *out = cml_tensor_init(ctx, tensor->rows, tensor->cols);
    if (out == NULL) return NULL;

    for (size_t r = 0; r < tensor->rows; r++) {
        const float *src = tensor->data + r * tensor->stride;
        float *dst = out->data + r * out->stride;
        for (size_t c = 0; c < tensor->cols; c++) {
            dst[c] = 1.0f / (1.0f + expf(-src[c]));
        }
    }

    return out;
}

cml_tensor_t *cml_tensor_relu(cml_context_t *ctx, cml_tensor_t *tensor) {
    if (ctx == NULL) return NULL;
    if (ctx->status != CML_OK) return NULL;
    if (tensor == NULL) {
        cml_context_error(ctx, CML_INVALID_ARG, "tensor is NULL");
        return NULL;
    }

    cml_tensor_t *out = cml_tensor_init(ctx, tensor->rows, tensor->cols);
    if (out == NULL) return NULL;

    for (size_t r = 0; r < tensor->rows; r++) {
        const float *src = tensor->data + r * tensor->stride;
        float *dst = out->data + r * out->stride;
        for (size_t c = 0; c < tensor->cols; c++) {
            dst[c] = src[c] > 0.0f ? src[c] : 0.0f;
        }
    }

    return out;
}

static cml_tensor_t *tensor_add_scaled(cml_context_t *ctx,
                                        cml_tensor_t *a, cml_tensor_t *b, float alpha) {
    if (a->rows != b->rows || a->cols != b->cols) {
        cml_context_error(ctx, CML_INVALID_ARG, "shape mismatch");
        return NULL;
    }

    cml_tensor_t *out = cml_tensor_init(ctx, a->rows, a->cols);
    if (out == NULL) return NULL;

    for (size_t r = 0; r < a->rows; r++) {
        cblas_scopy((int)a->cols, a->data + r * a->stride, 1,
                    out->data + r * out->stride, 1);
        cblas_saxpy((int)a->cols, alpha, b->data + r * b->stride, 1,
                    out->data + r * out->stride, 1);
    }

    return out;
}

cml_tensor_t *cml_tensor_add(cml_context_t *ctx, cml_tensor_t *a, cml_tensor_t *b) {
    if (ctx == NULL) return NULL;
    if (ctx->status != CML_OK) return NULL;
    if (a == NULL || b == NULL) {
        cml_context_error(ctx, CML_INVALID_ARG, "tensor argument is NULL");
        return NULL;
    }
    return tensor_add_scaled(ctx, a, b, 1.0f);
}

cml_tensor_t *cml_tensor_sub(cml_context_t *ctx, cml_tensor_t *a, cml_tensor_t *b) {
    if (ctx == NULL) return NULL;
    if (ctx->status != CML_OK) return NULL;
    if (a == NULL || b == NULL) {
        cml_context_error(ctx, CML_INVALID_ARG, "tensor argument is NULL");
        return NULL;
    }
    return tensor_add_scaled(ctx, a, b, -1.0f);
}

cml_tensor_t *cml_tensor_mul(cml_context_t *ctx, cml_tensor_t *a, cml_tensor_t *b) {
    if (ctx == NULL) return NULL;
    if (ctx->status != CML_OK) return NULL;
    if (a == NULL || b == NULL) {
        cml_context_error(ctx, CML_INVALID_ARG, "tensor argument is NULL");
        return NULL;
    }
    if (a->rows != b->rows || a->cols != b->cols) {
        cml_context_error(ctx, CML_INVALID_ARG, "shape mismatch");
        return NULL;
    }

    cml_tensor_t *out = cml_tensor_init(ctx, a->rows, a->cols);
    if (out == NULL) return NULL;

    for (size_t r = 0; r < a->rows; r++) {
        const float *ar = a->data + r * a->stride;
        const float *br = b->data + r * b->stride;
        float *dr = out->data + r * out->stride;
        for (size_t c = 0; c < a->cols; c++) {
            dr[c] = ar[c] * br[c];
        }
    }

    return out;
}

cml_tensor_t *cml_tensor_dot(cml_context_t *ctx, cml_tensor_t *a, cml_tensor_t *b) {
    if (ctx == NULL) return NULL;
    if (ctx->status != CML_OK) return NULL;
    if (a == NULL || b == NULL) {
        cml_context_error(ctx, CML_INVALID_ARG, "tensor argument is NULL");
        return NULL;
    }
    if (a->cols != b->rows) {
        cml_context_error(ctx, CML_INVALID_ARG, "incompatible shapes for matrix multiply");
        return NULL;
    }

    cml_tensor_t *out = cml_tensor_init(ctx, a->rows, b->cols);
    if (out == NULL) return NULL;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)a->rows, (int)b->cols, (int)a->cols,
                1.0f,
                a->data, (int)a->stride,
                b->data, (int)b->stride,
                0.0f,
                out->data, (int)out->stride);

    return out;
}

cml_tensor_t *cml_tensor_transpose(cml_context_t *ctx, cml_tensor_t *tensor) {
    if (ctx == NULL) return NULL;
    if (ctx->status != CML_OK) return NULL;
    if (tensor == NULL) {
        cml_context_error(ctx, CML_INVALID_ARG, "tensor is NULL");
        return NULL;
    }

    cml_tensor_t *out = cml_tensor_init(ctx, tensor->cols, tensor->rows);
    if (out == NULL) return NULL;

    for (size_t r = 0; r < tensor->rows; r++) {
        for (size_t c = 0; c < tensor->cols; c++) {
            out->data[c * out->stride + r] = tensor->data[r * tensor->stride + c];
        }
    }

    return out;
}

cml_tensor_t *cml_tensor_sum(cml_context_t *ctx, cml_tensor_t *tensor) {
    if (ctx == NULL) return NULL;
    if (ctx->status != CML_OK) return NULL;
    if (tensor == NULL) {
        cml_context_error(ctx, CML_INVALID_ARG, "tensor is NULL");
        return NULL;
    }

    cml_tensor_t *out = cml_tensor_init(ctx, 1, 1);
    if (out == NULL) return NULL;

    float total = 0.0f;
    for (size_t r = 0; r < tensor->rows; r++) {
        const float *row = tensor->data + r * tensor->stride;
        for (size_t c = 0; c < tensor->cols; c++) {
            total += row[c];
        }
    }

    out->data[0] = total;
    return out;
}
