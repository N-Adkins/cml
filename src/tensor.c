#include "tensor.h"
#include "context.h"
#include "backend/backend.h"
#include "random.h"
#include "tape.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static cml_tensor_t *tensor_alloc(cml_context_t *ctx, size_t rows, size_t cols) {
    cml_tensor_t *t = cml_arena_alloc(&ctx->arena, sizeof(struct cml_tensor_s));
    if (t == NULL) {
        cml_context_error(ctx, CML_OUT_OF_MEMORY, "tensor struct allocation failed");
        return NULL;
    }
    t->ctx = ctx;
    t->rows = rows;
    t->cols = cols;
    t->stride = cols;
    t->data = NULL;
    t->device_data = NULL;
    t->storage = t;
    t->data_offset = 0;
    t->host_valid = true;
    t->device_valid = false;
    t->grad = NULL;
    t->requires_grad = false;
    t->creator = NULL;
    return t;
}

cml_tensor_t *cml_tensor_init(cml_context_t *ctx, size_t rows, size_t cols) {
    if (ctx == NULL) return NULL;
    if (ctx->status != CML_OK) return NULL;

    // Guard rows*cols*sizeof(float) against size_t wrap.
    if (rows != 0 && cols != 0) {
        if (cols > SIZE_MAX / rows ||
            rows * cols > SIZE_MAX / sizeof(float)) {
            cml_context_error(ctx, CML_INVALID_ARG, "tensor dimensions overflow");
            return NULL;
        }
    }

    cml_tensor_t *t = tensor_alloc(ctx, rows, cols);
    if (t == NULL) return NULL;

    size_t bytes = rows * cols * sizeof(float);
    if (bytes == 0) return t; // empty tensor; no backing storage needed

    t->data = cml_arena_alloc(&ctx->arena, bytes);
    if (t->data == NULL) {
        cml_context_error(ctx, CML_OUT_OF_MEMORY, "tensor data allocation failed");
        return NULL;
    }

    // Zero tensors by default
    memset(t->data, 0, bytes);

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
    if (tensor == NULL || tensor->data == NULL) return 0.0f;
    if (row >= tensor->rows || col >= tensor->cols) {
        if (tensor->ctx != NULL) {
            cml_context_error(tensor->ctx, CML_INVALID_ARG, "tensor index out of bounds");
        }
        return 0.0f;
    }
    if (tensor->ctx != NULL && cml_backend_tensor_to_host(tensor->ctx, tensor) != CML_OK) {
        return 0.0f;
    }
    return tensor->data[row * tensor->stride + col];
}

void cml_tensor_set(cml_tensor_t *tensor, size_t row, size_t col, float value) {
    if (tensor == NULL || tensor->data == NULL) return;
    if (row >= tensor->rows || col >= tensor->cols) {
        if (tensor->ctx != NULL) {
            cml_context_error(tensor->ctx, CML_INVALID_ARG, "tensor index out of bounds");
        }
        return;
    }
    // Pull device-side data back to host first, otherwise this single-element
    // write lands on stale host memory and the rest of the buffer is lost when
    // we mark host as authoritative below.
    if (tensor->ctx != NULL && cml_backend_tensor_to_host(tensor->ctx, tensor) != CML_OK) {
        return;
    }
    tensor->data[row * tensor->stride + col] = value;
    cml_backend_mark_host_write(tensor);
}

float *cml_tensor_data(cml_tensor_t *tensor) {
    if (tensor == NULL) return NULL;
    if (tensor->ctx != NULL && cml_backend_tensor_to_host(tensor->ctx, tensor) != CML_OK) {
        return NULL;
    }
    cml_backend_mark_host_write(tensor);
    return tensor->data;
}

const float *cml_tensor_const_data(const cml_tensor_t *tensor) {
    if (tensor == NULL) return NULL;
    if (tensor->ctx != NULL && cml_backend_tensor_to_host(tensor->ctx, tensor) != CML_OK) {
        return NULL;
    }
    return tensor->data;
}

cml_status_t cml_tensor_to_host(cml_context_t *ctx, cml_tensor_t *tensor) {
    if (ctx == NULL || tensor == NULL) return CML_INVALID_ARG;
    return cml_backend_tensor_to_host(ctx, tensor);
}

cml_status_t cml_tensor_to_device(cml_context_t *ctx, cml_tensor_t *tensor) {
    if (ctx == NULL || tensor == NULL) return CML_INVALID_ARG;
    return cml_backend_tensor_to_device(ctx, tensor);
}

bool cml_tensor_has_device_copy(const cml_tensor_t *tensor) {
    if (tensor == NULL) return false;
    return cml_backend_tensor_has_device_copy(tensor);
}

// For a view, these only touch a sub-rectangle of the storage, so we must pull
// device data back to host first — otherwise marking host authoritative drops
// any data outside the view that only lives on the device.
void cml_tensor_zero(cml_tensor_t *tensor) {
    if (tensor == NULL || tensor->data == NULL) return;
    if (tensor->ctx != NULL && cml_backend_tensor_to_host(tensor->ctx, tensor) != CML_OK) return;

    for (size_t r = 0; r < tensor->rows; r++) {
        memset(tensor->data + r * tensor->stride, 0, tensor->cols * sizeof(float));
    }
    cml_backend_mark_host_write(tensor);
}

void cml_tensor_fill(cml_tensor_t *tensor, float value) {
    if (tensor == NULL || tensor->data == NULL) return;
    if (tensor->ctx != NULL && cml_backend_tensor_to_host(tensor->ctx, tensor) != CML_OK) return;

    for (size_t r = 0; r < tensor->rows; r++) {
        float *row = tensor->data + r * tensor->stride;
        for (size_t c = 0; c < tensor->cols; c++) {
            row[c] = value;
        }
    }
    cml_backend_mark_host_write(tensor);
}

void cml_tensor_rand(cml_tensor_t *tensor, float low, float high) {
    if (tensor == NULL || tensor->data == NULL) return;
    if (tensor->ctx == NULL) return;
    if (cml_backend_tensor_to_host(tensor->ctx, tensor) != CML_OK) return;

    float range = high - low;
    cml_rng_t *rng = &tensor->ctx->rng;
    for (size_t r = 0; r < tensor->rows; r++) {
        float *row = tensor->data + r * tensor->stride;
        for (size_t c = 0; c < tensor->cols; c++) {
            row[c] = low + range * cml_rng_next_uniform(rng);
        }
    }
    cml_backend_mark_host_write(tensor);
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

    if (cml_backend_copy(ctx, dst, src) != CML_OK) {
        return;
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
    view->storage = tensor->storage;
    view->data_offset = tensor->data_offset;
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
    view->stride = tensor->stride; // Non-contiguous: rows remain parent-spaced.
    view->storage = tensor->storage;
    view->data_offset = tensor->data_offset + start_row * tensor->stride + start_col;
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

    if (cml_backend_scale(ctx, out, tensor, scalar) != CML_OK) return NULL;

    cml_tape_record_scale(ctx, out, tensor, scalar);
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

    if (cml_backend_unary(ctx, out, tensor, CML_UNARY_SIGMOID) != CML_OK) return NULL;

    cml_tape_record_sigmoid(ctx, out, tensor);
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

    if (cml_backend_unary(ctx, out, tensor, CML_UNARY_RELU) != CML_OK) return NULL;

    cml_tape_record_relu(ctx, out, tensor);
    return out;
}

cml_tensor_t *cml_tensor_log(cml_context_t *ctx, cml_tensor_t *tensor) {
    if (ctx == NULL) return NULL;
    if (ctx->status != CML_OK) return NULL;
    if (tensor == NULL) {
        cml_context_error(ctx, CML_INVALID_ARG, "tensor is NULL");
        return NULL;
    }

    cml_tensor_t *out = cml_tensor_init(ctx, tensor->rows, tensor->cols);
    if (out == NULL) return NULL;

    if (cml_backend_unary(ctx, out, tensor, CML_UNARY_LOG) != CML_OK) return NULL;

    cml_tape_record_log(ctx, out, tensor);
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

    if (cml_backend_add_scaled(ctx, out, a, b, alpha) != CML_OK) return NULL;

    return out;
}

cml_tensor_t *cml_tensor_add(cml_context_t *ctx, cml_tensor_t *a, cml_tensor_t *b) {
    if (ctx == NULL) return NULL;
    if (ctx->status != CML_OK) return NULL;
    if (a == NULL || b == NULL) {
        cml_context_error(ctx, CML_INVALID_ARG, "tensor argument is NULL");
        return NULL;
    }
    cml_tensor_t *out = tensor_add_scaled(ctx, a, b, 1.0f);
    if (out == NULL) return NULL;
    cml_tape_record_add(ctx, out, a, b);
    return out;
}

cml_tensor_t *cml_tensor_sub(cml_context_t *ctx, cml_tensor_t *a, cml_tensor_t *b) {
    if (ctx == NULL) return NULL;
    if (ctx->status != CML_OK) return NULL;
    if (a == NULL || b == NULL) {
        cml_context_error(ctx, CML_INVALID_ARG, "tensor argument is NULL");
        return NULL;
    }
    cml_tensor_t *out = tensor_add_scaled(ctx, a, b, -1.0f);
    if (out == NULL) return NULL;
    cml_tape_record_sub(ctx, out, a, b);
    return out;
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

    if (cml_backend_mul(ctx, out, a, b) != CML_OK) return NULL;

    cml_tape_record_mul(ctx, out, a, b);
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

    if (cml_backend_dot(ctx, out, a, b) != CML_OK) return NULL;

    cml_tape_record_dot(ctx, out, a, b);
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

    if (cml_backend_transpose(ctx, out, tensor) != CML_OK) return NULL;

    cml_tape_record_transpose(ctx, out, tensor);
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
    if (cml_backend_sum(ctx, tensor, &total) != CML_OK) return NULL;
    out->data[0] = total;
    cml_backend_mark_host_write(out);
    cml_tape_record_sum(ctx, out, tensor);
    return out;
}

void cml_tensor_set_requires_grad(cml_tensor_t *tensor, bool requires_grad) {
    if (tensor == NULL) return;
    tensor->requires_grad = requires_grad;
}

bool cml_tensor_requires_grad(const cml_tensor_t *tensor) {
    if (tensor == NULL) return false;
    return tensor->requires_grad;
}

cml_tensor_t *cml_tensor_grad(const cml_tensor_t *tensor) {
    if (tensor == NULL) return NULL;
    return tensor->grad;
}
