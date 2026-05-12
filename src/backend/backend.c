#include "backend.h"
#include "../context.h"
#include "../tensor.h"

#include <limits.h>
#include <stdlib.h>

static void set_backend_error(cml_context_t *ctx, cml_status_t status, const char *msg) {
    if (ctx != NULL && status != CML_OK) {
        cml_context_error(ctx, status, msg);
    }
}

static cml_status_t check_int_dims(cml_context_t *ctx, const size_t *vals, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (vals[i] > (size_t)INT_MAX) {
            set_backend_error(ctx, CML_INVALID_ARG, "tensor dim or stride exceeds INT_MAX");
            return CML_INVALID_ARG;
        }
    }
    return CML_OK;
}

static cml_status_t check_tensor_ctx(cml_context_t *ctx, const cml_tensor_t *tensor) {
    if (tensor != NULL && tensor->ctx != NULL && tensor->ctx != ctx) {
        set_backend_error(ctx, CML_INVALID_ARG, "tensor belongs to a different context");
        return CML_INVALID_ARG;
    }
    return CML_OK;
}

static cml_tensor_t *tensor_storage(const cml_tensor_t *tensor) {
    if (tensor == NULL) return NULL;
    return tensor->storage;
}

static size_t storage_elements(const cml_tensor_t *storage) {
    return storage->rows * storage->stride;
}

static cml_status_t register_device_allocation(cml_context_t *ctx, float *ptr, cml_tensor_t *owner) {
    cml_device_alloc_t *node = malloc(sizeof(cml_device_alloc_t));
    if (node == NULL) {
        return CML_OUT_OF_MEMORY;
    }
    node->ptr = ptr;
    node->owner = owner;
    node->next = ctx->device_allocs;
    ctx->device_allocs = node;
    return CML_OK;
}

static cml_status_t ensure_device_allocated(cml_context_t *ctx, cml_tensor_t *storage) {
    if (storage->device_data != NULL) return CML_OK;
    if (ctx->backend_ops->alloc_device == NULL) return CML_INVALID_ARG;
    if (storage_elements(storage) == 0) return CML_OK; // nothing to allocate

    float *device_ptr = NULL;
    cml_status_t status = ctx->backend_ops->alloc_device(
        ctx->backend_state, storage_elements(storage), &device_ptr);
    if (status != CML_OK) return status;

    status = register_device_allocation(ctx, device_ptr, storage);
    if (status != CML_OK) {
        if (ctx->backend_ops->free_device != NULL) {
            ctx->backend_ops->free_device(ctx->backend_state, device_ptr);
        }
        return status;
    }

    storage->device_data = device_ptr;
    return CML_OK;
}

static cml_status_t ensure_host(cml_context_t *ctx, const cml_tensor_t *tensor) {
    cml_tensor_t *storage = tensor_storage(tensor);
    if (!ctx->backend_ops->uses_device) {
        storage->host_valid = true;
        return CML_OK;
    }
    if (storage->host_valid) return CML_OK;
    if (!storage->device_valid || storage->device_data == NULL) {
        return CML_INVALID_ARG;
    }
    cml_status_t status = ctx->backend_ops->copy_device_to_host(
        ctx->backend_state, storage->data, storage->device_data, storage_elements(storage));
    if (status != CML_OK) return status;
    storage->host_valid = true;
    return CML_OK;
}

static cml_status_t ensure_device(cml_context_t *ctx, const cml_tensor_t *tensor) {
    cml_tensor_t *storage = tensor_storage(tensor);
    if (!ctx->backend_ops->uses_device) return CML_OK;

    cml_status_t status = ensure_device_allocated(ctx, storage);
    if (status != CML_OK) return status;
    if (storage->device_valid) return CML_OK;
    if (!storage->host_valid) return CML_INVALID_ARG;

    status = ctx->backend_ops->copy_host_to_device(
        ctx->backend_state, storage->device_data, storage->data, storage_elements(storage));
    if (status != CML_OK) return status;
    storage->device_valid = true;
    return CML_OK;
}

static void mark_host_write(cml_tensor_t *tensor) {
    cml_tensor_t *storage = tensor_storage(tensor);
    storage->host_valid = true;
    storage->device_valid = false;
}

static void mark_device_write(cml_tensor_t *tensor) {
    cml_tensor_t *storage = tensor_storage(tensor);
    storage->host_valid = false;
    storage->device_valid = true;
}

static float *tensor_ptr(cml_context_t *ctx, cml_tensor_t *tensor) {
    cml_tensor_t *storage = tensor_storage(tensor);
    if (ctx->backend_ops->uses_device) {
        return storage->device_data + tensor->data_offset;
    }
    return storage->data + tensor->data_offset;
}

static const float *tensor_const_ptr(cml_context_t *ctx, const cml_tensor_t *tensor) {
    cml_tensor_t *storage = tensor_storage(tensor);
    if (ctx->backend_ops->uses_device) {
        return storage->device_data + tensor->data_offset;
    }
    return storage->data + tensor->data_offset;
}

bool cml_backend_cuda_compiled(void) {
#ifdef CML_ENABLE_CUDA
    return true;
#else
    return false;
#endif
}

cml_status_t cml_backend_init(cml_context_t *ctx, cml_backend_t backend) {
    if (ctx == NULL) return CML_INVALID_ARG;

    const cml_backend_ops_t *ops;
    if (backend == CML_BACKEND_CPU) {
        ops = &cml_cpu_backend_ops;
    } else if (backend == CML_BACKEND_CUDA) {
#ifdef CML_ENABLE_CUDA
        ops = &cml_cuda_backend_ops;
#else
        set_backend_error(ctx, CML_BACKEND_UNAVAILABLE, "CUDA backend was not compiled in");
        return CML_BACKEND_UNAVAILABLE;
#endif
    } else {
        set_backend_error(ctx, CML_INVALID_ARG, "unknown backend");
        return CML_INVALID_ARG;
    }

    void *state = NULL;
    cml_status_t status = ops->init(&state);
    if (status != CML_OK) {
        set_backend_error(ctx, status, "backend initialization failed");
        return status;
    }

    // Publish ops/state only after init succeeds so the deinit path can rely on
    // backend_ops == NULL meaning "no initialized backend".
    ctx->backend_ops = ops;
    ctx->backend_state = state;
    ctx->backend_kind = backend;
    ctx->device_allocs = NULL;
    return CML_OK;
}

void cml_backend_deinit(cml_context_t *ctx) {
    if (ctx == NULL || ctx->backend_ops == NULL) return;

    cml_device_alloc_t *node = ctx->device_allocs;
    while (node != NULL) {
        cml_device_alloc_t *next = node->next;
        if (ctx->backend_ops->uses_device && ctx->backend_ops->free_device != NULL && node->ptr != NULL) {
            ctx->backend_ops->free_device(ctx->backend_state, node->ptr);
        }
        free(node);
        node = next;
    }
    ctx->device_allocs = NULL;

    if (ctx->backend_ops->deinit != NULL) {
        ctx->backend_ops->deinit(ctx->backend_state);
    }
    ctx->backend_state = NULL;
    ctx->backend_ops = NULL;
}

void *cml_backend_device_mark(cml_context_t *ctx) {
    if (ctx == NULL) return NULL;
    return ctx->device_allocs;
}

void cml_backend_device_rewind(cml_context_t *ctx, void *mark) {
    if (ctx == NULL) return;
    cml_device_alloc_t *target = (cml_device_alloc_t *)mark;
    while (ctx->device_allocs != target) {
        cml_device_alloc_t *node = ctx->device_allocs;
        ctx->device_allocs = node->next;
        if (ctx->backend_ops != NULL && ctx->backend_ops->uses_device &&
                ctx->backend_ops->free_device != NULL && node->ptr != NULL) {
            ctx->backend_ops->free_device(ctx->backend_state, node->ptr);
        }
        // Clear the back-pointer so any tensor that outlives this rewind cannot
        // be observed holding a freed device buffer
        if (node->owner != NULL && node->owner->device_data == node->ptr) {
            node->owner->device_data = NULL;
            node->owner->device_valid = false;
        }
        free(node);
    }
}

cml_status_t cml_backend_tensor_to_host(cml_context_t *ctx, const cml_tensor_t *tensor) {
    if (ctx == NULL || tensor == NULL) return CML_INVALID_ARG;
    cml_status_t status = ensure_host(ctx, tensor);
    if (status != CML_OK) {
        set_backend_error(ctx, status, "failed to sync tensor to host");
    }
    return status;
}

cml_status_t cml_backend_tensor_to_device(cml_context_t *ctx, const cml_tensor_t *tensor) {
    if (ctx == NULL || tensor == NULL) return CML_INVALID_ARG;
    cml_status_t status = ensure_device(ctx, tensor);
    if (status != CML_OK) {
        set_backend_error(ctx, status, "failed to sync tensor to device");
    }
    return status;
}

void cml_backend_mark_host_write(cml_tensor_t *tensor) {
    if (tensor == NULL) return;
    mark_host_write(tensor);
}

bool cml_backend_tensor_has_device_copy(const cml_tensor_t *tensor) {
    cml_tensor_t *storage = tensor_storage(tensor);
    if (storage == NULL) return false;
    return storage->device_valid && storage->device_data != NULL;
}

cml_status_t cml_backend_copy(cml_context_t *ctx, cml_tensor_t *dst, const cml_tensor_t *src) {
    if (ctx == NULL || ctx->backend_ops == NULL) return CML_INVALID_ARG;
    if (ctx->status != CML_OK) return ctx->status;
    if (check_tensor_ctx(ctx, dst) != CML_OK) return CML_INVALID_ARG;
    if (check_tensor_ctx(ctx, src) != CML_OK) return CML_INVALID_ARG;
    size_t _dims[] = {dst->rows, dst->cols, dst->stride, src->stride};
    if (check_int_dims(ctx, _dims, sizeof(_dims)/sizeof(_dims[0])) != CML_OK) return CML_INVALID_ARG;
    cml_status_t status;
    if (ctx->backend_ops->uses_device) {
        status = ensure_device(ctx, src);
        if (status != CML_OK) goto fail;
        status = ensure_device_allocated(ctx, tensor_storage(dst));
        if (status != CML_OK) goto fail;
    } else {
        status = ensure_host(ctx, src);
        if (status != CML_OK) goto fail;
    }

    status = ctx->backend_ops->copy(
        ctx->backend_state,
        tensor_ptr(ctx, dst), dst->stride,
        tensor_const_ptr(ctx, src), src->stride,
        dst->rows, dst->cols);
    if (status != CML_OK) goto fail;

    if (ctx->backend_ops->uses_device) {
        mark_device_write(dst);
    } else {
        mark_host_write(dst);
    }
    return CML_OK;

fail:
    set_backend_error(ctx, status, "backend copy failed");
    return status;
}

cml_status_t cml_backend_add_scaled(cml_context_t *ctx, cml_tensor_t *out,
                                    const cml_tensor_t *a, const cml_tensor_t *b, float alpha) {
    if (ctx == NULL || ctx->backend_ops == NULL) return CML_INVALID_ARG;
    if (ctx->status != CML_OK) return ctx->status;
    if (check_tensor_ctx(ctx, out) != CML_OK) return CML_INVALID_ARG;
    if (check_tensor_ctx(ctx, a) != CML_OK) return CML_INVALID_ARG;
    if (check_tensor_ctx(ctx, b) != CML_OK) return CML_INVALID_ARG;
    size_t _dims[] = {out->rows, out->cols, out->stride, a->stride, b->stride};
    if (check_int_dims(ctx, _dims, sizeof(_dims)/sizeof(_dims[0])) != CML_OK) return CML_INVALID_ARG;
    cml_status_t status;
    if (ctx->backend_ops->uses_device) {
        status = ensure_device(ctx, a);
        if (status != CML_OK) goto fail;
        status = ensure_device(ctx, b);
        if (status != CML_OK) goto fail;
        status = ensure_device_allocated(ctx, tensor_storage(out));
        if (status != CML_OK) goto fail;
    } else {
        status = ensure_host(ctx, a);
        if (status != CML_OK) goto fail;
        status = ensure_host(ctx, b);
        if (status != CML_OK) goto fail;
    }

    status = ctx->backend_ops->add_scaled(
        ctx->backend_state,
        tensor_ptr(ctx, out), out->stride,
        tensor_const_ptr(ctx, a), a->stride,
        tensor_const_ptr(ctx, b), b->stride,
        out->rows, out->cols, alpha);
    if (status != CML_OK) goto fail;

    if (ctx->backend_ops->uses_device) {
        mark_device_write(out);
    } else {
        mark_host_write(out);
    }
    return CML_OK;

fail:
    set_backend_error(ctx, status, "backend add/sub failed");
    return status;
}

cml_status_t cml_backend_mul(cml_context_t *ctx, cml_tensor_t *out,
                             const cml_tensor_t *a, const cml_tensor_t *b) {
    if (ctx == NULL || ctx->backend_ops == NULL) return CML_INVALID_ARG;
    if (ctx->status != CML_OK) return ctx->status;
    if (check_tensor_ctx(ctx, out) != CML_OK) return CML_INVALID_ARG;
    if (check_tensor_ctx(ctx, a) != CML_OK) return CML_INVALID_ARG;
    if (check_tensor_ctx(ctx, b) != CML_OK) return CML_INVALID_ARG;
    size_t _dims[] = {out->rows, out->cols, out->stride, a->stride, b->stride};
    if (check_int_dims(ctx, _dims, sizeof(_dims)/sizeof(_dims[0])) != CML_OK) return CML_INVALID_ARG;
    cml_status_t status;
    if (ctx->backend_ops->uses_device) {
        status = ensure_device(ctx, a);
        if (status != CML_OK) goto fail;
        status = ensure_device(ctx, b);
        if (status != CML_OK) goto fail;
        status = ensure_device_allocated(ctx, tensor_storage(out));
        if (status != CML_OK) goto fail;
    } else {
        status = ensure_host(ctx, a);
        if (status != CML_OK) goto fail;
        status = ensure_host(ctx, b);
        if (status != CML_OK) goto fail;
    }

    status = ctx->backend_ops->mul(
        ctx->backend_state,
        tensor_ptr(ctx, out), out->stride,
        tensor_const_ptr(ctx, a), a->stride,
        tensor_const_ptr(ctx, b), b->stride,
        out->rows, out->cols);
    if (status != CML_OK) goto fail;

    if (ctx->backend_ops->uses_device) {
        mark_device_write(out);
    } else {
        mark_host_write(out);
    }
    return CML_OK;

fail:
    set_backend_error(ctx, status, "backend mul failed");
    return status;
}

cml_status_t cml_backend_scale(cml_context_t *ctx, cml_tensor_t *out,
                               const cml_tensor_t *in, float scalar) {
    if (ctx == NULL || ctx->backend_ops == NULL) return CML_INVALID_ARG;
    if (ctx->status != CML_OK) return ctx->status;
    if (check_tensor_ctx(ctx, out) != CML_OK) return CML_INVALID_ARG;
    if (check_tensor_ctx(ctx, in) != CML_OK) return CML_INVALID_ARG;
    size_t _dims[] = {out->rows, out->cols, out->stride, in->stride};
    if (check_int_dims(ctx, _dims, sizeof(_dims)/sizeof(_dims[0])) != CML_OK) return CML_INVALID_ARG;
    cml_status_t status;
    if (ctx->backend_ops->uses_device) {
        status = ensure_device(ctx, in);
        if (status != CML_OK) goto fail;
        status = ensure_device_allocated(ctx, tensor_storage(out));
        if (status != CML_OK) goto fail;
    } else {
        status = ensure_host(ctx, in);
        if (status != CML_OK) goto fail;
    }

    status = ctx->backend_ops->scale(
        ctx->backend_state,
        tensor_ptr(ctx, out), out->stride,
        tensor_const_ptr(ctx, in), in->stride,
        out->rows, out->cols, scalar);
    if (status != CML_OK) goto fail;

    if (ctx->backend_ops->uses_device) {
        mark_device_write(out);
    } else {
        mark_host_write(out);
    }
    return CML_OK;

fail:
    set_backend_error(ctx, status, "backend scale failed");
    return status;
}

cml_status_t cml_backend_dot(cml_context_t *ctx, cml_tensor_t *out,
                             const cml_tensor_t *a, const cml_tensor_t *b) {
    if (ctx == NULL || ctx->backend_ops == NULL) return CML_INVALID_ARG;
    if (ctx->status != CML_OK) return ctx->status;
    if (check_tensor_ctx(ctx, out) != CML_OK) return CML_INVALID_ARG;
    if (check_tensor_ctx(ctx, a) != CML_OK) return CML_INVALID_ARG;
    if (check_tensor_ctx(ctx, b) != CML_OK) return CML_INVALID_ARG;
    size_t _dims[] = {a->rows, b->cols, a->cols, a->stride, b->stride, out->stride};
    if (check_int_dims(ctx, _dims, sizeof(_dims)/sizeof(_dims[0])) != CML_OK) return CML_INVALID_ARG;
    cml_status_t status;
    if (ctx->backend_ops->uses_device) {
        status = ensure_device(ctx, a);
        if (status != CML_OK) goto fail;
        status = ensure_device(ctx, b);
        if (status != CML_OK) goto fail;
        status = ensure_device_allocated(ctx, tensor_storage(out));
        if (status != CML_OK) goto fail;
    } else {
        status = ensure_host(ctx, a);
        if (status != CML_OK) goto fail;
        status = ensure_host(ctx, b);
        if (status != CML_OK) goto fail;
    }

    status = ctx->backend_ops->dot(
        ctx->backend_state,
        tensor_ptr(ctx, out), out->stride,
        tensor_const_ptr(ctx, a), a->stride,
        tensor_const_ptr(ctx, b), b->stride,
        a->rows, b->cols, a->cols);
    if (status != CML_OK) goto fail;

    if (ctx->backend_ops->uses_device) {
        mark_device_write(out);
    } else {
        mark_host_write(out);
    }
    return CML_OK;

fail:
    set_backend_error(ctx, status, "backend dot failed");
    return status;
}

cml_status_t cml_backend_transpose(cml_context_t *ctx, cml_tensor_t *out, const cml_tensor_t *in) {
    if (ctx == NULL || ctx->backend_ops == NULL) return CML_INVALID_ARG;
    if (ctx->status != CML_OK) return ctx->status;
    if (check_tensor_ctx(ctx, out) != CML_OK) return CML_INVALID_ARG;
    if (check_tensor_ctx(ctx, in) != CML_OK) return CML_INVALID_ARG;
    size_t _dims[] = {in->rows, in->cols, in->stride, out->stride};
    if (check_int_dims(ctx, _dims, sizeof(_dims)/sizeof(_dims[0])) != CML_OK) return CML_INVALID_ARG;
    cml_status_t status;
    if (ctx->backend_ops->uses_device) {
        status = ensure_device(ctx, in);
        if (status != CML_OK) goto fail;
        status = ensure_device_allocated(ctx, tensor_storage(out));
        if (status != CML_OK) goto fail;
    } else {
        status = ensure_host(ctx, in);
        if (status != CML_OK) goto fail;
    }

    status = ctx->backend_ops->transpose(
        ctx->backend_state,
        tensor_ptr(ctx, out), out->stride,
        tensor_const_ptr(ctx, in), in->stride,
        in->rows, in->cols);
    if (status != CML_OK) goto fail;

    if (ctx->backend_ops->uses_device) {
        mark_device_write(out);
    } else {
        mark_host_write(out);
    }
    return CML_OK;

fail:
    set_backend_error(ctx, status, "backend transpose failed");
    return status;
}

cml_status_t cml_backend_sum(cml_context_t *ctx, const cml_tensor_t *in, float *out) {
    if (ctx == NULL || ctx->backend_ops == NULL) return CML_INVALID_ARG;
    if (ctx->status != CML_OK) return ctx->status;
    if (check_tensor_ctx(ctx, in) != CML_OK) return CML_INVALID_ARG;
    size_t _dims[] = {in->rows, in->cols, in->stride};
    if (check_int_dims(ctx, _dims, sizeof(_dims)/sizeof(_dims[0])) != CML_OK) return CML_INVALID_ARG;
    cml_status_t status;
    if (ctx->backend_ops->uses_device) {
        status = ensure_device(ctx, in);
    } else {
        status = ensure_host(ctx, in);
    }
    if (status != CML_OK) {
        set_backend_error(ctx, status, "backend sum input sync failed");
        return status;
    }

    status = ctx->backend_ops->sum(
        ctx->backend_state, tensor_const_ptr(ctx, in), in->stride, in->rows, in->cols, out);
    if (status != CML_OK) {
        set_backend_error(ctx, status, "backend sum failed");
    }
    return status;
}

cml_status_t cml_backend_unary(cml_context_t *ctx, cml_tensor_t *out,
                               const cml_tensor_t *in, cml_unary_op_t op) {
    if (ctx == NULL || ctx->backend_ops == NULL) return CML_INVALID_ARG;
    if (ctx->status != CML_OK) return ctx->status;
    if (check_tensor_ctx(ctx, out) != CML_OK) return CML_INVALID_ARG;
    if (check_tensor_ctx(ctx, in) != CML_OK) return CML_INVALID_ARG;
    size_t _dims[] = {in->rows, in->cols, in->stride, out->stride};
    if (check_int_dims(ctx, _dims, sizeof(_dims)/sizeof(_dims[0])) != CML_OK) return CML_INVALID_ARG;
    cml_status_t status;
    if (ctx->backend_ops->uses_device) {
        status = ensure_device(ctx, in);
        if (status != CML_OK) goto fail;
        status = ensure_device_allocated(ctx, tensor_storage(out));
        if (status != CML_OK) goto fail;
    } else {
        status = ensure_host(ctx, in);
        if (status != CML_OK) goto fail;
    }

    status = ctx->backend_ops->unary(
        ctx->backend_state,
        tensor_ptr(ctx, out), out->stride,
        tensor_const_ptr(ctx, in), in->stride,
        in->rows, in->cols, op);
    if (status != CML_OK) goto fail;

    if (ctx->backend_ops->uses_device) {
        mark_device_write(out);
    } else {
        mark_host_write(out);
    }
    return CML_OK;

fail:
    set_backend_error(ctx, status, "backend unary op failed");
    return status;
}

cml_status_t cml_backend_unary_grad(cml_context_t *ctx, cml_tensor_t *out,
                                    const cml_tensor_t *x, const cml_tensor_t *grad,
                                    cml_unary_op_t op) {
    if (ctx == NULL || ctx->backend_ops == NULL) return CML_INVALID_ARG;
    if (ctx->status != CML_OK) return ctx->status;
    if (check_tensor_ctx(ctx, out) != CML_OK) return CML_INVALID_ARG;
    if (check_tensor_ctx(ctx, x) != CML_OK) return CML_INVALID_ARG;
    if (check_tensor_ctx(ctx, grad) != CML_OK) return CML_INVALID_ARG;
    size_t _dims[] = {x->rows, x->cols, x->stride, out->stride, grad->stride};
    if (check_int_dims(ctx, _dims, sizeof(_dims)/sizeof(_dims[0])) != CML_OK) return CML_INVALID_ARG;
    cml_status_t status;
    if (ctx->backend_ops->uses_device) {
        status = ensure_device(ctx, x);
        if (status != CML_OK) goto fail;
        status = ensure_device(ctx, grad);
        if (status != CML_OK) goto fail;
        status = ensure_device_allocated(ctx, tensor_storage(out));
        if (status != CML_OK) goto fail;
    } else {
        status = ensure_host(ctx, x);
        if (status != CML_OK) goto fail;
        status = ensure_host(ctx, grad);
        if (status != CML_OK) goto fail;
    }

    status = ctx->backend_ops->unary_grad(
        ctx->backend_state,
        tensor_ptr(ctx, out), out->stride,
        tensor_const_ptr(ctx, x), x->stride,
        tensor_const_ptr(ctx, grad), grad->stride,
        x->rows, x->cols, op);
    if (status != CML_OK) goto fail;

    if (ctx->backend_ops->uses_device) {
        mark_device_write(out);
    } else {
        mark_host_write(out);
    }
    return CML_OK;

fail:
    set_backend_error(ctx, status, "backend unary_grad failed");
    return status;
}

cml_status_t cml_backend_sum_rows(cml_context_t *ctx, cml_tensor_t *out,
                                  const cml_tensor_t *in) {
    if (ctx == NULL || ctx->backend_ops == NULL) return CML_INVALID_ARG;
    if (ctx->status != CML_OK) return ctx->status;
    if (check_tensor_ctx(ctx, out) != CML_OK) return CML_INVALID_ARG;
    if (check_tensor_ctx(ctx, in) != CML_OK) return CML_INVALID_ARG;
    size_t _dims[] = {in->rows, in->cols, in->stride};
    if (check_int_dims(ctx, _dims, sizeof(_dims)/sizeof(_dims[0])) != CML_OK) return CML_INVALID_ARG;
    cml_status_t status;
    if (ctx->backend_ops->uses_device) {
        status = ensure_device(ctx, in);
        if (status != CML_OK) goto fail;
        status = ensure_device_allocated(ctx, tensor_storage(out));
        if (status != CML_OK) goto fail;
    } else {
        status = ensure_host(ctx, in);
        if (status != CML_OK) goto fail;
    }

    status = ctx->backend_ops->sum_rows(
        ctx->backend_state,
        tensor_ptr(ctx, out),
        tensor_const_ptr(ctx, in), in->stride,
        in->rows, in->cols);
    if (status != CML_OK) goto fail;

    if (ctx->backend_ops->uses_device) {
        mark_device_write(out);
    } else {
        mark_host_write(out);
    }
    return CML_OK;

fail:
    set_backend_error(ctx, status, "backend sum_rows failed");
    return status;
}

cml_status_t cml_backend_accum_scaled(cml_context_t *ctx, cml_tensor_t *dst,
                                      const cml_tensor_t *src, float scale) {
    if (ctx == NULL || ctx->backend_ops == NULL) return CML_INVALID_ARG;
    if (ctx->status != CML_OK) return ctx->status;
    if (check_tensor_ctx(ctx, dst) != CML_OK) return CML_INVALID_ARG;
    if (check_tensor_ctx(ctx, src) != CML_OK) return CML_INVALID_ARG;
    size_t _dims[] = {dst->rows, dst->cols, dst->stride, src->stride};
    if (check_int_dims(ctx, _dims, sizeof(_dims)/sizeof(_dims[0])) != CML_OK) return CML_INVALID_ARG;
    cml_status_t status;
    if (ctx->backend_ops->uses_device) {
        status = ensure_device(ctx, dst);
        if (status != CML_OK) goto fail;
        status = ensure_device(ctx, src);
        if (status != CML_OK) goto fail;
    } else {
        status = ensure_host(ctx, dst);
        if (status != CML_OK) goto fail;
        status = ensure_host(ctx, src);
        if (status != CML_OK) goto fail;
    }

    status = ctx->backend_ops->accum_scaled(
        ctx->backend_state,
        tensor_ptr(ctx, dst), dst->stride,
        tensor_const_ptr(ctx, src), src->stride,
        dst->rows, dst->cols, scale);
    if (status != CML_OK) goto fail;

    if (ctx->backend_ops->uses_device) {
        mark_device_write(dst);
    } else {
        mark_host_write(dst);
    }
    return CML_OK;

fail:
    set_backend_error(ctx, status, "backend accum failed");
    return status;
}

cml_status_t cml_backend_adam_step(cml_context_t *ctx,
                                   cml_tensor_t *p, cml_tensor_t *m, cml_tensor_t *v,
                                   const cml_tensor_t *g,
                                   float lr, float beta1, float beta2, float eps,
                                   float bc1, float bc2) {
    if (ctx == NULL || ctx->backend_ops == NULL) return CML_INVALID_ARG;
    if (ctx->status != CML_OK) return ctx->status;
    if (ctx->backend_ops->adam_step == NULL) return CML_INVALID_ARG;
    if (check_tensor_ctx(ctx, p) != CML_OK) return CML_INVALID_ARG;
    if (check_tensor_ctx(ctx, m) != CML_OK) return CML_INVALID_ARG;
    if (check_tensor_ctx(ctx, v) != CML_OK) return CML_INVALID_ARG;
    if (check_tensor_ctx(ctx, g) != CML_OK) return CML_INVALID_ARG;
    size_t _dims[] = {p->rows, p->cols, p->stride, m->stride, v->stride, g->stride};
    if (check_int_dims(ctx, _dims, sizeof(_dims)/sizeof(_dims[0])) != CML_OK) return CML_INVALID_ARG;
    cml_status_t status;
    if (ctx->backend_ops->uses_device) {
        status = ensure_device(ctx, p);
        if (status != CML_OK) goto fail;
        status = ensure_device(ctx, m);
        if (status != CML_OK) goto fail;
        status = ensure_device(ctx, v);
        if (status != CML_OK) goto fail;
        status = ensure_device(ctx, g);
        if (status != CML_OK) goto fail;
    } else {
        status = ensure_host(ctx, p);
        if (status != CML_OK) goto fail;
        status = ensure_host(ctx, m);
        if (status != CML_OK) goto fail;
        status = ensure_host(ctx, v);
        if (status != CML_OK) goto fail;
        status = ensure_host(ctx, g);
        if (status != CML_OK) goto fail;
    }

    status = ctx->backend_ops->adam_step(
        ctx->backend_state,
        tensor_ptr(ctx, p), p->stride,
        tensor_ptr(ctx, m), m->stride,
        tensor_ptr(ctx, v), v->stride,
        tensor_const_ptr(ctx, g), g->stride,
        p->rows, p->cols,
        lr, beta1, beta2, eps, bc1, bc2);
    if (status != CML_OK) goto fail;

    if (ctx->backend_ops->uses_device) {
        mark_device_write(p);
        mark_device_write(m);
        mark_device_write(v);
    } else {
        mark_host_write(p);
        mark_host_write(m);
        mark_host_write(v);
    }
    return CML_OK;

fail:
    set_backend_error(ctx, status, "backend adam_step failed");
    return status;
}

cml_status_t cml_backend_add_bias(cml_context_t *ctx, cml_tensor_t *out,
                                  const cml_tensor_t *a, const cml_tensor_t *b) {
    if (ctx == NULL || ctx->backend_ops == NULL) return CML_INVALID_ARG;
    if (ctx->status != CML_OK) return ctx->status;
    if (check_tensor_ctx(ctx, out) != CML_OK) return CML_INVALID_ARG;
    if (check_tensor_ctx(ctx, a) != CML_OK) return CML_INVALID_ARG;
    if (check_tensor_ctx(ctx, b) != CML_OK) return CML_INVALID_ARG;
    size_t _dims[] = {out->rows, out->cols, out->stride, a->stride, b->stride};
    if (check_int_dims(ctx, _dims, sizeof(_dims)/sizeof(_dims[0])) != CML_OK) return CML_INVALID_ARG;
    cml_status_t status;
    if (ctx->backend_ops->uses_device) {
        status = ensure_device(ctx, a);
        if (status != CML_OK) goto fail;
        status = ensure_device(ctx, b);
        if (status != CML_OK) goto fail;
        status = ensure_device_allocated(ctx, tensor_storage(out));
        if (status != CML_OK) goto fail;
    } else {
        status = ensure_host(ctx, a);
        if (status != CML_OK) goto fail;
        status = ensure_host(ctx, b);
        if (status != CML_OK) goto fail;
    }

    status = ctx->backend_ops->add_bias(
        ctx->backend_state,
        tensor_ptr(ctx, out), out->stride,
        tensor_const_ptr(ctx, a), a->stride,
        tensor_const_ptr(ctx, b), b->stride,
        out->rows, out->cols);
    if (status != CML_OK) goto fail;

    if (ctx->backend_ops->uses_device) {
        mark_device_write(out);
    } else {
        mark_host_write(out);
    }
    return CML_OK;

fail:
    set_backend_error(ctx, status, "backend add-bias failed");
    return status;
}

cml_status_t cml_backend_softmax_xent_forward(cml_context_t *ctx,
                                              cml_tensor_t *softmax_out,
                                              const cml_tensor_t *logits,
                                              const cml_tensor_t *targets,
                                              float *loss_out) {
    if (ctx == NULL || ctx->backend_ops == NULL) return CML_INVALID_ARG;
    if (ctx->status != CML_OK) return ctx->status;
    if (ctx->backend_ops->softmax_xent_forward == NULL) return CML_INVALID_ARG;
    if (check_tensor_ctx(ctx, softmax_out) != CML_OK) return CML_INVALID_ARG;
    if (check_tensor_ctx(ctx, logits) != CML_OK) return CML_INVALID_ARG;
    if (check_tensor_ctx(ctx, targets) != CML_OK) return CML_INVALID_ARG;
    size_t _dims[] = {softmax_out->rows, softmax_out->cols, softmax_out->stride,
                      logits->stride, targets->stride};
    if (check_int_dims(ctx, _dims, sizeof(_dims)/sizeof(_dims[0])) != CML_OK) return CML_INVALID_ARG;

    cml_status_t status;
    if (ctx->backend_ops->uses_device) {
        status = ensure_device(ctx, logits);
        if (status != CML_OK) goto fail;
        status = ensure_device(ctx, targets);
        if (status != CML_OK) goto fail;
        status = ensure_device_allocated(ctx, tensor_storage(softmax_out));
        if (status != CML_OK) goto fail;
    } else {
        status = ensure_host(ctx, logits);
        if (status != CML_OK) goto fail;
        status = ensure_host(ctx, targets);
        if (status != CML_OK) goto fail;
    }

    status = ctx->backend_ops->softmax_xent_forward(
        ctx->backend_state,
        tensor_ptr(ctx, softmax_out), softmax_out->stride,
        tensor_const_ptr(ctx, logits), logits->stride,
        tensor_const_ptr(ctx, targets), targets->stride,
        softmax_out->rows, softmax_out->cols, loss_out);
    if (status != CML_OK) goto fail;

    if (ctx->backend_ops->uses_device) {
        mark_device_write(softmax_out);
    } else {
        mark_host_write(softmax_out);
    }
    return CML_OK;

fail:
    set_backend_error(ctx, status, "backend softmax_xent_forward failed");
    return status;
}

cml_status_t cml_backend_softmax_xent_backward(cml_context_t *ctx,
                                               cml_tensor_t *dlogits,
                                               const cml_tensor_t *softmax,
                                               const cml_tensor_t *targets,
                                               float scale) {
    if (ctx == NULL || ctx->backend_ops == NULL) return CML_INVALID_ARG;
    if (ctx->status != CML_OK) return ctx->status;
    if (ctx->backend_ops->softmax_xent_backward == NULL) return CML_INVALID_ARG;
    if (check_tensor_ctx(ctx, dlogits) != CML_OK) return CML_INVALID_ARG;
    if (check_tensor_ctx(ctx, softmax) != CML_OK) return CML_INVALID_ARG;
    if (check_tensor_ctx(ctx, targets) != CML_OK) return CML_INVALID_ARG;
    size_t _dims[] = {dlogits->rows, dlogits->cols, dlogits->stride,
                      softmax->stride, targets->stride};
    if (check_int_dims(ctx, _dims, sizeof(_dims)/sizeof(_dims[0])) != CML_OK) return CML_INVALID_ARG;

    cml_status_t status;
    if (ctx->backend_ops->uses_device) {
        status = ensure_device(ctx, softmax);
        if (status != CML_OK) goto fail;
        status = ensure_device(ctx, targets);
        if (status != CML_OK) goto fail;
        status = ensure_device_allocated(ctx, tensor_storage(dlogits));
        if (status != CML_OK) goto fail;
    } else {
        status = ensure_host(ctx, softmax);
        if (status != CML_OK) goto fail;
        status = ensure_host(ctx, targets);
        if (status != CML_OK) goto fail;
    }

    status = ctx->backend_ops->softmax_xent_backward(
        ctx->backend_state,
        tensor_ptr(ctx, dlogits), dlogits->stride,
        tensor_const_ptr(ctx, softmax), softmax->stride,
        tensor_const_ptr(ctx, targets), targets->stride,
        dlogits->rows, dlogits->cols, scale);
    if (status != CML_OK) goto fail;

    if (ctx->backend_ops->uses_device) {
        mark_device_write(dlogits);
    } else {
        mark_host_write(dlogits);
    }
    return CML_OK;

fail:
    set_backend_error(ctx, status, "backend softmax_xent_backward failed");
    return status;
}
