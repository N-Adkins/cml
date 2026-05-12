#include "backend.h"
#include "../context.h"
#include "../tensor.h"

#include <stdlib.h>

static void set_backend_error(cml_context_t *ctx, cml_status_t status, const char *msg) {
    if (ctx != NULL && status != CML_OK) {
        cml_context_error(ctx, status, msg);
    }
}

static cml_tensor_t *tensor_storage(const cml_tensor_t *tensor) {
    if (tensor == NULL) return NULL;
    return tensor->storage;
}

static size_t storage_elements(const cml_tensor_t *storage) {
    return storage->rows * storage->stride;
}

static cml_status_t register_device_allocation(cml_context_t *ctx, float *ptr) {
    cml_device_alloc_t *node = malloc(sizeof(cml_device_alloc_t));
    if (node == NULL) {
        return CML_OUT_OF_MEMORY;
    }
    node->ptr = ptr;
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

    status = register_device_allocation(ctx, device_ptr);
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
