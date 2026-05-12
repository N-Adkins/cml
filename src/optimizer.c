#include <cml/optimizer.h>
#include <cml/context.h>
#include <cml/nn.h>
#include <cml/tensor.h>
#include "backend/backend.h"
#include "memory/arena.h"
#include "context.h"
#include "tensor.h"

#include <math.h>
#include <stdlib.h>

typedef enum {
    CML_OPT_SGD,
    CML_OPT_ADAM
} cml_opt_kind_t;

struct cml_optimizer_s {
    cml_opt_kind_t kind;
    union {
        struct {
            float lr;
        } sgd;
        struct {
            float lr, beta1, beta2, eps;
            // params[i] is the leaf tensor whose moments are stored at m[i], v[i]
            cml_tensor_t **params;
            cml_tensor_t **m;
            cml_tensor_t **v;
            size_t t, n;
        } adam;
    } inner;
};

cml_optimizer_t *cml_optimizer_sgd(cml_context_t *ctx, float lr) {
    if (ctx == NULL) {
        return NULL;
    }
    if (ctx->status != CML_OK) {
        return NULL;
    }

    cml_optimizer_t *opt = cml_arena_alloc(&ctx->arena, sizeof(cml_optimizer_t));
    if (opt == NULL) {
        cml_context_error(ctx, CML_OUT_OF_MEMORY, "sgd optimizer allocation failed");
        return NULL;
    }

    opt->kind = CML_OPT_SGD;
    opt->inner.sgd.lr = lr;
    return opt;
}

cml_optimizer_t *cml_optimizer_adam(cml_context_t *ctx, cml_module_t *module, float lr, float beta1, float beta2, float eps) {
    if (ctx == NULL || module == NULL) {
        return NULL;
    }
    if (ctx->status != CML_OK) {
        return NULL;
    }

    const size_t n_params = cml_module_param_count(module);

    // Rewind any partial state allocated below if init fails midway
    size_t arena_mark = cml_arena_mark(&ctx->arena);
    void *device_mark = cml_backend_device_mark(ctx);

    cml_optimizer_t *opt = cml_arena_alloc(&ctx->arena, sizeof(cml_optimizer_t));
    cml_tensor_t **params = NULL;
    cml_tensor_t **m = NULL;
    cml_tensor_t **v = NULL;
    if (n_params > 0) {
        params = cml_arena_alloc(&ctx->arena, n_params * sizeof(cml_tensor_t *));
        m = cml_arena_alloc(&ctx->arena, n_params * sizeof(cml_tensor_t *));
        v = cml_arena_alloc(&ctx->arena, n_params * sizeof(cml_tensor_t *));
    }

    if (opt == NULL || (n_params > 0 && (params == NULL || m == NULL || v == NULL))) {
        cml_backend_device_rewind(ctx, device_mark);
        cml_arena_rewind(&ctx->arena, arena_mark);
        // Rollback succeeded; surface a fresh, retry-able OOM rather than whatever
        // partial state the failed allocation left behind
        cml_context_clear_status(ctx);
        cml_context_error(ctx, CML_OUT_OF_MEMORY, "adam optimizer allocation failed");
        return NULL;
    }

    if (n_params > 0) {
        cml_module_collect_params(module, params, 0);

        for (size_t i = 0; i < n_params; i++) {
            cml_tensor_t *param = params[i];
            const size_t param_rows = cml_tensor_rows(param);
            const size_t param_cols = cml_tensor_cols(param);
            m[i] = cml_tensor_init(ctx, param_rows, param_cols);
            v[i] = cml_tensor_init(ctx, param_rows, param_cols);
            if (m[i] == NULL || v[i] == NULL) {
                cml_backend_device_rewind(ctx, device_mark);
                cml_arena_rewind(&ctx->arena, arena_mark);
                cml_context_clear_status(ctx);
                cml_context_error(ctx, CML_OUT_OF_MEMORY, "adam moments allocation failed");
                return NULL;
            }
            // Force device allocation now so these buffers live above any
            // per-epoch device mark a trainer might take. Otherwise the first
            // adam_step would lazily allocate them inside the epoch and the
            // trainer's device_rewind would free them between iterations.
            cml_backend_tensor_to_device(ctx, m[i]);
            cml_backend_tensor_to_device(ctx, v[i]);
            if (ctx->status != CML_OK) {
                cml_backend_device_rewind(ctx, device_mark);
                cml_arena_rewind(&ctx->arena, arena_mark);
                cml_context_clear_status(ctx);
                cml_context_error(ctx, CML_BACKEND_UNAVAILABLE,
                                  "adam: failed to stage moments on device");
                return NULL;
            }
        }
    }

    opt->kind = CML_OPT_ADAM;
    opt->inner.adam.n = n_params;
    opt->inner.adam.lr = lr;
    opt->inner.adam.beta1 = beta1;
    opt->inner.adam.beta2 = beta2;
    opt->inner.adam.eps = eps;
    opt->inner.adam.t = 0;
    opt->inner.adam.params = params;
    opt->inner.adam.m = m;
    opt->inner.adam.v = v;

    return opt;
}

static void sgd_step(cml_optimizer_t *opt, cml_tensor_t **params, size_t n_params) {
    for (size_t i = 0; i < n_params; i++) {
        cml_tensor_t *p = params[i];
        if (p == NULL || p->grad == NULL) continue;
        cml_backend_accum_scaled(p->ctx, p, p->grad, -opt->inner.sgd.lr);
    }
}

static cml_tensor_t **adam_find_moments(cml_optimizer_t *opt, cml_tensor_t *p,
                                         cml_tensor_t **m_out, cml_tensor_t **v_out) {
    for (size_t i = 0; i < opt->inner.adam.n; i++) {
        if (opt->inner.adam.params[i] == p) {
            *m_out = opt->inner.adam.m[i];
            *v_out = opt->inner.adam.v[i];
            return m_out;
        }
    }
    return NULL;
}

static void adam_step(cml_optimizer_t *opt, cml_tensor_t **params, size_t n_params) {
    opt->inner.adam.t++;

    const float bc1 = 1.0f - powf(opt->inner.adam.beta1, (float)opt->inner.adam.t);
    const float bc2 = 1.0f - powf(opt->inner.adam.beta2, (float)opt->inner.adam.t);

    for (size_t i = 0; i < n_params; i++) {
        cml_tensor_t *p = params[i];
        if (p == NULL) continue;
        cml_tensor_t *gradient = cml_tensor_grad(p);
        if (gradient == NULL) continue;

        cml_tensor_t *m = NULL;
        cml_tensor_t *v = NULL;
        if (adam_find_moments(opt, p, &m, &v) == NULL) {
            // Caller passed a param the optimizer wasn't built for
            cml_context_error(p->ctx, CML_INVALID_ARG,
                              "adam: parameter not registered with this optimizer");
            return;
        }

        cml_backend_adam_step(p->ctx, p, m, v, gradient,
                              opt->inner.adam.lr,
                              opt->inner.adam.beta1, opt->inner.adam.beta2,
                              opt->inner.adam.eps, bc1, bc2);
    }
}

void cml_optimizer_step(cml_optimizer_t *opt, cml_tensor_t **params, size_t n_params) {
    if (opt == NULL || params == NULL) return;
    switch (opt->kind) {
    case CML_OPT_SGD:
        sgd_step(opt, params, n_params);
        break;
    case CML_OPT_ADAM:
        adam_step(opt, params, n_params);
        break;
    }
}

void cml_optimizer_reset(cml_optimizer_t *opt) {
    if (opt == NULL) return;
    switch (opt->kind) {
    case CML_OPT_SGD:
        break;
    case CML_OPT_ADAM:
        opt->inner.adam.t = 0;
        for (size_t i = 0; i < opt->inner.adam.n; i++) {
            cml_tensor_zero(opt->inner.adam.m[i]);
            cml_tensor_zero(opt->inner.adam.v[i]);
        }
        break;
    }
}
