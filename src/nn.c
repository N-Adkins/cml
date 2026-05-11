#include <cml/nn.h>
#include <cml/linear.h>
#include "context.h"

typedef enum {
    CML_MODULE_LINEAR = 0,
    CML_MODULE_RELU,
    CML_MODULE_SEQUENTIAL,
} cml_module_kind_t;

struct cml_module_s {
    cml_module_kind_t kind;
    union {
        cml_linear_t *linear;
        struct {
            cml_module_t **modules;
            size_t n_modules;
        } sequential;
    } as;
};

static cml_module_t *module_alloc(cml_context_t *ctx, cml_module_kind_t kind) {
    cml_module_t *module = cml_arena_alloc(&ctx->arena, sizeof(struct cml_module_s));
    if (module == NULL) {
        cml_context_error(ctx, CML_OUT_OF_MEMORY, "module allocation failed");
        return NULL;
    }
    module->kind = kind;
    return module;
}

cml_module_t *cml_nn_linear(cml_context_t *ctx, size_t in_features, size_t out_features) {
    if (ctx == NULL) return NULL;
    if (ctx->status != CML_OK) return NULL;

    cml_module_t *module = module_alloc(ctx, CML_MODULE_LINEAR);
    if (module == NULL) return NULL;

    module->as.linear = cml_linear_init(ctx, in_features, out_features);
    if (module->as.linear == NULL) return NULL;
    return module;
}

cml_module_t *cml_nn_relu(cml_context_t *ctx) {
    if (ctx == NULL) return NULL;
    if (ctx->status != CML_OK) return NULL;
    return module_alloc(ctx, CML_MODULE_RELU);
}

cml_module_t *cml_nn_sequential(cml_context_t *ctx, cml_module_t **modules, size_t n_modules) {
    if (ctx == NULL) return NULL;
    if (ctx->status != CML_OK) return NULL;
    if (n_modules > 0 && modules == NULL) {
        cml_context_error(ctx, CML_INVALID_ARG, "modules is NULL");
        return NULL;
    }

    cml_module_t *module = module_alloc(ctx, CML_MODULE_SEQUENTIAL);
    if (module == NULL) return NULL;

    module->as.sequential.modules = NULL;
    module->as.sequential.n_modules = n_modules;
    if (n_modules == 0) return module;

    module->as.sequential.modules = cml_arena_alloc(&ctx->arena, n_modules * sizeof(cml_module_t *));
    if (module->as.sequential.modules == NULL) {
        cml_context_error(ctx, CML_OUT_OF_MEMORY, "sequential module list allocation failed");
        return NULL;
    }

    for (size_t i = 0; i < n_modules; i++) {
        if (modules[i] == NULL) {
            cml_context_error(ctx, CML_INVALID_ARG, "sequential module contains NULL child");
            return NULL;
        }
        module->as.sequential.modules[i] = modules[i];
    }

    return module;
}

cml_tensor_t *cml_module_forward(cml_context_t *ctx, cml_module_t *module, cml_tensor_t *x) {
    if (ctx == NULL) return NULL;
    if (ctx->status != CML_OK) return NULL;
    if (module == NULL || x == NULL) {
        cml_context_error(ctx, CML_INVALID_ARG, "argument is NULL");
        return NULL;
    }

    switch (module->kind) {
        case CML_MODULE_LINEAR:
            return cml_linear_forward(ctx, module->as.linear, x);
        case CML_MODULE_RELU:
            return cml_tensor_relu(ctx, x);
        case CML_MODULE_SEQUENTIAL: {
            cml_tensor_t *out = x;
            for (size_t i = 0; i < module->as.sequential.n_modules; i++) {
                out = cml_module_forward(ctx, module->as.sequential.modules[i], out);
                if (out == NULL) return NULL;
            }
            return out;
        }
        default:
            cml_context_error(ctx, CML_INVALID_ARG, "unknown module kind");
            return NULL;
    }
}

size_t cml_module_param_count(const cml_module_t *module) {
    if (module == NULL) return 0;

    switch (module->kind) {
        case CML_MODULE_LINEAR:
            return 2;
        case CML_MODULE_RELU:
            return 0;
        case CML_MODULE_SEQUENTIAL: {
            size_t total = 0;
            for (size_t i = 0; i < module->as.sequential.n_modules; i++) {
                total += cml_module_param_count(module->as.sequential.modules[i]);
            }
            return total;
        }
        default:
            return 0;
    }
}

size_t cml_module_collect_params(cml_module_t *module, cml_tensor_t **params, size_t offset) {
    if (module == NULL || params == NULL) return offset;

    switch (module->kind) {
        case CML_MODULE_LINEAR:
            return cml_linear_collect_params(module->as.linear, params, offset);
        case CML_MODULE_RELU:
            return offset;
        case CML_MODULE_SEQUENTIAL:
            for (size_t i = 0; i < module->as.sequential.n_modules; i++) {
                offset = cml_module_collect_params(module->as.sequential.modules[i], params, offset);
            }
            return offset;
        default:
            return offset;
    }
}
