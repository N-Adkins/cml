#include "sgd.h"
#include "context.h"
#include "tensor.h"

#include <cblas.h>

cml_sgd_t *cml_sgd_init(cml_context_t *ctx, float lr) {
    if (ctx == NULL) return NULL;
    if (ctx->status != CML_OK) return NULL;

    cml_sgd_t *opt = cml_arena_alloc(&ctx->arena, sizeof(struct cml_sgd_s));
    if (opt == NULL) {
        cml_context_error(ctx, CML_OUT_OF_MEMORY, "sgd allocation failed");
        return NULL;
    }
    opt->lr = lr;
    return opt;
}

void cml_sgd_step(cml_sgd_t *opt, cml_tensor_t **params, size_t n_params) {
    if (opt == NULL || params == NULL) return;

    for (size_t i = 0; i < n_params; i++) {
        cml_tensor_t *p = params[i];
        if (p == NULL || p->grad == NULL) continue;
        for (size_t r = 0; r < p->rows; r++) {
            cblas_saxpy((int)p->cols, -opt->lr,
                        p->grad->data + r * p->grad->stride, 1,
                        p->data + r * p->stride, 1);
        }
    }
}
