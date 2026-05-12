#include "context.h"
#include "backend/backend.h"

#include <stdlib.h>

cml_context_t *cml_init(size_t start_size) {
    return cml_init_with_backend(start_size, CML_BACKEND_CPU);
}

cml_context_t *cml_init_with_backend(size_t start_size, cml_backend_t backend) {
    struct cml_context_s *ctx = malloc(sizeof(struct cml_context_s));
    if (ctx == NULL) {
        return NULL;
    }

    cml_status_t arena_status = cml_arena_init(&ctx->arena, start_size);
    if (arena_status != CML_OK) {
        free(ctx);
        return NULL;
    }
    ctx->status = CML_OK;
    ctx->error_msg = NULL;
    ctx->backend_ops = NULL; // cml_backend_init populates this on success
    ctx->backend_state = NULL;
    ctx->device_allocs = NULL;
    ctx->tape_head = NULL;
    ctx->grad_enabled = true;
    cml_rng_seed(&ctx->rng, 0u);

    if (cml_backend_init(ctx, backend) != CML_OK) {
        cml_backend_deinit(ctx); // safe: backend_ops stays NULL on failure
        cml_arena_deinit(&ctx->arena);
        free(ctx);
        return NULL;
    }

    return ctx;
}

void cml_deinit(cml_context_t *ctx) {
    if (ctx == NULL) {
        return;
    }
    cml_backend_deinit(ctx);
    cml_arena_deinit(&ctx->arena);
    free(ctx);
}

cml_status_t cml_get_status(const cml_context_t *ctx) {
    if (ctx == NULL) return CML_INVALID_ARG;
    return ctx->status;
}

cml_backend_t cml_get_backend(const cml_context_t *ctx) {
    if (ctx == NULL) return CML_BACKEND_CPU;
    return ctx->backend_kind;
}

const char *cml_get_error_msg(const cml_context_t *ctx) {
    if (ctx == NULL) {
        return "context is NULL";
    }
    if (ctx->status == CML_OK) {
        return NULL;
    }
    return ctx->error_msg;
}

void cml_seed(cml_context_t *ctx, uint64_t seed) {
    if (ctx == NULL) return;
    cml_rng_seed(&ctx->rng, seed);
}

bool cml_cuda_is_available(void) {
    return cml_backend_cuda_compiled();
}

void cml_context_error(cml_context_t *ctx, cml_status_t status, const char *error_msg) {
    if (ctx->status != CML_OK) {
        // For now, we won't allow errors overwriting other ones - the logic being
        // that the program is already in an invalid state.
        return;
    }

    ctx->status = status;
    ctx->error_msg = error_msg;
}

void cml_context_clear_status(cml_context_t *ctx) {
    if (ctx == NULL) return;
    ctx->status = CML_OK;
    ctx->error_msg = NULL;
}
