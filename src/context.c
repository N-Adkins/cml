#include "context.h"

#include <stdlib.h>

cml_context_t *cml_init(size_t start_size) {
    struct cml_context_s *ctx = malloc(sizeof(struct cml_context_s));
    if (ctx == NULL) {
        return NULL;
    }
    
    cml_arena_init(&ctx->arena, start_size);
    ctx->status = CML_OK;
    ctx->error_msg = NULL;

    return ctx;
}

void cml_deinit(cml_context_t *ctx) {
    if (ctx == NULL) {
        return;
    }
    cml_arena_deinit(&ctx->arena);
    free(ctx);
}

cml_status_t cml_get_status(cml_context_t *ctx) {
    return ctx->status;
}

const char *cml_get_error_msg(cml_context_t *ctx) {
    if (ctx->status == CML_OK) {
        return NULL;
    }
    return ctx->error_msg;
}
