#include <cml/scope.h>

#include "context.h"
#include "backend/backend.h"
#include "memory/arena.h"

cml_scope_t cml_scope_begin(cml_context_t *ctx) {
    cml_scope_t scope = {0, NULL};
    if (ctx == NULL) return scope;
    scope.arena_mark = cml_arena_mark(&ctx->arena);
    scope.device_mark = cml_backend_device_mark(ctx);
    return scope;
}

void cml_scope_end(cml_context_t *ctx, cml_scope_t scope) {
    if (ctx == NULL) return;
    // Device first: it walks the alloc list and nulls device_data on the storage
    // tensors it frees, which would be UAF if their arena memory were already gone
    cml_backend_device_rewind(ctx, scope.device_mark);
    cml_arena_rewind(&ctx->arena, scope.arena_mark);
}
