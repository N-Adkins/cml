#ifndef CML_INTERNAL_CONTEXT_H
#define CML_INTERNAL_CONTEXT_H

#include <cml/context.h>
#include "memory/arena.h"

// Opaque pointer impl for cml_context_t
struct cml_context_s {
    cml_arena_t arena;
    cml_status_t status;
    const char *error_msg;
};

void cml_context_error(cml_context_t *ctx, cml_status_t status, const char *error_msg);

#endif
