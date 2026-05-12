#ifndef CML_SCOPE_H
#define CML_SCOPE_H

#include <stddef.h>
#include "context.h"

// Pairs an arena mark with a device-allocation mark, freed together at scope_end

typedef struct {
    size_t arena_mark;
    void *device_mark;
} cml_scope_t;

cml_scope_t cml_scope_begin(cml_context_t *ctx);
void cml_scope_end(cml_context_t *ctx, cml_scope_t scope);

#endif
