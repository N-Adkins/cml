#ifndef CML_INTERNAL_CONTEXT_H
#define CML_INTERNAL_CONTEXT_H

#include <cml/context.h>
#include "memory/arena.h"

#include <stdbool.h>

struct cml_tape_node_s;

struct cml_context_s {
    cml_arena_t arena;
    cml_status_t status;
    const char *error_msg;

    // Autograd
    struct cml_tape_node_s *tape_head;
    bool grad_enabled;
};

void cml_context_error(cml_context_t *ctx, cml_status_t status, const char *error_msg);

#endif
