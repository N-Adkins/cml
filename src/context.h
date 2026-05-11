#ifndef CML_INTERNAL_CONTEXT_H
#define CML_INTERNAL_CONTEXT_H

#include <cml/context.h>
#include "memory/arena.h"

#include <stdbool.h>

struct cml_tape_node_s;
struct cml_backend_ops_s;

typedef struct cml_device_alloc_s {
    float *ptr;
    struct cml_device_alloc_s *next;
} cml_device_alloc_t;

struct cml_context_s {
    cml_arena_t arena;
    cml_status_t status;
    const char *error_msg;

    // Backend abstraction
    cml_backend_t backend_kind;
    const struct cml_backend_ops_s *backend_ops;
    void *backend_state; // Used by CUDA
    cml_device_alloc_t *device_allocs;

    // Autograd
    struct cml_tape_node_s *tape_head;
    bool grad_enabled;
};

void cml_context_error(cml_context_t *ctx, cml_status_t status, const char *error_msg);

#endif
