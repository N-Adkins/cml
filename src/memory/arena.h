#ifndef CML_INTERNAL_MEMORY_ARENA_H
#define CML_INTERNAL_MEMORY_ARENA_H

#include <cml/status.h>
#include <stddef.h>

// Very simple bump allocator
typedef struct {
    char *buffer;
    size_t capacity;
    size_t offset;
} cml_arena_t;

cml_status_t cml_arena_init(cml_arena_t *arena, size_t capacity_bytes);
void cml_arena_deinit(cml_arena_t *arena);
void *cml_arena_alloc(cml_arena_t *arena, size_t size);
void cml_arena_reset(cml_arena_t *arena);

#endif
