#include "arena.h"

#include <stdlib.h>

cml_status_t cml_arena_init(cml_arena_t *arena, size_t capacity_bytes) {
    if (arena == NULL) {
        return CML_INVALID_ARG;
    }
    
    arena->buffer = malloc(capacity_bytes);
    if (arena->buffer == NULL) {
        return CML_OUT_OF_MEMORY;
    }

    arena->capacity = capacity_bytes;
    arena->offset = 0;

    return CML_OK;
}

void cml_arena_deinit(cml_arena_t *arena) {
    if (arena != NULL) {
        if (arena->buffer != NULL) {
            free(arena->buffer);
        }
    }
}

void *cml_arena_alloc(cml_arena_t *arena, size_t size) {
    size_t aligned = (size + 7) & ~7ULL; // aligned to 8 bytes
    if (arena->offset + aligned > arena->capacity) {
        return NULL;
    }

    void* ptr = arena->buffer + arena->offset;
    arena->offset += aligned;
    return ptr;
}

void cml_arena_reset(cml_arena_t *arena) {
    arena->offset = 0;
}
