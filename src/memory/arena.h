#ifndef CML_INTERNAL_MEMORY_ARENA_H
#define CML_INTERNAL_MEMORY_ARENA_H

#include <cml/context.h>
#include <stddef.h>

typedef struct cml_arena_chunk_s {
    char *buffer;
    size_t capacity;
    size_t used;
    struct cml_arena_chunk_s *next;
} cml_arena_chunk_t;

// Growing bump allocator: when an allocation would overflow the current
// chunk, a new (larger) chunk is appended. Existing chunks are never
// resized, so pointers returned by previous allocations remain valid until
// reset/deinit.
typedef struct {
    cml_arena_chunk_t *head;
    cml_arena_chunk_t *current;
    size_t offset; // total bytes allocated across all chunks
} cml_arena_t;

cml_status_t cml_arena_init(cml_arena_t *arena, size_t capacity_bytes);
void cml_arena_deinit(cml_arena_t *arena);
void *cml_arena_alloc(cml_arena_t *arena, size_t size);
void cml_arena_reset(cml_arena_t *arena);
size_t cml_arena_mark(const cml_arena_t *arena);
void cml_arena_rewind(cml_arena_t *arena, size_t mark);

#endif
