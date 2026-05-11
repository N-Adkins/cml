#include "arena.h"

#include <stdlib.h>

static cml_arena_chunk_t *chunk_create(size_t capacity) {
    cml_arena_chunk_t *chunk = malloc(sizeof(*chunk));
    if (chunk == NULL) {
        return NULL;
    }
    chunk->buffer = malloc(capacity);
    if (chunk->buffer == NULL) {
        free(chunk);
        return NULL;
    }
    chunk->capacity = capacity;
    chunk->used = 0;
    chunk->next = NULL;
    return chunk;
}

cml_status_t cml_arena_init(cml_arena_t *arena, size_t capacity_bytes) {
    if (arena == NULL) {
        return CML_INVALID_ARG;
    }

    arena->head = chunk_create(capacity_bytes);
    if (arena->head == NULL) {
        return CML_OUT_OF_MEMORY;
    }
    arena->current = arena->head;
    arena->offset = 0;

    return CML_OK;
}

void cml_arena_deinit(cml_arena_t *arena) {
    if (arena == NULL) {
        return;
    }
    cml_arena_chunk_t *chunk = arena->head;
    while (chunk != NULL) {
        cml_arena_chunk_t *next = chunk->next;
        free(chunk->buffer);
        free(chunk);
        chunk = next;
    }
    arena->head = NULL;
    arena->current = NULL;
    arena->offset = 0;
}

void *cml_arena_alloc(cml_arena_t *arena, size_t size) {
    size_t aligned = (size + 7) & ~7ULL; // aligned to 8 bytes
    cml_arena_chunk_t *chunk = arena->current;

    if (chunk->used + aligned > chunk->capacity) {
        // Try to reuse a later chunk that already has room (post-rewind /
        // post-reset chunks are empty), otherwise append a new one.
        cml_arena_chunk_t *next = chunk->next;
        while (next != NULL && next->capacity - next->used < aligned) {
            next = next->next;
        }
        if (next == NULL) {
            size_t grown = chunk->capacity * 2;
            if (grown < aligned) {
                grown = aligned;
            }
            cml_arena_chunk_t *fresh = chunk_create(grown);
            if (fresh == NULL) {
                return NULL;
            }
            cml_arena_chunk_t *tail = chunk;
            while (tail->next != NULL) {
                tail = tail->next;
            }
            tail->next = fresh;
            next = fresh;
        }
        chunk = next;
        arena->current = chunk;
    }

    void *ptr = chunk->buffer + chunk->used;
    chunk->used += aligned;
    arena->offset += aligned;
    return ptr;
}

void cml_arena_reset(cml_arena_t *arena) {
    if (arena == NULL) return;
    cml_arena_chunk_t *chunk = arena->head;
    while (chunk != NULL) {
        chunk->used = 0;
        chunk = chunk->next;
    }
    arena->current = arena->head;
    arena->offset = 0;
}

size_t cml_arena_mark(const cml_arena_t *arena) {
    if (arena == NULL) return 0;
    return arena->offset;
}

void cml_arena_rewind(cml_arena_t *arena, size_t mark) {
    if (arena == NULL) return;
    if (mark > arena->offset) return;

    size_t remaining = mark;
    cml_arena_chunk_t *chunk = arena->head;
    cml_arena_chunk_t *landed = arena->head;
    while (chunk != NULL) {
        if (remaining >= chunk->used) {
            remaining -= chunk->used;
            landed = chunk;
            chunk = chunk->next;
        } else {
            chunk->used = remaining;
            landed = chunk;
            chunk = chunk->next;
            while (chunk != NULL) {
                chunk->used = 0;
                chunk = chunk->next;
            }
            break;
        }
    }
    arena->current = landed;
    arena->offset = mark;
}
