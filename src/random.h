#ifndef CML_INTERNAL_RANDOM_H
#define CML_INTERNAL_RANDOM_H

#include <stddef.h>
#include <stdint.h>

// PCG32-based PRNG, owned per-context so contexts are independent and seedable

typedef struct {
    uint64_t state;
    uint64_t inc;
} cml_rng_t;

void cml_rng_seed(cml_rng_t *rng, uint64_t seed);
uint32_t cml_rng_next_u32(cml_rng_t *rng);
float cml_rng_next_uniform(cml_rng_t *rng);
size_t cml_rng_next_below(cml_rng_t *rng, size_t bound);

#endif
