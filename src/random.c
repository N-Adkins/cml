#include "random.h"

// PCG32 default multiplier
#define CML_RNG_MULTIPLIER 6364136223846793005ULL

void cml_rng_seed(cml_rng_t *rng, uint64_t seed) {
    rng->state = 0u;
    rng->inc = (seed << 1u) | 1u;
    (void)cml_rng_next_u32(rng);
    rng->state += seed;
    (void)cml_rng_next_u32(rng);
}

uint32_t cml_rng_next_u32(cml_rng_t *rng) {
    uint64_t oldstate = rng->state;
    rng->state = oldstate * CML_RNG_MULTIPLIER + rng->inc;
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((32u - rot) & 31u));
}

float cml_rng_next_uniform(cml_rng_t *rng) {
    // 24 bits for a uniform [0, 1) float
    uint32_t bits = cml_rng_next_u32(rng) >> 8;
    return (float)bits / (float)(1u << 24);
}

size_t cml_rng_next_below(cml_rng_t *rng, size_t bound) {
    if (bound <= 1) return 0;
    if (bound <= (size_t)0xFFFFFFFFu) {
        uint32_t b = (uint32_t)bound;
        // rejection sampling for unbiased modulo
        uint32_t threshold = (uint32_t)(((uint64_t)1u << 32) % (uint64_t)b);
        for (;;) {
            uint32_t r = cml_rng_next_u32(rng);
            if (r >= threshold) return (size_t)(r % b);
        }
    }
    uint64_t b = (uint64_t)bound;
    uint64_t limit = UINT64_MAX - (UINT64_MAX % b);
    for (;;) {
        uint64_t hi = (uint64_t)cml_rng_next_u32(rng) << 32;
        uint64_t lo = (uint64_t)cml_rng_next_u32(rng);
        uint64_t r = hi | lo;
        if (r < limit) return (size_t)(r % b);
    }
}
