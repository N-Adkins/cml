#ifndef CML_CONTEXT_H
#define CML_CONTEXT_H

#include <stdbool.h>
#include <stddef.h>

// This is the entire context for an AI program. It is opaque and hidden from the user - but
// it has some internals used for things like memory allocation.

typedef struct cml_context_s cml_context_t;

typedef enum {
    CML_OK = 0,
    CML_INVALID_ARG,
    CML_OUT_OF_MEMORY,
    CML_BACKEND_UNAVAILABLE,
} cml_status_t;

typedef enum {
    CML_BACKEND_CPU = 0,
    CML_BACKEND_CUDA,
} cml_backend_t;

// Spins up an entire CML instance - notably there is no global state so multiple should be able
// to run at the same time.
cml_context_t *cml_init(size_t size);
// Same as cml_init, but allows selecting CPU or CUDA backend at context creation.
cml_context_t *cml_init_with_backend(size_t size, cml_backend_t backend);
void cml_deinit(cml_context_t *ctx);

// Returns the current status code of the CML instance
cml_status_t cml_get_status(cml_context_t *ctx);
// Returns the compute backend selected for this context.
cml_backend_t cml_get_backend(cml_context_t *ctx);

// If cml_get_status returned anything but CML_OK, this will
// return a more detailed error message.
const char *cml_get_error_msg(cml_context_t *ctx);

// True when this build was compiled with CUDA backend support.
bool cml_cuda_is_available(void);

#endif
