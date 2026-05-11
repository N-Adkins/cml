#ifndef CML_CONTEXT_H
#define CML_CONTEXT_H

#include <stddef.h>

#include "status.h"

// This is the entire context for an AI program. It is opaque and hidden from the user - but
// it has some internals used for things like memory allocation.

typedef struct cml_context_s cml_context_t;

// Spins up an entire CML instance - notably there is no global state so multiple should be able
// to run at the same time.
cml_context_t *cml_init(size_t size);
void cml_deinit(cml_context_t *ctx);

// Returns the current status code of the CML instance
cml_status_t cml_get_status(cml_context_t *ctx);

// If cml_get_status returned anything but CML_OK, this will
// return a more detailed error message.
const char *cml_get_error_msg(cml_context_t *ctx);

#endif
