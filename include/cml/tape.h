#ifndef CML_TAPE_H
#define CML_TAPE_H

#include "context.h"
#include "tensor.h"

// This is a tape-machine based implementation for autograd. This is slower than the
// graph-based approach which requires a "build" step - for simplicitly just doing the tape for now.
//
// The entire tape is held within the passed ctx variable.

void cml_backward(cml_context_t *ctx, cml_tensor_t *loss);
void cml_tape_clear(cml_context_t *ctx);
void cml_grad_enable(cml_context_t *ctx);
void cml_grad_disable(cml_context_t *ctx);

#endif
