#ifndef CML_TAPE_H
#define CML_TAPE_H

#include "context.h"
#include "tensor.h"

// This is a tape-machine based implementation for autograd. This is slower than the
// graph-based approach which requires a "build" step - for simplicitly just doing the tape for now.
//
// The entire tape is held within the passed ctx variable.

// Walks the tape in reverse and accumulates gradients into the parameter
// tensors. All intermediate tensors needed during the walk are allocated 
// in the context arena and are NOT released by cml_tape_clear (which only 
// nulls leaf grad pointers so the same parameters can be reused). Wrap 
// each forward+backward in cml_scope_begin/cml_scope_end to reclaim that 
// arena memory; otherwise the arena will grow each step.
void cml_backward(cml_context_t *ctx, cml_tensor_t *loss);
void cml_tape_clear(cml_context_t *ctx);
void cml_grad_enable(cml_context_t *ctx);
void cml_grad_disable(cml_context_t *ctx);

#endif
