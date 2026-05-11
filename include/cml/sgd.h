#ifndef CML_SGD_H
#define CML_SGD_H

#include "context.h"
#include "tensor.h"

typedef struct cml_sgd_s cml_sgd_t;

// Initialize stochastic gradient descent
cml_sgd_t *cml_sgd_init(cml_context_t *ctx, float lr);

// Applies param -= lr * grad for each param that has a gradient.
// Call after cml_backward and before cml_tape_clear.
void cml_sgd_step(cml_sgd_t *opt, cml_tensor_t **params, size_t n_params);

#endif
