#ifndef CML_LINEAR_H
#define CML_LINEAR_H

#include "context.h"
#include "tensor.h"

typedef struct cml_linear_s cml_linear_t;

// Allocates a fully-connected layer with Xavier-uniform weights and zero bias.
// Weights are [in_features x out_features]; requires_grad is true on both.
cml_linear_t *cml_linear_init(cml_context_t *ctx, size_t in_features, size_t out_features);

// Computes x @ W + b. x must be [batch x in_features]; returns [batch x out_features].
cml_tensor_t *cml_linear_forward(cml_context_t *ctx, cml_linear_t *layer, cml_tensor_t *x);

cml_tensor_t *cml_linear_weight(const cml_linear_t *layer);
cml_tensor_t *cml_linear_bias(const cml_linear_t *layer);

#endif
