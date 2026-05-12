#ifndef CML_OPTIMIZER_H
#define CML_OPTIMIZER_H

#include <stddef.h>
#include "nn.h"
#include "tensor.h"

typedef struct cml_optimizer_s cml_optimizer_t;

void cml_optimizer_step(cml_optimizer_t *opt, cml_tensor_t **params, size_t n_params);

// Resets internal state of an optimizer
void cml_optimizer_reset(cml_optimizer_t *opt);

cml_optimizer_t *cml_optimizer_sgd(cml_context_t *ctx, float lr);

// Module is needed to get parameters
cml_optimizer_t *cml_optimizer_adam(cml_context_t *ctx, cml_module_t *module, float lr, 
                                    float beta1, float beta2, float eps);

#endif
