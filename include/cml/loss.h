#ifndef CML_LOSS_H
#define CML_LOSS_H

#include "context.h"
#include "tensor.h"

// Mean squared error: mean((pred - target)^2)
cml_tensor_t *cml_loss_mse(cml_context_t *ctx, cml_tensor_t *pred, cml_tensor_t *target);

// Categorical cross-entropy: -mean(target * log(pred))
// pred should contain class probabilities (e.g. softmax output); target is one-hot or soft labels.
cml_tensor_t *cml_loss_cross_entropy(cml_context_t *ctx, cml_tensor_t *pred, cml_tensor_t *target);

#endif
