#ifndef CML_LOSS_H
#define CML_LOSS_H

#include "context.h"
#include "tensor.h"

// Mean squared error: mean((pred - target)^2)
cml_tensor_t *cml_loss_mse(cml_context_t *ctx, cml_tensor_t *pred, cml_tensor_t *target);

// Categorical cross-entropy averaged over the batch: -(1/N) * sum(target * log(pred))
// pred should contain class probabilities (e.g. softmax output); target is one-hot or soft labels.
cml_tensor_t *cml_loss_cross_entropy(cml_context_t *ctx, cml_tensor_t *pred, cml_tensor_t *target);

// Numerically stable fused softmax + cross-entropy. Takes raw logits and one-hot
// or soft target probabilities; preferred over softmax followed by cross_entropy.
cml_tensor_t *cml_loss_softmax_cross_entropy(cml_context_t *ctx, cml_tensor_t *logits, cml_tensor_t *target);

#endif
