#ifndef CML_TRAINER_H
#define CML_TRAINER_H

#include <stdbool.h>
#include <stddef.h>
#include "context.h"
#include "tensor.h"

typedef struct cml_trainer_s cml_trainer_t;

// User-supplied function that runs a forward pass and returns a scalar loss tensor.
typedef cml_tensor_t *(*cml_loss_fn)(cml_context_t *ctx, void *model,
                                      cml_tensor_t *x, cml_tensor_t *y);

cml_trainer_t *cml_trainer_init(cml_context_t *ctx,
                                 void *model,
                                 cml_loss_fn loss_fn,
                                 cml_tensor_t **params,
                                 size_t n_params,
                                 float lr);

// Runs the training loop for `epochs` iterations.
// When verbose is true, renders a single-line progress bar with epoch and loss updates.
void cml_trainer_fit(cml_context_t *ctx, cml_trainer_t *trainer,
                     cml_tensor_t *x, cml_tensor_t *y,
                     size_t epochs, bool verbose);

#endif
