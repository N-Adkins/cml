#ifndef CML_TRAINER_H
#define CML_TRAINER_H

#include <stdbool.h>
#include <stddef.h>
#include "context.h"
#include "data.h"
#include "optimizer.h"
#include "tensor.h"

// This is a framework for training the models to make it easier with a nice ui.
// You basically just throw everything in it and provide it a loss function and it will work.
//
// Invariant: params, optimizer, and any other long-lived state must be created
// before cml_trainer_fit/fit_loader. Each batch runs inside an internal scope
// that reclaims everything allocated during that batch

typedef struct cml_trainer_s cml_trainer_t;

// User-supplied function that runs a forward pass and returns a scalar loss tensor.
typedef cml_tensor_t *(*cml_loss_fn)(cml_context_t *ctx, void *model,
                                      cml_tensor_t *x, cml_tensor_t *y);

cml_trainer_t *cml_trainer_init(cml_context_t *ctx,
                                 void *model,
                                 cml_loss_fn loss_fn,
                                 cml_tensor_t **params,
                                 size_t n_params,
                                 cml_optimizer_t *opt);

// Runs the training loop for `epochs` iterations.
// When verbose is true, renders a progress bar with epoch and loss updates.
void cml_trainer_fit(cml_context_t *ctx, cml_trainer_t *trainer,
                     cml_tensor_t *x, cml_tensor_t *y,
                     size_t epochs, bool verbose);

// Runs the training loop over mini-batches produced by a data loader.
void cml_trainer_fit_loader(cml_context_t *ctx, cml_trainer_t *trainer,
                            cml_data_loader_t *loader,
                            size_t epochs, bool verbose);

#endif
