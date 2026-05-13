#ifndef CML_NN_H
#define CML_NN_H

#include <stddef.h>
#include "context.h"
#include "tensor.h"

typedef struct cml_module_s cml_module_t;

// Layer constructors.
cml_module_t *cml_nn_linear(cml_context_t *ctx, size_t in_features, size_t out_features);
cml_module_t *cml_nn_relu(cml_context_t *ctx);
cml_module_t *cml_nn_sequential(cml_context_t *ctx, cml_module_t **modules, size_t n_modules);

// Runs a module on input tensor x.
cml_tensor_t *cml_module_forward(cml_context_t *ctx, cml_module_t *module, cml_tensor_t *x);

// Parameter introspection for optimizers.
size_t cml_module_param_count(const cml_module_t *module);

// Walks the module tree and writes parameter pointers into params[offset..].
// Returns the new offset (number of params written + initial offset). If
// writes would exceed `capacity`, the function stops early and returns the
// offset reached so far.
size_t cml_module_collect_params(cml_module_t *module, cml_tensor_t **params,
                                 size_t capacity, size_t offset);

#endif
