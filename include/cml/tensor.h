#ifndef CML_TENSOR_H
#define CML_TENSOR_H

#include <stdbool.h>
#include <stddef.h>
#include "context.h"

typedef struct cml_tensor_s cml_tensor_t;

// Creates a tensor that is owned and managed by the context
cml_tensor_t *cml_tensor_init(cml_context_t *ctx, size_t rows, size_t cols);

// Accessors

size_t cml_tensor_rows(const cml_tensor_t *tensor);
size_t cml_tensor_cols(const cml_tensor_t *tensor);
float cml_tensor_get(const cml_tensor_t *tensor, size_t row, size_t col);
void cml_tensor_set(cml_tensor_t *tensor, size_t row, size_t col, float value);
float *cml_tensor_data(cml_tensor_t *tensor);

// Value initialization

void cml_tensor_zero(cml_tensor_t *tensor);
void cml_tensor_fill(cml_tensor_t *tensor, float value);
void cml_tensor_rand(cml_tensor_t *tensor, float low, float high);
void cml_tensor_copy(cml_context_t *ctx, cml_tensor_t *dst, const cml_tensor_t *src);

// Views

cml_tensor_t *cml_tensor_reshape(cml_context_t *ctx, cml_tensor_t *tensor, size_t new_rows, size_t new_cols);
cml_tensor_t *cml_tensor_view(cml_context_t *ctx, cml_tensor_t *tensor, size_t start_row, size_t start_col,
                              size_t rows, size_t cols);

// Unary operators

cml_tensor_t *cml_tensor_scale(cml_context_t *ctx, cml_tensor_t *tensor, float scalar);
cml_tensor_t *cml_tensor_log(cml_context_t *ctx, cml_tensor_t *tensor);
cml_tensor_t *cml_tensor_sigmoid(cml_context_t *ctx, cml_tensor_t *tensor);
cml_tensor_t *cml_tensor_relu(cml_context_t *ctx, cml_tensor_t *tensor);

// Binary operators

cml_tensor_t *cml_tensor_add(cml_context_t *ctx, cml_tensor_t *a, cml_tensor_t *b);
cml_tensor_t *cml_tensor_sub(cml_context_t *ctx, cml_tensor_t *a, cml_tensor_t *b);
cml_tensor_t *cml_tensor_mul(cml_context_t *ctx, cml_tensor_t *a, cml_tensor_t *b);

// Linear algebra

cml_tensor_t *cml_tensor_dot(cml_context_t *ctx, cml_tensor_t *a, cml_tensor_t *b);
cml_tensor_t *cml_tensor_transpose(cml_context_t *ctx, cml_tensor_t *tensor);
cml_tensor_t *cml_tensor_sum(cml_context_t *ctx, cml_tensor_t *tensor);

// Autograd

void cml_tensor_set_requires_grad(cml_tensor_t *tensor, bool requires_grad);
bool cml_tensor_requires_grad(const cml_tensor_t *tensor);
cml_tensor_t *cml_tensor_grad(const cml_tensor_t *tensor);

#endif
