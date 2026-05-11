#ifndef CML_INTERNAL_TAPE_H
#define CML_INTERNAL_TAPE_H

#include <cml/tape.h>

void cml_tape_record_add(cml_context_t *ctx, cml_tensor_t *out, cml_tensor_t *a, cml_tensor_t *b);
void cml_tape_record_sub(cml_context_t *ctx, cml_tensor_t *out, cml_tensor_t *a, cml_tensor_t *b);
void cml_tape_record_mul(cml_context_t *ctx, cml_tensor_t *out, cml_tensor_t *a, cml_tensor_t *b);
void cml_tape_record_dot(cml_context_t *ctx, cml_tensor_t *out, cml_tensor_t *a, cml_tensor_t *b);
void cml_tape_record_scale(cml_context_t *ctx, cml_tensor_t *out, cml_tensor_t *tensor, float scalar);
void cml_tape_record_log(cml_context_t *ctx, cml_tensor_t *out, cml_tensor_t *tensor);
void cml_tape_record_sigmoid(cml_context_t *ctx, cml_tensor_t *out, cml_tensor_t *tensor);
void cml_tape_record_relu(cml_context_t *ctx, cml_tensor_t *out, cml_tensor_t *tensor);
void cml_tape_record_transpose(cml_context_t *ctx, cml_tensor_t *out, cml_tensor_t *tensor);
void cml_tape_record_sum(cml_context_t *ctx, cml_tensor_t *out, cml_tensor_t *tensor);

#endif
