#ifndef CML_INTERNAL_TENSOR_H
#define CML_INTERNAL_TENSOR_H

#include <cml/tensor.h>

#include <stdbool.h>

struct cml_tape_node_s;

struct cml_tensor_s {
    cml_context_t *ctx;
    size_t rows;
    size_t cols;
    size_t stride; // Elements between row starts; equals cols for contiguous tensors.
    float *data;
    float *device_data; // Only valid on storage tensors.
    struct cml_tensor_s *storage; // Root storage tensor that owns backing memory.
    size_t data_offset; // Element offset from the root storage base.
    bool host_valid; // Only meaningful on storage tensors.
    bool device_valid; // Only meaningful on storage tensors.
    struct cml_tensor_s *grad;
    bool requires_grad;
    struct cml_tape_node_s *creator;
};

#endif
