#ifndef CML_INTERNAL_TENSOR_H
#define CML_INTERNAL_TENSOR_H

#include <cml/tensor.h>

struct cml_tensor_s {
    size_t rows;
    size_t cols;
    size_t stride; /* elements between row starts; equals cols for contiguous tensors */
    float *data;
};

#endif
