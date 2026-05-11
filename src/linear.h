#ifndef CML_INTERNAL_LINEAR_H
#define CML_INTERNAL_LINEAR_H

#include <cml/linear.h>

struct cml_linear_s {
    cml_tensor_t *w; // [in_features x out_features]
    cml_tensor_t *b; // [1 x out_features]
};

#endif
