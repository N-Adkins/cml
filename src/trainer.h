#ifndef CML_INTERNAL_TRAINER_H
#define CML_INTERNAL_TRAINER_H

#include <cml/trainer.h>
#include "sgd.h"

struct cml_trainer_s {
    void *model;
    cml_loss_fn loss_fn;
    cml_tensor_t **params;
    size_t n_params;
    cml_sgd_t *opt;
};

#endif
