#ifndef CML_INTERNAL_TRAINER_H
#define CML_INTERNAL_TRAINER_H

#include <cml/optimizer.h>
#include <cml/trainer.h>

struct cml_trainer_s {
    void *model;
    cml_loss_fn loss_fn;
    cml_tensor_t **params;
    size_t n_params;
    cml_optimizer_t *opt;
    float last_loss;
    size_t last_epoch;
    bool has_loss;
};

#endif
