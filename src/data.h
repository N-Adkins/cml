#ifndef CML_INTERNAL_DATA_H
#define CML_INTERNAL_DATA_H

#include <cml/data.h>

struct cml_dataset_s {
    cml_tensor_t *features;
    cml_tensor_t *targets;
    const char **feature_names;
    const char **target_names;
    size_t num_samples;
    size_t num_features;
    size_t num_targets;
};

struct cml_data_loader_s {
    cml_dataset_t *dataset;
    size_t batch_size;
    bool shuffle;
    size_t position;
    size_t *indices;
    cml_tensor_t *shuffle_features_batch;
    cml_tensor_t *shuffle_targets_batch;
};

void cml_data_loader_prepare_device(cml_context_t *ctx, cml_data_loader_t *loader);

#endif
