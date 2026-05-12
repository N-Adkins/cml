#ifndef CML_DATA_H
#define CML_DATA_H

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include "context.h"
#include "tensor.h"

typedef struct cml_dataset_s cml_dataset_t;
typedef struct cml_data_loader_s cml_data_loader_t;

// Creates a dataset from feature and target tensors.
// Both tensors must have the same row count and at least one column.
cml_dataset_t *cml_dataset_from_tensors(cml_context_t *ctx,
                                        cml_tensor_t *features,
                                        cml_tensor_t *targets);

// Loads a numeric CSV dataset where each row is:
//   [feature_0, ..., feature_(feature_cols-1), target_0, ..., target_(target_cols-1)]
// If has_header is true, the first non-empty line is parsed as column names.
cml_dataset_t *cml_dataset_load_csv(cml_context_t *ctx,
                                    const char *path,
                                    size_t feature_cols,
                                    size_t target_cols,
                                    bool has_header);

size_t cml_dataset_num_samples(const cml_dataset_t *dataset);
size_t cml_dataset_num_features(const cml_dataset_t *dataset);
size_t cml_dataset_num_targets(const cml_dataset_t *dataset);
cml_tensor_t *cml_dataset_features(const cml_dataset_t *dataset);
cml_tensor_t *cml_dataset_targets(const cml_dataset_t *dataset);
const char *cml_dataset_feature_name(const cml_dataset_t *dataset, size_t feature_index);
const char *cml_dataset_target_name(const cml_dataset_t *dataset, size_t target_index);
void cml_dataset_print_rows(cml_context_t *ctx,
                            const cml_dataset_t *dataset,
                            size_t start_row,
                            size_t max_rows,
                            FILE *stream);

// Mini-batch data loader.
cml_data_loader_t *cml_data_loader_init(cml_context_t *ctx,
                                        cml_dataset_t *dataset,
                                        size_t batch_size,
                                        bool shuffle);
void cml_data_loader_reset(cml_context_t *ctx, cml_data_loader_t *loader);
bool cml_data_loader_next(cml_context_t *ctx,
                          cml_data_loader_t *loader,
                          cml_tensor_t **features_batch,
                          cml_tensor_t **targets_batch);
size_t cml_data_loader_batch_size(const cml_data_loader_t *loader);
size_t cml_data_loader_num_batches(const cml_data_loader_t *loader);
bool cml_data_loader_shuffle(const cml_data_loader_t *loader);

#endif
