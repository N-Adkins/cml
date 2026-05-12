#include "data.h"
#include "context.h"
#include "random.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char *start;
    size_t len;
} cml_csv_field_t;

static char *cml_dataset_alloc_string(cml_context_t *ctx, const char *src, size_t len) {
    if (len == SIZE_MAX) {
        cml_context_error(ctx, CML_INVALID_ARG, "string length overflow");
        return NULL;
    }
    char *dst = cml_arena_alloc(&ctx->arena, len + 1);
    if (dst == NULL) {
        cml_context_error(ctx, CML_OUT_OF_MEMORY, "string allocation failed");
        return NULL;
    }
    memcpy(dst, src, len);
    dst[len] = '\0';
    return dst;
}

static bool cml_dataset_alloc_indexed_names(cml_context_t *ctx, const char *prefix,
                                            size_t count, const char ***names_out) {
    if (count == 0) {
        *names_out = NULL;
        return true;
    }
    if (count > SIZE_MAX / sizeof(const char *)) {
        cml_context_error(ctx, CML_INVALID_ARG, "column count overflow");
        return false;
    }

    const char **names = cml_arena_alloc(&ctx->arena, count * sizeof(const char *));
    if (names == NULL) {
        cml_context_error(ctx, CML_OUT_OF_MEMORY, "column name array allocation failed");
        return false;
    }

    for (size_t i = 0; i < count; i++) {
        int needed = snprintf(NULL, 0, "%s_%zu", prefix, i);
        if (needed < 0) {
            cml_context_error(ctx, CML_INVALID_ARG, "failed to format column name");
            return false;
        }
        size_t len = (size_t)needed;
        if (len == SIZE_MAX) {
            cml_context_error(ctx, CML_INVALID_ARG, "column name length overflow");
            return false;
        }

        char *name = cml_arena_alloc(&ctx->arena, len + 1);
        if (name == NULL) {
            cml_context_error(ctx, CML_OUT_OF_MEMORY, "column name allocation failed");
            return false;
        }
        int written = snprintf(name, len + 1, "%s_%zu", prefix, i);
        if (written != needed) {
            cml_context_error(ctx, CML_INVALID_ARG, "failed to write column name");
            return false;
        }
        names[i] = name;
    }

    *names_out = names;
    return true;
}

static bool cml_dataset_set_default_names(cml_context_t *ctx, cml_dataset_t *dataset) {
    if (!cml_dataset_alloc_indexed_names(ctx, "feature", dataset->num_features, &dataset->feature_names)) {
        return false;
    }
    if (!cml_dataset_alloc_indexed_names(ctx, "target", dataset->num_targets, &dataset->target_names)) {
        return false;
    }
    return true;
}

// Reads one logical line into *buf, growing the buffer as needed. Strips a trailing CR/LF.
// Returns 1 if a line was read, 0 at clean EOF, -1 on allocation failure.
static int cml_csv_read_line(FILE *file, char **buf, size_t *cap, size_t *out_len) {
    size_t pos = 0;
    int ch;
    while ((ch = fgetc(file)) != EOF) {
        if (pos + 1 >= *cap) {
            size_t new_cap = (*cap == 0) ? 256 : *cap * 2;
            if (new_cap < pos + 2) new_cap = pos + 2;
            char *nb = realloc(*buf, new_cap);
            if (nb == NULL) return -1;
            *buf = nb;
            *cap = new_cap;
        }
        if (ch == '\n') break;
        (*buf)[pos++] = (char)ch;
    }
    if (ch == EOF && pos == 0) {
        if (*cap == 0) {
            char *nb = realloc(*buf, 1);
            if (nb == NULL) return -1;
            *buf = nb;
            *cap = 1;
        }
        (*buf)[0] = '\0';
        *out_len = 0;
        return 0;
    }
    if (pos > 0 && (*buf)[pos - 1] == '\r') pos--;
    (*buf)[pos] = '\0';
    *out_len = pos;
    return 1;
}

static bool cml_csv_line_is_blank(const char *line) {
    while (*line != '\0') {
        if (*line != ' ' && *line != '\t') return false;
        line++;
    }
    return true;
}

// Locale-independent decimal float parser. Accepts optional sign, integer
// and fraction parts, and optional decimal exponent. Returns true on success
// and sets *endp to the character past the parsed number
static bool cml_parse_float_c(const char *s, const char **endp, float *out) {
    const char *start = s;
    while (*s == ' ' || *s == '\t') s++;

    int sign = 1;
    if (*s == '+') s++;
    else if (*s == '-') { sign = -1; s++; }

    double mantissa = 0.0;
    int frac_digits = 0;
    bool past_point = false;
    bool has_digit = false;

    while (*s != '\0') {
        if (*s >= '0' && *s <= '9') {
            mantissa = mantissa * 10.0 + (double)(*s - '0');
            if (past_point) frac_digits++;
            has_digit = true;
            s++;
        } else if (*s == '.' && !past_point) {
            past_point = true;
            s++;
        } else {
            break;
        }
    }

    if (!has_digit) {
        *endp = start;
        return false;
    }

    int exp10 = -frac_digits;
    if (*s == 'e' || *s == 'E') {
        const char *exp_pos = s;
        s++;
        int esign = 1;
        if (*s == '+') s++;
        else if (*s == '-') { esign = -1; s++; }
        // Cap the accumulator well past any meaningful float magnitude; beyond this
        // the result is zero or inf anyway, and unbounded accumulation would overflow int
        const int EXP_CAP = 100000;
        int eval = 0;
        bool has_exp_digit = false;
        while (*s >= '0' && *s <= '9') {
            if (eval < EXP_CAP) eval = eval * 10 + (*s - '0');
            has_exp_digit = true;
            s++;
        }
        if (!has_exp_digit) {
            s = exp_pos;
        } else {
            exp10 += esign * eval;
        }
    }

    double value = (double)sign * mantissa * pow(10.0, (double)exp10);
    *out = (float)value;
    *endp = s;
    return true;
}

// Extracts one CSV field from *p (in-place rewriting for quoted fields to collapse "" -> ").
// On success: writes the field to *out (null-terminated), advances *p past the comma
// terminating the field, and sets *had_comma. Returns true if a field was parsed.
static bool cml_csv_next_field(char **p, cml_csv_field_t *out, bool *had_comma) {
    char *s = *p;
    while (*s == ' ' || *s == '\t') s++;

    if (*s == '"') {
        s++;
        char *field_start = s;
        char *w = s;
        while (*s != '\0') {
            if (*s == '"') {
                if (s[1] == '"') {
                    *w++ = '"';
                    s += 2;
                } else {
                    s++;
                    break;
                }
            } else {
                *w++ = *s++;
            }
        }
        out->start = field_start;
        out->len = (size_t)(w - field_start);
        // Write a null terminator into the (now-unused) gap before the next comma
        if (w < s) *w = '\0';

        while (*s == ' ' || *s == '\t') s++;
        *had_comma = (*s == ',');
        if (*had_comma) {
            s++;
        } else if (*s != '\0') {
            return false;
        }
        *p = s;
        return true;
    }

    char *field_start = s;
    while (*s != '\0' && *s != ',') s++;
    char *field_end = s;
    while (field_end > field_start && (field_end[-1] == ' ' || field_end[-1] == '\t')) {
        field_end--;
    }
    *had_comma = (*s == ',');
    if (*had_comma) s++;
    out->start = field_start;
    out->len = (size_t)(field_end - field_start);
    if (field_end < s) *field_end = '\0';
    *p = s;
    return true;
}

// Splits a CSV line into exactly `expected` fields. Returns true on exact match.
static bool cml_csv_split_line(char *line, size_t expected, cml_csv_field_t *fields) {
    char *p = line;
    for (size_t i = 0; i < expected; i++) {
        bool had_comma = false;
        if (!cml_csv_next_field(&p, &fields[i], &had_comma)) return false;
        bool last = (i + 1 == expected);
        if (last) {
            if (had_comma) return false; // too many fields
            while (*p == ' ' || *p == '\t') p++;
            if (*p != '\0') return false; // garbage after last field
        } else {
            if (!had_comma) return false; // not enough fields
        }
    }
    return true;
}

static bool cml_csv_parse_floats(cml_csv_field_t *fields, size_t n, float *out) {
    for (size_t i = 0; i < n; i++) {
        if (fields[i].len == 0) return false;
        // cml_csv_next_field already null-terminates fields[i].start[fields[i].len]
        const char *endp = NULL;
        if (!cml_parse_float_c(fields[i].start, &endp, &out[i])) return false;
        while (*endp == ' ' || *endp == '\t') endp++;
        if (*endp != '\0') return false;
    }
    return true;
}

cml_dataset_t *cml_dataset_from_tensors(cml_context_t *ctx,
                                        cml_tensor_t *features,
                                        cml_tensor_t *targets) {
    if (ctx == NULL) return NULL;
    if (ctx->status != CML_OK) return NULL;
    if (features == NULL || targets == NULL) {
        cml_context_error(ctx, CML_INVALID_ARG, "dataset tensor argument is NULL");
        return NULL;
    }

    size_t feature_rows = cml_tensor_rows(features);
    size_t target_rows = cml_tensor_rows(targets);
    size_t feature_cols = cml_tensor_cols(features);
    size_t target_cols = cml_tensor_cols(targets);

    if (feature_rows == 0 || target_rows == 0 || feature_cols == 0 || target_cols == 0) {
        cml_context_error(ctx, CML_INVALID_ARG, "dataset tensors must be non-empty");
        return NULL;
    }
    if (feature_rows != target_rows) {
        cml_context_error(ctx, CML_INVALID_ARG, "dataset feature/target row mismatch");
        return NULL;
    }

    cml_dataset_t *dataset = cml_arena_alloc(&ctx->arena, sizeof(struct cml_dataset_s));
    if (dataset == NULL) {
        cml_context_error(ctx, CML_OUT_OF_MEMORY, "dataset allocation failed");
        return NULL;
    }

    dataset->features = features;
    dataset->targets = targets;
    dataset->num_samples = feature_rows;
    dataset->num_features = feature_cols;
    dataset->num_targets = target_cols;
    dataset->feature_names = NULL;
    dataset->target_names = NULL;
    if (!cml_dataset_set_default_names(ctx, dataset)) {
        return NULL;
    }
    return dataset;
}

cml_dataset_t *cml_dataset_load_csv(cml_context_t *ctx, const char *path,
                                    size_t feature_cols, size_t target_cols,
                                    bool has_header) {
    if (ctx == NULL) return NULL;
    if (ctx->status != CML_OK) return NULL;
    if (path == NULL) {
        cml_context_error(ctx, CML_INVALID_ARG, "CSV path is NULL");
        return NULL;
    }
    if (feature_cols == 0 || target_cols == 0) {
        cml_context_error(ctx, CML_INVALID_ARG, "CSV feature and target columns must be > 0");
        return NULL;
    }
    if (feature_cols > SIZE_MAX - target_cols) {
        cml_context_error(ctx, CML_INVALID_ARG, "CSV column count overflow");
        return NULL;
    }
    size_t expected_cols = feature_cols + target_cols;
    // Guard the largest per-column allocation we'll do (the field metadata array)
    size_t max_elem = sizeof(cml_csv_field_t);
    if (sizeof(char *) > max_elem) max_elem = sizeof(char *);
    if (sizeof(float) > max_elem) max_elem = sizeof(float);
    if (expected_cols > SIZE_MAX / max_elem) {
        cml_context_error(ctx, CML_INVALID_ARG, "CSV column count overflow");
        return NULL;
    }

    FILE *file = fopen(path, "r");
    if (file == NULL) {
        cml_context_error(ctx, CML_INVALID_ARG, "failed to open CSV file");
        return NULL;
    }

    char *line = NULL;
    size_t line_cap = 0;
    cml_csv_field_t *fields = malloc(expected_cols * sizeof(cml_csv_field_t));
    float *row_values = malloc(expected_cols * sizeof(float));
    char **header_names = has_header ? calloc(expected_cols, sizeof(char *)) : NULL;
    float *flat = NULL; // growing buffer for parsed rows
    size_t flat_cap = 0;
    size_t flat_used = 0;
    size_t row_count = 0;
    bool header_consumed = !has_header;

    bool ok = (fields != NULL && row_values != NULL && (!has_header || header_names != NULL));
    if (!ok) {
        cml_context_error(ctx, CML_OUT_OF_MEMORY, "failed to allocate CSV parse buffers");
        goto cleanup;
    }

    for (;;) {
        size_t line_len = 0;
        int rc = cml_csv_read_line(file, &line, &line_cap, &line_len);
        if (rc < 0) {
            cml_context_error(ctx, CML_OUT_OF_MEMORY, "failed to grow CSV line buffer");
            ok = false;
            goto cleanup;
        }
        if (rc == 0) break;
        if (cml_csv_line_is_blank(line)) continue;

        if (!header_consumed) {
            if (!cml_csv_split_line(line, expected_cols, fields)) {
                cml_context_error(ctx, CML_INVALID_ARG, "CSV header has invalid format");
                ok = false;
                goto cleanup;
            }
            // Copy into the arena up front; fields[] gets reused for subsequent rows
            for (size_t i = 0; i < expected_cols; i++) {
                if (fields[i].len == 0) {
                    cml_context_error(ctx, CML_INVALID_ARG, "CSV header column name is empty");
                    ok = false;
                    goto cleanup;
                }
                header_names[i] = cml_dataset_alloc_string(ctx, fields[i].start, fields[i].len);
                if (header_names[i] == NULL) { ok = false; goto cleanup; }
            }
            header_consumed = true;
            continue;
        }

        if (!cml_csv_split_line(line, expected_cols, fields)) {
            cml_context_error(ctx, CML_INVALID_ARG, "CSV row has invalid format");
            ok = false;
            goto cleanup;
        }
        if (!cml_csv_parse_floats(fields, expected_cols, row_values)) {
            cml_context_error(ctx, CML_INVALID_ARG, "CSV row has non-numeric field");
            ok = false;
            goto cleanup;
        }

        if (flat_used + expected_cols > flat_cap) {
            size_t needed = flat_used + expected_cols;
            if (flat_used > SIZE_MAX - expected_cols) {
                cml_context_error(ctx, CML_INVALID_ARG, "CSV row buffer overflow");
                ok = false;
                goto cleanup;
            }
            size_t new_cap = flat_cap == 0 ? expected_cols * 16 : flat_cap;
            while (new_cap < needed) {
                if (new_cap > SIZE_MAX / 2) { new_cap = needed; break; }
                new_cap *= 2;
            }
            if (new_cap > SIZE_MAX / sizeof(float)) {
                cml_context_error(ctx, CML_INVALID_ARG, "CSV row buffer overflow");
                ok = false;
                goto cleanup;
            }
            float *nb = realloc(flat, new_cap * sizeof(float));
            if (nb == NULL) {
                cml_context_error(ctx, CML_OUT_OF_MEMORY, "failed to grow CSV row buffer");
                ok = false;
                goto cleanup;
            }
            flat = nb;
            flat_cap = new_cap;
        }
        memcpy(flat + flat_used, row_values, expected_cols * sizeof(float));
        flat_used += expected_cols;
        row_count++;
    }

    if (!ok) goto cleanup;
    if (row_count == 0) {
        cml_context_error(ctx, CML_INVALID_ARG, "CSV contains no data rows");
        ok = false;
        goto cleanup;
    }

    cml_tensor_t *features_tensor = cml_tensor_init(ctx, row_count, feature_cols);
    cml_tensor_t *targets_tensor = cml_tensor_init(ctx, row_count, target_cols);
    if (features_tensor == NULL || targets_tensor == NULL) { ok = false; goto cleanup; }

    float *fd = cml_tensor_data(features_tensor);
    float *td = cml_tensor_data(targets_tensor);
    if (fd == NULL || td == NULL) { ok = false; goto cleanup; }

    for (size_t r = 0; r < row_count; r++) {
        const float *src = flat + r * expected_cols;
        memcpy(fd + r * feature_cols, src, feature_cols * sizeof(float));
        memcpy(td + r * target_cols, src + feature_cols, target_cols * sizeof(float));
    }

    cml_dataset_t *dataset = cml_dataset_from_tensors(ctx, features_tensor, targets_tensor);
    if (dataset == NULL) { ok = false; goto cleanup; }

    if (has_header) {
        for (size_t i = 0; i < feature_cols; i++) {
            dataset->feature_names[i] = header_names[i];
        }
        for (size_t i = 0; i < target_cols; i++) {
            dataset->target_names[i] = header_names[feature_cols + i];
        }
    }

    free(fields);
    free(row_values);
    free(header_names);
    free(flat);
    free(line);
    fclose(file);
    return dataset;

cleanup:
    free(fields);
    free(row_values);
    free(header_names);
    free(flat);
    free(line);
    fclose(file);
    return NULL;
}

size_t cml_dataset_num_samples(const cml_dataset_t *dataset) {
    if (dataset == NULL) return 0;
    return dataset->num_samples;
}

size_t cml_dataset_num_features(const cml_dataset_t *dataset) {
    if (dataset == NULL) return 0;
    return dataset->num_features;
}

size_t cml_dataset_num_targets(const cml_dataset_t *dataset) {
    if (dataset == NULL) return 0;
    return dataset->num_targets;
}

cml_tensor_t *cml_dataset_features(const cml_dataset_t *dataset) {
    if (dataset == NULL) return NULL;
    return dataset->features;
}

cml_tensor_t *cml_dataset_targets(const cml_dataset_t *dataset) {
    if (dataset == NULL) return NULL;
    return dataset->targets;
}

const char *cml_dataset_feature_name(const cml_dataset_t *dataset, size_t feature_index) {
    if (dataset == NULL) return NULL;
    if (feature_index >= dataset->num_features) return NULL;
    return dataset->feature_names[feature_index];
}

const char *cml_dataset_target_name(const cml_dataset_t *dataset, size_t target_index) {
    if (dataset == NULL) return NULL;
    if (target_index >= dataset->num_targets) return NULL;
    return dataset->target_names[target_index];
}

void cml_dataset_print_rows(cml_context_t *ctx,
                            const cml_dataset_t *dataset,
                            size_t start_row,
                            size_t max_rows,
                            FILE *stream) {
    if (ctx == NULL) return;
    if (ctx->status != CML_OK) return;
    if (dataset == NULL) {
        cml_context_error(ctx, CML_INVALID_ARG, "dataset is NULL");
        return;
    }
    if (start_row >= dataset->num_samples) {
        cml_context_error(ctx, CML_INVALID_ARG, "start_row is out of range");
        return;
    }

    FILE *out = (stream != NULL) ? stream : stdout;
    fprintf(out, "row");
    for (size_t col = 0; col < dataset->num_features; col++) {
        fprintf(out, ",%s", dataset->feature_names[col]);
    }
    for (size_t col = 0; col < dataset->num_targets; col++) {
        fprintf(out, ",%s", dataset->target_names[col]);
    }
    fputc('\n', out);

    size_t remaining = dataset->num_samples - start_row;
    size_t rows_to_print = (max_rows < remaining) ? max_rows : remaining;
    for (size_t row_offset = 0; row_offset < rows_to_print; row_offset++) {
        size_t row = start_row + row_offset;
        fprintf(out, "%zu", row);
        for (size_t col = 0; col < dataset->num_features; col++) {
            fprintf(out, ",%.6f", (double)cml_tensor_get(dataset->features, row, col));
        }
        for (size_t col = 0; col < dataset->num_targets; col++) {
            fprintf(out, ",%.6f", (double)cml_tensor_get(dataset->targets, row, col));
        }
        fputc('\n', out);
    }
}

cml_data_loader_t *cml_data_loader_init(cml_context_t *ctx,
                                        cml_dataset_t *dataset,
                                        size_t batch_size,
                                        bool shuffle) {
    if (ctx == NULL) return NULL;
    if (ctx->status != CML_OK) return NULL;
    if (dataset == NULL) {
        cml_context_error(ctx, CML_INVALID_ARG, "dataset is NULL");
        return NULL;
    }
    if (batch_size == 0) {
        cml_context_error(ctx, CML_INVALID_ARG, "batch_size must be > 0");
        return NULL;
    }

    cml_data_loader_t *loader = cml_arena_alloc(&ctx->arena, sizeof(struct cml_data_loader_s));
    if (loader == NULL) {
        cml_context_error(ctx, CML_OUT_OF_MEMORY, "data loader allocation failed");
        return NULL;
    }

    loader->dataset = dataset;
    loader->batch_size = batch_size;
    loader->shuffle = shuffle;
    loader->position = 0;
    loader->indices = NULL;
    loader->shuffle_features_batch = NULL;
    loader->shuffle_targets_batch = NULL;

    if (shuffle) {
        if (dataset->num_samples > SIZE_MAX / sizeof(size_t)) {
            cml_context_error(ctx, CML_INVALID_ARG, "dataset is too large");
            return NULL;
        }
        loader->indices = cml_arena_alloc(&ctx->arena, dataset->num_samples * sizeof(size_t));
        if (loader->indices == NULL) {
            cml_context_error(ctx, CML_OUT_OF_MEMORY, "failed to allocate shuffle indices");
            return NULL;
        }

        loader->shuffle_features_batch = cml_tensor_init(ctx, batch_size, dataset->num_features);
        loader->shuffle_targets_batch = cml_tensor_init(ctx, batch_size, dataset->num_targets);
        if (loader->shuffle_features_batch == NULL || loader->shuffle_targets_batch == NULL) {
            return NULL;
        }
    }

    cml_data_loader_reset(ctx, loader);
    return loader;
}

void cml_data_loader_prepare_device(cml_context_t *ctx, cml_data_loader_t *loader) {
    if (ctx == NULL || loader == NULL) return;
    if (ctx->status != CML_OK) return;

    if (!loader->shuffle || loader->indices == NULL) {
        cml_tensor_to_device(ctx, loader->dataset->features);
        cml_tensor_to_device(ctx, loader->dataset->targets);
        return;
    }

    cml_tensor_to_device(ctx, loader->shuffle_features_batch);
    cml_tensor_to_device(ctx, loader->shuffle_targets_batch);
}

void cml_data_loader_reset(cml_context_t *ctx, cml_data_loader_t *loader) {
    if (ctx == NULL || loader == NULL) return;
    if (ctx->status != CML_OK) return;

    loader->position = 0;

    if (!loader->shuffle || loader->indices == NULL) return;

    size_t n = loader->dataset->num_samples;
    for (size_t i = 0; i < n; i++) loader->indices[i] = i;

    // Fisher-Yates using the context-local PRNG
    for (size_t i = n; i > 1; i--) {
        size_t swap_idx = cml_rng_next_below(&ctx->rng, i);
        size_t idx = i - 1;
        size_t tmp = loader->indices[idx];
        loader->indices[idx] = loader->indices[swap_idx];
        loader->indices[swap_idx] = tmp;
    }
}

bool cml_data_loader_next(cml_context_t *ctx,
                          cml_data_loader_t *loader,
                          cml_tensor_t **features_batch,
                          cml_tensor_t **targets_batch) {
    if (ctx == NULL) return false;
    if (ctx->status != CML_OK) return false;
    if (loader == NULL || features_batch == NULL || targets_batch == NULL) {
        cml_context_error(ctx, CML_INVALID_ARG, "data loader argument is NULL");
        return false;
    }

    *features_batch = NULL;
    *targets_batch = NULL;

    if (loader->position >= loader->dataset->num_samples) {
        return false;
    }

    size_t remaining = loader->dataset->num_samples - loader->position;
    size_t batch_rows = (remaining < loader->batch_size) ? remaining : loader->batch_size;

    size_t feature_cols = loader->dataset->num_features;
    size_t target_cols = loader->dataset->num_targets;

    if (!loader->shuffle || loader->indices == NULL) {
        size_t source_row = loader->position;
        cml_tensor_t *xb = cml_tensor_view(ctx, loader->dataset->features, source_row, 0,
                                           batch_rows, feature_cols);
        cml_tensor_t *yb = cml_tensor_view(ctx, loader->dataset->targets, source_row, 0,
                                           batch_rows, target_cols);
        if (xb == NULL || yb == NULL) {
            return false;
        }
        *features_batch = xb;
        *targets_batch = yb;
    } else {
        const float *features_data = cml_tensor_const_data(loader->dataset->features);
        const float *targets_data = cml_tensor_const_data(loader->dataset->targets);
        float *xb_data = cml_tensor_data(loader->shuffle_features_batch);
        float *yb_data = cml_tensor_data(loader->shuffle_targets_batch);
        if (features_data == NULL || targets_data == NULL || xb_data == NULL || yb_data == NULL) {
            return false;
        }

        for (size_t row = 0; row < batch_rows; row++) {
            size_t source_row = loader->indices[loader->position + row];
            memcpy(xb_data + row * feature_cols,
                   features_data + source_row * feature_cols,
                   feature_cols * sizeof(float));
            memcpy(yb_data + row * target_cols,
                   targets_data + source_row * target_cols,
                   target_cols * sizeof(float));
        }

        if (batch_rows == loader->batch_size) {
            *features_batch = loader->shuffle_features_batch;
            *targets_batch = loader->shuffle_targets_batch;
        } else {
            cml_tensor_t *xb = cml_tensor_view(ctx, loader->shuffle_features_batch, 0, 0,
                                               batch_rows, feature_cols);
            cml_tensor_t *yb = cml_tensor_view(ctx, loader->shuffle_targets_batch, 0, 0,
                                               batch_rows, target_cols);
            if (xb == NULL || yb == NULL) {
                return false;
            }
            *features_batch = xb;
            *targets_batch = yb;
        }
    }

    loader->position += batch_rows;
    return true;
}

size_t cml_data_loader_batch_size(const cml_data_loader_t *loader) {
    if (loader == NULL) return 0;
    return loader->batch_size;
}

size_t cml_data_loader_num_batches(const cml_data_loader_t *loader) {
    if (loader == NULL || loader->batch_size == 0) return 0;
    size_t batches = loader->dataset->num_samples / loader->batch_size;
    if (loader->dataset->num_samples % loader->batch_size != 0) {
        batches++;
    }
    return batches;
}

bool cml_data_loader_shuffle(const cml_data_loader_t *loader) {
    if (loader == NULL) return false;
    return loader->shuffle;
}
