#include <cml/context.h>
#include <cml/data.h>
#include <cml/tensor.h>
#include "unity.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CTX_SIZE (4 * 1024 * 1024)
#define TEST_CSV_PATH "test_data_dataset.csv"
#define TEST_ROWS_PATH "test_data_rows.txt"

static cml_context_t *ctx;

void setUp(void) {
    ctx = cml_init(CTX_SIZE);
}

void tearDown(void) {
    remove(TEST_CSV_PATH);
    remove(TEST_ROWS_PATH);
    cml_deinit(ctx);
}

static void write_csv(const char *contents) {
    FILE *f = fopen(TEST_CSV_PATH, "w");
    TEST_ASSERT_NOT_NULL(f);
    fputs(contents, f);
    fclose(f);
}

static void test_dataset_load_csv_success(void) {
    write_csv(
        "x1,x2,y\n"
        "1.0,2.0,3.0\n"
        "4.0,5.0,6.0\n"
        "7.0,8.0,9.0\n"
    );

    cml_dataset_t *dataset = cml_dataset_load_csv(ctx, TEST_CSV_PATH, 2, 1, true);
    TEST_ASSERT_NOT_NULL(dataset);
    TEST_ASSERT_EQUAL(CML_OK, cml_get_status(ctx));

    TEST_ASSERT_EQUAL_size_t(3, cml_dataset_num_samples(dataset));
    TEST_ASSERT_EQUAL_size_t(2, cml_dataset_num_features(dataset));
    TEST_ASSERT_EQUAL_size_t(1, cml_dataset_num_targets(dataset));

    cml_tensor_t *x = cml_dataset_features(dataset);
    cml_tensor_t *y = cml_dataset_targets(dataset);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, cml_tensor_get(x, 0, 0));
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 5.0f, cml_tensor_get(x, 1, 1));
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 9.0f, cml_tensor_get(y, 2, 0));
    TEST_ASSERT_EQUAL_STRING("x1", cml_dataset_feature_name(dataset, 0));
    TEST_ASSERT_EQUAL_STRING("x2", cml_dataset_feature_name(dataset, 1));
    TEST_ASSERT_EQUAL_STRING("y", cml_dataset_target_name(dataset, 0));
}

static void test_dataset_load_csv_invalid_row_errors(void) {
    write_csv(
        "x1,x2,y\n"
        "1.0,2.0,3.0\n"
        "4.0,not-a-number,6.0\n"
    );

    cml_dataset_t *dataset = cml_dataset_load_csv(ctx, TEST_CSV_PATH, 2, 1, true);
    TEST_ASSERT_NULL(dataset);
    TEST_ASSERT_EQUAL(CML_INVALID_ARG, cml_get_status(ctx));
}

static void test_data_loader_batches_without_shuffle(void) {
    cml_tensor_t *x = cml_tensor_init(ctx, 5, 2);
    cml_tensor_t *y = cml_tensor_init(ctx, 5, 1);
    for (size_t i = 0; i < 5; i++) {
        cml_tensor_set(x, i, 0, (float)(i * 10));
        cml_tensor_set(x, i, 1, (float)(i * 10 + 1));
        cml_tensor_set(y, i, 0, (float)(100 + i));
    }

    cml_dataset_t *dataset = cml_dataset_from_tensors(ctx, x, y);
    cml_data_loader_t *loader = cml_data_loader_init(ctx, dataset, 2, false);
    TEST_ASSERT_NOT_NULL(dataset);
    TEST_ASSERT_NOT_NULL(loader);
    TEST_ASSERT_EQUAL_size_t(3, cml_data_loader_num_batches(loader));

    cml_tensor_t *xb = NULL;
    cml_tensor_t *yb = NULL;

    TEST_ASSERT_TRUE(cml_data_loader_next(ctx, loader, &xb, &yb));
    TEST_ASSERT_EQUAL_size_t(2, cml_tensor_rows(xb));
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, cml_tensor_get(xb, 0, 0));
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 11.0f, cml_tensor_get(xb, 1, 1));
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 100.0f, cml_tensor_get(yb, 0, 0));

    TEST_ASSERT_TRUE(cml_data_loader_next(ctx, loader, &xb, &yb));
    TEST_ASSERT_EQUAL_size_t(2, cml_tensor_rows(xb));
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 20.0f, cml_tensor_get(xb, 0, 0));
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 31.0f, cml_tensor_get(xb, 1, 1));

    TEST_ASSERT_TRUE(cml_data_loader_next(ctx, loader, &xb, &yb));
    TEST_ASSERT_EQUAL_size_t(1, cml_tensor_rows(xb));
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 40.0f, cml_tensor_get(xb, 0, 0));
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 104.0f, cml_tensor_get(yb, 0, 0));

    TEST_ASSERT_FALSE(cml_data_loader_next(ctx, loader, &xb, &yb));
}

static void test_data_loader_shuffle_visits_all_samples(void) {
    cml_tensor_t *x = cml_tensor_init(ctx, 6, 1);
    cml_tensor_t *y = cml_tensor_init(ctx, 6, 1);
    for (size_t i = 0; i < 6; i++) {
        cml_tensor_set(x, i, 0, (float)i);
        cml_tensor_set(y, i, 0, (float)(10 + i));
    }

    cml_dataset_t *dataset = cml_dataset_from_tensors(ctx, x, y);
    srand(9);
    cml_data_loader_t *loader = cml_data_loader_init(ctx, dataset, 1, true);
    TEST_ASSERT_NOT_NULL(loader);

    bool seen[6] = {false, false, false, false, false, false};
    size_t seen_count = 0;

    cml_tensor_t *xb = NULL;
    cml_tensor_t *yb = NULL;
    while (cml_data_loader_next(ctx, loader, &xb, &yb)) {
        size_t value = (size_t)cml_tensor_get(xb, 0, 0);
        TEST_ASSERT_LESS_THAN_size_t(6, value);
        if (!seen[value]) {
            seen[value] = true;
            seen_count++;
        }
    }

    TEST_ASSERT_EQUAL_size_t(6, seen_count);
    for (size_t i = 0; i < 6; i++) {
        TEST_ASSERT_TRUE(seen[i]);
    }
}

static void test_dataset_from_tensors_default_names(void) {
    cml_tensor_t *x = cml_tensor_init(ctx, 2, 2);
    cml_tensor_t *y = cml_tensor_init(ctx, 2, 1);
    cml_dataset_t *dataset = cml_dataset_from_tensors(ctx, x, y);
    TEST_ASSERT_NOT_NULL(dataset);
    TEST_ASSERT_EQUAL_STRING("feature_0", cml_dataset_feature_name(dataset, 0));
    TEST_ASSERT_EQUAL_STRING("feature_1", cml_dataset_feature_name(dataset, 1));
    TEST_ASSERT_EQUAL_STRING("target_0", cml_dataset_target_name(dataset, 0));
}

static void test_dataset_print_rows_with_labels(void) {
    write_csv(
        "age,score,label\n"
        "21,10,0\n"
        "35,15,1\n"
        "50,22,1\n"
    );

    cml_dataset_t *dataset = cml_dataset_load_csv(ctx, TEST_CSV_PATH, 2, 1, true);
    TEST_ASSERT_NOT_NULL(dataset);

    FILE *f = fopen(TEST_ROWS_PATH, "w+");
    TEST_ASSERT_NOT_NULL(f);
    cml_dataset_print_rows(ctx, dataset, 1, 2, f);
    TEST_ASSERT_EQUAL(CML_OK, cml_get_status(ctx));
    fflush(f);
    rewind(f);

    char output[512];
    size_t bytes = fread(output, 1, sizeof(output) - 1, f);
    output[bytes] = '\0';
    fclose(f);

    TEST_ASSERT_NOT_NULL(strstr(output, "row,age,score,label\n"));
    TEST_ASSERT_NOT_NULL(strstr(output, "1,35.000000,15.000000,1.000000\n"));
    TEST_ASSERT_NOT_NULL(strstr(output, "2,50.000000,22.000000,1.000000\n"));
}

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_dataset_load_csv_success);
    RUN_TEST(test_dataset_load_csv_invalid_row_errors);
    RUN_TEST(test_data_loader_batches_without_shuffle);
    RUN_TEST(test_data_loader_shuffle_visits_all_samples);
    RUN_TEST(test_dataset_from_tensors_default_names);
    RUN_TEST(test_dataset_print_rows_with_labels);

    return UNITY_END();
}
