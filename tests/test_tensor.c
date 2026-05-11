#include <cml/context.h>
#include <cml/tensor.h>
#include "tensor.h"
#include "unity.h"

#define CTX_SIZE (1024 * 1024)

static cml_context_t *ctx;

void setUp(void) {
    ctx = cml_init(CTX_SIZE);
}

void tearDown(void) {
    cml_deinit(ctx);
}

/* --- init / accessors --- */

static void test_init_sets_shape(void) {
    cml_tensor_t *t = cml_tensor_init(ctx, 3, 4);
    TEST_ASSERT_NOT_NULL(t);
    TEST_ASSERT_EQUAL_size_t(3, cml_tensor_rows(t));
    TEST_ASSERT_EQUAL_size_t(4, cml_tensor_cols(t));
    TEST_ASSERT_EQUAL(CML_OK, cml_get_status(ctx));
}

static void test_init_stride_equals_cols(void) {
    cml_tensor_t *t = cml_tensor_init(ctx, 5, 7);
    TEST_ASSERT_NOT_NULL(t);
    TEST_ASSERT_EQUAL_size_t(7, t->stride);
}

static void test_init_null_ctx_returns_null(void) {
    TEST_ASSERT_NULL(cml_tensor_init(NULL, 3, 3));
}

static void test_accessor_null_tensor(void) {
    TEST_ASSERT_EQUAL_size_t(0, cml_tensor_rows(NULL));
    TEST_ASSERT_EQUAL_size_t(0, cml_tensor_cols(NULL));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, cml_tensor_get(NULL, 0, 0));
    TEST_ASSERT_NULL(cml_tensor_data(NULL));
}

/* --- set / get --- */

static void test_set_get_roundtrip(void) {
    cml_tensor_t *t = cml_tensor_init(ctx, 3, 3);
    cml_tensor_set(t, 1, 2, 7.5f);
    TEST_ASSERT_EQUAL_FLOAT(7.5f, cml_tensor_get(t, 1, 2));
}

static void test_data_pointer_matches_get(void) {
    cml_tensor_t *t = cml_tensor_init(ctx, 2, 3);
    cml_tensor_set(t, 0, 0, 1.0f);
    cml_tensor_set(t, 1, 2, 9.0f);
    float *d = cml_tensor_data(t);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, d[0]);
    TEST_ASSERT_EQUAL_FLOAT(9.0f, d[1 * 3 + 2]);
}

static void test_const_data_pointer_matches_get(void) {
    cml_tensor_t *t = cml_tensor_init(ctx, 2, 3);
    cml_tensor_set(t, 0, 1, 7.0f);
    cml_tensor_set(t, 1, 0, 4.0f);
    const float *d = cml_tensor_const_data(t);
    TEST_ASSERT_NOT_NULL(d);
    TEST_ASSERT_EQUAL_FLOAT(7.0f, d[1]);
    TEST_ASSERT_EQUAL_FLOAT(4.0f, d[3]);
}

static void test_sync_api_cpu_context(void) {
    cml_tensor_t *t = cml_tensor_init(ctx, 2, 2);
    cml_tensor_fill(t, 1.0f);
    TEST_ASSERT_EQUAL(CML_BACKEND_CPU, cml_get_backend(ctx));
    TEST_ASSERT_EQUAL(CML_OK, cml_tensor_to_device(ctx, t));
    TEST_ASSERT_FALSE(cml_tensor_has_device_copy(t));
    TEST_ASSERT_EQUAL(CML_OK, cml_tensor_to_host(ctx, t));
}

/* --- fill / zero / rand --- */

static void test_zero_clears_all_elements(void) {
    cml_tensor_t *t = cml_tensor_init(ctx, 4, 4);
    cml_tensor_fill(t, 99.0f);
    cml_tensor_zero(t);
    for (size_t r = 0; r < 4; r++) {
        for (size_t c = 0; c < 4; c++) {
            TEST_ASSERT_EQUAL_FLOAT(0.0f, cml_tensor_get(t, r, c));
        }
    }
}

static void test_fill_sets_all_elements(void) {
    cml_tensor_t *t = cml_tensor_init(ctx, 3, 5);
    cml_tensor_fill(t, 3.14f);
    for (size_t r = 0; r < 3; r++) {
        for (size_t c = 0; c < 5; c++) {
            TEST_ASSERT_EQUAL_FLOAT(3.14f, cml_tensor_get(t, r, c));
        }
    }
}

static void test_rand_stays_in_range(void) {
    cml_tensor_t *t = cml_tensor_init(ctx, 8, 8);
    cml_tensor_rand(t, -1.0f, 1.0f);
    for (size_t r = 0; r < 8; r++) {
        for (size_t c = 0; c < 8; c++) {
            float v = cml_tensor_get(t, r, c);
            TEST_ASSERT_GREATER_OR_EQUAL_FLOAT(-1.0f, v);
            TEST_ASSERT_LESS_OR_EQUAL_FLOAT(1.0f, v);
        }
    }
}

/* --- copy --- */

static void test_copy_duplicates_values(void) {
    cml_tensor_t *src = cml_tensor_init(ctx, 2, 3);
    cml_tensor_t *dst = cml_tensor_init(ctx, 2, 3);
    cml_tensor_fill(src, 5.0f);
    cml_tensor_set(src, 0, 1, 2.0f);
    cml_tensor_copy(ctx, dst, src);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, cml_tensor_get(dst, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(2.0f, cml_tensor_get(dst, 0, 1));
    TEST_ASSERT_EQUAL_FLOAT(5.0f, cml_tensor_get(dst, 1, 2));
}

static void test_copy_is_independent(void) {
    cml_tensor_t *src = cml_tensor_init(ctx, 2, 2);
    cml_tensor_t *dst = cml_tensor_init(ctx, 2, 2);
    cml_tensor_fill(src, 1.0f);
    cml_tensor_copy(ctx, dst, src);
    cml_tensor_set(src, 0, 0, 99.0f);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, cml_tensor_get(dst, 0, 0));
}

/* --- reshape --- */

static void test_reshape_changes_shape(void) {
    cml_tensor_t *t = cml_tensor_init(ctx, 2, 6);
    cml_tensor_fill(t, 1.0f);
    cml_tensor_t *r = cml_tensor_reshape(ctx, t, 3, 4);
    TEST_ASSERT_NOT_NULL(r);
    TEST_ASSERT_EQUAL_size_t(3, cml_tensor_rows(r));
    TEST_ASSERT_EQUAL_size_t(4, cml_tensor_cols(r));
    TEST_ASSERT_EQUAL(CML_OK, cml_get_status(ctx));
}

static void test_reshape_shares_data(void) {
    cml_tensor_t *t = cml_tensor_init(ctx, 1, 6);
    for (size_t c = 0; c < 6; c++) cml_tensor_set(t, 0, c, (float)c);
    cml_tensor_t *r = cml_tensor_reshape(ctx, t, 2, 3);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, cml_tensor_get(r, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(3.0f, cml_tensor_get(r, 1, 0));
    TEST_ASSERT_EQUAL_FLOAT(5.0f, cml_tensor_get(r, 1, 2));
}

static void test_reshape_element_count_mismatch_errors(void) {
    cml_tensor_t *t = cml_tensor_init(ctx, 2, 3);
    TEST_ASSERT_NULL(cml_tensor_reshape(ctx, t, 2, 4));
    TEST_ASSERT_EQUAL(CML_INVALID_ARG, cml_get_status(ctx));
}

/* --- view --- */

static void test_view_correct_shape(void) {
    cml_tensor_t *t = cml_tensor_init(ctx, 4, 4);
    cml_tensor_t *v = cml_tensor_view(ctx, t, 1, 1, 2, 2);
    TEST_ASSERT_NOT_NULL(v);
    TEST_ASSERT_EQUAL_size_t(2, cml_tensor_rows(v));
    TEST_ASSERT_EQUAL_size_t(2, cml_tensor_cols(v));
    TEST_ASSERT_EQUAL(CML_OK, cml_get_status(ctx));
}

static void test_view_reads_parent_data(void) {
    cml_tensor_t *t = cml_tensor_init(ctx, 3, 3);
    cml_tensor_zero(t);
    cml_tensor_set(t, 1, 1, 7.0f);
    cml_tensor_set(t, 1, 2, 8.0f);
    cml_tensor_t *v = cml_tensor_view(ctx, t, 1, 1, 1, 2);
    TEST_ASSERT_EQUAL_FLOAT(7.0f, cml_tensor_get(v, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(8.0f, cml_tensor_get(v, 0, 1));
}

static void test_view_write_visible_in_parent(void) {
    cml_tensor_t *t = cml_tensor_init(ctx, 3, 3);
    cml_tensor_zero(t);
    cml_tensor_t *v = cml_tensor_view(ctx, t, 0, 0, 2, 2);
    cml_tensor_set(v, 1, 1, 42.0f);
    TEST_ASSERT_EQUAL_FLOAT(42.0f, cml_tensor_get(t, 1, 1));
}

static void test_view_out_of_bounds_errors(void) {
    cml_tensor_t *t = cml_tensor_init(ctx, 2, 2);
    TEST_ASSERT_NULL(cml_tensor_view(ctx, t, 1, 1, 2, 2));
    TEST_ASSERT_EQUAL(CML_INVALID_ARG, cml_get_status(ctx));
}

static void test_view_inherits_parent_stride(void) {
    cml_tensor_t *t = cml_tensor_init(ctx, 4, 4);
    cml_tensor_t *v = cml_tensor_view(ctx, t, 0, 0, 2, 2);
    TEST_ASSERT_EQUAL_size_t(t->stride, v->stride);
}

/* --- scale --- */

static void test_scale_produces_scaled_copy(void) {
    cml_tensor_t *t = cml_tensor_init(ctx, 2, 2);
    cml_tensor_fill(t, 3.0f);
    cml_tensor_t *s = cml_tensor_scale(ctx, t, 2.0f);
    TEST_ASSERT_NOT_NULL(s);
    for (size_t r = 0; r < 2; r++) {
        for (size_t c = 0; c < 2; c++) {
            TEST_ASSERT_EQUAL_FLOAT(6.0f, cml_tensor_get(s, r, c));
        }
    }
}

static void test_scale_does_not_modify_original(void) {
    cml_tensor_t *t = cml_tensor_init(ctx, 2, 2);
    cml_tensor_fill(t, 4.0f);
    cml_tensor_scale(ctx, t, 0.5f);
    TEST_ASSERT_EQUAL_FLOAT(4.0f, cml_tensor_get(t, 0, 0));
}

/* --- add / sub / mul --- */

static void test_add_elementwise(void) {
    cml_tensor_t *a = cml_tensor_init(ctx, 2, 2);
    cml_tensor_t *b = cml_tensor_init(ctx, 2, 2);
    cml_tensor_fill(a, 1.0f);
    cml_tensor_fill(b, 2.0f);
    cml_tensor_t *c = cml_tensor_add(ctx, a, b);
    TEST_ASSERT_NOT_NULL(c);
    for (size_t r = 0; r < 2; r++) {
        for (size_t c2 = 0; c2 < 2; c2++) {
            TEST_ASSERT_EQUAL_FLOAT(3.0f, cml_tensor_get(c, r, c2));
        }
    }
}

static void test_sub_elementwise(void) {
    cml_tensor_t *a = cml_tensor_init(ctx, 2, 2);
    cml_tensor_t *b = cml_tensor_init(ctx, 2, 2);
    cml_tensor_fill(a, 5.0f);
    cml_tensor_fill(b, 3.0f);
    cml_tensor_t *c = cml_tensor_sub(ctx, a, b);
    TEST_ASSERT_NOT_NULL(c);
    for (size_t r = 0; r < 2; r++) {
        for (size_t c2 = 0; c2 < 2; c2++) {
            TEST_ASSERT_EQUAL_FLOAT(2.0f, cml_tensor_get(c, r, c2));
        }
    }
}

static void test_mul_elementwise(void) {
    cml_tensor_t *a = cml_tensor_init(ctx, 2, 2);
    cml_tensor_t *b = cml_tensor_init(ctx, 2, 2);
    cml_tensor_fill(a, 3.0f);
    cml_tensor_fill(b, 4.0f);
    cml_tensor_t *c = cml_tensor_mul(ctx, a, b);
    TEST_ASSERT_NOT_NULL(c);
    for (size_t r = 0; r < 2; r++) {
        for (size_t c2 = 0; c2 < 2; c2++) {
            TEST_ASSERT_EQUAL_FLOAT(12.0f, cml_tensor_get(c, r, c2));
        }
    }
}

static void test_add_shape_mismatch_errors(void) {
    cml_tensor_t *a = cml_tensor_init(ctx, 2, 3);
    cml_tensor_t *b = cml_tensor_init(ctx, 3, 2);
    TEST_ASSERT_NULL(cml_tensor_add(ctx, a, b));
    TEST_ASSERT_EQUAL(CML_INVALID_ARG, cml_get_status(ctx));
}

/* --- dot --- */

static void test_dot_identity(void) {
    /* A * I = A */
    cml_tensor_t *a = cml_tensor_init(ctx, 2, 2);
    cml_tensor_t *eye = cml_tensor_init(ctx, 2, 2);
    cml_tensor_zero(a);
    cml_tensor_zero(eye);
    cml_tensor_set(a, 0, 0, 1.0f); cml_tensor_set(a, 0, 1, 2.0f);
    cml_tensor_set(a, 1, 0, 3.0f); cml_tensor_set(a, 1, 1, 4.0f);
    cml_tensor_set(eye, 0, 0, 1.0f);
    cml_tensor_set(eye, 1, 1, 1.0f);
    cml_tensor_t *c = cml_tensor_dot(ctx, a, eye);
    TEST_ASSERT_NOT_NULL(c);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, cml_tensor_get(c, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(2.0f, cml_tensor_get(c, 0, 1));
    TEST_ASSERT_EQUAL_FLOAT(3.0f, cml_tensor_get(c, 1, 0));
    TEST_ASSERT_EQUAL_FLOAT(4.0f, cml_tensor_get(c, 1, 1));
}

static void test_dot_known_result(void) {
    /* [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]] */
    cml_tensor_t *a = cml_tensor_init(ctx, 2, 2);
    cml_tensor_t *b = cml_tensor_init(ctx, 2, 2);
    cml_tensor_set(a, 0, 0, 1.0f); cml_tensor_set(a, 0, 1, 2.0f);
    cml_tensor_set(a, 1, 0, 3.0f); cml_tensor_set(a, 1, 1, 4.0f);
    cml_tensor_set(b, 0, 0, 5.0f); cml_tensor_set(b, 0, 1, 6.0f);
    cml_tensor_set(b, 1, 0, 7.0f); cml_tensor_set(b, 1, 1, 8.0f);
    cml_tensor_t *c = cml_tensor_dot(ctx, a, b);
    TEST_ASSERT_EQUAL_FLOAT(19.0f, cml_tensor_get(c, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(22.0f, cml_tensor_get(c, 0, 1));
    TEST_ASSERT_EQUAL_FLOAT(43.0f, cml_tensor_get(c, 1, 0));
    TEST_ASSERT_EQUAL_FLOAT(50.0f, cml_tensor_get(c, 1, 1));
}

static void test_dot_non_square(void) {
    /* (2x3) * (3x1) = (2x1) */
    cml_tensor_t *a = cml_tensor_init(ctx, 2, 3);
    cml_tensor_t *b = cml_tensor_init(ctx, 3, 1);
    cml_tensor_set(a, 0, 0, 1.0f); cml_tensor_set(a, 0, 1, 2.0f); cml_tensor_set(a, 0, 2, 3.0f);
    cml_tensor_set(a, 1, 0, 4.0f); cml_tensor_set(a, 1, 1, 5.0f); cml_tensor_set(a, 1, 2, 6.0f);
    cml_tensor_set(b, 0, 0, 1.0f);
    cml_tensor_set(b, 1, 0, 1.0f);
    cml_tensor_set(b, 2, 0, 1.0f);
    cml_tensor_t *c = cml_tensor_dot(ctx, a, b);
    TEST_ASSERT_NOT_NULL(c);
    TEST_ASSERT_EQUAL_size_t(2, cml_tensor_rows(c));
    TEST_ASSERT_EQUAL_size_t(1, cml_tensor_cols(c));
    TEST_ASSERT_EQUAL_FLOAT(6.0f,  cml_tensor_get(c, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(15.0f, cml_tensor_get(c, 1, 0));
}

static void test_dot_incompatible_shapes_errors(void) {
    cml_tensor_t *a = cml_tensor_init(ctx, 2, 3);
    cml_tensor_t *b = cml_tensor_init(ctx, 2, 3);
    TEST_ASSERT_NULL(cml_tensor_dot(ctx, a, b));
    TEST_ASSERT_EQUAL(CML_INVALID_ARG, cml_get_status(ctx));
}

/* --- transpose --- */

static void test_transpose_swaps_shape(void) {
    cml_tensor_t *t = cml_tensor_init(ctx, 2, 5);
    cml_tensor_t *r = cml_tensor_transpose(ctx, t);
    TEST_ASSERT_NOT_NULL(r);
    TEST_ASSERT_EQUAL_size_t(5, cml_tensor_rows(r));
    TEST_ASSERT_EQUAL_size_t(2, cml_tensor_cols(r));
}

static void test_transpose_correct_values(void) {
    cml_tensor_t *t = cml_tensor_init(ctx, 2, 3);
    cml_tensor_set(t, 0, 0, 1.0f); cml_tensor_set(t, 0, 1, 2.0f); cml_tensor_set(t, 0, 2, 3.0f);
    cml_tensor_set(t, 1, 0, 4.0f); cml_tensor_set(t, 1, 1, 5.0f); cml_tensor_set(t, 1, 2, 6.0f);
    cml_tensor_t *r = cml_tensor_transpose(ctx, t);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, cml_tensor_get(r, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(4.0f, cml_tensor_get(r, 0, 1));
    TEST_ASSERT_EQUAL_FLOAT(2.0f, cml_tensor_get(r, 1, 0));
    TEST_ASSERT_EQUAL_FLOAT(5.0f, cml_tensor_get(r, 1, 1));
    TEST_ASSERT_EQUAL_FLOAT(3.0f, cml_tensor_get(r, 2, 0));
    TEST_ASSERT_EQUAL_FLOAT(6.0f, cml_tensor_get(r, 2, 1));
}

static void test_double_transpose_identity(void) {
    cml_tensor_t *t = cml_tensor_init(ctx, 3, 4);
    cml_tensor_rand(t, 0.0f, 1.0f);
    cml_tensor_t *tt = cml_tensor_transpose(ctx, cml_tensor_transpose(ctx, t));
    for (size_t r = 0; r < 3; r++) {
        for (size_t c = 0; c < 4; c++) {
            TEST_ASSERT_EQUAL_FLOAT(cml_tensor_get(t, r, c), cml_tensor_get(tt, r, c));
        }
    }
}

/* --- sum --- */

static void test_sum_known_value(void) {
    cml_tensor_t *t = cml_tensor_init(ctx, 3, 3);
    cml_tensor_fill(t, 2.0f);
    cml_tensor_t *s = cml_tensor_sum(ctx, t);
    TEST_ASSERT_NOT_NULL(s);
    TEST_ASSERT_EQUAL_size_t(1, cml_tensor_rows(s));
    TEST_ASSERT_EQUAL_size_t(1, cml_tensor_cols(s));
    TEST_ASSERT_EQUAL_FLOAT(18.0f, cml_tensor_get(s, 0, 0));
}

static void test_sum_single_element(void) {
    cml_tensor_t *t = cml_tensor_init(ctx, 1, 1);
    cml_tensor_set(t, 0, 0, 7.0f);
    cml_tensor_t *s = cml_tensor_sum(ctx, t);
    TEST_ASSERT_EQUAL_FLOAT(7.0f, cml_tensor_get(s, 0, 0));
}

/* --- sigmoid / relu --- */

static void test_sigmoid_range(void) {
    cml_tensor_t *t = cml_tensor_init(ctx, 1, 3);
    cml_tensor_set(t, 0, 0, -100.0f);
    cml_tensor_set(t, 0, 1,    0.0f);
    cml_tensor_set(t, 0, 2,  100.0f);
    cml_tensor_t *s = cml_tensor_sigmoid(ctx, t);
    TEST_ASSERT_NOT_NULL(s);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 0.0f, cml_tensor_get(s, 0, 0));
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 0.5f, cml_tensor_get(s, 0, 1));
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 1.0f, cml_tensor_get(s, 0, 2));
}

static void test_relu_zeroes_negatives(void) {
    cml_tensor_t *t = cml_tensor_init(ctx, 1, 4);
    cml_tensor_set(t, 0, 0, -3.0f);
    cml_tensor_set(t, 0, 1,  0.0f);
    cml_tensor_set(t, 0, 2,  2.0f);
    cml_tensor_set(t, 0, 3,  5.0f);
    cml_tensor_t *r = cml_tensor_relu(ctx, t);
    TEST_ASSERT_NOT_NULL(r);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, cml_tensor_get(r, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, cml_tensor_get(r, 0, 1));
    TEST_ASSERT_EQUAL_FLOAT(2.0f, cml_tensor_get(r, 0, 2));
    TEST_ASSERT_EQUAL_FLOAT(5.0f, cml_tensor_get(r, 0, 3));
}

static void test_relu_does_not_modify_original(void) {
    cml_tensor_t *t = cml_tensor_init(ctx, 1, 2);
    cml_tensor_set(t, 0, 0, -1.0f);
    cml_tensor_set(t, 0, 1,  3.0f);
    cml_tensor_relu(ctx, t);
    TEST_ASSERT_EQUAL_FLOAT(-1.0f, cml_tensor_get(t, 0, 0));
}

/* --- log --- */

static void test_log_known_values(void) {
    cml_tensor_t *t = cml_tensor_init(ctx, 1, 3);
    cml_tensor_set(t, 0, 0, 1.0f);   /* log(1) = 0         */
    cml_tensor_set(t, 0, 1, 2.0f);   /* log(2) ≈ 0.6931    */
    cml_tensor_set(t, 0, 2, 4.0f);   /* log(4) ≈ 1.3863    */
    cml_tensor_t *r = cml_tensor_log(ctx, t);
    TEST_ASSERT_NOT_NULL(r);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 0.0f,      cml_tensor_get(r, 0, 0));
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.693147f, cml_tensor_get(r, 0, 1));
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 1.386294f, cml_tensor_get(r, 0, 2));
}

static void test_log_output_shape_matches_input(void) {
    cml_tensor_t *t = cml_tensor_init(ctx, 3, 4);
    cml_tensor_fill(t, 1.0f);
    cml_tensor_t *r = cml_tensor_log(ctx, t);
    TEST_ASSERT_NOT_NULL(r);
    TEST_ASSERT_EQUAL_size_t(3, cml_tensor_rows(r));
    TEST_ASSERT_EQUAL_size_t(4, cml_tensor_cols(r));
}

static void test_log_does_not_modify_original(void) {
    cml_tensor_t *t = cml_tensor_init(ctx, 1, 2);
    cml_tensor_set(t, 0, 0, 2.0f);
    cml_tensor_set(t, 0, 1, 4.0f);
    cml_tensor_log(ctx, t);
    TEST_ASSERT_EQUAL_FLOAT(2.0f, cml_tensor_get(t, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(4.0f, cml_tensor_get(t, 0, 1));
}

static void test_log_null_tensor_errors(void) {
    TEST_ASSERT_NULL(cml_tensor_log(ctx, NULL));
    TEST_ASSERT_EQUAL(CML_INVALID_ARG, cml_get_status(ctx));
}

/* --- ctx error propagation --- */

static void test_errored_ctx_short_circuits(void) {
    cml_tensor_t *t = cml_tensor_init(ctx, 2, 2);
    cml_tensor_t *bad = cml_tensor_init(ctx, 3, 3);
    /* poison the context */
    cml_tensor_add(ctx, t, bad);
    TEST_ASSERT_NOT_EQUAL(CML_OK, cml_get_status(ctx));
    /* all subsequent ops should return NULL without crashing */
    TEST_ASSERT_NULL(cml_tensor_init(ctx, 1, 1));
    TEST_ASSERT_NULL(cml_tensor_scale(ctx, t, 2.0f));
    TEST_ASSERT_NULL(cml_tensor_sum(ctx, t));
}

/* --- view + binary op integration --- */

static void test_dot_through_view(void) {
    /* Build a 4x4 parent, take a 2x2 view of the top-left, dot with itself */
    cml_tensor_t *parent = cml_tensor_init(ctx, 4, 4);
    cml_tensor_zero(parent);
    cml_tensor_set(parent, 0, 0, 1.0f); cml_tensor_set(parent, 0, 1, 0.0f);
    cml_tensor_set(parent, 1, 0, 0.0f); cml_tensor_set(parent, 1, 1, 2.0f);
    cml_tensor_t *v = cml_tensor_view(ctx, parent, 0, 0, 2, 2);
    cml_tensor_t *c = cml_tensor_dot(ctx, v, v);
    TEST_ASSERT_NOT_NULL(c);
    /* [[1,0],[0,2]] * [[1,0],[0,2]] = [[1,0],[0,4]] */
    TEST_ASSERT_EQUAL_FLOAT(1.0f, cml_tensor_get(c, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, cml_tensor_get(c, 0, 1));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, cml_tensor_get(c, 1, 0));
    TEST_ASSERT_EQUAL_FLOAT(4.0f, cml_tensor_get(c, 1, 1));
    TEST_ASSERT_EQUAL(CML_OK, cml_get_status(ctx));
}

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_init_sets_shape);
    RUN_TEST(test_init_stride_equals_cols);
    RUN_TEST(test_init_null_ctx_returns_null);
    RUN_TEST(test_accessor_null_tensor);

    RUN_TEST(test_set_get_roundtrip);
    RUN_TEST(test_data_pointer_matches_get);
    RUN_TEST(test_const_data_pointer_matches_get);
    RUN_TEST(test_sync_api_cpu_context);

    RUN_TEST(test_zero_clears_all_elements);
    RUN_TEST(test_fill_sets_all_elements);
    RUN_TEST(test_rand_stays_in_range);

    RUN_TEST(test_copy_duplicates_values);
    RUN_TEST(test_copy_is_independent);

    RUN_TEST(test_reshape_changes_shape);
    RUN_TEST(test_reshape_shares_data);
    RUN_TEST(test_reshape_element_count_mismatch_errors);

    RUN_TEST(test_view_correct_shape);
    RUN_TEST(test_view_reads_parent_data);
    RUN_TEST(test_view_write_visible_in_parent);
    RUN_TEST(test_view_out_of_bounds_errors);
    RUN_TEST(test_view_inherits_parent_stride);

    RUN_TEST(test_scale_produces_scaled_copy);
    RUN_TEST(test_scale_does_not_modify_original);

    RUN_TEST(test_add_elementwise);
    RUN_TEST(test_sub_elementwise);
    RUN_TEST(test_mul_elementwise);
    RUN_TEST(test_add_shape_mismatch_errors);

    RUN_TEST(test_dot_identity);
    RUN_TEST(test_dot_known_result);
    RUN_TEST(test_dot_non_square);
    RUN_TEST(test_dot_incompatible_shapes_errors);

    RUN_TEST(test_transpose_swaps_shape);
    RUN_TEST(test_transpose_correct_values);
    RUN_TEST(test_double_transpose_identity);

    RUN_TEST(test_sum_known_value);
    RUN_TEST(test_sum_single_element);

    RUN_TEST(test_sigmoid_range);
    RUN_TEST(test_relu_zeroes_negatives);
    RUN_TEST(test_relu_does_not_modify_original);

    RUN_TEST(test_log_known_values);
    RUN_TEST(test_log_output_shape_matches_input);
    RUN_TEST(test_log_does_not_modify_original);
    RUN_TEST(test_log_null_tensor_errors);

    RUN_TEST(test_errored_ctx_short_circuits);
    RUN_TEST(test_dot_through_view);

    return UNITY_END();
}
