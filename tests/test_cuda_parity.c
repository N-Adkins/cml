#include <cml/context.h>
#include <cml/tensor.h>
#include "unity.h"

#define CTX_SIZE (4 * 1024 * 1024)
#define EPS 1e-4f

static cml_context_t *cpu_ctx;
static cml_context_t *cuda_ctx;

static void require_cuda_runtime(void) {
    if (cuda_ctx == NULL) {
        TEST_IGNORE_MESSAGE("CUDA backend/device unavailable at runtime");
    }
}

static cml_tensor_t *tensor_from_values(cml_context_t *ctx, size_t rows, size_t cols, const float *values) {
    cml_tensor_t *tensor = cml_tensor_init(ctx, rows, cols);
    TEST_ASSERT_NOT_NULL(tensor);
    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c++) {
            cml_tensor_set(tensor, r, c, values[r * cols + c]);
        }
    }
    return tensor;
}

static void assert_tensor_close(const cml_tensor_t *expected, const cml_tensor_t *actual,
                                size_t rows, size_t cols, float delta) {
    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c++) {
            TEST_ASSERT_FLOAT_WITHIN(delta, cml_tensor_get(expected, r, c), cml_tensor_get(actual, r, c));
        }
    }
}

void setUp(void) {
    cpu_ctx = cml_init_with_backend(CTX_SIZE, CML_BACKEND_CPU);
    cuda_ctx = cml_init_with_backend(CTX_SIZE, CML_BACKEND_CUDA);
}

void tearDown(void) {
    cml_deinit(cuda_ctx);
    cml_deinit(cpu_ctx);
}

static void test_cuda_compiled_in_when_enabled(void) {
    TEST_ASSERT_TRUE(cml_cuda_is_available());
}

static void test_cuda_backend_selected_for_cuda_context(void) {
    require_cuda_runtime();
    TEST_ASSERT_EQUAL(CML_BACKEND_CUDA, cml_get_backend(cuda_ctx));
}

static void test_cuda_parity_for_elementwise_ops(void) {
    static const float a_data[] = {
        1.5f, -2.0f, 3.5f,
        4.0f, 0.5f, -6.0f
    };
    static const float b_data[] = {
        -0.5f, 2.0f, -1.5f,
        3.0f, -4.5f, 2.0f
    };

    require_cuda_runtime();

    cml_tensor_t *a_cpu = tensor_from_values(cpu_ctx, 2, 3, a_data);
    cml_tensor_t *a_cuda = tensor_from_values(cuda_ctx, 2, 3, a_data);
    cml_tensor_t *b_cpu = tensor_from_values(cpu_ctx, 2, 3, b_data);
    cml_tensor_t *b_cuda = tensor_from_values(cuda_ctx, 2, 3, b_data);

    cml_tensor_t *add_cpu = cml_tensor_add(cpu_ctx, a_cpu, b_cpu);
    cml_tensor_t *add_cuda = cml_tensor_add(cuda_ctx, a_cuda, b_cuda);
    TEST_ASSERT_NOT_NULL(add_cpu);
    TEST_ASSERT_NOT_NULL(add_cuda);
    assert_tensor_close(add_cpu, add_cuda, 2, 3, EPS);

    cml_tensor_t *mul_cpu = cml_tensor_mul(cpu_ctx, a_cpu, b_cpu);
    cml_tensor_t *mul_cuda = cml_tensor_mul(cuda_ctx, a_cuda, b_cuda);
    TEST_ASSERT_NOT_NULL(mul_cpu);
    TEST_ASSERT_NOT_NULL(mul_cuda);
    assert_tensor_close(mul_cpu, mul_cuda, 2, 3, EPS);
}

static void test_cuda_parity_for_dot_and_transpose(void) {
    static const float lhs_data[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    static const float rhs_data[] = {
        2.0f, 1.0f,
        0.0f, -1.0f,
        3.0f, 4.0f
    };

    require_cuda_runtime();

    cml_tensor_t *lhs_cpu = tensor_from_values(cpu_ctx, 2, 3, lhs_data);
    cml_tensor_t *lhs_cuda = tensor_from_values(cuda_ctx, 2, 3, lhs_data);
    cml_tensor_t *rhs_cpu = tensor_from_values(cpu_ctx, 3, 2, rhs_data);
    cml_tensor_t *rhs_cuda = tensor_from_values(cuda_ctx, 3, 2, rhs_data);

    cml_tensor_t *dot_cpu = cml_tensor_dot(cpu_ctx, lhs_cpu, rhs_cpu);
    cml_tensor_t *dot_cuda = cml_tensor_dot(cuda_ctx, lhs_cuda, rhs_cuda);
    TEST_ASSERT_NOT_NULL(dot_cpu);
    TEST_ASSERT_NOT_NULL(dot_cuda);
    assert_tensor_close(dot_cpu, dot_cuda, 2, 2, EPS);

    cml_tensor_t *tp_cpu = cml_tensor_transpose(cpu_ctx, dot_cpu);
    cml_tensor_t *tp_cuda = cml_tensor_transpose(cuda_ctx, dot_cuda);
    TEST_ASSERT_NOT_NULL(tp_cpu);
    TEST_ASSERT_NOT_NULL(tp_cuda);
    assert_tensor_close(tp_cpu, tp_cuda, 2, 2, EPS);
}

static void test_cuda_parity_for_unary_and_sum(void) {
    static const float input_data[] = {
        0.25f, 1.0f, 2.5f,
        4.0f, 0.125f, 8.0f
    };

    require_cuda_runtime();

    cml_tensor_t *input_cpu = tensor_from_values(cpu_ctx, 2, 3, input_data);
    cml_tensor_t *input_cuda = tensor_from_values(cuda_ctx, 2, 3, input_data);

    cml_tensor_t *sig_cpu = cml_tensor_sigmoid(cpu_ctx, input_cpu);
    cml_tensor_t *sig_cuda = cml_tensor_sigmoid(cuda_ctx, input_cuda);
    TEST_ASSERT_NOT_NULL(sig_cpu);
    TEST_ASSERT_NOT_NULL(sig_cuda);
    assert_tensor_close(sig_cpu, sig_cuda, 2, 3, EPS);

    cml_tensor_t *sum_cpu = cml_tensor_sum(cpu_ctx, sig_cpu);
    cml_tensor_t *sum_cuda = cml_tensor_sum(cuda_ctx, sig_cuda);
    TEST_ASSERT_NOT_NULL(sum_cpu);
    TEST_ASSERT_NOT_NULL(sum_cuda);
    TEST_ASSERT_FLOAT_WITHIN(EPS, cml_tensor_get(sum_cpu, 0, 0), cml_tensor_get(sum_cuda, 0, 0));
}

static void test_cuda_set_after_op_preserves_other_values(void) {
    /* Regression: cml_tensor_set used to skip the device→host sync, so writing
     * a single element after a CUDA op corrupted everything else in the tensor. */
    require_cuda_runtime();

    cml_tensor_t *a = cml_tensor_init(cuda_ctx, 2, 3);
    cml_tensor_t *b = cml_tensor_init(cuda_ctx, 2, 3);
    cml_tensor_fill(a, 1.0f);
    cml_tensor_fill(b, 2.0f);

    /* This op leaves c authoritative on device only. */
    cml_tensor_t *c = cml_tensor_add(cuda_ctx, a, b);
    TEST_ASSERT_NOT_NULL(c);
    TEST_ASSERT_TRUE(cml_tensor_has_device_copy(c));

    /* Overwrite a single element. Every other cell should retain its 3.0f value. */
    cml_tensor_set(c, 0, 0, 99.0f);

    TEST_ASSERT_EQUAL_FLOAT(99.0f, cml_tensor_get(c, 0, 0));
    for (size_t r = 0; r < 2; r++) {
        for (size_t cc = 0; cc < 3; cc++) {
            if (r == 0 && cc == 0) continue;
            TEST_ASSERT_EQUAL_FLOAT(3.0f, cml_tensor_get(c, r, cc));
        }
    }
}

static void test_cuda_sum_is_deterministic(void) {
    /* The old implementation used a single-accumulator atomicAdd whose result
     * varied run-to-run. The block-reduction sum should be bit-exact across
     * back-to-back invocations on the same data. */
    require_cuda_runtime();

    const size_t n = 4096;
    cml_tensor_t *t = cml_tensor_init(cuda_ctx, 1, n);
    for (size_t i = 0; i < n; i++) {
        /* A mix of magnitudes so non-determinism in float add order would show. */
        cml_tensor_set(t, 0, i, ((float)i - (float)n / 2.0f) * 0.01f);
    }

    float first = cml_tensor_get(cml_tensor_sum(cuda_ctx, t), 0, 0);
    for (int trial = 0; trial < 5; trial++) {
        float again = cml_tensor_get(cml_tensor_sum(cuda_ctx, t), 0, 0);
        TEST_ASSERT_EQUAL_FLOAT(first, again);
    }
}

static void test_cuda_empty_tensor_ops_do_not_crash(void) {
    require_cuda_runtime();
    cml_tensor_t *t = cml_tensor_init(cuda_ctx, 0, 4);
    TEST_ASSERT_NOT_NULL(t);
    cml_tensor_zero(t);
    cml_tensor_fill(t, 1.0f);
    TEST_ASSERT_EQUAL(CML_OK, cml_get_status(cuda_ctx));
}

static void test_cuda_device_sync_roundtrip_matches_cpu(void) {
    static const float input_data[] = {
        9.0f, 8.0f,
        7.0f, 6.0f
    };

    require_cuda_runtime();

    cml_tensor_t *cpu_tensor = tensor_from_values(cpu_ctx, 2, 2, input_data);
    cml_tensor_t *cuda_tensor = tensor_from_values(cuda_ctx, 2, 2, input_data);

    TEST_ASSERT_EQUAL(CML_OK, cml_tensor_to_device(cuda_ctx, cuda_tensor));
    TEST_ASSERT_TRUE(cml_tensor_has_device_copy(cuda_tensor));
    TEST_ASSERT_EQUAL(CML_OK, cml_tensor_to_host(cuda_ctx, cuda_tensor));
    assert_tensor_close(cpu_tensor, cuda_tensor, 2, 2, EPS);
}

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_cuda_compiled_in_when_enabled);
    RUN_TEST(test_cuda_backend_selected_for_cuda_context);
    RUN_TEST(test_cuda_parity_for_elementwise_ops);
    RUN_TEST(test_cuda_parity_for_dot_and_transpose);
    RUN_TEST(test_cuda_parity_for_unary_and_sum);
    RUN_TEST(test_cuda_device_sync_roundtrip_matches_cpu);
    RUN_TEST(test_cuda_set_after_op_preserves_other_values);
    RUN_TEST(test_cuda_sum_is_deterministic);
    RUN_TEST(test_cuda_empty_tensor_ops_do_not_crash);

    return UNITY_END();
}
