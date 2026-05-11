#include "backend.h"

#include <cblas.h>
#include <math.h>

static cml_status_t cpu_init(void **state) {
    if (state != NULL) {
        *state = NULL;
    }
    return CML_OK;
}

static void cpu_deinit(void *state) {
    (void)state;
}

static cml_status_t cpu_copy(void *state,
                             float *dst, size_t dst_stride,
                             const float *src, size_t src_stride,
                             size_t rows, size_t cols) {
    (void)state;
    for (size_t r = 0; r < rows; r++) {
        cblas_scopy((int)cols, src + r * src_stride, 1, dst + r * dst_stride, 1);
    }
    return CML_OK;
}

static cml_status_t cpu_add_scaled(void *state,
                                   float *out, size_t out_stride,
                                   const float *a, size_t a_stride,
                                   const float *b, size_t b_stride,
                                   size_t rows, size_t cols, float alpha) {
    (void)state;
    for (size_t r = 0; r < rows; r++) {
        cblas_scopy((int)cols, a + r * a_stride, 1, out + r * out_stride, 1);
        cblas_saxpy((int)cols, alpha, b + r * b_stride, 1, out + r * out_stride, 1);
    }
    return CML_OK;
}

static cml_status_t cpu_mul(void *state,
                            float *out, size_t out_stride,
                            const float *a, size_t a_stride,
                            const float *b, size_t b_stride,
                            size_t rows, size_t cols) {
    (void)state;
    for (size_t r = 0; r < rows; r++) {
        const float *ar = a + r * a_stride;
        const float *br = b + r * b_stride;
        float *orow = out + r * out_stride;
        for (size_t c = 0; c < cols; c++) {
            orow[c] = ar[c] * br[c];
        }
    }
    return CML_OK;
}

static cml_status_t cpu_scale(void *state,
                              float *out, size_t out_stride,
                              const float *in, size_t in_stride,
                              size_t rows, size_t cols, float scalar) {
    (void)state;
    for (size_t r = 0; r < rows; r++) {
        cblas_scopy((int)cols, in + r * in_stride, 1, out + r * out_stride, 1);
        cblas_sscal((int)cols, scalar, out + r * out_stride, 1);
    }
    return CML_OK;
}

static cml_status_t cpu_dot(void *state,
                            float *out, size_t out_stride,
                            const float *a, size_t a_stride,
                            const float *b, size_t b_stride,
                            size_t m, size_t n, size_t k) {
    (void)state;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)m, (int)n, (int)k,
                1.0f,
                a, (int)a_stride,
                b, (int)b_stride,
                0.0f,
                out, (int)out_stride);
    return CML_OK;
}

static cml_status_t cpu_transpose(void *state,
                                  float *out, size_t out_stride,
                                  const float *in, size_t in_stride,
                                  size_t rows, size_t cols) {
    (void)state;
    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c++) {
            out[c * out_stride + r] = in[r * in_stride + c];
        }
    }
    return CML_OK;
}

static cml_status_t cpu_sum(void *state,
                            const float *in, size_t in_stride,
                            size_t rows, size_t cols, float *out) {
    (void)state;
    float total = 0.0f;
    for (size_t r = 0; r < rows; r++) {
        const float *row = in + r * in_stride;
        for (size_t c = 0; c < cols; c++) {
            total += row[c];
        }
    }
    *out = total;
    return CML_OK;
}

static cml_status_t cpu_unary(void *state,
                              float *out, size_t out_stride,
                              const float *in, size_t in_stride,
                              size_t rows, size_t cols, cml_unary_op_t op) {
    (void)state;
    for (size_t r = 0; r < rows; r++) {
        const float *src = in + r * in_stride;
        float *dst = out + r * out_stride;
        for (size_t c = 0; c < cols; c++) {
            float v = src[c];
            switch (op) {
                case CML_UNARY_LOG:
                    dst[c] = logf(v);
                    break;
                case CML_UNARY_SIGMOID:
                    dst[c] = 1.0f / (1.0f + expf(-v));
                    break;
                case CML_UNARY_RELU:
                    dst[c] = v > 0.0f ? v : 0.0f;
                    break;
                default:
                    return CML_INVALID_ARG;
            }
        }
    }
    return CML_OK;
}

static cml_status_t cpu_unary_grad(void *state,
                                   float *out, size_t out_stride,
                                   const float *x, size_t x_stride,
                                   const float *grad, size_t grad_stride,
                                   size_t rows, size_t cols, cml_unary_op_t op) {
    (void)state;
    for (size_t r = 0; r < rows; r++) {
        const float *xr = x + r * x_stride;
        const float *gr = grad + r * grad_stride;
        float *or_ = out + r * out_stride;
        for (size_t c = 0; c < cols; c++) {
            float xv = xr[c];
            float deriv;
            switch (op) {
                case CML_UNARY_LOG:
                    deriv = xv != 0.0f ? 1.0f / xv : 0.0f;
                    break;
                case CML_UNARY_SIGMOID: {
                    float s = 1.0f / (1.0f + expf(-xv));
                    deriv = s * (1.0f - s);
                    break;
                }
                case CML_UNARY_RELU:
                    deriv = xv > 0.0f ? 1.0f : 0.0f;
                    break;
                default:
                    return CML_INVALID_ARG;
            }
            or_[c] = gr[c] * deriv;
        }
    }
    return CML_OK;
}

static cml_status_t cpu_sum_rows(void *state,
                                 float *out,
                                 const float *in, size_t in_stride,
                                 size_t rows, size_t cols) {
    (void)state;
    for (size_t c = 0; c < cols; c++) out[c] = 0.0f;
    for (size_t r = 0; r < rows; r++) {
        const float *row = in + r * in_stride;
        for (size_t c = 0; c < cols; c++) out[c] += row[c];
    }
    return CML_OK;
}

static cml_status_t cpu_accum_scaled(void *state,
                                     float *dst, size_t dst_stride,
                                     const float *src, size_t src_stride,
                                     size_t rows, size_t cols, float scale) {
    (void)state;
    for (size_t r = 0; r < rows; r++) {
        cblas_saxpy((int)cols, scale, src + r * src_stride, 1, dst + r * dst_stride, 1);
    }
    return CML_OK;
}

static cml_status_t cpu_add_bias(void *state,
                                 float *out, size_t out_stride,
                                 const float *a, size_t a_stride,
                                 const float *b, size_t b_stride,
                                 size_t rows, size_t cols) {
    (void)state;
    (void)b_stride;
    for (size_t r = 0; r < rows; r++) {
        cblas_scopy((int)cols, a + r * a_stride, 1, out + r * out_stride, 1);
        cblas_saxpy((int)cols, 1.0f, b, 1, out + r * out_stride, 1);
    }
    return CML_OK;
}

const cml_backend_ops_t cml_cpu_backend_ops = {
    .uses_device = false,
    .init = cpu_init,
    .deinit = cpu_deinit,
    .alloc_device = NULL,
    .free_device = NULL,
    .copy_host_to_device = NULL,
    .copy_device_to_host = NULL,
    .copy = cpu_copy,
    .add_scaled = cpu_add_scaled,
    .mul = cpu_mul,
    .scale = cpu_scale,
    .dot = cpu_dot,
    .transpose = cpu_transpose,
    .sum = cpu_sum,
    .unary = cpu_unary,
    .unary_grad = cpu_unary_grad,
    .sum_rows = cpu_sum_rows,
    .accum_scaled = cpu_accum_scaled,
    .add_bias = cpu_add_bias,
};
