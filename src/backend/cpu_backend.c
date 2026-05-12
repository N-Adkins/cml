#include "backend.h"

#include <cblas.h>
#include <float.h>
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
    // Kahan-Neumaier compensated summation to retain precision on large inputs.
    float total = 0.0f;
    float comp = 0.0f;
    for (size_t r = 0; r < rows; r++) {
        const float *row = in + r * in_stride;
        for (size_t c = 0; c < cols; c++) {
            float y = row[c] - comp;
            float t = total + y;
            comp = (t - total) - y;
            total = t;
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
                    // Floor at FLT_MIN so log(0) returns a finite (very negative)
                    // value instead of -inf, which would NaN-poison gradients.
                    dst[c] = logf(v > FLT_MIN ? v : FLT_MIN);
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

static cml_status_t cpu_adam_step(void *state,
                                  float *p, size_t p_stride,
                                  float *m, size_t m_stride,
                                  float *v, size_t v_stride,
                                  const float *g, size_t g_stride,
                                  size_t rows, size_t cols,
                                  float lr, float beta1, float beta2, float eps,
                                  float bc1, float bc2) {
    (void)state;
    const float one_minus_b1 = 1.0f - beta1;
    const float one_minus_b2 = 1.0f - beta2;
    for (size_t r = 0; r < rows; r++) {
        float *pr = p + r * p_stride;
        float *mr = m + r * m_stride;
        float *vr = v + r * v_stride;
        const float *gr = g + r * g_stride;
        for (size_t c = 0; c < cols; c++) {
            float gv = gr[c];
            float new_m = beta1 * mr[c] + one_minus_b1 * gv;
            float new_v = beta2 * vr[c] + one_minus_b2 * gv * gv;
            mr[c] = new_m;
            vr[c] = new_v;
            float m_hat = new_m / bc1;
            float v_hat = new_v / bc2;
            pr[c] -= lr * m_hat / (sqrtf(v_hat) + eps);
        }
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
    .adam_step = cpu_adam_step,
};
