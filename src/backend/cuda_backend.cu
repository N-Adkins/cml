#include "backend.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <math.h>
#include <stdlib.h>

typedef struct {
    cublasHandle_t cublas;
    float *scalar_buf;
} cml_cuda_state_t;

static cml_status_t map_cuda_status(cudaError_t status) {
    if (status == cudaSuccess) return CML_OK;
    if (status == cudaErrorMemoryAllocation) return CML_OUT_OF_MEMORY;
    return CML_BACKEND_UNAVAILABLE;
}

static cml_status_t map_cublas_status(cublasStatus_t status) {
    if (status == CUBLAS_STATUS_SUCCESS) return CML_OK;
    if (status == CUBLAS_STATUS_ALLOC_FAILED) return CML_OUT_OF_MEMORY;
    return CML_BACKEND_UNAVAILABLE;
}

static cml_status_t finish_kernel(void) {
    cudaError_t status = cudaGetLastError();
    return map_cuda_status(status);
}

static __global__ void kernel_copy(float *dst, size_t dst_stride,
                                   const float *src, size_t src_stride,
                                   size_t rows, size_t cols) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t total = rows * cols;
    if (idx >= total) return;
    size_t r = idx / cols;
    size_t c = idx % cols;
    dst[r * dst_stride + c] = src[r * src_stride + c];
}

static __global__ void kernel_add_scaled(float *out, size_t out_stride,
                                         const float *a, size_t a_stride,
                                         const float *b, size_t b_stride,
                                         size_t rows, size_t cols, float alpha) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t total = rows * cols;
    if (idx >= total) return;
    size_t r = idx / cols;
    size_t c = idx % cols;
    out[r * out_stride + c] = a[r * a_stride + c] + alpha * b[r * b_stride + c];
}

static __global__ void kernel_mul(float *out, size_t out_stride,
                                  const float *a, size_t a_stride,
                                  const float *b, size_t b_stride,
                                  size_t rows, size_t cols) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t total = rows * cols;
    if (idx >= total) return;
    size_t r = idx / cols;
    size_t c = idx % cols;
    out[r * out_stride + c] = a[r * a_stride + c] * b[r * b_stride + c];
}

static __global__ void kernel_scale(float *out, size_t out_stride,
                                    const float *in, size_t in_stride,
                                    size_t rows, size_t cols, float scalar) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t total = rows * cols;
    if (idx >= total) return;
    size_t r = idx / cols;
    size_t c = idx % cols;
    out[r * out_stride + c] = in[r * in_stride + c] * scalar;
}

static __global__ void kernel_transpose(float *out, size_t out_stride,
                                        const float *in, size_t in_stride,
                                        size_t rows, size_t cols) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t total = rows * cols;
    if (idx >= total) return;
    size_t r = idx / cols;
    size_t c = idx % cols;
    out[c * out_stride + r] = in[r * in_stride + c];
}

static __global__ void kernel_unary(float *out, size_t out_stride,
                                    const float *in, size_t in_stride,
                                    size_t rows, size_t cols, int op) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t total = rows * cols;
    if (idx >= total) return;
    size_t r = idx / cols;
    size_t c = idx % cols;
    float v = in[r * in_stride + c];
    float outv = v;
    if (op == CML_UNARY_LOG) {
        outv = logf(v);
    } else if (op == CML_UNARY_SIGMOID) {
        outv = 1.0f / (1.0f + expf(-v));
    } else if (op == CML_UNARY_RELU) {
        outv = v > 0.0f ? v : 0.0f;
    }
    out[r * out_stride + c] = outv;
}

static __global__ void kernel_accum_scaled(float *dst, size_t dst_stride,
                                           const float *src, size_t src_stride,
                                           size_t rows, size_t cols, float scale) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t total = rows * cols;
    if (idx >= total) return;
    size_t r = idx / cols;
    size_t c = idx % cols;
    dst[r * dst_stride + c] += scale * src[r * src_stride + c];
}

static __global__ void kernel_add_bias(float *out, size_t out_stride,
                                       const float *a, size_t a_stride,
                                       const float *b, size_t b_stride,
                                       size_t rows, size_t cols) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t total = rows * cols;
    if (idx >= total) return;
    size_t r = idx / cols;
    size_t c = idx % cols;
    (void)b_stride;
    out[r * out_stride + c] = a[r * a_stride + c] + b[c];
}

static __global__ void kernel_sum_atomic(const float *in, size_t in_stride,
                                         size_t rows, size_t cols, float *out) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t total = rows * cols;
    if (idx >= total) return;
    size_t r = idx / cols;
    size_t c = idx % cols;
    atomicAdd(out, in[r * in_stride + c]);
}

static cml_status_t cuda_init(void **state) {
    cml_cuda_state_t *cuda_state = (cml_cuda_state_t *)malloc(sizeof(cml_cuda_state_t));
    if (cuda_state == NULL) return CML_OUT_OF_MEMORY;

    cublasStatus_t cublas_status = cublasCreate(&cuda_state->cublas);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        free(cuda_state);
        return map_cublas_status(cublas_status);
    }

    cublas_status = cublasSetPointerMode(cuda_state->cublas, CUBLAS_POINTER_MODE_HOST);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(cuda_state->cublas);
        free(cuda_state);
        return map_cublas_status(cublas_status);
    }

    cudaError_t cuda_status = cudaMalloc((void **)&cuda_state->scalar_buf, sizeof(float));
    if (cuda_status != cudaSuccess) {
        cublasDestroy(cuda_state->cublas);
        free(cuda_state);
        return map_cuda_status(cuda_status);
    }

    *state = cuda_state;
    return CML_OK;
}

static void cuda_deinit(void *state) {
    if (state == NULL) return;
    cml_cuda_state_t *cuda_state = (cml_cuda_state_t *)state;
    cudaFree(cuda_state->scalar_buf);
    cublasDestroy(cuda_state->cublas);
    free(cuda_state);
}

static cml_status_t cuda_alloc_device(void *state, size_t elements, float **ptr) {
    (void)state;
    cudaError_t status = cudaMalloc((void **)ptr, elements * sizeof(float));
    return map_cuda_status(status);
}

static void cuda_free_device(void *state, float *ptr) {
    (void)state;
    if (ptr != NULL) {
        cudaFree(ptr);
    }
}

static cml_status_t cuda_copy_host_to_device(void *state, float *dst, const float *src, size_t elements) {
    (void)state;
    cudaError_t status = cudaMemcpy(dst, src, elements * sizeof(float), cudaMemcpyHostToDevice);
    return map_cuda_status(status);
}

static cml_status_t cuda_copy_device_to_host(void *state, float *dst, const float *src, size_t elements) {
    (void)state;
    cudaError_t status = cudaMemcpy(dst, src, elements * sizeof(float), cudaMemcpyDeviceToHost);
    return map_cuda_status(status);
}

static cml_status_t cuda_copy(void *state,
                              float *dst, size_t dst_stride,
                              const float *src, size_t src_stride,
                              size_t rows, size_t cols) {
    (void)state;
    size_t total = rows * cols;
    int block = 256;
    int grid = (int)((total + (size_t)block - 1U) / (size_t)block);
    kernel_copy<<<grid, block>>>(dst, dst_stride, src, src_stride, rows, cols);
    return finish_kernel();
}

static cml_status_t cuda_add_scaled(void *state,
                                    float *out, size_t out_stride,
                                    const float *a, size_t a_stride,
                                    const float *b, size_t b_stride,
                                    size_t rows, size_t cols, float alpha) {
    (void)state;
    size_t total = rows * cols;
    int block = 256;
    int grid = (int)((total + (size_t)block - 1U) / (size_t)block);
    kernel_add_scaled<<<grid, block>>>(out, out_stride, a, a_stride, b, b_stride, rows, cols, alpha);
    return finish_kernel();
}

static cml_status_t cuda_mul(void *state,
                             float *out, size_t out_stride,
                             const float *a, size_t a_stride,
                             const float *b, size_t b_stride,
                             size_t rows, size_t cols) {
    (void)state;
    size_t total = rows * cols;
    int block = 256;
    int grid = (int)((total + (size_t)block - 1U) / (size_t)block);
    kernel_mul<<<grid, block>>>(out, out_stride, a, a_stride, b, b_stride, rows, cols);
    return finish_kernel();
}

static cml_status_t cuda_scale(void *state,
                               float *out, size_t out_stride,
                               const float *in, size_t in_stride,
                               size_t rows, size_t cols, float scalar) {
    (void)state;
    size_t total = rows * cols;
    int block = 256;
    int grid = (int)((total + (size_t)block - 1U) / (size_t)block);
    kernel_scale<<<grid, block>>>(out, out_stride, in, in_stride, rows, cols, scalar);
    return finish_kernel();
}

static cml_status_t cuda_dot(void *state,
                             float *out, size_t out_stride,
                             const float *a, size_t a_stride,
                             const float *b, size_t b_stride,
                             size_t m, size_t n, size_t k) {
    cml_cuda_state_t *cuda_state = (cml_cuda_state_t *)state;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasStatus_t status = cublasSgemm(
        cuda_state->cublas,
        CUBLAS_OP_N, CUBLAS_OP_N,
        (int)n, (int)m, (int)k,
        &alpha,
        b, (int)b_stride,
        a, (int)a_stride,
        &beta,
        out, (int)out_stride);
    return map_cublas_status(status);
}

static cml_status_t cuda_transpose(void *state,
                                   float *out, size_t out_stride,
                                   const float *in, size_t in_stride,
                                   size_t rows, size_t cols) {
    (void)state;
    size_t total = rows * cols;
    int block = 256;
    int grid = (int)((total + (size_t)block - 1U) / (size_t)block);
    kernel_transpose<<<grid, block>>>(out, out_stride, in, in_stride, rows, cols);
    return finish_kernel();
}

static cml_status_t cuda_sum(void *state,
                             const float *in, size_t in_stride,
                             size_t rows, size_t cols, float *out) {
    cml_cuda_state_t *cuda_state = (cml_cuda_state_t *)state;

    cudaError_t cuda_status = cudaMemset(cuda_state->scalar_buf, 0, sizeof(float));
    if (cuda_status != cudaSuccess) return map_cuda_status(cuda_status);

    size_t total = rows * cols;
    int block = 256;
    int grid = (int)((total + (size_t)block - 1U) / (size_t)block);
    kernel_sum_atomic<<<grid, block>>>(in, in_stride, rows, cols, cuda_state->scalar_buf);
    cml_status_t status = finish_kernel();
    if (status != CML_OK) return status;

    cuda_status = cudaMemcpy(out, cuda_state->scalar_buf, sizeof(float), cudaMemcpyDeviceToHost);
    return map_cuda_status(cuda_status);
}

static cml_status_t cuda_unary(void *state,
                               float *out, size_t out_stride,
                               const float *in, size_t in_stride,
                               size_t rows, size_t cols, cml_unary_op_t op) {
    (void)state;
    size_t total = rows * cols;
    int block = 256;
    int grid = (int)((total + (size_t)block - 1U) / (size_t)block);
    kernel_unary<<<grid, block>>>(out, out_stride, in, in_stride, rows, cols, (int)op);
    return finish_kernel();
}

static cml_status_t cuda_accum_scaled(void *state,
                                      float *dst, size_t dst_stride,
                                      const float *src, size_t src_stride,
                                      size_t rows, size_t cols, float scale) {
    (void)state;
    size_t total = rows * cols;
    int block = 256;
    int grid = (int)((total + (size_t)block - 1U) / (size_t)block);
    kernel_accum_scaled<<<grid, block>>>(dst, dst_stride, src, src_stride, rows, cols, scale);
    return finish_kernel();
}

static cml_status_t cuda_add_bias(void *state,
                                  float *out, size_t out_stride,
                                  const float *a, size_t a_stride,
                                  const float *b, size_t b_stride,
                                  size_t rows, size_t cols) {
    (void)state;
    size_t total = rows * cols;
    int block = 256;
    int grid = (int)((total + (size_t)block - 1U) / (size_t)block);
    kernel_add_bias<<<grid, block>>>(out, out_stride, a, a_stride, b, b_stride, rows, cols);
    return finish_kernel();
}

static __global__ void kernel_unary_grad(float *out, size_t out_stride,
                                         const float *x, size_t x_stride,
                                         const float *grad, size_t grad_stride,
                                         size_t rows, size_t cols, int op) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t total = rows * cols;
    if (idx >= total) return;
    size_t r = idx / cols;
    size_t c = idx % cols;
    float xv = x[r * x_stride + c];
    float gv = grad[r * grad_stride + c];
    float deriv;
    if (op == CML_UNARY_LOG) {
        deriv = xv != 0.0f ? 1.0f / xv : 0.0f;
    } else if (op == CML_UNARY_SIGMOID) {
        float s = 1.0f / (1.0f + expf(-xv));
        deriv = s * (1.0f - s);
    } else {
        deriv = xv > 0.0f ? 1.0f : 0.0f;
    }
    out[r * out_stride + c] = gv * deriv;
}

static cml_status_t cuda_unary_grad(void *state,
                                    float *out, size_t out_stride,
                                    const float *x, size_t x_stride,
                                    const float *grad, size_t grad_stride,
                                    size_t rows, size_t cols, cml_unary_op_t op) {
    (void)state;
    size_t total = rows * cols;
    int block = 256;
    int grid = (int)((total + (size_t)block - 1U) / (size_t)block);
    kernel_unary_grad<<<grid, block>>>(out, out_stride, x, x_stride, grad, grad_stride, rows, cols, (int)op);
    return finish_kernel();
}

static __global__ void kernel_sum_rows(const float *in, size_t in_stride,
                                       float *out, size_t cols, size_t rows) {
    size_t c = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (c >= cols) return;
    float sum = 0.0f;
    for (size_t r = 0; r < rows; r++) sum += in[r * in_stride + c];
    out[c] = sum;
}

static cml_status_t cuda_sum_rows(void *state,
                                  float *out,
                                  const float *in, size_t in_stride,
                                  size_t rows, size_t cols) {
    (void)state;
    int block = 256;
    int grid = (int)((cols + (size_t)block - 1U) / (size_t)block);
    kernel_sum_rows<<<grid, block>>>(in, in_stride, out, cols, rows);
    return finish_kernel();
}

extern "C" const cml_backend_ops_t cml_cuda_backend_ops = {
    .uses_device = true,
    .init = cuda_init,
    .deinit = cuda_deinit,
    .alloc_device = cuda_alloc_device,
    .free_device = cuda_free_device,
    .copy_host_to_device = cuda_copy_host_to_device,
    .copy_device_to_host = cuda_copy_device_to_host,
    .copy = cuda_copy,
    .add_scaled = cuda_add_scaled,
    .mul = cuda_mul,
    .scale = cuda_scale,
    .dot = cuda_dot,
    .transpose = cuda_transpose,
    .sum = cuda_sum,
    .unary = cuda_unary,
    .unary_grad = cuda_unary_grad,
    .sum_rows = cuda_sum_rows,
    .accum_scaled = cuda_accum_scaled,
    .add_bias = cuda_add_bias,
};
