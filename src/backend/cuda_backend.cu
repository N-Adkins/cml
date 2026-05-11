#include "backend.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#define CML_CUDA_SUM_BLOCK 256
#define CML_CUDA_SUM_MAX_BLOCKS 1024

typedef struct {
    cublasHandle_t cublas;
    float *scalar_buf;
    float *sum_partials; // device-side scratch for deterministic sum (CML_CUDA_SUM_MAX_BLOCKS floats)
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
    if (status != cudaSuccess) return map_cuda_status(status);
    // Force runtime errors (out-of-bounds, illegal memory access) to surface here
    // rather than at the next API call.
    status = cudaDeviceSynchronize();
    return map_cuda_status(status);
}

static int sum_grid_for(size_t total) {
    size_t blocks = (total + (size_t)CML_CUDA_SUM_BLOCK - 1) / (size_t)CML_CUDA_SUM_BLOCK;
    if (blocks > (size_t)CML_CUDA_SUM_MAX_BLOCKS) blocks = CML_CUDA_SUM_MAX_BLOCKS;
    if (blocks == 0) blocks = 1;
    return (int)blocks;
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
        // Floor at FLT_MIN to match the CPU backend and avoid -inf propagating
        // into gradients.
        outv = logf(v > FLT_MIN ? v : FLT_MIN);
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

// Block-level reduction with grid-stride loop. Each block produces one partial
// sum in `partials[blockIdx.x]`. The host then performs a small final reduction.
// The mapping between elements and blocks is fixed by index, which makes the
// per-run result bit-exact deterministic (unlike a single shared atomic).
static __global__ void kernel_sum_block(const float *in, size_t in_stride,
                                        size_t rows, size_t cols, float *partials) {
    __shared__ float sdata[CML_CUDA_SUM_BLOCK];
    size_t tid = (size_t)threadIdx.x;
    size_t total = rows * cols;
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + tid;
    size_t step = (size_t)gridDim.x * (size_t)blockDim.x;

    float local = 0.0f;
    while (idx < total) {
        size_t r = idx / cols;
        size_t c = idx % cols;
        local += in[r * in_stride + c];
        idx += step;
    }
    sdata[tid] = local;
    __syncthreads();

    for (size_t s = (size_t)blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) partials[blockIdx.x] = sdata[0];
}

static cml_status_t cuda_init(void **state) {
    cml_cuda_state_t *cuda_state = (cml_cuda_state_t *)malloc(sizeof(cml_cuda_state_t));
    if (cuda_state == NULL) return CML_OUT_OF_MEMORY;
    cuda_state->cublas = NULL;
    cuda_state->scalar_buf = NULL;
    cuda_state->sum_partials = NULL;

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

    cuda_status = cudaMalloc((void **)&cuda_state->sum_partials,
                              CML_CUDA_SUM_MAX_BLOCKS * sizeof(float));
    if (cuda_status != cudaSuccess) {
        cudaFree(cuda_state->scalar_buf);
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
    if (cuda_state->sum_partials != NULL) cudaFree(cuda_state->sum_partials);
    if (cuda_state->scalar_buf != NULL) cudaFree(cuda_state->scalar_buf);
    if (cuda_state->cublas != NULL) cublasDestroy(cuda_state->cublas);
    free(cuda_state);
}

static cml_status_t cuda_alloc_device(void *state, size_t elements, float **ptr) {
    (void)state;
    if (elements == 0) {
        *ptr = NULL;
        return CML_OK;
    }
    if (elements > SIZE_MAX / sizeof(float)) return CML_INVALID_ARG;
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
    if (elements == 0) return CML_OK;
    cudaError_t status = cudaMemcpy(dst, src, elements * sizeof(float), cudaMemcpyHostToDevice);
    return map_cuda_status(status);
}

static cml_status_t cuda_copy_device_to_host(void *state, float *dst, const float *src, size_t elements) {
    (void)state;
    if (elements == 0) return CML_OK;
    cudaError_t status = cudaMemcpy(dst, src, elements * sizeof(float), cudaMemcpyDeviceToHost);
    return map_cuda_status(status);
}

static cml_status_t cuda_copy(void *state,
                              float *dst, size_t dst_stride,
                              const float *src, size_t src_stride,
                              size_t rows, size_t cols) {
    (void)state;
    size_t total = rows * cols;
    if (total == 0) return CML_OK;
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
    if (total == 0) return CML_OK;
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
    if (total == 0) return CML_OK;
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
    if (total == 0) return CML_OK;
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
    if (m == 0 || n == 0 || k == 0) return CML_OK;
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
    if (total == 0) return CML_OK;
    int block = 256;
    int grid = (int)((total + (size_t)block - 1U) / (size_t)block);
    kernel_transpose<<<grid, block>>>(out, out_stride, in, in_stride, rows, cols);
    return finish_kernel();
}

static cml_status_t cuda_sum(void *state,
                             const float *in, size_t in_stride,
                             size_t rows, size_t cols, float *out) {
    cml_cuda_state_t *cuda_state = (cml_cuda_state_t *)state;
    size_t total = rows * cols;
    if (total == 0) {
        *out = 0.0f;
        return CML_OK;
    }

    int grid = sum_grid_for(total);
    kernel_sum_block<<<grid, CML_CUDA_SUM_BLOCK>>>(
        in, in_stride, rows, cols, cuda_state->sum_partials);
    cml_status_t status = finish_kernel();
    if (status != CML_OK) return status;

    float partials[CML_CUDA_SUM_MAX_BLOCKS];
    cudaError_t cuda_status = cudaMemcpy(partials, cuda_state->sum_partials,
                                         (size_t)grid * sizeof(float),
                                         cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) return map_cuda_status(cuda_status);

    // Final reduce on host with Kahan compensation for cross-block precision.
    float total_sum = 0.0f;
    float comp = 0.0f;
    for (int i = 0; i < grid; i++) {
        float y = partials[i] - comp;
        float t = total_sum + y;
        comp = (t - total_sum) - y;
        total_sum = t;
    }
    *out = total_sum;
    return CML_OK;
}

static cml_status_t cuda_unary(void *state,
                               float *out, size_t out_stride,
                               const float *in, size_t in_stride,
                               size_t rows, size_t cols, cml_unary_op_t op) {
    (void)state;
    size_t total = rows * cols;
    if (total == 0) return CML_OK;
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
    if (total == 0) return CML_OK;
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
    if (total == 0) return CML_OK;
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
    if (total == 0) return CML_OK;
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
    if (cols == 0) return CML_OK;
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
