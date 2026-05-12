#ifndef CML_INTERNAL_BACKEND_H
#define CML_INTERNAL_BACKEND_H

#include <cml/context.h>
#include <cml/tensor.h>

#include <stdbool.h>
#include <stddef.h>

// This is an abstraction over all of the big math operations I might need, as well as
// allocation. This is what allows cuda + cpu to use the same code. It's basically just a huge
// vtable.

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CML_UNARY_LOG = 0,
    CML_UNARY_SIGMOID,
    CML_UNARY_RELU,
} cml_unary_op_t;

typedef struct cml_backend_ops_s {
    bool uses_device;
    cml_status_t (*init)(void **state);
    void (*deinit)(void *state);
    cml_status_t (*alloc_device)(void *state, size_t elements, float **ptr);
    void (*free_device)(void *state, float *ptr);
    cml_status_t (*copy_host_to_device)(void *state, float *dst, const float *src, size_t elements);
    cml_status_t (*copy_device_to_host)(void *state, float *dst, const float *src, size_t elements);
    cml_status_t (*copy)(void *state,
                         float *dst, size_t dst_stride,
                         const float *src, size_t src_stride,
                         size_t rows, size_t cols);
    cml_status_t (*add_scaled)(void *state,
                               float *out, size_t out_stride,
                               const float *a, size_t a_stride,
                               const float *b, size_t b_stride,
                               size_t rows, size_t cols, float alpha);
    cml_status_t (*mul)(void *state,
                        float *out, size_t out_stride,
                        const float *a, size_t a_stride,
                        const float *b, size_t b_stride,
                        size_t rows, size_t cols);
    cml_status_t (*scale)(void *state,
                          float *out, size_t out_stride,
                          const float *in, size_t in_stride,
                          size_t rows, size_t cols, float scalar);
    cml_status_t (*dot)(void *state,
                        float *out, size_t out_stride,
                        const float *a, size_t a_stride,
                        const float *b, size_t b_stride,
                        size_t m, size_t n, size_t k);
    cml_status_t (*transpose)(void *state,
                              float *out, size_t out_stride,
                              const float *in, size_t in_stride,
                              size_t rows, size_t cols);
    cml_status_t (*sum)(void *state, const float *in, size_t in_stride, size_t rows, size_t cols, float *out);
    cml_status_t (*unary)(void *state,
                          float *out, size_t out_stride,
                          const float *in, size_t in_stride,
                          size_t rows, size_t cols, cml_unary_op_t op);
    cml_status_t (*unary_grad)(void *state,
                               float *out, size_t out_stride,
                               const float *x, size_t x_stride,
                               const float *grad, size_t grad_stride,
                               size_t rows, size_t cols, cml_unary_op_t op);
    cml_status_t (*sum_rows)(void *state,
                             float *out,
                             const float *in, size_t in_stride,
                             size_t rows, size_t cols);
    cml_status_t (*accum_scaled)(void *state,
                                 float *dst, size_t dst_stride,
                                 const float *src, size_t src_stride,
                                 size_t rows, size_t cols, float scale);
    cml_status_t (*add_bias)(void *state,
                             float *out, size_t out_stride,
                             const float *a, size_t a_stride,
                             const float *b, size_t b_stride,
                             size_t rows, size_t cols);
    cml_status_t (*adam_step)(void *state,
                              float *p, size_t p_stride,
                              float *m, size_t m_stride,
                              float *v, size_t v_stride,
                              const float *g, size_t g_stride,
                              size_t rows, size_t cols,
                              float lr, float beta1, float beta2, float eps,
                              float bc1, float bc2);
} cml_backend_ops_t;

extern const cml_backend_ops_t cml_cpu_backend_ops;
#ifdef CML_ENABLE_CUDA
extern const cml_backend_ops_t cml_cuda_backend_ops;
#endif

bool cml_backend_cuda_compiled(void);
cml_status_t cml_backend_init(cml_context_t *ctx, cml_backend_t backend);
void cml_backend_deinit(cml_context_t *ctx);
void *cml_backend_device_mark(cml_context_t *ctx);
void cml_backend_device_rewind(cml_context_t *ctx, void *mark);

cml_status_t cml_backend_tensor_to_host(cml_context_t *ctx, const cml_tensor_t *tensor);
cml_status_t cml_backend_tensor_to_device(cml_context_t *ctx, const cml_tensor_t *tensor);
void cml_backend_mark_host_write(cml_tensor_t *tensor);
bool cml_backend_tensor_has_device_copy(const cml_tensor_t *tensor);

cml_status_t cml_backend_copy(cml_context_t *ctx, cml_tensor_t *dst, const cml_tensor_t *src);
cml_status_t cml_backend_add_scaled(cml_context_t *ctx, cml_tensor_t *out,
                                    const cml_tensor_t *a, const cml_tensor_t *b, float alpha);
cml_status_t cml_backend_mul(cml_context_t *ctx, cml_tensor_t *out,
                             const cml_tensor_t *a, const cml_tensor_t *b);
cml_status_t cml_backend_scale(cml_context_t *ctx, cml_tensor_t *out,
                               const cml_tensor_t *in, float scalar);
cml_status_t cml_backend_dot(cml_context_t *ctx, cml_tensor_t *out,
                             const cml_tensor_t *a, const cml_tensor_t *b);
cml_status_t cml_backend_transpose(cml_context_t *ctx, cml_tensor_t *out, const cml_tensor_t *in);
cml_status_t cml_backend_sum(cml_context_t *ctx, const cml_tensor_t *in, float *out);
cml_status_t cml_backend_unary(cml_context_t *ctx, cml_tensor_t *out,
                               const cml_tensor_t *in, cml_unary_op_t op);
cml_status_t cml_backend_unary_grad(cml_context_t *ctx, cml_tensor_t *out,
                                    const cml_tensor_t *x, const cml_tensor_t *grad,
                                    cml_unary_op_t op);
cml_status_t cml_backend_sum_rows(cml_context_t *ctx, cml_tensor_t *out,
                                  const cml_tensor_t *in);
cml_status_t cml_backend_accum_scaled(cml_context_t *ctx, cml_tensor_t *dst,
                                      const cml_tensor_t *src, float scale);
cml_status_t cml_backend_add_bias(cml_context_t *ctx, cml_tensor_t *out,
                                  const cml_tensor_t *a, const cml_tensor_t *b);
cml_status_t cml_backend_adam_step(cml_context_t *ctx,
                                   cml_tensor_t *p, cml_tensor_t *m, cml_tensor_t *v,
                                   const cml_tensor_t *g,
                                   float lr, float beta1, float beta2, float eps,
                                   float bc1, float bc2);

#ifdef __cplusplus
}
#endif

#endif
