#include "tape.h"
#include "context.h"
#include "tensor.h"

#include <cblas.h>

struct cml_tape_node_s {
    cml_tensor_t *output; // tensor produced by this op; owns the grad allocation
    cml_tensor_t **inputs; // operands captured at record time (arena-allocated array)
    size_t n_inputs;
    void (*backward)(cml_context_t *, struct cml_tape_node_s *); // op-specific grad fn
    void *aux; // optional op data (e.g. scalar for scale); arena-allocated
    struct cml_tape_node_s *prev; // intrusive singly-linked list; head = most recent op
};

static void grad_accum_scaled(cml_context_t *ctx, cml_tensor_t *t,
                               cml_tensor_t *delta, float scale) {
    if (!t->requires_grad || delta == NULL) return;
    if (t->grad == NULL) {
        t->grad = cml_tensor_init(ctx, t->rows, t->cols);
        if (t->grad == NULL) return;
        cml_tensor_zero(t->grad);
    }
    for (size_t r = 0; r < delta->rows; r++) {
        cblas_saxpy((int)delta->cols, scale,
                    delta->data + r * delta->stride, 1,
                    t->grad->data + r * t->grad->stride, 1);
    }
}

static void backward_add(cml_context_t *ctx, struct cml_tape_node_s *node) {
    grad_accum_scaled(ctx, node->inputs[0], node->output->grad,  1.0f);
    grad_accum_scaled(ctx, node->inputs[1], node->output->grad,  1.0f);
}

static void backward_sub(cml_context_t *ctx, struct cml_tape_node_s *node) {
    grad_accum_scaled(ctx, node->inputs[0], node->output->grad,  1.0f);
    grad_accum_scaled(ctx, node->inputs[1], node->output->grad, -1.0f);
}

static void backward_mul(cml_context_t *ctx, struct cml_tape_node_s *node) {
    cml_tensor_t *a = node->inputs[0];
    cml_tensor_t *b = node->inputs[1];
    cml_tensor_t *g = node->output->grad;
    if (a->requires_grad) {
        grad_accum_scaled(ctx, a, cml_tensor_mul(ctx, g, b), 1.0f);
    }
    if (b->requires_grad) {
        grad_accum_scaled(ctx, b, cml_tensor_mul(ctx, g, a), 1.0f);
    }
}

static void backward_scale(cml_context_t *ctx, struct cml_tape_node_s *node) {
    grad_accum_scaled(ctx, node->inputs[0], node->output->grad, *(float *)node->aux);
}

static void backward_dot(cml_context_t *ctx, struct cml_tape_node_s *node) {
    cml_tensor_t *a = node->inputs[0];
    cml_tensor_t *b = node->inputs[1];
    cml_tensor_t *g = node->output->grad;
    if (a->requires_grad) {
        cml_tensor_t *bt = cml_tensor_transpose(ctx, b);
        if (bt != NULL) {
            grad_accum_scaled(ctx, a, cml_tensor_dot(ctx, g, bt), 1.0f);
        }
    }
    if (b->requires_grad) {
        cml_tensor_t *at = cml_tensor_transpose(ctx, a);
        if (at != NULL) {
            grad_accum_scaled(ctx, b, cml_tensor_dot(ctx, at, g), 1.0f);
        }
    }
}

static void backward_transpose(cml_context_t *ctx, struct cml_tape_node_s *node) {
    grad_accum_scaled(ctx, node->inputs[0],
                      cml_tensor_transpose(ctx, node->output->grad), 1.0f);
}

static void backward_sigmoid(cml_context_t *ctx, struct cml_tape_node_s *node) {
    cml_tensor_t *out = node->output;

    cml_tensor_t *ones = cml_tensor_init(ctx, out->rows, out->cols);
    if (ones == NULL) return;
    cml_tensor_fill(ones, 1.0f);

    cml_tensor_t *one_minus = cml_tensor_sub(ctx, ones, out);
    if (one_minus == NULL) return;

    cml_tensor_t *d = cml_tensor_mul(ctx, out, one_minus);
    if (d == NULL) return;

    grad_accum_scaled(ctx, node->inputs[0], cml_tensor_mul(ctx, out->grad, d), 1.0f);
}

static void backward_relu(cml_context_t *ctx, struct cml_tape_node_s *node) {
    cml_tensor_t *x = node->inputs[0];
    cml_tensor_t *mask = cml_tensor_init(ctx, x->rows, x->cols);
    if (mask == NULL) return;

    for (size_t r = 0; r < x->rows; r++) {
        for (size_t c = 0; c < x->cols; c++) {
            cml_tensor_set(mask, r, c,
                           cml_tensor_get(x, r, c) > 0.0f ? 1.0f : 0.0f);
        }
    }

    grad_accum_scaled(ctx, x, cml_tensor_mul(ctx, node->output->grad, mask), 1.0f);
}

static void backward_sum(cml_context_t *ctx, struct cml_tape_node_s *node) {
    cml_tensor_t *x = node->inputs[0];
    float g = cml_tensor_get(node->output->grad, 0, 0);

    cml_tensor_t *expanded = cml_tensor_init(ctx, x->rows, x->cols);
    if (expanded == NULL) return;

    cml_tensor_fill(expanded, g);
    grad_accum_scaled(ctx, x, expanded, 1.0f);
}

static struct cml_tape_node_s *tape_push(cml_context_t *ctx, cml_tensor_t *out,
                                          cml_tensor_t **inputs, size_t n,
                                          void (*bwd)(cml_context_t *, struct cml_tape_node_s *),
                                          void *aux) {
    struct cml_tape_node_s *node = cml_arena_alloc(&ctx->arena,
                                                    sizeof(struct cml_tape_node_s));
    if (node == NULL) {
        cml_context_error(ctx, CML_OUT_OF_MEMORY, "tape node allocation failed");
        return NULL;
    }

    cml_tensor_t **inp = cml_arena_alloc(&ctx->arena, n * sizeof(cml_tensor_t *));
    if (inp == NULL) {
        cml_context_error(ctx, CML_OUT_OF_MEMORY, "tape inputs allocation failed");
        return NULL;
    }
    for (size_t i = 0; i < n; i++) {
        inp[i] = inputs[i];
    }

    node->output = out;
    node->inputs = inp;
    node->n_inputs = n;
    node->backward = bwd;
    node->aux = aux;
    node->prev = ctx->tape_head;
    ctx->tape_head = node;

    out->requires_grad = true;
    out->creator = node;
    return node;
}

static bool needs_grad(cml_context_t *ctx, cml_tensor_t *a, cml_tensor_t *b) {
    return ctx->grad_enabled &&
           (a->requires_grad || (b != NULL && b->requires_grad));
}

void cml_tape_record_add(cml_context_t *ctx, cml_tensor_t *out,
                          cml_tensor_t *a, cml_tensor_t *b) {
    if (!needs_grad(ctx, a, b)) return;

    cml_tensor_t *inputs[] = {a, b};
    tape_push(ctx, out, inputs, 2, backward_add, NULL);
}

void cml_tape_record_sub(cml_context_t *ctx, cml_tensor_t *out,
                          cml_tensor_t *a, cml_tensor_t *b) {
    if (!needs_grad(ctx, a, b)) return;

    cml_tensor_t *inputs[] = {a, b};
    tape_push(ctx, out, inputs, 2, backward_sub, NULL);
}

void cml_tape_record_mul(cml_context_t *ctx, cml_tensor_t *out,
                          cml_tensor_t *a, cml_tensor_t *b) {
    if (!needs_grad(ctx, a, b)) return;

    cml_tensor_t *inputs[] = {a, b};
    tape_push(ctx, out, inputs, 2, backward_mul, NULL);
}

void cml_tape_record_dot(cml_context_t *ctx, cml_tensor_t *out,
                          cml_tensor_t *a, cml_tensor_t *b) {
    if (!needs_grad(ctx, a, b)) return;

    cml_tensor_t *inputs[] = {a, b};
    tape_push(ctx, out, inputs, 2, backward_dot, NULL);
}

void cml_tape_record_scale(cml_context_t *ctx, cml_tensor_t *out,
                            cml_tensor_t *tensor, float scalar) {
    if (!needs_grad(ctx, tensor, NULL)) return;

    float *aux = cml_arena_alloc(&ctx->arena, sizeof(float));
    if (aux == NULL) {
        cml_context_error(ctx, CML_OUT_OF_MEMORY, "tape aux allocation failed");
        return;
    }
    *aux = scalar;

    cml_tensor_t *inputs[] = {tensor};
    tape_push(ctx, out, inputs, 1, backward_scale, aux);
}

void cml_tape_record_sigmoid(cml_context_t *ctx, cml_tensor_t *out,
                              cml_tensor_t *tensor) {
    if (!needs_grad(ctx, tensor, NULL)) return;

    cml_tensor_t *inputs[] = {tensor};
    tape_push(ctx, out, inputs, 1, backward_sigmoid, NULL);
}

void cml_tape_record_relu(cml_context_t *ctx, cml_tensor_t *out,
                           cml_tensor_t *tensor) {
    if (!needs_grad(ctx, tensor, NULL)) return;

    cml_tensor_t *inputs[] = {tensor};
    tape_push(ctx, out, inputs, 1, backward_relu, NULL);
}

void cml_tape_record_transpose(cml_context_t *ctx, cml_tensor_t *out,
                                cml_tensor_t *tensor) {
    if (!needs_grad(ctx, tensor, NULL)) return;

    cml_tensor_t *inputs[] = {tensor};
    tape_push(ctx, out, inputs, 1, backward_transpose, NULL);
}

void cml_tape_record_sum(cml_context_t *ctx, cml_tensor_t *out,
                          cml_tensor_t *tensor) {
    if (!needs_grad(ctx, tensor, NULL)) return;

    cml_tensor_t *inputs[] = {tensor};
    tape_push(ctx, out, inputs, 1, backward_sum, NULL);
}

void cml_backward(cml_context_t *ctx, cml_tensor_t *loss) {
    if (ctx == NULL || loss == NULL) return;
    if (ctx->status != CML_OK) return;

    loss->grad = cml_tensor_init(ctx, loss->rows, loss->cols);
    if (loss->grad == NULL) return;
    cml_tensor_fill(loss->grad, 1.0f);

    bool was_enabled = ctx->grad_enabled;
    ctx->grad_enabled = false;

    for (struct cml_tape_node_s *n = ctx->tape_head;
         n != NULL && ctx->status == CML_OK;
         n = n->prev) {
        if (n->output->grad != NULL)
            n->backward(ctx, n);
    }

    ctx->grad_enabled = was_enabled;
}

void cml_tape_clear(cml_context_t *ctx) {
    if (ctx == NULL) return;

    for (struct cml_tape_node_s *n = ctx->tape_head; n != NULL; n = n->prev) {
        n->output->grad = NULL;
        n->output->creator = NULL;
        n->output->requires_grad = false;

        for (size_t i = 0; i < n->n_inputs; i++) {
            cml_tensor_t *inp = n->inputs[i];
            if (inp->creator == NULL)
                inp->grad = NULL;
        }
    }

    ctx->tape_head = NULL;
}

void cml_grad_enable(cml_context_t *ctx) {
    if (ctx != NULL) ctx->grad_enabled = true;
}

void cml_grad_disable(cml_context_t *ctx) {
    if (ctx != NULL) ctx->grad_enabled = false;
}
