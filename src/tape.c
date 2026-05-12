#include "tape.h"
#include "backend/backend.h"
#include "context.h"
#include "tensor.h"

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
    if (ctx->status != CML_OK) return;
    if (!t->requires_grad || delta == NULL) return;
    if (t->grad == NULL) {
        t->grad = cml_tensor_init(ctx, t->rows, t->cols);
        if (t->grad == NULL) return;
        cml_tensor_zero(t->grad);
    }
    cml_backend_accum_scaled(ctx, t->grad, delta, scale);
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
        if (ctx->status != CML_OK) return;
    }
    if (b->requires_grad) {
        grad_accum_scaled(ctx, b, cml_tensor_mul(ctx, g, a), 1.0f);
    }
}

static void backward_scale(cml_context_t *ctx, struct cml_tape_node_s *node) {
    grad_accum_scaled(ctx, node->inputs[0], node->output->grad, *(float *)node->aux);
}

static void backward_add_bias(cml_context_t *ctx, struct cml_tape_node_s *node) {
    cml_tensor_t *a = node->inputs[0]; /* [M x N] */
    cml_tensor_t *b = node->inputs[1]; /* [1 x N] */
    cml_tensor_t *g = node->output->grad;

    grad_accum_scaled(ctx, a, g, 1.0f);
    if (ctx->status != CML_OK) return;

    if (b->requires_grad) {
        cml_tensor_t *gb = cml_tensor_init(ctx, 1, b->cols);
        if (gb == NULL) return;
        if (cml_backend_sum_rows(ctx, gb, g) != CML_OK) return;
        grad_accum_scaled(ctx, b, gb, 1.0f);
    }
}

static void backward_unary(cml_context_t *ctx, struct cml_tape_node_s *node) {
    cml_unary_op_t op = *(cml_unary_op_t *)node->aux;
    cml_tensor_t *x = node->inputs[0];
    cml_tensor_t *g = node->output->grad;
    cml_tensor_t *dg = cml_tensor_init(ctx, x->rows, x->cols);
    if (dg == NULL) return;
    if (cml_backend_unary_grad(ctx, dg, x, g, op) != CML_OK) return;
    grad_accum_scaled(ctx, x, dg, 1.0f);
}

static void backward_dot(cml_context_t *ctx, struct cml_tape_node_s *node) {
    cml_tensor_t *a = node->inputs[0];
    cml_tensor_t *b = node->inputs[1];
    cml_tensor_t *g = node->output->grad;
    if (a->requires_grad) {
        cml_tensor_t *bt = cml_tensor_transpose(ctx, b);
        if (bt == NULL) return;
        grad_accum_scaled(ctx, a, cml_tensor_dot(ctx, g, bt), 1.0f);
        if (ctx->status != CML_OK) return;
    }
    if (b->requires_grad) {
        cml_tensor_t *at = cml_tensor_transpose(ctx, a);
        if (at == NULL) return;
        grad_accum_scaled(ctx, b, cml_tensor_dot(ctx, at, g), 1.0f);
    }
}

static void backward_transpose(cml_context_t *ctx, struct cml_tape_node_s *node) {
    grad_accum_scaled(ctx, node->inputs[0],
                      cml_tensor_transpose(ctx, node->output->grad), 1.0f);
}


static void backward_sum(cml_context_t *ctx, struct cml_tape_node_s *node) {
    cml_tensor_t *x = node->inputs[0];
    float g = cml_tensor_get(node->output->grad, 0, 0);

    cml_tensor_t *expanded = cml_tensor_init(ctx, x->rows, x->cols);
    if (expanded == NULL) return;

    cml_tensor_fill(expanded, g);
    grad_accum_scaled(ctx, x, expanded, 1.0f);
}

static void backward_softmax_xent(cml_context_t *ctx, struct cml_tape_node_s *node) {
    cml_tensor_t *logits = node->inputs[0];
    cml_tensor_t *target = node->inputs[1];
    cml_tensor_t *softmax_buf = (cml_tensor_t *)node->aux;
    if (softmax_buf == NULL) return;

    float g = cml_tensor_get(node->output->grad, 0, 0);
    float scale = g / (float)logits->rows;

    if (logits->requires_grad) {
        cml_tensor_t *dlogits = cml_tensor_init(ctx, logits->rows, logits->cols);
        if (dlogits == NULL) return;
        if (cml_backend_softmax_xent_backward(ctx, dlogits, softmax_buf, target, scale) != CML_OK) return;
        grad_accum_scaled(ctx, logits, dlogits, 1.0f);
    }
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

void cml_tape_record_add_bias(cml_context_t *ctx, cml_tensor_t *out,
                               cml_tensor_t *a, cml_tensor_t *b) {
    if (!needs_grad(ctx, a, b)) return;
    cml_tensor_t *inputs[] = {a, b};
    tape_push(ctx, out, inputs, 2, backward_add_bias, NULL);
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

static void record_unary(cml_context_t *ctx, cml_tensor_t *out,
                         cml_tensor_t *tensor, cml_unary_op_t op) {
    if (!needs_grad(ctx, tensor, NULL)) return;
    cml_unary_op_t *aux = cml_arena_alloc(&ctx->arena, sizeof(cml_unary_op_t));
    if (aux == NULL) {
        cml_context_error(ctx, CML_OUT_OF_MEMORY, "tape aux allocation failed");
        return;
    }
    *aux = op;
    cml_tensor_t *inputs[] = {tensor};
    tape_push(ctx, out, inputs, 1, backward_unary, aux);
}

void cml_tape_record_log(cml_context_t *ctx, cml_tensor_t *out,
                         cml_tensor_t *tensor) {
    record_unary(ctx, out, tensor, CML_UNARY_LOG);
}

void cml_tape_record_sigmoid(cml_context_t *ctx, cml_tensor_t *out,
                              cml_tensor_t *tensor) {
    record_unary(ctx, out, tensor, CML_UNARY_SIGMOID);
}

void cml_tape_record_relu(cml_context_t *ctx, cml_tensor_t *out,
                           cml_tensor_t *tensor) {
    record_unary(ctx, out, tensor, CML_UNARY_RELU);
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

void cml_tape_record_softmax_xent(cml_context_t *ctx, cml_tensor_t *loss,
                                   cml_tensor_t *logits, cml_tensor_t *target,
                                   cml_tensor_t *softmax_buf) {
    if (!needs_grad(ctx, logits, target)) return;

    cml_tensor_t *inputs[] = {logits, target};
    tape_push(ctx, loss, inputs, 2, backward_softmax_xent, softmax_buf);
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
