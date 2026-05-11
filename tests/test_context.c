#include <cml/context.h>
#include "context.h"
#include "memory/arena.h"
#include "unity.h"

static cml_context_t *ctx;

void setUp(void) {
    ctx = NULL;
}

void tearDown(void) {
    cml_deinit(ctx);
}

/* --- cml_init --- */

static void test_init_creates_valid_context(void) {
    ctx = cml_init(1024);
    TEST_ASSERT_NOT_NULL(ctx);
    TEST_ASSERT_NOT_NULL(ctx->arena.buffer);
    TEST_ASSERT_EQUAL_size_t(1024, ctx->arena.capacity);
    TEST_ASSERT_EQUAL_size_t(0, ctx->arena.offset);
    TEST_ASSERT_EQUAL(CML_OK, ctx->status);
    TEST_ASSERT_NULL(ctx->error_msg);
}

static void test_init_status_ok(void) {
    ctx = cml_init(256);
    TEST_ASSERT_EQUAL(CML_OK, cml_get_status(ctx));
}

static void test_init_no_error_msg(void) {
    ctx = cml_init(256);
    TEST_ASSERT_NULL(cml_get_error_msg(ctx));
}

static void test_init_different_sizes(void) {
    ctx = cml_init(64);
    TEST_ASSERT_NOT_NULL(ctx);
    TEST_ASSERT_EQUAL_size_t(64, ctx->arena.capacity);
    cml_deinit(ctx);

    ctx = cml_init(1024 * 1024);
    TEST_ASSERT_NOT_NULL(ctx);
    TEST_ASSERT_EQUAL_size_t(1024 * 1024, ctx->arena.capacity);
}

/* --- cml_deinit --- */

static void test_deinit_null_is_safe(void) {
    cml_deinit(NULL); /* must not crash */
}

/* --- cml_get_status / cml_get_error_msg --- */

static void test_get_status_returns_ok_initially(void) {
    ctx = cml_init(256);
    TEST_ASSERT_EQUAL(CML_OK, cml_get_status(ctx));
}

static void test_get_error_msg_null_when_ok(void) {
    ctx = cml_init(256);
    TEST_ASSERT_NULL(cml_get_error_msg(ctx));
}

/* --- cml_context_error --- */

static void test_error_sets_status(void) {
    ctx = cml_init(256);
    cml_context_error(ctx, CML_INVALID_ARG, "bad argument");
    TEST_ASSERT_EQUAL(CML_INVALID_ARG, cml_get_status(ctx));
}

static void test_error_sets_message(void) {
    ctx = cml_init(256);
    cml_context_error(ctx, CML_INVALID_ARG, "bad argument");
    TEST_ASSERT_EQUAL_STRING("bad argument", cml_get_error_msg(ctx));
}

static void test_error_first_error_wins(void) {
    ctx = cml_init(256);
    cml_context_error(ctx, CML_INVALID_ARG, "first");
    cml_context_error(ctx, CML_OUT_OF_MEMORY, "second");
    TEST_ASSERT_EQUAL(CML_INVALID_ARG, cml_get_status(ctx));
    TEST_ASSERT_EQUAL_STRING("first", cml_get_error_msg(ctx));
}

static void test_error_out_of_memory_status(void) {
    ctx = cml_init(256);
    cml_context_error(ctx, CML_OUT_OF_MEMORY, "oom");
    TEST_ASSERT_EQUAL(CML_OUT_OF_MEMORY, cml_get_status(ctx));
    TEST_ASSERT_EQUAL_STRING("oom", cml_get_error_msg(ctx));
}

/* --- arena integration --- */

static void test_arena_alloc_advances_offset(void) {
    ctx = cml_init(256);
    size_t before = ctx->arena.offset;
    void *p = cml_arena_alloc(&ctx->arena, 16);
    TEST_ASSERT_NOT_NULL(p);
    TEST_ASSERT_GREATER_THAN(before, ctx->arena.offset);
}

static void test_arena_alloc_is_8byte_aligned(void) {
    ctx = cml_init(256);
    void *p = cml_arena_alloc(&ctx->arena, 1);
    TEST_ASSERT_NOT_NULL(p);
    /* offset bumped to next 8-byte boundary */
    TEST_ASSERT_EQUAL_size_t(0, ctx->arena.offset % 8);
}

static void test_arena_alloc_returns_null_when_full(void) {
    ctx = cml_init(32);
    /* exhaust the arena */
    cml_arena_alloc(&ctx->arena, 32);
    void *p = cml_arena_alloc(&ctx->arena, 1);
    TEST_ASSERT_NULL(p);
}

static void test_arena_alloc_sequential_pointers_are_ordered(void) {
    ctx = cml_init(256);
    void *a = cml_arena_alloc(&ctx->arena, 8);
    void *b = cml_arena_alloc(&ctx->arena, 8);
    TEST_ASSERT_NOT_NULL(a);
    TEST_ASSERT_NOT_NULL(b);
    TEST_ASSERT_GREATER_THAN((uintptr_t)a, (uintptr_t)b);
}

static void test_arena_reset_resets_offset(void) {
    ctx = cml_init(256);
    cml_arena_alloc(&ctx->arena, 64);
    TEST_ASSERT_GREATER_THAN(0, ctx->arena.offset);
    cml_arena_reset(&ctx->arena);
    TEST_ASSERT_EQUAL_size_t(0, ctx->arena.offset);
}

static void test_arena_reset_allows_realloc(void) {
    ctx = cml_init(64);
    void *first = cml_arena_alloc(&ctx->arena, 64);
    TEST_ASSERT_NOT_NULL(first);
    cml_arena_reset(&ctx->arena);
    void *second = cml_arena_alloc(&ctx->arena, 64);
    TEST_ASSERT_NOT_NULL(second);
    TEST_ASSERT_EQUAL_PTR(first, second);
}

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_init_creates_valid_context);
    RUN_TEST(test_init_status_ok);
    RUN_TEST(test_init_no_error_msg);
    RUN_TEST(test_init_different_sizes);

    RUN_TEST(test_deinit_null_is_safe);

    RUN_TEST(test_get_status_returns_ok_initially);
    RUN_TEST(test_get_error_msg_null_when_ok);

    RUN_TEST(test_error_sets_status);
    RUN_TEST(test_error_sets_message);
    RUN_TEST(test_error_first_error_wins);
    RUN_TEST(test_error_out_of_memory_status);

    RUN_TEST(test_arena_alloc_advances_offset);
    RUN_TEST(test_arena_alloc_is_8byte_aligned);
    RUN_TEST(test_arena_alloc_returns_null_when_full);
    RUN_TEST(test_arena_alloc_sequential_pointers_are_ordered);
    RUN_TEST(test_arena_reset_resets_offset);
    RUN_TEST(test_arena_reset_allows_realloc);

    return UNITY_END();
}
