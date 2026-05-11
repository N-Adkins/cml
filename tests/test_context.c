#include <cml/context.h>
#include "context.h"
#include "unity.h"

void setUp(void)
{
}

void tearDown(void)
{
}

static void test_init_creates_valid_context(void) {
    cml_context_t *ctx = cml_init(1024);

    TEST_ASSERT_NOT_NULL(ctx);
    TEST_ASSERT_NOT_NULL(ctx->arena.buffer);
    TEST_ASSERT_EQUAL(1024, ctx->arena.capacity);
    TEST_ASSERT_EQUAL(0, ctx->arena.offset);
    TEST_ASSERT_EQUAL(CML_OK, ctx->status);

    cml_deinit(ctx);
}

int main(void)
{
    UNITY_BEGIN();

    RUN_TEST(test_init_creates_valid_context);

    return UNITY_END();
}
