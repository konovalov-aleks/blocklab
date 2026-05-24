#include "blocklab/Hash.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <type_traits>

static_assert(blocklab::hash(123U) == blocklab::hash(123U));
static_assert(std::is_same_v<decltype(blocklab::hashCombine(1U, 2U)), uint32_t>);

TEST_CASE("hashCombine is deterministic and order-sensitive", "[hash]")
{
    CHECK(blocklab::hashCombine(1U, 2U, 3U) == blocklab::hashCombine(1U, 2U, 3U));
    CHECK(blocklab::hashCombine(1U, 2U, 3U) != blocklab::hashCombine(1U, 3U, 2U));
    CHECK(blocklab::hashCombine(1U, blocklab::hashCombine(2U, 3U)) == blocklab::hashCombine(1U, 2U, 3U));
}

TEST_CASE("randomUInt supports single seed and bounded values", "[hash]")
{
    CHECK(blocklab::hash(42U) == blocklab::hash(42U));
    CHECK(blocklab::hash(42U) != 42U);
}

TEST_CASE("randomFloat01 returns normalized deterministic values", "[hash]")
{
    const float value = blocklab::randomFloat01(blocklab::hashCombine(42U, 7U));
    CHECK(value >= 0.0f);
    CHECK(value <= 1.0f);
    CHECK(value == blocklab::randomFloat01(blocklab::hashCombine(42U, 7U)));
}
