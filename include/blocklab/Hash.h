#pragma once

#include "CudaHelpers.h"

#include <concepts>
#include <cstdint>

namespace blocklab {

template <std::integral... T>
BLOCKLAB_HOST_DEVICE constexpr uint32_t hashCombine(uint32_t h1, uint32_t h2, T... other) noexcept
{
    if constexpr (sizeof...(other) == 0) {
        h1 ^= h2 + 0x9e3779b9u + (h1 << 6) + (h1 >> 2);
        return h1;
    } else
        return hashCombine(h1, hashCombine(h2, other...));
}

BLOCKLAB_HOST_DEVICE constexpr uint32_t hash(uint32_t value)
{
    value ^= value >> 16U;
    value *= 0x7feb352dU;
    value ^= value >> 15U;
    value *= 0x846ca68bU;
    value ^= value >> 16U;
    return value;
}

BLOCKLAB_HOST_DEVICE constexpr float randomFloat01(uint32_t seed)
{
    return static_cast<float>(hash(seed) & 0x00ffffffU) / static_cast<float>(0x00ffffffU);
}

} // namespace blocklab
