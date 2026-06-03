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

// Maps a seed plus caller-chosen salt to a deterministic signed offset in [-range, range].
// Different salts let one seed produce independent offsets for unrelated domains, e.g. X and Z axes.
BLOCKLAB_HOST_DEVICE constexpr int32_t seedOffset(uint32_t seed, uint32_t salt, uint32_t range = 4096U)
{
    return static_cast<int32_t>(hashCombine(seed, salt) % (range * 2U + 1U)) - static_cast<int32_t>(range);
}

} // namespace blocklab
