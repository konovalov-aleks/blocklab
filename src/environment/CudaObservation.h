#pragma once

#include <cstdint>

namespace blocklab {

void convertRgba8ToFloatNchw(const void* rgba8, float* nchw, std::uint32_t batchSize, std::uint32_t width,
    std::uint32_t height, std::uintptr_t streamHandle);

} // namespace blocklab
