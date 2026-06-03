#pragma once

#include <cstdint>

namespace blocklab {

void convertRgba8ToFloatNchw(
    const void* rgba8, float* nchw, uint32_t batchSize, uint32_t width, uint32_t height, uintptr_t streamHandle);

} // namespace blocklab
