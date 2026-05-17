#pragma once

#include <cstdint>

namespace blocklab {

enum class ObservationDevice {
    None,
    Cpu,
    SdlGpuTexture,
    Cuda,
};

enum class ObservationFormat {
    RGBA8,
};

struct Observation {
    int32_t width = 0;
    int32_t height = 0;
    int32_t channels = 4;
    ObservationDevice device = ObservationDevice::None;
    ObservationFormat format = ObservationFormat::RGBA8;
    uintptr_t handle = 0;
    uint64_t version = 0;
};

} // namespace blocklab
