#pragma once

#include <blocklab/Math.h>

#include <cstddef>
#include <cstdint>

namespace blocklab {

enum VisibleFace : uint8_t {
    Left   = 1 << 0,
    Right  = 1 << 1,
    Top    = 1 << 2,
    Bottom = 1 << 3,
    Front  = 1 << 4,
    Back   = 1 << 5
};

struct Voxel;
static constexpr size_t VoxelSize = 16;

} // namespace blocklab
