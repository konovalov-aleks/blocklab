#pragma once

#include <blocklab/Block.h>
#include <blocklab/Math.h>
#include <blocklab/Voxel.h>

#include <cstdint>

namespace blocklab {

enum class Material : uint8_t {
    GrassTop = 0,
    Dirt = 1,
    Stone = 2,
    GrassSide = 3,
    PigSkin = 4,
    PigSnout = 5,
    VertexColor = 8,
};

struct MeshVertex {
    Vec4 position;
    Vec4 colorAndShade;
    Vec4 uvMaterial;
};

} // namespace blocklab
