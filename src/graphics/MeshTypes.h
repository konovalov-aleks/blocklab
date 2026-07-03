#pragma once

#include <blocklab/utility/Math.h>
#include <world/Block.h>
#include <world/Voxel.h>

#include <cstdint>

namespace blocklab {

enum class Material : std::uint8_t {
    GrassTop = 0,
    Dirt = 1,
    Stone = 2,
    GrassSide = 3,
    PigSkin = 4,
    PigSnout = 5,
    TorchSide = 6,
    TorchTop = 7,
    VertexColor = 8,
};

struct MeshVertex {
    Vec4 position;
    Vec4 normal;
    Vec4 uvMaterial;
    Vec3 color;
    float padding;
};

} // namespace blocklab
