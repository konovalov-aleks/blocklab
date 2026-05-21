#pragma once

#include "blocklab/BlockTypes.h"
#include "blocklab/Math.h"

#include <cstdint>

namespace blocklab {

enum class MeshMaterial : uint32_t {
    GrassTop = 0,
    Dirt = 1,
    Stone = 2,
    GrassSide = 3,
    PigSkin = 4,
    PigSnout = 5,
    VertexColor = 8,
};

constexpr float meshMaterialId(MeshMaterial material)
{
    return static_cast<float>(static_cast<uint32_t>(material));
}

struct MeshVertex {
    Vec4 position;
    Vec4 colorAndShade;
    Vec4 uvMaterial;
};

struct MeshBuildConfig {
    int32_t radius = 32;
};

struct TerrainBlockOverride {
    int32_t x = 0;
    int32_t y = 0;
    int32_t z = 0;
    uint8_t block = BlockId::Air;
};

} // namespace blocklab
