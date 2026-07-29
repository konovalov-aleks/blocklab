#include "Drop.h"

#include <graphics/Mesh.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cmath>

namespace blocklab {

static constexpr float s_size = 0.25f;
static const float s_diagonalSize = s_size * std::sqrt(2.0f);

HitCylinder Drop::hitVolume() const
{
    return {
        .dimensions = {
            .radius = s_diagonalSize / 2.0f,
            .height = s_size,
        },
        .position = m_position,
    };
}

std::array<MeshVertex, Drop::s_meshVertexCount> Drop::makeMesh()
{
    std::array<MeshVertex, Drop::s_meshVertexCount> vertices;
    MeshBuilder mb(vertices);
    mb.appendMeshCuboid(
        { -s_size / 2.0f,       0, -s_size / 2.0f },
        {  s_size / 2.0f,  s_size,  s_size / 2.0f },
        Material::TorchSide, { 1.0f, 1.0f, 1.0f }
    );
    for (std::size_t i = 0; i < 6; ++i)
        vertices[i].material = static_cast<std::uint32_t>(Material::TorchTop);
    assert(mb.verticesWritten() == vertices.size());
    return vertices;
}

} // namespace blocklab
