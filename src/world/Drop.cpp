#include "Drop.h"

#include <graphics/Mesh.h>

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace blocklab {

std::array<MeshVertex, Drop::s_meshVertexCount> Drop::makeMesh()
{
    std::array<MeshVertex, Drop::s_meshVertexCount> vertices;
    MeshBuilder mb(vertices);
    mb.appendMeshCuboid(
        { -0.125f, -0.125f, -0.125f },
        {  0.125f,  0.125f,  0.125f },
        Material::TorchSide, { 1.0f, 1.0f, 1.0f }
    );
    for (std::size_t i = 0; i < 6; ++i)
        vertices[i].material = static_cast<std::uint32_t>(Material::TorchTop);
    assert(mb.verticesWritten() == vertices.size());
    return vertices;
}

} // namespace blocklab
