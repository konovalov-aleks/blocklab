#include "PigMesh.h"

#include <graphics/Mesh.h>

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace blocklab {

std::span<MeshVertex> PigMesh::generate()
{
    MeshBuilder mb(m_vertices);
    const Vec3 skin { 0.88f, 0.56f, 0.65f };
    const Vec3 snout { 0.96f, 0.66f, 0.73f };
    const Vec3 dark { 0.08f, 0.06f, 0.07f };
    //const Material material = color.g > 0.6f ? Material::PigSnout : Material::PigSkin;
    mb.appendMeshCuboid(
        { -0.36f, 0.24f, -0.58f },
        {  0.36f, 0.78f,  0.58f },
        Material::PigSkin, skin
    );
    mb.appendMeshCuboid(
        { -0.30f, 0.34f, 0.50f },
        {  0.30f, 0.84f, 0.98f },
        Material::PigSkin, skin
    );
    mb.appendMeshCuboid(
        { -0.16f, 0.48f, 0.94f },
        {  0.16f, 0.66f, 1.08f },
        Material::PigSnout, snout
    );
    mb.appendMeshPatch(
        { -0.23f, 0.65f, 0.0f },
        { -0.13f, 0.75f, 0.0f },
        0.982f, dark
    );
    mb.appendMeshPatch(
        { 0.13f, 0.65f, 0.0f },
        { 0.23f, 0.75f, 0.0f },
        0.982f, dark
    );
    mb.appendMeshPatch(
        { -0.10f, 0.54f, 0.0f },
        { -0.04f, 0.61f, 0.0f },
        1.082f, dark
    );
    mb.appendMeshPatch(
        { 0.04f, 0.54f, 0.0f },
        { 0.10f, 0.61f, 0.0f },
        1.082f, dark
    );
    mb.appendMeshCuboid(
        { -0.31f, 0.00f, -0.45f },
        { -0.15f, 0.28f, -0.25f },
        Material::PigSkin, skin, 2.0f
    );
    mb.appendMeshCuboid(
        { 0.15f, 0.00f, -0.45f },
        { 0.31f, 0.28f, -0.25f },
        Material::PigSkin, skin, -2.0f
    );
    mb.appendMeshCuboid(
        { -0.31f, 0.00f, 0.25f },
        { -0.15f, 0.28f, 0.45f },
        Material::PigSkin, skin, -2.0f
    );
    mb.appendMeshCuboid(
        { 0.15f, 0.00f, 0.25f },
        { 0.31f, 0.28f, 0.45f },
        Material::PigSkin, skin, 2.0f
    );
    assert(mb.verticesWritten() == verticesCount());
    return { m_vertices.data(), mb.verticesWritten() };
}

} // namespace blocklab
