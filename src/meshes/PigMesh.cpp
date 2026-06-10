#include "blocklab/meshes/PigMesh.h"

#include <cassert>
#include <cstdint>

namespace blocklab {
namespace {

    constexpr float meshMaterialId(Material material) { return static_cast<float>(static_cast<uint32_t>(material)); }

    void appendMeshFace(MeshVertex*& vertices, Vec3 p0, Vec3 p1, Vec3 p2, Vec3 p3, Vec3 color, float shade,
        Material material, float animationPhase)
    {
        const Vec4 packedColor { color, shade };
        const float materialId = meshMaterialId(material);
        vertices[0] = { .position = { p0, animationPhase },
            .colorAndShade = packedColor,
            .uvMaterial = { 0.0f, 0.0f, materialId, 0.0f } };
        vertices[1] = { .position = { p1, animationPhase },
            .colorAndShade = packedColor,
            .uvMaterial = { 1.0f, 0.0f, materialId, 0.0f } };
        vertices[2] = { .position = { p2, animationPhase },
            .colorAndShade = packedColor,
            .uvMaterial = { 1.0f, 1.0f, materialId, 0.0f } };
        vertices[3] = { .position = { p0, animationPhase },
            .colorAndShade = packedColor,
            .uvMaterial = { 0.0f, 0.0f, materialId, 0.0f } };
        vertices[4] = { .position = { p2, animationPhase },
            .colorAndShade = packedColor,
            .uvMaterial = { 1.0f, 1.0f, materialId, 0.0f } };
        vertices[5] = { .position = { p3, animationPhase },
            .colorAndShade = packedColor,
            .uvMaterial = { 0.0f, 1.0f, materialId, 0.0f } };
        vertices += 6;
    }

    void appendMeshCuboid(MeshVertex*& vertices, Vec3 min, Vec3 max, Vec3 color, float animationPhase = 0.0f)
    {
        const Vec3 p000 { min.x, min.y, min.z };
        const Vec3 p100 { max.x, min.y, min.z };
        const Vec3 p010 { min.x, max.y, min.z };
        const Vec3 p110 { max.x, max.y, min.z };
        const Vec3 p001 { min.x, min.y, max.z };
        const Vec3 p101 { max.x, min.y, max.z };
        const Vec3 p011 { min.x, max.y, max.z };
        const Vec3 p111 { max.x, max.y, max.z };
        const Material material = color.g > 0.6f ? Material::PigSnout : Material::PigSkin;
        appendMeshFace(vertices, p010, p011, p111, p110, color, 1.0f, material, animationPhase);
        appendMeshFace(vertices, p000, p100, p101, p001, color, 0.48f, material, animationPhase);
        appendMeshFace(vertices, p100, p110, p111, p101, color, 0.78f, material, animationPhase);
        appendMeshFace(vertices, p000, p001, p011, p010, color, 0.78f, material, animationPhase);
        appendMeshFace(vertices, p001, p101, p111, p011, color, 0.68f, material, animationPhase);
        appendMeshFace(vertices, p000, p010, p110, p100, color, 0.68f, material, animationPhase);
    }

    void appendMeshPatch(MeshVertex*& vertices, Vec3 min, Vec3 max, float z, Vec3 color)
    {
        appendMeshFace(vertices, { min.x, min.y, z }, { max.x, min.y, z }, { max.x, max.y, z }, { min.x, max.y, z },
            color, 1.0f, Material::VertexColor, 0.0f);
    }

} // namespace

std::span<MeshVertex> PigMesh::generate()
{
    MeshVertex* const destination = m_vertices.data();
    MeshVertex* vertices = destination;
    const Vec3 skin { 0.88f, 0.56f, 0.65f };
    const Vec3 snout { 0.96f, 0.66f, 0.73f };
    const Vec3 dark { 0.08f, 0.06f, 0.07f };
    appendMeshCuboid(vertices, { -0.36f, 0.24f, -0.58f }, { 0.36f, 0.78f, 0.58f }, skin);
    appendMeshCuboid(vertices, { -0.30f, 0.34f, 0.50f }, { 0.30f, 0.84f, 0.98f }, skin);
    appendMeshCuboid(vertices, { -0.16f, 0.48f, 0.94f }, { 0.16f, 0.66f, 1.08f }, snout);
    appendMeshPatch(vertices, { -0.23f, 0.65f, 0.0f }, { -0.13f, 0.75f, 0.0f }, 0.982f, dark);
    appendMeshPatch(vertices, { 0.13f, 0.65f, 0.0f }, { 0.23f, 0.75f, 0.0f }, 0.982f, dark);
    appendMeshPatch(vertices, { -0.10f, 0.54f, 0.0f }, { -0.04f, 0.61f, 0.0f }, 1.082f, dark);
    appendMeshPatch(vertices, { 0.04f, 0.54f, 0.0f }, { 0.10f, 0.61f, 0.0f }, 1.082f, dark);
    appendMeshCuboid(vertices, { -0.31f, 0.00f, -0.45f }, { -0.15f, 0.28f, -0.25f }, skin, 2.0f);
    appendMeshCuboid(vertices, { 0.15f, 0.00f, -0.45f }, { 0.31f, 0.28f, -0.25f }, skin, -2.0f);
    appendMeshCuboid(vertices, { -0.31f, 0.00f, 0.25f }, { -0.15f, 0.28f, 0.45f }, skin, -2.0f);
    appendMeshCuboid(vertices, { 0.15f, 0.00f, 0.25f }, { 0.31f, 0.28f, 0.45f }, skin, 2.0f);
    assert(static_cast<uint32_t>(vertices - destination) == verticesCount());
    return { m_vertices.data(), static_cast<std::size_t>(vertices - destination) };
}

} // namespace blocklab
