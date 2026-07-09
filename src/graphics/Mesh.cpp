#include "Mesh.h"

#include <blocklab/utility/Error.h>
#include <blocklab/utility/Math.h>

#include <cassert>
#include <cstdint>

namespace blocklab {

void MeshBuilder::appendMeshCuboid(Vec3 min, Vec3 max, Material material, Vec3 color, float animationPhase)
{
    const Vec3 p000 { min.x, min.y, min.z };
    const Vec3 p100 { max.x, min.y, min.z };
    const Vec3 p010 { min.x, max.y, min.z };
    const Vec3 p110 { max.x, max.y, min.z };
    const Vec3 p001 { min.x, min.y, max.z };
    const Vec3 p101 { max.x, min.y, max.z };
    const Vec3 p011 { min.x, max.y, max.z };
    const Vec3 p111 { max.x, max.y, max.z };
    appendMeshFace(p010, p011, p111, p110, color, material, animationPhase);
    appendMeshFace(p000, p100, p101, p001, color, material, animationPhase);
    appendMeshFace(p100, p110, p111, p101, color, material, animationPhase);
    appendMeshFace(p000, p001, p011, p010, color, material, animationPhase);
    appendMeshFace(p001, p101, p111, p011, color, material, animationPhase);
    appendMeshFace(p000, p010, p110, p100, color, material, animationPhase);
}

void MeshBuilder::appendMeshPatch(Vec3 min, Vec3 max, float z, Vec3 color)
{
    appendMeshFace(
        { min.x, min.y, z },
        { max.x, min.y, z },
        { max.x, max.y, z },
        { min.x, max.y, z },
        color, Material::VertexColor, 0.0f
    );
}

void MeshBuilder::appendMeshFace(Vec3 p0, Vec3 p1, Vec3 p2, Vec3 p3,
                                Vec3 color, Material material, float animationPhase)
{
    if (m_data + 6 > m_end) [[unlikely]]
        fatalError("MeshBuilder: the destination buffer is too small");

    const Vec3 normal = glm::normalize(glm::cross(p1 - p0, p2 - p0));
    const std::uint32_t materialId = static_cast<std::uint32_t>(material);
    *(m_data++) = {
        .position = p0,
        .animationPhase = animationPhase,
        .normal = normal,
        .uv = { 0.0f, 0.0f },
        .material = materialId,
        .color = color };
    *(m_data++) = {
        .position = p1,
        .animationPhase = animationPhase,
        .normal = normal,
        .uv = { 1.0f, 0.0f },
        .material = materialId,
        .color = color };
    *(m_data++) = {
        .position = p2,
        .animationPhase = animationPhase,
        .normal = normal,
        .uv = { 1.0f, 1.0f },
        .material = materialId,
        .color = color };
    *(m_data++) = {
        .position = p0,
        .animationPhase = animationPhase,
        .normal = normal,
        .uv = { 0.0f, 0.0f },
        .material = materialId,
        .color = color };
    *(m_data++) = {
        .position = p2,
        .animationPhase = animationPhase,
        .normal = normal,
        .uv = { 1.0f, 1.0f },
        .material = materialId,
        .color = color };
    *(m_data++) = {
        .position = p3,
        .animationPhase = animationPhase,
        .normal = normal,
        .uv = { 0.0f, 1.0f },
        .material = materialId,
        .color = color };
    assert(m_data <= m_end);
}

} // namespace blocklab
