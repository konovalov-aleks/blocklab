#pragma once

#include <blocklab/utility/Math.h>

#include <cstddef>
#include <cstdint>
#include <span>

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
    Vec2 uv;
    std::uint32_t material;
    std::uint32_t padding0;
    Vec3 color;
    float padding1;
};

class MeshBuilder {
public:
    MeshBuilder(std::span<MeshVertex> dstBuffer)
        : m_begin(dstBuffer.data())
        , m_data(m_begin)
        , m_end(dstBuffer.data() + dstBuffer.size())
    {}

    void appendMeshCuboid(Vec3 min, Vec3 max, Material, Vec3 color, float animationPhase = 0.0f);
    void appendMeshPatch(Vec3 min, Vec3 max, float z, Vec3 color);

    std::size_t verticesWritten() const { return m_data - m_begin; }

private:
    void appendMeshFace(Vec3 p0, Vec3 p1, Vec3 p2, Vec3 p3,
                        Vec3 color, Material material, float animationPhase);

    MeshVertex* const m_begin;
    MeshVertex* m_data;
    MeshVertex* const m_end;
};

} // namespace blocklab
