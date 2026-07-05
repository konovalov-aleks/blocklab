#pragma once

#include <blocklab/inventory/Item.h>
#include <blocklab/utility/Math.h>
#include <graphics/Mesh.h>

#include <array>
#include <cstdint>

namespace blocklab {

class Drop {
public:
    static constexpr std::uint64_t s_lifetimeMs = 5 * 60 * 1000;

    Drop(std::uint64_t creationTime, IVec3 pos, Item item)
        : m_creationTime(creationTime)
        , m_position(pos)
        , m_item(item)
    {}

    std::uint64_t creationTime() const { return m_creationTime; }
    const Item& item() const { return m_item; }

    IVec3 position() const { return m_position; }
    void setPosition(IVec3 pos) { m_position = pos; }

    static constexpr std::uint32_t s_meshVertexCount = 36;
    static std::array<MeshVertex, s_meshVertexCount> makeMesh();

private:
    std::uint64_t m_creationTime;
    IVec3 m_position;
    Item m_item;
};

} // namespace blocklab
