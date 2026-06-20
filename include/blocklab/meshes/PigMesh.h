#pragma once

#include "blocklab/MeshTypes.h"

#include <array>
#include <cstdint>
#include <span>

namespace blocklab {

class PigMesh {
public:
    static constexpr std::uint32_t verticesCount() { return s_verticesCount; }

    std::span<MeshVertex> generate();

private:
    static constexpr std::uint32_t s_cuboidCount = 7;
    static constexpr std::uint32_t s_patchCount = 4;
    static constexpr std::uint32_t s_verticesCount = s_cuboidCount * 36U + s_patchCount * 6U;

    std::array<MeshVertex, s_verticesCount> m_vertices;
};

} // namespace blocklab
