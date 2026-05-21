#pragma once

#include "blocklab/Math.h"
#include "blocklab/MeshTypes.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace blocklab {

class CudaTerrainMeshBuilder final {
public:
    CudaTerrainMeshBuilder();
    ~CudaTerrainMeshBuilder();

    CudaTerrainMeshBuilder(const CudaTerrainMeshBuilder&) = delete;
    CudaTerrainMeshBuilder& operator=(const CudaTerrainMeshBuilder&) = delete;

    void rebuild(uint32_t seed, IVec3 center, int32_t radius, const std::vector<TerrainBlockOverride>& overrides,
        std::vector<MeshVertex>& outVertices);

private:
    struct State;
    std::unique_ptr<State> m_state;
};

} // namespace blocklab
