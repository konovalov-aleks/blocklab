#pragma once

#include "blocklab/Math.h"
#include "blocklab/MeshTypes.h"
#include "blocklab/PageLockedVector.h"

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

    uint32_t rebuild(uint32_t seed, IVec3 center, int32_t halfExtent,
        const std::vector<TerrainBlockOverride>& overrides, MeshVertex* outVertices, uint32_t maxVertices,
        PageLockedVector<uint8_t>& outBlocks);

private:
    struct State;
    std::unique_ptr<State> m_state;
};

} // namespace blocklab
