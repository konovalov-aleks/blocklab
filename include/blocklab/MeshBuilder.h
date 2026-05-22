#pragma once

#include "blocklab/Agent.h"
#include "blocklab/MeshTypes.h"
#include "blocklab/World.h"

#include <memory>
#include <vector>

namespace blocklab {

class CudaTerrainMeshBuilder;

class MeshBuilder final {
public:
    explicit MeshBuilder(MeshBuildConfig config = {});
    ~MeshBuilder();

    uint32_t rebuild(const World& world, const AgentState& agent, MeshVertex* outVertices, uint32_t maxVertices);

private:
    MeshBuildConfig m_config;
    std::vector<BlockOverride> m_overrides;
    std::vector<TerrainBlockOverride> m_terrainOverrides;
    std::vector<uint8_t> m_solidBlocks;
    std::unique_ptr<blocklab::CudaTerrainMeshBuilder> m_cudaTerrain;
};

} // namespace blocklab
