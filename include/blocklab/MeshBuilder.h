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

    void rebuild(const World& world, const AgentState& agent);
    const std::vector<MeshVertex>& vertices() const { return m_vertices; }

private:
    MeshBuildConfig m_config;
    std::vector<MeshVertex> m_vertices;
    std::vector<BlockOverride> m_overrides;
    std::vector<TerrainBlockOverride> m_terrainOverrides;
    std::unique_ptr<blocklab::CudaTerrainMeshBuilder> m_cudaTerrain;
};

} // namespace blocklab
