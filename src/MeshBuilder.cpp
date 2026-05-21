#include "blocklab/MeshBuilder.h"

#include "blocklab/CudaMeshBuilder.h"

#include <memory>

namespace blocklab {

MeshBuilder::MeshBuilder(MeshBuildConfig config)
    : m_config(config)
    , m_cudaTerrain(std::make_unique<CudaTerrainMeshBuilder>())
{
}

MeshBuilder::~MeshBuilder() = default;

void MeshBuilder::rebuild(const World& world, const AgentState& agent)
{
    const IVec3 center { floorToInt(agent.position.x), floorToInt(agent.position.y), floorToInt(agent.position.z) };
    const IVec3 origin { center.x - m_config.radius, 0, center.z - m_config.radius };
    const IVec3 size { m_config.radius * 2 + 1, Chunk::SizeY, m_config.radius * 2 + 1 };
    world.collectOverridesInRegion(origin, size, m_overrides);
    m_terrainOverrides.clear();
    m_terrainOverrides.reserve(m_overrides.size());
    for (const BlockOverride& blockOverride : m_overrides) {
        m_terrainOverrides.push_back({
            .x = blockOverride.coord.x,
            .y = blockOverride.coord.y,
            .z = blockOverride.coord.z,
            .block = blockId(blockOverride.block),
        });
    }
    m_cudaTerrain->rebuild(world.seed(), center, m_config.radius, m_terrainOverrides, m_vertices);
}

} // namespace blocklab
