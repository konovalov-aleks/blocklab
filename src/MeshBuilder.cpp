#include "blocklab/MeshBuilder.h"

#include "blocklab/CudaMeshBuilder.h"

#include <memory>
#include <utility>

namespace blocklab {

MeshBuilder::MeshBuilder(MeshBuildConfig config)
    : m_config(config)
    , m_cudaTerrain(std::make_unique<CudaTerrainMeshBuilder>())
{
}

MeshBuilder::~MeshBuilder() = default;

uint32_t MeshBuilder::rebuild(
    const World& world, const AgentState& agent, MeshVertex* outVertices, uint32_t maxVertices)
{
    const IVec3 center { floorToInt(agent.position.x), floorToInt(agent.position.y), floorToInt(agent.position.z) };
    const int32_t extent = m_config.halfExtent * 2;
    // The cached/rendered area is half-open: [center - halfExtent, center + halfExtent).
    const IVec3 origin { center.x - m_config.halfExtent, 0, center.z - m_config.halfExtent };
    const IVec3 size { extent, Chunk::SizeY, extent };
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

    World::BlocksCache& cache = world.collisionCacheMutable();
    cache.origin = origin;
    cache.size = size;
    cache.version = world.version();

    const uint32_t vertexCount = m_cudaTerrain->rebuild(
        world.seed(), center, m_config.halfExtent, m_terrainOverrides, outVertices, maxVertices, cache.blocks);
    assert(static_cast<std::size_t>(size.x) * static_cast<std::size_t>(size.y) * static_cast<std::size_t>(size.z)
        == cache.blocks.size());
    return vertexCount;
}

} // namespace blocklab
