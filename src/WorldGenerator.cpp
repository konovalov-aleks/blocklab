#include "blocklab/WorldGenerator.h"

#include "blocklab/CudaWorldGenerator.h"
#include "blocklab/World.h"

#include <memory>
#include <utility>

namespace blocklab {

WorldGenerator::WorldGenerator(WorldGenerationConfig config)
    : m_config(config)
    , m_cudaGenerator(std::make_unique<CudaWorldGenerator>())
{
}

WorldGenerator::~WorldGenerator() = default;

CudaFuture<WorldGenerationOutput> WorldGenerator::generate(
    const World& world, const AgentState& agent, WorldGenerationBuffers&& buffers)
{
    const IVec3 center {
        floorToInt32(agent.position.x),
        floorToInt32(agent.position.y),
        floorToInt32(agent.position.z),
    };
    const int32_t extent = m_config.halfExtent * 2;
    // The cached/rendered area is half-open: [center - halfExtent, center + halfExtent).
    const IVec3 origin { center.x - m_config.halfExtent, 0, center.z - m_config.halfExtent };
    const IVec3 size { extent, Chunk::SizeY, extent };

    world.collectOverridesInRegion(origin, size, m_overrides);
    const WorldGenerationInput input {
        .seed = world.seed(),
        .worldVersion = world.version(),
        .center = center,
        .origin = origin,
        .size = size,
        .halfExtent = m_config.halfExtent,
        .overrides = m_overrides,
    };
    return m_cudaGenerator->generate(input, std::move(buffers));
}

} // namespace blocklab
