#include "WorldGenerator.h"

#include "CudaWorldGenerator.h"
#include "World.h"

#include <cstdint>
#include <memory>
#include <utility>

namespace blocklab {

WorldGenerator::WorldGenerator(WorldGenerationConfig config)
    : m_config(config)
    , m_cudaGenerator(std::make_unique<CudaWorldGenerator>())
{
}

WorldGenerator::~WorldGenerator() = default;

cudaStream_t WorldGenerator::stream() const { return m_cudaGenerator->stream(); }

CudaFuture<WorldGenerationOutput> WorldGenerator::generate(
    const World& world, const AgentState& agent, WorldGenerationBuffers&& buffers)
{
    const IVec3 center {
        floorToInt32(agent.position.x),
        floorToInt32(agent.position.y),
        floorToInt32(agent.position.z),
    };
    const std::int32_t extent = m_config.halfExtent * 2;
    // The cached/rendered area is half-open: [center - halfExtent, center + halfExtent).
    const std::int32_t originX = center.x - m_config.halfExtent;
    const std::int32_t originZ = center.z - m_config.halfExtent;
    const IVec3 size { extent, Chunk::SizeY, extent };

    world.collectOverridesInRegion({ originX, 0, originZ }, size, m_overrides);
    const WorldGenerationInput input {
        .seed = world.seed(),
        .worldVersion = world.version(),
        .center = center,
        .size = size,
        .originX = originX,
        .originZ = originZ,
        .halfExtent = m_config.halfExtent,
        .overrides = m_overrides,
    };
    return m_cudaGenerator->generate(input, std::move(buffers));
}

} // namespace blocklab
