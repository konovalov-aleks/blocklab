#include "WorldGenerator.h"

#include "World.h"

#include <cstdint>
#include <utility>

namespace blocklab {

CudaFuture<WorldGenerationOutput> WorldGenerator::generate(
    const World& world, const AgentState& agent, WorldGenerationBuffers&& buffers)
{
    const IVec3 center {
        floorToInt32(agent.position.x),
        floorToInt32(agent.position.y),
        floorToInt32(agent.position.z),
    };
    const std::uint32_t extent = m_config.halfExtent * 2;
    // The cached/rendered area is half-open: [center - halfExtent, center + halfExtent).
    const std::int32_t originX = center.x - m_config.halfExtent;
    const std::int32_t originZ = center.z - m_config.halfExtent;
    const UVec3 size { extent, World::s_height, extent };

    world.collectOverridesInRegion({ originX, World::s_minY, originZ }, size, m_overrides);
    const WorldGenerationInput input {
        .seed = world.seed(),
        .worldVersion = world.version(),
        .center = center,
        .size = size,
        .origin = { originX, World::s_minY, originZ },
        .overrides = m_overrides,
    };
    return generate(input, std::move(buffers));
}

} // namespace blocklab
