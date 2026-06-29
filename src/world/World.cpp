#include "World.h"

#include <blocklab/utility/Error.h>
#include <characters/PigCharacter.h>
#include <utility/Hash.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace blocklab {

namespace {

    std::int32_t floorDiv(std::int32_t value, std::int32_t divisor)
    {
        if (value >= 0)
            return value / divisor;
        return -((-value + divisor - 1) / divisor);
    }

    std::int32_t overrideClusterLocalCoord(std::int32_t value, std::int32_t cluster)
    {
        return value - cluster * OverrideCluster::Edge;
    }

    std::int32_t overrideClusterIndex(std::int32_t x, std::int32_t y, std::int32_t z)
    {
        return x + z * OverrideCluster::Edge + y * OverrideCluster::Edge * OverrideCluster::Edge;
    }

} // namespace

BlockInfo& World::BlockCache::operator[](IVec3 pos)
{
    assert(!empty());
    assert(m_state == State::Ready);
    assert(m_blocks.size() == static_cast<std::size_t>(m_size.x) * m_size.y * m_size.z);
    assert(isInsideBounds(pos));
    return m_blocks[denseBlockIndex(pos - m_origin)];
}

std::int32_t World::BlockCache::terrainHeight(IVec2 xz)
{
    waitIfPending();
    assert(!empty());
    assert(m_state == State::Ready);
    assert(m_heightMap.size() == static_cast<std::size_t>(m_size.x) * m_size.z);

    const std::int32_t localX = xz[0] - m_origin.x;
    const std::int32_t localZ = xz[1] - m_origin.z;
    assert(localX >= 0 && localX < m_size.x);
    assert(localZ >= 0 && localZ < m_size.z);
    return m_heightMap[localX + m_size.x * localZ];
}

bool World::BlockCache::isInsideBounds(IVec3 pos) const
{
    return pos.x >= m_origin.x && pos.y >= m_origin.y && pos.z >= m_origin.z && pos.x < m_origin.x + m_size.x
        && pos.y < m_origin.y + m_size.y && pos.z < m_origin.z + m_size.z;
}

void World::BlockCache::clear()
{
    waitIfPending();
    m_origin = {};
    m_size = {};
    m_blocks.clear();
    m_state = State::Empty;
}

void World::BlockCache::waitIfPending()
{
    if (m_state == State::Borrowed) [[unlikely]]
        fatalError("World block cache is borrowed for generation");

    if (!m_pendingFuture.valid())
        return;

    WorldGenerationOutput& output = m_pendingFuture.get();
    m_origin = output.origin;
    m_size = output.size;
    m_blocks = std::move(output.buffers.cpuCache.blocks);
    m_heightMap = std::move(output.buffers.cpuCache.heightMap);
    m_pendingFuture = {};
    m_state = State::Ready;
}

CPUCacheGenerationBuffers World::BlockCache::borrowGenerationBuffers()
{
    waitIfPending();
    m_state = State::Borrowed;
    return {
        .blocks = std::move(m_blocks),
        .heightMap = std::move(m_heightMap),
    };
}

void World::BlockCache::update(CudaSharedFuture<WorldGenerationOutput> gen)
{
    if (m_state != State::Borrowed) [[unlikely]]
        fatalError("World generation buffers were not borrowed");

    m_pendingFuture = std::move(gen);
    m_state = State::Pending;
}

std::size_t World::BlockCache::denseBlockIndex(IVec3 local) const
{
    assert(glm::all(glm::greaterThanEqual(local, IVec3(0, 0, 0))) && glm::all(glm::lessThan(local, m_size)));
    return local.x + m_size.x * (local.y + m_size.y * local.z);
}

void World::waitForGeneration() const { m_blockCache.waitIfPending(); }

void World::resetSeed(std::uint32_t seed)
{
    m_seed = seed;
    m_logicalTimeMs = 0;
    m_overrideColumns.clear();
    m_overrideCount = 0;
    m_blockCache.clear();
    m_dayTimeShiftTicks = hash(seed) % s_ticksPerGameDay;
    ++m_version;
}

void World::resetCharacters()
{
    m_blockCache.waitIfPending();

    if (m_blockCache.empty()) [[unlikely]]
        fatalError("World generation cache is not ready");

    m_nextEntityId = 1;
    m_characters.clear();
    spawnTestPigs();
}

Block World::getBlock(IVec3 pos) const
{
    m_blockCache.waitIfPending();

    if (!isValidHeight(pos.y))
        return Block::Air;

    if (!m_blockCache.isInsideBounds(pos)) [[unlikely]] {
        fatalError(
            "Requested block (", pos.x, ", ", pos.y, ", ", pos.z, ") is outside of the world generation cache bounds");
    }

    if (m_blockCache.empty()) [[unlikely]]
        fatalError("World generation cache blocks are not ready");

    return m_blockCache[pos].blockType;
}

void World::setBlock(IVec3 pos, Block block)
{
    m_blockCache.waitIfPending();
    if (!isValidHeight(pos.y))
        return;

    const Block oldBlock = getBlock(pos);
    if (oldBlock == block)
        return;

    // add override
    const std::int32_t clusterX = floorDiv(pos.x, OverrideCluster::Edge);
    const std::int32_t clusterY = floorDiv(pos.y, OverrideCluster::Edge);
    const std::int32_t clusterZ = floorDiv(pos.z, OverrideCluster::Edge);
    const std::int32_t localX = overrideClusterLocalCoord(pos.x, clusterX);
    const std::int32_t localY = overrideClusterLocalCoord(pos.y, clusterY);
    const std::int32_t localZ = overrideClusterLocalCoord(pos.z, clusterZ);
    const std::size_t localIndex = static_cast<std::size_t>(overrideClusterIndex(localX, localY, localZ));

    auto columnIt = m_overrideColumns.find(clusterX, clusterZ);
    if (columnIt == m_overrideColumns.end()) {
        m_overrideColumns.insert(clusterX, clusterZ, OverrideClusterColumn {});
        columnIt = m_overrideColumns.find(clusterX, clusterZ);
    }

    OverrideCluster& cluster = (*columnIt)[clusterY];
    if (cluster.set(localIndex, block))
        ++m_overrideCount;

    // TODO: Revisit cache patching when lighting becomes incremental. This keeps CPU collision data coherent, but does
    // not update derived lighting data; GPU-side unlight/light should eventually handle that without full regeneration.
    if (m_blockCache.isInsideBounds(pos))
        m_blockCache[pos].blockType = block;

    ++m_version;

    const IVec3 above = pos + IVec3 { 0, 1, 0 };
    if (isSolidBlock(oldBlock) && !isSolidBlock(block) && getBlock(above) == Block::Torch)
        setBlock(above, Block::Air);
}

bool World::hasSolidBlockInArea(IVec3 min, IVec3 max) const
{
    m_blockCache.waitIfPending();

    if (m_blockCache.empty()) [[unlikely]]
        fatalError("World generation cache is not ready");

    min = glm::max(min, m_blockCache.origin());
    max = glm::min(max, m_blockCache.origin() + m_blockCache.size() - IVec3 { 1 });

    for (std::int32_t y = min.y; y <= max.y; ++y) {
        for (std::int32_t z = min.z; z <= max.z; ++z) {
            for (std::int32_t x = min.x; x <= max.x; ++x) {
                const Block block = m_blockCache[{ x, y, z }].blockType;
                if (isSolidBlock(block))
                    return true;
            }
        }
    }
    return false;
}

void World::update(float dt, Vec3 threatPosition)
{
    updateCharacters(dt, threatPosition);
    m_logicalTimeMs += static_cast<std::uint64_t>(dt * 1000.0f);
}

void World::updateCharacters(float dt, Vec3 threatPosition)
{
    for (std::unique_ptr<NPC>& character : m_characters) {
        if (!character->isAlive())
            continue;
        character->update(*this, threatPosition, dt);
    }
}

void World::collectOverridesInRegion(IVec3 origin, UVec3 size, std::vector<BlockOverride>& out) const
{
    out.clear();
    const IVec3 end = origin + static_cast<IVec3>(size);
    const std::int32_t startClusterX = floorDiv(origin.x, OverrideCluster::Edge);
    const std::int32_t startClusterY = floorDiv(origin.y, OverrideCluster::Edge);
    const std::int32_t startClusterZ = floorDiv(origin.z, OverrideCluster::Edge);
    const std::int32_t endClusterX = floorDiv(end.x - 1, OverrideCluster::Edge);
    const std::int32_t endClusterY = floorDiv(end.y - 1, OverrideCluster::Edge);
    const std::int32_t endClusterZ = floorDiv(end.z - 1, OverrideCluster::Edge);

    for (std::int32_t clusterZ = startClusterZ; clusterZ <= endClusterZ; ++clusterZ) {
        for (std::int32_t clusterX = startClusterX; clusterX <= endClusterX; ++clusterX) {
            const auto columnIt = m_overrideColumns.find(clusterX, clusterZ);
            if (columnIt == m_overrideColumns.end())
                continue;

            const auto startYIt = columnIt->lower_bound(startClusterY);
            const auto endYIt = columnIt->upper_bound(endClusterY);
            for (auto clusterIt = startYIt; clusterIt != endYIt; ++clusterIt) {
                const std::int32_t clusterY = clusterIt->first;
                const OverrideCluster& cluster = clusterIt->second;
                const IVec3 clusterOrigin {
                    clusterX * OverrideCluster::Edge,
                    clusterY * OverrideCluster::Edge,
                    clusterZ * OverrideCluster::Edge,
                };
                const IVec3 clusterEnd
                    = clusterOrigin + IVec3 { OverrideCluster::Edge, OverrideCluster::Edge, OverrideCluster::Edge };
                if (clusterEnd.x <= origin.x || clusterEnd.y <= origin.y || clusterEnd.z <= origin.z
                    || clusterOrigin.x >= end.x || clusterOrigin.y >= end.y || clusterOrigin.z >= end.z) {
                    continue;
                }

                const std::int32_t localStartX = std::max(0, origin.x - clusterOrigin.x);
                const std::int32_t localStartY = std::max(0, origin.y - clusterOrigin.y);
                const std::int32_t localStartZ = std::max(0, origin.z - clusterOrigin.z);
                const std::int32_t localEndX = std::min(OverrideCluster::Edge, end.x - clusterOrigin.x);
                const std::int32_t localEndY = std::min(OverrideCluster::Edge, end.y - clusterOrigin.y);
                const std::int32_t localEndZ = std::min(OverrideCluster::Edge, end.z - clusterOrigin.z);
                for (std::int32_t y = localStartY; y < localEndY; ++y) {
                    for (std::int32_t z = localStartZ; z < localEndZ; ++z) {
                        for (std::int32_t x = localStartX; x < localEndX; ++x) {
                            const std::optional<Block> stored
                                = cluster.get(static_cast<std::size_t>(overrideClusterIndex(x, y, z)));
                            if (!stored)
                                continue;
                            out.push_back({
                                .coord = IVec3 { clusterOrigin.x + x, clusterOrigin.y + y, clusterOrigin.z + z },
                                .block = *stored,
                            });
                        }
                    }
                }
            }
        }
    }
}

void World::spawnTestPigs()
{
    static constexpr std::int32_t PigCount = 32;
    static constexpr float SpawnRadius = 14.0f;
    static constexpr float MinAgentDistance = 3.5f;

    for (std::int32_t i = 0; i < PigCount; ++i) {
        const float angle = randomFloat01(hashCombine(m_seed, static_cast<std::uint32_t>(i), 0x27d4eb2dU)) * 2.0f * Pi;
        const float t
            = (static_cast<float>(i) + randomFloat01(hashCombine(m_seed, static_cast<std::uint32_t>(i), 0x85ebca6bU)))
            / static_cast<float>(PigCount);
        const float radius = MinAgentDistance + std::sqrt(t) * (SpawnRadius - MinAgentDistance);
        const std::int32_t x = floorToInt32(0.5f + std::cos(angle) * radius);
        const std::int32_t z = floorToInt32(0.5f + std::sin(angle) * radius);
        const Vec3 position {
            x,
            static_cast<float>(terrainHeight({ x, z })) + 1.05f,
            z,
        };
        m_characters.push_back(std::make_unique<PigCharacter>(m_nextEntityId++, position));
    }
}

} // namespace blocklab
