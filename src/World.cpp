#include "blocklab/World.h"

#include "blocklab/Error.h"
#include "blocklab/Hash.h"
#include "blocklab/characters/PigCharacter.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace blocklab {

namespace {

    int32_t floorDiv(int32_t value, int32_t divisor)
    {
        if (value >= 0)
            return value / divisor;
        return -((-value + divisor - 1) / divisor);
    }

    int32_t overrideClusterLocalCoord(int32_t value, int32_t cluster)
    {
        return value - cluster * OverrideCluster::Edge;
    }

    int32_t overrideClusterIndex(int32_t x, int32_t y, int32_t z)
    {
        return x + z * OverrideCluster::Edge + y * OverrideCluster::Edge * OverrideCluster::Edge;
    }

    std::size_t denseBlockIndex(IVec3 local, IVec3 size)
    {
        return static_cast<std::size_t>(local.x) + static_cast<std::size_t>(local.y) * static_cast<std::size_t>(size.x)
            + static_cast<std::size_t>(local.z) * static_cast<std::size_t>(size.x) * static_cast<std::size_t>(size.y);
    }

} // namespace

Block Chunk::get(int32_t x, int32_t y, int32_t z) const
{
    if (x < 0 || x >= SizeX || y < 0 || y >= SizeY || z < 0 || z >= SizeZ)
        return Block::Air;
    const int32_t index = (y * SizeZ + z) * SizeX + x;
    return m_blocks[static_cast<std::size_t>(index)];
}

void Chunk::set(int32_t x, int32_t y, int32_t z, Block block)
{
    if (x < 0 || x >= SizeX || y < 0 || y >= SizeY || z < 0 || z >= SizeZ)
        return;
    const int32_t index = (y * SizeZ + z) * SizeX + x;
    m_blocks[static_cast<std::size_t>(index)] = block;
}

OverrideCluster::OverrideCluster() { m_blocks.fill(NoOverride); }

OverrideCluster::Mask OverrideCluster::bitFor(std::size_t index) { return Mask { 1 } << index; }

bool OverrideCluster::hasOverride(std::size_t index) const { return (m_overrideMask & bitFor(index)) != 0; }

bool OverrideCluster::hasSolidOverride(std::size_t index) const { return (m_solidMask & bitFor(index)) != 0; }

std::optional<Block> OverrideCluster::get(std::size_t index) const
{
    if (!hasOverride(index))
        return std::nullopt;
    const uint8_t stored = m_blocks[index];
    return static_cast<Block>(stored);
}

bool OverrideCluster::set(std::size_t index, Block block)
{
    uint8_t& stored = m_blocks[index];
    const Mask bit = bitFor(index);
    const bool inserted = (m_overrideMask & bit) == 0;
    if (inserted)
        ++m_count;

    m_overrideMask |= bit;
    if (block == Block::Air)
        m_solidMask &= ~bit;
    else
        m_solidMask |= bit;

    stored = static_cast<uint8_t>(block);
    return inserted;
}

bool OverrideCluster::clear(std::size_t index)
{
    const Mask bit = bitFor(index);
    if ((m_overrideMask & bit) == 0)
        return false;

    m_blocks[index] = NoOverride;
    m_overrideMask &= ~bit;
    m_solidMask &= ~bit;
    --m_count;
    return true;
}

void World::BlocksCache::waitIfPending()
{
    if (state == State::Borrowed) [[unlikely]]
        fatalError("World block cache is borrowed for generation");

    if (!pendingFuture.valid())
        return;

    WorldGenerationOutput& output = pendingFuture.get();
    origin = output.origin;
    size = output.size;
    blocks = std::move(output.buffers.blocks);
    pendingFuture = {};
    state = State::Ready;
}

WorldGenerationBuffers World::borrowGenerationBuffers(std::span<MeshVertex> meshVertices) const
{
    m_blockCache.waitIfPending();
    m_blockCache.state = BlocksCache::State::Borrowed;
    return {
        .meshVertices = meshVertices,
        .blocks = std::move(m_blockCache.blocks),
    };
}

void World::updateGeneration(CudaSharedFuture<WorldGenerationOutput> generation) const
{
    if (m_blockCache.state != BlocksCache::State::Borrowed) [[unlikely]]
        fatalError("World generation buffers were not borrowed");

    m_blockCache.pendingFuture = std::move(generation);
    m_blockCache.state = BlocksCache::State::Pending;
}

void World::waitForGeneration() const
{
    m_blockCache.waitIfPending();
}

void World::resetSeed(uint32_t seed)
{
    m_seed = seed;
    m_overrideColumns.clear();
    m_overrideCount = 0;
    m_blockCache.clear();
    ++m_version;
}

void World::resetCharacters()
{
    if (m_blockCache.state == BlocksCache::State::Empty) [[unlikely]]
        fatalError("World generation cache is not ready");

    m_nextEntityId = 1;
    m_characters.clear();
    spawnTestPigs();
}

bool World::isInsideCacheBounds(IVec3 pos) const
{
    return pos.x >= m_blockCache.origin.x
        && pos.y >= m_blockCache.origin.y
        && pos.z >= m_blockCache.origin.z
        && pos.x < m_blockCache.origin.x + m_blockCache.size.x
        && pos.y < m_blockCache.origin.y + m_blockCache.size.y
        && pos.z < m_blockCache.origin.z + m_blockCache.size.z;
}

Block World::getBlock(IVec3 pos) const
{
    m_blockCache.waitIfPending();

    if (pos.y < 0 || pos.y >= Chunk::SizeY)
        return Block::Air;

    if (!isInsideCacheBounds(pos)) [[unlikely]]
        fatalError("Requested block (", pos.x, ", ", pos.y, ", ", pos.z, ") is outside of the world generation cache bounds");

    if (m_blockCache.blocks.empty()) [[unlikely]]
        fatalError("World generation cache blocks are not ready");

    const IVec3 local = pos - m_blockCache.origin;
    return static_cast<Block>(m_blockCache.blocks[denseBlockIndex(local, m_blockCache.size)]);
}

void World::setBlock(IVec3 pos, Block block)
{
    m_blockCache.waitIfPending();
    if (pos.y < 0 || pos.y >= Chunk::SizeY)
        return;

    if (getBlock(pos) == block)
        return;

    // add override
    const int32_t clusterX = floorDiv(pos.x, OverrideCluster::Edge);
    const int32_t clusterY = floorDiv(pos.y, OverrideCluster::Edge);
    const int32_t clusterZ = floorDiv(pos.z, OverrideCluster::Edge);
    const int32_t localX = overrideClusterLocalCoord(pos.x, clusterX);
    const int32_t localY = overrideClusterLocalCoord(pos.y, clusterY);
    const int32_t localZ = overrideClusterLocalCoord(pos.z, clusterZ);
    const std::size_t localIndex = static_cast<std::size_t>(overrideClusterIndex(localX, localY, localZ));

    auto columnIt = m_overrideColumns.find(clusterX, clusterZ);
    if (columnIt == m_overrideColumns.end()) {
        m_overrideColumns.insert(clusterX, clusterZ, OverrideClusterColumn {});
        columnIt = m_overrideColumns.find(clusterX, clusterZ);
    }

    OverrideCluster& cluster = columnIt->clusters[clusterY];
    if (cluster.set(localIndex, block))
        ++m_overrideCount;

    // update the cache
    if (isInsideCacheBounds(pos)) {
        const IVec3 local = pos - m_blockCache.origin;
        m_blockCache.blocks[denseBlockIndex(local, m_blockCache.size)] = static_cast<uint8_t>(block);
    }

    ++m_version;
}

bool World::hasSolidBlockInArea(IVec3 min, IVec3 max) const
{
    m_blockCache.waitIfPending();

    if (m_blockCache.blocks.empty()) [[unlikely]]
        fatalError("World generation cache blocks are not ready");

    min = glm::max(min, m_blockCache.origin);
    max = glm::min(max, m_blockCache.origin + m_blockCache.size - IVec3 { 1 });

    return cachedSolidBlockInArea(min, max);
}

bool World::cachedSolidBlockInArea(IVec3 min, IVec3 max) const
{
    const IVec3 localMin = min - m_blockCache.origin;
    const IVec3 localMax = max - m_blockCache.origin;
    for (int32_t y = localMin.y; y <= localMax.y; ++y) {
        for (int32_t z = localMin.z; z <= localMax.z; ++z) {
            for (int32_t x = localMin.x; x <= localMax.x; ++x) {
                if (m_blockCache.blocks[denseBlockIndex({ x, y, z }, m_blockCache.size)]
                    != static_cast<uint8_t>(Block::Air)) {
                    return true;
                }
            }
        }
    }
    return false;
}

void World::update(float dt, Vec3 threatPosition)
{
    updateCharacters(dt, threatPosition);
    m_logicalTimeMs += static_cast<uint64_t>(dt * 1000.0f);
}

void World::updateCharacters(float dt, Vec3 threatPosition)
{
    for (std::unique_ptr<NPC>& character : m_characters) {
        if (!character->isAlive())
            continue;
        character->update(*this, threatPosition, dt);
    }
}

float World::groundHeight(float x, float z) const
{
    // TODO compute on GPU and cache
    const int32_t wx = floorToInt32(x);
    const int32_t wz = floorToInt32(z);
    for (int32_t y = Chunk::SizeY - 1; y >= 0; --y) {
        if (isSolid({ wx, y, wz }))
            return static_cast<float>(y + 1);
    }
    return 0.0f;
}

void World::collectOverridesInRegion(IVec3 origin, IVec3 size, std::vector<BlockOverride>& out) const
{
    out.clear();
    const IVec3 end = origin + size;
    const int32_t startClusterX = floorDiv(origin.x, OverrideCluster::Edge);
    const int32_t startClusterY = floorDiv(origin.y, OverrideCluster::Edge);
    const int32_t startClusterZ = floorDiv(origin.z, OverrideCluster::Edge);
    const int32_t endClusterX = floorDiv(end.x - 1, OverrideCluster::Edge);
    const int32_t endClusterY = floorDiv(end.y - 1, OverrideCluster::Edge);
    const int32_t endClusterZ = floorDiv(end.z - 1, OverrideCluster::Edge);

    for (int32_t clusterZ = startClusterZ; clusterZ <= endClusterZ; ++clusterZ) {
        for (int32_t clusterX = startClusterX; clusterX <= endClusterX; ++clusterX) {
            const auto columnIt = m_overrideColumns.find(clusterX, clusterZ);
            if (columnIt == m_overrideColumns.end())
                continue;

            const auto startYIt = columnIt->clusters.lower_bound(startClusterY);
            const auto endYIt = columnIt->clusters.upper_bound(endClusterY);
            for (auto clusterIt = startYIt; clusterIt != endYIt; ++clusterIt) {
                const int32_t clusterY = clusterIt->first;
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

                const int32_t localStartX = std::max(0, origin.x - clusterOrigin.x);
                const int32_t localStartY = std::max(0, origin.y - clusterOrigin.y);
                const int32_t localStartZ = std::max(0, origin.z - clusterOrigin.z);
                const int32_t localEndX = std::min(OverrideCluster::Edge, end.x - clusterOrigin.x);
                const int32_t localEndY = std::min(OverrideCluster::Edge, end.y - clusterOrigin.y);
                const int32_t localEndZ = std::min(OverrideCluster::Edge, end.z - clusterOrigin.z);
                for (int32_t y = localStartY; y < localEndY; ++y) {
                    for (int32_t z = localStartZ; z < localEndZ; ++z) {
                        for (int32_t x = localStartX; x < localEndX; ++x) {
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
    static constexpr int32_t PigCount = 32;
    static constexpr float SpawnRadius = 14.0f;
    static constexpr float MinAgentDistance = 3.5f;

    for (int32_t i = 0; i < PigCount; ++i) {
        const float angle = randomFloat01(hashCombine(m_seed, static_cast<uint32_t>(i), 0x27d4eb2dU)) * 2.0f * Pi;
        const float t
            = (static_cast<float>(i) + randomFloat01(hashCombine(m_seed, static_cast<uint32_t>(i), 0x85ebca6bU)))
            / static_cast<float>(PigCount);
        const float radius = MinAgentDistance + std::sqrt(t) * (SpawnRadius - MinAgentDistance);
        const float x = 0.5f + std::cos(angle) * radius;
        const float z = 0.5f + std::sin(angle) * radius;
        const Vec3 position {
            x,
            groundHeight(x, z) + 0.05f,
            z,
        };
        m_characters.push_back(std::make_unique<PigCharacter>(m_nextEntityId++, position));
    }
}

} // namespace blocklab
