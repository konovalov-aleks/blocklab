#include "blocklab/World.h"

#include "blocklab/Error.h"
#include "blocklab/Hash.h"
#include "blocklab/characters/PigCharacter.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace blocklab {

namespace {

    float valueNoise(uint32_t seed, int32_t x, int32_t z)
    {
        const uint32_t hash = hashCombine(seed, x, z);
        return static_cast<float>(hash & 0x00ffffffU) / static_cast<float>(0x00ffffffU) * 2.0f - 1.0f;
    }

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

    OverrideCluster::Mask overrideClusterColumnMask(int32_t x, int32_t z, int32_t minY, int32_t maxY)
    {
        OverrideCluster::Mask mask = 0;
        for (int32_t y = minY; y <= maxY; ++y)
            mask |= OverrideCluster::Mask { 1 } << static_cast<std::size_t>(overrideClusterIndex(x, y, z));
        return mask;
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

World::World(uint32_t seed)
    : m_seed(seed)
{
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
    version = output.worldVersion;
    blocks = std::move(output.buffers.blocks);
    pendingFuture = {};
    state = State::Ready;
}

WorldGenerationBuffers World::borrowGenerationBuffers(std::span<MeshVertex> meshVertices) const
{
    m_blocksCache.waitIfPending();
    m_blocksCache.state = BlocksCache::State::Borrowed;
    return {
        .meshVertices = meshVertices,
        .blocks = std::move(m_blocksCache.blocks),
    };
}

void World::updateGeneration(CudaSharedFuture<WorldGenerationOutput> generation) const
{
    if (m_blocksCache.state != BlocksCache::State::Borrowed) [[unlikely]]
        fatalError("World generation buffers were not borrowed");

    m_blocksCache.pendingFuture = std::move(generation);
    m_blocksCache.state = BlocksCache::State::Pending;
}

void World::reset(uint32_t seed)
{
    m_seed = seed;
    m_overrideColumns.clear();
    m_overrideCount = 0;
    m_blocksCache.clear();
    m_nextEntityId = 1;
    m_characters.clear();
    spawnTestPigs();
    ++m_version;
}

Block World::getBlock(int32_t x, int32_t y, int32_t z) const
{
    m_blocksCache.waitIfPending();

    if (m_blocksCache.version == m_version && !m_blocksCache.blocks.empty() && x >= m_blocksCache.origin.x
        && y >= m_blocksCache.origin.y && z >= m_blocksCache.origin.z
        && x < m_blocksCache.origin.x + m_blocksCache.size.x && y < m_blocksCache.origin.y + m_blocksCache.size.y
        && z < m_blocksCache.origin.z + m_blocksCache.size.z) [[likely]] {

        const IVec3 local { x - m_blocksCache.origin.x, y - m_blocksCache.origin.y, z - m_blocksCache.origin.z };
        return static_cast<Block>(m_blocksCache.blocks[denseBlockIndex(local, m_blocksCache.size)]);
    }

    if (const std::optional<Block> block = overriddenBlock(x, y, z))
        return *block;
    return generatedBlock(x, y, z);
}

std::optional<Block> World::overriddenBlock(int32_t x, int32_t y, int32_t z) const
{
    const int32_t clusterX = floorDiv(x, OverrideCluster::Edge);
    const int32_t clusterY = floorDiv(y, OverrideCluster::Edge);
    const int32_t clusterZ = floorDiv(z, OverrideCluster::Edge);
    const auto columnIt = m_overrideColumns.find(clusterX, clusterZ);
    if (columnIt != m_overrideColumns.end()) {
        const int32_t localX = overrideClusterLocalCoord(x, clusterX);
        const int32_t localY = overrideClusterLocalCoord(y, clusterY);
        const int32_t localZ = overrideClusterLocalCoord(z, clusterZ);
        const auto clusterIt = columnIt->clusters.find(clusterY);
        if (clusterIt != columnIt->clusters.end()) {
            const std::optional<Block> stored
                = clusterIt->second.get(static_cast<std::size_t>(overrideClusterIndex(localX, localY, localZ)));
            if (stored)
                return *stored;
        }
    }
    return std::nullopt;
}

void World::setBlock(int32_t x, int32_t y, int32_t z, Block block)
{
    m_blocksCache.waitIfPending();
    if (y < 0 || y >= Chunk::SizeY)
        return;

    if (getBlock(x, y, z) == block)
        return;

    const int32_t clusterX = floorDiv(x, OverrideCluster::Edge);
    const int32_t clusterY = floorDiv(y, OverrideCluster::Edge);
    const int32_t clusterZ = floorDiv(z, OverrideCluster::Edge);
    const int32_t localX = overrideClusterLocalCoord(x, clusterX);
    const int32_t localY = overrideClusterLocalCoord(y, clusterY);
    const int32_t localZ = overrideClusterLocalCoord(z, clusterZ);
    const std::size_t localIndex = static_cast<std::size_t>(overrideClusterIndex(localX, localY, localZ));

    if (generatedBlock(x, y, z) == block) {
        auto columnIt = m_overrideColumns.find(clusterX, clusterZ);
        if (columnIt != m_overrideColumns.end()) {
            auto clusterIt = columnIt->clusters.find(clusterY);
            if (clusterIt != columnIt->clusters.end()) {
                if (clusterIt->second.clear(localIndex)) {
                    --m_overrideCount;
                    if (clusterIt->second.isEmpty()) {
                        columnIt->clusters.erase(clusterIt);
                        if (columnIt->isEmpty())
                            m_overrideColumns.erase(columnIt);
                    }
                }
            }
        }
    } else {
        auto columnIt = m_overrideColumns.find(clusterX, clusterZ);
        if (columnIt == m_overrideColumns.end()) {
            m_overrideColumns.insert(clusterX, clusterZ, OverrideClusterColumn {});
            columnIt = m_overrideColumns.find(clusterX, clusterZ);
        }

        OverrideCluster& cluster = columnIt->clusters[clusterY];
        if (cluster.set(localIndex, block))
            ++m_overrideCount;
    }

    ++m_version;
}

bool World::isSolid(int32_t x, int32_t y, int32_t z) const { return getBlock(x, y, z) != Block::Air; }

OverrideCluster::Mask World::generatedSolidColumnMask(
    int32_t x, int32_t z, int32_t clusterY, int32_t localX, int32_t localZ, int32_t localMinY, int32_t localMaxY) const
{
    OverrideCluster::Mask mask = 0;
    const int32_t clusterOriginY = clusterY * OverrideCluster::Edge;
    for (int32_t localY = localMinY; localY <= localMaxY; ++localY) {
        if (generatedBlock(x, clusterOriginY + localY, z) == Block::Air)
            continue;
        mask |= OverrideCluster::Mask { 1 } << static_cast<std::size_t>(overrideClusterIndex(localX, localY, localZ));
    }
    return mask;
}

bool World::hasSolidBlockInArea(IVec3 min, IVec3 max) const
{
    m_blocksCache.waitIfPending();

    if (max.y < 0)
        return true;
    if (min.y >= Chunk::SizeY)
        return false;

    min.y = std::max(min.y, 0);
    max.y = std::min(max.y, Chunk::SizeY - 1);

    if (m_blocksCache.version == m_version && !m_blocksCache.blocks.empty() && min.x >= m_blocksCache.origin.x
        && min.y >= m_blocksCache.origin.y && min.z >= m_blocksCache.origin.z
        && max.x < m_blocksCache.origin.x + m_blocksCache.size.x
        && max.y < m_blocksCache.origin.y + m_blocksCache.size.y
        && max.z < m_blocksCache.origin.z + m_blocksCache.size.z) [[likely]]
        return cachedSolidBlockInArea(min, max);

    if (m_overrideCount == 0) {
        for (int32_t z = min.z; z <= max.z; ++z) {
            for (int32_t x = min.x; x <= max.x; ++x) {
                for (int32_t y = min.y; y <= max.y; ++y) {
                    if (generatedBlock(x, y, z) != Block::Air)
                        return true;
                }
            }
        }
        return false;
    }

    const int32_t startClusterY = floorDiv(min.y, OverrideCluster::Edge);
    const int32_t endClusterY = floorDiv(max.y, OverrideCluster::Edge);

    for (int32_t z = min.z; z <= max.z; ++z) {
        const int32_t clusterZ = floorDiv(z, OverrideCluster::Edge);
        const int32_t localZ = overrideClusterLocalCoord(z, clusterZ);
        for (int32_t x = min.x; x <= max.x; ++x) {
            const int32_t clusterX = floorDiv(x, OverrideCluster::Edge);
            const int32_t localX = overrideClusterLocalCoord(x, clusterX);
            const auto columnIt = m_overrideColumns.find(clusterX, clusterZ);

            for (int32_t clusterY = startClusterY; clusterY <= endClusterY; ++clusterY) {
                if (columnIt == m_overrideColumns.end())
                    break;

                const auto clusterIt = columnIt->clusters.find(clusterY);
                if (clusterIt == columnIt->clusters.end())
                    continue;

                const int32_t clusterOriginY = clusterY * OverrideCluster::Edge;
                const int32_t localMinY = std::max(0, min.y - clusterOriginY);
                const int32_t localMaxY = std::min(OverrideCluster::Edge - 1, max.y - clusterOriginY);
                const OverrideCluster::Mask queryMask = overrideClusterColumnMask(localX, localZ, localMinY, localMaxY);
                if (clusterIt->second.hasSolidOverrideInMask(queryMask))
                    return true;
            }

            for (int32_t clusterY = startClusterY; clusterY <= endClusterY; ++clusterY) {
                const int32_t clusterOriginY = clusterY * OverrideCluster::Edge;
                const int32_t localMinY = std::max(0, min.y - clusterOriginY);
                const int32_t localMaxY = std::min(OverrideCluster::Edge - 1, max.y - clusterOriginY);
                const OverrideCluster::Mask generatedMask
                    = generatedSolidColumnMask(x, z, clusterY, localX, localZ, localMinY, localMaxY);
                if (generatedMask == 0)
                    continue;

                OverrideCluster::Mask airOverrideMask = 0;
                if (columnIt != m_overrideColumns.end()) {
                    const auto clusterIt = columnIt->clusters.find(clusterY);
                    if (clusterIt != columnIt->clusters.end()) {
                        const OverrideCluster& cluster = clusterIt->second;
                        airOverrideMask = (cluster.overrideMask() & ~cluster.solidMask()) & generatedMask;
                    }
                }
                if ((generatedMask & ~airOverrideMask) != 0)
                    return true;
            }
        }
    }

    return false;
}

bool World::cachedSolidBlockInArea(IVec3 min, IVec3 max) const
{
    const IVec3 localMin = min - m_blocksCache.origin;
    const IVec3 localMax = max - m_blocksCache.origin;
    for (int32_t y = localMin.y; y <= localMax.y; ++y) {
        for (int32_t z = localMin.z; z <= localMax.z; ++z) {
            for (int32_t x = localMin.x; x <= localMax.x; ++x) {
                if (m_blocksCache.blocks[denseBlockIndex({ x, y, z }, m_blocksCache.size)]
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
    const int32_t wx = floorToInt32(x);
    const int32_t wz = floorToInt32(z);
    for (int32_t y = Chunk::SizeY - 1; y >= 0; --y) {
        if (isSolid(wx, y, wz))
            return static_cast<float>(y + 1);
    }
    return 0.0f;
}

std::vector<IVec3> World::visibleBlocksNear(Vec3 center, int32_t radius) const
{
    std::vector<IVec3> blocks;
    const int32_t cx = floorToInt32(center.x);
    const int32_t cy = floorToInt32(center.y);
    const int32_t cz = floorToInt32(center.z);
    blocks.reserve(static_cast<std::size_t>((radius * 2 + 1) * (radius * 2 + 1) * 8));

    for (int32_t z = cz - radius; z <= cz + radius; ++z) {
        for (int32_t y = std::max(0, cy - radius); y <= std::min(Chunk::SizeY - 1, cy + radius); ++y) {
            for (int32_t x = cx - radius; x <= cx + radius; ++x) {
                if (!isSolid(x, y, z))
                    continue;
                if (!isSolid(x + 1, y, z) || !isSolid(x - 1, y, z) || !isSolid(x, y + 1, z) || !isSolid(x, y - 1, z)
                    || !isSolid(x, y, z + 1) || !isSolid(x, y, z - 1)) {
                    blocks.push_back({ x, y, z });
                }
            }
        }
    }

    return blocks;
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

Block World::generatedBlock(int32_t x, int32_t y, int32_t z) const
{
    if (y < 0)
        return Block::Stone;

    if (y >= Chunk::SizeY)
        return Block::Air;

    const int32_t height = std::clamp(static_cast<int32_t>(terrainHeight(x, z)), 2, Chunk::SizeY - 2);
    if (y > height)
        return Block::Air;

    if (y == height)
        return Block::Grass;

    if (y > height - 4)
        return Block::Dirt;

    return Block::Stone;
}

float World::terrainHeight(int32_t x, int32_t z) const
{
    const int32_t sampleX = x + seedOffset(m_seed, 0x4f1bbc21U);
    const int32_t sampleZ = z + seedOffset(m_seed, 0x9a7c15d3U);
    const float low = std::sin(static_cast<float>(sampleX) * 0.17f) * 2.2f;
    const float high = std::cos(static_cast<float>(sampleZ) * 0.13f) * 1.8f;
    const float diagonal = std::sin(static_cast<float>(sampleX + sampleZ) * 0.08f) * 2.0f;
    const float rough = valueNoise(m_seed, sampleX / 3, sampleZ / 3) * 0.55f;
    return 9.0f + low + high + diagonal + rough;
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
