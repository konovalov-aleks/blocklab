#include "blocklab/World.h"

#include <algorithm>
#include <cmath>

namespace blocklab {

namespace {

    uint32_t mixBits(uint32_t value)
    {
        value ^= value >> 16U;
        value *= 0x7feb352dU;
        value ^= value >> 15U;
        value *= 0x846ca68bU;
        value ^= value >> 16U;
        return value;
    }

    float valueNoise(uint32_t seed, int32_t x, int32_t z)
    {
        const uint32_t hash
            = mixBits(seed ^ static_cast<uint32_t>(x) * 0x9e3779b9U ^ static_cast<uint32_t>(z) * 0x85ebca6bU);
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

std::size_t BlockCoordHash::operator()(const BlockCoord& coord) const noexcept
{
    uint32_t hash = mixBits(static_cast<uint32_t>(coord.x));
    hash ^= mixBits(static_cast<uint32_t>(coord.y) + 0x9e3779b9U);
    hash ^= mixBits(static_cast<uint32_t>(coord.z) + 0x85ebca6bU);
    return static_cast<std::size_t>(hash);
}

OverrideCluster::OverrideCluster() { m_blocks.fill(NoOverride); }

std::optional<Block> OverrideCluster::get(std::size_t index) const
{
    const uint8_t stored = m_blocks[index];
    if (stored == NoOverride)
        return std::nullopt;
    return static_cast<Block>(stored);
}

bool OverrideCluster::set(std::size_t index, Block block)
{
    uint8_t& stored = m_blocks[index];
    const bool inserted = stored == NoOverride;
    if (inserted)
        ++m_count;
    stored = static_cast<uint8_t>(block);
    return inserted;
}

bool OverrideCluster::clear(std::size_t index)
{
    uint8_t& stored = m_blocks[index];
    if (stored == NoOverride)
        return false;

    stored = NoOverride;
    --m_count;
    return true;
}

World::World(uint32_t seed)
    : m_seed(seed)
{
}

void World::reset(uint32_t seed)
{
    m_seed = seed;
    m_overrideColumns.clear();
    m_overrideCount = 0;
    ++m_version;
}

Block World::getBlock(int32_t x, int32_t y, int32_t z) const
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
    return generatedBlock(x, y, z);
}

void World::setBlock(int32_t x, int32_t y, int32_t z, Block block)
{
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
            m_overrideColumns.insert(clusterX, clusterZ, OverrideClusterColumn { });
            columnIt = m_overrideColumns.find(clusterX, clusterZ);
        }

        OverrideCluster& cluster = columnIt->clusters[clusterY];
        if (cluster.set(localIndex, block))
            ++m_overrideCount;
    }

    ++m_version;
}

bool World::isSolid(int32_t x, int32_t y, int32_t z) const { return getBlock(x, y, z) != Block::Air; }

float World::groundHeight(float x, float z) const
{
    const int32_t wx = floorToInt(x);
    const int32_t wz = floorToInt(z);
    for (int32_t y = Chunk::SizeY - 1; y >= 0; --y) {
        if (isSolid(wx, y, wz))
            return static_cast<float>(y + 1);
    }
    return 0.0f;
}

std::vector<IVec3> World::visibleBlocksNear(Vec3 center, int32_t radius) const
{
    std::vector<IVec3> blocks;
    const int32_t cx = floorToInt(center.x);
    const int32_t cy = floorToInt(center.y);
    const int32_t cz = floorToInt(center.z);
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
                                .coord = {
                                    .x = clusterOrigin.x + x,
                                    .y = clusterOrigin.y + y,
                                    .z = clusterOrigin.z + z,
                                },
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
    const float low = std::sin((static_cast<float>(x) + static_cast<float>(m_seed) * 0.013f) * 0.17f) * 2.2f;
    const float high = std::cos((static_cast<float>(z) - static_cast<float>(m_seed) * 0.019f) * 0.13f) * 1.8f;
    const float diagonal = std::sin(static_cast<float>(x + z) * 0.08f + static_cast<float>(m_seed) * 0.001f) * 2.0f;
    const float rough = valueNoise(m_seed, x / 3, z / 3) * 0.55f;
    return 9.0f + low + high + diagonal + rough;
}

} // namespace blocklab
