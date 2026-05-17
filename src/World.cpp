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

World::World(uint32_t seed)
    : m_seed(seed)
{
}

void World::reset(uint32_t seed)
{
    m_seed = seed;
    m_overrides.clear();
    ++m_version;
}

Block World::getBlock(int32_t x, int32_t y, int32_t z) const
{
    const auto it = m_overrides.find({ x, y, z });
    if (it != m_overrides.end())
        return it->second;
    return generatedBlock(x, y, z);
}

void World::setBlock(int32_t x, int32_t y, int32_t z, Block block)
{
    if (y < 0 || y >= Chunk::SizeY)
        return;

    const BlockCoord coord { x, y, z };
    if (getBlock(x, y, z) == block)
        return;

    if (generatedBlock(x, y, z) == block)
        m_overrides.erase(coord);
    else
        m_overrides[coord] = block;

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
    for (const auto& [coord, block] : m_overrides) {
        if (coord.x < origin.x || coord.y < origin.y || coord.z < origin.z || coord.x >= origin.x + size.x
            || coord.y >= origin.y + size.y || coord.z >= origin.z + size.z) {
            continue;
        }
        out.push_back({
            .coord = coord,
            .block = block,
        });
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
