#pragma once

#include "blocklab/Math.h"
#include "blocklab/QuadTree.h"
#include "blocklab/characters/NPC.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace blocklab {

enum class Block : uint8_t {
    Air = 0,
    Grass,
    Dirt,
    Stone,
};

class Chunk {
public:
    static constexpr int32_t SizeX = 16;
    static constexpr int32_t SizeY = 32;
    static constexpr int32_t SizeZ = 16;
    static constexpr int32_t Volume = SizeX * SizeY * SizeZ;

    Block get(int32_t x, int32_t y, int32_t z) const;
    void set(int32_t x, int32_t y, int32_t z, Block block);

private:
    std::array<Block, Volume> m_blocks { };
};

struct BlockCoord {
    int32_t x = 0;
    int32_t y = 0;
    int32_t z = 0;

    bool operator==(const BlockCoord& other) const { return x == other.x && y == other.y && z == other.z; }
};

struct BlockCoordHash {
    std::size_t operator()(const BlockCoord& coord) const noexcept;
};

class OverrideCluster {
public:
    using Mask = uint64_t;

    static constexpr int32_t Edge = 4;
    static constexpr int32_t Volume = Edge * Edge * Edge;
    static constexpr uint8_t NoOverride = std::numeric_limits<uint8_t>::max();

    OverrideCluster();
    std::optional<Block> get(std::size_t index) const;
    bool set(std::size_t index, Block block);
    bool clear(std::size_t index);
    bool hasOverride(std::size_t index) const;
    bool hasSolidOverride(std::size_t index) const;
    bool hasOverrideInMask(Mask mask) const { return (m_overrideMask & mask) != 0; }
    bool hasSolidOverrideInMask(Mask mask) const { return (m_solidMask & mask) != 0; }
    uint16_t count() const { return m_count; }
    bool isEmpty() const { return m_count == 0; }
    Mask overrideMask() const { return m_overrideMask; }
    Mask solidMask() const { return m_solidMask; }

private:
    static Mask bitFor(std::size_t index);

    uint16_t m_count = 0;
    Mask m_overrideMask = 0;
    Mask m_solidMask = 0;
    std::array<uint8_t, Volume> m_blocks;
};

static_assert(OverrideCluster::Volume <= std::numeric_limits<OverrideCluster::Mask>::digits);
static_assert(sizeof(Block) == sizeof(uint8_t),
    "OverrideCluster stores dense uint8_t block ids. If Block becomes heavier, consider storing compact ids or "
    "pointers.");
static_assert(static_cast<uint8_t>(Block::Air) != OverrideCluster::NoOverride);
static_assert(static_cast<uint8_t>(Block::Grass) != OverrideCluster::NoOverride);
static_assert(static_cast<uint8_t>(Block::Dirt) != OverrideCluster::NoOverride);
static_assert(static_cast<uint8_t>(Block::Stone) != OverrideCluster::NoOverride);

struct BlockOverride {
    BlockCoord coord;
    Block block = Block::Air;
};

struct OverrideClusterColumn {
    std::map<int32_t, OverrideCluster> clusters;

    bool isEmpty() const { return clusters.empty(); }
};

class World {
public:
    explicit World(uint32_t seed = 1);

    void reset(uint32_t seed);
    Block getBlock(int32_t x, int32_t y, int32_t z) const;
    void setBlock(int32_t x, int32_t y, int32_t z, Block block);
    void updateCharacters(float dt, Vec3 threatPosition);
    bool isSolid(int32_t x, int32_t y, int32_t z) const;
    bool hasSolidBlockInArea(IVec3 min, IVec3 max) const;
    float groundHeight(float x, float z) const;
    std::vector<IVec3> visibleBlocksNear(Vec3 center, int32_t radius) const;
    void collectOverridesInRegion(IVec3 origin, IVec3 size, std::vector<BlockOverride>& out) const;

    uint32_t seed() const { return m_seed; }
    std::size_t overrideCount() const { return m_overrideCount; }
    const std::vector<std::unique_ptr<NPC>>& characters() const { return m_characters; }
    uint64_t version() const { return m_version; }

private:
    uint32_t m_seed = 1;
    uint64_t m_version = 1;
    std::size_t m_overrideCount = 0;
    EntityId m_nextEntityId = 1;
    QuadTree<OverrideClusterColumn> m_overrideColumns;
    std::vector<std::unique_ptr<NPC>> m_characters;

    std::optional<Block> overriddenBlock(int32_t x, int32_t y, int32_t z) const;
    Block generatedBlock(int32_t x, int32_t y, int32_t z) const;
    OverrideCluster::Mask generatedSolidColumnMask(int32_t x, int32_t z, int32_t clusterY, int32_t localX,
        int32_t localZ, int32_t localMinY, int32_t localMaxY) const;
    float terrainHeight(int32_t x, int32_t z) const;
    void spawnTestPigs();
};

} // namespace blocklab
