#pragma once

#include "Block.h"
#include "WorldGenerator.h"

#include <blocklab/gpu/cuda/CudaSharedFuture.h>
#include <blocklab/utility/Math.h>
#include <characters/NPC.h>
#include <containers/QuadTree.h>
#include <gpu/cuda/PageLockedVector.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace blocklab {

class OverrideCluster {
public:
    using Mask = std::uint64_t;

    static constexpr std::int32_t Edge = 4;
    static constexpr std::int32_t Volume = Edge * Edge * Edge;
    static constexpr std::uint8_t NoOverride = BlockId::NoOverride;

    OverrideCluster();
    std::optional<Block> get(std::size_t index) const;
    bool set(std::size_t index, Block block);
    bool clear(std::size_t index);
    bool hasOverride(std::size_t index) const;
    bool hasSolidOverride(std::size_t index) const;
    bool hasOverrideInMask(Mask mask) const { return (m_overrideMask & mask) != 0; }
    bool hasSolidOverrideInMask(Mask mask) const { return (m_solidMask & mask) != 0; }
    std::uint16_t count() const { return m_count; }
    bool isEmpty() const { return m_count == 0; }
    Mask overrideMask() const { return m_overrideMask; }
    Mask solidMask() const { return m_solidMask; }

private:
    static Mask bitFor(std::size_t index);

    std::uint16_t m_count = 0;
    Mask m_overrideMask = 0;
    Mask m_solidMask = 0;
    std::array<std::uint8_t, Volume> m_blocks;
};

static_assert(OverrideCluster::Volume <= std::numeric_limits<OverrideCluster::Mask>::digits);
static_assert(sizeof(Block) == sizeof(std::uint8_t),
    "OverrideCluster stores dense std::uint8_t block ids. If Block becomes heavier, consider storing compact ids or "
    "pointers.");
static_assert(BlockId::Air != OverrideCluster::NoOverride);
static_assert(BlockId::Grass != OverrideCluster::NoOverride);
static_assert(BlockId::Dirt != OverrideCluster::NoOverride);
static_assert(BlockId::Stone != OverrideCluster::NoOverride);

struct OverrideClusterColumn {
    std::map<std::int32_t, OverrideCluster> clusters;

    bool isEmpty() const { return clusters.empty(); }
};

class World {
public:
    struct BlocksCache {
        enum class State : std::uint8_t {
            Empty,
            Ready,
            Borrowed,
            Pending,
        };

        BlocksCache() = default;
        ~BlocksCache() { waitIfPending(); }

        BlocksCache(const BlocksCache&) = delete;
        BlocksCache& operator=(const BlocksCache&) = delete;

        BlocksCache(BlocksCache&& other) noexcept
            : origin(other.origin)
            , size(other.size)
            , blocks(std::move(other.blocks))
            , pendingFuture(std::move(other.pendingFuture))
            , state(std::exchange(other.state, State::Ready))
        {
        }

        BlocksCache& operator=(BlocksCache&& other) noexcept
        {
            if (this == &other)
                return *this;

            waitIfPending();
            other.waitIfPending();
            origin = other.origin;
            size = other.size;
            blocks = std::move(other.blocks);
            pendingFuture = std::move(other.pendingFuture);
            state = std::exchange(other.state, State::Ready);
            return *this;
        }

        void clear()
        {
            waitIfPending();
            origin = {};
            size = {};
            blocks.clear();
            state = State::Empty;
        }

        void waitIfPending();

        IVec3 origin {};
        IVec3 size {};
        PageLockedVector<std::uint8_t> blocks;
        CudaSharedFuture<WorldGenerationOutput> pendingFuture;
        State state = State::Empty;
    };

    World() = default;

    void update(float dt, Vec3 threatPosition);

    // resets the world to the initial state with the given seed,
    // invalidating all block overrides and cached blocks
    void resetSeed(std::uint32_t seed);
    // spawns characters and resets their states, without modifying blocks or overrides
    // note: the block cache must be initialized
    void resetCharacters();

    Block getBlock(IVec3 pos) const;
    void setBlock(IVec3 pos, Block block);
    bool isSolid(IVec3 pos) const { return getBlock(pos) != Block::Air; }
    bool hasSolidBlockInArea(IVec3 min, IVec3 max) const;
    float groundHeight(float x, float z) const;

    void collectOverridesInRegion(IVec3 origin, IVec3 size, std::vector<BlockOverride>& out) const;

    PageLockedVector<std::uint8_t> borrowGenerationBuffers() const;
    void updateGeneration(CudaSharedFuture<WorldGenerationOutput>) const;
    void waitForGeneration() const;

    std::uint32_t seed() const { return m_seed; }
    std::uint64_t version() const { return m_version; }
    std::uint64_t logicalTimeMs() const { return m_logicalTimeMs; }

    std::size_t overrideCount() const { return m_overrideCount; }

    const std::vector<std::unique_ptr<NPC>>& characters() const { return m_characters; }

private:
    bool isInsideCacheBounds(IVec3) const;
    bool cachedSolidBlockInArea(IVec3 min, IVec3 max) const;
    void updateCharacters(float dt, Vec3 threatPosition);
    void spawnTestPigs();

    std::uint32_t m_seed = 0;
    std::uint64_t m_version = 1;
    std::uint64_t m_logicalTimeMs = 0;
    std::size_t m_overrideCount = 0;
    EntityId m_nextEntityId = 1;
    QuadTree<OverrideClusterColumn> m_overrideColumns;
    std::vector<std::unique_ptr<NPC>> m_characters;

    mutable BlocksCache m_blockCache;
};

} // namespace blocklab
