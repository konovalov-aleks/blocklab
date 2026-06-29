#pragma once

#include "Block.h"
#include "OverrideCluster.h"
#include "WorldGenerator.h"
#include "WorldTime.h"

#include <blocklab/gpu/cuda/CudaSharedFuture.h>
#include <blocklab/utility/Math.h>
#include <characters/NPC.h>
#include <containers/QuadTree.h>
#include <gpu/cuda/PageLockedVector.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace blocklab {

class World {
public:
    static constexpr std::int32_t s_height = 32;
    static constexpr std::int32_t s_minY = 0;
    static constexpr std::int32_t s_maxY = s_minY + s_height - 1;
    static_assert(std::numeric_limits<HeightMapValueT>::min() <= s_minY);
    static_assert(std::numeric_limits<HeightMapValueT>::max() >= s_maxY);

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
    bool isSolid(IVec3 pos) const { return isSolidBlock(getBlock(pos)); }
    bool hasSolidBlockInArea(IVec3 min, IVec3 max) const;
    std::int32_t terrainHeight(IVec2 xz) const { return m_blockCache.terrainHeight(xz); }

    void collectOverridesInRegion(IVec3 origin, UVec3 size, std::vector<BlockOverride>& out) const;

    CPUCacheGenerationBuffers borrowGenerationBuffers() const { return m_blockCache.borrowGenerationBuffers(); }
    void updateGeneration(CudaSharedFuture<WorldGenerationOutput> gen) const { m_blockCache.update(std::move(gen)); }
    void waitForGeneration() const;

    std::uint32_t seed() const { return m_seed; }
    std::uint64_t version() const { return m_version; }

    std::uint64_t logicalTimeMs() const { return m_logicalTimeMs; }
    WorldTime dayTime() const { return (m_logicalTimeMs / s_tickPeriodMs + m_dayTimeShiftTicks) % s_ticksPerGameDay; }

    std::size_t overrideCount() const { return m_overrideCount; }

    const std::vector<std::unique_ptr<NPC>>& characters() const { return m_characters; }

    static constexpr bool isValidHeight(std::int32_t y) { return y >= s_minY && y <= s_maxY; }

private:
    using OverrideClusterColumn = std::map<std::int32_t, OverrideCluster>;

    class BlockCache {
    public:
        enum class State : std::uint8_t {
            Empty,
            Ready,
            Borrowed,
            Pending,
        };

        BlockCache() = default;
        ~BlockCache() { waitIfPending(); }

        BlockCache(const BlockCache&) = delete;
        BlockCache& operator=(const BlockCache&) = delete;

        BlockInfo& operator[](IVec3);
        std::int32_t terrainHeight(IVec2 xz);

        IVec3 size() const { return m_size; }
        IVec3 origin() const { return m_origin; }
        bool isInsideBounds(IVec3) const;

        bool empty() const { return m_state == State::Empty || m_blocks.empty(); }
        void clear();
        void waitIfPending();

        CPUCacheGenerationBuffers borrowGenerationBuffers();
        void update(CudaSharedFuture<WorldGenerationOutput>);

    private:
        std::size_t denseBlockIndex(IVec3 local) const;

        IVec3 m_origin = {};
        IVec3 m_size {};

        PageLockedVector<BlockInfo> m_blocks;
        PageLockedVector<HeightMapValueT> m_heightMap;

        CudaSharedFuture<WorldGenerationOutput> m_pendingFuture;
        State m_state = State::Empty;
    };

    void updateCharacters(float dt, Vec3 threatPosition);
    void spawnTestPigs();

    std::size_t m_overrideCount = 0;
    std::uint64_t m_version = 1;
    std::uint64_t m_logicalTimeMs = 0;
    std::uint32_t m_seed = 0;
    WorldTime m_dayTimeShiftTicks = 0;

    EntityId m_nextEntityId = 1;
    QuadTree<OverrideClusterColumn> m_overrideColumns;
    std::vector<std::unique_ptr<NPC>> m_characters;
    mutable BlockCache m_blockCache;
};

} // namespace blocklab
