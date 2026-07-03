#pragma once

#include "Block.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

namespace blocklab {

class OverrideCluster {
public:
    using Mask = std::uint64_t;
    using BlockTypeT = std::underlying_type_t<Block>;

    static constexpr std::int32_t Edge = 4;
    static constexpr std::int32_t Volume = Edge * Edge * Edge;
    static constexpr BlockTypeT NoOverride = std::numeric_limits<BlockTypeT>::max();

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
static_assert(static_cast<OverrideCluster::BlockTypeT>(Block::Air) != OverrideCluster::NoOverride);
static_assert(static_cast<OverrideCluster::BlockTypeT>(Block::Grass) != OverrideCluster::NoOverride);
static_assert(static_cast<OverrideCluster::BlockTypeT>(Block::Dirt) != OverrideCluster::NoOverride);
static_assert(static_cast<OverrideCluster::BlockTypeT>(Block::Stone) != OverrideCluster::NoOverride);
static_assert(static_cast<OverrideCluster::BlockTypeT>(Block::Torch) != OverrideCluster::NoOverride);

} // namespace blocklab
