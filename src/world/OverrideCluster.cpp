#include "OverrideCluster.h"

#include <cstddef>
#include <cstdint>

namespace blocklab {

OverrideCluster::OverrideCluster() { m_blocks.fill(NoOverride); }

OverrideCluster::Mask OverrideCluster::bitFor(std::size_t index) { return Mask { 1 } << index; }

bool OverrideCluster::hasOverride(std::size_t index) const { return (m_overrideMask & bitFor(index)) != 0; }

bool OverrideCluster::hasSolidOverride(std::size_t index) const { return (m_solidMask & bitFor(index)) != 0; }

std::optional<Block> OverrideCluster::get(std::size_t index) const
{
    if (!hasOverride(index))
        return std::nullopt;
    const std::uint8_t stored = m_blocks[index];
    return static_cast<Block>(stored);
}

bool OverrideCluster::set(std::size_t index, Block block)
{
    std::uint8_t& stored = m_blocks[index];
    const Mask bit = bitFor(index);
    const bool inserted = (m_overrideMask & bit) == 0;
    if (inserted)
        ++m_count;

    m_overrideMask |= bit;
    if (!isSolidBlock(block))
        m_solidMask &= ~bit;
    else
        m_solidMask |= bit;

    stored = static_cast<std::uint8_t>(block);
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


} // namespace blocklab
