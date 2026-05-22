#pragma once

#include <cstdint>
#include <limits>

namespace blocklab {

enum class Block : uint8_t {
    Air = 0,
    Grass,
    Dirt,
    Stone,
};

struct BlockId {
    static constexpr uint8_t Air = static_cast<uint8_t>(Block::Air);
    static constexpr uint8_t Grass = static_cast<uint8_t>(Block::Grass);
    static constexpr uint8_t Dirt = static_cast<uint8_t>(Block::Dirt);
    static constexpr uint8_t Stone = static_cast<uint8_t>(Block::Stone);
    static constexpr uint8_t NoOverride = std::numeric_limits<uint8_t>::max();
};

constexpr uint8_t blockId(Block block) { return static_cast<uint8_t>(block); }

} // namespace blocklab
