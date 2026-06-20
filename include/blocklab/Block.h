#pragma once

#include "blocklab/Math.h"

#include <array>
#include <cstdint>
#include <limits>

namespace blocklab {

enum class Block : std::uint8_t {
    Air = 0,
    Grass,
    Dirt,
    Stone,

    COUNT
};

struct BlockId {
    static constexpr std::uint8_t Air = static_cast<std::uint8_t>(Block::Air);
    static constexpr std::uint8_t Grass = static_cast<std::uint8_t>(Block::Grass);
    static constexpr std::uint8_t Dirt = static_cast<std::uint8_t>(Block::Dirt);
    static constexpr std::uint8_t Stone = static_cast<std::uint8_t>(Block::Stone);
    static constexpr std::uint8_t NoOverride = std::numeric_limits<std::uint8_t>::max();
};

constexpr std::uint8_t blockId(Block block) { return static_cast<std::uint8_t>(block); }

struct BlockOverride {
    IVec3 coord {};
    Block block = Block::Air;
};

class Chunk {
public:
    static constexpr std::int32_t SizeX = 16;
    static constexpr std::int32_t SizeY = 32;
    static constexpr std::int32_t SizeZ = 16;
    static constexpr std::int32_t Volume = SizeX * SizeY * SizeZ;

    Block get(std::int32_t x, std::int32_t y, std::int32_t z) const;
    void set(std::int32_t x, std::int32_t y, std::int32_t z, Block block);

private:
    std::array<Block, Volume> m_blocks {};
};

} // namespace blocklab
