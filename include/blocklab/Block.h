#pragma once

#include "blocklab/Math.h"

#include <array>
#include <cstdint>
#include <limits>

namespace blocklab {

enum class Block : uint8_t {
    Air = 0,
    Grass,
    Dirt,
    Stone,

    COUNT
};

struct BlockId {
    static constexpr uint8_t Air = static_cast<uint8_t>(Block::Air);
    static constexpr uint8_t Grass = static_cast<uint8_t>(Block::Grass);
    static constexpr uint8_t Dirt = static_cast<uint8_t>(Block::Dirt);
    static constexpr uint8_t Stone = static_cast<uint8_t>(Block::Stone);
    static constexpr uint8_t NoOverride = std::numeric_limits<uint8_t>::max();
};

constexpr uint8_t blockId(Block block) { return static_cast<uint8_t>(block); }

struct BlockOverride {
    IVec3 coord {};
    Block block = Block::Air;
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
    std::array<Block, Volume> m_blocks {};
};

} // namespace blocklab
