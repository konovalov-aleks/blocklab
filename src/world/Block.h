#pragma once

#include <blocklab/gpu/cuda/CudaHelpers.h>
#include <blocklab/utility/Math.h>

#include <array>
#include <cstdint>

namespace blocklab {

enum class Block : std::uint8_t {
    Air = 0,
    Grass,
    Dirt,
    Stone,
    Torch,

    COUNT
};

struct BlockInfo {
    Block blockType;
    // TODO use more compact representation
    // std::uint8_t blockLight : 4;
    // std::uint8_t skyLight : 4;
    std::uint32_t blockLight;
    std::uint32_t skyLight;
};

constexpr BLOCKLAB_HOST_DEVICE bool isSolidBlock(Block block) { return block != Block::Air && block != Block::Torch; }
constexpr BLOCKLAB_HOST_DEVICE bool isSolidBlock(const BlockInfo& block) { return isSolidBlock(block.blockType); }

constexpr BLOCKLAB_HOST_DEVICE bool isOpaqueBlock(Block block) { return block != Block::Air && block != Block::Torch; }
constexpr BLOCKLAB_HOST_DEVICE bool isOpaqueBlock(const BlockInfo& block) { return isOpaqueBlock(block.blockType); }

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
