#pragma once

#include <blocklab/gpu/cuda/CudaHelpers.h>
#include <blocklab/utility/Math.h>

#include <cstdint>
#include <cstdlib>

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

// Surface slipperiness of a block, matching Minecraft: higher values mean less drag (more
// sliding), lower values mean more friction. Air has slipperiness 1.0, which yields the plain
// 0.91 in-air drag, so a character standing on a slippery block (e.g. future ice) slides farther.
constexpr BLOCKLAB_HOST_DEVICE float slipperiness(Block block)
{
    switch (block) {
    case Block::Grass:
    case Block::Dirt:
    case Block::Stone:
    case Block::Torch:
        return 0.6f;
    case Block::Air:
        return 1.0f;
    case Block::COUNT:
        break;
    }
    [[unlikely]] std::abort(); // unreachable
}

struct BlockOverride {
    IVec3 coord {};
    Block block = Block::Air;
};

} // namespace blocklab
