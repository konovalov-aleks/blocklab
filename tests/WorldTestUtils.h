#pragma once

#include <world/Block.h>
#include <world/Voxel.h>

#include <blocklab/utility/Math.h>

#include <cstdint>
#include <initializer_list>
#include <string_view>
#include <vector>

namespace blocklab {

class World;

namespace test {

    enum class VoxelFace : std::uint32_t {
        Left = 0,
        Right = 1,
        Top = 2,
        Bottom = 3,
        Front = 4,
        Back = 5,
    };

    struct TestVoxel {
        std::uint32_t data;
        std::uint32_t blockLight;
        std::uint32_t skyLight;
    };
    static_assert(sizeof(TestVoxel) == VoxelSize);

    struct GeneratedVoxels {
        IVec3 origin;
        IVec3 size;
        std::vector<BlockInfo> blocks;
        std::vector<TestVoxel> voxels;
    };

    void updateWorldCacheAt(const World&, IVec3 center);
    GeneratedVoxels generateVoxelsAt(const World&, IVec3 center);

    void fillBlocks(World&, IVec3 origin, IVec3 size, Block);
    void clearBlocks(World&, IVec3 origin, IVec3 size);

    // Clears the [origin, origin + size) volume, then fills the horizontal XZ plane at world-space groundY with block.
    void generateFlatWorld(World&, IVec3 origin, IVec3 size, std::int32_t groundY, Block = Block::Stone);

    // Fills the [origin, origin + size) volume with wall blocks, then clears the inner
    // [origin + (1,1,1), origin + size - (1,1,1)) volume to air.
    void generateCave(World&, IVec3 origin, IVec3 size, Block wall = Block::Stone);

    Block voxelBlock(const TestVoxel&);
    const TestVoxel& requireVoxelAt(const GeneratedVoxels&, IVec3 pos);
    std::uint8_t blockLight(const TestVoxel&, VoxelFace);

    // Checks a block-light volume using one string layer per Y level. Inside each layer, rows are Z and columns are X.
    // Hex digits 0-F mean a non-solid block with the corresponding blockLight value, S means a solid block with
    // blockLight == 0, and . skips the cell.
    void checkBlockLight(
        const GeneratedVoxels&, IVec3 origin, IVec3 size, std::initializer_list<std::string_view> layers);

} // namespace test

} // namespace blocklab
