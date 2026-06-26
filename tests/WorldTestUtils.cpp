#include "WorldTestUtils.h"

#include <blocklab/gpu/cuda/CudaHelpers.h>
#include <blocklab/gpu/cuda/CudaSharedFuture.h>
#include <world/World.h>
#include <world/WorldGenerator.h>

#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace blocklab::test {

namespace {

    constexpr std::int32_t TestGenerationHalfExtent = 32;
    constexpr std::uint32_t TestGenerationExtent = TestGenerationHalfExtent * 2;
    constexpr std::uint32_t TestMaxTerrainVoxels
        = static_cast<std::uint32_t>(TestGenerationExtent * Chunk::SizeY * TestGenerationExtent);

    std::size_t denseBlockIndex(IVec3 local, IVec3 size)
    {
        return static_cast<std::size_t>(local.x) + static_cast<std::size_t>(local.y) * static_cast<std::size_t>(size.x)
            + static_cast<std::size_t>(local.z) * static_cast<std::size_t>(size.x) * static_cast<std::size_t>(size.y);
    }

    const BlockInfo& generatedBlockAt(const GeneratedVoxels& generated, IVec3 pos)
    {
        const IVec3 local = pos - generated.origin;
        REQUIRE(local.x >= 0);
        REQUIRE(local.y >= 0);
        REQUIRE(local.z >= 0);
        REQUIRE(local.x < generated.size.x);
        REQUIRE(local.y < generated.size.y);
        REQUIRE(local.z < generated.size.z);
        return generated.blocks[denseBlockIndex(local, generated.size)];
    }

    std::uint8_t lightMaskValue(char c)
    {
        if (c >= '0' && c <= '9')
            return static_cast<std::uint32_t>(c - '0');
        if (c >= 'A' && c <= 'F')
            return static_cast<std::uint32_t>(c - 'A' + 10);
        if (c >= 'a' && c <= 'f')
            return static_cast<std::uint32_t>(c - 'a' + 10);

        FAIL("Unexpected light mask character");
        return 0;
    }

    IVec3 voxelPosition(const GeneratedVoxels& generated, const TestVoxel& voxel)
    {
        const std::int32_t localZ = static_cast<std::int32_t>((voxel.data >> 11U) & ((1U << 6U) - 1U));
        const std::int32_t localY = static_cast<std::int32_t>((voxel.data >> 17U) & ((1U << 9U) - 1U));
        const std::int32_t localX = static_cast<std::int32_t>((voxel.data >> 26U) & ((1U << 6U) - 1U));
        return generated.origin + IVec3 { localX, localY, localZ };
    }

} // namespace

void updateWorldCacheAt(const World& world, IVec3 center)
{
    WorldGenerator generator({ .halfExtent = TestGenerationHalfExtent });
    void* voxelMemory = nullptr;
    cudaCheck(cudaMalloc(&voxelMemory, VoxelSize * TestMaxTerrainVoxels), "cudaMalloc test terrain voxels");
    auto* const voxels = static_cast<Voxel*>(voxelMemory);
    TerrainHeader* terrainHeader = nullptr;
    cudaCheck(cudaMalloc(&terrainHeader, sizeof(TerrainHeader)), "cudaMalloc test terrain header");
    AgentState agent;
    agent.position = { static_cast<float>(center.x), static_cast<float>(center.y), static_cast<float>(center.z) };
    CudaSharedFuture<WorldGenerationOutput> generation = generator
                                                             .generate(world, agent,
                                                                 {
                                                                     .header = terrainHeader,
                                                                     .voxels = voxels,
                                                                     .maxVoxels = TestMaxTerrainVoxels,
                                                                     .blocks = world.borrowGenerationBuffers(),
                                                                 })
                                                             .share();
    world.updateGeneration(generation);
    world.waitForGeneration();
    cudaCheck(cudaFree(terrainHeader), "cudaFree test terrain header");
    cudaCheck(cudaFree(voxels), "cudaFree test terrain voxels");
}

GeneratedVoxels generateVoxelsAt(const World& world, IVec3 center)
{
    WorldGenerator generator({ .halfExtent = TestGenerationHalfExtent });
    void* voxelMemory = nullptr;
    cudaCheck(cudaMalloc(&voxelMemory, VoxelSize * TestMaxTerrainVoxels), "cudaMalloc test terrain voxels");
    auto* const voxels = static_cast<Voxel*>(voxelMemory);
    TerrainHeader* terrainHeader = nullptr;
    cudaCheck(cudaMalloc(&terrainHeader, sizeof(TerrainHeader)), "cudaMalloc test terrain header");
    AgentState agent;
    agent.position = { static_cast<float>(center.x), static_cast<float>(center.y), static_cast<float>(center.z) };
    CudaSharedFuture<WorldGenerationOutput> generation = generator
                                                             .generate(world, agent,
                                                                 {
                                                                     .header = terrainHeader,
                                                                     .voxels = voxels,
                                                                     .maxVoxels = TestMaxTerrainVoxels,
                                                                     .blocks = world.borrowGenerationBuffers(),
                                                                 })
                                                             .share();

    WorldGenerationOutput& output = generation.get();
    GeneratedVoxels result;
    result.origin = output.origin;
    result.size = output.size;
    result.blocks.assign(output.buffers.blocks.begin(), output.buffers.blocks.end());
    result.voxels.resize(output.voxelCount);
    cudaCheck(cudaMemcpy(result.voxels.data(), voxels, VoxelSize * result.voxels.size(), cudaMemcpyDeviceToHost),
        "cudaMemcpy test terrain voxels");

    world.updateGeneration(generation);
    world.waitForGeneration();
    cudaCheck(cudaFree(terrainHeader), "cudaFree test terrain header");
    cudaCheck(cudaFree(voxels), "cudaFree test terrain voxels");
    return result;
}

void fillBlocks(World& world, IVec3 origin, IVec3 size, Block block)
{
    for (std::int32_t y = 0; y < size.y; ++y) {
        for (std::int32_t z = 0; z < size.z; ++z) {
            for (std::int32_t x = 0; x < size.x; ++x)
                world.setBlock(origin + IVec3 { x, y, z }, block);
        }
    }
}

void clearBlocks(World& world, IVec3 origin, IVec3 size) { fillBlocks(world, origin, size, Block::Air); }

void generateFlatWorld(World& world, IVec3 origin, IVec3 size, std::int32_t groundY, Block block)
{
    clearBlocks(world, origin, size);
    for (std::int32_t z = 0; z < size.z; ++z) {
        for (std::int32_t x = 0; x < size.x; ++x)
            world.setBlock({ origin.x + x, groundY, origin.z + z }, block);
    }
}

void generateCave(World& world, IVec3 origin, IVec3 size, Block wall)
{
    fillBlocks(world, origin, size, wall);
    if (size.x <= 2 || size.y <= 2 || size.z <= 2)
        return;
    clearBlocks(world, origin + IVec3 { 1, 1, 1 }, size - IVec3 { 2, 2, 2 });
}

Block voxelBlock(const TestVoxel& voxel) { return static_cast<Block>(voxel.data & ((1U << 5U) - 1U)); }

const TestVoxel& requireVoxelAt(const GeneratedVoxels& generated, IVec3 pos)
{
    const TestVoxel* result = nullptr;
    for (const TestVoxel& voxel : generated.voxels) {
        if (voxelPosition(generated, voxel) == pos) {
            result = &voxel;
            break;
        }
    }
    REQUIRE(result != nullptr);
    return *result;
}

std::uint8_t blockLight(const TestVoxel& voxel, VoxelFace face)
{
    return static_cast<std::uint8_t>((voxel.blockLight >> (static_cast<std::uint32_t>(face) * 4U)) & 0xFU);
}

void checkBlockLight(
    const GeneratedVoxels& generated, IVec3 origin, IVec3 size, std::initializer_list<std::string_view> layers)
{
    REQUIRE(static_cast<std::int32_t>(layers.size()) == size.y);

    std::int32_t y = 0;
    for (std::string_view layer : layers) {
        std::int32_t x = 0;
        std::int32_t z = 0;
        for (char c : layer) {
            if (c == '\r')
                continue;
            if (c == '\n') {
                REQUIRE(x == size.x);
                x = 0;
                ++z;
                continue;
            }

            REQUIRE(x < size.x);
            REQUIRE(z < size.z);
            const IVec3 pos = origin + IVec3 { x, y, z };
            const BlockInfo& block = generatedBlockAt(generated, pos);
            INFO("pos=(" << pos.x << ", " << pos.y << ", " << pos.z << "), mask='" << c << "'");
            if (c == 'S') {
                CHECK(isSolidBlock(block));
                CHECK(block.blockLight == 0);
            } else if (c != '.') {
                CHECK(!isSolidBlock(block));
                CHECK(block.blockLight == lightMaskValue(c));
            }
            ++x;
        }

        if (x) {
            REQUIRE(x == size.x);
            ++z;
        }
        REQUIRE(z == size.z);
        ++y;
    }
}

} // namespace blocklab::test
