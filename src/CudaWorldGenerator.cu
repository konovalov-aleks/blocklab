#include "blocklab/CudaWorldGenerator.h"

#include "blocklab/CudaHelpers.h"
#include "blocklab/Hash.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace blocklab {

struct alignas(16) Voxel {
    __device__ Voxel(int3 pos, uint8_t material, uint8_t visibleFaces)
        : position(pos)
        , blockTypeAndVisibleFaces((static_cast<uint32_t>(material) << 8) | visibleFaces)
    {
    }

    int3 position;
    uint32_t blockTypeAndVisibleFaces; // [blockType:8 bits][visible faces: 8 bits]    };
};
static_assert(sizeof(Voxel) == VoxelSize);

namespace {

    constexpr int32_t ChunkSizeY = 32;
    constexpr int32_t CoordBias = 1 << 20;
    constexpr int64_t CoordMask = (int64_t { 1 } << 21) - 1;

    struct CudaOverride {
        int64_t key;
        uint8_t block;
    };

    struct CudaBuildParams {
        uint32_t seed;
        int32_t centerX;
        int32_t centerY;
        int32_t centerZ;
        int32_t width;
        int32_t height;
        int32_t depth;
        int32_t originX;
        int32_t originY;
        int32_t originZ;
        int32_t overrideCount;
        uint32_t maxVoxels;
    };

    int64_t packKeyHost(int32_t x, int32_t y, int32_t z)
    {
        return ((int64_t { x + CoordBias } & CoordMask) << 42) | ((int64_t { y + CoordBias } & CoordMask) << 21)
            | (int64_t { z + CoordBias } & CoordMask);
    }

    __device__ int64_t packKeyDevice(int32_t x, int32_t y, int32_t z)
    {
        return ((int64_t { x + CoordBias } & CoordMask) << 42) | ((int64_t { y + CoordBias } & CoordMask) << 21)
            | (int64_t { z + CoordBias } & CoordMask);
    }

    __device__ float valueNoise(uint32_t seed, int32_t x, int32_t z)
    {
        const uint32_t hash = hashCombine(seed, x, z);
        return static_cast<float>(hash & 0x00ffffffU) / static_cast<float>(0x00ffffffU) * 2.0f - 1.0f;
    }

    __device__ float terrainHeight(uint32_t seed, int32_t x, int32_t z)
    {
        const int32_t sampleX = x + seedOffset(seed, 0x4f1bbc21U);
        const int32_t sampleZ = z + seedOffset(seed, 0x9a7c15d3U);
        const float low = __sinf(static_cast<float>(sampleX) * 0.17f) * 2.2f;
        const float high = __cosf(static_cast<float>(sampleZ) * 0.13f) * 1.8f;
        const float diagonal = __sinf(static_cast<float>(sampleX + sampleZ) * 0.08f) * 2.0f;
        const float rough = valueNoise(seed, sampleX / 3, sampleZ / 3) * 0.55f;
        return 9.0f + low + high + diagonal + rough;
    }

    __device__ uint8_t generatedBlock(uint32_t seed, int32_t x, int32_t y, int32_t z)
    {
        if (y < 0)
            return BlockId::Stone;
        if (y >= ChunkSizeY)
            return BlockId::Air;

        int32_t height = static_cast<int32_t>(terrainHeight(seed, x, z));
        height = min(max(height, 2), ChunkSizeY - 2);
        if (y > height)
            return BlockId::Air;
        if (y == height)
            return BlockId::Grass;
        if (y > height - 4)
            return BlockId::Dirt;
        return BlockId::Stone;
    }

    __device__ uint8_t overriddenBlock(const CudaOverride* overrides, int32_t count, int32_t x, int32_t y, int32_t z)
    {
        const int64_t key = packKeyDevice(x, y, z);
        int32_t left = 0;
        int32_t right = count - 1;
        while (left <= right) {
            const int32_t mid = left + (right - left) / 2;
            const int64_t midKey = overrides[mid].key;
            if (midKey == key)
                return overrides[mid].block;
            if (midKey < key)
                left = mid + 1;
            else
                right = mid - 1;
        }
        return BlockId::NoOverride;
    }

    __device__ uint8_t blockAt(
        const CudaBuildParams params, const CudaOverride* overrides, int32_t x, int32_t y, int32_t z)
    {
        const uint8_t overrideBlock = overriddenBlock(overrides, params.overrideCount, x, y, z);
        if (overrideBlock != BlockId::NoOverride)
            return overrideBlock;
        return generatedBlock(params.seed, x, y, z);
    }

    __global__ void buildTerrainKernel(
        CudaBuildParams params, const CudaOverride* overrides, Voxel* voxels, uint32_t* voxelCount, uint8_t* blocks)
    {
        const int32_t index = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
        const int32_t volume = params.width * params.height * params.depth;
        if (index >= volume)
            return;

        const int32_t localX = index % params.width;
        const int32_t localY = (index / params.width) % params.height;
        const int32_t localZ = index / (params.width * params.height);
        const int32_t x = params.originX + localX;
        const int32_t y = params.originY + localY;
        const int32_t z = params.originZ + localZ;

        const uint8_t block = blockAt(params, overrides, x, y, z);
        blocks[index] = block;
        if (block == BlockId::Air)
            return;

        const uint8_t rightBlock = blockAt(params, overrides, x + 1, y, z);
        const uint8_t leftBlock = blockAt(params, overrides, x - 1, y, z);
        const uint8_t topBlock = blockAt(params, overrides, x, y + 1, z);
        const uint8_t bottomBlock = blockAt(params, overrides, x, y - 1, z);
        const uint8_t frontBlock = blockAt(params, overrides, x, y, z + 1);
        const uint8_t backBlock = blockAt(params, overrides, x, y, z - 1);

        if (rightBlock != BlockId::Air && leftBlock != BlockId::Air && topBlock != BlockId::Air
            && bottomBlock != BlockId::Air && frontBlock != BlockId::Air && backBlock != BlockId::Air)
            return;

        uint8_t visibleFaces = 0;
        visibleFaces |= rightBlock == BlockId::Air ? VisibleFace::Right : 0;
        visibleFaces |= leftBlock == BlockId::Air ? VisibleFace::Left : 0;
        visibleFaces |= topBlock == BlockId::Air ? VisibleFace::Top : 0;
        visibleFaces |= bottomBlock == BlockId::Air ? VisibleFace::Bottom : 0;
        visibleFaces |= frontBlock == BlockId::Air ? VisibleFace::Front : 0;
        visibleFaces |= backBlock == BlockId::Air ? VisibleFace::Back : 0;

        const uint32_t voxelIndex = atomicAdd(voxelCount, 1);
        if (voxelIndex >= params.maxVoxels)
            return;

        voxels[voxelIndex] = Voxel(make_int3(x, y, z), block, visibleFaces);
    }

} // namespace

struct CudaWorldGenerator::State {
    CudaOverride* overrides = nullptr;
    uint32_t* voxelCount = nullptr;
    uint32_t* voxelCountHost = nullptr;
    cudaStream_t stream = nullptr;
    PageLockedVector<CudaOverride> packedOverrides;
    uint32_t overrideCapacity = 0;
    bool pending = false;
};

CudaWorldGenerator::CudaWorldGenerator()
    : m_state(std::make_unique<State>())
{
    cudaCheck(cudaStreamCreateWithFlags(&m_state->stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    cudaCheck(cudaMalloc(&m_state->voxelCount, sizeof(uint32_t)), "cudaMalloc voxelCount");
    cudaCheck(cudaMallocHost(&m_state->voxelCountHost, sizeof(uint32_t)), "cudaMallocHost voxelCount");
}

CudaWorldGenerator::~CudaWorldGenerator()
{
    if (!m_state)
        return;
    cudaStreamSynchronize(m_state->stream);
    cudaFree(m_state->overrides);
    cudaFree(m_state->voxelCount);
    cudaFreeHost(m_state->voxelCountHost);
    cudaStreamDestroy(m_state->stream);
}

cudaStream_t CudaWorldGenerator::stream() const
{
    assert(m_state);
    return m_state->stream;
}

CudaFuture<WorldGenerationOutput> CudaWorldGenerator::generate(
    const WorldGenerationInput& input, WorldGenerationBuffers&& buffers)
{
    const auto waitForPendingBuild = [this] {
        cudaCheck(cudaStreamSynchronize(m_state->stream), "cudaStreamSynchronize previous terrain mesh build");
        m_state->pending = false;
    };

    const int32_t width = input.halfExtent * 2;
    const int32_t height = ChunkSizeY;
    const int32_t depth = input.halfExtent * 2;
    const int32_t volume = width * height * depth;
    const auto blockCount = static_cast<std::size_t>(volume);
    // Do not resize the borrowed cache buffer in-place: it may still be part of an async
    // CUDA/Vulkan handoff chain.
    PageLockedVector<uint8_t> outBlocks;
    if (buffers.blocks.size() >= blockCount)
        outBlocks = std::move(buffers.blocks);
    outBlocks.uninitializedResize(blockCount);

    if (m_state->pending)
        waitForPendingBuild();

    PageLockedVector<CudaOverride>& packedOverrides = m_state->packedOverrides;
    packedOverrides.clear();
    packedOverrides.reserve(input.overrides.size());
    for (const BlockOverride& blockOverride : input.overrides) {
        packedOverrides.push_back({
            .key = packKeyHost(blockOverride.coord.x, blockOverride.coord.y, blockOverride.coord.z),
            .block = blockId(blockOverride.block),
        });
    }
    std::sort(packedOverrides.begin(), packedOverrides.end(),
        [](const CudaOverride& a, const CudaOverride& b) { return a.key < b.key; });

    if (m_state->overrideCapacity < packedOverrides.size()) {
        cudaFree(m_state->overrides);
        m_state->overrides = nullptr;
        m_state->overrideCapacity = static_cast<uint32_t>(std::max<std::size_t>(packedOverrides.size(), 1U));
        cudaCheck(
            cudaMalloc(&m_state->overrides, sizeof(CudaOverride) * m_state->overrideCapacity), "cudaMalloc overrides");
    }
    if (!packedOverrides.empty()) {
        cudaCheck(cudaMemcpyAsync(m_state->overrides, packedOverrides.data(),
                      sizeof(CudaOverride) * packedOverrides.size(), cudaMemcpyHostToDevice, m_state->stream),
            "cudaMemcpyAsync overrides");
    }

    cudaCheck(cudaMemsetAsync(m_state->voxelCount, 0, sizeof(uint32_t), m_state->stream), "cudaMemsetAsync voxelCount");
    const CudaBuildParams params {
        .seed = input.seed,
        .centerX = input.center.x,
        .centerY = input.center.y,
        .centerZ = input.center.z,
        .width = width,
        .height = height,
        .depth = depth,
        .originX = input.origin.x,
        .originY = 0,
        .originZ = input.origin.z,
        .overrideCount = static_cast<int32_t>(packedOverrides.size()),
        .maxVoxels = static_cast<uint32_t>(buffers.maxVoxelCount),
    };

    constexpr int32_t ThreadCount = 256;
    buildTerrainKernel<<<(volume + ThreadCount - 1) / ThreadCount, ThreadCount, 0, m_state->stream>>>(
        params, m_state->overrides, buffers.voxels, m_state->voxelCount, outBlocks.data());
    cudaCheck(cudaGetLastError(), "buildTerrainKernel");

    cudaCheck(cudaMemcpyAsync(m_state->voxelCountHost, m_state->voxelCount, sizeof(*m_state->voxelCountHost),
                  cudaMemcpyDeviceToHost, m_state->stream),
        "cudaMemcpyAsync voxelCount");

    m_state->pending = true;
    WorldGenerationOutput output;
    output.origin = input.origin;
    output.size = input.size;
    output.worldVersion = input.worldVersion;
    output.buffers = std::move(buffers);
    output.buffers.blocks = std::move(outBlocks);
    auto outputStorage = std::make_shared<WorldGenerationOutput>(std::move(output));
    return CudaFuture<WorldGenerationOutput>(m_state->stream, [state = m_state.get(), outputStorage]() mutable {
        state->pending = false;
        outputStorage->voxelCount
            = std::min(*state->voxelCountHost, static_cast<uint32_t>(outputStorage->buffers.maxVoxelCount));
        return std::move(*outputStorage);
    });
}

} // namespace blocklab
