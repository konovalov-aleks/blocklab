#include <blocklab/world/CudaWorldGenerator.h>

#include <blocklab/gpu/cuda/CudaHelpers.h>
#include <blocklab/utility/Hash.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace blocklab {

class Voxel {
public:
    __device__ Voxel(uint3 pos, std::uint8_t blockType, std::uint8_t visibleFaces)
        : m_data(blockType | (visibleFaces << 5) | (pos.z << 11) | (pos.y << 17) | (pos.x << 26))
    {
        assert((blockType & ((1 << 5) - 1)) == blockType);
        assert((visibleFaces & ((1 << 6) - 1)) == visibleFaces);
        assert((pos.z & ((1 << 6) - 1)) == pos.z);
        assert((pos.y & ((1 << 9) - 1)) == pos.y);
        assert((pos.x & ((1 << 6) - 1)) == pos.x);
    }

private:
    // x : 6
    // y : 9
    // z : 6
    // visibleFacesMask: 6
    // blockType: 5
    std::uint32_t m_data;
};
static_assert(sizeof(Voxel) == VoxelSize);
static_assert(static_cast<std::uint8_t>(Block::COUNT) <= (1 << 5));

namespace {

    constexpr std::int32_t ChunkSizeY = 32;
    constexpr std::int32_t CoordBias = 1 << 20;
    constexpr std::int64_t CoordMask = (std::int64_t { 1 } << 21) - 1;

    struct CudaOverride {
        std::int64_t key;
        std::uint8_t block;
    };

    struct CudaBuildParams {
        std::uint32_t seed;
        std::int32_t centerX;
        std::int32_t centerY;
        std::int32_t centerZ;
        std::int32_t width;
        std::int32_t height;
        std::int32_t depth;
        std::int32_t originX;
        std::int32_t originY;
        std::int32_t originZ;
        std::int32_t overrideCount;
        std::uint32_t maxVoxels;
    };

    std::int64_t packKeyHost(std::int32_t x, std::int32_t y, std::int32_t z)
    {
        return ((std::int64_t { x + CoordBias } & CoordMask) << 42)
            | ((std::int64_t { y + CoordBias } & CoordMask) << 21) | (std::int64_t { z + CoordBias } & CoordMask);
    }

    __device__ std::int64_t packKeyDevice(std::int32_t x, std::int32_t y, std::int32_t z)
    {
        return ((std::int64_t { x + CoordBias } & CoordMask) << 42)
            | ((std::int64_t { y + CoordBias } & CoordMask) << 21) | (std::int64_t { z + CoordBias } & CoordMask);
    }

    __device__ float valueNoise(std::uint32_t seed, std::int32_t x, std::int32_t z)
    {
        const std::uint32_t hash = hashCombine(seed, x, z);
        return static_cast<float>(hash & 0x00ffffffU) / static_cast<float>(0x00ffffffU) * 2.0f - 1.0f;
    }

    __device__ float terrainHeight(std::uint32_t seed, std::int32_t x, std::int32_t z)
    {
        const std::int32_t sampleX = x + seedOffset(seed, 0x4f1bbc21U);
        const std::int32_t sampleZ = z + seedOffset(seed, 0x9a7c15d3U);
        const float low = __sinf(static_cast<float>(sampleX) * 0.17f) * 2.2f;
        const float high = __cosf(static_cast<float>(sampleZ) * 0.13f) * 1.8f;
        const float diagonal = __sinf(static_cast<float>(sampleX + sampleZ) * 0.08f) * 2.0f;
        const float rough = valueNoise(seed, sampleX / 3, sampleZ / 3) * 0.55f;
        return 9.0f + low + high + diagonal + rough;
    }

    __device__ std::uint8_t generatedBlock(std::uint32_t seed, std::int32_t x, std::int32_t y, std::int32_t z)
    {
        if (y < 0)
            return BlockId::Stone;
        if (y >= ChunkSizeY)
            return BlockId::Air;

        std::int32_t height = static_cast<std::int32_t>(terrainHeight(seed, x, z));
        height = min(max(height, 2), ChunkSizeY - 2);
        if (y > height)
            return BlockId::Air;
        if (y == height)
            return BlockId::Grass;
        if (y > height - 4)
            return BlockId::Dirt;
        return BlockId::Stone;
    }

    __device__ std::uint8_t overriddenBlock(
        const CudaOverride* overrides, std::int32_t count, std::int32_t x, std::int32_t y, std::int32_t z)
    {
        const std::int64_t key = packKeyDevice(x, y, z);
        std::int32_t left = 0;
        std::int32_t right = count - 1;
        while (left <= right) {
            const std::int32_t mid = left + (right - left) / 2;
            const std::int64_t midKey = overrides[mid].key;
            if (midKey == key)
                return overrides[mid].block;
            if (midKey < key)
                left = mid + 1;
            else
                right = mid - 1;
        }
        return BlockId::NoOverride;
    }

    __device__ std::uint8_t blockAt(
        const CudaBuildParams params, const CudaOverride* overrides, std::int32_t x, std::int32_t y, std::int32_t z)
    {
        const std::uint8_t overrideBlock = overriddenBlock(overrides, params.overrideCount, x, y, z);
        if (overrideBlock != BlockId::NoOverride)
            return overrideBlock;
        return generatedBlock(params.seed, x, y, z);
    }

    __global__ void buildTerrainKernel(CudaBuildParams params, const CudaOverride* overrides, TerrainHeader* header,
        Voxel* voxels, std::uint32_t* voxelCount, std::uint8_t* blocks)
    {
        const std::int32_t index = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
        if (index == 0) {
            header->originX = params.originX;
            header->originZ = params.originZ;
        }

        const std::int32_t volume = params.width * params.height * params.depth;
        if (index >= volume)
            return;

        const std::int32_t localX = index % params.width;
        const std::int32_t localY = (index / params.width) % params.height;
        const std::int32_t localZ = index / (params.width * params.height);
        const std::int32_t x = params.originX + localX;
        const std::int32_t y = params.originY + localY;
        const std::int32_t z = params.originZ + localZ;

        const std::uint8_t block = blockAt(params, overrides, x, y, z);
        blocks[index] = block;
        if (block == BlockId::Air)
            return;

        const std::uint8_t rightBlock = blockAt(params, overrides, x + 1, y, z);
        const std::uint8_t leftBlock = blockAt(params, overrides, x - 1, y, z);
        const std::uint8_t topBlock = blockAt(params, overrides, x, y + 1, z);
        const std::uint8_t bottomBlock = blockAt(params, overrides, x, y - 1, z);
        const std::uint8_t frontBlock = blockAt(params, overrides, x, y, z + 1);
        const std::uint8_t backBlock = blockAt(params, overrides, x, y, z - 1);

        if (rightBlock != BlockId::Air && leftBlock != BlockId::Air && topBlock != BlockId::Air
            && bottomBlock != BlockId::Air && frontBlock != BlockId::Air && backBlock != BlockId::Air)
            return;

        std::uint8_t visibleFaces = 0;
        visibleFaces |= rightBlock == BlockId::Air ? VisibleFace::Right : 0;
        visibleFaces |= leftBlock == BlockId::Air ? VisibleFace::Left : 0;
        visibleFaces |= topBlock == BlockId::Air ? VisibleFace::Top : 0;
        visibleFaces |= bottomBlock == BlockId::Air ? VisibleFace::Bottom : 0;
        visibleFaces |= frontBlock == BlockId::Air ? VisibleFace::Front : 0;
        visibleFaces |= backBlock == BlockId::Air ? VisibleFace::Back : 0;

        const std::uint32_t voxelIndex = atomicAdd(voxelCount, 1);
        if (voxelIndex >= params.maxVoxels)
            return;

        voxels[voxelIndex] = Voxel(make_uint3(localX, localY, localZ), block, visibleFaces);
    }

} // namespace

struct CudaWorldGenerator::State {
    CudaOverride* overrides = nullptr;
    std::uint32_t* voxelCount = nullptr;
    std::uint32_t* voxelCountHost = nullptr;
    cudaStream_t stream = nullptr;
    PageLockedVector<CudaOverride> packedOverrides;
    std::uint32_t overrideCapacity = 0;
    bool pending = false;
};

CudaWorldGenerator::CudaWorldGenerator()
    : m_state(std::make_unique<State>())
{
    cudaCheck(cudaStreamCreateWithFlags(&m_state->stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    cudaCheck(cudaMalloc(&m_state->voxelCount, sizeof(m_state->voxelCount)), "cudaMalloc voxelCount");
    cudaCheck(cudaMallocHost(&m_state->voxelCountHost, sizeof(m_state->voxelCountHost)), "cudaMallocHost voxelCount");
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

    const std::int32_t width = input.halfExtent * 2;
    const std::int32_t height = ChunkSizeY;
    const std::int32_t depth = input.halfExtent * 2;
    const std::int32_t volume = width * height * depth;
    const auto blockCount = static_cast<std::size_t>(volume);
    // Do not resize the borrowed cache buffer in-place: it may still be part of an async
    // CUDA/Vulkan handoff chain.
    PageLockedVector<std::uint8_t> outBlocks;
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
        m_state->overrideCapacity = static_cast<std::uint32_t>(std::max<std::size_t>(packedOverrides.size(), 1U));
        cudaCheck(
            cudaMalloc(&m_state->overrides, sizeof(CudaOverride) * m_state->overrideCapacity), "cudaMalloc overrides");
    }
    if (!packedOverrides.empty()) {
        cudaCheck(cudaMemcpyAsync(m_state->overrides, packedOverrides.data(),
                      sizeof(CudaOverride) * packedOverrides.size(), cudaMemcpyHostToDevice, m_state->stream),
            "cudaMemcpyAsync overrides");
    }

    cudaCheck(
        cudaMemsetAsync(m_state->voxelCount, 0, sizeof(std::uint32_t), m_state->stream), "cudaMemsetAsync voxelCount");
    const CudaBuildParams params {
        .seed = input.seed,
        .centerX = input.center.x,
        .centerY = input.center.y,
        .centerZ = input.center.z,
        .width = width,
        .height = height,
        .depth = depth,
        .originX = input.originX,
        .originY = 0,
        .originZ = input.originZ,
        .overrideCount = static_cast<std::int32_t>(packedOverrides.size()),
        .maxVoxels = static_cast<std::uint32_t>(buffers.maxVoxelCount),
    };

    constexpr std::int32_t ThreadCount = 256;
    buildTerrainKernel<<<(volume + ThreadCount - 1) / ThreadCount, ThreadCount, 0, m_state->stream>>>(
        params, m_state->overrides, buffers.header, buffers.voxels, m_state->voxelCount, outBlocks.data());
    cudaCheck(cudaGetLastError(), "buildTerrainKernel");

    cudaCheck(cudaMemcpyAsync(m_state->voxelCountHost, m_state->voxelCount, sizeof(*m_state->voxelCountHost),
                  cudaMemcpyDeviceToHost, m_state->stream),
        "cudaMemcpyAsync voxelCount");

    m_state->pending = true;
    WorldGenerationOutput output;
    output.origin = { input.originX, 0, input.originZ };
    output.size = input.size;
    output.worldVersion = input.worldVersion;
    output.buffers = std::move(buffers);
    output.buffers.blocks = std::move(outBlocks);
    auto outputStorage = std::make_shared<WorldGenerationOutput>(std::move(output));
    return CudaFuture<WorldGenerationOutput>(m_state->stream, [state = m_state.get(), outputStorage]() mutable {
        state->pending = false;
        outputStorage->voxelCount
            = std::min(*state->voxelCountHost, static_cast<std::uint32_t>(outputStorage->buffers.maxVoxelCount));
        return std::move(*outputStorage);
    });
}

} // namespace blocklab
