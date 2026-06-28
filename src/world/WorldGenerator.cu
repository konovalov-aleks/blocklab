#include "WorldGenerator.h"

#include <blocklab/gpu/cuda/CudaHelpers.h>
#include <gpu/cuda/PageLockedVector.h>
#include <utility/Hash.h>

#include <cuda_runtime.h>
#include <thrust/optional.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

namespace blocklab {

using VoxelLighting = std::uint8_t[6];

static constexpr std::uint32_t s_maxLightSourcesPerGeneration = 512;

static constexpr std::uint32_t s_maxLightPropagationDistance = 14;

// 4 * N^2 + 2
static constexpr std::int32_t s_maxLightComputeQueueLength = 786;

class Voxel {
public:
    __device__ Voxel(uint3 pos, Block blockType, std::uint8_t visibleFaces, const VoxelLighting& blockLight,
        const VoxelLighting& skyLight)
        : m_data(static_cast<std::uint8_t>(blockType) | (visibleFaces << 5) | (pos.z << 11) | (pos.y << 17)
              | (pos.x << 26))
        , m_blockLight(packLighting(blockLight))
        , m_skyLight(packLighting(skyLight))
    {
        assert((static_cast<unsigned>(blockType) & ((1 << 5) - 1)) == static_cast<unsigned>(blockType));
        assert((visibleFaces & ((1 << 6) - 1)) == visibleFaces);
        assert((pos.z & ((1 << 6) - 1)) == pos.z);
        assert((pos.y & ((1 << 9) - 1)) == pos.y);
        assert((pos.x & ((1 << 6) - 1)) == pos.x);
    }

private:
    static __device__ std::uint32_t packLighting(const VoxelLighting& l)
    {
        std::uint32_t result = 0;
        for (std::size_t i = 0; i < 6; ++i) {
            assert((l[i] & 0xF) == l[i]);
            result |= (l[i] << (i * 4));
        }
        return result;
    }

    // x : 6
    // y : 9
    // z : 6
    // visibleFacesMask: 6
    // blockType: 5
    std::uint32_t m_data;
    std::uint32_t m_blockLight = 0;
    std::uint32_t m_skyLight = 0;
};
static_assert(sizeof(Voxel) == VoxelSize);
static_assert(static_cast<std::uint8_t>(Block::COUNT) <= (1 << 5));

namespace {

    constexpr std::int32_t CoordBias = 1 << 20;
    constexpr std::int64_t CoordMask = (std::int64_t { 1 } << 21) - 1;

    struct CudaOverride {
        std::int64_t key;
        Block block;
    };

    struct CudaBuildParams {
        std::uint32_t seed;
        std::int32_t centerX;
        std::int32_t centerY;
        std::int32_t centerZ;
        std::uint32_t width;
        std::uint32_t height;
        std::uint32_t depth;
        std::int32_t originX;
        std::int32_t originY;
        std::int32_t originZ;
        std::int32_t overrideCount;
        std::uint32_t maxVoxels;
        std::uint32_t maxLightSources;
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

    __device__ Block generatedBlock(std::uint32_t seed, std::int32_t x, std::int32_t y, std::int32_t z)
    {
        std::int32_t height = static_cast<std::int32_t>(terrainHeight(seed, x, z));
        if (y > height)
            return Block::Air;
        if (y == height)
            return Block::Grass;
        if (y > height - 4)
            return Block::Dirt;
        return Block::Stone;
    }

    __device__ thrust::optional<Block> overriddenBlock(
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
        return thrust::nullopt;
    }

    __device__ Block blockAt(
        const CudaBuildParams params, const CudaOverride* overrides, std::int32_t x, std::int32_t y, std::int32_t z)
    {
        if (thrust::optional<Block> overrideBlock = overriddenBlock(overrides, params.overrideCount, x, y, z))
            return *overrideBlock;
        return generatedBlock(params.seed, x, y, z);
    }

    __global__ void buildBlockCacheKernel(CudaBuildParams params, const CudaOverride* overrides, TerrainHeader* header,
        uchar3* lightSources, std::uint32_t* lightSourceCount, BlockInfo* blocks, HeightMapValueT* heightMap)
    {
        if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
            header->originX = params.originX;
            header->originY = params.originY;
            header->originZ = params.originZ;
        }

        const std::int32_t localY = threadIdx.x;
        const std::int32_t localX = blockIdx.x;
        const std::int32_t localZ = blockIdx.y;
        const std::int32_t index = localX + params.width * (localY + params.height * localZ);

        const std::int32_t x = params.originX + localX;
        const std::int32_t y = params.originY + localY;
        const std::int32_t z = params.originZ + localZ;

        const Block block = blockAt(params, overrides, x, y, z);

        blocks[index].blockType = block;
        blocks[index].blockLight = 0;
        blocks[index].skyLight = 0;

        if (block == Block::Torch) {
            const std::uint32_t lightSourceIndex = atomicAdd(lightSourceCount, 1);
            if (lightSourceIndex < params.maxLightSources)
                lightSources[lightSourceIndex] = make_uchar3(localX, localY, localZ);
        }

        __shared__ std::int32_t maxHeight;
        if (threadIdx.x == 0)
            maxHeight = params.originY - 1;
        __syncthreads();

        if (maxHeight < y && isSolidBlock(block))
            atomicMax(&maxHeight, y);
        __syncthreads();

        if (threadIdx.x == 0)
            heightMap[localX + params.width * localZ] = maxHeight;
    }

    __global__ void buildTerrainKernel(
        CudaBuildParams params, Voxel* voxels, std::uint32_t* voxelCount, BlockInfo* blocks)
    {
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        const std::int32_t volume = params.width * params.height * params.depth;
        if (index >= volume)
            return;

        const std::int32_t localX = index % params.width;
        const std::int32_t localY = (index / params.width) % params.height;
        const std::int32_t localZ = index / (params.width * params.height);

        const Block block = blocks[index].blockType;
        if (block == Block::Air)
            return;

        std::uint8_t visibleFaces = 0;
        VoxelLighting blockLight;
        const VoxelLighting skyLight = {};

        if (block == Block::Torch) {
            visibleFaces
                = VisibleFace::Right | VisibleFace::Left | VisibleFace::Top | VisibleFace::Front | VisibleFace::Back;
            std::memset(blockLight, 15, sizeof(blockLight));
        } else {
            const auto blockAt = [&params, blocks](int x, int y, int z) -> BlockInfo {
                if (x < 0 || y < 0 || z < 0 || x >= params.width || y >= params.height || z >= params.depth) {
                    // TODO: Revisit unloaded-neighbor semantics when terrain is generated as multiple resident chunks.
                    // Returning air renders boundary faces, which is useful for chunk seams, but creates currently
                    // hidden extra faces while the generated region moves with the player.
                    return {
                        .blockType = Block::Air,
                        .blockLight = 0,
                        .skyLight = 0,
                    };
                }
                return blocks[x + params.width * (y + params.height * z)];
            };

            const BlockInfo& leftBlock = blockAt(localX - 1, localY, localZ);
            const BlockInfo& rightBlock = blockAt(localX + 1, localY, localZ);
            const BlockInfo& topBlock = blockAt(localX, localY + 1, localZ);
            const BlockInfo& bottomBlock = blockAt(localX, localY - 1, localZ);
            const BlockInfo& frontBlock = blockAt(localX, localY, localZ + 1);
            const BlockInfo& backBlock = blockAt(localX, localY, localZ - 1);

            if (isOpaqueBlock(rightBlock) && isOpaqueBlock(leftBlock) && isOpaqueBlock(topBlock)
                && isOpaqueBlock(bottomBlock) && isOpaqueBlock(frontBlock) && isOpaqueBlock(backBlock))
                return;

            visibleFaces |= !isOpaqueBlock(leftBlock) ? VisibleFace::Left : 0;
            visibleFaces |= !isOpaqueBlock(rightBlock) ? VisibleFace::Right : 0;
            visibleFaces |= !isOpaqueBlock(topBlock) ? VisibleFace::Top : 0;
            visibleFaces |= !isOpaqueBlock(bottomBlock) ? VisibleFace::Bottom : 0;
            visibleFaces |= !isOpaqueBlock(frontBlock) ? VisibleFace::Front : 0;
            visibleFaces |= !isOpaqueBlock(backBlock) ? VisibleFace::Back : 0;

            blockLight[0] = leftBlock.blockLight;
            blockLight[1] = rightBlock.blockLight;
            blockLight[2] = topBlock.blockLight;
            blockLight[3] = bottomBlock.blockLight;
            blockLight[4] = frontBlock.blockLight;
            blockLight[5] = backBlock.blockLight;
        }

        const std::uint32_t voxelIndex = atomicAdd(voxelCount, 1);
        if (voxelIndex >= params.maxVoxels)
            return;

        voxels[voxelIndex] = Voxel(make_uint3(localX, localY, localZ), block, visibleFaces, blockLight, skyLight);
    }

    struct LightingBlockCache {
        // isSolid : 1
        // light: 7

        void __device__ init(bool solid) { data = solid << 7; }

        bool __device__ isSolid() { return data & 0x80; }
        std::uint8_t __device__ light() { return light(data); }

        void __device__ setLight(std::uint8_t light) { data = withLight(data, light); }

        __device__ bool atomicSetLightIfGreater(std::uint8_t newLight)
        {
            // CUDA has no atomicCAS for uint8_t
            // this is a workaround via uint32_t CAS operation

            auto wordAddr
                = reinterpret_cast<unsigned int*>(reinterpret_cast<std::uintptr_t>(&data) & ~std::uintptr_t(3));

            const unsigned int byteOffset = reinterpret_cast<std::uintptr_t>(&data) & 3;

            const unsigned int shift = byteOffset * 8;
            const unsigned int mask = 0xFFu << shift;

            std::int32_t oldWord = *wordAddr;

            for (;;) {
                std::uint8_t oldByte = (oldWord >> shift) & 0xFFu;
                if (light(oldByte) >= newLight)
                    return false;

                std::uint8_t newByte = withLight(oldByte, newLight);
                std::uint32_t newWord = (oldWord & ~mask) | (std::uint32_t(newByte) << shift);

                std::uint32_t prev = atomicCAS(wordAddr, oldWord, newWord);
                if (prev == oldWord)
                    return true;
                oldWord = prev;
            }
        }

        static __device__ std::uint8_t light(std::uint8_t data) { return data & 0x7f; }

        static __device__ std::uint8_t withLight(std::uint8_t d, std::uint8_t newLight)
        {
            return (d & 0x80) | newLight;
        }

        std::uint8_t data;
    };

    __global__ void computeBlockLights(
        CudaBuildParams params, uchar3* lightSources, std::uint32_t* lightSourceCount, BlockInfo* blocks)
    {
        const int lightIndex = blockIdx.x;
        if (lightIndex >= *lightSourceCount)
            return;
        const uchar3& light = lightSources[lightIndex];

        const auto center = s_maxLightPropagationDistance;
        const auto dim = s_maxLightPropagationDistance * 2 + 1;
        __shared__ __align__(4) LightingBlockCache blocksCache[dim][dim][dim];

        const auto blockAt = [blocks, &params](int x, int y, int z) -> BlockInfo& {
            return blocks[x + params.width * (y + params.height * z)];
        };

        const int totalElements = dim * dim * dim;
        for (int i = threadIdx.x; i < totalElements; i += blockDim.x) {
            const int x = i % dim;
            const int y = (i / dim) % dim;
            const int z = i / (dim * dim);

            const int worldX = light.x + x - center;
            const int worldY = light.y + y - center;
            const int worldZ = light.z + z - center;

            const bool isOutOfBounds = worldX < 0 || worldY < 0 || worldZ < 0 || worldX >= params.width
                || worldY >= params.height || worldZ >= params.depth;

            const bool isSolid = isOutOfBounds || isOpaqueBlock(blockAt(worldX, worldY, worldZ).blockType);
            blocksCache[x][y][z].init(isSolid);
        }

        __shared__ uchar3 queue1[s_maxLightComputeQueueLength];
        __shared__ uchar3 queue2[s_maxLightComputeQueueLength];

        __shared__ int curQueueLength;
        __shared__ int newQueueLength;
        __shared__ uchar3* curQueue;
        __shared__ uchar3* newQueue;

        if (threadIdx.x == 0) {
            curQueue = queue1;
            newQueue = queue2;
            newQueue[0]
                = { s_maxLightPropagationDistance, s_maxLightPropagationDistance, s_maxLightPropagationDistance };
            newQueueLength = 1;
            blocksCache[center][center][center].setLight(15);
        }

        const auto updateLight = [](int x, int y, int z, std::uint8_t light) {
            if (blocksCache[x][y][z].isSolid())
                return;

            if (blocksCache[x][y][z].atomicSetLightIfGreater(light)) {
                int index = atomicAdd(&newQueueLength, 1);
                newQueue[index] = make_uchar3(x, y, z);
            }
        };

        const std::int32_t queueIndex = threadIdx.x;
        for (int iter = 0; iter < s_maxLightPropagationDistance; ++iter) {
            __syncthreads();
            if (threadIdx.x == 0) {
                const auto tmp = curQueue;
                curQueue = newQueue;
                newQueue = tmp;

                curQueueLength = newQueueLength;
                newQueueLength = 0;
            }
            __syncthreads();

            if (!curQueueLength)
                break;

            if (curQueueLength <= queueIndex)
                continue;

            const uchar3 pos = curQueue[queueIndex];
            std::uint8_t light = blocksCache[pos.x][pos.y][pos.z].light();
            if (light > 0) {
                --light;
                updateLight(pos.x - 1, pos.y, pos.z, light);
                updateLight(pos.x + 1, pos.y, pos.z, light);
                updateLight(pos.x, pos.y - 1, pos.z, light);
                updateLight(pos.x, pos.y + 1, pos.z, light);
                updateLight(pos.x, pos.y, pos.z - 1, light);
                updateLight(pos.x, pos.y, pos.z + 1, light);
            }
        }

        for (int i = threadIdx.x; i < totalElements; i += blockDim.x) {
            const int x = i % dim;
            const int y = (i / dim) % dim;
            const int z = i / (dim * dim);

            const int worldX = light.x + x - center;
            const int worldY = light.y + y - center;
            const int worldZ = light.z + z - center;

            const bool isOutOfBounds = worldX < 0 || worldY < 0 || worldZ < 0 || worldX >= params.width
                || worldY >= params.height || worldZ >= params.depth;
            if (isOutOfBounds)
                continue;

            atomicMax(&blockAt(worldX, worldY, worldZ).blockLight, blocksCache[x][y][z].light());
        }
    }

} // namespace

class WorldGenerator::Impl {
public:
    Impl()
    {
        cudaCheck(cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");

        cudaCheck(cudaMalloc(&m_voxelCountDevice, sizeof(*m_voxelCountDevice)), "cudaMalloc voxelCount");
        cudaCheck(cudaMallocHost(&m_voxelCountHost, sizeof(*m_voxelCountHost)), "cudaMallocHost voxelCount");

        cudaCheck(cudaMalloc(&m_lightSourcesDevice, sizeof(*m_lightSourcesDevice) * s_maxLightSourcesPerGeneration),
            "cudaMalloc lightSource");
        cudaCheck(
            cudaMalloc(&m_lightSourceCountDevice, sizeof(*m_lightSourceCountDevice)), "cudaMalloc lightSourceCount");
    }

    ~Impl()
    {
        cudaStreamSynchronize(m_stream);

        cudaFree(m_overridesDevice);

        cudaFree(m_voxelCountDevice);
        cudaFreeHost(m_voxelCountHost);

        cudaFree(m_blockCacheDevice);
        cudaFree(m_heightMapDevice);

        cudaFree(m_lightSourcesDevice);
        cudaFree(m_lightSourceCountDevice);

        cudaStreamDestroy(m_stream);
    }

    cudaStream_t stream() const { return m_stream; }
    CudaFuture<WorldGenerationOutput> generate(const WorldGenerationInput&, WorldGenerationBuffers&&);

private:
    PageLockedVector<CudaOverride> m_packedOverrides;
    CudaOverride* m_overridesDevice = nullptr;
    std::uint32_t m_overridesCapacity = 0;

    std::uint32_t* m_voxelCountDevice = nullptr;
    std::uint32_t* m_voxelCountHost = nullptr;

    BlockInfo* m_blockCacheDevice = nullptr;
    std::uint32_t m_blockCacheCapacity = 0;

    HeightMapValueT* m_heightMapDevice = nullptr;
    std::uint32_t m_heightMapCapacity = 0;

    uchar3* m_lightSourcesDevice = nullptr;
    std::uint32_t* m_lightSourceCountDevice = nullptr;

    cudaStream_t m_stream = nullptr;

    bool m_pending = false;
};

WorldGenerator::WorldGenerator(WorldGenerationConfig config)
    : m_config(config)
    , m_impl(std::make_unique<Impl>())
{
}

WorldGenerator::~WorldGenerator() = default;

cudaStream_t WorldGenerator::stream() const
{
    assert(m_impl);
    return m_impl->stream();
}

CudaFuture<WorldGenerationOutput> WorldGenerator::generate(
    const WorldGenerationInput& input, WorldGenerationBuffers&& buffers)
{
    assert(m_impl);
    return m_impl->generate(input, std::move(buffers));
}

CudaFuture<WorldGenerationOutput> WorldGenerator::Impl::generate(
    const WorldGenerationInput& input, WorldGenerationBuffers&& buffers)
{
    if (m_pending) {
        cudaCheck(cudaStreamSynchronize(m_stream), "cudaStreamSynchronize previous terrain mesh build");
        m_pending = false;
    }

    const std::uint32_t blockCount = static_cast<std::uint32_t>(input.size.x * input.size.y * input.size.z);
    if (m_blockCacheCapacity < blockCount) {
        cudaFree(m_blockCacheDevice);
        m_blockCacheCapacity = blockCount;
        cudaCheck(cudaMalloc(&m_blockCacheDevice, sizeof(*m_blockCacheDevice) * m_blockCacheCapacity),
            "cudaMalloc blockCache");
    }

    const std::uint32_t heightMapSize = static_cast<std::uint32_t>(input.size.x * input.size.z);
    if (m_heightMapCapacity < heightMapSize) {
        cudaFree(m_heightMapDevice);
        m_heightMapCapacity = heightMapSize;
        cudaCheck(
            cudaMalloc(&m_heightMapDevice, sizeof(*m_heightMapDevice) * m_heightMapCapacity), "cudaMalloc heightMap");
    }

    PageLockedVector<CudaOverride>& packedOverrides = m_packedOverrides;
    packedOverrides.clear();
    packedOverrides.reserve(input.overrides.size());
    for (const BlockOverride& blockOverride : input.overrides) {
        packedOverrides.push_back({
            .key = packKeyHost(blockOverride.coord.x, blockOverride.coord.y, blockOverride.coord.z),
            .block = blockOverride.block,
        });
    }
    std::sort(packedOverrides.begin(), packedOverrides.end(),
        [](const CudaOverride& a, const CudaOverride& b) { return a.key < b.key; });

    if (m_overridesCapacity < packedOverrides.size()) {
        cudaFree(m_overridesDevice);
        m_overridesDevice = nullptr;
        m_overridesCapacity = static_cast<std::uint32_t>(std::max<std::size_t>(packedOverrides.size(), 1U));
        cudaCheck(cudaMalloc(&m_overridesDevice, sizeof(CudaOverride) * m_overridesCapacity), "cudaMalloc overrides");
    }
    if (!packedOverrides.empty()) {
        cudaCheck(cudaMemcpyAsync(m_overridesDevice, packedOverrides.data(),
                      sizeof(CudaOverride) * packedOverrides.size(), cudaMemcpyHostToDevice, m_stream),
            "cudaMemcpyAsync overrides");
    }

    cudaCheck(
        cudaMemsetAsync(m_voxelCountDevice, 0, sizeof(*m_voxelCountDevice), m_stream), "cudaMemsetAsync voxelCount");
    cudaCheck(cudaMemsetAsync(m_lightSourceCountDevice, 0, sizeof(*m_lightSourceCountDevice), m_stream),
        "cudaMemsetAsync lightSourceCount");

    const CudaBuildParams params {
        .seed = input.seed,
        .centerX = input.center.x,
        .centerY = input.center.y,
        .centerZ = input.center.z,
        .width = input.size.x,
        .height = input.size.y,
        .depth = input.size.z,
        .originX = input.origin.x,
        .originY = input.origin.y,
        .originZ = input.origin.z,
        .overrideCount = static_cast<std::int32_t>(packedOverrides.size()),
        .maxVoxels = buffers.terrain.maxVoxels,
        .maxLightSources = s_maxLightSourcesPerGeneration,
    };

    // 1. build a dense block cache
    {
        buildBlockCacheKernel<<<{ params.width, params.depth }, params.height, 0, m_stream>>>(params, m_overridesDevice,
            buffers.terrain.header, m_lightSourcesDevice, m_lightSourceCountDevice, m_blockCacheDevice,
            m_heightMapDevice);
        cudaCheck(cudaGetLastError(), "buildBlockCacheKernel");
    }
    // 2. compute lighting
    {
        constexpr std::int32_t threadCount = s_maxLightComputeQueueLength;
        computeBlockLights<<<s_maxLightSourcesPerGeneration, threadCount, 0, m_stream>>>(
            params, m_lightSourcesDevice, m_lightSourceCountDevice, m_blockCacheDevice);
        cudaCheck(cudaGetLastError(), "computeBlockLightsKernel");
    }
    // 3. build a terrain for rendering
    {
        constexpr std::int32_t threadCount = 256;
        buildTerrainKernel<<<(blockCount + threadCount - 1) / threadCount, threadCount, 0, m_stream>>>(
            params, buffers.terrain.voxels, m_voxelCountDevice, m_blockCacheDevice);
        cudaCheck(cudaGetLastError(), "buildTerrainKernel");
    }

    cudaCheck(cudaMemcpyAsync(
                  m_voxelCountHost, m_voxelCountDevice, sizeof(*m_voxelCountHost), cudaMemcpyDeviceToHost, m_stream),
        "cudaMemcpyAsync voxelCount");

    buffers.cpuCache.blocks.uninitializedResize(blockCount);
    assert(m_blockCacheCapacity >= buffers.cpuCache.blocks.size());
    cudaCheck(cudaMemcpyAsync(buffers.cpuCache.blocks.data(), m_blockCacheDevice,
                  sizeof(*m_blockCacheDevice) * buffers.cpuCache.blocks.size(), cudaMemcpyDeviceToHost, m_stream),
        "cudaMemcpyAsync blockCache");

    buffers.cpuCache.heightMap.uninitializedResize(heightMapSize);
    assert(m_heightMapCapacity >= buffers.cpuCache.heightMap.size());
    cudaCheck(cudaMemcpyAsync(buffers.cpuCache.heightMap.data(), m_heightMapDevice,
                  sizeof(*m_heightMapDevice) * buffers.cpuCache.heightMap.size(), cudaMemcpyDeviceToHost, m_stream),
        "cudaMemcpyAsync heightMap");

    m_pending = true;
    WorldGenerationOutput output;
    output.origin = input.origin;
    output.size = input.size;
    output.worldVersion = input.worldVersion;
    output.buffers = std::move(buffers);
    auto outputStorage = std::make_shared<WorldGenerationOutput>(std::move(output));
    return CudaFuture<WorldGenerationOutput>(m_stream, [this, outputStorage]() mutable {
        m_pending = false;
        outputStorage->voxelCount = std::min(*m_voxelCountHost, outputStorage->buffers.terrain.maxVoxels);
        return std::move(*outputStorage);
    });
}

} // namespace blocklab
