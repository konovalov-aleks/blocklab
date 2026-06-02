#include "blocklab/CudaWorldGenerator.h"

#include "blocklab/CudaHelpers.h"
#include "blocklab/Hash.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace blocklab {
namespace {

    constexpr int32_t ChunkSizeY = 32;
    constexpr int32_t CoordBias = 1 << 20;
    constexpr int64_t CoordMask = (int64_t { 1 } << 21) - 1;

    struct CudaVertex {
        float4 position;
        float4 colorAndShade;
        float4 uvMaterial;
    };

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
        uint32_t maxVertices;
    };

    static_assert(sizeof(MeshVertex) == sizeof(CudaVertex));

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

    __device__ float3 topColor(uint8_t block)
    {
        if (block == BlockId::Grass)
            return make_float3(0.28f, 0.62f, 0.24f);
        if (block == BlockId::Dirt)
            return make_float3(0.49f, 0.33f, 0.20f);
        if (block == BlockId::Stone)
            return make_float3(0.50f, 0.52f, 0.54f);
        return make_float3(1.0f, 0.0f, 1.0f);
    }

    __device__ float3 sideColor(uint8_t block)
    {
        if (block == BlockId::Grass)
            return make_float3(0.43f, 0.35f, 0.20f);
        return topColor(block);
    }

    __device__ MeshMaterial topMaterial(uint8_t block)
    {
        if (block == BlockId::Grass)
            return MeshMaterial::GrassTop;
        if (block == BlockId::Dirt)
            return MeshMaterial::Dirt;
        if (block == BlockId::Stone)
            return MeshMaterial::Stone;
        return MeshMaterial::VertexColor;
    }

    __device__ MeshMaterial sideMaterial(uint8_t block)
    {
        if (block == BlockId::Grass)
            return MeshMaterial::GrassSide;
        return topMaterial(block);
    }

    __device__ void writeVertex(
        CudaVertex& vertex, float3 position, float3 color, float shade, float2 uv, MeshMaterial material)
    {
        vertex.position = make_float4(position.x, position.y, position.z, 1.0f);
        vertex.colorAndShade = make_float4(color.x, color.y, color.z, shade);
        vertex.uvMaterial = make_float4(uv.x, uv.y, static_cast<float>(static_cast<uint32_t>(material)), 0.0f);
    }

    __device__ float2 faceUv(
        float3 p, float3 p0, float3 p1, float3 p2, float3 p3, float2 fallback, MeshMaterial material)
    {
        if (material != MeshMaterial::GrassSide)
            return fallback;

        const float minY = fminf(fminf(p0.y, p1.y), fminf(p2.y, p3.y));
        const float maxY = fmaxf(fmaxf(p0.y, p1.y), fmaxf(p2.y, p3.y));
        const float minX = fminf(fminf(p0.x, p1.x), fminf(p2.x, p3.x));
        const float maxX = fmaxf(fmaxf(p0.x, p1.x), fmaxf(p2.x, p3.x));
        const float minZ = fminf(fminf(p0.z, p1.z), fminf(p2.z, p3.z));
        const float maxZ = fmaxf(fmaxf(p0.z, p1.z), fmaxf(p2.z, p3.z));
        const float xSpan = fmaxf(maxX - minX, 0.001f);
        const float ySpan = fmaxf(maxY - minY, 0.001f);
        const float zSpan = fmaxf(maxZ - minZ, 0.001f);
        const float horizontal = xSpan >= zSpan ? (p.x - minX) / xSpan : (p.z - minZ) / zSpan;
        return make_float2(horizontal, (p.y - minY) / ySpan);
    }

    __device__ void appendFace(CudaVertex* vertices, uint32_t* vertexCount, const CudaBuildParams params, float3 p0,
        float3 p1, float3 p2, float3 p3, float3 color, float shade, MeshMaterial material)
    {
        const uint32_t base = atomicAdd(vertexCount, 6U);
        if (base + 5U >= params.maxVertices)
            return;
        const float2 uv0 = faceUv(p0, p0, p1, p2, p3, make_float2(0.0f, 0.0f), material);
        const float2 uv1 = faceUv(p1, p0, p1, p2, p3, make_float2(1.0f, 0.0f), material);
        const float2 uv2 = faceUv(p2, p0, p1, p2, p3, make_float2(1.0f, 1.0f), material);
        const float2 uv3 = faceUv(p3, p0, p1, p2, p3, make_float2(0.0f, 1.0f), material);
        writeVertex(vertices[base + 0U], p0, color, shade, uv0, material);
        writeVertex(vertices[base + 1U], p1, color, shade, uv1, material);
        writeVertex(vertices[base + 2U], p2, color, shade, uv2, material);
        writeVertex(vertices[base + 3U], p0, color, shade, uv0, material);
        writeVertex(vertices[base + 4U], p2, color, shade, uv2, material);
        writeVertex(vertices[base + 5U], p3, color, shade, uv3, material);
    }

    __global__ void buildTerrainMeshKernel(CudaBuildParams params, const CudaOverride* overrides, CudaVertex* vertices,
        uint32_t* vertexCount, uint8_t* blocks)
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
        if (blockAt(params, overrides, x + 1, y, z) != 0 && blockAt(params, overrides, x - 1, y, z) != 0
            && blockAt(params, overrides, x, y + 1, z) != 0 && blockAt(params, overrides, x, y - 1, z) != 0
            && blockAt(params, overrides, x, y, z + 1) != 0 && blockAt(params, overrides, x, y, z - 1) != 0) {
            return;
        }

        const float fx = static_cast<float>(x);
        const float fy = static_cast<float>(y);
        const float fz = static_cast<float>(z);
        const float3 p000 = make_float3(fx + 0.0f, fy + 0.0f, fz + 0.0f);
        const float3 p100 = make_float3(fx + 1.0f, fy + 0.0f, fz + 0.0f);
        const float3 p010 = make_float3(fx + 0.0f, fy + 1.0f, fz + 0.0f);
        const float3 p110 = make_float3(fx + 1.0f, fy + 1.0f, fz + 0.0f);
        const float3 p001 = make_float3(fx + 0.0f, fy + 0.0f, fz + 1.0f);
        const float3 p101 = make_float3(fx + 1.0f, fy + 0.0f, fz + 1.0f);
        const float3 p011 = make_float3(fx + 0.0f, fy + 1.0f, fz + 1.0f);
        const float3 p111 = make_float3(fx + 1.0f, fy + 1.0f, fz + 1.0f);
        const float3 top = topColor(block);
        const float3 side = sideColor(block);
        const MeshMaterial topTile = topMaterial(block);
        const MeshMaterial sideTile = sideMaterial(block);

        if (blockAt(params, overrides, x, y + 1, z) == 0)
            appendFace(vertices, vertexCount, params, p010, p011, p111, p110, top, 1.00f, topTile);
        if (blockAt(params, overrides, x, y - 1, z) == 0)
            appendFace(vertices, vertexCount, params, p000, p100, p101, p001, side, 0.48f, sideTile);
        if (blockAt(params, overrides, x + 1, y, z) == 0)
            appendFace(vertices, vertexCount, params, p100, p110, p111, p101, side, 0.78f, sideTile);
        if (blockAt(params, overrides, x - 1, y, z) == 0)
            appendFace(vertices, vertexCount, params, p000, p001, p011, p010, side, 0.78f, sideTile);
        if (blockAt(params, overrides, x, y, z + 1) == 0)
            appendFace(vertices, vertexCount, params, p001, p101, p111, p011, side, 0.68f, sideTile);
        if (blockAt(params, overrides, x, y, z - 1) == 0)
            appendFace(vertices, vertexCount, params, p000, p010, p110, p100, side, 0.68f, sideTile);
    }

} // namespace

struct CudaWorldGenerator::State {
    CudaOverride* overrides = nullptr;
    uint32_t* vertexCount = nullptr;
    uint32_t* vertexCountHost = nullptr;
    cudaStream_t stream = nullptr;
    PageLockedVector<CudaOverride> packedOverrides;
    uint32_t overrideCapacity = 0;
    bool pending = false;
};

CudaWorldGenerator::CudaWorldGenerator()
    : m_state(std::make_unique<State>())
{
    cudaCheck(cudaStreamCreateWithFlags(&m_state->stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    cudaCheck(cudaMalloc(&m_state->vertexCount, sizeof(uint32_t)), "cudaMalloc vertexCount");
    cudaCheck(cudaMallocHost(&m_state->vertexCountHost, sizeof(uint32_t)), "cudaMallocHost vertexCount");
}

CudaWorldGenerator::~CudaWorldGenerator()
{
    if (!m_state)
        return;
    cudaStreamSynchronize(m_state->stream);
    cudaFree(m_state->overrides);
    cudaFree(m_state->vertexCount);
    cudaFreeHost(m_state->vertexCountHost);
    cudaStreamDestroy(m_state->stream);
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

    cudaCheck(
        cudaMemsetAsync(m_state->vertexCount, 0, sizeof(uint32_t), m_state->stream), "cudaMemsetAsync vertexCount");
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
        .maxVertices = static_cast<uint32_t>(buffers.meshVertices.size()),
    };

    constexpr int32_t ThreadCount = 256;
    buildTerrainMeshKernel<<<(volume + ThreadCount - 1) / ThreadCount, ThreadCount, 0, m_state->stream>>>(params,
        m_state->overrides, reinterpret_cast<CudaVertex*>(buffers.meshVertices.data()), m_state->vertexCount,
        outBlocks.data());
    cudaCheck(cudaGetLastError(), "buildTerrainMeshKernel");

    cudaCheck(cudaMemcpyAsync(m_state->vertexCountHost, m_state->vertexCount, sizeof(*m_state->vertexCountHost),
                  cudaMemcpyDeviceToHost, m_state->stream),
        "cudaMemcpyAsync vertexCount");

    m_state->pending = true;
    WorldGenerationOutput output;
    output.origin = input.origin;
    output.size = input.size;
    output.worldVersion = input.worldVersion;
    output.buffers = std::move(buffers);
    output.buffers.blocks = std::move(outBlocks);
    auto outputStorage = std::make_shared<WorldGenerationOutput>(std::move(output));
    return CudaFuture<WorldGenerationOutput>(
        m_state->stream, [state = m_state.get(), outputStorage]() mutable {
            state->pending = false;
            outputStorage->meshVertexCount
                = std::min(*state->vertexCountHost, static_cast<uint32_t>(outputStorage->buffers.meshVertices.size()));
            return std::move(*outputStorage);
        });
}

} // namespace blocklab
