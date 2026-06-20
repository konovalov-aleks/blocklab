#pragma once

#include "Block.h"
#include "Voxel.h"

#include <blocklab/environment/Agent.h>
#include <blocklab/gpu/cuda/CudaFuture.h>
#include <blocklab/gpu/cuda/PageLockedVector.h>
#include <blocklab/utility/Math.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace blocklab {

class CudaWorldGenerator;

struct TerrainHeader {
    std::int32_t originX;
    std::int32_t originZ;
};
static_assert(sizeof(TerrainHeader) == 8, "The layout must match the structure layout in the voxel shader");

struct WorldGenerationConfig {
    std::int32_t halfExtent = 32;
};

struct WorldGenerationInput {
    std::uint32_t seed = 1;
    std::uint64_t worldVersion = 0;
    IVec3 center {};
    IVec3 size {};
    std::int32_t originX = 0;
    std::int32_t originZ = 0;
    std::int32_t halfExtent = 0;
    std::span<const BlockOverride> overrides;
};

struct WorldGenerationBuffers {
    TerrainHeader* header;

    Voxel* voxels;
    std::size_t maxVoxelCount;

    PageLockedVector<std::uint8_t> blocks;
};

struct WorldGenerationOutput {
    IVec3 origin {};
    IVec3 size {};
    std::uint64_t worldVersion = 0;
    // TODO is this field necessary? it seems, we can use buffers.voxels.size() instead
    std::uint32_t voxelCount = 0;
    WorldGenerationBuffers buffers;
};

class WorldGenerator final {
public:
    explicit WorldGenerator(WorldGenerationConfig config = {});
    ~WorldGenerator();

    // TODO join WorldGenerator and CudaWorldGenerator
    cudaStream_t stream() const;

    CudaFuture<WorldGenerationOutput> generate(const World&, const AgentState&, WorldGenerationBuffers&& buffers);

private:
    WorldGenerationConfig m_config;
    std::vector<BlockOverride> m_overrides;
    std::unique_ptr<blocklab::CudaWorldGenerator> m_cudaGenerator;
};

} // namespace blocklab
