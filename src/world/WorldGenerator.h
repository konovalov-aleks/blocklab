#pragma once

#include "Block.h"
#include "Voxel.h"

#include <blocklab/gpu/cuda/CudaFuture.h>
#include <blocklab/utility/Math.h>
#include <environment/Agent.h>
#include <gpu/cuda/PageLockedVector.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace blocklab {

class World;

struct TerrainHeader {
    std::int32_t originX;
    std::int32_t originY;
    std::int32_t originZ;
};
static_assert(sizeof(TerrainHeader) == 12, "The layout must match the structure layout in the voxel shader");

struct WorldGenerationConfig {
    std::int32_t halfExtent = 32;
};

struct WorldGenerationInput {
    std::uint32_t seed = 1;
    std::uint64_t worldVersion = 0;
    IVec3 center {};
    IVec3 size {};
    IVec3 origin {};
    std::span<const BlockOverride> overrides;
};

struct WorldGenerationBuffers {
    TerrainHeader* header;

    Voxel* voxels;
    std::uint32_t maxVoxels;

    PageLockedVector<BlockInfo> blocks;
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

    WorldGenerator(const WorldGenerator&) = delete;
    WorldGenerator& operator=(const WorldGenerator&) = delete;

    cudaStream_t stream() const;

    CudaFuture<WorldGenerationOutput> generate(const World&, const AgentState&, WorldGenerationBuffers&& buffers);

private:
    CudaFuture<WorldGenerationOutput> generate(const WorldGenerationInput&, WorldGenerationBuffers&&);

    struct Impl;

    WorldGenerationConfig m_config;
    std::vector<BlockOverride> m_overrides;
    std::unique_ptr<Impl> m_impl;
};

} // namespace blocklab
