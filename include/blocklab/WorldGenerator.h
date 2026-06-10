#pragma once

#include "blocklab/Agent.h"
#include "blocklab/Block.h"
#include "blocklab/CudaFuture.h"
#include "blocklab/Math.h"
#include "blocklab/PageLockedVector.h"
#include "blocklab/Voxel.h"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace blocklab {

class CudaWorldGenerator;

struct TerrainHeader {
    int32_t originX;
    int32_t originZ;
};
static_assert(sizeof(TerrainHeader) == 8, "The layout must match the structure layout in the voxel shader");

struct WorldGenerationConfig {
    int32_t halfExtent = 32;
};

struct WorldGenerationInput {
    uint32_t seed = 1;
    uint64_t worldVersion = 0;
    IVec3 center {};
    IVec3 size {};
    int32_t originX = 0;
    int32_t originZ = 0;
    int32_t halfExtent = 0;
    std::span<const BlockOverride> overrides;
};

struct WorldGenerationBuffers {
    TerrainHeader* header;

    Voxel* voxels;
    size_t maxVoxelCount;

    PageLockedVector<uint8_t> blocks;
};

struct WorldGenerationOutput {
    IVec3 origin {};
    IVec3 size {};
    uint64_t worldVersion = 0;
    // TODO is this field necessary? it seems, we can use buffers.voxels.size() instead
    uint32_t voxelCount = 0;
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
