#pragma once

#include "blocklab/Environment.h"

#include <SDL3/SDL.h>

#include <cstdint>
#include <vector>

namespace blocklab {

struct RenderConfig {
    int32_t width = 320;
    int32_t height = 180;
    bool presentToWindow = true;
};

class Renderer final : public ObservationRenderer {
public:
    Renderer(SDL_Window* window, RenderConfig config);
    ~Renderer();

    void resize(int width, int height);
    Observation renderObservation(const World& world, const AgentState& agent) override;
    void present();
    const Observation& observation() const { return m_observation; }

private:
    struct RenderParams {
        Vec4 origin;
        Vec4 forward;
        Vec4 right;
        Vec4 up;
        IVec4 worldOriginAndWidth;
        IVec4 regionAndHeight;
        IVec4 overrideInfo;
        Vec4 tuning;
    };

    struct GpuBlockOverride {
        int32_t x = 0;
        int32_t y = 0;
        int32_t z = 0;
        int32_t block = 0;
    };

    RenderConfig m_config;
    SDL_Window* m_window = nullptr;
    SDL_GPUDevice* m_device = nullptr;
    SDL_GPUTexture* m_observationTarget = nullptr;
    SDL_GPUBuffer* m_blockBuffer = nullptr;
    SDL_GPUBuffer* m_overrideBuffer = nullptr;
    SDL_GPUTransferBuffer* m_overrideTransferBuffer = nullptr;
    SDL_GPUBuffer* m_meshVertexBuffer = nullptr;
    SDL_GPUBuffer* m_meshDrawBuffer = nullptr;
    SDL_GPUTexture* m_blockAtlas = nullptr;
    SDL_GPUSampler* m_blockSampler = nullptr;
    SDL_GPUTexture* m_depthTarget = nullptr;
    SDL_GPUComputePipeline* m_generatePipeline = nullptr;
    SDL_GPUComputePipeline* m_applyOverridesPipeline = nullptr;
    SDL_GPUComputePipeline* m_resetMeshDrawPipeline = nullptr;
    SDL_GPUComputePipeline* m_buildMeshPipeline = nullptr;
    SDL_GPUGraphicsPipeline* m_meshPipeline = nullptr;
    int m_width = 1280;
    int m_height = 720;
    Observation m_observation;
    std::vector<BlockOverride> m_activeOverrides;
    std::vector<GpuBlockOverride> m_gpuOverrides;
    IVec3 m_cachedRegionOrigin { 0, 0, 0 };
    uint64_t m_cachedWorldVersion = 0;
    bool m_blockBufferValid = false;

    void initializeGpuResources();
    void releaseGpuResources();
    void uploadBlockAtlas();
    RenderParams buildRenderParams(const AgentState& agent) const;
    bool updateBlockBuffer(const World& world, const RenderParams& params, SDL_GPUCommandBuffer* commandBuffer);
    void rebuildMeshBuffer(const RenderParams& params, SDL_GPUCommandBuffer* commandBuffer);
    void renderGpuFrame(const World& world, const AgentState& agent);
};

} // namespace blocklab
