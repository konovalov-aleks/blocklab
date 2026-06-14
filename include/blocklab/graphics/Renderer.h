#pragma once

#include <blocklab/Environment.h>
#include <blocklab/WorldGenerator.h>
#include <blocklab/graphics/Vulkan.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <vector>

namespace blocklab {

enum class RenderEntityKind : uint32_t {
    None = 0,
    Pig = 1,
};

constexpr float renderEntityKindId(RenderEntityKind kind) { return static_cast<float>(static_cast<uint32_t>(kind)); }

struct RenderConfig {
    int32_t width = 320;
    int32_t height = 180;
    uint32_t batchSize = 1;
};

class Renderer final : public ObservationRenderer {
public:
    Renderer(Vulkan&, RenderConfig config = {});
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    const Observation& renderObservations(std::span<const World>, std::span<const AgentState>) override;
    const Observation& observation() const { return m_observation; }
    std::size_t cudaObservationTensorBytes() const;

    struct VulkanState;
    struct RenderParams {
        struct alignas(16) FrameInfo {
            int32_t animationTimeMs = 0;
        };

        Vec4 origin;
        Vec4 forward;
        Vec4 right;
        Vec4 up;
        IVec4 worldOriginAndWidth;
        IVec4 regionAndHeight;
        FrameInfo frameInfo;
        Vec4 tuning;
    };
    struct EntityInstance {
        Vec4 positionAndYaw;
        Vec4 velocityAndKind;
    };
    struct DrawPushConstants {
        uint32_t envIndex = 0;
        uint32_t layerIndex = 0;
        uint32_t padding0 = 0;
        uint32_t padding1 = 0;
    };
    static_assert(sizeof(RenderParams::FrameInfo) == sizeof(IVec4));

private:
    struct RenderSlot {
        IVec3 lastMeshCenter { std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::min(),
            std::numeric_limits<int32_t>::min() };

        uint32_t terrainVoxelOffset = 0;
        uint32_t terrainVoxelCount = 0;

        uint32_t pigVertexCount = 0;

        uint32_t instanceOffset = 0;
        uint32_t instanceCount = 0;

        CudaSharedFuture<WorldGenerationOutput> pendingGeneration;
        uint64_t lastWorldVersion = 0;
    };

    RenderParams buildRenderParams(const AgentState&, const World&) const;
    void uploadInstances(std::size_t slot, const World&);
    void drawFrame();
    void initializeBatchData();
    void renderObservationSlot(std::size_t slot, const World&, const AgentState&);

    RenderConfig m_config;

    Vulkan& m_vk;
    std::unique_ptr<VulkanState> m_state;

    WorldGenerator m_worldGenerator;
    Observation m_observation;
    std::vector<EntityInstance> m_instances;
    std::unique_ptr<RenderSlot[]> m_slots;
    std::unique_ptr<RenderParams[]> m_renderParams;
    uint32_t m_batchSize = 0;
    uint32_t m_pigMeshVertexCount = 0;
    uint64_t m_observationVersion = 0;
    bool m_pigMeshUploaded = false;
};

} // namespace blocklab
