#pragma once

#include <blocklab/environment/Observation.h>
#include <blocklab/utility/Math.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace blocklab {

class AgentState;
class Environment;
class Vulkan;
class World;
class WorldGenerator;

enum class RenderEntityKind : std::uint32_t {
    None = 0,
    Pig = 1,
};

struct RenderConfig {
    std::int32_t width = 320;
    std::int32_t height = 180;
    std::uint32_t batchSize = 1;
};

class Renderer final {
public:
    Renderer(Vulkan&, RenderConfig config = {});
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    const Observation& observation() const { return m_observation; }
    std::size_t cudaObservationTensorBytes() const;

    struct RenderParams {
        struct alignas(16) FrameInfo {
            std::int32_t animationTimeMs = 0;
            std::int32_t padding[3] = {};
        };
        struct alignas(16) ProjectionInfo {
            float farPlane;
            float fovRadians;
            float fogStart;
            float fogEnd;
        };
        struct alignas(16) SkyInfo {
            Vec3 skyColor;
            // currentLight = max(0, block.skyLight - skyLightDimming)
            std::uint32_t skyLightDimming;
        };

        Vec4 origin;
        Vec4 forward;
        Vec4 right;
        Vec4 up;
        IVec4 worldOriginAndWidth;
        IVec4 regionAndHeight;
        FrameInfo frameInfo;
        ProjectionInfo projectionInfo;
        SkyInfo skyInfo;
    };
    struct EntityInstance {
        Vec4 positionAndYaw;
        Vec4 velocityAndKind;
    };
    struct DrawPushConstants {
        std::uint32_t envIndex = 0;
        std::uint32_t layerIndex = 0;
        std::uint32_t padding0 = 0;
        std::uint32_t padding1 = 0;
    };
    static_assert(sizeof(RenderParams::FrameInfo) == sizeof(IVec4));
    static_assert(sizeof(RenderParams::ProjectionInfo) == sizeof(Vec4));
    static_assert(sizeof(RenderParams::SkyInfo) == sizeof(Vec4));

    struct RenderSlot;
    struct VulkanState;

private:
    const Observation& renderObservations(std::span<const World>, std::span<const AgentState>);
    RenderParams buildRenderParams(const AgentState&, const World&) const;
    void uploadInstances(std::size_t slot, const World&);
    void drawFrame();
    void initializeBatchData();
    void renderObservationSlot(std::size_t slot, const World&, const AgentState&);

    RenderConfig m_config;

    Vulkan& m_vk;
    std::unique_ptr<VulkanState> m_state;

    std::unique_ptr<WorldGenerator> m_worldGenerator;
    Observation m_observation;
    std::vector<EntityInstance> m_instances;
    std::unique_ptr<RenderSlot[]> m_slots;
    std::unique_ptr<RenderParams[]> m_renderParams;
    std::uint32_t m_batchSize = 0;
    std::uint32_t m_pigMeshVertexCount = 0;
    std::uint64_t m_observationVersion = 0;
    bool m_pigMeshUploaded = false;

    friend class Environment;
};

} // namespace blocklab
