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

enum class RenderEntityFamily : std::uint32_t {
    None = 0,
    Character = 1,
    Drop = 2,
};

inline constexpr std::uint32_t RenderEntityFamilyShift = 24;
inline constexpr std::uint32_t RenderEntityLocalIdMask = (1U << RenderEntityFamilyShift) - 1U;

consteval std::uint32_t renderEntityId(RenderEntityFamily family, std::uint32_t localId)
{
    return (static_cast<std::uint32_t>(family) << RenderEntityFamilyShift) | localId;
}

enum class RenderEntityKind : std::uint32_t {
    None = 0,

    Pig = renderEntityId(RenderEntityFamily::Character, 1),

    DirtDrop = renderEntityId(RenderEntityFamily::Drop, 1),
    StoneDrop = renderEntityId(RenderEntityFamily::Drop, 2),
    TorchDrop = renderEntityId(RenderEntityFamily::Drop, 3),
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
            std::uint32_t animationTimeMs = 0;
            std::uint32_t _padding[3];
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
            Vec3 skyLightDirection;
            float _padding;
        };

        Vec3 origin;
        float _padding1;
        Vec3 forward;
        float _padding2;
        Vec3 right;
        float _padding3;
        Vec3 up;
        float _padding4;
        IVec3 worldOrigin;
        std::int32_t viewportWidth;
        IVec3 regionOrigin;
        std::int32_t viewportHeight;
        FrameInfo frameInfo;
        ProjectionInfo projectionInfo;
        SkyInfo skyInfo;
    };
    struct EntityInstance {
        Vec3 position;
        float yaw;
        Vec4 velocity;

        std::uint32_t kind;
        float blockLight;
        float skyLight;
        float _padding;
    };
    struct DrawPushConstants {
        std::uint32_t envIndex = 0;
        std::uint32_t layerIndex = 0;
        std::uint32_t padding0 = 0;
        std::uint32_t padding1 = 0;
    };
    static_assert(sizeof(RenderParams::FrameInfo) == sizeof(IVec4));
    static_assert(sizeof(RenderParams::ProjectionInfo) == sizeof(Vec4));
    static_assert(sizeof(RenderParams::SkyInfo) == sizeof(Vec4) * 2);
    static_assert(sizeof(EntityInstance) == sizeof(Vec4) * 3);

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
    std::uint32_t m_dropMeshVertexOffset = 0;
    std::uint32_t m_dropMeshVertexCount = 0;
    std::uint64_t m_observationVersion = 0;
    bool m_entityMeshesUploaded = false;

    friend class Environment;
};

} // namespace blocklab
