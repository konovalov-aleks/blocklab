#pragma once

#include "blocklab/Environment.h"
#include "blocklab/MeshBuilder.h"

#include <cstdint>
#include <limits>
#include <span>
#include <vector>

struct GLFWwindow;

namespace blocklab {

enum class RenderEntityKind : uint32_t {
    None = 0,
    Pig = 1,
};

constexpr float renderEntityKindId(RenderEntityKind kind) { return static_cast<float>(static_cast<uint32_t>(kind)); }

struct RenderConfig {
    int32_t width = 320;
    int32_t height = 180;
    bool visible = true;
    bool present = true;
};

class Renderer final : public ObservationRenderer {
public:
    explicit Renderer(RenderConfig config = {});
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    bool shouldClose() const;
    void pollEvents();
    GLFWwindow* window() const { return m_window; }
    void resize(int32_t width, int32_t height);
    Observation renderObservation(const World& world, const AgentState& agent) override;
    const Observation& observation() const { return m_observation; }

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
    static_assert(sizeof(RenderParams::FrameInfo) == sizeof(IVec4));

private:
    RenderParams buildRenderParams(const AgentState&, const World&) const;
    void uploadInstances(const World&);
    void drawFrame(const RenderParams&);

    RenderConfig m_config;
    GLFWwindow* m_window = nullptr;
    VulkanState* m_vk = nullptr;
    MeshBuilder m_meshBuilder;
    Observation m_observation;
    uint64_t m_lastWorldVersion = 0;
    IVec3 m_lastMeshCenter { std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::min(),
        std::numeric_limits<int32_t>::min() };
    std::vector<MeshVertex> m_pigMesh;
    std::vector<EntityInstance> m_instances;
    uint32_t m_terrainVertexCount = 0;
    uint32_t m_pigVertexOffset = 0;
    uint32_t m_pigVertexCount = 0;
    uint32_t m_instanceCount = 0;
};

} // namespace blocklab
