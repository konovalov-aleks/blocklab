#pragma once

#include "blocklab/Math.h"
#include "blocklab/characters/Character.h"

#include <cstdint>

namespace blocklab {

struct AgentAction {
    float forward = 0.0f;
    float right = 0.0f;
    bool jump = false;
    bool dig = false;
    bool place = false;
    float yawDelta = 0.0f;
    float pitchDelta = 0.0f;
};

struct AgentState {
    Vec3 position { 0.0f, 14.0f, 0.0f };
    Vec3 velocity {};
    float yaw = 0.0f;
    float pitch = 0.0f;
    bool onGround = false;
    int32_t blocksCollected = 0;
    int32_t blocksPlaced = 0;
};

class Agent final : public Character {
public:
    Agent();

    const AgentState& state() const { return m_state; }
    AgentState& mutableState() { return m_state; }

    void reset(Vec3 position);
    void step(World& world, const AgentAction& action, float dt);

private:
    AgentState m_state;

    void interact(World& world, const AgentAction& action);
    void syncStateFromBody();
};

} // namespace blocklab
