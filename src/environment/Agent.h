#pragma once

#include <blocklab/environment/AgentAction.h>
#include <blocklab/inventory/Inventory.h>
#include <blocklab/utility/Math.h>
#include <characters/AgentCharacter.h>

#include <cstdint>

namespace blocklab {

class World;

struct AgentState {
    Vec3 position { 0.0f, 14.0f, 0.0f };
    Vec3 velocity {};
    float yaw = 0.0f;
    float pitch = 0.0f;
    bool grounded = false;
    std::int32_t blocksCollected = 0;
    std::int32_t blocksPlaced = 0;
};

class Agent {
public:
    Agent(World&);

    void reset(Vec3 position);
    void step(const AgentAction&, float dt);

    const Inventory& inventory() const { return m_inventory; }
    const AgentState& state() const { return m_state; }
    AgentState& mutableState() { return m_state; }

private:
    void pickDrops();
    void interact(const AgentAction&);
    void syncStateFromBody();

    World& world() const { return m_character.world(); }

    AgentState m_state;
    AgentCharacter m_character;
    Inventory m_inventory;
};

} // namespace blocklab
