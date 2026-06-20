#pragma once

#include "blocklab/characters/Character.h"

#include <cstdint>

namespace blocklab {

enum class CharacterStateKind : std::uint8_t {
    Idle,
    MoveTo,
    EatGrass,
    ChaseTarget,
    Attack,
    Panic,
};

struct CharacterState {
    CharacterStateKind kind = CharacterStateKind::Idle;
    Vec3 target {};
    float timer = 0.0f;
};

class NPC : public Character {
public:
    NPC(EntityId id, CharacterKind kind, Vec3 position);

    CharacterStateKind stateKind() const { return m_state.kind; }
    void update(World& world, Vec3 threatPosition, float dt);

protected:
    void setState(CharacterState state);
    virtual void updateState(World& world, Vec3 threatPosition, float dt) = 0;

    CharacterState m_state;
};

} // namespace blocklab
