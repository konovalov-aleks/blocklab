#pragma once

#include "NPC.h"

#include <cstdint>

namespace blocklab {

class PigCharacter final : public NPC {
public:
    PigCharacter(EntityId id, Vec3 position);

private:
    void updateState(World& world, Vec3 threatPosition, float dt) override;

    std::uint32_t m_walkSeed = 0;
    std::int32_t m_walkStep = 0;
    float m_blockedTimer = 0.0f;
};

} // namespace blocklab
