#include "AgentCharacter.h"

namespace blocklab {

static constexpr Character::HitBox s_agentHitBox = { .radius = 0.32f, .height = 1.8f };

AgentCharacter::AgentCharacter(Vec3 position)
    : Character(0, CharacterKind::Agent, position, s_agentHitBox)
{
}

} // namespace blocklab
