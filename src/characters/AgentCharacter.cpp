#include "AgentCharacter.h"

namespace blocklab {

static constexpr CylinderDimensions s_agentHitCylinder = { .radius = 0.3f, .height = 1.8f };

AgentCharacter::AgentCharacter(Vec3 position)
    : Character(0, CharacterKind::Agent, position, s_agentHitCylinder)
{
}

} // namespace blocklab
