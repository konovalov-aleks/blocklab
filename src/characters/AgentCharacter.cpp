#include "AgentCharacter.h"

namespace blocklab {

static constexpr CylinderDimensions s_agentHitCylinder = { .radius = 0.3f, .height = 1.8f };

AgentCharacter::AgentCharacter(World& world, Vec3 position)
    : Character(world, 0, CharacterKind::Agent, position, s_agentHitCylinder)
{
}

} // namespace blocklab
