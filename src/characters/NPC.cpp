#include "NPC.h"

namespace blocklab {

NPC::NPC(EntityId id, CharacterKind kind, Vec3 position, CylinderDimensions hitCylinder)
    : Character(id, kind, position, hitCylinder)
{
}

void NPC::update(World& world, Vec3 threatPosition, float dt)
{
    if (!isAlive())
        return;

    m_hasHorizontalMovement = false;
    updateState(world, threatPosition, dt);
    if (!m_hasHorizontalMovement)
        dampHorizontalMovement(dt);
    applyPhysics(world, dt);
}

void NPC::setState(CharacterState state) { m_state = state; }

} // namespace blocklab
