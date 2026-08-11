#include "NPC.h"

namespace blocklab {

NPC::NPC(World& world, EntityId id, CharacterKind kind, Vec3 position, CylinderDimensions hitCylinder)
    : Character(world, id, kind, position, hitCylinder)
{
}

void NPC::update(Vec3 threatPosition, float dt)
{
    if (!alive())
        return;

    m_hasHorizontalMovement = false;
    updateState(threatPosition, dt);
    // Apply surface friction even on frames without movement input, so idle characters
    // come to a halt with the same drag model as during movement.
    if (!m_hasHorizontalMovement)
        applyHorizontalFriction(dt);
    applyPhysics(dt);
}

void NPC::setState(CharacterState state) { m_state = state; }

} // namespace blocklab
