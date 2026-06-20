#pragma once

#include "Character.h"

#include <blocklab/utility/Math.h>

namespace blocklab {

class AgentCharacter : public Character {
public:
    AgentCharacter(Vec3 position);

    using Character::applyPhysics;
    using Character::requestJump;
    using Character::resetBody;
    using Character::setHorizontalMovement;
};

} // namespace blocklab
