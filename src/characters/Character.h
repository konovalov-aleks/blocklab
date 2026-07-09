#pragma once

#include "CharacterKind.h"

#include <blocklab/utility/Math.h>
#include <world/HitVolume.h>

#include <cstdint>

namespace blocklab {

class World;

using EntityId = std::uint32_t;

class Character {
public:
    Character(EntityId, CharacterKind, Vec3 position, CylinderDimensions hitCylinder);
    virtual ~Character() = default;

    EntityId id() const { return m_id; }
    CharacterKind kind() const { return m_kind; }

    std::int32_t health() const { return m_health; }
    bool isAlive() const { return m_health > 0; }

    const Vec3& position() const { return m_position; }
    const Vec3& velocity() const { return m_velocity; }
    const Vec3& direction() const { return m_forward; }

    HitCylinder hitVolume() const
    {
        return { m_hitCylinderDimensions, m_position };
    }

    bool onGround() const { return m_onGround; }
    bool horizontalBlocked() const { return m_horizontalBlocked; }
    bool occupiesBlock(IVec3 block) const;

protected:
    void moveToward(World&, Vec3 target, float speed, float dt);
    void fleeFrom(World&, Vec3 threatPosition, float speed, float dt);
    void dampHorizontalMovement(float dt);
    void resetBody(Vec3 position);
    void setAutoJump(bool enabled, float jumpSpeed);
    void setTurnSpeed(float turnSpeed);
    void setHorizontalMovement(Vec3 direction, float speed, float acceleration, float dt);
    void requestJump(float jumpSpeed);
    void applyPhysics(World&, float dt);

    Vec3 m_position {};
    Vec3 m_velocity {};
    Vec3 m_home {};
    Vec3 m_forward { 0.0f, 0.0f, 1.0f };

    const CylinderDimensions m_hitCylinderDimensions;
    std::int32_t m_health = 1;
    bool m_onGround = false;
    bool m_hasHorizontalMovement = false;
    bool m_horizontalBlocked = false;
    bool m_autoJump = false;
    float m_autoJumpSpeed = 6.0f;
    float m_turnSpeed = 8.0f;

private:
    bool collides(const World& world, Vec3 position) const;

    const EntityId m_id;
    const CharacterKind m_kind;
};

} // namespace blocklab
