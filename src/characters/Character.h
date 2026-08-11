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
    Character(World&, EntityId, CharacterKind, Vec3 position, CylinderDimensions hitCylinder);
    virtual ~Character() = default;

    World& world() const { return m_world; }
    EntityId id() const { return m_id; }
    CharacterKind kind() const { return m_kind; }

    std::int32_t health() const { return m_health; }
    bool alive() const { return m_health > 0; }

    const Vec3& position() const { return m_position; }
    const Vec3& velocity() const { return m_velocity; }
    const Vec3& direction() const { return m_forward; }

    HitCylinder hitVolume() const
    {
        return { m_hitCylinderDimensions, m_position };
    }

    bool grounded() const { return m_onGround; }
    bool horizontalBlocked() const { return m_horizontalBlocked; }
    bool occupiesBlock(IVec3 block) const;

    virtual void onHit(Vec3 direction, float power, int damage);

protected:
    void moveToward(Vec3 target, float speed, float dt);
    void fleeFrom(Vec3 threatPosition, float speed, float dt);
    void applyHorizontalFriction(float dt);
    void resetBody(Vec3 position);
    void setAutoJump(bool enabled, float jumpSpeed);
    void setTurnSpeed(float turnSpeed);
    void setHorizontalMovement(Vec3 direction, float speed, float acceleration, float dt);
    void requestJump(float jumpSpeed);
    void applyImpulse(Vec3 direction, float amplitude);

    void applyPhysics(float dt);

    virtual float knockbackResistance() const { return 0.0f; }

    Vec3 m_position {};
    Vec3 m_velocity {};
    Vec3 m_home {};
    Vec3 m_forward { 0.0f, 0.0f, 1.0f };

    const CylinderDimensions m_hitCylinderDimensions;
    std::int32_t m_health = 1;
    // Minecraft's invulnerableTime: after taking damage the entity is immune to further damage
    // and knockback for a short window, so rapid hits don't stack vertical knockback into a launch.
    float m_invulnerableTime = 0.0f;
    bool m_onGround = false;
    bool m_hasHorizontalMovement = false;
    bool m_horizontalBlocked = false;
    bool m_autoJump = false;
    float m_autoJumpSpeed = 6.0f;
    float m_turnSpeed = 8.0f;

private:
    bool collides(Vec3 position) const;
    float surfaceSlipperiness() const;

    const EntityId m_id;
    const CharacterKind m_kind;
    World& m_world;
};

} // namespace blocklab
