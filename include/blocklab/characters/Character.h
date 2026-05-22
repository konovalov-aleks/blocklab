#pragma once

#include "blocklab/Math.h"

#include <cstdint>

namespace blocklab {

class World;

using EntityId = uint32_t;

enum class CharacterKind : uint8_t {
    Agent,
    Pig,
};

struct CharacterSnapshot {
    EntityId id = 0;
    CharacterKind kind = CharacterKind::Pig;
    Vec3 position {};
    Vec3 velocity {};
    Vec3 forward { 0.0f, 0.0f, 1.0f };
    float radius = 0.35f;
    float height = 0.8f;
    int32_t health = 1;
};

class Character {
public:
    Character(EntityId id, CharacterKind kind, Vec3 position);
    virtual ~Character() = default;

    EntityId id() const { return m_id; }
    CharacterKind kind() const { return m_kind; }
    const Vec3& position() const { return m_position; }
    const Vec3& velocity() const { return m_velocity; }
    int32_t health() const { return m_health; }
    bool onGround() const { return m_onGround; }
    bool horizontalBlocked() const { return m_horizontalBlocked; }
    bool isAlive() const { return m_health > 0; }

    CharacterSnapshot snapshot() const;

protected:
    void moveToward(World& world, Vec3 target, float speed, float dt);
    void fleeFrom(World& world, Vec3 threatPosition, float speed, float dt);
    void dampHorizontalMovement(float dt);
    void resetBody(Vec3 position);
    void setCollisionShape(float radius, float height);
    void setAutoJump(bool enabled, float jumpSpeed);
    void setTurnSpeed(float turnSpeed);
    void setHorizontalMovement(Vec3 direction, float speed, float acceleration, float dt);
    void requestJump(float jumpSpeed);
    void applyPhysics(World& world, float dt);
    bool occupiesBlock(IVec3 block) const;

    Vec3 m_position {};
    Vec3 m_velocity {};
    Vec3 m_home {};
    Vec3 m_forward { 0.0f, 0.0f, 1.0f };
    float m_radius = 0.35f;
    float m_height = 0.8f;
    int32_t m_health = 1;
    bool m_onGround = false;
    bool m_hasHorizontalMovement = false;
    bool m_horizontalBlocked = false;
    bool m_autoJump = false;
    float m_autoJumpSpeed = 6.0f;
    float m_turnSpeed = 8.0f;

private:
    EntityId m_id = 0;
    CharacterKind m_kind = CharacterKind::Pig;

    bool collides(const World& world, Vec3 position) const;
};

} // namespace blocklab
