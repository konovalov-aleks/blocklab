#include "blocklab/characters/Character.h"

#include "blocklab/World.h"

#include <algorithm>
#include <cmath>

namespace blocklab {

namespace {

    float length2D(Vec3 value) { return std::sqrt(value.x * value.x + value.z * value.z); }

    float wrapAngle(float angle)
    {
        constexpr float Pi = 3.14159265358979323846f;
        while (angle > Pi)
            angle -= 2.0f * Pi;
        while (angle < -Pi)
            angle += 2.0f * Pi;
        return angle;
    }

    Vec3 normalized2D(Vec3 value)
    {
        const float length = length2D(value);
        if (length <= 0.0001f)
            return { };
        return { value.x / length, 0.0f, value.z / length };
    }

    constexpr float CharacterAcceleration = 24.0f;
    constexpr float CharacterGravity = -22.0f;
    constexpr float CharacterMaxFallSpeed = -40.0f;

} // namespace

Character::Character(EntityId id, CharacterKind kind, Vec3 position)
    : m_position(position)
    , m_home(position)
    , m_id(id)
    , m_kind(kind)
{
}

CharacterSnapshot Character::snapshot() const
{
    return {
        .id = m_id,
        .kind = m_kind,
        .position = m_position,
        .velocity = m_velocity,
        .forward = m_forward,
        .radius = m_radius,
        .height = m_height,
        .health = m_health,
    };
}

void Character::moveToward(World& world, Vec3 target, float speed, float dt)
{
    (void)world;
    const Vec3 direction = normalized2D(target - m_position);
    setHorizontalMovement(direction, speed, CharacterAcceleration, dt);
}

void Character::fleeFrom(World& world, Vec3 threatPosition, float speed, float dt)
{
    (void)world;
    const Vec3 direction = normalized2D(m_position - threatPosition);
    setHorizontalMovement(direction, speed, CharacterAcceleration, dt);
}

void Character::dampHorizontalMovement(float dt)
{
    const float damping = std::min(1.0f, CharacterAcceleration * dt);
    m_velocity.x += (0.0f - m_velocity.x) * damping;
    m_velocity.z += (0.0f - m_velocity.z) * damping;
}

void Character::resetBody(Vec3 position)
{
    m_position = position;
    m_velocity = { };
    m_onGround = false;
    m_hasHorizontalMovement = false;
    m_horizontalBlocked = false;
}

void Character::setCollisionShape(float radius, float height)
{
    m_radius = radius;
    m_height = height;
}

void Character::setAutoJump(bool enabled, float jumpSpeed)
{
    m_autoJump = enabled;
    m_autoJumpSpeed = jumpSpeed;
}

void Character::setTurnSpeed(float turnSpeed) { m_turnSpeed = turnSpeed; }

void Character::setHorizontalMovement(Vec3 direction, float speed, float acceleration, float dt)
{
    if (direction.x != 0.0f || direction.z != 0.0f) {
        const float currentYaw = std::atan2(m_forward.x, m_forward.z);
        const float targetYaw = std::atan2(direction.x, direction.z);
        const float delta = wrapAngle(targetYaw - currentYaw);
        const float maxTurn = m_turnSpeed * dt;
        const float nextYaw = currentYaw + std::clamp(delta, -maxTurn, maxTurn);
        m_forward = { std::sin(nextYaw), 0.0f, std::cos(nextYaw) };
    }

    const Vec3 targetVelocity = direction * speed;
    const float blend = std::min(1.0f, acceleration * dt);
    m_velocity.x += (targetVelocity.x - m_velocity.x) * blend;
    m_velocity.z += (targetVelocity.z - m_velocity.z) * blend;
    m_hasHorizontalMovement = true;
}

void Character::requestJump(float jumpSpeed)
{
    if (!m_onGround)
        return;

    m_velocity.y = jumpSpeed;
    m_onGround = false;
}

bool Character::collides(const World& world, Vec3 position) const
{
    return world.hasSolidBlockInArea(
        {
            floorToInt(position.x - m_radius),
            floorToInt(position.y),
            floorToInt(position.z - m_radius),
        },
        {
            floorToInt(position.x + m_radius),
            floorToInt(position.y + m_height),
            floorToInt(position.z + m_radius),
        });
}

bool Character::occupiesBlock(IVec3 block) const
{
    const int32_t minX = floorToInt(m_position.x - m_radius);
    const int32_t maxX = floorToInt(m_position.x + m_radius);
    const int32_t minY = floorToInt(m_position.y);
    const int32_t maxY = floorToInt(m_position.y + m_height);
    const int32_t minZ = floorToInt(m_position.z - m_radius);
    const int32_t maxZ = floorToInt(m_position.z + m_radius);
    return block.x >= minX && block.x <= maxX && block.y >= minY && block.y <= maxY && block.z >= minZ
        && block.z <= maxZ;
}

void Character::applyPhysics(World& world, float dt)
{
    m_horizontalBlocked = false;
    m_velocity.y += CharacterGravity * dt;
    m_velocity.y = std::max(m_velocity.y, CharacterMaxFallSpeed);

    Vec3 next = m_position;

    next.x += m_velocity.x * dt;
    if (collides(world, next)) {
        next.x = m_position.x;
        m_horizontalBlocked = true;
        if (m_autoJump && m_onGround && std::abs(m_velocity.x) > 0.01f)
            requestJump(m_autoJumpSpeed);
        m_velocity.x = 0.0f;
    }

    next.z += m_velocity.z * dt;
    if (collides(world, next)) {
        next.z = m_position.z;
        m_horizontalBlocked = true;
        if (m_autoJump && m_onGround && std::abs(m_velocity.z) > 0.01f)
            requestJump(m_autoJumpSpeed);
        m_velocity.z = 0.0f;
    }

    next.y += m_velocity.y * dt;
    m_onGround = false;
    if (collides(world, next)) {
        if (m_velocity.y < 0.0f)
            m_onGround = true;
        next.y = m_position.y;
        m_velocity.y = 0.0f;
    }

    m_position = next;
}

} // namespace blocklab
