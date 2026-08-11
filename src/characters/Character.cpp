#include "Character.h"

#include <blocklab/utility/Math.h>
#include <world/World.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace blocklab {

namespace {

    float length2D(Vec3 value) { return std::sqrt(value.x * value.x + value.z * value.z); }

    Vec3 normalized2D(Vec3 value)
    {
        const float length = length2D(value);
        if (length <= 0.0001f)
            return {};
        return { value.x / length, 0.0f, value.z / length };
    }

    constexpr float s_tickDuration = 0.05f;      // one Minecraft tick (20 TPS)
    constexpr float s_baseDrag = 0.91f;           // Minecraft base horizontal drag (air slipperiness == 1.0)
    constexpr float s_airControlFactor = 0.02f;   // Minecraft airborne movement factor (air control strength)
    constexpr float CharacterAcceleration = 24.0f;
    constexpr float CharacterGravity = -22.0f;
    constexpr float CharacterMaxFallSpeed = -40.0f;

    // Minecraft knockback vertical: Entity.takeKnockback adds a fixed +0.4 blocks/tick upward
    // whenever the strength is positive. Minecraft's fall model is a -0.08 blocks/tick gravity
    // increment plus a 0.98 vertical drag per tick (initial acceleration ~32 m/s^2, terminal
    // 3.92 bpt), while this game uses a plain constant -22 m/s^2 gravity. Under that weaker
    // model the raw 0.4 bpt (= 8 m/s) would hop ~1.45 m, so 7.0 m/s keeps the original's
    // ~1.1-block hop. Like the agent's jump speed (7.42 -> 1.25 m, MC's jump height), the
    // vertical constants are scaled so observable results match Minecraft.
    constexpr float s_knockbackVertical = 7.0f;
    // Minecraft LivingEntity.invulnerableTime: 20 ticks of immunity after taking damage, so
    // repeated hits within the window do no damage and no knockback (otherwise rapid clicks or a
    // spamming agent would stack the vertical hop and launch the target into the air).
    constexpr float s_invulnerableDuration = 1.0f;

} // namespace

Character::Character(World& world, EntityId id, CharacterKind kind, Vec3 position,
                     CylinderDimensions hitCylinder)
    : m_position(position)
    , m_home(position)
    , m_hitCylinderDimensions(hitCylinder)
    , m_id(id)
    , m_kind(kind)
    , m_world(world)
{
}

void Character::moveToward(Vec3 target, float speed, float dt)
{
    const Vec3 direction = normalized2D(target - m_position);
    if (glm::length2(direction) <= sqr(std::numeric_limits<float>::epsilon()))
        return;
    setHorizontalMovement(direction, speed, CharacterAcceleration, dt);
}

void Character::fleeFrom(Vec3 threatPosition, float speed, float dt)
{
    const Vec3 direction = normalized2D(m_position - threatPosition);
    if (glm::length2(direction) <= sqr(std::numeric_limits<float>::epsilon()))
        return;
    setHorizontalMovement(direction, speed, CharacterAcceleration, dt);
}

void Character::applyHorizontalFriction(float dt)
{
    // Exponential horizontal decay, frame-rate independent. Ground drag is the surface
    // slipperiness scaled by Minecraft's base 0.91; in the air slipperiness is 1.0,
    // which yields the plain 0.91 air drag.
    const float dragStep = std::pow(surfaceSlipperiness() * s_baseDrag, dt / s_tickDuration);
    m_velocity.x *= dragStep;
    m_velocity.z *= dragStep;
}

void Character::resetBody(Vec3 position)
{
    m_position = position;
    m_velocity = {};
    m_onGround = false;
    m_hasHorizontalMovement = false;
    m_horizontalBlocked = false;
}

void Character::setAutoJump(bool enabled, float jumpSpeed)
{
    m_autoJump = enabled;
    m_autoJumpSpeed = jumpSpeed;
}

void Character::setTurnSpeed(float turnSpeed) { m_turnSpeed = turnSpeed; }

void Character::setHorizontalMovement(Vec3 wishDir, float speed, float acceleration, float dt)
{
    const float tickFactor = dt / s_tickDuration;
    const float dragStep = std::pow(surfaceSlipperiness() * s_baseDrag, tickFactor);

    // 1. Exponential decay of the existing horizontal velocity (Minecraft-like friction).
    m_velocity.x *= dragStep;
    m_velocity.z *= dragStep;

    // No movement input: keep decaying and let inertia carry the character to a halt.
    if (glm::length2(wishDir) <= sqr(std::numeric_limits<float>::epsilon())) {
        m_hasHorizontalMovement = false;
        return;
    }

    // 2. Smoothly rotate the facing toward the movement direction (used for rendering).
    const float currentYaw = std::atan2(m_forward.x, m_forward.z);
    const float targetYaw = std::atan2(wishDir.x, wishDir.z);
    const float nextYaw = currentYaw + std::clamp(wrapAngle(targetYaw - currentYaw), -m_turnSpeed * dt, m_turnSpeed * dt);
    m_forward = { std::sin(nextYaw), 0.0f, std::cos(nextYaw) };

    // 3. Impulse that keeps the velocity at the target speed against drag:
    //    terminal velocity = targetSpeed  =>  impulse = targetSpeed * (1 - dragStep).
    const float idealTerminalImpulse = speed * (1.0f - dragStep);
    float appliedAcceleration;
    if (m_onGround) {
        // Ground control reaches terminal velocity progressively each tick.
        appliedAcceleration = idealTerminalImpulse;
    } else {
        // Air control is much weaker than ground control (Minecraft airborne factor),
        // capped so the character can steer mid-air but never accelerate past its
        // ground speed.
        appliedAcceleration = std::min(acceleration * s_airControlFactor * tickFactor, idealTerminalImpulse);
    }

    // 4. Add the impulse in the wished direction.
    m_velocity.x += wishDir.x * appliedAcceleration;
    m_velocity.z += wishDir.z * appliedAcceleration;
    m_hasHorizontalMovement = true;
}

void Character::requestJump(float jumpSpeed)
{
    if (!grounded())
        return;

    m_velocity.y = jumpSpeed;
    m_onGround = false;
}

bool Character::collides(Vec3 position) const
{
    return world().hasSolidBlockInArea(
        {
            floorToInt32(position.x - m_hitCylinderDimensions.radius),
            floorToInt32(position.y),
            floorToInt32(position.z - m_hitCylinderDimensions.radius),
        },
        {
            floorToInt32(position.x + m_hitCylinderDimensions.radius),
            floorToInt32(position.y + m_hitCylinderDimensions.height),
            floorToInt32(position.z + m_hitCylinderDimensions.radius),
        });
}

float Character::surfaceSlipperiness() const
{
    // In the air the character is not touching any surface, so it always uses the
    // plain 0.91 air drag (slipperiness == 1.0), like Minecraft.
    if (!m_onGround)
        return 1.0f;

    // The feet sit on the top face of the block below, so nudge the y by a small
    // epsilon to land on that block's cell rather than the one at the feet level.
    const IVec3 below {
        floorToInt32(m_position.x),
        floorToInt32(m_position.y - 0.001f),
        floorToInt32(m_position.z),
    };
    return slipperiness(m_world.blockType(below));
}

bool Character::occupiesBlock(IVec3 block) const
{
    const std::int32_t minX = floorToInt32(m_position.x - m_hitCylinderDimensions.radius);
    const std::int32_t maxX = floorToInt32(m_position.x + m_hitCylinderDimensions.radius);
    const std::int32_t minY = floorToInt32(m_position.y);
    const std::int32_t maxY = floorToInt32(m_position.y + m_hitCylinderDimensions.height);
    const std::int32_t minZ = floorToInt32(m_position.z - m_hitCylinderDimensions.radius);
    const std::int32_t maxZ = floorToInt32(m_position.z + m_hitCylinderDimensions.radius);
    return block.x >= minX && block.x <= maxX
        && block.y >= minY && block.y <= maxY
        && block.z >= minZ && block.z <= maxZ;
}

void Character::onHit(Vec3 direction, float power, int damage)
{
    assert(std::abs(glm::length(direction) - 1.0f) < 0.0001f);
    // MC invulnerability window: a target already recently hit ignores further damage and
    // knockback until the window lapses, so consecutive hits can't stack the vertical hop.
    if (m_invulnerableTime > 0.0f)
        return;

    m_health -= damage;
    if (!alive()) {
        world().onCharacterKilled();
        world().throwDrop(floorToInt32(position()), Item::Type::Torch);
        return;
    }
    m_invulnerableTime = s_invulnerableDuration;

    // Minecraft knockback (Entity.takeKnockback): push along the attacker's horizontal facing
    // by `power`, plus a fixed upward hop whenever the strength is positive. `power` is in m/s
    // and matches the base-punch strength of 0.4 blocks/tick = 8 m/s. Resistance scales the
    // strength; the hop applies only while strength is positive (also like Minecraft).
    const float strength = power * (1.0f - knockbackResistance());
    if (strength <= 0.0f)
        return;

    const Vec3 horizontal { direction.x, 0.0f, direction.z };
    const float horizontalLength = length2D(horizontal);
    if (horizontalLength > 0.0001f)
        applyImpulse(horizontal / horizontalLength, strength);
    m_velocity.y += s_knockbackVertical;

    m_onGround = false;
}

void Character::applyImpulse(Vec3 direction, float amplitude)
{
    if (amplitude < 0.0001f)
        return;

    assert(std::abs(glm::length(direction) - 1.0f) < 0.0001f);
    m_velocity += direction * amplitude;
}

void Character::applyPhysics(float dt)
{
    m_invulnerableTime = std::max(0.0f, m_invulnerableTime - dt);

    m_horizontalBlocked = false;

    Vec3 next = m_position;

    next.x += m_velocity.x * dt;
    if (collides(next)) {
        next.x = m_position.x;
        m_horizontalBlocked = true;
        if (m_autoJump && m_onGround && std::abs(m_velocity.x) > 0.01f)
            requestJump(m_autoJumpSpeed);
        m_velocity.x = 0.0f;
    }

    next.z += m_velocity.z * dt;
    if (collides(next)) {
        next.z = m_position.z;
        m_horizontalBlocked = true;
        if (m_autoJump && m_onGround && std::abs(m_velocity.z) > 0.01f)
            requestJump(m_autoJumpSpeed);
        m_velocity.z = 0.0f;
    }

    const float previousVelocityY = m_velocity.y;
    m_velocity.y = std::max(m_velocity.y + CharacterGravity * dt, CharacterMaxFallSpeed);
    next.y += 0.5f * (previousVelocityY + m_velocity.y) * dt;
    m_onGround = false;
    if (collides(next)) {
        if (m_velocity.y < 0.0f) {
            m_onGround = true;
            // Land flush on the top face of the block below, like Minecraft. Without this the
            // character hovers a few millimetres above the surface: it stays grounded (the box
            // dips into the block each frame and is pushed back) but any ground query at the
            // feet (e.g. surface slipperiness) resolves to the air cell above the block instead
            // of the block itself, which silently swapped ground friction for air drag.
            next.y = static_cast<float>(floorToInt32(next.y) + 1);
        } else {
            // Hitting a ceiling: keep the pre-penetration height.
            next.y = m_position.y;
        }
        m_velocity.y = 0.0f;
    }

    m_position = next;
}

} // namespace blocklab
