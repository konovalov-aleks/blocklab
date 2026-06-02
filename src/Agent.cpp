#include "blocklab/Agent.h"

#include "blocklab/Math.h"
#include "blocklab/World.h"

#include <algorithm>
#include <cmath>

namespace blocklab {

namespace {

    constexpr float EyeHeight = 1.62f;
    constexpr float AgentHeight = 1.8f;
    constexpr float Radius = 0.32f;

    Vec3 forwardFromYaw(float yaw) { return { std::sin(yaw), 0.0f, std::cos(yaw) }; }

    Vec3 rightFromYaw(float yaw) { return { std::cos(yaw), 0.0f, -std::sin(yaw) }; }

    Vec3 forwardFromAngles(float yaw, float pitch)
    {
        const float pitchCos = std::cos(pitch);
        return {
            std::sin(yaw) * pitchCos,
            std::sin(pitch),
            std::cos(yaw) * pitchCos,
        };
    }

} // namespace

Agent::Agent()
    : Character(0, CharacterKind::Agent, { 0.0f, 14.0f, 0.0f })
{
    setCollisionShape(Radius, AgentHeight);
}

void Agent::reset(Vec3 position)
{
    m_state = {};
    resetBody(position);
    syncStateFromBody();
}

void Agent::step(World& world, const AgentAction& action, float dt)
{
    m_state.yaw += action.yawDelta;
    m_state.pitch = std::clamp(m_state.pitch + action.pitchDelta, -Pi / 2.0f + 0.05f, Pi / 2.0f - 0.05f);
    if (m_state.yaw > Pi)
        m_state.yaw -= 2.0f * Pi;
    else if (m_state.yaw < -Pi)
        m_state.yaw += 2.0f * Pi;

    Vec3 wishDir = forwardFromYaw(m_state.yaw) * action.forward + rightFromYaw(m_state.yaw) * action.right;
    if (glm::length(wishDir) > 0.00001f)
        wishDir = glm::normalize(wishDir);

    constexpr float moveSpeed = 5.0f;
    constexpr float acceleration = 28.0f;
    constexpr float jumpSpeed = 7.0f;

    setHorizontalMovement(wishDir, moveSpeed, acceleration, dt);
    if (action.jump)
        requestJump(jumpSpeed);
    applyPhysics(world, dt);
    syncStateFromBody();
    interact(world, action);
    syncStateFromBody();
}

void Agent::interact(World& world, const AgentAction& action)
{
    if (!action.dig && !action.place)
        return;

    const Vec3 eye = position() + Vec3 { 0.0f, EyeHeight, 0.0f };
    const Vec3 forward = forwardFromAngles(m_state.yaw, m_state.pitch);

    IVec3 previousAir { floorToInt32(eye.x), floorToInt32(eye.y), floorToInt32(eye.z) };
    for (float distance = 0.5f; distance <= 4.0f; distance += 0.2f) {
        const Vec3 sample = eye + forward * distance;
        const IVec3 blockPos { floorToInt32(sample.x), floorToInt32(sample.y), floorToInt32(sample.z) };
        if (world.isSolid(blockPos)) {
            if (action.dig) {
                world.setBlock(blockPos, Block::Air);
                ++m_state.blocksCollected;
            } else if (action.place && !occupiesBlock(previousAir)) {
                world.setBlock(previousAir, Block::Dirt);
                ++m_state.blocksPlaced;
            }
            return;
        }
        previousAir = blockPos;
    }
}

void Agent::syncStateFromBody()
{
    m_state.position = position();
    m_state.velocity = velocity();
    m_state.onGround = onGround();
}

} // namespace blocklab
