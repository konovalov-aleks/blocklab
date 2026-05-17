#include "blocklab/Agent.h"

#include "blocklab/World.h"

#include <algorithm>
#include <cmath>

namespace blocklab {

namespace {

    constexpr float Pi = 3.14159265358979323846f;
    constexpr float HalfPi = Pi * 0.5f;
    constexpr float EyeHeight = 1.62f;
    constexpr float AgentHeight = 1.8f;
    constexpr float Radius = 0.32f;

    bool collides(const World& world, Vec3 position)
    {
        const int32_t minX = floorToInt(position.x - Radius);
        const int32_t maxX = floorToInt(position.x + Radius);
        const int32_t minY = floorToInt(position.y);
        const int32_t maxY = floorToInt(position.y + AgentHeight);
        const int32_t minZ = floorToInt(position.z - Radius);
        const int32_t maxZ = floorToInt(position.z + Radius);

        for (int32_t z = minZ; z <= maxZ; ++z) {
            for (int32_t y = minY; y <= maxY; ++y) {
                for (int32_t x = minX; x <= maxX; ++x) {
                    if (world.isSolid(x, y, z))
                        return true;
                }
            }
        }
        return false;
    }

    bool occupiesBlock(Vec3 position, IVec3 block)
    {
        const int32_t minX = floorToInt(position.x - Radius);
        const int32_t maxX = floorToInt(position.x + Radius);
        const int32_t minY = floorToInt(position.y);
        const int32_t maxY = floorToInt(position.y + AgentHeight);
        const int32_t minZ = floorToInt(position.z - Radius);
        const int32_t maxZ = floorToInt(position.z + Radius);
        return block.x >= minX && block.x <= maxX && block.y >= minY && block.y <= maxY && block.z >= minZ
            && block.z <= maxZ;
    }

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

void Agent::reset(Vec3 position)
{
    m_state = { };
    m_state.position = position;
}

void Agent::step(World& world, const AgentAction& action, float dt)
{
    m_state.yaw += action.yawDelta;
    m_state.pitch = std::clamp(m_state.pitch + action.pitchDelta, -HalfPi + 0.05f, HalfPi - 0.05f);
    if (m_state.yaw > Pi)
        m_state.yaw -= 2.0f * Pi;
    else if (m_state.yaw < -Pi)
        m_state.yaw += 2.0f * Pi;

    Vec3 wishDir = forwardFromYaw(m_state.yaw) * action.forward + rightFromYaw(m_state.yaw) * action.right;
    if (glm::length(wishDir) > 0.00001f)
        wishDir = glm::normalize(wishDir);

    constexpr float moveSpeed = 5.0f;
    constexpr float acceleration = 28.0f;
    constexpr float gravity = -22.0f;
    constexpr float jumpSpeed = 7.0f;

    const Vec3 targetVelocity = wishDir * moveSpeed;
    m_state.velocity.x += (targetVelocity.x - m_state.velocity.x) * std::min(1.0f, acceleration * dt);
    m_state.velocity.z += (targetVelocity.z - m_state.velocity.z) * std::min(1.0f, acceleration * dt);

    if (action.jump && m_state.onGround) {
        m_state.velocity.y = jumpSpeed;
        m_state.onGround = false;
    }

    m_state.velocity.y += gravity * dt;
    m_state.velocity.y = std::max(m_state.velocity.y, -40.0f);

    Vec3 next = m_state.position;

    next.x += m_state.velocity.x * dt;
    if (collides(world, next)) {
        next.x = m_state.position.x;
        m_state.velocity.x = 0.0f;
    }

    next.z += m_state.velocity.z * dt;
    if (collides(world, next)) {
        next.z = m_state.position.z;
        m_state.velocity.z = 0.0f;
    }

    next.y += m_state.velocity.y * dt;
    m_state.onGround = false;
    if (collides(world, next)) {
        if (m_state.velocity.y < 0.0f)
            m_state.onGround = true;
        next.y = m_state.position.y;
        m_state.velocity.y = 0.0f;
    }

    m_state.position = next;
    interact(world, action);
}

void Agent::interact(World& world, const AgentAction& action)
{
    if (!action.dig && !action.place)
        return;

    const Vec3 eye = m_state.position + Vec3 { 0.0f, EyeHeight, 0.0f };
    const Vec3 forward = forwardFromAngles(m_state.yaw, m_state.pitch);

    IVec3 previousAir { floorToInt(eye.x), floorToInt(eye.y), floorToInt(eye.z) };
    for (float distance = 0.5f; distance <= 4.0f; distance += 0.2f) {
        const Vec3 sample = eye + forward * distance;
        const IVec3 blockPos { floorToInt(sample.x), floorToInt(sample.y), floorToInt(sample.z) };
        if (world.isSolid(blockPos.x, blockPos.y, blockPos.z)) {
            if (action.dig) {
                world.setBlock(blockPos.x, blockPos.y, blockPos.z, Block::Air);
                ++m_state.blocksCollected;
            } else if (action.place && !occupiesBlock(m_state.position, previousAir)) {
                world.setBlock(previousAir.x, previousAir.y, previousAir.z, Block::Dirt);
                ++m_state.blocksPlaced;
            }
            return;
        }
        previousAir = blockPos;
    }
}

} // namespace blocklab
