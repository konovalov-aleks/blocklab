#include "Agent.h"

#include <blocklab/utility/Error.h>
#include <blocklab/utility/Math.h>
#include <world/World.h>

#include <algorithm>
#include <cmath>

namespace blocklab {

namespace {

    constexpr float s_eyeHeight = 1.62f;

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

    Block itemBlock(Item::Type item)
    {
        switch (item) {
        case Item::Type::Torch:
            return Block::Torch;
        case Item::Type::Dirt:
            return Block::Dirt;
        case Item::Type::Stone:
            return Block::Stone;
        case Item::Type::COUNT:
            break;
        }
        fatalError("Invalid item type: ", static_cast<int>(item));
    }

} // namespace

Agent::Agent()
    : m_character({ 0.0f, 14.0f, 0.0f })
{
}

void Agent::reset(Vec3 position)
{
    m_state = {};
    m_inventory = {};
    m_inventory[Inventory::hotbarSlotId(0)] = Item(Item::Type::Torch, Item::stackSize(Item::Type::Torch));
    m_inventory[Inventory::hotbarSlotId(1)] = Item(Item::Type::Dirt, Item::stackSize(Item::Type::Dirt));
    m_inventory[Inventory::hotbarSlotId(2)] = Item(Item::Type::Stone, Item::stackSize(Item::Type::Stone));
    m_character.resetBody(position);
    syncStateFromBody();
}

void Agent::step(World& world, const AgentAction& action, float dt)
{
    if (action.activeHotbarSlot) {
        if (!Inventory::isHotbarSlot(*action.activeHotbarSlot)) [[unlikely]]
            fatalError("Agent::step received an invalid active hotbar slot ID: ", static_cast<unsigned>(*action.activeHotbarSlot));
        m_inventory.setActiveHotbarSlot(*action.activeHotbarSlot);
    }

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
    constexpr float jumpSpeed = 7.42f;

    m_character.setHorizontalMovement(wishDir, moveSpeed, acceleration, dt);
    if (action.jump)
        m_character.requestJump(jumpSpeed);
    m_character.applyPhysics(world, dt);
    syncStateFromBody();
    pickDrops(world);
    interact(world, action);
    syncStateFromBody();
}

void Agent::pickDrops(World& world)
{
    const auto hitVolume = m_character.hitVolume();
    const auto& drops = world.drops();
    for (std::size_t i = 0; i < drops.size(); ++i) {
        const Drop& drop = drops[i];
        if (drop.alive() && collides(hitVolume, drop.hitVolume()))
            world.moveDropItemsToInventory(i, m_inventory);
    }
}

void Agent::interact(World& world, const AgentAction& action)
{
    if (!action.attack && !action.use)
        return;

    const Vec3 eye = m_character.position() + Vec3 { 0.0f, s_eyeHeight, 0.0f };
    const Vec3 forward = forwardFromAngles(m_state.yaw, m_state.pitch);

    IVec3 previousAir = floorToInt32(eye);
    constexpr float maxBlockInteractionDistance = 4.5f;
    for (float distance = 0.5f; distance <= maxBlockInteractionDistance; distance += 0.2f) {
        const Vec3 sample = eye + forward * distance;
        const IVec3 blockPos = floorToInt32(sample);
        const Block hitBlock = world.blockType(blockPos);
        if (hitBlock != Block::Air) {
            if (action.attack) {
                if (world.setBlock(blockPos, Block::Air, true))
                    ++m_state.blocksCollected;
                return;
            }

            if (!isSolidBlock(hitBlock)) {
                previousAir = blockPos;
                continue;
            }

            if (action.use) {
                Item& activeItem = m_inventory[m_inventory.activeHotbarSlot()];
                if (activeItem.empty())
                    return;

                const Block block = itemBlock(activeItem.type());
                if (isSolidBlock(block) && m_character.occupiesBlock(previousAir))
                    return;
                if (block != Block::Torch || previousAir == blockPos + IVec3 { 0, 1, 0 }) {
                    if (world.setBlock(previousAir, block)) {
                        activeItem.remove(1);
                        ++m_state.blocksPlaced;
                    }
                }
            }
            return;
        }
        previousAir = blockPos;
    }
}

void Agent::syncStateFromBody()
{
    m_state.position = m_character.position();
    m_state.velocity = m_character.velocity();
    m_state.onGround = m_character.onGround();
}

} // namespace blocklab
