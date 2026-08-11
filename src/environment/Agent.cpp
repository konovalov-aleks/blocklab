#include "Agent.h"

#include <algorithms/Raycast.h>
#include <characters/Character.h>
#include <blocklab/utility/Error.h>
#include <blocklab/utility/Math.h>
#include <world/World.h>

#include <glm/gtx/norm.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <variant>

namespace blocklab {

namespace {

    struct BlockTarget {
        IVec3 position;
        std::optional<IVec3> normal;
    };
    using CharacterTarget = std::reference_wrapper<Character>;
    using CursorTarget = std::variant<std::monostate, BlockTarget, CharacterTarget>;

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

    Block inventoryItemBlock(Item::Type item)
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

    constexpr float s_maxBlockInteractionDistance = 4.5f;
    constexpr float s_maxCharacterInteractionDistance = 3.0f;

    CursorTarget cursorTarget(World& world, Vec3 position, Vec3 direction)
    {
        float blockDistance = std::numeric_limits<float>::max();
        std::optional<IVec3> blockPos;
        std::optional<IVec3> blockNormal;

        raycast(position, direction,
            [&world, &blockDistance, &blockPos, &blockNormal](const RaycastCbParams& p) {
                if (p.distance > s_maxBlockInteractionDistance)
                    return RaycastCommand::Break;

                const Block hitBlock = world.blockType(p.pos);
                if (hitBlock != Block::Air) {
                    blockDistance = p.distance;
                    blockPos = p.pos;
                    blockNormal = p.normal;
                    return RaycastCommand::Break;
                }
                return RaycastCommand::Continue;
            }
        );

        // Distance at which the ray first enters any character's body, regardless of reach: a
        // body in the way occludes what's behind it even when it is out of melee range.
        float occlusionDistance = std::numeric_limits<float>::max();

        float closestCharacterSqr = std::numeric_limits<float>::max();
        Character* character = nullptr;
        const float maxCharacterDistanceSqr = sqr(s_maxCharacterInteractionDistance);
        for (const std::unique_ptr<NPC>& c : world.characters()) {
            assert(c);
            assert(c->alive());
            const std::optional<float> entry = rayCylinderEntryDistance({ position, direction }, c->hitVolume());
            if (!entry)
                continue;
            occlusionDistance = std::min(occlusionDistance, *entry);
            const float dx = c->position().x - position.x;
            const float dz = c->position().z - position.z;
            if (dx * dx + dz * dz > maxCharacterDistanceSqr)
                continue;
            const float distanceSqr = glm::length2(position - c->position());
            if (distanceSqr < closestCharacterSqr) {
                closestCharacterSqr = distanceSqr;
                character = c.get();
            }
        }

        // An entity body in the way shields the block behind it
        if (occlusionDistance < blockDistance)
            blockDistance = std::numeric_limits<float>::max();

        if (blockPos && blockDistance * blockDistance < closestCharacterSqr)
            return BlockTarget { .normal = blockNormal, .position = *blockPos };
        if (character)
            return *character;

        return {};
    }

} // namespace

Agent::Agent(World& world)
    : m_character(world, { 0.0f, 14.0f, 0.0f })
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

void Agent::step(const AgentAction& action, float dt)
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
    if (glm::length2(wishDir) > 1.0E-10f)
        wishDir = glm::normalize(wishDir);

    constexpr float moveSpeed = 5.0f;
    constexpr float acceleration = 28.0f;
    constexpr float jumpSpeed = 7.42f;

    m_character.setHorizontalMovement(wishDir, moveSpeed, acceleration, dt);
    if (action.jump)
        m_character.requestJump(jumpSpeed);
    m_character.applyPhysics(dt);
    syncStateFromBody();
    pickDrops();
    interact(action);
    syncStateFromBody();
}

void Agent::pickDrops()
{
    const auto hitVolume = m_character.hitVolume();
    const auto& drops = world().drops();
    for (std::size_t i = 0; i < drops.size(); ++i) {
        const Drop& drop = drops[i];
        if (drop.alive() && collides(hitVolume, drop.hitVolume()))
            world().moveDropItemsToInventory(i, m_inventory);
    }
}

void Agent::interact(const AgentAction& action)
{
    if (!action.attack && !action.use)
        return;

    const Vec3 eye = m_character.position() + Vec3 { 0.0f, s_eyeHeight, 0.0f };
    const Vec3 forward = forwardFromAngles(m_state.yaw, m_state.pitch);

    CursorTarget target = cursorTarget(world(), eye, forward);
    if (action.attack) {
        if (BlockTarget* block = std::get_if<BlockTarget>(&target)) {
            assert(world().blockType(block->position) != Block::Air);
            if (world().setBlock(block->position, Block::Air, true))
                ++m_state.blocksCollected;
        } else if (CharacterTarget* c = std::get_if<CharacterTarget>(&target)) {
            // Minecraft base-punch knockback strength: 0.4 blocks/tick = 8 m/s. Sprint hits are
            // 0.9 bpt = 18 m/s, which is what the previous arbitrary 20 m/s approximated and why
            // the pig used to fly ~5.7 m.
            constexpr float hitKnockback = 8.0f;
            constexpr int damage = 1;
            c->get().onHit(forward, hitKnockback, damage);
        }
    } else if (action.use) {
        if (BlockTarget* bt = std::get_if<BlockTarget>(&target)) {
            Item& activeItem = m_inventory[m_inventory.activeHotbarSlot()];
            if (activeItem.empty())
                return;
            const Block itemBlock = inventoryItemBlock(activeItem.type());

            const Block block = world().blockType(bt->position);
            if (!isSolidBlock(block))
                return;

            const IVec3 prevBlock = bt->normal ? bt->position + *bt->normal
                                               : bt->position; // the agent is inside a solid block
            assert(world().isInsideLoadedCache(prevBlock));
            if (isSolidBlock(itemBlock) && m_character.occupiesBlock(prevBlock))
                return;

            if (itemBlock == Block::Torch && bt->normal != IVec3 { 0, 1, 0 }) {
                // currently Torch can be only placed on a top block surface
                return;
            }

            if (world().setBlock(prevBlock, itemBlock)) {
                activeItem.remove(1);
                ++m_state.blocksPlaced;
            }
        }
    }
}

void Agent::syncStateFromBody()
{
    m_state.position = m_character.position();
    m_state.velocity = m_character.velocity();
    m_state.grounded = m_character.grounded();
}

} // namespace blocklab
