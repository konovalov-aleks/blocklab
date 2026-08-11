#include "WorldTestUtils.h"

#include <blocklab/environment/Environment.h>
#include <blocklab/gpu/vulkan/Vulkan.h>
#include <characters/Character.h>
#include <characters/PigCharacter.h>
#include <blocklab/graphics/Renderer.h>
#include <blocklab/inventory/Inventory.h>
#include <environment/Agent.h>
#include <environment/EnvInstance.h>
#include <world/World.h>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <sstream>
#include <vector>

namespace blocklab::test {

class EnvironmentInternalAccessTestHelper {
public:
    static World& world(Environment& env, std::size_t i) { return env.m_instances[i].world; }
    static const AgentState& agentState(Environment& env, std::size_t i) { return env.m_instances[i].agent.state(); }
};

namespace {
    constexpr float TestStepDt = 1.0f / 60.0f;

    struct TestRenderContext {
        explicit TestRenderContext(std::uint32_t batchSize = 1)
            : vkInstance(false)
            , vk(std::make_shared<Vulkan>(vkInstance))
            , renderer(*vk, { .batchSize = batchSize })
        {
        }

        VulkanInstance vkInstance;
        std::shared_ptr<Vulkan> vk;
        Renderer renderer;
    };

    std::uint32_t itemCount(const Inventory& inventory, Item::Type type)
    {
        std::uint32_t count = 0;
        const auto addSlots = [&](std::span<const Item> slots) {
            for (const Item& item : slots) {
                if (!item.empty() && item.type() == type)
                    count += item.count();
            }
        };
        addSlots(inventory.hotbarSlots());
        addSlots(inventory.storageSlots());
        return count;
    }

    void setupEmptyTestArea(World& world)
    {
        world.resetSeed(17);
        updateWorldCacheAt(world, { 0, 14, 0 });
        clearBlocks(world, { -3, 0, -3 }, { 7, 22, 7 });
    }

    void stepAgent(World& world, Agent& agent, const AgentAction& action, int steps = 1, float dt = TestStepDt)
    {
        for (int i = 0; i < steps; ++i) {
            agent.step(action, dt);
            world.update(dt, agent.state().position);
        }
    }

    AgentAction lookDownAttack()
    {
        AgentAction action;
        action.pitchDelta = -Pi;
        action.attack = true;
        return action;
    }

} // namespace

#define BLOCKLAB_INFO_DROPS(world) \
    std::ostringstream blocklabDropInfo; \
    blocklabDropInfo << "drops = " << (world).drops().size(); \
    for (std::size_t blocklabDropIndex = 0; blocklabDropIndex < (world).drops().size(); ++blocklabDropIndex) { \
        const Drop& blocklabDrop = (world).drops()[blocklabDropIndex]; \
        const Vec3 blocklabDropPosition = blocklabDrop.position(); \
        blocklabDropInfo << "\ndrop[" << blocklabDropIndex << "]" \
            << " alive=" << blocklabDrop.alive() \
            << " x=" << blocklabDropPosition.x \
            << " y=" << blocklabDropPosition.y \
            << " z=" << blocklabDropPosition.z; \
    } \
    INFO(blocklabDropInfo.str())

TEST_CASE("Environment can step a moving agent", "[environment]")
{
    TestRenderContext renderContext;
    Environment env(renderContext.renderer, 1);
    env.reset(42);
    const AgentState initialState = EnvironmentInternalAccessTestHelper::agentState(env, 0);
    for (int i = 0; i < 120; ++i) {
        AgentAction action;
        action.forward = 1.0f;
        action.yawDelta = 0.01f;
        action.pitchDelta = 0.001f;
        const AgentAction actions[] { action };
        const StepResult result = env.step(actions).front();
        CHECK(result.reward > -10.0f);
    }
    const AgentState finalState = EnvironmentInternalAccessTestHelper::agentState(env, 0);
    const float dx = finalState.position.x - initialState.position.x;
    const float dz = finalState.position.z - initialState.position.z;
    CHECK(std::sqrt(dx * dx + dz * dz) > 0.1f);
}

TEST_CASE("Environment reset rebuilds cache around the initial spawn", "[environment]")
{
    TestRenderContext renderContext;
    Environment env(renderContext.renderer, 1);
    env.reset(42);
    AgentAction action;
    action.forward = 1.0f;
    const AgentAction actions[] { action };
    // Move the agent into another generated cache region, then verify reset rebuilds the cache around the initial
    // spawn instead of reusing the previous agent-centered region.
    for (int i = 0; i < 500; ++i)
        env.step(actions);

    env.reset(43);

    const AgentState& state = EnvironmentInternalAccessTestHelper::agentState(env, 0);
    CHECK(state.position.x == 0.5f);
    CHECK(state.position.z == 0.5f);
    CHECK(EnvironmentInternalAccessTestHelper::world(env, 0).characters().size() == 32);
}

TEST_CASE("Agent cannot place a block into its own body", "[environment]")
{
    TestRenderContext renderContext;
    Environment placeEnv(renderContext.renderer, 1);
    World& world = EnvironmentInternalAccessTestHelper::world(placeEnv, 0);
    placeEnv.reset(7);
    const AgentState& state = EnvironmentInternalAccessTestHelper::agentState(placeEnv, 0);
    const std::int32_t occupiedX = floorToInt32(state.position.x);
    const std::int32_t occupiedY = floorToInt32(state.position.y + 1.0f);
    const std::int32_t occupiedZ = floorToInt32(state.position.z);
    world.setBlock({ occupiedX, occupiedY, occupiedZ + 2 }, Block::Dirt);

    AgentAction placeIntoSelf;
    placeIntoSelf.pitchDelta = 1.2f;
    placeIntoSelf.activeHotbarSlot = Inventory::hotbarSlotId(1);
    placeIntoSelf.use = true;
    const AgentAction placeActions[] { placeIntoSelf };
    placeEnv.step(placeActions);
    CHECK(world.blockType({ occupiedX, occupiedY, occupiedZ }) == Block::Air);
}

TEST_CASE("Agent interaction ray can leave the world vertically", "[environment]")
{
    World world;
    world.resetSeed(17);
    updateWorldCacheAt(world, { 0, 0, 0 });

    // Keep the downward interaction ray in air until it crosses below World::s_minY.
    clearBlocks(world, { -1, World::s_minY, -1 }, { 3, 4, 3 });

    Agent agent(world);
    agent.reset({ 0.5f, 0.0f, 0.5f });

    AgentAction action;
    action.pitchDelta = -Pi;
    action.attack = true;

    agent.step(action, 0.0f);
    CHECK(agent.state().blocksCollected == 0);
}

TEST_CASE("Agent picks up a mined block after falling through it", "[environment][inventory][drop]")
{
    World world;
    setupEmptyTestArea(world);
    world.setBlock({ 0, 10, 0 }, Block::Stone);
    world.setBlock({ 0, 13, 0 }, Block::Dirt);

    Agent agent(world);
    agent.reset({ 0.5f, 14.0f, 0.5f });
    REQUIRE(itemCount(agent.inventory(), Item::Type::Dirt) == 64);

    stepAgent(world, agent, lookDownAttack(), 1, 0.0f);
    REQUIRE(world.blockType({ 0, 13, 0 }) == Block::Air);
    CHECK(itemCount(agent.inventory(), Item::Type::Dirt) == 64);

    stepAgent(world, agent, {}, 120);
    INFO("agent y = " << agent.state().position.y);
    BLOCKLAB_INFO_DROPS(world);
    CHECK(itemCount(agent.inventory(), Item::Type::Dirt) == 65);
    CHECK(world.drops().empty());
}

TEST_CASE("Agent picks up multiple drops with their original item types", "[environment][inventory][drop]")
{
    World world;
    setupEmptyTestArea(world);
    world.setBlock({ 0, 12, 0 }, Block::Stone);
    world.setBlock({ 0, 14, 0 }, Block::Stone);
    world.setBlock({ 0, 15, 0 }, Block::Dirt);
    world.setBlock({ 0, 16, 0 }, Block::Dirt);
    world.setBlock({ 0, 17, 0 }, Block::Torch);

    Agent agent(world);
    agent.reset({ 0.5f, 17.0f, 0.5f });
    REQUIRE(itemCount(agent.inventory(), Item::Type::Torch) == 64);
    REQUIRE(itemCount(agent.inventory(), Item::Type::Dirt) == 64);
    REQUIRE(itemCount(agent.inventory(), Item::Type::Stone) == 64);

    AgentAction attack = lookDownAttack();
    stepAgent(world, agent, attack, 1, 0.0f);
    stepAgent(world, agent, attack, 1, 0.0f);
    stepAgent(world, agent, attack, 1, 0.0f);
    stepAgent(world, agent, attack, 1, 0.0f);

    REQUIRE(world.blockType({ 0, 17, 0 }) == Block::Air);
    REQUIRE(world.blockType({ 0, 16, 0 }) == Block::Air);
    REQUIRE(world.blockType({ 0, 15, 0 }) == Block::Air);
    REQUIRE(world.blockType({ 0, 14, 0 }) == Block::Air);

    stepAgent(world, agent, {}, 180);
    BLOCKLAB_INFO_DROPS(world);
    CHECK(itemCount(agent.inventory(), Item::Type::Torch) == 65);
    CHECK(itemCount(agent.inventory(), Item::Type::Dirt) == 66);
    CHECK(itemCount(agent.inventory(), Item::Type::Stone) == 65);
    CHECK(world.drops().empty());
}

TEST_CASE("Agent use places the active hotbar item and consumes it", "[environment][inventory]")
{
    World world;
    setupEmptyTestArea(world);
    world.setBlock({ 0, 10, 0 }, Block::Stone);
    world.setBlock({ 0, 15, 2 }, Block::Stone);

    Agent agent(world);
    agent.reset({ 0.5f, 14.0f, 0.5f });
    REQUIRE(itemCount(agent.inventory(), Item::Type::Dirt) == 64);

    AgentAction action;
    action.activeHotbarSlot = Inventory::hotbarSlotId(1);
    action.use = true;
    stepAgent(world, agent, action, 1, 0.0f);

    CHECK(world.blockType({ 0, 15, 1 }) == Block::Dirt);
    CHECK(itemCount(agent.inventory(), Item::Type::Dirt) == 63);
}

TEST_CASE("Agent use with an empty active hotbar slot does not place a block", "[environment][inventory]")
{
    World world;
    setupEmptyTestArea(world);
    world.setBlock({ 0, 10, 0 }, Block::Stone);
    world.setBlock({ 0, 15, 2 }, Block::Stone);

    Agent agent(world);
    agent.reset({ 0.5f, 14.0f, 0.5f });

    const auto trySetBlock = [&](Inventory::SlotId slot) {
        AgentAction action;
        action.activeHotbarSlot = slot;
        action.use = true;
        stepAgent(world, agent, action, 1, 0.0f);
    };

    trySetBlock(Inventory::hotbarSlotId(8));
    CHECK(world.blockType({ 0, 15, 1 }) == Block::Air);
    CHECK(itemCount(agent.inventory(), Item::Type::Torch) == 64);
    CHECK(itemCount(agent.inventory(), Item::Type::Dirt) == 64);
    CHECK(itemCount(agent.inventory(), Item::Type::Stone) == 64);

    const Inventory::SlotId stoneSlot = Inventory::hotbarSlotId(2);
    REQUIRE(agent.inventory()[stoneSlot].count() == 64);

    trySetBlock(stoneSlot);
    CHECK(world.blockType({ 0, 15, 1 }) == Block::Stone);
    CHECK(agent.inventory()[stoneSlot].count() == 63);
    CHECK(itemCount(agent.inventory(), Item::Type::Stone) == 63);
}

TEST_CASE("World collision queries respect air and solid override masks", "[world]")
{
    World world;
    world.resetSeed(17);
    const std::int32_t x = 5;
    const std::int32_t z = -3;
    updateWorldCacheAt(world, { x, 0, z });
    const std::int32_t groundY = world.terrainHeight({ x, z });
    const IVec3 groundBlock { x, groundY, z };
    REQUIRE(world.blockType({ x, groundY, z }) != Block::Air);
    CHECK(world.hasSolidBlockInArea(groundBlock, groundBlock));

    world.setBlock({ x, groundY, z }, Block::Air);
    CHECK(!world.hasSolidBlockInArea(groundBlock, groundBlock));
    updateWorldCacheAt(world, { x, 0, z });
    CHECK(world.blockType(groundBlock) == Block::Air);
    CHECK(!world.hasSolidBlockInArea(groundBlock, groundBlock));

    const IVec3 airBlock { x, World::s_maxY, z };
    REQUIRE(world.blockType(airBlock) == Block::Air);
    CHECK(!world.hasSolidBlockInArea(airBlock, airBlock));

    world.setBlock(airBlock, Block::Stone);
    CHECK(world.hasSolidBlockInArea(airBlock, airBlock));
    updateWorldCacheAt(world, { x, 0, z });
    CHECK(world.blockType(airBlock) == Block::Stone);
    CHECK(world.hasSolidBlockInArea(airBlock, airBlock));
}

TEST_CASE("World can remove a solid block at the top height", "[world]")
{
    World world;
    world.resetSeed(17);
    const IVec3 topBlock { 5, World::s_maxY, -3 };
    updateWorldCacheAt(world, { topBlock.x, 0, topBlock.z });

    world.setBlock(topBlock, Block::Stone);
    REQUIRE(world.blockType(topBlock) == Block::Stone);

    world.setBlock(topBlock, Block::Air);
    CHECK(world.blockType(topBlock) == Block::Air);
}

TEST_CASE("World block cache follows the requested agent-centered region", "[world]")
{
    World world;
    world.resetSeed(17);
    const IVec3 firstCenter { 0, 0, 0 };
    const IVec3 secondCenter { 48, 0, -48 };
    updateWorldCacheAt(world, firstCenter);
    CHECK(world.blockType({ 0, 0, 0 }) != Block::Air);

    updateWorldCacheAt(world, secondCenter);
    CHECK(world.blockType({ 48, 0, -48 }) != Block::Air);
}

TEST_CASE("OverrideCluster keeps count consistent with stored blocks", "[world]")
{
    OverrideCluster cluster;
    const OverrideCluster::Mask bit3 = OverrideCluster::Mask { 1 } << 3;
    const OverrideCluster::Mask bit7 = OverrideCluster::Mask { 1 } << 7;

    CHECK(cluster.isEmpty());
    CHECK(cluster.count() == 0);
    CHECK(!cluster.get(3));
    CHECK(!cluster.hasOverride(3));
    CHECK(!cluster.hasSolidOverride(3));
    CHECK(!cluster.hasOverrideInMask(bit3));
    CHECK(!cluster.hasSolidOverrideInMask(bit3));

    CHECK(cluster.set(3, Block::Dirt));
    CHECK(!cluster.isEmpty());
    CHECK(cluster.count() == 1);
    REQUIRE(cluster.get(3));
    CHECK(*cluster.get(3) == Block::Dirt);
    CHECK(cluster.hasOverride(3));
    CHECK(cluster.hasSolidOverride(3));
    CHECK(cluster.hasOverrideInMask(bit3));
    CHECK(cluster.hasSolidOverrideInMask(bit3));

    CHECK(!cluster.set(3, Block::Stone));
    CHECK(cluster.count() == 1);
    REQUIRE(cluster.get(3));
    CHECK(*cluster.get(3) == Block::Stone);

    CHECK(cluster.set(7, Block::Grass));
    CHECK(cluster.count() == 2);
    CHECK(cluster.hasOverrideInMask(bit3 | bit7));
    CHECK(cluster.hasSolidOverrideInMask(bit3 | bit7));

    CHECK(cluster.clear(3));
    CHECK(cluster.count() == 1);
    CHECK(!cluster.get(3));
    CHECK(!cluster.hasOverride(3));
    CHECK(!cluster.hasSolidOverride(3));
    REQUIRE(cluster.get(7));
    CHECK(*cluster.get(7) == Block::Grass);

    CHECK(!cluster.clear(3));
    CHECK(cluster.count() == 1);
    CHECK(cluster.clear(7));
    CHECK(cluster.count() == 0);
    CHECK(cluster.isEmpty());
}

TEST_CASE("OverrideCluster tracks air overrides separately from solid overrides", "[world]")
{
    OverrideCluster cluster;
    const OverrideCluster::Mask bit5 = OverrideCluster::Mask { 1 } << 5;

    CHECK(cluster.set(5, Block::Air));
    CHECK(cluster.count() == 1);
    REQUIRE(cluster.get(5));
    CHECK(*cluster.get(5) == Block::Air);
    CHECK(cluster.hasOverride(5));
    CHECK(!cluster.hasSolidOverride(5));
    CHECK(cluster.hasOverrideInMask(bit5));
    CHECK(!cluster.hasSolidOverrideInMask(bit5));
    CHECK(cluster.overrideMask() == bit5);
    CHECK(cluster.solidMask() == 0);

    CHECK(!cluster.set(5, Block::Stone));
    CHECK(cluster.count() == 1);
    CHECK(cluster.hasOverride(5));
    CHECK(cluster.hasSolidOverride(5));
    CHECK(cluster.overrideMask() == bit5);
    CHECK(cluster.solidMask() == bit5);

    CHECK(!cluster.set(5, Block::Air));
    CHECK(cluster.count() == 1);
    CHECK(cluster.hasOverride(5));
    CHECK(!cluster.hasSolidOverride(5));
    CHECK(cluster.overrideMask() == bit5);
    CHECK(cluster.solidMask() == 0);

    CHECK(cluster.clear(5));
    CHECK(cluster.isEmpty());
    CHECK(cluster.overrideMask() == 0);
    CHECK(cluster.solidMask() == 0);
}

TEST_CASE("World collects only overrides inside a requested region", "[world]")
{
    World world;
    world.resetSeed(13);
    updateWorldCacheAt(world, { 1, 0, 1 });
    world.setBlock({ 1, 30, 1 }, Block::Stone);
    world.setBlock({ 9, 30, 1 }, Block::Stone);
    world.setBlock({ 1, 20, 1 }, Block::Stone);
    updateWorldCacheAt(world, { 40, 0, 40 });
    world.setBlock({ 40, 30, 40 }, Block::Stone);

    std::vector<BlockOverride> overrides;
    world.collectOverridesInRegion({ 0, 24, 0 }, { 16, 8, 16 }, overrides);

    std::vector<IVec3> coords;
    for (const BlockOverride& blockOverride : overrides)
        coords.push_back(blockOverride.coord);
    std::sort(coords.begin(), coords.end(), [](const IVec3& a, const IVec3& b) {
        if (a.x != b.x)
            return a.x < b.x;
        if (a.y != b.y)
            return a.y < b.y;
        return a.z < b.z;
    });

    REQUIRE(coords.size() == 2);
    CHECK(coords[0] == IVec3 { 1, 30, 1 });
    CHECK(coords[1] == IVec3 { 9, 30, 1 });
}

TEST_CASE("World spawns test pigs around the agent on reset", "[world][characters]")
{
    World world;
    world.resetSeed(21);
    updateWorldCacheAt(world, { 0, 0, 0 });
    world.resetCharacters();

    REQUIRE(world.characters().size() == 32);
    REQUIRE(world.characters().front() != nullptr);
    const Character& pig = *world.characters().front();
    CHECK(pig.kind() == CharacterKind::Pig);
    CHECK(world.characters().front()->stateKind() == CharacterStateKind::Idle);
    CHECK(pig.health() == 3);

    for (const std::unique_ptr<NPC>& character : world.characters()) {
        REQUIRE(character != nullptr);
        const Vec3 position = character->position();
        const float dx = position.x - 0.5f;
        const float dz = position.z - 0.5f;
        CHECK(std::sqrt(dx * dx + dz * dz) >= 3.0f);
        CHECK(position.y > 0.0f);
    }
}

TEST_CASE("Pig starts walking after world character updates", "[world][characters]")
{
    World world;
    world.resetSeed(21);
    updateWorldCacheAt(world, { 0, 0, 0 });
    world.resetCharacters();
    REQUIRE(world.characters().size() == 32);
    std::vector<Vec3> initialPositions;
    initialPositions.reserve(world.characters().size());
    for (const std::unique_ptr<NPC>& character : world.characters())
        initialPositions.push_back(character->position());

    for (int i = 0; i < 360; ++i)
        world.update(1.0f / 60.0f, { 1000.0f, 0.0f, 1000.0f });

    bool anyPigMoved = false;
    for (std::size_t i = 0; i < world.characters().size(); ++i) {
        const Vec3 movedPosition = world.characters()[i]->position();
        const float dx = movedPosition.x - initialPositions[i].x;
        const float dz = movedPosition.z - initialPositions[i].z;
        anyPigMoved = anyPigMoved || std::sqrt(dx * dx + dz * dz) > 0.1f;
    }
    CHECK(anyPigMoved);
}

TEST_CASE("Pig panics when threat is close", "[world][characters]")
{
    World world;
    world.resetSeed(21);
    updateWorldCacheAt(world, { 0, 0, 0 });
    world.resetCharacters();
    REQUIRE(world.characters().size() == 32);

    world.update(1.0f / 60.0f, world.characters().front()->position());
    CHECK(world.characters().front()->stateKind() == CharacterStateKind::Panic);
}

TEST_CASE("Physics: agent reaches ground target speed on flat stone", "[physics][characters]")
{
    World world;
    world.resetSeed(17);
    updateWorldCacheAt(world, { 0, 14, 0 });
    // Flat stone floor at y = 8, clear everything above it.
    fillBlocks(world, { -8, 8, -8 }, { 20, 1, 20 }, Block::Stone);
    clearBlocks(world, { -8, 9, -8 }, { 20, 15, 20 });

    Agent agent(world);
    agent.reset({ 0.5f, 10.0f, 0.5f });

    AgentAction action;
    action.forward = 1.0f;

    float lastSpeed = 0.0f;
    for (int i = 0; i < 60; ++i) {
        agent.step(action, 1.0f / 60.0f);
        world.update(1.0f / 60.0f, agent.state().position);
        const Vec3 v = agent.state().velocity;
        lastSpeed = std::sqrt(v.x * v.x + v.z * v.z);
    }
    // Walk target is 5.0 m/s; before the fix the 60Hz responsiveness bug capped it at ~2.33.
    CHECK(std::abs(lastSpeed - 5.0f) < 0.2f);
}

TEST_CASE("Physics: grounded feet resolve to the surface block, not the air cell above it", "[physics][characters]")
{
    World world;
    world.resetSeed(17);
    updateWorldCacheAt(world, { 0, 14, 0 });
    // Flat stone floor at y = 8 (top face at y = 9), clear everything above it.
    fillBlocks(world, { -8, 8, -8 }, { 20, 1, 20 }, Block::Stone);
    clearBlocks(world, { -8, 9, -8 }, { 20, 15, 20 });

    Agent agent(world);
    // Drop from well above the floor: a fast landing leaves the character hovering a few
    // millimetres (or more) above the surface unless the landing is snapped flush, and then a
    // surface query at the feet resolves to the air cell above the block instead of the block.
    // Regression for "grass feels like ice": a grounded character was silently decelerating
    // with air drag (0.91/tick) instead of the surface friction (0.546/tick on stone/grass).
    agent.reset({ 0.5f, 16.0f, 0.5f });

    AgentAction fwd;
    fwd.forward = 1.0f;
    AgentAction idle;

    // Fall, land, and reach walking speed.
    for (int i = 0; i < 90; ++i) {
        agent.step(fwd, 1.0f / 60.0f);
        world.update(1.0f / 60.0f, agent.state().position);
    }

    // Release input and measure the per-frame horizontal decay rate.
    const auto speed = [](const AgentState& s) {
        return std::sqrt(s.velocity.x * s.velocity.x + s.velocity.z * s.velocity.z);
    };
    float prev = speed(agent.state());
    float ratioSum = 0.0f;
    int samples = 0;
    for (int i = 0; i < 20 && prev > 0.1f; ++i) {
        agent.step(idle, 1.0f / 60.0f);
        world.update(1.0f / 60.0f, agent.state().position);
        const float cur = speed(agent.state());
        ratioSum += cur / prev;
        prev = cur;
        ++samples;
    }
    REQUIRE(samples > 0);

    // Ground friction: 0.6 slipperiness * 0.91 = 0.546 per tick -> 0.8173 per 1/60s frame.
    // Air drag: 0.91 per tick -> 0.9691 per frame. The measured rate must match the ground.
    const float avgRatio = ratioSum / static_cast<float>(samples);
    CHECK(std::abs(avgRatio - 0.8173f) < 0.02f);
    CHECK(agent.state().grounded);
}

namespace {
    // A bare character with no AI so knockback can be measured in isolation. The pig hit
    // cylinder matches PigCharacter's so the result matches what the player sees on real pigs.
    class KnockbackVictim final : public Character {
    public:
        KnockbackVictim(World& world, Vec3 position)
            : Character(world, 99, CharacterKind::Pig, position, { .radius = 0.35f, .height = 0.8f })
        {
            m_health = 3;
        }
        using Character::applyHorizontalFriction;
        using Character::applyPhysics;
    };
} // namespace

TEST_CASE("Physics: knockback matches Minecraft's base punch", "[physics][characters]")
{
    World world;
    world.resetSeed(17);
    updateWorldCacheAt(world, { 0, 14, 0 });
    // Flat stone floor at y = 8 (top face at y = 9), clear everything above it.
    fillBlocks(world, { -8, 8, -8 }, { 20, 1, 20 }, Block::Stone);
    clearBlocks(world, { -8, 9, -8 }, { 20, 15, 20 });

    KnockbackVictim victim(world, { 0.5f, 10.0f, 0.5f });
    // Land and settle on the floor, mirroring NPC::update (friction then physics).
    for (int i = 0; i < 60; ++i) {
        victim.applyHorizontalFriction(1.0f / 60.0f);
        victim.applyPhysics(1.0f / 60.0f);
    }
    const Vec3 start = victim.position();

    // Minecraft base punch: strength 0.4 blocks/tick = 8 m/s horizontally + a fixed ~1-block
    // hop. Direction is the attacker's horizontal facing (here straight +Z).
    victim.onHit({ 0.0f, 0.0f, 1.0f }, 8.0f, 1);

    float maxY = start.y;
    for (int i = 0; i < 150; ++i) {
        victim.applyHorizontalFriction(1.0f / 60.0f);
        victim.applyPhysics(1.0f / 60.0f);
        maxY = std::max(maxY, victim.position().y);
    }

    const Vec3 displacement = victim.position() - start;
    const float distance = std::sqrt(displacement.x * displacement.x + displacement.z * displacement.z);
    // Expected: ~3 m push + ~0.2 m ground slide. The old arbitrary power launched ~5.7 m, and
    // treating power as a pure full-vector impulse (with the fixed 0.2 up-tilt) slid only ~1.4 m.
    CHECK(distance > 2.0f);
    CHECK(distance < 4.5f);
    // A ~1.1 m hop, not a launch high into the air.
    CHECK(maxY - start.y < 2.0f);
    CHECK(maxY - start.y > 0.3f);
}

TEST_CASE("Physics: repeated hits within Minecraft's invulnerability window do not stack knockback", "[physics][characters]")
{
    World world;
    world.resetSeed(17);
    updateWorldCacheAt(world, { 0, 14, 0 });
    fillBlocks(world, { -8, 8, -8 }, { 20, 1, 20 }, Block::Stone);
    clearBlocks(world, { -8, 9, -8 }, { 20, 15, 20 });

    KnockbackVictim victim(world, { 0.5f, 10.0f, 0.5f });
    for (int i = 0; i < 60; ++i) {
        victim.applyHorizontalFriction(1.0f / 60.0f);
        victim.applyPhysics(1.0f / 60.0f);
    }
    const Vec3 start = victim.position();

    // Two quick hits (e.g. rapid clicks or a spamming agent). MC's invulnerableTime (1 s) makes
    // the second one a no-op; without it the +7 m/s vertical hops stack and the target rockets up.
    victim.onHit({ 0.0f, 0.0f, 1.0f }, 8.0f, 1);
    victim.applyPhysics(1.0f / 60.0f);
    victim.onHit({ 0.0f, 0.0f, 1.0f }, 8.0f, 1);

    float maxY = start.y;
    for (int i = 0; i < 150; ++i) {
        victim.applyHorizontalFriction(1.0f / 60.0f);
        victim.applyPhysics(1.0f / 60.0f);
        maxY = std::max(maxY, victim.position().y);
    }

    // Regression: two stacked hits previously launched the target ~4.3 m; with the window the
    // rise stays the single-hit ~1.1 m. The second hit must also deal no damage (health 3 -> 2).
    CHECK(maxY - start.y < 2.0f);
    CHECK(maxY - start.y > 0.3f);
    CHECK(victim.health() == 2);
}

TEST_CASE("Agent can only attack characters within melee reach", "[environment][characters]")
{
    const auto attackPigAt = [](float pigZ) {
        World world;
        world.resetSeed(17);
        updateWorldCacheAt(world, { 0, 14, 0 });
        // Flat stone floor at y = 8 (top face at y = 9), clear everything above it.
        fillBlocks(world, { -8, 8, -8 }, { 20, 1, 20 }, Block::Stone);
        clearBlocks(world, { -8, 9, -8 }, { 20, 15, 20 });

        // Agent and pig share the x = 0.5 axis, so yaw 0 aims along +Z. The pig never moves:
        // only agent.step runs, never world.update.
        Agent agent(world);
        agent.reset({ 0.5f, 9.0f, 0.5f });
        world.addCharacter(std::make_unique<PigCharacter>(world, 1, Vec3 { 0.5f, 9.0f, pigZ }));

        // Aim at the pig's centre: the eye is 1.62 above the feet, the hit cylinder is 0.8 tall.
        AgentAction attack;
        const float eyeY = 9.0f + 1.62f;
        const float pigCentreY = 9.0f + 0.4f;
        attack.pitchDelta = std::atan2(pigCentreY - eyeY, pigZ - 0.5f);
        attack.attack = true;
        agent.step(attack, TestStepDt);
        return world.characters().front()->health();
    };

    // 2.5 m away horizontally: inside the 3 m melee reach -> the hit lands (3 HP - 1 damage).
    CHECK(attackPigAt(3.0f) == 2);
    // 5.5 m away: the ray still crosses the pig, but the reach gate rejects the hit.
    CHECK(attackPigAt(6.0f) == 3);
}

TEST_CASE("Agent cannot mine a block occluded by an out-of-reach character", "[environment][characters]")
{
    const auto probe = [](bool withPig) {
        World world;
        world.resetSeed(17);
        updateWorldCacheAt(world, { 0, 14, 0 });
        // Flat stone floor at y = 8 (top face at y = 9), clear everything above it.
        fillBlocks(world, { -8, 8, -8 }, { 20, 1, 20 }, Block::Stone);
        clearBlocks(world, { -8, 9, -8 }, { 20, 15, 20 });
        // A target block on the floor ahead of the agent.
        world.setBlock({ 0, 9, 4 }, Block::Stone);

        Agent agent(world);
        agent.reset({ 0.5f, 9.0f, 0.5f });
        if (withPig) {
            // Pig 3.2 m from the eye (beyond the 3 m melee reach) standing between the eye and
            // the block, so its body is in the line of fire.
            world.addCharacter(std::make_unique<PigCharacter>(world, 1, Vec3 { 0.5f, 9.0f, 3.7f }));
        }

        // Aim at the block's front face (z = 4) at chest height.
        AgentAction attack;
        attack.pitchDelta = std::atan2((9.0f + 0.5f) - (9.0f + 1.62f), 4.0f - 0.5f);
        attack.attack = true;
        agent.step(attack, TestStepDt);

        const Block block = world.blockType({ 0, 9, 4 });
        const int pigHealth = withPig ? world.characters().front()->health() : 0;
        return std::pair<Block, int> { block, pigHealth };
    };

    // Without the pig the block is within the 4.5 m block reach and is mined.
    const std::pair<Block, int> noPig = probe(false);
    CHECK(noPig.first == Block::Air);

    // With the pig in front (but out of melee reach) its body occludes the block: the block is
    // not mined and the pig itself is not hit (it stays at full health).
    const std::pair<Block, int> pigInTheWay = probe(true);
    CHECK(pigInTheWay.first == Block::Stone);
    CHECK(pigInTheWay.second == 3);
}

TEST_CASE("Agent cannot mine a block beyond the block interaction reach", "[environment]")
{
    World world;
    world.resetSeed(17);
    updateWorldCacheAt(world, { 0, 14, 0 });
    fillBlocks(world, { -8, 8, -8 }, { 20, 1, 20 }, Block::Stone);
    clearBlocks(world, { -8, 9, -8 }, { 20, 15, 20 });
    // A block ~5.6 m from the eye, beyond the 4.5 m block interaction reach.
    world.setBlock({ 0, 9, 6 }, Block::Stone);

    Agent agent(world);
    agent.reset({ 0.5f, 9.0f, 0.5f });

    // Aim at the block's front face (z = 6) at chest height.
    AgentAction attack;
    attack.pitchDelta = std::atan2((9.0f + 0.5f) - (9.0f + 1.62f), 6.0f - 0.5f);
    attack.attack = true;
    agent.step(attack, TestStepDt);

    CHECK(world.blockType({ 0, 9, 6 }) == Block::Stone);
}

} // namespace blocklab::test
