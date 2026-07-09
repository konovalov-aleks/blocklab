#include "WorldTestUtils.h"

#include <blocklab/environment/Environment.h>
#include <blocklab/gpu/vulkan/Vulkan.h>
#include <blocklab/graphics/Renderer.h>
#include <blocklab/inventory/Inventory.h>
#include <environment/Agent.h>
#include <world/World.h>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <vector>

namespace blocklab::test {

class EnvironmentInternalAccessTestHelper {
public:
    static World& world(Environment& env, std::size_t i) { return env.m_worlds[i]; }
    static const AgentState& agentState(Environment& env, std::size_t i) { return env.m_agents[i].state(); }
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
            agent.step(world, action, dt);
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

    Agent agent;
    agent.reset({ 0.5f, 0.0f, 0.5f });

    AgentAction action;
    action.pitchDelta = -Pi;
    action.attack = true;

    agent.step(world, action, 0.0f);
    CHECK(agent.state().blocksCollected == 0);
}

TEST_CASE("Agent picks up a mined block after falling through it", "[environment][inventory][drop]")
{
    World world;
    setupEmptyTestArea(world);
    world.setBlock({ 0, 10, 0 }, Block::Stone);
    world.setBlock({ 0, 13, 0 }, Block::Dirt);

    Agent agent;
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

    Agent agent;
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

    Agent agent;
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

    Agent agent;
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

} // namespace blocklab::test
