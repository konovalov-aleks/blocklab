#include "blocklab/Environment.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace {

class CountingRenderer final : public blocklab::ObservationRenderer {
public:
    const blocklab::Observation& renderObservations(
        std::span<const blocklab::World> worlds, std::span<const blocklab::AgentState> agents) override
    {
        (void)agents;
        m_observation.reset(4U, 4U, 4U, blocklab::ObservationDevice::Cpu, blocklab::ObservationFormat::FloatNCHW,
            static_cast<uint32_t>(worlds.size()));
        m_observation.setVersion(++m_version);
        for (uint32_t i = 0; i < worlds.size(); ++i)
            m_observation.setSlot(i, 0x1234);
        return m_observation;
    }

private:
    blocklab::Observation m_observation;
    uint64_t m_version = 0;
};

} // namespace

TEST_CASE("Environment reset returns renderer observation", "[environment]")
{
    CountingRenderer renderer;
    blocklab::Environment env(renderer, 1, 2);
    env.reset(42);
    const blocklab::Observation& initial = env.observe();
    CHECK(initial.device() == blocklab::ObservationDevice::Cpu);
    CHECK(initial.handle(0) == 0x1234);
}

TEST_CASE("Environment can step a moving agent", "[environment]")
{
    CountingRenderer renderer;
    blocklab::Environment env(renderer, 1, 2);
    env.reset(42);
    for (int i = 0; i < 120; ++i) {
        blocklab::AgentAction action;
        action.forward = 1.0f;
        action.yawDelta = 0.01f;
        action.pitchDelta = 0.001f;
        const blocklab::AgentAction actions[] { action };
        const blocklab::StepResult result = env.step(actions).front();
        CHECK(env.observe().device() == blocklab::ObservationDevice::Cpu);
        CHECK(result.reward > -10.0f);
    }
}

TEST_CASE("Agent cannot place a block into its own body", "[environment]")
{
    CountingRenderer renderer;
    blocklab::Environment placeEnv(renderer, 1, 2);
    placeEnv.reset(7);
    const blocklab::AgentState& state = placeEnv.agent(0).state();
    const int32_t occupiedX = blocklab::floorToInt32(state.position.x);
    const int32_t occupiedY = blocklab::floorToInt32(state.position.y + 1.0f);
    const int32_t occupiedZ = blocklab::floorToInt32(state.position.z);
    placeEnv.mutableWorld(0).setBlock(occupiedX, occupiedY, occupiedZ + 2, blocklab::Block::Dirt);

    blocklab::AgentAction placeIntoSelf;
    placeIntoSelf.pitchDelta = 1.2f;
    placeIntoSelf.place = true;
    const blocklab::AgentAction placeActions[] { placeIntoSelf };
    placeEnv.step(placeActions);
    CHECK(placeEnv.world(0).getBlock(occupiedX, occupiedY, occupiedZ) == blocklab::Block::Air);
}

TEST_CASE("World keeps procedural blocks and sparse overrides consistent", "[world]")
{
    blocklab::World infiniteWorld(11);
    CHECK(infiniteWorld.getBlock(100000, 0, -100000) == blocklab::Block::Stone);

    const blocklab::Block generated = infiniteWorld.getBlock(100000, 12, -100000);
    infiniteWorld.setBlock(100000, 12, -100000, blocklab::Block::Stone);
    CHECK(infiniteWorld.getBlock(100000, 12, -100000) == blocklab::Block::Stone);
    CHECK(infiniteWorld.overrideCount() > 0);

    infiniteWorld.setBlock(100000, 12, -100000, generated);
    CHECK(infiniteWorld.getBlock(100000, 12, -100000) == generated);
    CHECK(infiniteWorld.overrideCount() == 0);
}

TEST_CASE("World seed materially changes procedural terrain", "[world]")
{
    blocklab::World first(1);
    blocklab::World second(2);
    first.reset(1);
    second.reset(2);

    int differentHeights = 0;
    for (int32_t z = -8; z <= 8; ++z) {
        for (int32_t x = -8; x <= 8; ++x) {
            if (std::abs(first.groundHeight(static_cast<float>(x), static_cast<float>(z))
                    - second.groundHeight(static_cast<float>(x), static_cast<float>(z)))
                > 0.5f) {
                ++differentHeights;
            }
        }
    }
    CHECK(differentHeights > 96);
}

TEST_CASE("World collision queries respect air and solid override masks", "[world]")
{
    blocklab::World world(17);
    const int32_t x = 5;
    const int32_t z = -3;
    const int32_t groundY
        = blocklab::floorToInt32(world.groundHeight(static_cast<float>(x), static_cast<float>(z))) - 1;
    const blocklab::IVec3 groundBlock { x, groundY, z };
    REQUIRE(world.getBlock(x, groundY, z) != blocklab::Block::Air);
    CHECK(world.hasSolidBlockInArea(groundBlock, groundBlock));

    world.setBlock(x, groundY, z, blocklab::Block::Air);
    CHECK(!world.hasSolidBlockInArea(groundBlock, groundBlock));

    const blocklab::IVec3 airBlock { x, blocklab::Chunk::SizeY - 1, z };
    REQUIRE(world.getBlock(airBlock.x, airBlock.y, airBlock.z) == blocklab::Block::Air);
    CHECK(!world.hasSolidBlockInArea(airBlock, airBlock));

    world.setBlock(airBlock.x, airBlock.y, airBlock.z, blocklab::Block::Stone);
    CHECK(world.hasSolidBlockInArea(airBlock, airBlock));
}

TEST_CASE("World block cache uses local dense coordinates", "[world]")
{
    blocklab::World world(17);
    blocklab::World::BlocksCache& cache = world.collisionCacheMutable();
    cache.origin = { -4, 0, 8 };
    cache.size = { 3, 4, 2 };
    cache.version = world.version();
    cache.blocks.resize(static_cast<std::size_t>(cache.size.x * cache.size.y * cache.size.z));
    std::fill(cache.blocks.begin(), cache.blocks.end(), blocklab::BlockId::Air);

    const blocklab::IVec3 stone { -3, 2, 9 };
    const blocklab::IVec3 local = stone - cache.origin;
    const std::size_t index = static_cast<std::size_t>(local.x)
        + static_cast<std::size_t>(local.y) * static_cast<std::size_t>(cache.size.x)
        + static_cast<std::size_t>(local.z) * static_cast<std::size_t>(cache.size.x)
            * static_cast<std::size_t>(cache.size.y);
    cache.blocks[index] = blocklab::BlockId::Stone;

    CHECK(world.getBlock(stone.x, stone.y, stone.z) == blocklab::Block::Stone);
    CHECK(world.hasSolidBlockInArea(stone, stone));
    CHECK(!world.hasSolidBlockInArea({ -4, 0, 8 }, { -4, 0, 8 }));
}

TEST_CASE("OverrideCluster keeps count consistent with stored blocks", "[world]")
{
    blocklab::OverrideCluster cluster;
    const blocklab::OverrideCluster::Mask bit3 = blocklab::OverrideCluster::Mask { 1 } << 3;
    const blocklab::OverrideCluster::Mask bit7 = blocklab::OverrideCluster::Mask { 1 } << 7;

    CHECK(cluster.isEmpty());
    CHECK(cluster.count() == 0);
    CHECK(!cluster.get(3));
    CHECK(!cluster.hasOverride(3));
    CHECK(!cluster.hasSolidOverride(3));
    CHECK(!cluster.hasOverrideInMask(bit3));
    CHECK(!cluster.hasSolidOverrideInMask(bit3));

    CHECK(cluster.set(3, blocklab::Block::Dirt));
    CHECK(!cluster.isEmpty());
    CHECK(cluster.count() == 1);
    REQUIRE(cluster.get(3));
    CHECK(*cluster.get(3) == blocklab::Block::Dirt);
    CHECK(cluster.hasOverride(3));
    CHECK(cluster.hasSolidOverride(3));
    CHECK(cluster.hasOverrideInMask(bit3));
    CHECK(cluster.hasSolidOverrideInMask(bit3));

    CHECK(!cluster.set(3, blocklab::Block::Stone));
    CHECK(cluster.count() == 1);
    REQUIRE(cluster.get(3));
    CHECK(*cluster.get(3) == blocklab::Block::Stone);

    CHECK(cluster.set(7, blocklab::Block::Grass));
    CHECK(cluster.count() == 2);
    CHECK(cluster.hasOverrideInMask(bit3 | bit7));
    CHECK(cluster.hasSolidOverrideInMask(bit3 | bit7));

    CHECK(cluster.clear(3));
    CHECK(cluster.count() == 1);
    CHECK(!cluster.get(3));
    CHECK(!cluster.hasOverride(3));
    CHECK(!cluster.hasSolidOverride(3));
    REQUIRE(cluster.get(7));
    CHECK(*cluster.get(7) == blocklab::Block::Grass);

    CHECK(!cluster.clear(3));
    CHECK(cluster.count() == 1);
    CHECK(cluster.clear(7));
    CHECK(cluster.count() == 0);
    CHECK(cluster.isEmpty());
}

TEST_CASE("OverrideCluster tracks air overrides separately from solid overrides", "[world]")
{
    blocklab::OverrideCluster cluster;
    const blocklab::OverrideCluster::Mask bit5 = blocklab::OverrideCluster::Mask { 1 } << 5;

    CHECK(cluster.set(5, blocklab::Block::Air));
    CHECK(cluster.count() == 1);
    REQUIRE(cluster.get(5));
    CHECK(*cluster.get(5) == blocklab::Block::Air);
    CHECK(cluster.hasOverride(5));
    CHECK(!cluster.hasSolidOverride(5));
    CHECK(cluster.hasOverrideInMask(bit5));
    CHECK(!cluster.hasSolidOverrideInMask(bit5));
    CHECK(cluster.overrideMask() == bit5);
    CHECK(cluster.solidMask() == 0);

    CHECK(!cluster.set(5, blocklab::Block::Stone));
    CHECK(cluster.count() == 1);
    CHECK(cluster.hasOverride(5));
    CHECK(cluster.hasSolidOverride(5));
    CHECK(cluster.overrideMask() == bit5);
    CHECK(cluster.solidMask() == bit5);

    CHECK(!cluster.set(5, blocklab::Block::Air));
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
    blocklab::World world(13);
    world.setBlock(1, 30, 1, blocklab::Block::Stone);
    world.setBlock(9, 30, 1, blocklab::Block::Stone);
    world.setBlock(1, 20, 1, blocklab::Block::Stone);
    world.setBlock(40, 30, 40, blocklab::Block::Stone);

    std::vector<blocklab::BlockOverride> overrides;
    world.collectOverridesInRegion({ 0, 24, 0 }, { 16, 8, 16 }, overrides);

    std::vector<blocklab::IVec3> coords;
    for (const blocklab::BlockOverride& blockOverride : overrides)
        coords.push_back(blockOverride.coord);
    std::sort(coords.begin(), coords.end(), [](const blocklab::IVec3& a, const blocklab::IVec3& b) {
        if (a.x != b.x)
            return a.x < b.x;
        if (a.y != b.y)
            return a.y < b.y;
        return a.z < b.z;
    });

    REQUIRE(coords.size() == 2);
    CHECK(coords[0] == blocklab::IVec3 { 1, 30, 1 });
    CHECK(coords[1] == blocklab::IVec3 { 9, 30, 1 });
}

TEST_CASE("World spawns test pigs around the agent on reset", "[world][characters]")
{
    blocklab::World world(13);
    world.reset(21);

    REQUIRE(world.characters().size() == 32);
    const blocklab::CharacterSnapshot pig = world.characters().front()->snapshot();
    CHECK(pig.kind == blocklab::CharacterKind::Pig);
    CHECK(world.characters().front()->stateKind() == blocklab::CharacterStateKind::Idle);
    CHECK(pig.health == 3);

    for (const std::unique_ptr<blocklab::NPC>& character : world.characters()) {
        const blocklab::Vec3 position = character->position();
        const float dx = position.x - 0.5f;
        const float dz = position.z - 0.5f;
        CHECK(std::sqrt(dx * dx + dz * dz) >= 3.0f);
        CHECK(position.y > 0.0f);
    }
}

TEST_CASE("Pig starts walking after world character updates", "[world][characters]")
{
    blocklab::World world(13);
    world.reset(21);
    REQUIRE(world.characters().size() == 32);
    std::vector<blocklab::Vec3> initialPositions;
    initialPositions.reserve(world.characters().size());
    for (const std::unique_ptr<blocklab::NPC>& character : world.characters())
        initialPositions.push_back(character->position());

    for (int i = 0; i < 360; ++i)
        world.update(1.0f / 60.0f, { 1000.0f, 0.0f, 1000.0f });

    bool anyPigMoved = false;
    for (std::size_t i = 0; i < world.characters().size(); ++i) {
        const blocklab::Vec3 movedPosition = world.characters()[i]->position();
        const float dx = movedPosition.x - initialPositions[i].x;
        const float dz = movedPosition.z - initialPositions[i].z;
        anyPigMoved = anyPigMoved || std::sqrt(dx * dx + dz * dz) > 0.1f;
    }
    CHECK(anyPigMoved);
}

TEST_CASE("Pig panics when threat is close", "[world][characters]")
{
    blocklab::World world(13);
    world.reset(21);
    REQUIRE(world.characters().size() == 32);

    world.update(1.0f / 60.0f, world.characters().front()->position());
    CHECK(world.characters().front()->stateKind() == blocklab::CharacterStateKind::Panic);
}

TEST_CASE("Environment returns renderer observations when renderer is installed", "[environment]")
{
    CountingRenderer renderer;
    blocklab::Environment renderEnv(renderer, 1, 2);
    renderEnv.reset(9);
    const uint64_t renderedResetVersion = renderEnv.observe().version();
    CHECK(renderEnv.observe().device() == blocklab::ObservationDevice::Cpu);
    CHECK(renderEnv.observe().handle(0) == 0x1234);
    CHECK(renderedResetVersion != 0);

    const blocklab::AgentAction action;
    const blocklab::AgentAction actions[] { action };
    const blocklab::StepResult renderedStep = renderEnv.step(actions).front();
    CHECK(renderEnv.observe().device() == blocklab::ObservationDevice::Cpu);
    CHECK(renderEnv.observe().handle(0) == 0x1234);
    CHECK(renderEnv.observe().version() > renderedResetVersion);
    CHECK(!renderedStep.terminated);
}
