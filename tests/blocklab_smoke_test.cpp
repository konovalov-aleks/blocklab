#include "blocklab/Environment.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <vector>

namespace {

class CountingRenderer final : public blocklab::ObservationRenderer {
public:
    blocklab::Observation renderObservation(const blocklab::World&, const blocklab::AgentState&) override
    {
        return {
            .width = 4,
            .height = 4,
            .channels = 4,
            .device = blocklab::ObservationDevice::Cpu,
            .format = blocklab::ObservationFormat::RGBA8,
            .handle = 0x1234,
            .version = ++m_version,
        };
    }

private:
    uint64_t m_version = 0;
};

} // namespace

TEST_CASE("Environment reset returns an empty observation without renderer", "[environment]")
{
    blocklab::Environment env(2);
    const blocklab::Observation initial = env.reset(42);
    CHECK(initial.device == blocklab::ObservationDevice::None);
    CHECK(initial.handle == 0);
}

TEST_CASE("Environment can step a moving agent", "[environment]")
{
    blocklab::Environment env(2);
    env.reset(42);
    for (int i = 0; i < 120; ++i) {
        blocklab::AgentAction action;
        action.forward = 1.0f;
        action.yawDelta = 0.01f;
        action.pitchDelta = 0.001f;
        const blocklab::StepResult result = env.step(action);
        CHECK(result.observation.device == blocklab::ObservationDevice::None);
        CHECK(result.reward > -10.0f);
    }
}

TEST_CASE("Agent cannot place a block into its own body", "[environment]")
{
    blocklab::Environment placeEnv(2);
    placeEnv.reset(7);
    const blocklab::AgentState& state = placeEnv.agent().state();
    const int32_t occupiedX = blocklab::floorToInt(state.position.x);
    const int32_t occupiedY = blocklab::floorToInt(state.position.y + 1.0f);
    const int32_t occupiedZ = blocklab::floorToInt(state.position.z);
    placeEnv.mutableWorld().setBlock(occupiedX, occupiedY, occupiedZ + 2, blocklab::Block::Dirt);

    blocklab::AgentAction placeIntoSelf;
    placeIntoSelf.pitchDelta = 1.2f;
    placeIntoSelf.place = true;
    placeEnv.step(placeIntoSelf);
    CHECK(placeEnv.world().getBlock(occupiedX, occupiedY, occupiedZ) == blocklab::Block::Air);
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

TEST_CASE("OverrideCluster keeps count consistent with stored blocks", "[world]")
{
    blocklab::OverrideCluster cluster;
    CHECK(cluster.isEmpty());
    CHECK(cluster.count() == 0);
    CHECK(!cluster.get(3));

    CHECK(cluster.set(3, blocklab::Block::Dirt));
    CHECK(!cluster.isEmpty());
    CHECK(cluster.count() == 1);
    REQUIRE(cluster.get(3));
    CHECK(*cluster.get(3) == blocklab::Block::Dirt);

    CHECK(!cluster.set(3, blocklab::Block::Stone));
    CHECK(cluster.count() == 1);
    REQUIRE(cluster.get(3));
    CHECK(*cluster.get(3) == blocklab::Block::Stone);

    CHECK(cluster.set(7, blocklab::Block::Grass));
    CHECK(cluster.count() == 2);

    CHECK(cluster.clear(3));
    CHECK(cluster.count() == 1);
    CHECK(!cluster.get(3));
    REQUIRE(cluster.get(7));
    CHECK(*cluster.get(7) == blocklab::Block::Grass);

    CHECK(!cluster.clear(3));
    CHECK(cluster.count() == 1);
    CHECK(cluster.clear(7));
    CHECK(cluster.count() == 0);
    CHECK(cluster.isEmpty());
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

    std::vector<blocklab::BlockCoord> coords;
    for (const blocklab::BlockOverride& blockOverride : overrides)
        coords.push_back(blockOverride.coord);
    std::sort(coords.begin(), coords.end(), [](const blocklab::BlockCoord& a, const blocklab::BlockCoord& b) {
        if (a.x != b.x)
            return a.x < b.x;
        if (a.y != b.y)
            return a.y < b.y;
        return a.z < b.z;
    });

    REQUIRE(coords.size() == 2);
    CHECK(coords[0] == blocklab::BlockCoord { .x = 1, .y = 30, .z = 1 });
    CHECK(coords[1] == blocklab::BlockCoord { .x = 9, .y = 30, .z = 1 });
}

TEST_CASE("Environment returns renderer observations when renderer is installed", "[environment]")
{
    blocklab::Environment renderEnv(2);
    CountingRenderer renderer;
    renderEnv.setObservationRenderer(&renderer);
    const blocklab::Observation renderedReset = renderEnv.reset(9);
    CHECK(renderedReset.device == blocklab::ObservationDevice::Cpu);
    CHECK(renderedReset.handle == 0x1234);
    CHECK(renderedReset.version != 0);

    const blocklab::StepResult renderedStep = renderEnv.step({ });
    CHECK(renderedStep.observation.device == blocklab::ObservationDevice::Cpu);
    CHECK(renderedStep.observation.handle == 0x1234);
    CHECK(renderedStep.observation.version > renderedReset.version);
}
