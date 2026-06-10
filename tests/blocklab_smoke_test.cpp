#include "blocklab/Environment.h"

#include "blocklab/CudaHelpers.h"

#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace {

constexpr int32_t TestGenerationHalfExtent = 32;
constexpr uint32_t TestGenerationExtent = TestGenerationHalfExtent * 2;
constexpr uint32_t TestMaxTerrainVoxels
    = static_cast<uint32_t>(TestGenerationExtent * blocklab::Chunk::SizeY * TestGenerationExtent);

void updateWorldCacheAt(const blocklab::World& world, blocklab::IVec3 center)
{
    blocklab::WorldGenerator generator({ .halfExtent = TestGenerationHalfExtent });
    void* voxelMemory = nullptr;
    blocklab::cudaCheck(
        cudaMalloc(&voxelMemory, blocklab::VoxelSize * TestMaxTerrainVoxels), "cudaMalloc test terrain voxels");
    auto* const voxels = static_cast<blocklab::Voxel*>(voxelMemory);
    blocklab::TerrainHeader* terrainHeader = nullptr;
    blocklab::cudaCheck(cudaMalloc(&terrainHeader, sizeof(blocklab::TerrainHeader)), "cudaMalloc test terrain header");
    blocklab::AgentState agent;
    agent.position = { static_cast<float>(center.x), static_cast<float>(center.y), static_cast<float>(center.z) };
    blocklab::CudaSharedFuture<blocklab::WorldGenerationOutput> generation
        = generator
              .generate(world, agent,
                  {
                      .header = terrainHeader,
                      .voxels = voxels,
                      .maxVoxelCount = TestMaxTerrainVoxels,
                      .blocks = world.borrowGenerationBuffers(),
                  })
              .share();
    world.updateGeneration(generation);
    world.waitForGeneration();
    blocklab::cudaCheck(cudaFree(terrainHeader), "cudaFree test terrain header");
    blocklab::cudaCheck(cudaFree(voxels), "cudaFree test terrain voxels");
}

class CountingRenderer final : public blocklab::ObservationRenderer {
public:
    const blocklab::Observation& renderObservations(
        std::span<const blocklab::World> worlds, std::span<const blocklab::AgentState> agents) override
    {
        for (std::size_t i = 0; i < worlds.size(); ++i) {
            updateWorldCacheAt(worlds[i],
                { blocklab::floorToInt32(agents[i].position.x), blocklab::floorToInt32(agents[i].position.y),
                    blocklab::floorToInt32(agents[i].position.z) });
        }
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
    blocklab::Environment env(renderer, 1);
    env.reset(42);
    const blocklab::Observation& initial = env.observe();
    CHECK(initial.device() == blocklab::ObservationDevice::Cpu);
    CHECK(initial.handle(0) == 0x1234);
}

TEST_CASE("Environment can step a moving agent", "[environment]")
{
    CountingRenderer renderer;
    blocklab::Environment env(renderer, 1);
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

TEST_CASE("Environment reset rebuilds cache around the initial spawn", "[environment]")
{
    CountingRenderer renderer;
    blocklab::Environment env(renderer, 1);
    env.reset(42);
    blocklab::AgentAction action;
    action.forward = 1.0f;
    const blocklab::AgentAction actions[] { action };
    // Move the agent into another generated cache region, then verify reset rebuilds the cache around the initial
    // spawn instead of reusing the previous agent-centered region.
    for (int i = 0; i < 500; ++i)
        env.step(actions);

    env.reset(43);

    CHECK(env.agent(0).state().position.x == 0.5f);
    CHECK(env.agent(0).state().position.z == 0.5f);
    CHECK(env.world(0).characters().size() == 32);
}

TEST_CASE("Agent cannot place a block into its own body", "[environment]")
{
    CountingRenderer renderer;
    blocklab::Environment placeEnv(renderer, 1);
    placeEnv.reset(7);
    const blocklab::AgentState& state = placeEnv.agent(0).state();
    const int32_t occupiedX = blocklab::floorToInt32(state.position.x);
    const int32_t occupiedY = blocklab::floorToInt32(state.position.y + 1.0f);
    const int32_t occupiedZ = blocklab::floorToInt32(state.position.z);
    placeEnv.mutableWorld(0).setBlock({ occupiedX, occupiedY, occupiedZ + 2 }, blocklab::Block::Dirt);

    blocklab::AgentAction placeIntoSelf;
    placeIntoSelf.pitchDelta = 1.2f;
    placeIntoSelf.place = true;
    const blocklab::AgentAction placeActions[] { placeIntoSelf };
    placeEnv.step(placeActions);
    CHECK(placeEnv.world(0).getBlock({ occupiedX, occupiedY, occupiedZ }) == blocklab::Block::Air);
}

TEST_CASE("World collision queries respect air and solid override masks", "[world]")
{
    blocklab::World world;
    world.resetSeed(17);
    const int32_t x = 5;
    const int32_t z = -3;
    updateWorldCacheAt(world, { x, 0, z });
    const int32_t groundY
        = blocklab::floorToInt32(world.groundHeight(static_cast<float>(x), static_cast<float>(z))) - 1;
    const blocklab::IVec3 groundBlock { x, groundY, z };
    REQUIRE(world.getBlock({ x, groundY, z }) != blocklab::Block::Air);
    CHECK(world.hasSolidBlockInArea(groundBlock, groundBlock));

    world.setBlock({ x, groundY, z }, blocklab::Block::Air);
    CHECK(!world.hasSolidBlockInArea(groundBlock, groundBlock));
    updateWorldCacheAt(world, { x, 0, z });
    CHECK(world.getBlock(groundBlock) == blocklab::Block::Air);
    CHECK(!world.hasSolidBlockInArea(groundBlock, groundBlock));

    const blocklab::IVec3 airBlock { x, blocklab::Chunk::SizeY - 1, z };
    REQUIRE(world.getBlock(airBlock) == blocklab::Block::Air);
    CHECK(!world.hasSolidBlockInArea(airBlock, airBlock));

    world.setBlock(airBlock, blocklab::Block::Stone);
    CHECK(world.hasSolidBlockInArea(airBlock, airBlock));
    updateWorldCacheAt(world, { x, 0, z });
    CHECK(world.getBlock(airBlock) == blocklab::Block::Stone);
    CHECK(world.hasSolidBlockInArea(airBlock, airBlock));
}

TEST_CASE("World block cache follows the requested agent-centered region", "[world]")
{
    blocklab::World world;
    world.resetSeed(17);
    const blocklab::IVec3 firstCenter { 0, 0, 0 };
    const blocklab::IVec3 secondCenter { 48, 0, -48 };
    updateWorldCacheAt(world, firstCenter);
    CHECK(world.getBlock({ 0, 0, 0 }) != blocklab::Block::Air);

    updateWorldCacheAt(world, secondCenter);
    CHECK(world.getBlock({ 48, 0, -48 }) != blocklab::Block::Air);
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
    blocklab::World world;
    world.resetSeed(13);
    updateWorldCacheAt(world, { 1, 0, 1 });
    world.setBlock({ 1, 30, 1 }, blocklab::Block::Stone);
    world.setBlock({ 9, 30, 1 }, blocklab::Block::Stone);
    world.setBlock({ 1, 20, 1 }, blocklab::Block::Stone);
    updateWorldCacheAt(world, { 40, 0, 40 });
    world.setBlock({ 40, 30, 40 }, blocklab::Block::Stone);

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
    blocklab::World world;
    world.resetSeed(21);
    updateWorldCacheAt(world, { 0, 0, 0 });
    world.resetCharacters();

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
    blocklab::World world;
    world.resetSeed(21);
    updateWorldCacheAt(world, { 0, 0, 0 });
    world.resetCharacters();
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
    blocklab::World world;
    world.resetSeed(21);
    updateWorldCacheAt(world, { 0, 0, 0 });
    world.resetCharacters();
    REQUIRE(world.characters().size() == 32);

    world.update(1.0f / 60.0f, world.characters().front()->position());
    CHECK(world.characters().front()->stateKind() == blocklab::CharacterStateKind::Panic);
}

TEST_CASE("Environment returns renderer observations when renderer is installed", "[environment]")
{
    CountingRenderer renderer;
    blocklab::Environment renderEnv(renderer, 1);
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
