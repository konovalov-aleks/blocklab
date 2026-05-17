#include "blocklab/Environment.h"

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

int main()
{
    blocklab::Environment env(2);
    const blocklab::Observation initial = env.reset(42);
    if (initial.device != blocklab::ObservationDevice::None || initial.handle != 0)
        return EXIT_FAILURE;

    for (int i = 0; i < 120; ++i) {
        blocklab::AgentAction action;
        action.forward = 1.0f;
        action.yawDelta = 0.01f;
        action.pitchDelta = 0.001f;
        const blocklab::StepResult result = env.step(action);
        if (result.observation.device != blocklab::ObservationDevice::None || result.reward <= -10.0f)
            return EXIT_FAILURE;
    }

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
    if (placeEnv.world().getBlock(occupiedX, occupiedY, occupiedZ) != blocklab::Block::Air)
        return EXIT_FAILURE;

    blocklab::World infiniteWorld(11);
    if (infiniteWorld.getBlock(100000, 0, -100000) != blocklab::Block::Stone)
        return EXIT_FAILURE;

    const blocklab::Block generated = infiniteWorld.getBlock(100000, 12, -100000);
    infiniteWorld.setBlock(100000, 12, -100000, blocklab::Block::Stone);
    if (infiniteWorld.getBlock(100000, 12, -100000) != blocklab::Block::Stone || infiniteWorld.overrideCount() == 0)
        return EXIT_FAILURE;

    infiniteWorld.setBlock(100000, 12, -100000, generated);
    if (infiniteWorld.getBlock(100000, 12, -100000) != generated || infiniteWorld.overrideCount() != 0)
        return EXIT_FAILURE;

    blocklab::Environment renderEnv(2);
    CountingRenderer renderer;
    renderEnv.setObservationRenderer(&renderer);
    const blocklab::Observation renderedReset = renderEnv.reset(9);
    if (renderedReset.device != blocklab::ObservationDevice::Cpu || renderedReset.handle != 0x1234
        || renderedReset.version == 0)
        return EXIT_FAILURE;

    const blocklab::StepResult renderedStep = renderEnv.step({ });
    if (renderedStep.observation.device != blocklab::ObservationDevice::Cpu || renderedStep.observation.handle != 0x1234
        || renderedStep.observation.version <= renderedReset.version)
        return EXIT_FAILURE;

    return EXIT_SUCCESS;
}
