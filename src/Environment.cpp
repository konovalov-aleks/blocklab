#include "blocklab/Environment.h"

#include "blocklab/Error.h"
#include "blocklab/Hash.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace blocklab {

static constexpr Vec3 InitialAgentPosition { 0.5f, 14.0f, 0.5f };

Environment::Environment(ObservationRenderer& renderer, std::uint32_t numEnvs)
    : m_batchSize(numEnvs)
    , m_worlds(std::make_unique<World[]>(numEnvs))
    , m_agents(std::make_unique<Agent[]>(numEnvs))
    , m_observationRenderer(renderer)
    , m_stepCounts(std::make_unique<std::int32_t[]>(numEnvs))
    , m_stepResults(std::make_unique<StepResult[]>(numEnvs))
    , m_renderAgents(std::make_unique<AgentState[]>(numEnvs))
{
    if (numEnvs == 0) [[unlikely]]
        fatalError("Environment batch size must be positive");
}

void Environment::reset(std::uint32_t seed)
{
    for (std::uint32_t i = 0; i < m_batchSize; ++i) {
        World& world = m_worlds[i];
        world.resetSeed(hash(seed + i));
        m_agents[i].reset(InitialAgentPosition);
    }

    // update block cache before character's initialization
    updateObservation();

    for (std::uint32_t i = 0; i < m_batchSize; ++i) {
        World& world = m_worlds[i];
        m_stepCounts[i] = 0;
        world.resetCharacters();
        const float spawnY = world.groundHeight(0.0f, 0.0f) + 0.05f;
        m_agents[i].reset({ InitialAgentPosition.x, spawnY, InitialAgentPosition.z });
    }

    // we have moved the agent's initial position according to the ground height,
    // so we need to update the observation again
    updateObservation();
}

std::span<const StepResult> Environment::step(std::span<const AgentAction> actions)
{
    if (actions.size() != m_batchSize)
        fatalError("Action batch size does not match environment batch size");

    for (std::uint32_t i = 0; i < m_batchSize; ++i) {
        World& world = m_worlds[i];
        Agent& agent = m_agents[i];
        StepResult& result = m_stepResults[i];
        const AgentState before = agent.state();
        agent.step(world, actions[i], s_fixedDt);
        world.update(s_fixedDt, agent.state().position);
        ++m_stepCounts[i];

        const AgentState& after = agent.state();
        const float dx = after.position.x - before.position.x;
        const float dz = after.position.z - before.position.z;
        float reward = 0.01f + std::min(0.05f, std::sqrt(dx * dx + dz * dz) * 0.02f);
        reward += static_cast<float>(after.blocksCollected - before.blocksCollected) * 0.5f;
        reward += static_cast<float>(after.blocksPlaced - before.blocksPlaced) * 0.2f;
        if (after.position.y < -8.0f)
            reward -= 5.0f;
        result.reward = reward;
        result.terminated = after.position.y < -8.0f;
        result.truncated = m_stepCounts[i] >= s_maxSteps;
    }

    updateObservation();
    return { m_stepResults.get(), m_batchSize };
}

const Observation& Environment::updateObservation()
{
    for (std::uint32_t i = 0; i < m_batchSize; ++i)
        m_renderAgents[i] = m_agents[i].state();
    const Observation& observation = m_observationRenderer.renderObservations(
        { m_worlds.get(), m_batchSize }, { m_renderAgents.get(), m_batchSize });
    if (observation.batchSize() != m_batchSize) [[unlikely]]
        fatalError("Observation renderer returned an unexpected batch size");
    m_observation = &observation;
    return *m_observation;
}

} // namespace blocklab
