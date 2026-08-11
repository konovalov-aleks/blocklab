#include <blocklab/environment/Environment.h>

#include "EnvInstance.h"

#include <blocklab/graphics/Renderer.h>
#include <blocklab/utility/Error.h>
#include <utility/Hash.h>
#include <world/World.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace blocklab {

static constexpr Vec3 InitialAgentPosition { 0.5f, 14.0f, 0.5f };

Environment::Environment(Renderer& renderer, std::uint32_t numEnvs, std::uint32_t maxSteps)
    : m_instances(std::make_unique<EnvInstance[]>(numEnvs))
    , m_observation(numEnvs)
    , m_renderer(renderer)
    , m_stepCounts(std::make_unique<std::uint32_t[]>(numEnvs))
    , m_stepResults(std::make_unique<StepResult[]>(numEnvs))
    , m_renderAgents(std::make_unique<AgentState[]>(numEnvs))
    , m_batchSize(numEnvs)
    , m_maxSteps(maxSteps)
{
    if (numEnvs == 0) [[unlikely]]
        fatalError("Environment batch size must be positive");

    for (std::uint32_t i = 0; i < numEnvs; ++i)
        m_observation.inventories().set(i, m_instances[i].agent.inventory());
}

Environment::~Environment() = default;

void Environment::reset(std::uint32_t seed)
{
    for (std::uint32_t i = 0; i < m_batchSize; ++i) {
        World& world = m_instances[i].world;
        world.resetSeed(hash(seed + i));
        m_instances[i].agent.reset(InitialAgentPosition);
    }

    // update block cache before character's initialization
    updateObservation();

    for (std::uint32_t i = 0; i < m_batchSize; ++i) {
        World& world = m_instances[i].world;
        m_stepCounts[i] = 0;
        world.resetCharacters();
        const float spawnY = static_cast<float>(world.terrainHeight({ 0, 0 })) + 1.05f;
        m_instances[i].agent.reset({ InitialAgentPosition.x, spawnY, InitialAgentPosition.z });
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
        Agent& agent = m_instances[i].agent;
        StepResult& result = m_stepResults[i];
        const AgentState before = agent.state();
        agent.step(actions[i], s_fixedDt);
        m_instances[i].world.update(s_fixedDt, agent.state().position);
        ++m_stepCounts[i];

        const AgentState& after = agent.state();
        const float dx = after.position.x - before.position.x;
        const float dz = after.position.z - before.position.z;
        float reward = 0.01f + std::min(0.05f, std::sqrt(dx * dx + dz * dz) * 0.02f);
        reward += static_cast<float>(after.blocksCollected - before.blocksCollected) * 0.5f;
        reward += static_cast<float>(after.blocksPlaced - before.blocksPlaced) * 0.2f;
        if (after.position.y < World::s_minY)
            reward -= 5.0f;
        result.reward = reward;
        result.terminated = after.position.y < World::s_minY;
        result.truncated = m_maxSteps > 0 && m_stepCounts[i] >= m_maxSteps;
    }

    updateObservation();
    return { m_stepResults.get(), m_batchSize };
}

const Observation& Environment::updateObservation()
{
    for (std::uint32_t i = 0; i < m_batchSize; ++i)
        m_renderAgents[i] = m_instances[i].agent.state();
    const Renderer::RenderResult renderObservation
        = m_renderer.renderObservations({ m_instances.get(), m_batchSize });
    if (renderObservation.images.batchSize() != m_batchSize) [[unlikely]]
        fatalError("Observation renderer returned an unexpected batch size");
    m_observation.setImageBatchRef(renderObservation.images);
    m_observation.setVersion(renderObservation.version);
    return m_observation;
}

} // namespace blocklab
