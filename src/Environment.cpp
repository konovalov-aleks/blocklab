#include "blocklab/Environment.h"

#include <algorithm>
#include <cmath>

namespace blocklab {

Environment::Environment(int32_t worldRadiusChunks)
    : m_world(worldRadiusChunks)
{
    reset();
}

void Environment::setObservationRenderer(ObservationRenderer* renderer)
{
    m_observationRenderer = renderer;
    updateObservation();
}

Observation Environment::reset(uint32_t seed)
{
    m_world.reset(seed);
    m_stepCount = 0;
    const float spawnY = m_world.groundHeight(0.0f, 0.0f) + 0.05f;
    m_agent.reset({ 0.5f, spawnY, 0.5f });
    return updateObservation();
}

StepResult Environment::step(const AgentAction& action)
{
    const AgentState before = m_agent.state();
    m_agent.step(m_world, action, s_fixedDt);
    m_world.updateCharacters(s_fixedDt, m_agent.state().position);
    ++m_stepCount;

    const AgentState& after = m_agent.state();
    const float dx = after.position.x - before.position.x;
    const float dz = after.position.z - before.position.z;
    float reward = 0.01f + std::min(0.05f, std::sqrt(dx * dx + dz * dz) * 0.02f);
    reward += static_cast<float>(after.blocksCollected - before.blocksCollected) * 0.5f;
    reward += static_cast<float>(after.blocksPlaced - before.blocksPlaced) * 0.2f;
    if (after.position.y < -8.0f)
        reward -= 5.0f;

    return {
        .observation = updateObservation(),
        .reward = reward,
        .terminated = after.position.y < -8.0f,
        .truncated = m_stepCount >= s_maxSteps,
    };
}

Observation Environment::updateObservation()
{
    if (!m_observationRenderer) {
        m_observation = {};
        return m_observation;
    }
    m_observation = m_observationRenderer->renderObservation(m_world, m_agent.state());
    return m_observation;
}

} // namespace blocklab
