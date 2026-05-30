#pragma once

#include "blocklab/Agent.h"
#include "blocklab/Observation.h"
#include "blocklab/World.h"

#include <cstdint>
#include <memory>
#include <span>

namespace blocklab {

class ObservationRenderer {
public:
    virtual ~ObservationRenderer() = default;

    virtual const Observation& renderObservations(std::span<const World>, std::span<const AgentState>) = 0;
};

struct StepResult {
    float reward = 0.0f;
    bool terminated = false;
    bool truncated = false;
};

class Environment {
public:
    Environment(ObservationRenderer&, uint32_t numEnvs, int32_t worldRadiusChunks);

    void reset(uint32_t seed = 1);
    std::span<const StepResult> step(std::span<const AgentAction> actions);
    // Returned observation remains valid until the next reset() or step() call.
    const Observation& observe() const { return *m_observation; }

    uint32_t batchSize() const { return m_batchSize; }
    const World& world(uint32_t batchId) const { return m_worlds[batchId]; }
    World& mutableWorld(uint32_t batchId) { return m_worlds[batchId]; }
    const Agent& agent(uint32_t batchId) const { return m_agents[batchId]; }

private:
    static constexpr float s_fixedDt = 1.0f / 60.0f;
    static constexpr int32_t s_maxSteps = 60 * 60 * 5;

    uint32_t m_batchSize = 0;
    std::unique_ptr<World[]> m_worlds;
    std::unique_ptr<Agent[]> m_agents;
    const Observation* m_observation = nullptr;
    ObservationRenderer& m_observationRenderer;
    std::unique_ptr<int32_t[]> m_stepCounts;
    std::unique_ptr<StepResult[]> m_stepResults;
    std::unique_ptr<AgentState[]> m_renderAgents;

    const Observation& updateObservation();
};

} // namespace blocklab
