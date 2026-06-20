#pragma once

#include "AgentAction.h"
#include "Observation.h"

#include <cstdint>
#include <memory>
#include <span>

namespace blocklab {

class Agent;
class AgentState;
class Renderer;
class World;

struct StepResult {
    float reward = 0.0f;
    bool terminated = false;
    bool truncated = false;
};

class Environment {
public:
    Environment(Renderer&, std::uint32_t numEnvs);
    ~Environment();

    // TODO implement per-batch reset
    void reset(std::uint32_t seed = 1);
    std::span<const StepResult> step(std::span<const AgentAction> actions);
    // Returned observation remains valid until the next reset() or step() call.
    const Observation& observe() const { return *m_observation; }

    std::uint32_t batchSize() const { return m_batchSize; }

private:
    static constexpr float s_fixedDt = 1.0f / 60.0f;
    static constexpr std::int32_t s_maxSteps = 60 * 60 * 5;

    std::uint32_t m_batchSize = 0;
    std::unique_ptr<World[]> m_worlds;
    std::unique_ptr<Agent[]> m_agents;
    const Observation* m_observation = nullptr;
    Renderer& m_renderer;
    std::unique_ptr<std::int32_t[]> m_stepCounts;
    std::unique_ptr<StepResult[]> m_stepResults;
    std::unique_ptr<AgentState[]> m_renderAgents;

    const Observation& updateObservation();

    friend class EnvironmentInternalAccessTestHelper;
    friend class EnvironmentInternalAccessBenchmarkHelper;
};

} // namespace blocklab
