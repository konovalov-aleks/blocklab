#pragma once

#include "observation/Observation.h"
#include "AgentAction.h"

#include <cstdint>
#include <memory>
#include <span>

namespace blocklab {

class Agent;
struct AgentState;
class Renderer;
class World;

namespace test {
    class EnvironmentInternalAccessTestHelper;
} // namespace test

struct StepResult {
    float reward = 0.0f;
    bool terminated = false;
    bool truncated = false;
};

class Environment {
public:
    Environment(Renderer&, std::uint32_t numEnvs, std::uint32_t maxSteps = 0);
    ~Environment();

    // TODO implement per-batch reset
    void reset(std::uint32_t seed = 1);
    std::span<const StepResult> step(std::span<const AgentAction> actions);
    // Returned observation remains valid until the next reset() or step() call.
    const Observation& observe() const { return m_observation; }

    std::uint32_t batchSize() const { return m_batchSize; }

private:
    const Observation& updateObservation();

    static constexpr float s_fixedDt = 1.0f / 60.0f;

    std::unique_ptr<World[]> m_worlds;
    std::unique_ptr<Agent[]> m_agents;
    Observation m_observation;
    Renderer& m_renderer;
    std::unique_ptr<std::uint32_t[]> m_stepCounts;
    std::unique_ptr<StepResult[]> m_stepResults;
    std::unique_ptr<AgentState[]> m_renderAgents;
    const std::uint32_t m_batchSize;
    const std::uint32_t m_maxSteps;

    friend class test::EnvironmentInternalAccessTestHelper;
    friend class EnvironmentInternalAccessBenchmarkHelper;
};

} // namespace blocklab
