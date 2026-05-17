#pragma once

#include "blocklab/Agent.h"
#include "blocklab/Observation.h"
#include "blocklab/World.h"

#include <cstdint>

namespace blocklab {

class ObservationRenderer {
public:
    virtual ~ObservationRenderer() = default;

    virtual Observation renderObservation(const World& world, const AgentState& agent) = 0;
};

struct StepResult {
    Observation observation;
    float reward = 0.0f;
    bool terminated = false;
    bool truncated = false;
};

class Environment {
public:
    explicit Environment(int32_t worldRadiusChunks = 3);

    void setObservationRenderer(ObservationRenderer* renderer);
    Observation reset(uint32_t seed = 1);
    StepResult step(const AgentAction& action);
    const Observation& observe() const { return m_observation; }

    const World& world() const { return m_world; }
    World& mutableWorld() { return m_world; }
    const Agent& agent() const { return m_agent; }

private:
    static constexpr float s_fixedDt = 1.0f / 60.0f;
    static constexpr int32_t s_maxSteps = 60 * 60 * 5;

    World m_world;
    Agent m_agent;
    ObservationRenderer* m_observationRenderer = nullptr;
    Observation m_observation;
    int32_t m_stepCount = 0;

    Observation updateObservation();
};

} // namespace blocklab
