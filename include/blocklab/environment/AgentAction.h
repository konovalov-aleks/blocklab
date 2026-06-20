#pragma once

namespace blocklab {

struct AgentAction {
    float forward = 0.0f;
    float right = 0.0f;
    bool jump = false;
    bool dig = false;
    bool place = false;
    float yawDelta = 0.0f;
    float pitchDelta = 0.0f;
};

} // namespace blocklab
