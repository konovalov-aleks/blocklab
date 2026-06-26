#pragma once

#include <cstdint>

namespace blocklab {

enum class PlacementBlock : std::uint8_t {
    Torch = 1,
    Dirt = 2,
    Stone = 3,
};

struct AgentAction {
    float forward = 0.0f;
    float right = 0.0f;
    bool jump = false;
    bool dig = false;
    bool place = false;
    PlacementBlock placementBlock = PlacementBlock::Dirt;
    float yawDelta = 0.0f;
    float pitchDelta = 0.0f;
};

} // namespace blocklab
