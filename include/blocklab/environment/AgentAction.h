#pragma once

#include <blocklab/inventory/Inventory.h>

#include <optional>

namespace blocklab {

struct AgentAction {
    float forward = 0.0f;
    float right = 0.0f;
    bool jump = false;
    bool attack = false;
    bool use = false;
    std::optional<Inventory::SlotId> activeHotbarSlot;
    float yawDelta = 0.0f;
    float pitchDelta = 0.0f;
};

} // namespace blocklab
