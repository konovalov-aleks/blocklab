#pragma once

#include "Agent.h"

#include <world/World.h>

namespace blocklab {

struct EnvInstance {
    EnvInstance() : agent(world) {}

    World world;
    Agent agent;
};

} // namespace
