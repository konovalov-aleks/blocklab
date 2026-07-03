#pragma once

#include "WorldTime.h"
#include <blocklab/utility/Math.h>

#include <cstdint>
#include <utility>

namespace blocklab {

// returns sky light level as an integer [0..15]
std::uint8_t skyLightAtTime(WorldTime);

// returns sky light level as a float value [0.0 .. 1.0]
float skyLightFactorAtTime(WorldTime);

Vec3 skyColorAtTime(WorldTime);

std::pair<Vec3, float> skyColorAndLightFactorAtTime(WorldTime);

Vec3 skyLightDirectionAtTime(WorldTime);

} // namespace blocklab
