#pragma once

#include <glm/glm.hpp>

#include <cmath>
#include <cstdint>

namespace blocklab {

using Vec2 = glm::vec2;
using Vec3 = glm::vec3;
using Vec4 = glm::vec4;
using IVec3 = glm::ivec3;
using IVec4 = glm::ivec4;

inline static constexpr float Pi = 3.14159265358979323846f;

inline constexpr int32_t floorToInt(float value) { return static_cast<int32_t>(std::floor(value)); }

} // namespace blocklab
