#pragma once

#include <glm/glm.hpp>

#include <cmath>
#include <cstdint>
#include <numbers>

namespace blocklab {

using Vec2 = glm::vec2;
using Vec3 = glm::vec3;
using Vec4 = glm::vec4;
using IVec2 = glm::ivec2;
using IVec3 = glm::ivec3;
using IVec4 = glm::ivec4;
using UVec3 = glm::uvec3;

inline static constexpr float Pi = std::numbers::pi_v<float>;

inline constexpr std::int32_t floorToInt32(float value)
{
    return static_cast<std::int32_t>(std::floor(value));
}

inline constexpr IVec3 floorToInt32(Vec3 v)
{
    return {
        floorToInt32(v.x),
        floorToInt32(v.y),
        floorToInt32(v.z),
    };
}

inline float wrapAngle(float angle)
{
    while (angle > Pi)
        angle -= 2.0f * Pi;
    while (angle < -Pi)
        angle += 2.0f * Pi;
    return angle;
}

} // namespace blocklab
