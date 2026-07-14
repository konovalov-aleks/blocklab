#pragma once

#include <blocklab/utility/Math.h>

#include <glm/gtx/norm.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <limits>
#include <optional>

namespace blocklab {

namespace details::raycast {

    struct RaycastAxis {
        // movement direction along the axis
        int step;
        // max t value to reach the cell boundary
        float tMax;
        // t distance to traverse a cell
        float tDelta;
    };

    inline RaycastAxis initAxis(float origin, int cellIndex, float speed)
    {
        if (std::abs(speed) < std::numeric_limits<float>::epsilon()) {
            return {
                .step = 0,
                .tMax = std::numeric_limits<float>::max(),
                .tDelta = std::numeric_limits<float>::max(),
            };
        }

        float nextBoundary;
        int step;
        if (speed > 0) {
            nextBoundary = static_cast<float>(cellIndex + 1);
            step = 1;
        } else {
            nextBoundary = static_cast<float>(cellIndex);
            step = -1;
        }
        return {
            .step = step,
            .tMax = (nextBoundary - origin) / speed,
            .tDelta = static_cast<float>(step) / speed,
        };
    }

} // namespace details::raycast

enum class RaycastCommand {
    Break,
    Continue
};

struct RaycastCbParams {
    IVec3 pos;
    // nullopt is only possible in the origin cell
    std::optional<IVec3> normal;
    float distance;
};

template <typename T>
concept RaycastHandlerT =
    std::invocable<T, RaycastCbParams>
 && std::same_as<std::invoke_result_t<T, RaycastCbParams>, RaycastCommand>;

template <RaycastHandlerT Callback>
bool raycast(Vec3 origin, Vec3 direction, const Callback& cb)
{
    using namespace blocklab::details::raycast;

    assert(std::abs(glm::length2(direction) - 1.0f) < 1E-5);
    IVec3 cell = floorToInt32(origin);
    float t = 0.0f;

    RaycastAxis axisX = initAxis(origin.x, cell.x, direction.x);
    RaycastAxis axisY = initAxis(origin.y, cell.y, direction.y);
    RaycastAxis axisZ = initAxis(origin.z, cell.z, direction.z);

    std::optional<IVec3> normal;

    for (;;) {
        if (cb({ cell, normal, t }) == RaycastCommand::Break)
            return true;

        t = std::min({ axisX.tMax, axisY.tMax, axisZ.tMax });

        if (t == axisX.tMax) {
            cell.x += axisX.step;
            axisX.tMax += axisX.tDelta;
            normal = { -axisX.step, 0, 0 };

        } else if (t == axisY.tMax) {
            cell.y += axisY.step;
            axisY.tMax += axisY.tDelta;
            normal = { 0, -axisY.step, 0 };

        } else {
            cell.z += axisZ.step;
            axisZ.tMax += axisZ.tDelta;
            normal = { 0, 0, -axisZ.step };
        }
    }
    return false;
}

} // namespace blocklab
