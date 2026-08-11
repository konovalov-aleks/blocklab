#pragma once

#include <blocklab/utility/Math.h>

#include <glm/gtx/norm.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

namespace blocklab {

struct CylinderDimensions {
    float radius;
    float height;
};

struct HitCylinder {
    CylinderDimensions dimensions;

    // bottom center point
    Vec3 position;
};

struct Ray {
    Vec3 origin;
    Vec3 direction;
};

inline bool collides(const HitCylinder& c1, const HitCylinder& c2)
{
    if (c1.position.y + c1.dimensions.height < c2.position.y
     || c2.position.y + c2.dimensions.height < c1.position.y)
        return false;

    const float xzDistanceSqr =
        sqr(c1.position.x - c2.position.x) + sqr(c1.position.z - c2.position.z);
    return xzDistanceSqr <= sqr(c1.dimensions.radius + c2.dimensions.radius);
}

// Distance along the ray (in metres; the ray direction must be unit length) at which the ray
// first enters the cylinder, or nullopt if it misses.
inline std::optional<float> rayCylinderEntryDistance(const Ray& r, const HitCylinder& c)
{
    const Vec2 rDirection2d = { r.direction.x, r.direction.z };
    const float horizontalLength = std::sqrtf(glm::length2(rDirection2d));
    if (horizontalLength < std::numeric_limits<float>::epsilon()) {
        // vertical ray
        const bool insideFootprint =
            r.origin.x <= c.position.x + c.dimensions.radius
            && r.origin.x >= c.position.x - c.dimensions.radius
            && r.origin.z <= c.position.z + c.dimensions.radius
            && r.origin.z >= c.position.z - c.dimensions.radius;
        const bool entering = r.direction.y > 0 ? r.origin.y <= c.position.y
                                                : r.origin.y >= c.position.y + c.dimensions.height;
        if (!insideFootprint || !entering)
            return std::nullopt;
        // enter through whichever cap the ray points at
        const float entryY = r.direction.y > 0 ? c.position.y : c.position.y + c.dimensions.height;
        return std::max(0.0f, (entryY - r.origin.y) / r.direction.y);
    }

    const Vec2 rDirection2dUnit = rDirection2d / horizontalLength;
    const Vec2 rOrigin2d = { r.origin.x, r.origin.z };
    const Vec2 oc = Vec2 { c.position.x, c.position.z } - rOrigin2d;

    const float tCenter = glm::dot(oc, rDirection2dUnit);
    const float ocLen2 = glm::length2(oc);
    const float h2 = ocLen2 - sqr(tCenter);

    const float r2 = sqr(c.dimensions.radius);
    if (h2 > r2)
        return std::nullopt;

    const float a = std::sqrtf(r2 - h2);
    // t is the distance along the horizontal unit direction; the 3D distance along the ray is
    // t / horizontalLength, because the ray direction has that horizontal projection.
    const float t1 = tCenter - a;
    const float t2 = tCenter + a;

    if (t1 >= 0) {
        const float y1 = r.origin.y + r.direction.y * (t1 / horizontalLength);
        if (y1 >= c.position.y && y1 <= c.position.y + c.dimensions.height)
            return t1 / horizontalLength;
    }
    if (t2 >= 0) {
        const float y2 = r.origin.y + r.direction.y * (t2 / horizontalLength);
        if (y2 >= c.position.y && y2 <= c.position.y + c.dimensions.height)
            return t2 / horizontalLength;
    }
    return std::nullopt;
}

inline bool collides(const Ray& r, const HitCylinder& c)
{
    return rayCylinderEntryDistance(r, c).has_value();
}

} // namespace blocklab
