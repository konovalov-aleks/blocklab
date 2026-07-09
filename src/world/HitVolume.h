#pragma once

#include <blocklab/utility/Math.h>

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

inline bool collides(const HitCylinder& c1, const HitCylinder& c2)
{
    if (c1.position.y + c1.dimensions.height < c2.position.y
     || c2.position.y + c2.dimensions.height < c1.position.y)
        return false;

    const float xzDistanceSqr =
        sqr(c1.position.x - c2.position.x) + sqr(c1.position.z - c2.position.z);
    return xzDistanceSqr <= sqr(c1.dimensions.radius + c2.dimensions.radius);
}

} // namespace blocklab
