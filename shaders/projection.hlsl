#ifndef PROJECTION_HLSL
#define PROJECTION_HLSL

#include "render_params.hlsl"

struct ViewPosition {
    float x;
    float y;
    float z;
};

ViewPosition worldToView(float3 worldPosition, RenderParams params)
{
    float3 relative = worldPosition - params.origin;

    ViewPosition result;
    result.x = dot(relative, params.right);
    result.y = dot(relative, params.up);
    result.z = dot(relative, params.forward);
    return result;
}

float4 projectViewPosition(ViewPosition viewPosition, RenderParams params)
{
    float nearPlane = 0.05;
    float farPlane = params.projectionInfo.farPlane;
    float tanHalfFov = tan(params.projectionInfo.fovRadians * 0.5);
    float aspect = float(params.viewportWidth) / float(params.viewportHeight);

    float4 result;
    result.x = viewPosition.x / (aspect * tanHalfFov);
    result.y = -viewPosition.y / tanHalfFov;
    result.z = viewPosition.z * farPlane / (farPlane - nearPlane) - farPlane * nearPlane / (farPlane - nearPlane);
    result.w = viewPosition.z;
    return result;
}

#endif // PROJECTION_HLSL
