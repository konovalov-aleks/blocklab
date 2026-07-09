#ifndef FOG_HLSL
#define FOG_HLSL

#include "render_params.hlsl"

float fogIntensity(float viewZ, RenderParams params)
{
    return saturate(
        (viewZ - params.projectionInfo.fogStart) / max(params.projectionInfo.fogEnd - params.projectionInfo.fogStart, 0.001));
}

#endif // FOG_HLSL
