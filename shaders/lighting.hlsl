#include "render_params.hlsl"

float faceSkyLight(float3 faceNormal, RenderParams params)
{
    float lightToFaceNormalDot = dot(faceNormal, params.skyInfo.skyLightDirection);
    float faceDirectionLight = lerp(0.4, 1.0, saturate(lightToFaceNormalDot));

    // We switch to a fallback lighting when the sky light source approaches the horizon to avoid visual
    // artifacts from the sudden change in light direction.
    float fallbackLightFactor = 1.0 - smoothstep(0.0, 0.6, params.skyInfo.skyLightDirection.y);
    if (fallbackLightFactor < 1E-3)
        return faceDirectionLight;

    static const float3 fallbackLightDirection = normalize(float3(0.0, 1.0, 0.3));

    float fallbackLightToFaceNormalDot = dot(faceNormal, fallbackLightDirection);
    float fallbackLight = lerp(0.4, 1.0, saturate(fallbackLightToFaceNormalDot));
    return lerp(faceDirectionLight, fallbackLight, fallbackLightFactor);
}


