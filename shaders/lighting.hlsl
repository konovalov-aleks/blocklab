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

float faceBlockLight(float3 faceNormal)
{
    // Block light has no tracked source direction. Use a stable fake direction to keep torch-lit meshes from looking flat.
    static const float3 fakeBlockKeyLightDirection = normalize(float3(0.35, 0.85, 0.45));
    static const float3 fakeBlockFillLightDirection = normalize(float3(-0.55, 0.65, -0.25));

    float keyLight = saturate(dot(faceNormal, fakeBlockKeyLightDirection));
    float fillLight = saturate(dot(faceNormal, fakeBlockFillLightDirection));
    return 0.45 + keyLight * 0.40 + fillLight * 0.15;
}
