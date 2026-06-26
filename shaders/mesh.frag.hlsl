#include "material.hlsl"

static const float3 SkyColor = float3(0.148302, 0.374624, 0.717623);

struct FragmentInput {
    float4 color : TEXCOORD0;
    float3 worldPosition : TEXCOORD1;
    float shade : TEXCOORD2;
    float fog : TEXCOORD3;
    float3 uvMaterial : TEXCOORD4;
};

float hash21(float2 p)
{
    float3 p3 = frac(float3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return frac((p3.x + p3.y) * p3.z);
}

float3 srgbToLinear(float3 color)
{
    return pow(saturate(color), 2.2);
}

float3 palette(float3 color)
{
    return srgbToLinear(color);
}

float3 texelColor(uint material, float2 uv, float3 worldPosition, float3 fallback)
{
    if (material == MaterialVertexColor)
        return srgbToLinear(fallback);

    float2 texel = material == MaterialGrassTop ? floor(worldPosition.xz * 16.0) : floor(saturate(uv) * 15.999);
    float n = hash21(texel + float(material) * 17.0);

    if (material == MaterialGrassTop) {
        float blade = step(0.68, hash21(texel * 1.7 + 4.0));
        float3 grass = lerp(float3(0.22, 0.50, 0.18), float3(0.36, 0.64, 0.27), n);
        return palette(grass + blade * float3(0.02, 0.04, 0.01));
    }
    if (material == MaterialDirt)
        return palette(lerp(float3(0.35, 0.20, 0.10), float3(0.56, 0.36, 0.18), n));
    if (material == MaterialStone)
        return palette(lerp(float3(0.43, 0.44, 0.44), float3(0.57, 0.58, 0.58), n));
    if (material == MaterialGrassSide) {
        float3 dirt = lerp(float3(0.32, 0.20, 0.11), float3(0.48, 0.31, 0.17), n);
        float3 grass = lerp(float3(0.18, 0.46, 0.15), float3(0.30, 0.58, 0.21), n);
        float dirtChance = 1.0;
        if (texel.y >= 15.0)
            dirtChance = 0.0;
        else if (texel.y >= 14.0)
            dirtChance = 0.12;
        else if (texel.y >= 13.0)
            dirtChance = 0.32;
        else if (texel.y >= 12.0)
            dirtChance = 0.58;
        else
            dirtChance = 1.0;
        float dirtMask = step(1.0 - dirtChance, hash21(texel + 11.0));
        return palette(lerp(grass, dirt, dirtMask));
    }
    if (material == MaterialPigSkin)
        return palette(lerp(float3(0.86, 0.54, 0.63), float3(0.96, 0.66, 0.73), n));
    if (material == MaterialPigSnout)
        return palette(lerp(float3(0.90, 0.60, 0.68), float3(1.00, 0.72, 0.79), n));
    if (material == MaterialTorchSide) {
        float3 wood = lerp(float3(0.46, 0.25, 0.10), float3(0.58, 0.34, 0.16), n);
        float3 ember = lerp(float3(1.00, 0.52, 0.08), float3(1.00, 0.88, 0.24), n);
        return palette(lerp(wood, ember, step(0.72, uv.y)));
    }
    if (material == MaterialTorchTop)
        return palette(lerp(float3(1.00, 0.52, 0.08), float3(1.00, 0.88, 0.24), n));
    return srgbToLinear(fallback);
}

float4 meshFragmentMain(FragmentInput input) : SV_Target0
{
    uint material = uint(input.uvMaterial.z + 0.5);
    float3 albedo = texelColor(material, input.uvMaterial.xy, input.worldPosition, input.color.rgb);
    float3 lit = saturate(albedo * input.shade);
    return float4(lerp(lit, SkyColor, input.fog), 1.0);
}
