Texture2D<float4> blockAtlas : register(t0, space2);
SamplerState blockSampler : register(s0, space2);

static const float3 SkyColor = float3(0.42, 0.64, 0.86);

struct FragmentInput {
    float2 uv : TEXCOORD0;
    float shade : TEXCOORD1;
    float fog : TEXCOORD2;
};

float4 meshFragmentMain(FragmentInput input) : SV_Target0
{
    float4 texel = blockAtlas.Sample(blockSampler, input.uv);
    float3 litColor = texel.rgb * input.shade;
    return float4(lerp(litColor, SkyColor, input.fog), texel.a);
}
