struct RenderParams {
    float4 origin;
    float4 forward;
    float4 right;
    float4 up;
    int4 worldOriginAndWidth;
    int4 regionAndHeight;
    int4 overrideInfo;
    float4 tuning;
};

cbuffer Params : register(b0, space1)
{
    RenderParams params;
};

