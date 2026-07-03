#ifndef RENDER_PARAMS_HLSL
#define RENDER_PARAMS_HLSL

struct RenderFrameInfo {
    uint animationTimeMs;
    uint _padding1;
    uint _padding2;
    uint _padding3;
};

struct RenderProjectionInfo {
    float farPlane;
    float fovRadians;
    float fogStart;
    float fogEnd;
};

struct RenderSkyInfo {
    float3 skyColor;
    uint skyLightDimming;
    float3 skyLightDirection;
    float _padding;
};

struct RenderParams {
    float4 origin;
    float4 forward;
    float4 right;
    float4 up;
    int4 worldOriginAndWidth;
    int4 regionAndHeight;
    RenderFrameInfo frameInfo;
    RenderProjectionInfo projectionInfo;
    RenderSkyInfo skyInfo;
};

struct DrawPushConstants {
    uint envIndex;
    uint layerIndex;
    uint _padding0;
    uint _padding1;
};

#endif // RENDER_PARAMS_HLSL
