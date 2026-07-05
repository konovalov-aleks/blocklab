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
    float3 origin;
    float _padding1;
    float3 forward;
    float _padding2;
    float3 right;
    float _padding3;
    float3 up;
    float _padding4;
    int3 worldOrigin;
    int viewportWidth;
    int3 regionOrigin;
    int viewportHeight;
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
