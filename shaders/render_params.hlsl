struct RenderFrameInfo {
    int animationTimeMs;
};

struct RenderParams {
    float4 origin;
    float4 forward;
    float4 right;
    float4 up;
    int4 worldOriginAndWidth;
    int4 regionAndHeight;
    RenderFrameInfo frameInfo;
    float4 tuning;
};

struct DrawPushConstants {
    uint envIndex;
    uint layerIndex;
    uint _padding0;
    uint _padding1;
};
