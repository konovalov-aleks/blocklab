struct VertexOutput {
    float4 position : SV_Position;
    float4 color : TEXCOORD0;
    float3 worldPosition : TEXCOORD1;
    float light : TEXCOORD2;
    float4 fog : TEXCOORD3;
    float2 uv : TEXCOORD4;
    nointerpolation uint material : TEXCOORD5;
    uint layer : SV_RenderTargetArrayIndex;
};
