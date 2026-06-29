struct VertexOutput {
    float4 position : SV_Position;
    float4 color : TEXCOORD0;
    float3 worldPosition : TEXCOORD1;
    float light : TEXCOORD2;
    float4 fog : TEXCOORD3;
    float3 uvMaterial : TEXCOORD4;
    uint layer : SV_RenderTargetArrayIndex;
};

