#include "graphics_common.hlsl"

struct MeshVertex {
    float4 position;
    float4 uvAndShade;
};

struct VertexOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    float shade : TEXCOORD1;
    float fog : TEXCOORD2;
};

StructuredBuffer<MeshVertex> vertices : register(t0, space0);

VertexOutput meshVertexMain(uint vertexId : SV_VertexID)
{
    MeshVertex vertex = vertices[vertexId];
    float3 worldPosition = vertex.position.xyz;
    float3 relative = worldPosition - params.origin.xyz;
    float viewX = dot(relative, params.right.xyz);
    float viewY = dot(relative, params.up.xyz);
    float viewZ = dot(relative, params.forward.xyz);

    float nearPlane = 0.05;
    float farPlane = params.tuning.x;
    float tanHalfFov = tan(params.tuning.y * 0.5);
    float aspect = float(params.worldOriginAndWidth.w) / float(params.regionAndHeight.w);

    VertexOutput output;
    output.position.x = viewX / (aspect * tanHalfFov);
    output.position.y = viewY / tanHalfFov;
    output.position.z = viewZ * farPlane / (farPlane - nearPlane) - farPlane * nearPlane / (farPlane - nearPlane);
    output.position.w = viewZ;
    output.uv = vertex.uvAndShade.xy;
    output.shade = vertex.uvAndShade.z;
    output.fog = saturate((viewZ - params.tuning.z) / max(params.tuning.w - params.tuning.z, 0.001));
    return output;
}
