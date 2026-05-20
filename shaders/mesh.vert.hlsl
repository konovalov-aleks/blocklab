#include "graphics_common.hlsl"

struct MeshVertex {
    float4 position;
    float4 uvAndShade;
};

struct EntityInstance {
    float4 positionAndYaw;
    float4 velocityAndKind;
};

struct VertexOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    float shade : TEXCOORD1;
    float fog : TEXCOORD2;
};

StructuredBuffer<MeshVertex> vertices : register(t0, space0);
StructuredBuffer<EntityInstance> instances : register(t1, space0);

VertexOutput meshVertexMain(uint vertexId : SV_VertexID, uint instanceId : SV_InstanceID)
{
    MeshVertex vertex = vertices[vertexId];
    float3 worldPosition = vertex.position.xyz;
    EntityInstance instance = instances[instanceId];
    if (instance.velocityAndKind.w > 0.5) {
        float yaw = instance.positionAndYaw.w;
        float3 entityRight = float3(cos(yaw), 0.0, -sin(yaw));
        float3 entityForward = float3(sin(yaw), 0.0, cos(yaw));
        worldPosition = instance.positionAndYaw.xyz
            + entityRight * vertex.position.x
            + float3(0.0, vertex.position.y, 0.0)
            + entityForward * vertex.position.z;
        float phase = vertex.position.w > 0.0 ? 0.0 : 3.14159265;
        float animationTime = float(params.overrideInfo.z) * 0.001;
        float stride = sin(animationTime * 9.0 + phase);
        if (abs(vertex.position.w) > 1.5 && instance.velocityAndKind.x > 0.05) {
            worldPosition += entityForward * (stride * 0.08);
            worldPosition.y += max(0.0, -stride) * 0.05;
        }
    }
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
