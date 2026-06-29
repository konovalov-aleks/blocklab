#include "render_params.hlsl"
#include "vertex_output.hlsl"

struct MeshVertex {
    float4 position;
    float4 colorAndLight;
    float4 uvMaterial;
};

struct EntityInstance {
    float4 positionAndYaw;
    float4 velocityAndKind;
};

static const uint EntityKindNone = 0;
static const uint EntityKindPig = 1;

StructuredBuffer<MeshVertex> vertices : register(t0, space0);
StructuredBuffer<EntityInstance> instances : register(t1, space0);
StructuredBuffer<RenderParams> paramsBuffer : register(t2, space0);

[[vk::push_constant]] cbuffer PushConstants
{
    DrawPushConstants pushConstants;
};

VertexOutput meshVertexMain(uint vertexId : SV_VertexID, uint instanceId : SV_InstanceID)
{
    RenderParams params = paramsBuffer[pushConstants.envIndex];
    MeshVertex vertex = vertices[vertexId];
    float3 worldPosition = vertex.position.xyz;
    EntityInstance instance = instances[instanceId];
    uint entityKind = uint(instance.velocityAndKind.w + 0.5);
    if (entityKind != EntityKindNone) {
        float yaw = instance.positionAndYaw.w;
        float3 entityRight = float3(cos(yaw), 0.0, -sin(yaw));
        float3 entityForward = float3(sin(yaw), 0.0, cos(yaw));
        worldPosition = instance.positionAndYaw.xyz + entityRight * vertex.position.x
            + float3(0.0, vertex.position.y, 0.0) + entityForward * vertex.position.z;
        // Pig mesh uses position.w as a leg animation marker: 0 disables it, sign selects the stride phase.
        float legPhaseMarker = vertex.position.w;
        float stridePhase = legPhaseMarker > 0.0 ? 0.0 : 3.14159265;
        float animationTime = float(params.frameInfo.animationTimeMs) * 0.001;
        float stride = sin(animationTime * 9.0 + stridePhase);
        float horizontalSpeedSq = dot(instance.velocityAndKind.xz, instance.velocityAndKind.xz);
        if (abs(legPhaseMarker) > 1.5 && horizontalSpeedSq > 0.0025) {
            worldPosition += entityForward * (stride * 0.08);
            worldPosition.y += max(0.0, -stride) * 0.05;
        }
    }
    float3 relative = worldPosition - params.origin.xyz;
    float viewX = dot(relative, params.right.xyz);
    float viewY = dot(relative, params.up.xyz);
    float viewZ = dot(relative, params.forward.xyz);

    float nearPlane = 0.05;
    float farPlane = params.projectionInfo.farPlane;
    float tanHalfFov = tan(params.projectionInfo.fovRadians * 0.5);
    float aspect = float(params.worldOriginAndWidth.w) / float(params.regionAndHeight.w);

    float fogIntensity = saturate(
        (viewZ - params.projectionInfo.fogStart) / max(params.projectionInfo.fogEnd - params.projectionInfo.fogStart, 0.001));

    VertexOutput output;
    output.position.x = viewX / (aspect * tanHalfFov);
    output.position.y = -viewY / tanHalfFov;
    output.position.z = viewZ * farPlane / (farPlane - nearPlane) - farPlane * nearPlane / (farPlane - nearPlane);
    output.position.w = viewZ;
    output.color = float4(vertex.colorAndLight.rgb, 1.0);
    output.worldPosition = worldPosition;
    output.light = vertex.colorAndLight.a;
    output.uvMaterial = vertex.uvMaterial.xyz;
    output.fog = float4(params.skyInfo.skyColor, fogIntensity);
    output.layer = pushConstants.layerIndex;
    return output;
}
