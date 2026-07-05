#include "lighting.hlsl"
#include "material.hlsl"
#include "render_params.hlsl"
#include "vertex_output.hlsl"

struct MeshVertex {
    float4 position;
    float4 normal;
    float2 uv;
    uint material;
    uint _padding0;
    float3 color;
    float _padding1;
};

struct EntityInstance {
    float3 position;
    float yaw;
    float4 velocity;
    uint kind;
    float blockLight;
    float skyLight;
    float _padding;
};

static const uint EntityKindNone = 0;
static const uint RenderEntityFamilyShift = 24;
static const uint RenderEntityLocalIdMask = 0x00FFFFFF;
static const uint RenderEntityFamilyCharacter = 1;
static const uint RenderEntityFamilyDrop = 2;
static const uint EntityKindPig = (RenderEntityFamilyCharacter << RenderEntityFamilyShift) | 1;
static const uint EntityKindDirtDrop = (RenderEntityFamilyDrop << RenderEntityFamilyShift) | 1;
static const uint EntityKindStoneDrop = (RenderEntityFamilyDrop << RenderEntityFamilyShift) | 2;
static const uint EntityKindTorchDrop = (RenderEntityFamilyDrop << RenderEntityFamilyShift) | 3;

uint renderEntityFamily(uint kind)
{
    return kind >> RenderEntityFamilyShift;
}

uint renderEntityLocalId(uint kind)
{
    return kind & RenderEntityLocalIdMask;
}

uint dropMaterial(uint kind, uint meshMaterial)
{
    uint localId = renderEntityLocalId(kind);
    if (localId == renderEntityLocalId(EntityKindDirtDrop))
        return MaterialDirt;
    if (localId == renderEntityLocalId(EntityKindStoneDrop))
        return MaterialStone;
    if (localId == renderEntityLocalId(EntityKindTorchDrop))
        return meshMaterial;
    return MaterialVertexColor;
}

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
    float3 worldNormal = vertex.normal.xyz;
    EntityInstance instance = instances[instanceId];
    uint entityKind = instance.kind;
    uint entityFamily = renderEntityFamily(entityKind);
    float3 itemLocalPosition = vertex.position.xyz;
    if (entityKind != EntityKindNone) {
        float3 localPosition = itemLocalPosition;
        float yaw = instance.yaw;
        float animationTime = float(params.frameInfo.animationTimeMs) * 0.001;
        float dropBob = 0.0;
        if (entityFamily == RenderEntityFamilyDrop) {
            // Slow item rotation
            yaw += animationTime * 1.5;
            // Small vertical bobbing
            dropBob = sin(animationTime * 3.0 + instance.yaw) * 0.06;
            if (entityKind == EntityKindTorchDrop) {
                // Torch drops are item-like: thinner than block drops, slightly taller, and less floaty
                dropBob *= 0.5;
                localPosition = float3(localPosition.x * 0.36, localPosition.y * 1.4 - 0.08, localPosition.z * 0.36);
            }
            itemLocalPosition = localPosition;
        }

        float3 entityRight = float3(cos(yaw), 0.0, -sin(yaw));
        float3 entityForward = float3(sin(yaw), 0.0, cos(yaw));

        worldPosition = instance.position + entityRight * localPosition.x
            + float3(0.0, localPosition.y, 0.0) + entityForward * localPosition.z;
        worldPosition.y += dropBob;
        worldNormal = entityRight * vertex.normal.x
            + float3(0.0, vertex.normal.y, 0.0) + entityForward * vertex.normal.z;

        // Pig mesh uses position.w as a leg animation marker: 0 disables it, sign selects the stride phase.
        float legPhaseMarker = vertex.position.w;
        float stridePhase = legPhaseMarker > 0.0 ? 0.0 : 3.14159265;
        float stride = sin(animationTime * 9.0 + stridePhase);
        float horizontalSpeedSq = dot(instance.velocity.xz, instance.velocity.xz);
        if (entityFamily == RenderEntityFamilyCharacter && abs(legPhaseMarker) > 1.5 && horizontalSpeedSq > 0.0025) {
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

    float faceSkyLightFactor = faceSkyLight(normalize(worldNormal), params);
    float light = max(faceSkyLightFactor * (instance.skyLight - params.skyInfo.skyLightDimming / 15.0f), instance.blockLight);

    VertexOutput output;
    output.position.x = viewX / (aspect * tanHalfFov);
    output.position.y = -viewY / tanHalfFov;
    output.position.z = viewZ * farPlane / (farPlane - nearPlane) - farPlane * nearPlane / (farPlane - nearPlane);
    output.position.w = viewZ;
    output.color = float4(vertex.color, 1.0);
    output.worldPosition = worldPosition;
    output.light = 1.0 * light;
    output.uv = vertex.uv;
    output.material = vertex.material;
    if (entityFamily == RenderEntityFamilyDrop) {
        uint material = dropMaterial(entityKind, vertex.material);
        output.material = material;
        if (entityKind == EntityKindTorchDrop && material == MaterialTorchSide)
            output.uv.y = saturate((itemLocalPosition.y + 0.255) / 0.35);
    }
    output.fog = float4(params.skyInfo.skyColor, fogIntensity);
    output.layer = pushConstants.layerIndex;
    return output;
}
