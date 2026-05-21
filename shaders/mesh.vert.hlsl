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

struct MeshVertex {
    float4 position;
    float4 colorAndShade;
    float4 uvMaterial;
};

struct EntityInstance {
    float4 positionAndYaw;
    float4 velocityAndKind;
};

static const uint EntityKindNone = 0;
static const uint EntityKindPig = 1;

struct VertexOutput {
    float4 position : SV_Position;
    float4 color : TEXCOORD0;
    float3 worldPosition : TEXCOORD1;
    float shade : TEXCOORD2;
    float fog : TEXCOORD3;
    float3 uvMaterial : TEXCOORD4;
};

StructuredBuffer<MeshVertex> vertices : register(t0, space0);
StructuredBuffer<EntityInstance> instances : register(t1, space0);

cbuffer Params : register(b0, space1)
{
    RenderParams params;
};

VertexOutput meshVertexMain(uint vertexId : SV_VertexID, uint instanceId : SV_InstanceID)
{
    MeshVertex vertex = vertices[vertexId];
    float3 worldPosition = vertex.position.xyz;
    EntityInstance instance = instances[instanceId];
    uint entityKind = uint(instance.velocityAndKind.w + 0.5);
    if (entityKind != EntityKindNone) {
        float yaw = instance.positionAndYaw.w;
        float3 entityRight = float3(cos(yaw), 0.0, -sin(yaw));
        float3 entityForward = float3(sin(yaw), 0.0, cos(yaw));
        worldPosition = instance.positionAndYaw.xyz
            + entityRight * vertex.position.x
            + float3(0.0, vertex.position.y, 0.0)
            + entityForward * vertex.position.z;
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
    output.position.y = -viewY / tanHalfFov;
    output.position.z = viewZ * farPlane / (farPlane - nearPlane) - farPlane * nearPlane / (farPlane - nearPlane);
    output.position.w = viewZ;
    output.color = float4(vertex.colorAndShade.rgb, 1.0);
    output.worldPosition = worldPosition;
    output.shade = vertex.colorAndShade.a;
    output.uvMaterial = vertex.uvMaterial.xyz;
    output.fog = saturate((viewZ - params.tuning.z) / max(params.tuning.w - params.tuning.z, 0.001));
    return output;
}
