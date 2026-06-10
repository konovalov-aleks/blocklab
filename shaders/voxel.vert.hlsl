#include "material.hlsl"
#include "render_params.hlsl"
#include "vertex_output.hlsl"

struct TerrainHeader {
    int originX;
    int originZ;
};

struct Voxel {
    // x : 6
    // y : 9
    // z : 6
    // visibleFacesMask: 6
    // blockType: 5
    uint data;
};

static const int N_VERTICES_PER_VOXEL = 36;

// this should correspond to VisibleFace in Voxel.h
static const uint FACE_MASKS[] = {
    1 << 0, // left
    1 << 1, // right
    1 << 2, // top
    1 << 3, // bottom
    1 << 4, // front
    1 << 5, // back
};

static const uint FACE_MATERIALS[][6] = {
    // left / right / top / bottom / front / back

    // 0 - Air
    {  -1, -1, -1, -1, -1, -1 }, // invisible
    // 1 - Grass
    { MaterialGrassSide, MaterialGrassSide, MaterialGrassTop, MaterialDirt, MaterialGrassSide, MaterialGrassSide },
    // 2 - Dirt
    { MaterialDirt, MaterialDirt, MaterialDirt, MaterialDirt, MaterialDirt, MaterialDirt },
    // 3 - Stone
    { MaterialStone, MaterialStone, MaterialStone, MaterialStone, MaterialStone, MaterialStone },
};

static const float FACE_SHADES[6] = {
    0.78, // left
    0.78, // right
    1.0, // top
    0.48, // bottom
    0.68, // front
    0.68, // back
};

static const float3 CUBE_VERTICES[36] = {
    // left face
    float3(0,0,0), float3(0,0,1), float3(0,1,1), float3(0,0,0), float3(0,1,1), float3(0,1,0),
    // right face
    float3(1,0,0), float3(1,1,0), float3(1,1,1), float3(1,0,0), float3(1,1,1), float3(1,0,1),
    // top face
    float3(0,1,0), float3(0,1,1), float3(1,1,1), float3(0,1,0), float3(1,1,1), float3(1,1,0),
    // bottom face
    float3(0,0,0), float3(1,0,0), float3(1,0,1), float3(0,0,0), float3(1,0,1), float3(0,0,1),
    // front face
    float3(0,0,1), float3(1,0,1), float3(1,1,1), float3(0,0,1), float3(1,1,1), float3(0,1,1),
    // back face
    float3(0,0,0), float3(0,1,0), float3(1,1,0), float3(0,0,0), float3(1,1,0), float3(1,0,0),
};

static const float2 CUBE_UV[36] = {
    // left face
    float2(0,0), float2(1,0), float2(1,1), float2(0,0), float2(1,1), float2(0,1),
    // right face
    float2(0,0), float2(0,1), float2(1,1), float2(0,0), float2(1,1), float2(1,0),
    // top face
    float2(0,0), float2(0,1), float2(1,1), float2(0,0), float2(1,1), float2(1,0),
    // bottom face
    float2(0,0), float2(0,1), float2(1,1), float2(0,0), float2(1,1), float2(1,0),
    // front face
    float2(0,0), float2(1,0), float2(1,1), float2(0,0), float2(1,1), float2(0,1),
    // back face
    float2(0,0), float2(0,1), float2(1,1), float2(0,0), float2(1,1), float2(1,0),
};

StructuredBuffer<TerrainHeader> terrainHeaders : register(t0, space0);
StructuredBuffer<Voxel> voxels : register(t1, space0);
StructuredBuffer<RenderParams> paramsBuffer : register(t2, space0);

[[vk::push_constant]] cbuffer PushConstants
{
    DrawPushConstants pushConstants;
};

VertexOutput voxelVertexMain(uint vertexId : SV_VertexID)
{
    const int voxelIndex = vertexId / N_VERTICES_PER_VOXEL;
    const int vertexInVoxelIndex = vertexId % N_VERTICES_PER_VOXEL;
    const int faceIndex = vertexInVoxelIndex / 6;

    RenderParams params = paramsBuffer[pushConstants.envIndex];
    Voxel voxel = voxels[voxelIndex];
    // x : 6
    // y : 9
    // z : 6
    // visibleFacesMask: 6
    // blockType: 5
    const uint blockType = voxel.data & ((1 << 5) - 1);
    const uint visibleFaces = (voxel.data >> 5) & ((1 << 6) - 1);
    const int localPosZ = (voxel.data >> 11) & ((1 << 6) - 1);
    const int localPosY = (voxel.data >> 17) & ((1 << 9) - 1);
    const int localPosX = (voxel.data >> 26) & ((1 << 6) - 1);
    const int posZ = localPosZ + terrainHeaders[pushConstants.envIndex].originZ;
    const int posX = localPosX + terrainHeaders[pushConstants.envIndex].originX;
    const int3 blockPos = int3(posX, localPosY, posZ);

    // use a degenerate triangle for invisible faces, so we don't have to discard anything in the pixel shader
    float multiplier = (visibleFaces & FACE_MASKS[faceIndex]) ? 1.0 : 0.0;

    float3 vertexOffset = CUBE_VERTICES[vertexInVoxelIndex];
    float3 worldPosition = multiplier * (blockPos + vertexOffset);
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
    output.color = float4(1.0, 0.0, 1.0, 1.0); // color is not used for voxels, but we set pink color for debugging purposes
    output.worldPosition = worldPosition;
    output.shade = FACE_SHADES[faceIndex];
    output.uvMaterial = float3(CUBE_UV[vertexInVoxelIndex], FACE_MATERIALS[blockType][faceIndex]);
    output.fog = saturate((viewZ - params.tuning.z) / max(params.tuning.w - params.tuning.z, 0.001));
    output.layer = pushConstants.layerIndex;
    return output;
}
