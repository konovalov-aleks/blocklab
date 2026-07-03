#include "lighting.hlsl"
#include "material.hlsl"
#include "render_params.hlsl"
#include "vertex_output.hlsl"

struct TerrainHeader {
    int originX;
    int originY;
    int originZ;
};

struct Voxel {
    // x : 6
    // y : 9
    // z : 6
    // visibleFacesMask: 6
    // blockType: 5
    uint data;

    uint blockLight;
    uint skyLight;
};

static const int N_VERTICES_PER_VOXEL = 36;
static const uint BlockTorch = 4;

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
    // 4 - Torch
    { MaterialTorchSide, MaterialTorchSide, MaterialTorchTop, MaterialTorchSide, MaterialTorchSide, MaterialTorchSide },
};

static const float3 FACE_NORMALS[6] = {
    float3(-1.0,  0.0,  0.0), // left
    float3( 1.0,  0.0,  0.0), // right
    float3( 0.0,  1.0,  0.0), // top
    float3( 0.0, -1.0,  0.0), // bottom
    float3( 0.0,  0.0,  1.0), // front
    float3( 0.0,  0.0, -1.0), // back
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

float3 blockVertex(uint blockType, uint vertexInVoxelIndex)
{
    float3 vertex = CUBE_VERTICES[vertexInVoxelIndex];
    if (blockType == BlockTorch)
        return float3(0.4375 + vertex.x * 0.125, vertex.y * 0.65, 0.4375 + vertex.z * 0.125);
    return vertex;
}

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
    const int posY = localPosY + terrainHeaders[pushConstants.envIndex].originY;
    const int posX = localPosX + terrainHeaders[pushConstants.envIndex].originX;
    const int3 blockPos = int3(posX, posY, posZ);

    // unpack light
    const uint blockLight = (voxel.blockLight >> (faceIndex * 4)) & 0xF;
    const uint skyLight = (voxel.skyLight >> (faceIndex * 4)) & 0xF;

    // use a degenerate triangle for invisible faces, so we don't have to discard anything in the pixel shader
    float multiplier = (visibleFaces & FACE_MASKS[faceIndex]) ? 1.0 : 0.0;

    float3 vertexOffset = blockVertex(blockType, vertexInVoxelIndex);
    float3 worldPosition = multiplier * (blockPos + vertexOffset);
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

    float faceSkyLightFactor = faceSkyLight(FACE_NORMALS[faceIndex], params);

    VertexOutput output;
    output.position.x = viewX / (aspect * tanHalfFov);
    output.position.y = -viewY / tanHalfFov;
    output.position.z = viewZ * farPlane / (farPlane - nearPlane) - farPlane * nearPlane / (farPlane - nearPlane);
    output.position.w = viewZ;
    output.color = float4(1.0, 0.0, 1.0, 1.0); // color is not used for voxels, but we set pink color for debugging purposes
    output.worldPosition = worldPosition;
    uint currentSkyLight = skyLight - min(skyLight, params.skyInfo.skyLightDimming);
    output.light = max(float(currentSkyLight) * faceSkyLightFactor, float(blockLight)) / 15.0;
    output.uvMaterial = float3(CUBE_UV[vertexInVoxelIndex], FACE_MATERIALS[blockType][faceIndex]);
    output.fog = float4(params.skyInfo.skyColor, fogIntensity);
    output.layer = pushConstants.layerIndex;
    return output;
}
