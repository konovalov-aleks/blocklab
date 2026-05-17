#include "common.hlsl"

struct MeshVertex {
    float4 position;
    float4 uvAndShade;
};

StructuredBuffer<uint> blocks : register(t0, space0);
RWStructuredBuffer<MeshVertex> vertices : register(u0, space1);
RWStructuredBuffer<uint> drawArgs : register(u1, space1);

static const uint AtlasTileSizePixels = 16;
static const uint AtlasColumns = 4;
static const uint AtlasRows = 2;
static const float AtlasTileSize = float(AtlasTileSizePixels);
static const float2 AtlasSize = float2(float(AtlasTileSizePixels * AtlasColumns), float(AtlasTileSizePixels * AtlasRows));

uint blockAt(int3 p, int3 regionSize)
{
    if (p.x < 0 || p.y < 0 || p.z < 0 || p.x >= regionSize.x || p.y >= regionSize.y || p.z >= regionSize.z) {
        return BlockAir;
    }
    return blocks[p.x + p.z * regionSize.x + p.y * regionSize.x * regionSize.z];
}

uint tileForFace(uint block, int3 normal)
{
    switch (block) {
    case BlockGrass:
        if (normal.y > 0) {
            return TileGrassTop;
        }
        if (normal.y < 0) {
            return TileDirt;
        }
        return TileGrassSide;
    case BlockDirt:
        return TileDirt;
    case BlockStone:
        return TileStone;
    default:
        return TileDirt;
    }
}

float faceShade(int3 normal)
{
    if (normal.y > 0) {
        return 1.0;
    }
    if (normal.y < 0) {
        return 0.48;
    }
    if (normal.x != 0) {
        return 0.78;
    }
    return 0.68;
}

float2 atlasUv(uint tile, float2 uv)
{
    float2 tileOrigin = float2(float(tile % AtlasColumns), float(tile / AtlasColumns)) * AtlasTileSize;
    return (tileOrigin + uv * (AtlasTileSize - 1.0) + 0.5) / AtlasSize;
}

void emitVertex(uint index, float3 position, float2 uv, float shade)
{
    vertices[index].position = float4(position, 1.0);
    vertices[index].uvAndShade = float4(uv, shade, 0.0);
}

void faceUvs(int3 normal, out float2 uv0, out float2 uv1, out float2 uv2, out float2 uv3)
{
    if (normal.y > 0) {
        uv0 = float2(0.0, 0.0);
        uv1 = float2(0.0, 1.0);
        uv2 = float2(1.0, 1.0);
        uv3 = float2(1.0, 0.0);
        return;
    }
    if (normal.y < 0) {
        uv0 = float2(0.0, 0.0);
        uv1 = float2(1.0, 0.0);
        uv2 = float2(1.0, 1.0);
        uv3 = float2(0.0, 1.0);
        return;
    }

    uv0 = float2(0.0, 1.0);
    uv1 = normal.x < 0 || normal.z > 0 ? float2(1.0, 1.0) : float2(0.0, 0.0);
    uv2 = float2(1.0, 0.0);
    uv3 = normal.x < 0 || normal.z > 0 ? float2(0.0, 0.0) : float2(1.0, 1.0);
}

void emitFace(float3 p0, float3 p1, float3 p2, float3 p3, uint tile, float shade, int3 normal)
{
    uint baseIndex;
    InterlockedAdd(drawArgs[0], 6, baseIndex);
    float2 uv0;
    float2 uv1;
    float2 uv2;
    float2 uv3;
    faceUvs(normal, uv0, uv1, uv2, uv3);
    emitVertex(baseIndex + 0, p0, atlasUv(tile, uv0), shade);
    emitVertex(baseIndex + 1, p1, atlasUv(tile, uv1), shade);
    emitVertex(baseIndex + 2, p2, atlasUv(tile, uv2), shade);
    emitVertex(baseIndex + 3, p0, atlasUv(tile, uv0), shade);
    emitVertex(baseIndex + 4, p2, atlasUv(tile, uv2), shade);
    emitVertex(baseIndex + 5, p3, atlasUv(tile, uv3), shade);
}

void emitVisibleFace(
    uint block, int3 local, int3 normal, float3 p0, float3 p1, float3 p2, float3 p3, int3 regionSize)
{
    int3 neighbor = local + normal;
    if (blockAt(neighbor, regionSize) == BlockAir) {
        emitFace(p0, p1, p2, p3, tileForFace(block, normal), faceShade(normal), normal);
    }
}

[numthreads(256, 1, 1)]
void buildMeshMain(uint3 gid : SV_DispatchThreadID)
{
    int3 regionSize = params.regionAndHeight.xyz;
    uint volume = uint(regionSize.x * regionSize.y * regionSize.z);
    uint index = gid.x;
    if (index >= volume) {
        return;
    }

    int x = int(index % uint(regionSize.x));
    int z = int((index / uint(regionSize.x)) % uint(regionSize.z));
    int y = int(index / uint(regionSize.x * regionSize.z));
    int3 local = int3(x, y, z);
    uint block = blockAt(local, regionSize);
    if (block == BlockAir) {
        return;
    }

    float3 base = float3(params.worldOriginAndWidth.x + x, y, params.worldOriginAndWidth.z + z);

    float3 p000 = base + float3(0.0, 0.0, 0.0);
    float3 p100 = base + float3(1.0, 0.0, 0.0);
    float3 p010 = base + float3(0.0, 1.0, 0.0);
    float3 p110 = base + float3(1.0, 1.0, 0.0);
    float3 p001 = base + float3(0.0, 0.0, 1.0);
    float3 p101 = base + float3(1.0, 0.0, 1.0);
    float3 p011 = base + float3(0.0, 1.0, 1.0);
    float3 p111 = base + float3(1.0, 1.0, 1.0);

    emitVisibleFace(block, local, int3(0, 1, 0), p010, p011, p111, p110, regionSize);
    emitVisibleFace(block, local, int3(0, -1, 0), p000, p100, p101, p001, regionSize);
    emitVisibleFace(block, local, int3(1, 0, 0), p100, p110, p111, p101, regionSize);
    emitVisibleFace(block, local, int3(-1, 0, 0), p000, p001, p011, p010, regionSize);
    emitVisibleFace(block, local, int3(0, 0, 1), p001, p101, p111, p011, regionSize);
    emitVisibleFace(block, local, int3(0, 0, -1), p000, p010, p110, p100, regionSize);
}
