#include "common.hlsl"

RWStructuredBuffer<uint> blocks : register(u0, space1);

uint mixBits(uint value)
{
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

float valueNoise(uint seed, int x, int z)
{
    uint hash = mixBits(seed ^ uint(x) * 0x9e3779b9u ^ uint(z) * 0x85ebca6bu);
    return float(hash & 0x00ffffffu) / float(0x00ffffffu) * 2.0 - 1.0;
}

uint generatedBlock(uint seed, int x, int y, int z, int worldHeight)
{
    if (y < 0) {
        return BlockStone;
    }
    if (y >= worldHeight) {
        return BlockAir;
    }

    float low = sin((float(x) + float(seed) * 0.013) * 0.17) * 2.2;
    float high = cos((float(z) - float(seed) * 0.019) * 0.13) * 1.8;
    float diagonal = sin(float(x + z) * 0.08 + float(seed) * 0.001) * 2.0;
    float rough = valueNoise(seed, x / 3, z / 3) * 0.55;
    int height = clamp(int(9.0 + low + high + diagonal + rough), 2, worldHeight - 2);
    if (y > height) {
        return BlockAir;
    }
    if (y == height) {
        return BlockGrass;
    }
    if (y > height - 4) {
        return BlockDirt;
    }
    return BlockStone;
}

[numthreads(256, 1, 1)]
void generateMain(uint3 gid : SV_DispatchThreadID)
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
    int worldX = params.worldOriginAndWidth.x + x;
    int worldZ = params.worldOriginAndWidth.z + z;
    uint seed = uint(params.overrideInfo.y);
    blocks[index] = generatedBlock(seed, worldX, y, worldZ, regionSize.y);
}
