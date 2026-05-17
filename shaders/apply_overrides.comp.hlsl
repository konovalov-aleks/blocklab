#include "common.hlsl"

StructuredBuffer<int4> overrides : register(t0, space0);
RWStructuredBuffer<uint> blocks : register(u0, space1);

[numthreads(256, 1, 1)]
void applyOverridesMain(uint3 gid : SV_DispatchThreadID)
{
    if (gid.x >= uint(params.overrideInfo.x)) {
        return;
    }

    int4 edit = overrides[gid.x];
    int3 local = edit.xyz - params.worldOriginAndWidth.xyz;
    int3 regionSize = params.regionAndHeight.xyz;
    if (local.x < 0 || local.y < 0 || local.z < 0 || local.x >= regionSize.x || local.y >= regionSize.y
        || local.z >= regionSize.z) {
        return;
    }

    int index = local.x + local.z * regionSize.x + local.y * regionSize.x * regionSize.z;
    blocks[index] = uint(edit.w);
}

