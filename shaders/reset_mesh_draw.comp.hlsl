RWStructuredBuffer<uint> drawArgs : register(u0, space1);

[numthreads(1, 1, 1)]
void resetMeshDrawMain(uint3 gid : SV_DispatchThreadID)
{
    drawArgs[0] = 0;
    drawArgs[1] = 1;
    drawArgs[2] = 0;
    drawArgs[3] = 0;
}

