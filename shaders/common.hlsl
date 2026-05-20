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

enum BlockId {
    BlockAir = 0,
    BlockGrass = 1,
    BlockDirt = 2,
    BlockStone = 3,
};

enum AtlasTile {
    TileGrassTop = 0,
    TileDirt = 1,
    TileStone = 2,
    TileGrassSide = 3,
    TilePigSkin = 4,
    TileWood = 5,
    TileLeaves = 6,
    TilePigFace = 7,
};

cbuffer Params : register(b0, space2)
{
    RenderParams params;
};
