#include "WorldTestUtils.h"

#include <world/World.h>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>

namespace blocklab::test {

namespace {

    constexpr IVec3 Center { 0, 0, 0 };
    constexpr IVec3 CaveOrigin { -4, 12, -4 };
    constexpr IVec3 CaveSize { 9, 7, 9 };

    void generateCaveWithEntrance(World& world)
    {
        generateCave(world, CaveOrigin, CaveSize);
        for (std::int32_t y = 13; y <= 15; ++y) {
            for (std::int32_t x = -1; x <= 1; ++x)
                world.setBlock({ x, y, -4 }, Block::Air);
        }
    }

    void closeCaveEntrance(World& world)
    {
        for (std::int32_t y = 13; y <= 15; ++y) {
            for (std::int32_t x = -1; x <= 1; ++x)
                world.setBlock({ x, y, -4 }, Block::Stone);
        }
    }

    void partiallyCloseCaveEntrance(World& world)
    {
        for (std::int32_t y = 13; y <= 15; ++y)
            world.setBlock({ 0, y, -4 }, Block::Stone);
    }

} // namespace

TEST_CASE("Generated terrain block light matches an explicit torch volume", "[world][lighting]")
{
    World world;
    world.resetSeed(23);
    updateWorldCacheAt(world, Center);

    const IVec3 torch { 0, 16, 0 };
    const IVec3 volumeOrigin { -3, 13, -3 };
    const IVec3 volumeSize { 7, 7, 7 };

    // Keep this volume deliberately small and fully explicit: the mask is the specification, not a second light solver.
    clearBlocks(world, { -16, 0, -16 }, { 33, 32, 33 });
    world.setBlock(torch, Block::Torch);

    const GeneratedVoxels generated = generateVoxelsAt(world, Center);
    checkBlockLight(generated, volumeOrigin, volumeSize,
        {
            // Y = 13, rows are Z = -3..3.
            "6789876\n"
            "789A987\n"
            "89ABA98\n"
            "9ABCBA9\n"
            "89ABA98\n"
            "789A987\n"
            "6789876",

            // Y = 14, rows are Z = -3..3.
            "789A987\n"
            "89ABA98\n"
            "9ABCBA9\n"
            "ABCDCBA\n"
            "9ABCBA9\n"
            "89ABA98\n"
            "789A987",

            // Y = 15, rows are Z = -3..3.
            "89ABA98\n"
            "9ABCBA9\n"
            "ABCDCBA\n"
            "BCDEDCB\n"
            "ABCDCBA\n"
            "9ABCBA9\n"
            "89ABA98",

            // Y = 16, rows are Z = -3..3.
            "9ABCBA9\n"
            "ABCDCBA\n"
            "BCDEDCB\n"
            "CDEFEDC\n"
            "BCDEDCB\n"
            "ABCDCBA\n"
            "9ABCBA9",

            // Y = 17, rows are Z = -3..3.
            "89ABA98\n"
            "9ABCBA9\n"
            "ABCDCBA\n"
            "BCDEDCB\n"
            "ABCDCBA\n"
            "9ABCBA9\n"
            "89ABA98",

            // Y = 18, rows are Z = -3..3.
            "789A987\n"
            "89ABA98\n"
            "9ABCBA9\n"
            "ABCDCBA\n"
            "9ABCBA9\n"
            "89ABA98\n"
            "789A987",

            // Y = 19, rows are Z = -3..3.
            "6789876\n"
            "789A987\n"
            "89ABA98\n"
            "9ABCBA9\n"
            "89ABA98\n"
            "789A987\n"
            "6789876",
        });

    // Explicitly check the lit-to-dark edge on different sides of the torch.
    checkBlockLight(generated, { 11, 14, 0 }, { 5, 5, 1 },
        {
            // Y = 14, row is Z = 0.
            "21000",
            // Y = 15, row is Z = 0.
            "32100",
            // Y = 16, row is Z = 0.
            "43210",
            // Y = 17, row is Z = 0.
            "32100",
            // Y = 18, row is Z = 0.
            "21000",
        });
    checkBlockLight(generated, { -2, 27, 0 }, { 5, 5, 1 },
        {
            // Y = 27, row is Z = 0.
            "23432",
            // Y = 28, row is Z = 0.
            "12321",
            // Y = 29, row is Z = 0.
            "01210",
            // Y = 30, row is Z = 0.
            "00100",
            // Y = 31, row is Z = 0.
            "00000",
        });
}

TEST_CASE("Generated terrain block light enters a cave from an outside roof torch", "[world][lighting]")
{
    World world;
    world.resetSeed(23);
    updateWorldCacheAt(world, Center);

    const IVec3 checkedOrigin { -3, 13, -4 };
    const IVec3 checkedSize { 7, 5, 6 };
    const IVec3 torch { 0, 19, -3 };

    // Build a cave with a front entrance. The torch sits outside on the roof near that entrance, so light must travel
    // around solid cave blocks instead of relying on direct empty-space propagation.
    generateCaveWithEntrance(world);

    {
        const GeneratedVoxels generated = generateVoxelsAt(world, Center);
        checkBlockLight(generated, checkedOrigin, checkedSize,
            {
                // Y = 13, rows are Z = -4..1.
                "SS000SS\n"
                "0000000\n"
                "0000000\n"
                "0000000\n"
                "0000000\n"
                "0000000",

                // Y = 14, rows are Z = -4..1.
                "SS000SS\n"
                "0000000\n"
                "0000000\n"
                "0000000\n"
                "0000000\n"
                "0000000",

                // Y = 15, rows are Z = -4..1.
                "SS000SS\n"
                "0000000\n"
                "0000000\n"
                "0000000\n"
                "0000000\n"
                "0000000",

                // Y = 16, rows are Z = -4..1.
                "SSSSSSS\n"
                "0000000\n"
                "0000000\n"
                "0000000\n"
                "0000000\n"
                "0000000",

                // Y = 17, rows are Z = -4..1.
                "SSSSSSS\n"
                "0000000\n"
                "0000000\n"
                "0000000\n"
                "0000000\n"
                "0000000",
            });
    }

    // The outside roof torch lights the cave through the entrance, while solid parts of the front wall remain unlit.
    world.setBlock(torch, Block::Torch);
    {
        const GeneratedVoxels generated = generateVoxelsAt(world, Center);
        checkBlockLight(generated, checkedOrigin, checkedSize,
            {
                // Y = 13, rows are Z = -4..1.
                "SS565SS\n"
                "2345432\n"
                "1234321\n"
                "0123210\n"
                "0012100\n"
                "0001000",

                // Y = 14, rows are Z = -4..1.
                "SS676SS\n"
                "3456543\n"
                "2345432\n"
                "1234321\n"
                "0123210\n"
                "0012100",

                // Y = 15, rows are Z = -4..1.
                "SS787SS\n"
                "4567654\n"
                "3456543\n"
                "2345432\n"
                "1234321\n"
                "0123210",

                // Y = 16, rows are Z = -4..1.
                "SSSSSSS\n"
                "3456543\n"
                "2345432\n"
                "1234321\n"
                "0123210\n"
                "0012100",

                // Y = 17, rows are Z = -4..1.
                "SSSSSSS\n"
                "2345432\n"
                "1234321\n"
                "0123210\n"
                "0012100\n"
                "0001000",
            });
    }

    // Partially closing the entrance keeps side passages open, but reduces the light that reaches the cave interior.
    partiallyCloseCaveEntrance(world);
    {
        const GeneratedVoxels generated = generateVoxelsAt(world, Center);
        checkBlockLight(generated, checkedOrigin, checkedSize,
            {
                // Y = 13, rows are Z = -4..1.
                "SS5S5SS\n"
                "2343432\n"
                "1232321\n"
                "0121210\n"
                "0010100\n"
                "0000000",

                // Y = 14, rows are Z = -4..1.
                "SS6S6SS\n"
                "3454543\n"
                "2343432\n"
                "1232321\n"
                "0121210\n"
                "0010100",

                // Y = 15, rows are Z = -4..1.
                "SS7S7SS\n"
                "4565654\n"
                "3454543\n"
                "2343432\n"
                "1232321\n"
                "0121210",

                // Y = 16, rows are Z = -4..1.
                "SSSSSSS\n"
                "3454543\n"
                "2343432\n"
                "1232321\n"
                "0121210\n"
                "0010100",

                // Y = 17, rows are Z = -4..1.
                "SSSSSSS\n"
                "2343432\n"
                "1232321\n"
                "0121210\n"
                "0010100\n"
                "0000000",
            });
    }

    // Closing the entrance blocks the only path into the cave.
    closeCaveEntrance(world);
    {
        const GeneratedVoxels generated = generateVoxelsAt(world, Center);
        checkBlockLight(generated, checkedOrigin, checkedSize,
            {
                // Y = 13, rows are Z = -4..1.
                "SSSSSSS\n"
                "0000000\n"
                "0000000\n"
                "0000000\n"
                "0000000\n"
                "0000000",

                // Y = 14, rows are Z = -4..1.
                "SSSSSSS\n"
                "0000000\n"
                "0000000\n"
                "0000000\n"
                "0000000\n"
                "0000000",

                // Y = 15, rows are Z = -4..1.
                "SSSSSSS\n"
                "0000000\n"
                "0000000\n"
                "0000000\n"
                "0000000\n"
                "0000000",

                // Y = 16, rows are Z = -4..1.
                "SSSSSSS\n"
                "0000000\n"
                "0000000\n"
                "0000000\n"
                "0000000\n"
                "0000000",

                // Y = 17, rows are Z = -4..1.
                "SSSSSSS\n"
                "0000000\n"
                "0000000\n"
                "0000000\n"
                "0000000\n"
                "0000000",
            });
    }
}

TEST_CASE("Generated terrain block light exits a cave from an inside torch", "[world][lighting]")
{
    World world;
    world.resetSeed(23);
    updateWorldCacheAt(world, Center);

    const IVec3 checkedOrigin { -3, 13, -8 };
    const IVec3 checkedSize { 7, 5, 6 };
    const IVec3 torch { 0, 13, -2 };

    generateCaveWithEntrance(world);
    world.setBlock(torch, Block::Torch);

    {
        const GeneratedVoxels generated = generateVoxelsAt(world, Center);
        checkBlockLight(generated, checkedOrigin, checkedSize,
            {
                // Y = 13, rows are Z = -8..-3.
                "6789876\n"
                "789A987\n"
                "89ABA98\n"
                "9ABCBA9\n"
                "SSCDCSS\n"
                "BCDEDCB",

                // Y = 14, rows are Z = -8..-3.
                "5678765\n"
                "6789876\n"
                "789A987\n"
                "89ABA98\n"
                "SSBCBSS\n"
                "ABCDCBA",

                // Y = 15, rows are Z = -8..-3.
                "4567654\n"
                "5678765\n"
                "6789876\n"
                "789A987\n"
                "SSABASS\n"
                "9ABCBA9",

                // Y = 16, rows are Z = -8..-3.
                "3456543\n"
                "4567654\n"
                "5678765\n"
                "6789876\n"
                "SSSSSSS\n"
                "89ABA98",

                // Y = 17, rows are Z = -8..-3.
                "2345432\n"
                "3456543\n"
                "4567654\n"
                "5678765\n"
                "SSSSSSS\n"
                "789A987",
            });
    }

    // Partially closing the entrance still lets light escape through the side openings, but the outside light drops.
    partiallyCloseCaveEntrance(world);
    {
        const GeneratedVoxels generated = generateVoxelsAt(world, Center);
        checkBlockLight(generated, checkedOrigin, checkedSize,
            {
                // Y = 13, rows are Z = -8..-3.
                "6787876\n"
                "7898987\n"
                "89A9A98\n"
                "9ABABA9\n"
                "SSCSCSS\n"
                "BCDEDCB",

                // Y = 14, rows are Z = -8..-3.
                "5676765\n"
                "6787876\n"
                "7898987\n"
                "89A9A98\n"
                "SSBSBSS\n"
                "ABCDCBA",

                // Y = 15, rows are Z = -8..-3.
                "4565654\n"
                "5676765\n"
                "6787876\n"
                "7898987\n"
                "SSASASS\n"
                "9ABCBA9",

                // Y = 16, rows are Z = -8..-3.
                "3454543\n"
                "4565654\n"
                "5676765\n"
                "6787876\n"
                "SSSSSSS\n"
                "89ABA98",

                // Y = 17, rows are Z = -8..-3.
                "2343432\n"
                "3454543\n"
                "4565654\n"
                "5676765\n"
                "SSSSSSS\n"
                "789A987",
            });
    }

    // Closing the entrance keeps the cave interior lit but stops light from escaping outside.
    closeCaveEntrance(world);
    {
        const GeneratedVoxels generated = generateVoxelsAt(world, Center);
        checkBlockLight(generated, checkedOrigin, checkedSize,
            {
                // Y = 13, rows are Z = -8..-3.
                "0000000\n"
                "0000000\n"
                "0000000\n"
                "0000000\n"
                "SSSSSSS\n"
                "BCDEDCB",

                // Y = 14, rows are Z = -8..-3.
                "0000000\n"
                "0000000\n"
                "0000000\n"
                "0000000\n"
                "SSSSSSS\n"
                "ABCDCBA",

                // Y = 15, rows are Z = -8..-3.
                "0000000\n"
                "0000000\n"
                "0000000\n"
                "0000000\n"
                "SSSSSSS\n"
                "9ABCBA9",

                // Y = 16, rows are Z = -8..-3.
                "0000000\n"
                "0000000\n"
                "0000000\n"
                "0000000\n"
                "SSSSSSS\n"
                "89ABA98",

                // Y = 17, rows are Z = -8..-3.
                "0000000\n"
                "0000000\n"
                "0000000\n"
                "0000000\n"
                "SSSSSSS\n"
                "789A987",
            });
    }
}

TEST_CASE("Generated terrain lighting follows torch state transitions", "[world][lighting]")
{
    World world;
    world.resetSeed(23);
    updateWorldCacheAt(world, Center);

    const IVec3 leftTorch { 0, 16, 0 };
    const IVec3 leftTorchSupport { 0, 15, 0 };
    const IVec3 rightTorch { 4, 16, 0 };
    const IVec3 rightTorchSupport { 4, 15, 0 };
    const IVec3 sliceOrigin { -2, 12, -1 };
    const IVec3 sliceSize { 9, 9, 3 };

    // The full-volume test above verifies 3D falloff. Here we only check the central Z slice that changes across the
    // state transitions.
    clearBlocks(world, { -4, 14, -2 }, { 13, 5, 5 });
    world.setBlock(leftTorchSupport, Block::Stone);
    world.setBlock(rightTorchSupport, Block::Stone);

    // No light sources: the checked slice is dark.
    {
        const GeneratedVoxels generated = generateVoxelsAt(world, Center);
        checkBlockLight(generated, sliceOrigin, sliceSize,
            {
                // Y = 12, rows are Z = -1..1.
                "000000000\n"
                "000000000\n"
                "000000000",
                // Y = 13, rows are Z = -1..1.
                "000000000\n"
                "000000000\n"
                "000000000",
                // Y = 14, rows are Z = -1..1.
                "000000000\n"
                "000000000\n"
                "000000000",
                // Y = 15, rows are Z = -1..1.
                "000000000\n"
                "00S000S00\n"
                "000000000",
                // Y = 16, rows are Z = -1..1.
                "000000000\n"
                "000000000\n"
                "000000000",
                // Y = 17, rows are Z = -1..1.
                "000000000\n"
                "000000000\n"
                "000000000",
                // Y = 18, rows are Z = -1..1.
                "000000000\n"
                "000000000\n"
                "000000000",
                // Y = 19, rows are Z = -1..1.
                "000000000\n"
                "000000000\n"
                "000000000",
                // Y = 20, rows are Z = -1..1.
                "000000000\n"
                "000000000\n"
                "000000000",
            });
    }

    // A single torch lights the central slice from left to right.
    world.setBlock(leftTorch, Block::Torch);
    {
        const GeneratedVoxels generated = generateVoxelsAt(world, Center);
        checkBlockLight(generated, sliceOrigin, sliceSize,
            {
                // Y = 12, rows are Z = -1..1.
                "89A987654\n"
                "9A9A98765\n"
                "89A987654",
                // Y = 13, rows are Z = -1..1.
                "9ABA98765\n"
                "ABABA9876\n"
                "9ABA98765",
                // Y = 14, rows are Z = -1..1.
                "ABCBA9876\n"
                "BCBCBA987\n"
                "ABCBA9876",
                // Y = 15, rows are Z = -1..1.
                "BCDCBA987\n"
                "CDSDCBS98\n"
                "BCDCBA987",
                // Y = 16, rows are Z = -1..1.
                "CDEDCBA98\n"
                "DEFEDCBA9\n"
                "CDEDCBA98",
                // Y = 17, rows are Z = -1..1.
                "BCDCBA987\n"
                "CDEDCBA98\n"
                "BCDCBA987",
                // Y = 18, rows are Z = -1..1.
                "ABCBA9876\n"
                "BCDCBA987\n"
                "ABCBA9876",
                // Y = 19, rows are Z = -1..1.
                "9ABA98765\n"
                "ABCBA9876\n"
                "9ABA98765",
                // Y = 20, rows are Z = -1..1.
                "89A987654\n"
                "9ABA98765\n"
                "89A987654",
            });
    }

    // A second torch combines with the first one by taking the maximum light for each air block in the slice.
    world.setBlock(rightTorch, Block::Torch);
    {
        const GeneratedVoxels generated = generateVoxelsAt(world, Center);
        checkBlockLight(generated, sliceOrigin, sliceSize,
            {
                // Y = 12, rows are Z = -1..1.
                "89A989A98\n"
                "9A9A9A9A9\n"
                "89A989A98",
                // Y = 13, rows are Z = -1..1.
                "9ABA9ABA9\n"
                "ABABABABA\n"
                "9ABA9ABA9",
                // Y = 14, rows are Z = -1..1.
                "ABCBABCBA\n"
                "BCBCBCBCB\n"
                "ABCBABCBA",
                // Y = 15, rows are Z = -1..1.
                "BCDCBCDCB\n"
                "CDSDCDSDC\n"
                "BCDCBCDCB",
                // Y = 16, rows are Z = -1..1.
                "CDEDCDEDC\n"
                "DEFEDEFED\n"
                "CDEDCDEDC",
                // Y = 17, rows are Z = -1..1.
                "BCDCBCDCB\n"
                "CDEDCDEDC\n"
                "BCDCBCDCB",
                // Y = 18, rows are Z = -1..1.
                "ABCBABCBA\n"
                "BCDCBCDCB\n"
                "ABCBABCBA",
                // Y = 19, rows are Z = -1..1.
                "9ABA9ABA9\n"
                "ABCBABCBA\n"
                "9ABA9ABA9",
                // Y = 20, rows are Z = -1..1.
                "89A989A98\n"
                "9ABA9ABA9\n"
                "89A989A98",
            });
    }

    // Removing one torch weakens only the air blocks that depended on it in the slice.
    world.setBlock(rightTorch, Block::Air);
    {
        const GeneratedVoxels generated = generateVoxelsAt(world, Center);
        checkBlockLight(generated, sliceOrigin, sliceSize,
            {
                // Y = 12, rows are Z = -1..1.
                "89A987654\n"
                "9A9A98765\n"
                "89A987654",
                // Y = 13, rows are Z = -1..1.
                "9ABA98765\n"
                "ABABA9876\n"
                "9ABA98765",
                // Y = 14, rows are Z = -1..1.
                "ABCBA9876\n"
                "BCBCBA987\n"
                "ABCBA9876",
                // Y = 15, rows are Z = -1..1.
                "BCDCBA987\n"
                "CDSDCBS98\n"
                "BCDCBA987",
                // Y = 16, rows are Z = -1..1.
                "CDEDCBA98\n"
                "DEFEDCBA9\n"
                "CDEDCBA98",
                // Y = 17, rows are Z = -1..1.
                "BCDCBA987\n"
                "CDEDCBA98\n"
                "BCDCBA987",
                // Y = 18, rows are Z = -1..1.
                "ABCBA9876\n"
                "BCDCBA987\n"
                "ABCBA9876",
                // Y = 19, rows are Z = -1..1.
                "9ABA98765\n"
                "ABCBA9876\n"
                "9ABA98765",
                // Y = 20, rows are Z = -1..1.
                "89A987654\n"
                "9ABA98765\n"
                "89A987654",
            });
    }

    // Destroying the support block removes the torch above and its remaining light contribution.
    world.setBlock(leftTorchSupport, Block::Air);
    CHECK(world.blockType(leftTorch) == Block::Air);
    {
        const GeneratedVoxels generated = generateVoxelsAt(world, Center);
        checkBlockLight(generated, sliceOrigin, sliceSize,
            {
                // Y = 12, rows are Z = -1..1.
                "000000000\n"
                "000000000\n"
                "000000000",
                // Y = 13, rows are Z = -1..1.
                "000000000\n"
                "000000000\n"
                "000000000",
                // Y = 14, rows are Z = -1..1.
                "000000000\n"
                "000000000\n"
                "000000000",
                // Y = 15, rows are Z = -1..1.
                "000000000\n"
                "000000S00\n"
                "000000000",
                // Y = 16, rows are Z = -1..1.
                "000000000\n"
                "000000000\n"
                "000000000",
                // Y = 17, rows are Z = -1..1.
                "000000000\n"
                "000000000\n"
                "000000000",
                // Y = 18, rows are Z = -1..1.
                "000000000\n"
                "000000000\n"
                "000000000",
                // Y = 19, rows are Z = -1..1.
                "000000000\n"
                "000000000\n"
                "000000000",
                // Y = 20, rows are Z = -1..1.
                "000000000\n"
                "000000000\n"
                "000000000",
            });
    }
}

} // namespace blocklab::test
