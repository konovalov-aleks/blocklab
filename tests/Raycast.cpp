#include <algorithms/Raycast.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <optional>
#include <vector>

namespace blocklab::test {

namespace {

    struct RaycastStep {
        IVec3 pos;
        std::optional<IVec3> normal;
        float distance;
    };

    std::vector<RaycastStep> collectSteps(Vec3 origin, Vec3 direction, std::size_t stepCount)
    {
        std::vector<RaycastStep> steps;
        steps.reserve(stepCount);
        raycast(origin, direction, [&](const RaycastCbParams& p) {
            steps.push_back({
                .pos = p.pos,
                .normal = p.normal,
                .distance = p.distance,
            });
            return steps.size() == stepCount ? RaycastCommand::Break : RaycastCommand::Continue;
        });
        return steps;
    }

    void checkVec(IVec3 actual, IVec3 expected)
    {
        CHECK(actual.x == expected.x);
        CHECK(actual.y == expected.y);
        CHECK(actual.z == expected.z);
    }

    void checkNormal(const std::optional<IVec3>& actual, IVec3 expected)
    {
        REQUIRE(actual.has_value());
        checkVec(*actual, expected);
    }

    void checkAxisRay(Vec3 origin, Vec3 direction, IVec3 expectedOrigin, IVec3 step, IVec3 expectedNormal, float firstDistance)
    {
        const std::vector<RaycastStep> steps = collectSteps(origin, direction, 4);

        REQUIRE(steps.size() == 4);
        checkVec(steps[0].pos, expectedOrigin);
        CHECK(!steps[0].normal);
        CHECK(steps[0].distance == 0.0f);

        for (std::size_t i = 1; i < steps.size(); ++i) {
            checkVec(steps[i].pos, expectedOrigin + static_cast<int>(i) * step);
            checkNormal(steps[i].normal, expectedNormal);
            CHECK(steps[i].distance == Catch::Approx(firstDistance + static_cast<float>(i - 1)));
        }
    }

} // namespace

TEST_CASE("Raycast visits the origin cell first", "[raycast]")
{
    const std::vector<RaycastStep> steps =
        collectSteps({ 1.25f, 2.5f, -1.5f }, { 1.0f, 0.0f, 0.0f }, 1);

    REQUIRE(steps.size() == 1);
    checkVec(steps[0].pos, { 1, 2, -2 });
    CHECK(!steps[0].normal);
    CHECK(steps[0].distance == 0.0f);
}

TEST_CASE("Raycast advances along cardinal axes", "[raycast]")
{
    SECTION("+X")
    {
        checkAxisRay({ 0.25f, 2.5f, -1.5f }, { 1.0f, 0.0f, 0.0f },
            { 0, 2, -2 }, { 1, 0, 0 }, { -1, 0, 0 }, 0.75f);
    }

    SECTION("-X")
    {
        checkAxisRay({ 0.25f, 2.5f, -1.5f }, { -1.0f, 0.0f, 0.0f },
            { 0, 2, -2 }, { -1, 0, 0 }, { 1, 0, 0 }, 0.25f);
    }

    SECTION("+Y")
    {
        checkAxisRay({ 1.2f, 5.8f, 0.0f }, { 0.0f, 1.0f, 0.0f },
            { 1, 5, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, 0.2f);
    }

    SECTION("-Y")
    {
        checkAxisRay({ 1.2f, 5.8f, 0.0f }, { 0.0f, -1.0f, 0.0f },
            { 1, 5, 0 }, { 0, -1, 0 }, { 0, 1, 0 }, 0.8f);
    }

    SECTION("+Z")
    {
        checkAxisRay({ -2.0f, 3.25f, 7.4f }, { 0.0f, 0.0f, 1.0f },
            { -2, 3, 7 }, { 0, 0, 1 }, { 0, 0, -1 }, 0.6f);
    }

    SECTION("-Z")
    {
        checkAxisRay({ -2.0f, 3.25f, 7.4f }, { 0.0f, 0.0f, -1.0f },
            { -2, 3, 7 }, { 0, 0, -1 }, { 0, 0, 1 }, 0.4f);
    }
}

TEST_CASE("Raycast chooses the nearest voxel boundary on diagonal rays", "[raycast]")
{
    const float invLength = 1.0f / std::sqrt(5.0f);
    const Vec3 direction = { invLength, 2.0f * invLength, 0.0f };
    const std::vector<RaycastStep> steps = collectSteps({ 0.5f, 0.5f, 0.5f }, direction, 4);

    REQUIRE(steps.size() == 4);
    checkVec(steps[0].pos, { 0, 0, 0 });
    CHECK(!steps[0].normal);
    CHECK(steps[0].distance == 0.0f);

    checkVec(steps[1].pos, { 0, 1, 0 });
    checkNormal(steps[1].normal, { 0, -1, 0 });
    CHECK(steps[1].distance == Catch::Approx(0.25 * std::sqrt(5.0)));

    checkVec(steps[2].pos, { 1, 1, 0 });
    checkNormal(steps[2].normal, { -1, 0, 0 });
    CHECK(steps[2].distance == Catch::Approx(0.5 * std::sqrt(5.0)));

    checkVec(steps[3].pos, { 1, 2, 0 });
    checkNormal(steps[3].normal, { 0, -1, 0 });
    CHECK(steps[3].distance == Catch::Approx(0.75 * std::sqrt(5.0)));
}

TEST_CASE("Raycast distance advances until the callback stops a miss", "[raycast]")
{
    constexpr float MaxDistance = 4.5f;
    std::size_t visited = 0;
    bool stoppedByDistance = false;
    bool hitStepGuard = false;

    raycast({ 0.25f, 0.25f, 0.25f }, { 1.0f, 0.0f, 0.0f }, [&](const RaycastCbParams& p) {
        ++visited;
        if (p.distance > MaxDistance) {
            stoppedByDistance = true;
            return RaycastCommand::Break;
        }
        if (visited > 100) {
            hitStepGuard = true;
            return RaycastCommand::Break;
        }
        return RaycastCommand::Continue;
    });

    CHECK(stoppedByDistance);
    CHECK(!hitStepGuard);
    CHECK(visited < 100);
}

TEST_CASE("Raycast handles rays starting exactly on voxel boundaries", "[raycast]")
{
    SECTION("Ray starts on integer coordinates")
    {
        const std::vector<RaycastStep> steps = collectSteps({ 1.0f, 2.0f, 3.0f }, { 1.0f, 0.0f, 0.0f }, 3);

        REQUIRE(steps.size() == 3);
        // When starting exactly on a boundary, we should visit the voxel at the starting position first
        checkVec(steps[0].pos, { 1, 2, 3 });
        CHECK(!steps[0].normal);
        CHECK(steps[0].distance == 0.0f);

        // Then move to the next voxel
        checkVec(steps[1].pos, { 2, 2, 3 });
        checkNormal(steps[1].normal, { -1, 0, 0 });
        CHECK(steps[1].distance == Catch::Approx(1.0f));

        // And the next
        checkVec(steps[2].pos, { 3, 2, 3 });
        checkNormal(steps[2].normal, { -1, 0, 0 });
        CHECK(steps[2].distance == Catch::Approx(2.0f));
    }

    SECTION("Ray starts on integer X boundary")
    {
        const std::vector<RaycastStep> steps = collectSteps({ 2.0f, 1.5f, 3.5f }, { 0.0f, 1.0f, 0.0f }, 3);

        REQUIRE(steps.size() == 3);
        checkVec(steps[0].pos, { 2, 1, 3 });
        CHECK(!steps[0].normal);
        CHECK(steps[0].distance == 0.0f);

        checkVec(steps[1].pos, { 2, 2, 3 });
        checkNormal(steps[1].normal, { 0, -1, 0 });
        CHECK(steps[1].distance == Catch::Approx(0.5f));

        checkVec(steps[2].pos, { 2, 3, 3 });
        checkNormal(steps[2].normal, { 0, -1, 0 });
        CHECK(steps[2].distance == Catch::Approx(1.5f));
    }

    SECTION("Ray starts on integer Y boundary")
    {
        const std::vector<RaycastStep> steps = collectSteps({ 1.5f, 4.0f, 2.5f }, { 0.0f, 0.0f, 1.0f }, 3);

        REQUIRE(steps.size() == 3);
        checkVec(steps[0].pos, { 1, 4, 2 });
        CHECK(!steps[0].normal);
        CHECK(steps[0].distance == 0.0f);

        checkVec(steps[1].pos, { 1, 4, 3 });
        checkNormal(steps[1].normal, { 0, 0, -1 });
        CHECK(steps[1].distance == Catch::Approx(0.5f));

        checkVec(steps[2].pos, { 1, 4, 4 });
        checkNormal(steps[2].normal, { 0, 0, -1 });
        CHECK(steps[2].distance == Catch::Approx(1.5f));
    }

    SECTION("Ray starts on integer Z boundary")
    {
        const std::vector<RaycastStep> steps = collectSteps({ 3.5f, 2.5f, 5.0f }, { 1.0f, 0.0f, 0.0f }, 3);

        REQUIRE(steps.size() == 3);
        checkVec(steps[0].pos, { 3, 2, 5 });
        CHECK(!steps[0].normal);
        CHECK(steps[0].distance == 0.0f);

        checkVec(steps[1].pos, { 4, 2, 5 });
        checkNormal(steps[1].normal, { -1, 0, 0 });
        CHECK(steps[1].distance == Catch::Approx(0.5f));

        checkVec(steps[2].pos, { 5, 2, 5 });
        checkNormal(steps[2].normal, { -1, 0, 0 });
        CHECK(steps[2].distance == Catch::Approx(1.5f));
    }
}

} // namespace blocklab::test
