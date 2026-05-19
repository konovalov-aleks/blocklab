#include "blocklab/QuadTree.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

uint64_t pointKey(int32_t x, int32_t z)
{
    return (static_cast<uint64_t>(static_cast<uint32_t>(x)) << 32) | static_cast<uint32_t>(z);
}

int32_t keyX(uint64_t key) { return static_cast<int32_t>(static_cast<uint32_t>(key >> 32)); }

int32_t keyZ(uint64_t key) { return static_cast<int32_t>(static_cast<uint32_t>(key)); }

} // namespace

TEST_CASE("QuadTree finds values by point", "[quad-tree]")
{
    blocklab::QuadTree<int> tree;
    tree.insert(0, 0, 1);
    tree.insert(5, 5, 2);
    tree.insert(-4, 3, 3);
    tree.insert(16, -8, 4);

    REQUIRE(tree.find(0, 0) != tree.end());
    REQUIRE(tree.find(5, 5) != tree.end());
    REQUIRE(tree.find(-4, 3) != tree.end());
    REQUIRE(tree.find(16, -8) != tree.end());
    CHECK(*tree.find(0, 0) == 1);
    CHECK(*tree.find(5, 5) == 2);
    CHECK(*tree.find(-4, 3) == 3);
    CHECK(*tree.find(16, -8) == 4);
    CHECK(tree.find(1, 1) == tree.end());
}

TEST_CASE("QuadTree supports const point lookup", "[quad-tree]")
{
    blocklab::QuadTree<int> tree;
    tree.insert(5, 5, 2);

    const blocklab::QuadTree<int>& constTree = tree;
    REQUIRE(constTree.find(5, 5) != constTree.end());
    CHECK(*constTree.find(5, 5) == 2);
    CHECK(constTree.find(6, 5) == constTree.end());
}

TEST_CASE("QuadTree updates an existing point without growing", "[quad-tree]")
{
    blocklab::QuadTree<int> tree;
    tree.insert(5, 5, 2);
    tree.insert(5, 5, 20);

    CHECK(tree.size() == 1);
    REQUIRE(tree.find(5, 5) != tree.end());
    CHECK(*tree.find(5, 5) == 20);
}

TEST_CASE("QuadTree clears all values", "[quad-tree]")
{
    blocklab::QuadTree<int> tree;
    tree.insert(0, 0, 1);
    tree.insert(5, 5, 2);
    tree.insert(-4, 3, 3);

    tree.clear();
    CHECK(tree.empty());
    CHECK(tree.size() == 0);
    CHECK(tree.begin() == tree.end());
    CHECK(tree.cbegin() == tree.cend());
    CHECK(tree.find(0, 0) == tree.end());
    CHECK(tree.find(5, 5) == tree.end());
    CHECK(tree.find(-4, 3) == tree.end());

    tree.insert(12, -7, 4);
    CHECK(tree.size() == 1);
    REQUIRE(tree.find(12, -7) != tree.end());
    CHECK(*tree.find(12, -7) == 4);
}

TEST_CASE("QuadTree erases matching values inside a rectangle", "[quad-tree]")
{
    blocklab::QuadTree<int> tree;
    tree.insert(0, 0, 1);
    tree.insert(5, 5, 2);
    tree.insert(-4, 3, 3);
    tree.insert(16, -8, 4);

    const std::size_t erased = tree.eraseIf(
        { .x = -10, .z = -10, .width = 20, .height = 20 }, [](int32_t, int32_t, int value) { return value % 2 == 1; });
    CHECK(erased == 2);
    CHECK(tree.size() == 2);
    CHECK(tree.find(0, 0) == tree.end());
    CHECK(tree.find(-4, 3) == tree.end());
    REQUIRE(tree.find(5, 5) != tree.end());
    REQUIRE(tree.find(16, -8) != tree.end());
    CHECK(*tree.find(5, 5) == 2);
    CHECK(*tree.find(16, -8) == 4);
}

TEST_CASE("QuadTree erases a point by iterator", "[quad-tree]")
{
    blocklab::QuadTree<int> tree;
    tree.insert(0, 0, 1);
    tree.insert(5, 5, 2);

    auto it = tree.find(0, 0);
    auto expectedNext = it;
    ++expectedNext;
    auto next = tree.erase(it);
    CHECK(tree.size() == 1);
    CHECK(tree.find(0, 0) == tree.end());
    REQUIRE(tree.find(5, 5) != tree.end());
    CHECK(*tree.find(5, 5) == 2);
    CHECK(next == expectedNext);
}

TEST_CASE("QuadTree erases all values inside a rectangle", "[quad-tree]")
{
    blocklab::QuadTree<int> tree;
    int value = 0;
    for (int32_t z = -3; z <= 3; ++z) {
        for (int32_t x = -3; x <= 3; ++x)
            tree.insert(x, z, ++value);
    }

    const std::size_t erased = tree.erase({ .x = -1, .z = -1, .width = 3, .height = 3 });
    CHECK(erased == 9);
    CHECK(tree.size() == 40);

    for (int32_t z = -3; z <= 3; ++z) {
        for (int32_t x = -3; x <= 3; ++x) {
            const bool inside = x >= -1 && x <= 1 && z >= -1 && z <= 1;
            if (inside) {
                CHECK(tree.find(x, z) == tree.end());
                continue;
            }
            CHECK(tree.find(x, z) != tree.end());
        }
    }
}

TEST_CASE("QuadTree iterator returned by find advances to following values", "[quad-tree]")
{
    blocklab::QuadTree<int> tree;
    tree.insert(-16, -16, 1);
    tree.insert(16, -16, 2);
    tree.insert(-16, 16, 3);
    tree.insert(16, 16, 4);
    tree.insert(64, 64, 5);

    std::vector<uint64_t> orderedKeys;
    for (auto it = tree.begin(); it != tree.end(); ++it)
        orderedKeys.push_back(pointKey(it.x(), it.z()));
    REQUIRE(orderedKeys.size() >= 4);

    const uint64_t startKey = orderedKeys[1];
    const uint64_t stopKey = orderedKeys[orderedKeys.size() - 1];
    auto it = tree.find(keyX(startKey), keyZ(startKey));
    const auto stop = tree.find(keyX(stopKey), keyZ(stopKey));

    std::vector<uint64_t> suffixKeys;
    for (; it != stop; ++it)
        suffixKeys.push_back(pointKey(it.x(), it.z()));

    REQUIRE(suffixKeys.size() == orderedKeys.size() - 2);
    for (std::size_t i = 0; i < suffixKeys.size(); ++i)
        CHECK(suffixKeys[i] == orderedKeys[i + 1]);
}

TEST_CASE("QuadTree iterator returned by find can advance to end", "[quad-tree]")
{
    blocklab::QuadTree<int> tree;
    tree.insert(-16, -16, 1);
    tree.insert(16, -16, 2);
    tree.insert(-16, 16, 3);
    tree.insert(16, 16, 4);

    std::vector<uint64_t> orderedKeys;
    for (auto it = tree.cbegin(); it != tree.cend(); ++it)
        orderedKeys.push_back(pointKey(it.x(), it.z()));
    REQUIRE(!orderedKeys.empty());

    auto it = tree.find(keyX(orderedKeys.front()), keyZ(orderedKeys.front()));
    std::size_t visited = 0;
    for (; it != tree.end(); ++it)
        ++visited;
    CHECK(visited == orderedKeys.size());
}

TEST_CASE("QuadTree supports copy construction and assignment", "[quad-tree]")
{
    blocklab::QuadTree<int> tree;
    tree.insert(-2, 4, 10);
    tree.insert(8, -6, 20);

    blocklab::QuadTree<int> copied(tree);
    tree.insert(-2, 4, 30);

    CHECK(copied.size() == 2);
    REQUIRE(copied.find(-2, 4) != copied.end());
    REQUIRE(copied.find(8, -6) != copied.end());
    CHECK(*copied.find(-2, 4) == 10);
    CHECK(*copied.find(8, -6) == 20);

    blocklab::QuadTree<int> assigned;
    assigned = copied;
    copied.clear();

    CHECK(assigned.size() == 2);
    REQUIRE(assigned.find(-2, 4) != assigned.end());
    REQUIRE(assigned.find(8, -6) != assigned.end());
    CHECK(*assigned.find(-2, 4) == 10);
    CHECK(*assigned.find(8, -6) == 20);
}

TEST_CASE("QuadTree stores and removes a large random point set", "[quad-tree]")
{
    constexpr std::size_t pointCount = 4096;
    std::mt19937 rng(0xB10C1AB);
    std::uniform_int_distribution<int32_t> coordinateDistribution(-50000, 50000);
    std::uniform_int_distribution<int> valueDistribution(-1000000, 1000000);

    std::unordered_map<uint64_t, int> expected;
    expected.reserve(pointCount);

    while (expected.size() < pointCount) {
        const int32_t x = coordinateDistribution(rng);
        const int32_t z = coordinateDistribution(rng);
        const uint64_t key = pointKey(x, z);
        if (expected.find(key) != expected.end())
            continue;

        expected.emplace(key, valueDistribution(rng));
    }

    blocklab::QuadTree<int> tree;
    for (const auto& [key, value] : expected)
        tree.insert(keyX(key), keyZ(key), value);

    CHECK(tree.size() == expected.size());
    for (const auto& [key, value] : expected) {
        auto it = tree.find(keyX(key), keyZ(key));
        REQUIRE(it != tree.end());
        CHECK(*it == value);
    }

    std::unordered_map<uint64_t, int> actual;
    actual.reserve(expected.size());
    for (auto it = tree.cbegin(); it != tree.cend(); ++it)
        actual.emplace(pointKey(it.x(), it.z()), *it);
    CHECK(actual == expected);

    for (const auto& entry : expected) {
        const uint64_t key = entry.first;
        const std::size_t erased = tree.erase({ .x = keyX(key), .z = keyZ(key), .width = 1, .height = 1 });
        CHECK(erased == 1);
    }

    CHECK(tree.empty());
    CHECK(tree.size() == 0);
    CHECK(tree.begin() == tree.end());
    CHECK(tree.cbegin() == tree.cend());
    for (const auto& entry : expected) {
        const uint64_t key = entry.first;
        CHECK(tree.find(keyX(key), keyZ(key)) == tree.end());
    }
}

TEST_CASE("QuadTree supports move construction and assignment", "[quad-tree]")
{
    blocklab::QuadTree<int> tree;
    tree.insert(-2, 4, 10);
    tree.insert(8, -6, 20);

    blocklab::QuadTree<int> moved(std::move(tree));
    CHECK(tree.empty());
    CHECK(moved.size() == 2);

    blocklab::QuadTree<int> assigned;
    assigned = std::move(moved);
    CHECK(moved.empty());
    CHECK(assigned.size() == 2);

    REQUIRE(assigned.find(-2, 4) != assigned.end());
    REQUIRE(assigned.find(8, -6) != assigned.end());
    CHECK(*assigned.find(-2, 4) == 10);
    CHECK(*assigned.find(8, -6) == 20);
}
