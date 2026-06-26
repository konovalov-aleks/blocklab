#include <gpu/cuda/PageLockedVector.h>

#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <type_traits>

namespace {

void requireCuda(cudaError_t result) { REQUIRE(result == cudaSuccess); }

struct Sample {
    std::int32_t x = 0;
    float y = 0.0f;
};

static_assert(std::is_trivially_copyable_v<Sample>);

} // namespace

TEST_CASE("PageLockedVector grows and preserves existing values", "[cuda][page-locked-vector]")
{
    blocklab::PageLockedVector<std::int32_t> values;
    CHECK(values.empty());
    CHECK(values.size() == 0);
    CHECK(values.capacity() == 0);
    CHECK(values.data() == nullptr);

    values.push_back(7);
    values.push_back(11);
    values.push_back(13);
    values.push_back(17);
    values.push_back(19);

    REQUIRE(values.size() == 5);
    CHECK(values.capacity() >= values.size());
    CHECK(values[0] == 7);
    CHECK(values[1] == 11);
    CHECK(values[2] == 13);
    CHECK(values[3] == 17);
    CHECK(values[4] == 19);
}

TEST_CASE("PageLockedVector resize zero-initializes new storage", "[cuda][page-locked-vector]")
{
    blocklab::PageLockedVector<Sample> values;
    values.push_back({ .x = 3, .y = 1.5f });
    values.resize(4);

    REQUIRE(values.size() == 4);
    CHECK(values[0].x == 3);
    CHECK(values[0].y == 1.5f);
    for (std::size_t i = 1; i < values.size(); ++i) {
        CHECK(values[i].x == 0);
        CHECK(values[i].y == 0.0f);
    }

    values.resize(2);
    REQUIRE(values.size() == 2);
    CHECK(values[0].x == 3);
    CHECK(values[1].x == 0);
}

TEST_CASE("PageLockedVector uninitializedResize updates size without reallocating within capacity",
    "[cuda][page-locked-vector]")
{
    blocklab::PageLockedVector<std::uint8_t> values;
    values.reserve(16);
    std::uint8_t* const data = values.data();
    const std::size_t capacity = values.capacity();

    values.uninitializedResize(12);
    CHECK(values.size() == 12);
    CHECK(values.capacity() == capacity);
    CHECK(values.data() == data);

    for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = static_cast<std::uint8_t>(i + 1U);

    values.clear();
    CHECK(values.empty());
    CHECK(values.capacity() == capacity);
    CHECK(values.data() == data);

    values.uninitializedResize(12);
    CHECK(values.data() == data);
    CHECK(values[0] == 1);
    CHECK(values[11] == 12);
}

TEST_CASE("PageLockedVector move transfers ownership and leaves source empty", "[cuda][page-locked-vector]")
{
    blocklab::PageLockedVector<std::int32_t> source;
    source.push_back(10);
    source.push_back(20);
    std::int32_t* const originalData = source.data();

    blocklab::PageLockedVector<std::int32_t> moved(std::move(source));
    CHECK(source.empty());
    CHECK(source.capacity() == 0);
    CHECK(source.data() == nullptr);
    REQUIRE(moved.size() == 2);
    CHECK(moved.data() == originalData);
    CHECK(moved[0] == 10);
    CHECK(moved[1] == 20);

    blocklab::PageLockedVector<std::int32_t> assigned;
    assigned.push_back(1);
    assigned = std::move(moved);
    CHECK(moved.empty());
    CHECK(moved.capacity() == 0);
    CHECK(moved.data() == nullptr);
    REQUIRE(assigned.size() == 2);
    CHECK(assigned.data() == originalData);
    CHECK(assigned[0] == 10);
    CHECK(assigned[1] == 20);
}

TEST_CASE("PageLockedVector memory can be used as CUDA memcpy host memory", "[cuda][page-locked-vector]")
{
    blocklab::PageLockedVector<std::uint32_t> values;
    values.resize(8);
    for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = static_cast<std::uint32_t>(100U + i);

    std::uint32_t* device = nullptr;
    requireCuda(cudaMalloc(&device, sizeof(std::uint32_t) * values.size()));
    requireCuda(cudaMemcpy(device, values.data(), sizeof(std::uint32_t) * values.size(), cudaMemcpyHostToDevice));

    std::array<std::uint32_t, 8> copied {};
    requireCuda(cudaMemcpy(copied.data(), device, sizeof(std::uint32_t) * copied.size(), cudaMemcpyDeviceToHost));
    requireCuda(cudaFree(device));

    for (std::size_t i = 0; i < copied.size(); ++i)
        CHECK(copied[i] == static_cast<std::uint32_t>(100U + i));
}

TEST_CASE("PageLockedVector iterators cover mutable elements", "[cuda][page-locked-vector]")
{
    blocklab::PageLockedVector<std::int32_t> values;
    for (std::int32_t value : { 1, 2, 3, 4 })
        values.push_back(value);

    CHECK(values.begin() == values.data());
    CHECK(values.end() == values.data() + values.size());
    CHECK(std::distance(values.begin(), values.end()) == static_cast<std::ptrdiff_t>(values.size()));

    for (std::int32_t& value : values)
        value *= 3;

    CHECK(values[0] == 3);
    CHECK(values[1] == 6);
    CHECK(values[2] == 9);
    CHECK(values[3] == 12);
}

TEST_CASE("PageLockedVector const iterators expose read-only range", "[cuda][page-locked-vector]")
{
    blocklab::PageLockedVector<std::int32_t> values;
    values.push_back(5);
    values.push_back(7);
    values.push_back(11);

    const blocklab::PageLockedVector<std::int32_t>& constValues = values;
    CHECK(constValues.begin() == values.data());
    CHECK(constValues.cbegin() == values.data());
    CHECK(constValues.end() == values.data() + values.size());
    CHECK(constValues.cend() == values.data() + values.size());

    const std::int32_t sum = std::accumulate(constValues.begin(), constValues.end(), 0);
    CHECK(sum == 23);
}

TEST_CASE("PageLockedVector iterators work with standard algorithms", "[cuda][page-locked-vector]")
{
    blocklab::PageLockedVector<std::int32_t> values;
    values.resize(6);

    std::fill(values.begin(), values.end(), 4);
    CHECK(std::all_of(values.cbegin(), values.cend(), [](std::int32_t value) { return value == 4; }));

    values[2] = 9;
    auto found = std::find(values.begin(), values.end(), 9);
    REQUIRE(found != values.end());
    *found = 12;
    CHECK(values[2] == 12);
}

TEST_CASE("PageLockedVector empty iterators form an empty range", "[cuda][page-locked-vector]")
{
    blocklab::PageLockedVector<std::int32_t> values;
    CHECK(values.begin() == values.end());
    CHECK(values.cbegin() == values.cend());

    values.reserve(8);
    CHECK(values.begin() == values.end());
    CHECK(values.cbegin() == values.cend());
}
