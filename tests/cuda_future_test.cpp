#include "blocklab/CudaFuture.h"
#include "blocklab/CudaSharedFuture.h"

#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include <memory>
#include <utility>

namespace {

class TestStream {
public:
    TestStream() { REQUIRE(cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking) == cudaSuccess); }

    ~TestStream()
    {
        if (m_stream)
            cudaStreamDestroy(m_stream);
    }

    TestStream(const TestStream&) = delete;
    TestStream& operator=(const TestStream&) = delete;

    cudaStream_t get() const { return m_stream; }

private:
    cudaStream_t m_stream = nullptr;
};

} // namespace

TEST_CASE("CudaFuture exposes validity and move-only ownership", "[cuda][cuda-future]")
{
    blocklab::CudaFuture<int> empty;
    CHECK_FALSE(empty.valid());
    CHECK_FALSE(empty.ready());
    empty.wait();

    TestStream stream;
    auto readCount = std::make_shared<int>(0);
    blocklab::CudaFuture<int> future(stream.get(), [readCount] {
        ++*readCount;
        return 42;
    });

    CHECK(future.valid());

    blocklab::CudaFuture<int> moved(std::move(future));
    CHECK(moved.valid());
    CHECK_FALSE(future.valid());

    moved.wait();
    moved.wait();
    CHECK(moved.ready());
    CHECK(*readCount == 0);
    CHECK(moved.get() == 42);
    CHECK(moved.get() == 42);
    CHECK(*readCount == 1);
    CHECK(moved.valid());
}

TEST_CASE("CudaFuture move assignment transfers cached result", "[cuda][cuda-future]")
{
    TestStream stream;
    auto readCount = std::make_shared<int>(0);
    blocklab::CudaFuture<int> source(stream.get(), [readCount] {
        ++*readCount;
        return 7;
    });
    CHECK(source.get() == 7);

    blocklab::CudaFuture<int> assigned;
    assigned = std::move(source);

    CHECK(assigned.valid());
    CHECK_FALSE(source.valid());
    CHECK(assigned.get() == 7);
    CHECK(*readCount == 1);
}

TEST_CASE("CudaSharedFuture is empty when built from empty CudaFuture", "[cuda][cuda-future]")
{
    blocklab::CudaFuture<int> empty;
    blocklab::CudaSharedFuture<int> shared(std::move(empty));

    CHECK_FALSE(shared.valid());
    CHECK_FALSE(shared.ready());
    CHECK_FALSE(empty.valid());
    shared.wait();
}

TEST_CASE("CudaSharedFuture copies share wait and result state", "[cuda][cuda-future]")
{
    TestStream stream;
    auto readCount = std::make_shared<int>(0);
    blocklab::CudaFuture<int> future(stream.get(), [readCount] {
        ++*readCount;
        return 99;
    });

    blocklab::CudaSharedFuture<int> shared = future.share();
    blocklab::CudaSharedFuture<int> copy = shared;

    CHECK(shared.valid());
    CHECK(copy.valid());
    CHECK_FALSE(future.valid());

    shared.wait();
    copy.wait();
    CHECK(shared.ready());
    CHECK(copy.ready());
    CHECK(*readCount == 0);
    CHECK(shared.get() == 99);
    CHECK(copy.get() == 99);
    CHECK(*readCount == 1);
}

TEST_CASE("CudaSharedFuture can take an already materialized CudaFuture", "[cuda][cuda-future]")
{
    TestStream stream;
    auto readCount = std::make_shared<int>(0);
    blocklab::CudaFuture<int> future(stream.get(), [readCount] {
        ++*readCount;
        return 123;
    });
    CHECK(future.get() == 123);

    blocklab::CudaSharedFuture<int> shared(std::move(future));

    CHECK(shared.valid());
    CHECK_FALSE(future.valid());
    CHECK(shared.get() == 123);
    CHECK(*readCount == 1);
}

TEST_CASE("CudaFuture share moves state into CudaSharedFuture", "[cuda][cuda-future]")
{
    TestStream stream;
    auto readCount = std::make_shared<int>(0);
    blocklab::CudaFuture<int> future(stream.get(), [readCount] {
        ++*readCount;
        return 314;
    });

    blocklab::CudaSharedFuture<int> shared = future.share();

    CHECK(shared.valid());
    CHECK_FALSE(future.valid());
    CHECK(shared.get() == 314);
    CHECK(*readCount == 1);
}
