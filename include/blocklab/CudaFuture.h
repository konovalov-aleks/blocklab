#pragma once

#include "blocklab/CudaHelpers.h"

#include <cuda_runtime.h>

#include <functional>
#include <optional>
#include <utility>

namespace blocklab {

template <typename T>
class CudaSharedFuture;

template <typename T>
class CudaFuture {
public:
    CudaFuture() = default;

    CudaFuture(cudaStream_t stream, std::function<T()> readResult)
        : m_stream(stream)
        , m_readResult(std::move(readResult))
        , m_pending(true)
    {
    }

    CudaFuture(const CudaFuture&) = delete;
    CudaFuture& operator=(const CudaFuture&) = delete;

    CudaFuture(CudaFuture&& other) noexcept
        : m_stream(std::exchange(other.m_stream, nullptr))
        , m_readResult(std::move(other.m_readResult))
        , m_result(std::move(other.m_result))
        , m_pending(std::exchange(other.m_pending, false))
    {
    }

    CudaFuture& operator=(CudaFuture&& other) noexcept
    {
        if (this == &other)
            return *this;

        m_stream = std::exchange(other.m_stream, nullptr);
        m_readResult = std::move(other.m_readResult);
        m_result = std::move(other.m_result);
        m_pending = std::exchange(other.m_pending, false);
        return *this;
    }

    void wait()
    {
        if (!m_pending)
            return;

        cudaCheck(cudaStreamSynchronize(m_stream), "cudaStreamSynchronize future");
        m_pending = false;
    }

    T get()
    {
        wait();
        if (!m_result)
            m_result = m_readResult();
        return *m_result;
    }

private:
    friend class CudaSharedFuture<T>;

    cudaStream_t m_stream = nullptr;
    std::function<T()> m_readResult;
    std::optional<T> m_result;
    bool m_pending = false;
};

} // namespace blocklab
