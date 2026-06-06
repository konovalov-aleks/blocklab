#pragma once

#include "blocklab/CudaHelpers.h"

#include <cuda_runtime.h>

#include <functional>
#include <optional>
#include <utility>

namespace blocklab {

template <typename T>
class CudaFutureControlBlock {
public:
    CudaFutureControlBlock() = default;

    CudaFutureControlBlock(cudaStream_t stream, std::function<T()> readResult)
        : m_stream(stream)
        , m_readResult(std::move(readResult))
        , m_pending(true)
    {
        cudaCheck(cudaEventCreateWithFlags(&m_done, cudaEventDisableTiming), "cudaEventCreateWithFlags future");
        cudaCheck(cudaEventRecord(m_done, m_stream), "cudaEventRecord future");
    }

    ~CudaFutureControlBlock() { destroyEvent(); }

    CudaFutureControlBlock(const CudaFutureControlBlock&) = delete;
    CudaFutureControlBlock& operator=(const CudaFutureControlBlock&) = delete;

    CudaFutureControlBlock(CudaFutureControlBlock&& other) noexcept
        : m_stream(std::exchange(other.m_stream, nullptr))
        , m_done(std::exchange(other.m_done, nullptr))
        , m_readResult(std::move(other.m_readResult))
        , m_result(std::move(other.m_result))
        , m_pending(std::exchange(other.m_pending, false))
    {
        other.m_result.reset();
    }

    CudaFutureControlBlock& operator=(CudaFutureControlBlock&& other) noexcept
    {
        if (this == &other)
            return *this;

        destroyEvent();
        m_stream = std::exchange(other.m_stream, nullptr);
        m_done = std::exchange(other.m_done, nullptr);
        m_readResult = std::move(other.m_readResult);
        m_result = std::move(other.m_result);
        other.m_result.reset();
        m_pending = std::exchange(other.m_pending, false);
        return *this;
    }

    void wait()
    {
        if (!m_pending)
            return;

        cudaCheck(cudaEventSynchronize(m_done), "cudaEventSynchronize future");
        m_pending = false;
        destroyEvent();
    }

    T& get()
    {
        wait();
        if (!m_result)
            m_result = m_readResult();
        return *m_result;
    }

    bool valid() const { return m_stream || m_result.has_value(); }

    bool ready() const
    {
        if (!valid())
            return false;
        if (!m_pending)
            return true;

        const cudaError_t queryResult = cudaEventQuery(m_done);
        if (queryResult == cudaSuccess)
            return true;
        if (queryResult == cudaErrorNotReady)
            return false;
        cudaCheck(queryResult, "cudaEventQuery future");
        return false;
    }

private:
    void destroyEvent()
    {
        if (!m_done)
            return;
        cudaCheck(cudaEventDestroy(m_done), "cudaEventDestroy future");
        m_done = nullptr;
    }

    cudaStream_t m_stream = nullptr;
    cudaEvent_t m_done = nullptr;
    std::function<T()> m_readResult;
    std::optional<T> m_result;
    bool m_pending = false;
};

} // namespace blocklab
