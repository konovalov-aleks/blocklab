#pragma once

#include "blocklab/CudaHelpers.h"

#include <cuda_runtime.h>

#include <cassert>
#include <functional>
#include <optional>
#include <utility>

namespace blocklab {

namespace details {

    class CudaFutureControlBlockImplBase {
    public:
        CudaFutureControlBlockImplBase() = default;

        CudaFutureControlBlockImplBase(cudaStream_t stream)
            : m_stream(stream)
            , m_pending(true)
        {
            cudaCheck(cudaEventCreateWithFlags(&m_done, cudaEventDisableTiming), "cudaEventCreateWithFlags future");
            cudaCheck(cudaEventRecord(m_done, m_stream), "cudaEventRecord future");
        }

        ~CudaFutureControlBlockImplBase() { destroyEvent(); }

        CudaFutureControlBlockImplBase(const CudaFutureControlBlockImplBase&) = delete;
        CudaFutureControlBlockImplBase& operator=(const CudaFutureControlBlockImplBase&) = delete;

        CudaFutureControlBlockImplBase(CudaFutureControlBlockImplBase&& other) noexcept
            : m_stream(std::exchange(other.m_stream, nullptr))
            , m_done(std::exchange(other.m_done, nullptr))
            , m_pending(std::exchange(other.m_pending, false))
        {
        }

        CudaFutureControlBlockImplBase& operator=(CudaFutureControlBlockImplBase&& other) noexcept
        {
            if (this == &other)
                return *this;

            destroyEvent();
            m_stream = std::exchange(other.m_stream, nullptr);
            m_done = std::exchange(other.m_done, nullptr);
            m_pending = std::exchange(other.m_pending, false);
            return *this;
        }

        void wait()
        {
            if (!m_pending || !valid())
                return;

            cudaCheck(cudaEventSynchronize(m_done), "cudaEventSynchronize future");
            m_pending = false;
            destroyEvent();
        }

        void enqueueGPUWait(cudaStream_t stream)
        {
            if (!m_pending || !valid())
                return;

            cudaCheck(cudaStreamWaitEvent(stream, m_done, 0), "cudaStreamWaitEvent future");
        }

        bool valid() const { return m_stream; }

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
        bool m_pending = false;
    };

} // namespace details

template <typename T>
class CudaFutureControlBlock : public details::CudaFutureControlBlockImplBase {
    using BaseT = details::CudaFutureControlBlockImplBase;

public:
    CudaFutureControlBlock() = default;

    CudaFutureControlBlock(cudaStream_t stream, std::function<T()> readResult)
        : BaseT(stream)
        , m_readResult(std::move(readResult))
    {
    }

    CudaFutureControlBlock(CudaFutureControlBlock&& other) noexcept
        : BaseT(std::move(other))
        , m_readResult(std::move(other.m_readResult))
        , m_result(std::move(other.m_result))
    {
        other.m_result.reset();
    }

    CudaFutureControlBlock& operator=(CudaFutureControlBlock&& other) noexcept
    {
        if (this == &other)
            return *this;

        BaseT::operator=(std::move(other));

        m_readResult = std::move(other.m_readResult);
        m_result = std::move(other.m_result);
        other.m_result.reset();

        return *this;
    }

    T& get()
    {
        assert(valid());
        wait();
        if (!m_result)
            m_result = m_readResult();
        return *m_result;
    }

private:
    std::function<T()> m_readResult;
    std::optional<T> m_result;
};

template <>
class CudaFutureControlBlock<void> : public details::CudaFutureControlBlockImplBase {
    using BaseT = details::CudaFutureControlBlockImplBase;

public:
    using BaseT::BaseT;
};

} // namespace blocklab
