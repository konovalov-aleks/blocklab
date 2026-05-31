#pragma once

#include "blocklab/CudaFuture.h"

#include <cuda_runtime.h>

#include <functional>
#include <memory>
#include <optional>
#include <utility>

namespace blocklab {

template <typename T>
class CudaSharedFuture {
public:
    CudaSharedFuture() = default;

    CudaSharedFuture(CudaFuture<T>&& future)
        : m_control(std::make_shared<ControlBlock>(std::move(future)))
    {
    }

    void wait() const
    {
        if (m_control)
            m_control->wait();
    }

    T get() const { return m_control->get(); }

    explicit operator bool() const { return static_cast<bool>(m_control); }

private:
    struct ControlBlock {
        explicit ControlBlock(CudaFuture<T>&& future)
            : stream(std::exchange(future.m_stream, nullptr))
            , readResult(std::move(future.m_readResult))
            , result(std::move(future.m_result))
            , pending(std::exchange(future.m_pending, false))
        {
        }

        void wait()
        {
            if (!pending)
                return;

            cudaCheck(cudaStreamSynchronize(stream), "cudaStreamSynchronize shared future");
            pending = false;
        }

        T get()
        {
            wait();
            if (!result)
                result = readResult();
            return *result;
        }

        cudaStream_t stream = nullptr;
        std::function<T()> readResult;
        std::optional<T> result;
        bool pending = false;
    };

    std::shared_ptr<ControlBlock> m_control;
};

} // namespace blocklab
