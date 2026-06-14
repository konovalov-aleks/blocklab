#pragma once

#include "blocklab/CudaFuture.h"

#include <cuda_runtime.h>

#include <cassert>
#include <memory>
#include <type_traits>
#include <utility>

namespace blocklab {

template <typename T>
class CudaSharedFuture {
public:
    CudaSharedFuture() = default;

    CudaSharedFuture(CudaFuture<T>&& future)
    {
        if (future.valid())
            m_control = std::make_shared<CudaFutureControlBlock<T>>(std::move(future.m_control));
    }

    auto& get() const
        requires(!std::is_void_v<T>)
    {
        assert(m_control);
        return m_control->get();
    }

    void wait() const
    {
        if (m_control) [[likely]]
            m_control->wait();
    }

    void enqueueGPUWait(cudaStream_t stream) const
    {
        if (m_control) [[likely]]
            m_control->enqueueGPUWait(stream);
    }

    bool valid() const { return m_control && m_control->valid(); }
    bool ready() const { return m_control && m_control->ready(); }

private:
    std::shared_ptr<CudaFutureControlBlock<T>> m_control;
};

template <typename T>
CudaSharedFuture<T> CudaFuture<T>::share()
{
    return CudaSharedFuture<T>(std::move(*this));
}

} // namespace blocklab
