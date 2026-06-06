#pragma once

#include "blocklab/CudaFuture.h"

#include <cuda_runtime.h>

#include <memory>
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

    void wait() const
    {
        if (m_control)
            m_control->wait();
    }

    T& get() const { return m_control->get(); }

    bool valid() const { return static_cast<bool>(m_control); }

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
