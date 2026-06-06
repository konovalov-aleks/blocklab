#pragma once

#include "blocklab/CudaFutureControlBlock.h"

#include <cuda_runtime.h>

#include <functional>
#include <utility>

namespace blocklab {

template <typename T>
class CudaSharedFuture;

template <typename T>
class CudaFuture {
public:
    CudaFuture() = default;

    CudaFuture(cudaStream_t stream, std::function<T()> readResult)
        : m_control(stream, std::move(readResult))
    {
    }

    CudaFuture(const CudaFuture&) = delete;
    CudaFuture& operator=(const CudaFuture&) = delete;

    CudaFuture(CudaFuture&&) noexcept = default;
    CudaFuture& operator=(CudaFuture&&) noexcept = default;

    void wait() { m_control.wait(); }
    T& get() { return m_control.get(); }
    bool valid() const { return m_control.valid(); }
    bool ready() const { return m_control.ready(); }

    CudaSharedFuture<T> share();

private:
    friend class CudaSharedFuture<T>;

    CudaFutureControlBlock<T> m_control;
};

} // namespace blocklab
