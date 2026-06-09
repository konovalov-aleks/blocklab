#pragma once

#include "blocklab/CudaFutureControlBlock.h"

#include <cuda_runtime.h>

#include <functional>
#include <type_traits>
#include <utility>

namespace blocklab {

template <typename T>
class CudaSharedFuture;

template <typename T>
class CudaFuture {
public:
    CudaFuture() = default;

    CudaFuture(cudaStream_t stream)
        requires std::is_void_v<T>
        : m_control(stream)
    {
    }

    CudaFuture(cudaStream_t stream, std::function<T()> readResult)
        requires(!std::is_void_v<T>)
        : m_control(stream, std::move(readResult))
    {
    }

    CudaFuture(const CudaFuture&) = delete;
    CudaFuture& operator=(const CudaFuture&) = delete;

    CudaFuture(CudaFuture&&) noexcept = default;
    CudaFuture& operator=(CudaFuture&&) noexcept = default;

    auto& get()
        requires(!std::is_void_v<T>)
    {
        return m_control.get();
    }

    void wait() { m_control.wait(); }
    void enqueueGPUWait(cudaStream_t stream) { m_control.enqueueGPUWait(stream); }

    bool valid() const { return m_control.valid(); }
    bool ready() const { return m_control.ready(); }

    CudaSharedFuture<T> share();

private:
    friend class CudaSharedFuture<T>;

    CudaFutureControlBlock<T> m_control;
};

} // namespace blocklab
