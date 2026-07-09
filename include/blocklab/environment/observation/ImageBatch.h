#pragma once

#include <blocklab/gpu/cuda/CudaSharedFuture.h>

#include <cstdint>
#include <utility>

namespace blocklab {

class ImageBatch {
public:
    std::uint32_t width() const { return m_width; }
    std::uint32_t height() const { return m_height; }
    std::uint32_t channels() const { return 3; }
    std::uint32_t batchSize() const { return m_batchSize; }

    float* data() const { return m_data; }
    bool ready() const { return m_ready.ready(); }
    void enqueueReadyWait(cudaStream_t stream) const { m_ready.enqueueGPUWait(stream); }

    void reset(std::uint32_t width, std::uint32_t height, std::uint32_t batchSize)
    {
        m_width = width;
        m_height = height;
        m_batchSize = batchSize;
        m_data = nullptr;
        m_ready = {};
    }

    void setData(float* data, CudaSharedFuture<void> ready)
    {
        m_data = data;
        m_ready = std::move(ready);
    }

private:
    float* m_data = nullptr;
    CudaSharedFuture<void> m_ready;
    std::uint32_t m_width = 0;
    std::uint32_t m_height = 0;
    std::uint32_t m_batchSize = 0;
};

} // namespace blocklab
