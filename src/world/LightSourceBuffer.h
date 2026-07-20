#pragma once

#include <blocklab/gpu/cuda/CudaHelpers.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace blocklab::worldgen {

class LightSourceBuffer {
public:
    struct DeviceData {
        uchar3* data;
        std::uint32_t capacity;
        std::uint32_t* size;
    };

    LightSourceBuffer()
    {
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&m_sizeDevice), sizeof(*m_sizeDevice)),
            "cudaMalloc (LightSourceBuffer::m_sizeDevice)");
        cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&m_sizeHost), sizeof(*m_sizeHost)),
            "cudaMallocHost (LightSourceBuffer::m_sizeHost)");

        *m_sizeHost = 0;
    }

    LightSourceBuffer(std::uint32_t initialCapacity)
        : LightSourceBuffer()
    {
        reserve(initialCapacity);
    }

    ~LightSourceBuffer()
    {
        cudaFree(m_data);
        cudaFree(m_sizeDevice);
        cudaFreeHost(m_sizeHost);
    }

    void reserve(std::uint32_t newCapacity)
    {
        if (newCapacity <= m_capacity)
            return;
        if (m_data)
            cudaFree(m_data);

        if (m_capacity == 0)
            m_capacity = newCapacity;
        else {
            while (m_capacity < newCapacity)
                m_capacity = std::max(m_capacity + 1, m_capacity + m_capacity / 2);
        }

        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&m_data), sizeof(*m_data) * m_capacity),
            "cudaMalloc (LightSourceBuffer::m_data)");
    }

    std::uint32_t capacity() const { return m_capacity; }
    std::uint32_t size() const { return *m_sizeHost; }

    DeviceData deviceData() const { return { m_data, m_capacity, m_sizeDevice }; }

    void enqueueClear(cudaStream_t stream)
    {
        cudaCheck(cudaMemsetAsync(m_sizeDevice, 0, sizeof(*m_sizeDevice), stream),
            "cudaMemsetAsync (LightSourceBuffer::enqueueClear)");
    }

    void enqueueUploadSize(cudaStream_t stream)
    {
        static_assert(std::is_same_v<decltype(*m_sizeHost), decltype(*m_sizeDevice)>);
        cudaCheck(cudaMemcpyAsync(m_sizeHost, m_sizeDevice, sizeof(*m_sizeHost), cudaMemcpyDeviceToHost, stream),
            "cudaMemcpyAsync (LightSourceBuffer::enqueueUploadSize)");
    }

private:
    LightSourceBuffer(const LightSourceBuffer&) = delete;
    LightSourceBuffer& operator=(const LightSourceBuffer&) = delete;

    uchar3* m_data = nullptr;
    std::uint32_t* m_sizeDevice;
    std::uint32_t* m_sizeHost;
    std::uint32_t m_capacity = 0;
};

} // namespace blocklab
