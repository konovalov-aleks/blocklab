#pragma once

#include <blocklab/gpu/vulkan/VulkanBuffer.h>

#include <cuda_runtime.h>
#include <utility>

namespace blocklab {

class VulkanCudaInteropBuffer : public VulkanBuffer {
    using BaseT = VulkanBuffer;

public:
    VulkanCudaInteropBuffer() = default;

    VulkanCudaInteropBuffer(Vulkan&, vk::DeviceSize, vk::BufferUsageFlags);

    VulkanCudaInteropBuffer(VulkanCudaInteropBuffer&& other) { *this = std::move(other); }
    VulkanCudaInteropBuffer& operator=(VulkanCudaInteropBuffer&& other);

    ~VulkanCudaInteropBuffer() { reset(); }

    template <typename T = void>
    T* cudaPtr() const
    {
        return reinterpret_cast<T*>(m_cudaPtr);
    }

private:
    void reset();

    cudaExternalMemory_t m_cudaMemory = nullptr;
    void* m_cudaPtr = nullptr;
};

} // namespace blocklab
