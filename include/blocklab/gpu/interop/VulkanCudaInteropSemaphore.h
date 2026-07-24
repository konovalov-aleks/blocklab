#pragma once

#include <blocklab/gpu/vulkan/VulkanSemaphore.h>

#include <cuda_runtime.h>
#include <vulkan/vulkan.hpp>

#include <cassert>
#include <cstdint>

namespace blocklab {

namespace details {

    class CudaSemaphoreBase {
    public:
        CudaSemaphoreBase() = default;
        CudaSemaphoreBase(vk::Device, vk::Semaphore, cudaExternalSemaphoreHandleType);
        ~CudaSemaphoreBase() { resetCudaSemaphore(); }

        CudaSemaphoreBase(CudaSemaphoreBase&&) noexcept;
        CudaSemaphoreBase& operator=(CudaSemaphoreBase&&) noexcept;

        CudaSemaphoreBase(const CudaSemaphoreBase&) = delete;
        CudaSemaphoreBase& operator=(const CudaSemaphoreBase&) = delete;

        const cudaExternalSemaphore_t& cudaSemaphore() const
        {
            assert(m_cudaSemaphore);
            return m_cudaSemaphore;
        }

    protected:
        void resetCudaSemaphore();

    private:
        cudaExternalSemaphore_t m_cudaSemaphore = nullptr;
    };

} // namespace details

#ifndef CUDA_CPU_FALLBACK_MODE

class VulkanCudaInteropBinarySemaphore : public VulkanBinarySemaphore, public details::CudaSemaphoreBase {
public:
    VulkanCudaInteropBinarySemaphore() = default;
    VulkanCudaInteropBinarySemaphore(vk::Device);

    VulkanCudaInteropBinarySemaphore(VulkanCudaInteropBinarySemaphore&& other) noexcept = default;
    VulkanCudaInteropBinarySemaphore& operator=(VulkanCudaInteropBinarySemaphore&& other) noexcept;

    void enqueueWait(cudaStream_t);
    void enqueueSignal(cudaStream_t);
};
#endif // !CUDA_CPU_FALLBACK_MODE

class VulkanCudaInteropTimelineSemaphore : public VulkanTimelineSemaphore, public details::CudaSemaphoreBase {
public:
    VulkanCudaInteropTimelineSemaphore() = default;
    VulkanCudaInteropTimelineSemaphore(vk::Device, std::uint64_t initialValue = 0);

    VulkanCudaInteropTimelineSemaphore(VulkanCudaInteropTimelineSemaphore&& other) noexcept = default;
    VulkanCudaInteropTimelineSemaphore& operator=(VulkanCudaInteropTimelineSemaphore&& other) noexcept;

    void enqueueWait(cudaStream_t, std::uint64_t value);
    void enqueueSignal(cudaStream_t, std::uint64_t value);
};

} // namespace blocklab