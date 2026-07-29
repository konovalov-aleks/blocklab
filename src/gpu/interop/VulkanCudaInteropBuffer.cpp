#include <blocklab/gpu/interop/VulkanCudaInteropBuffer.h>

#include <blocklab/gpu/cuda/CudaHelpers.h>
#include <blocklab/utility/Error.h>

#include <utility>

namespace blocklab {

VulkanCudaInteropBuffer::VulkanCudaInteropBuffer(Vulkan& vk, vk::DeviceSize size, vk::BufferUsageFlags usage)
    #ifndef CUDA_CPU_FALLBACK_MODE
    : VulkanBuffer(vk, size, usage, vk::MemoryPropertyFlagBits::eDeviceLocal, true)
    #else
    : VulkanBuffer(
        vk, size, usage ,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, false)
    #endif
{
    #ifndef CUDA_CPU_FALLBACK_MODE
    auto vkGetMemoryFdKHRFn
        = reinterpret_cast<PFN_vkGetMemoryFdKHR>(vkGetDeviceProcAddr(vk.device(), "vkGetMemoryFdKHR"));
    if (!vkGetMemoryFdKHRFn) [[unlikely]]
        fatalError("vkGetMemoryFdKHR is unavailable");

    const VkMemoryGetFdInfoKHR fdInfo {
        .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
        .memory = vkMemory(),
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    };
    int fd = -1;
    vkCheck(vkGetMemoryFdKHRFn(vk.device(), &fdInfo, &fd), "vkGetMemoryFdKHR");

    cudaExternalMemoryHandleDesc externalMemoryDesc {
        .type = cudaExternalMemoryHandleTypeOpaqueFd,
        .handle = { .fd = fd },
        .size = static_cast<unsigned long long>(physicalSize()),
    };
    #else
    cudaExternalMemoryHandleDesc externalMemoryDesc {
        .device = vk.device(),
        .deviceMemory = vkMemory(),
        .size = static_cast<unsigned long long>(physicalSize()),
    };
    #endif // !CUDA_CPU_FALLBACK_MODE
    cudaCheck(cudaImportExternalMemory(&m_cudaMemory, &externalMemoryDesc), "cudaImportExternalMemory");

    cudaExternalMemoryBufferDesc bufferDesc {};
    bufferDesc.offset = 0;
    bufferDesc.size = static_cast<unsigned long long>(this->size());
    cudaCheck(
        cudaExternalMemoryGetMappedBuffer(&m_cudaPtr, m_cudaMemory, &bufferDesc), "cudaExternalMemoryGetMappedBuffer");
}

VulkanCudaInteropBuffer& VulkanCudaInteropBuffer::operator=(VulkanCudaInteropBuffer&& other)
{
    if (this != &other) {
        reset();
        BaseT::operator=(std::move(other));
        std::swap(m_cudaMemory, other.m_cudaMemory);
        std::swap(m_cudaPtr, other.m_cudaPtr);
    }
    return *this;
}

void VulkanCudaInteropBuffer::reset()
{
    if (m_cudaPtr) {
        cudaCheck(cudaFree(m_cudaPtr), "cudaFree external buffer mapping");
        m_cudaPtr = nullptr;
    }
    if (m_cudaMemory) {
        cudaCheck(cudaDestroyExternalMemory(m_cudaMemory), "cudaDestroyExternalMemory");
        m_cudaMemory = nullptr;
    }
}

} // namespace blocklab
