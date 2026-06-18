#include <blocklab/graphics/VulkanCudaInteropSemaphore.h>

#include <blocklab/CudaHelpers.h>
#include <blocklab/Error.h>
#include <blocklab/graphics/Vulkan.h>

#include <cuda_runtime.h>
#include <vulkan/vulkan.hpp>

#include <cstdint>
#include <utility>

namespace blocklab {

namespace details {

    CudaSemaphoreBase::CudaSemaphoreBase(
        vk::Device device, vk::Semaphore vkSemaphore, cudaExternalSemaphoreHandleType cudaHandleType)
    {
        auto vkGetSemaphoreFdKHRFn
            = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(vkGetDeviceProcAddr(device, "vkGetSemaphoreFdKHR"));
        if (!vkGetSemaphoreFdKHRFn) [[unlikely]]
            fatalError("vkGetSemaphoreFdKHR is unavailable");

        const VkSemaphoreGetFdInfoKHR fdInfo {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
            .semaphore = vkSemaphore,
            .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT,
        };
        int fd = -1;
        vkCheck(vkGetSemaphoreFdKHRFn(device, &fdInfo, &fd), "vkGetSemaphoreFdKHR");

        cudaExternalSemaphoreHandleDesc externalSemaphoreDesc {};
        externalSemaphoreDesc.type = cudaHandleType;
        externalSemaphoreDesc.handle.fd = fd;
        cudaCheck(cudaImportExternalSemaphore(&m_cudaSemaphore, &externalSemaphoreDesc),
            "cudaImportExternalSemaphore render");
    }

    CudaSemaphoreBase::CudaSemaphoreBase(CudaSemaphoreBase&& other) noexcept
        : m_cudaSemaphore(std::exchange(other.m_cudaSemaphore, nullptr))
    {
    }

    CudaSemaphoreBase& CudaSemaphoreBase::operator=(CudaSemaphoreBase&& other) noexcept
    {
        std::swap(m_cudaSemaphore, other.m_cudaSemaphore);
        return *this;
    }

    void CudaSemaphoreBase::resetCudaSemaphore()
    {
        if (m_cudaSemaphore) {
            cudaCheck(cudaDestroyExternalSemaphore(m_cudaSemaphore), "cudaDestroyExternalSemaphore");
            m_cudaSemaphore = nullptr;
        }
    }

} // namespace details

VulkanCudaInteropBinarySemaphore::VulkanCudaInteropBinarySemaphore(vk::Device device)
    : VulkanBinarySemaphore(device, true)
    , CudaSemaphoreBase(device, vkSemaphore(), cudaExternalSemaphoreHandleTypeOpaqueFd)
{
}

VulkanCudaInteropBinarySemaphore& VulkanCudaInteropBinarySemaphore::operator=(
    VulkanCudaInteropBinarySemaphore&& other) noexcept
{
    if (this == &other)
        return *this;

    resetCudaSemaphore();
    VulkanBinarySemaphore::operator=(std::move(other));
    details::CudaSemaphoreBase::operator=(std::move(other));
    return *this;
}

void VulkanCudaInteropBinarySemaphore::enqueueWait(cudaStream_t stream)
{
    cudaExternalSemaphoreWaitParams waitParams = {};
    cudaCheck(cudaWaitExternalSemaphoresAsync(&cudaSemaphore(), &waitParams, 1, stream),
        "cudaWaitExternalSemaphoresAsync[binary]");
}

void VulkanCudaInteropBinarySemaphore::enqueueSignal(cudaStream_t stream)
{
    cudaExternalSemaphoreSignalParams signalParams = {};
    cudaCheck(cudaSignalExternalSemaphoresAsync(&cudaSemaphore(), &signalParams, 1, stream),
        "cudaSignalExternalSemaphoresAsync[binary]");
}

VulkanCudaInteropTimelineSemaphore::VulkanCudaInteropTimelineSemaphore(vk::Device device, std::uint64_t initialValue)
    : VulkanTimelineSemaphore(device, initialValue, true)
    , CudaSemaphoreBase(device, vkSemaphore(), cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd)
{
}

VulkanCudaInteropTimelineSemaphore& VulkanCudaInteropTimelineSemaphore::operator=(
    VulkanCudaInteropTimelineSemaphore&& other) noexcept
{
    if (this == &other)
        return *this;

    resetCudaSemaphore();
    VulkanTimelineSemaphore::operator=(std::move(other));
    details::CudaSemaphoreBase::operator=(std::move(other));
    return *this;
}

void VulkanCudaInteropTimelineSemaphore::enqueueWait(cudaStream_t stream, std::uint64_t value)
{
    cudaExternalSemaphoreWaitParams waitParams = {};
    waitParams.params.fence.value = value;
    cudaCheck(cudaWaitExternalSemaphoresAsync(&cudaSemaphore(), &waitParams, 1, stream),
        "cudaWaitExternalSemaphoresAsync[timeline]");
}

void VulkanCudaInteropTimelineSemaphore::enqueueSignal(cudaStream_t stream, std::uint64_t value)
{
    cudaExternalSemaphoreSignalParams signalParams = {};
    signalParams.params.fence.value = value;
    cudaCheck(cudaSignalExternalSemaphoresAsync(&cudaSemaphore(), &signalParams, 1, stream),
        "cudaSignalExternalSemaphoresAsync[timeline]");
}

} // namespace blocklab
