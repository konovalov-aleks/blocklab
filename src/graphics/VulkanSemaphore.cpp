#include <blocklab/graphics/VulkanSemaphore.h>

#include <blocklab/graphics/Vulkan.h>

#include <cstdint>
#include <utility>

namespace blocklab {

VulkanSemaphore::VulkanSemaphore(vk::Device device, vk::SemaphoreType type, std::uint64_t initialValue, bool exported)
{
    const vk::ExportSemaphoreCreateInfo exportInfo {
        .handleTypes = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd,
    };
    vk::SemaphoreTypeCreateInfo typeCreateInfo = {
        .pNext = exported ? &exportInfo : nullptr,
        .semaphoreType = type,
        .initialValue = initialValue,
    };
    const vk::SemaphoreCreateInfo semaphoreInfo {
        .pNext = &typeCreateInfo,
    };
    m_semaphore = vkCheck(device.createSemaphoreUnique(semaphoreInfo), "vk::Device::createSemaphore");
}

} // namespace blocklab
