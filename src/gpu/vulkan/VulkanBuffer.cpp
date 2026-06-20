#include <blocklab/gpu/vulkan/VulkanBuffer.h>

#include "Memory.h"

#include <cassert>
#include <cstddef>
#include <cstring>
#include <utility>

namespace blocklab {

VulkanBuffer& VulkanBuffer::operator=(VulkanBuffer&& other)
{
    if (this != &other) {
        m_vkBuffer = std::move(other.m_vkBuffer);
        m_vkMemory = std::move(other.m_vkMemory);
        m_size = std::exchange(other.m_size, 0);
        m_memorySize = std::exchange(other.m_memorySize, 0);
        m_usage = std::exchange(other.m_usage, {});
    }
    return *this;
}

VulkanBuffer::VulkanBuffer(
    Vulkan& vk, vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, bool exportMemory)
    : m_size(size)
    , m_usage(usage)
{
    const vk::ExternalMemoryBufferCreateInfo externalBufferInfo {
        .handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd,
    };

    const vk::BufferCreateInfo bufferInfo {
        .pNext = exportMemory ? &externalBufferInfo : nullptr,
        .size = size,
        .usage = usage,
        .sharingMode = vk::SharingMode::eExclusive,
    };

    m_vkBuffer = vkCheck(vk.device().createBufferUnique(bufferInfo), "VkDevice::createBuffer");

    vk::MemoryRequirements requirements = vk.device().getBufferMemoryRequirements(*m_vkBuffer);
    m_memorySize = requirements.size;

    const vk::ExportMemoryAllocateInfo exportAllocInfo { .handleTypes
        = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd };
    const vk::MemoryAllocateInfo allocInfo {
        .pNext = exportMemory ? &exportAllocInfo : nullptr,
        .allocationSize = requirements.size,
        .memoryTypeIndex = findMemoryType(vk.physicalDevice(), requirements.memoryTypeBits, properties),
    };
    m_vkMemory = vkCheck(vk.device().allocateMemoryUnique(allocInfo), "VkDevice::allocateMemory");
    vkCheck(vk.device().bindBufferMemory(*m_vkBuffer, *m_vkMemory, 0), "VkDevice::bindBufferMemory");
}

void VulkanBuffer::copySync(
    Vulkan& vk, vk::CommandPool commandPool, const VulkanBuffer& source, VulkanBuffer& destination, vk::DeviceSize size)
{
    assert(size > 0);
    assert(size <= source.size());
    assert(size <= destination.size());
    assert(source.usage() & vk::BufferUsageFlagBits::eTransferSrc);
    assert(destination.usage() & vk::BufferUsageFlagBits::eTransferDst);

    const vk::CommandBufferAllocateInfo allocInfo {
        .commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1
    };
    vk::CommandBuffer commandBufferRaw;
    vkCheck(vk.device().allocateCommandBuffers(&allocInfo, &commandBufferRaw), "VkDevice::allocateCommandBuffers");
    vk::UniqueCommandBuffer commandBuffer = wrapPoolUnique(commandBufferRaw, vk.device(), commandPool);

    const vk::CommandBufferBeginInfo beginInfo { .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit };
    vkCheck(commandBuffer->begin(beginInfo), "VkCommandBuffer::begin");
    const vk::BufferCopy copyRegion {
        .size = size,
    };
    commandBuffer->copyBuffer(source.vkBuffer(), destination.vkBuffer(), 1, &copyRegion);
    vkCheck(commandBuffer->end(), "VkCommandBuffer::end");

    vk::SubmitInfo submitInfo {
        .commandBufferCount = 1,
        .pCommandBuffers = &*commandBuffer,
    };
    vkCheck(vk.graphicsQueue().submit(1, &submitInfo, {}), "VkQueue::submit");
    vkCheck(vk.graphicsQueue().waitIdle(), "VkQueue::waitIdle");
}

void VulkanHostBuffer::upload(Vulkan& vk, vk::DeviceSize dstOffset, const void* data, vk::DeviceSize size)
{
    assert(data);
    assert(dstOffset < this->size());
    assert(size > 0 && (size <= this->size() - dstOffset));

    void* mapped = vkCheck(vk.device().mapMemory(vkMemory(), dstOffset, size), "vkMapMemory");
    std::memcpy(mapped, data, static_cast<std::size_t>(size));
    vk.device().unmapMemory(vkMemory());
}

void VulkanDeviceBuffer::uploadSync(Vulkan& vk, vk::CommandPool commandPool, const void* data, vk::DeviceSize size)
{
    assert(data);
    assert(size > 0 && size <= this->size());

    VulkanHostBuffer staging(vk, size, vk::BufferUsageFlagBits::eTransferSrc);
    staging.upload(vk, 0, data, size);
    VulkanBuffer::copySync(vk, commandPool, staging, *this, size);
}

} // namespace blocklab
