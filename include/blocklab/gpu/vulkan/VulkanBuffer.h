#pragma once

#include "Vulkan.h"

#include <vulkan/vulkan.hpp>

#include <cassert>
#include <utility>

namespace blocklab {

class VulkanBuffer {
public:
    VulkanBuffer() = default;

    VulkanBuffer(const VulkanBuffer&) = delete;
    VulkanBuffer& operator=(const VulkanBuffer&) = delete;

    VulkanBuffer(VulkanBuffer&& other) { *this = std::move(other); }
    VulkanBuffer& operator=(VulkanBuffer&& other);

    vk::DeviceSize size() const { return m_size; }
    vk::DeviceSize physicalSize() const { return m_memorySize; }

    vk::DeviceMemory vkMemory() const
    {
        assert(m_vkMemory);
        return *m_vkMemory;
    }

    vk::Buffer vkBuffer() const
    {
        assert(m_vkBuffer);
        return *m_vkBuffer;
    }

    vk::BufferUsageFlags usage() const { return m_usage; }

    vk::DescriptorBufferInfo info() const { return vk::DescriptorBufferInfo { .buffer = vkBuffer(), .range = size() }; }

    static void copySync(
        Vulkan&, vk::CommandPool, const VulkanBuffer& source, VulkanBuffer& destination, vk::DeviceSize size);

protected:
    VulkanBuffer(Vulkan&, vk::DeviceSize size, vk::BufferUsageFlags, vk::MemoryPropertyFlags, bool exportMemory);

private:
    vk::UniqueBuffer m_vkBuffer;
    vk::UniqueDeviceMemory m_vkMemory;
    vk::DeviceSize m_size = 0;
    vk::DeviceSize m_memorySize = 0;
    vk::BufferUsageFlags m_usage = {};
};

class VulkanHostBuffer : public VulkanBuffer {
public:
    VulkanHostBuffer() = default;

    VulkanHostBuffer(Vulkan& vk, vk::DeviceSize size, vk::BufferUsageFlags usage)
        : VulkanBuffer(vk, size, usage,
              vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, false)
    {
    }

    void upload(Vulkan&, vk::DeviceSize dstOffset, const void* data, vk::DeviceSize size);
};

class VulkanDeviceBuffer : public VulkanBuffer {
public:
    VulkanDeviceBuffer() = default;

    VulkanDeviceBuffer(Vulkan& vk, vk::DeviceSize size, vk::BufferUsageFlags usage)
        : VulkanBuffer(vk, size, usage, vk::MemoryPropertyFlagBits::eDeviceLocal, false)
    {
    }

    void uploadSync(Vulkan&, vk::CommandPool commandPool, const void* data, vk::DeviceSize size);
};

} // namespace blocklab
