#pragma once

#include <vulkan/vulkan.hpp>

#include <cassert>
#include <cstdint>

namespace blocklab {

class VulkanSemaphore {
public:
    VulkanSemaphore(VulkanSemaphore&& other) noexcept = default;
    VulkanSemaphore& operator=(VulkanSemaphore&& other) noexcept = default;

    VulkanSemaphore(const VulkanSemaphore&) = delete;
    VulkanSemaphore& operator=(const VulkanSemaphore&) = delete;

    vk::Semaphore vkSemaphore() const
    {
        assert(m_semaphore);
        return *m_semaphore;
    }

protected:
    VulkanSemaphore() = default;
    ~VulkanSemaphore() = default;

    VulkanSemaphore(vk::Device, vk::SemaphoreType, std::uint64_t initialValue, bool external);

private:
    vk::UniqueSemaphore m_semaphore;
};

class VulkanBinarySemaphore : public VulkanSemaphore {
public:
    VulkanBinarySemaphore() = default;
    VulkanBinarySemaphore(vk::Device device)
        : VulkanBinarySemaphore(device, false)
    {
    }

protected:
    VulkanBinarySemaphore(vk::Device device, bool exported)
        : VulkanSemaphore(device, vk::SemaphoreType::eBinary, 0, exported)
    {
    }
};

class VulkanTimelineSemaphore : public VulkanSemaphore {
public:
    VulkanTimelineSemaphore() = default;
    VulkanTimelineSemaphore(vk::Device device, std::uint64_t initialValue)
        : VulkanTimelineSemaphore(device, initialValue, false)
    {
    }

protected:
    VulkanTimelineSemaphore(vk::Device device, std::uint64_t initialValue, bool exported)
        : VulkanSemaphore(device, vk::SemaphoreType::eTimeline, initialValue, exported)
    {
    }
};

} // namespace
