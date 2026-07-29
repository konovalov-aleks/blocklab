#pragma once

#include <blocklab/utility/Error.h>

#include <vulkan/vulkan.hpp>

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace blocklab {

inline void vkCheck(VkResult result, const char* operation)
{
    if (result != VK_SUCCESS) [[unlikely]]
        fatalError(operation, " failed with VkResult ", result);
}

inline void vkCheck(vk::Result result, const char* operation)
{
    if (result != vk::Result::eSuccess) [[unlikely]]
        fatalError(operation, " failed with vk::Result ", static_cast<int>(result));
}

template <typename T>
[[nodiscard]] T vkCheck(vk::ResultValue<T>&& rv, const char* operation)
{
    if (rv.result != vk::Result::eSuccess) [[unlikely]]
        fatalError(operation, " failed with vk::Result ", static_cast<int>(rv.result));
    return std::move(rv.value);
}

template <typename T, typename PoolT>
auto wrapPoolUnique(T rawObject, vk::Device device, PoolT pool)
{
    using DefaultDispatch = VULKAN_HPP_DEFAULT_DISPATCHER_TYPE;
    using Deleter = typename vk::UniqueHandleTraits<T, DefaultDispatch>::deleter;
    return vk::UniqueHandle<T, DefaultDispatch>(rawObject, Deleter(device, pool));
}

class VulkanInstance {
public:
    VulkanInstance(bool useGLFW);

    const vk::Instance& get() const { return *m_instance; }

private:
    vk::UniqueInstance m_instance;
};

class Display;

class Vulkan {
public:
    Vulkan(VulkanInstance&, vk::SurfaceKHR = nullptr);

    vk::Instance instance() const { return m_instance.get(); }
    vk::PhysicalDevice physicalDevice() const { return m_physicalDevice; }
    vk::Device device() const { return *m_device; }

    vk::Queue graphicsQueue() const { return m_graphicsQueue; }
    std::uint32_t graphicsQueueFamilyIndex() const { return m_graphicsQueueIndex; }

    vk::Queue presentQueue() const { return m_presentQueue; }
    std::uint32_t presentQueueFamilyIndex() const { return m_presentQueueIndex.value_or(m_graphicsQueueIndex); }

private:
    using RequiredExtensions = std::vector<const char*>;

    RequiredExtensions requiredExtensions(bool presentationSupport);
    void choosePhysicalDevice(const RequiredExtensions&, vk::SurfaceKHR presentSurface);
    void createDevice(const RequiredExtensions&);

    VulkanInstance& m_instance;
    vk::PhysicalDevice m_physicalDevice;
    vk::UniqueDevice m_device;

    vk::Queue m_graphicsQueue;
    vk::Queue m_presentQueue;

    std::uint32_t m_graphicsQueueIndex = static_cast<std::uint32_t>(-1);
    std::optional<std::uint32_t> m_presentQueueIndex;
};

} // namespace blocklab
