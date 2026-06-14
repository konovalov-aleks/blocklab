#pragma once

#include <blocklab/Error.h>

#include <vulkan/vulkan.hpp>

#include <cstdint>

namespace blocklab {

inline std::uint32_t findMemoryType(
    vk::PhysicalDevice physicalDevice, std::uint32_t typeBits, vk::MemoryPropertyFlags flags)
{
    vk::PhysicalDeviceMemoryProperties properties = physicalDevice.getMemoryProperties();
    for (std::uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
        if ((typeBits & (1U << i)) && (properties.memoryTypes[i].propertyFlags & flags) == flags)
            return i;
    }
    fatalError("No compatible Vulkan memory type");
}

} // namespace blocklab
