#include <blocklab/gpu/vulkan/Vulkan.h>

#include <blocklab/graphics/Display.h>

#include <GLFW/glfw3.h>

#include <cstdint>
#include <optional>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace blocklab {

VulkanInstance::VulkanInstance(bool useGLFW)
{
    vk::ApplicationInfo appInfo {
        .pApplicationName = "BlockLab",
        .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
        .pEngineName = "BlockLab",
        .engineVersion = VK_MAKE_VERSION(0, 1, 0),
        .apiVersion = VK_API_VERSION_1_2,
    };

    std::uint32_t extensionCount = 0;
    const char** extensions = nullptr;

    if (useGLFW)
        extensions = glfwGetRequiredInstanceExtensions(&extensionCount);

    vk::InstanceCreateInfo info {
        .pApplicationInfo = &appInfo,
        .enabledExtensionCount = extensionCount,
        .ppEnabledExtensionNames = extensions,
    };

    m_instance = vkCheck(vk::createInstanceUnique(info), "vk::createInstance");
}

Vulkan::Vulkan(VulkanInstance& instance, vk::SurfaceKHR presentSurface)
    : m_instance(instance)
{
    const RequiredExtensions extensions = requiredExtensions(presentSurface != nullptr);
    choosePhysicalDevice(extensions, presentSurface);
    createDevice(extensions);
}

Vulkan::RequiredExtensions Vulkan::requiredExtensions(bool presentationSupport)
{
    RequiredExtensions result = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
        VK_EXT_SHADER_VIEWPORT_INDEX_LAYER_EXTENSION_NAME,
    };
    if (presentationSupport)
        result.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    return result;
}

void Vulkan::choosePhysicalDevice(const RequiredExtensions& requiredExtensions, vk::SurfaceKHR presentSurface)
{
    std::vector<vk::PhysicalDevice> devices
        = vkCheck(instance().enumeratePhysicalDevices(), "VkInstance::enumeratePhysicalDevices");

    std::unordered_set<std::string_view> propNames;

    for (const vk::PhysicalDevice& device : devices) {
        std::vector<vk::QueueFamilyProperties> queueFamilies = device.getQueueFamilyProperties();
        std::optional<std::uint32_t> graphicsQueueIndex;
        std::optional<std::uint32_t> presentQueueIndex;
        for (std::uint32_t i = 0; i < queueFamilies.size(); ++i) {
            if (queueFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics)
                graphicsQueueIndex = i;
            if (presentSurface) {
                if (vkCheck(device.getSurfaceSupportKHR(i, presentSurface), "VkPhysicalDevice::getSurfaceSupportKHR"))
                    presentQueueIndex = i;
            }
        }
        if (!graphicsQueueIndex || (presentSurface && !presentQueueIndex))
            continue;

        std::vector<vk::ExtensionProperties> extensionProperties = vkCheck(
            device.enumerateDeviceExtensionProperties(), "VkPhysicalDevice::enumerateDeviceExtensionProperties");
        propNames.clear();
        propNames.reserve(extensionProperties.size());
        for (const vk::ExtensionProperties& p : extensionProperties)
            propNames.insert(p.extensionName);

        bool supportsExtensions = true;
        for (const char* extName : requiredExtensions) {
            if (!propNames.contains(extName)) {
                supportsExtensions = false;
                break;
            }
        }
        if (!supportsExtensions)
            continue;

        vk::PhysicalDeviceVulkan12Features features12;
        vk::PhysicalDeviceFeatures2 features2 {
            .pNext = &features12,
        };
        device.getFeatures2(&features2);
        if (features12.shaderOutputLayer != vk::True || features12.shaderOutputViewportIndex != vk::True)
            continue;

        m_physicalDevice = device;
        m_graphicsQueueIndex = *graphicsQueueIndex;
        m_presentQueueIndex = presentQueueIndex;
        return;
    }
    fatalError("No suitable Vulkan physical device found");
}

void Vulkan::createDevice(const RequiredExtensions& extensions)
{
    vk::PhysicalDeviceVulkan12Features features12 {
        .timelineSemaphore = vk::True,
        .shaderOutputViewportIndex = vk::True,
        .shaderOutputLayer = vk::True,
    };

    const float priority[] = { 1.0f };
    std::uint32_t infoCount = 1;
    vk::DeviceQueueCreateInfo queueCreateInfos[2] {
        {
            .queueFamilyIndex = m_graphicsQueueIndex,
            .queueCount = 1,
            .pQueuePriorities = priority,
        },
    };
    if (m_presentQueueIndex && *m_presentQueueIndex != m_graphicsQueueIndex) {
        ++infoCount;
        queueCreateInfos[1] = vk::DeviceQueueCreateInfo {
            .queueFamilyIndex = *m_presentQueueIndex,
            .queueCount = 1,
            .pQueuePriorities = priority,
        };
    }

    const vk::DeviceCreateInfo devInfo {
        .pNext = &features12,
        .queueCreateInfoCount = infoCount,
        .pQueueCreateInfos = queueCreateInfos,
        .enabledExtensionCount = static_cast<std::uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data(),
    };

    m_device = vkCheck(physicalDevice().createDeviceUnique(devInfo), "VkPhysicalDevice::createDevice");

    m_graphicsQueue = m_device->getQueue(m_graphicsQueueIndex, 0);
    if (m_presentQueueIndex)
        m_presentQueue = m_device->getQueue(*m_presentQueueIndex, 0);
}

} // namespace
