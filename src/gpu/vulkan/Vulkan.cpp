#ifdef __APPLE__
#   define VK_ENABLE_BETA_EXTENSIONS
#endif // __APPLE__

#include <blocklab/gpu/vulkan/Vulkan.h>

#include <blocklab/graphics/Display.h>

#include <GLFW/glfw3.h>

#include <cstdint>
#include <optional>
#include <string_view>
#include <unordered_set>
#include <vector>

// #define DEBUG_LOGS

#ifdef DEBUG_LOGS
#   include <iostream>
#endif // DEBUG_LOGS

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

    vk::InstanceCreateFlags flags = {};

#ifdef __APPLE__
    std::vector<const char*> instanceExtensions(extensions, extensions + extensionCount);
    instanceExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    extensions = instanceExtensions.data();
    extensionCount = static_cast<std::uint32_t>(instanceExtensions.size());
    flags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
#endif // __APPLE__

    const vk::InstanceCreateInfo info {
        .flags = flags,
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
    #ifndef CUDA_CPU_FALLBACK_MODE
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
    #endif // CUDA_CPU_FALLBACK_MODE
        VK_EXT_SHADER_VIEWPORT_INDEX_LAYER_EXTENSION_NAME,
    #ifdef __APPLE__
        VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME,
    #endif
    };
    if (presentationSupport)
        result.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    return result;
}

void Vulkan::choosePhysicalDevice(const RequiredExtensions& requiredExtensions, vk::SurfaceKHR presentSurface)
{
    std::vector<vk::PhysicalDevice> devices
        = vkCheck(instance().enumeratePhysicalDevices(), "VkInstance::enumeratePhysicalDevices");

    #ifdef DEBUG_LOGS
    std::cout << "\n========================================" << std::endl;
    std::cout << "Found " << devices.size() << " physical devices" << std::endl;
    std::cout << "========================================\n" << std::endl;
    #endif

    std::unordered_set<std::string_view> propNames;

    for (const vk::PhysicalDevice& device : devices) {
        #ifdef DEBUG_LOGS
        vk::PhysicalDeviceProperties props;
        device.getProperties(&props);

        std::cout << "--- Device: " << props.deviceName << " ---" << std::endl;
        std::cout << "  Type: " << vk::to_string(props.deviceType) << std::endl;
        std::cout << "  API Version: " << VK_API_VERSION_MAJOR(props.apiVersion) << "."
                  << VK_API_VERSION_MINOR(props.apiVersion) << "."
                  << VK_API_VERSION_PATCH(props.apiVersion) << std::endl;
        #endif // DEBUG_LOGS

        // check queues
        std::vector<vk::QueueFamilyProperties> queueFamilies = device.getQueueFamilyProperties();
        std::optional<std::uint32_t> graphicsQueueIndex;
        std::optional<std::uint32_t> presentQueueIndex;

        #ifdef DEBUG_LOGS
        std::cout << "  Queue families: " << queueFamilies.size() << std::endl;
        #endif
        for (std::uint32_t i = 0; i < queueFamilies.size(); ++i) {
            #ifdef DEBUG_LOGS
            std::cout << "    Family " << i << ": flags=" << vk::to_string(queueFamilies[i].queueFlags) << std::endl;
            #endif
            if (queueFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics) {
                graphicsQueueIndex = i;
                #ifdef DEBUG_LOGS
                std::cout << "      -> Graphics queue found at index " << i << std::endl;
                #endif
            }
            if (presentSurface) {
                if (vkCheck(device.getSurfaceSupportKHR(i, presentSurface), "VkPhysicalDevice::getSurfaceSupportKHR")) {
                    presentQueueIndex = i;
                    #ifdef DEBUG_LOGS
                    std::cout << "      -> Present queue found at index " << i << std::endl;
                    #endif
                }
            }
        }

        if (!graphicsQueueIndex) {
            #ifdef DEBUG_LOGS
            std::cout << "  [FAIL] No graphics queue" << std::endl;
            #endif
            continue;
        }
        if (presentSurface && !presentQueueIndex) {
            #ifdef DEBUG_LOGS
            std::cout << "  [FAIL] No present queue (surface requires presentation)" << std::endl;
            #endif
            continue;
        }
        #ifdef DEBUG_LOGS
        std::cout << "  [OK] Graphics and present queues available" << std::endl;
        #endif

        // check extensions
        std::vector<vk::ExtensionProperties> extensionProperties = vkCheck(
            device.enumerateDeviceExtensionProperties(), "VkPhysicalDevice::enumerateDeviceExtensionProperties");
        propNames.clear();
        propNames.reserve(extensionProperties.size());
        for (const vk::ExtensionProperties& p : extensionProperties)
            propNames.insert(p.extensionName);

        #ifdef DEBUG_LOGS
        std::cout << "  Required extensions (" << requiredExtensions.size() << "):" << std::endl;
        #endif
        bool supportsExtensions = true;
        for (const char* extName : requiredExtensions) {
            bool supported = propNames.contains(extName);
            #ifdef DEBUG_LOGS
            std::cout << "    " << extName << ": " << (supported ? "YES" : "NO") << std::endl;
            #endif
            if (!supported) {
                supportsExtensions = false;
            }
        }
        if (!supportsExtensions) {
            #ifdef DEBUG_LOGS
            std::cout << "  [FAIL] Missing required extensions" << std::endl;
            #endif
            continue;
        }
        #ifdef DEBUG_LOGS
        std::cout << "  [OK] All required extensions supported" << std::endl;
        #endif

        // check Vulkan 1.2 features
        vk::PhysicalDeviceVulkan12Features features12 = {};
        vk::PhysicalDeviceFeatures2 features2 = {};
        features2.pNext = &features12;
        device.getFeatures2(&features2);

        #ifdef DEBUG_LOGS
        std::cout << "  Vulkan 1.2 features:" << std::endl;
        std::cout << "    shaderOutputLayer: " << (features12.shaderOutputLayer ? "YES" : "NO") << std::endl;
        std::cout << "    shaderOutputViewportIndex: " << (features12.shaderOutputViewportIndex ? "YES" : "NO") << std::endl;
        #endif

        if (features12.shaderOutputLayer != vk::True || features12.shaderOutputViewportIndex != vk::True) {
            #ifdef DEBUG_LOGS
            std::cout << "  [FAIL] Missing required Vulkan 1.2 features" << std::endl;
            #endif
            continue;
        }
        #ifdef DEBUG_LOGS
        std::cout << "  [OK] All required Vulkan 1.2 features supported" << std::endl;
        #endif

        m_physicalDevice = device;
        m_graphicsQueueIndex = *graphicsQueueIndex;
        m_presentQueueIndex = presentQueueIndex;

        #ifdef DEBUG_LOGS
        std::cout << "\n[SUCCESS] Selected device: " << props.deviceName << std::endl;
        std::cout << "========================================\n" << std::endl;
        #endif
        return;
    }

    #ifdef DEBUG_LOGS
    std::cout << "\n[FAIL] No suitable Vulkan physical device found!" << std::endl;
    std::cout << "========================================\n" << std::endl;
    #endif
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
