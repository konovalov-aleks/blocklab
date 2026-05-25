#include "blocklab/Renderer.h"

#include "blocklab/CudaHelpers.h"
#include "blocklab/Error.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace blocklab {
namespace {

    constexpr float EyeHeight = 1.62f;
    constexpr VkFormat ColorFormat = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr uint32_t OffscreenFrameCount = 8;
    constexpr uint32_t MaxEntityInstances = 256;
    constexpr uint32_t MaxPigVertices = 1024;
    constexpr int32_t TerrainMeshHalfExtent = 32;
    constexpr uint32_t TerrainMeshExtent = TerrainMeshHalfExtent * 2;
    constexpr uint32_t MaxTerrainVertices
        = static_cast<uint32_t>(TerrainMeshExtent * Chunk::SizeY * TerrainMeshExtent * 36);

    Vec3 cameraForward(float yaw, float pitch)
    {
        const float pitchCos = std::cos(pitch);
        return glm::normalize(Vec3 { std::sin(yaw) * pitchCos, std::sin(pitch), std::cos(yaw) * pitchCos });
    }

    std::vector<char> readFile(const std::filesystem::path path)
    {
        std::ifstream file(path, std::ios::ate | std::ios::binary);
        if (!file) [[unlikely]]
            fatalError("Failed to open", path);
        const std::streamsize size = file.tellg();
        std::vector<char> bytes(static_cast<std::size_t>(size));
        file.seekg(0);
        file.read(bytes.data(), size);
        return bytes;
    }

    void vkCheck(VkResult result, const char* operation)
    {
        if (result != VK_SUCCESS) [[unlikely]]
            fatalError(operation, "failed with VkResult", result);
    }

    uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeBits, VkMemoryPropertyFlags flags)
    {
        VkPhysicalDeviceMemoryProperties properties {};
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &properties);
        for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
            if ((typeBits & (1U << i)) && (properties.memoryTypes[i].propertyFlags & flags) == flags)
                return i;
        }
        fatalError("No compatible Vulkan memory type");
    }

    struct QueueFamilies {
        std::optional<uint32_t> graphics;
        std::optional<uint32_t> present;

        bool complete(bool requirePresent) const { return graphics && (!requirePresent || present); }
    };

    QueueFamilies findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface)
    {
        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());

        QueueFamilies result;
        for (uint32_t i = 0; i < count; ++i) {
            if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
                result.graphics = i;
            if (surface) {
                VkBool32 present = VK_FALSE;
                vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &present);
                if (present)
                    result.present = i;
            }
        }
        return result;
    }

    struct Buffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize size = 0;
        VkDeviceSize memorySize = 0;
        cudaExternalMemory_t cudaMemory = nullptr;
        void* cudaPtr = nullptr;
    };

    struct Image {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
    };

    struct OffscreenFrame {
        Image color;
        Image depth;
        VkFramebuffer framebuffer = VK_NULL_HANDLE;
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        Buffer paramsBuffer;
        VkDescriptorSet paramsDescriptorSet = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
    };

    bool hasStencilComponent(VkFormat format)
    {
        return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
    }

    bool hasDeviceExtension(VkPhysicalDevice device, std::string_view name)
    {
        uint32_t extensionCount = 0;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> extensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, extensions.data());
        return std::any_of(extensions.begin(), extensions.end(),
            [name](const VkExtensionProperties& extension) { return extension.extensionName == name; });
    }

    VkFormat findSupportedDepthFormat(VkPhysicalDevice physicalDevice)
    {
        const VkFormat candidates[] {
            VK_FORMAT_D32_SFLOAT,
            VK_FORMAT_D32_SFLOAT_S8_UINT,
            VK_FORMAT_D24_UNORM_S8_UINT,
            VK_FORMAT_D16_UNORM,
        };

        for (VkFormat format : candidates) {
            VkFormatProperties properties {};
            vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &properties);
            if ((properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
                == VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
                return format;
            }
        }

        fatalError("No supported Vulkan depth format");
    }

} // namespace

struct Renderer::VulkanState {
    VkInstance instance = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkQueue presentQueue = VK_NULL_HANDLE;
    uint32_t graphicsFamily = 0;
    VkExtent2D renderExtent {};
    VkFormat colorFormat = ColorFormat;
    VkFormat depthFormat = VK_FORMAT_UNDEFINED;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkFormat swapchainFormat = VK_FORMAT_UNDEFINED;
    VkExtent2D swapchainExtent {};
    std::vector<VkImage> swapchainImages;
    std::vector<VkImageView> swapchainImageViews;
    Image depthImage;
    std::vector<OffscreenFrame> offscreenFrames;
    std::size_t nextOffscreenFrame = 0;
    std::size_t lastSubmittedOffscreenFrame = 0;
    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkDescriptorSetLayout vertexSetLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout paramsSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> framebuffers;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet vertexDescriptorSet = VK_NULL_HANDLE;
    VkDescriptorSet paramsDescriptorSet = VK_NULL_HANDLE;
    Buffer vertexBuffer;
    Buffer instanceBuffer;
    Buffer paramsBuffer;
    Buffer observationBuffer;
    VkSemaphore imageAvailable = VK_NULL_HANDLE;
    VkSemaphore renderFinished = VK_NULL_HANDLE;
    VkFence inFlight = VK_NULL_HANDLE;
    VkDeviceSize vertexCapacityBytes = 0;
};

namespace {

    Buffer createBuffer(Renderer::VulkanState& vk, VkDeviceSize size, VkBufferUsageFlags usage,
        VkMemoryPropertyFlags properties, bool exportMemory = false)
    {
        Buffer buffer;
        buffer.size = size;
        const VkExternalMemoryBufferCreateInfo externalBufferInfo {
            .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
            .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
        };
        const VkBufferCreateInfo bufferInfo {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = exportMemory ? &externalBufferInfo : nullptr,
            .size = size,
            .usage = usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        vkCheck(vkCreateBuffer(vk.device, &bufferInfo, nullptr, &buffer.buffer), "vkCreateBuffer");

        VkMemoryRequirements requirements {};
        vkGetBufferMemoryRequirements(vk.device, buffer.buffer, &requirements);
        buffer.memorySize = requirements.size;
        const VkExportMemoryAllocateInfo exportAllocInfo {
            .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
            .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
        };
        const VkMemoryAllocateInfo allocInfo {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = exportMemory ? &exportAllocInfo : nullptr,
            .allocationSize = requirements.size,
            .memoryTypeIndex = findMemoryType(vk.physicalDevice, requirements.memoryTypeBits, properties),
        };
        vkCheck(vkAllocateMemory(vk.device, &allocInfo, nullptr, &buffer.memory), "vkAllocateMemory");
        vkCheck(vkBindBufferMemory(vk.device, buffer.buffer, buffer.memory, 0), "vkBindBufferMemory");
        return buffer;
    }

    Buffer createHostBuffer(Renderer::VulkanState& vk, VkDeviceSize size, VkBufferUsageFlags usage)
    {
        return createBuffer(
            vk, size, usage, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }

    Buffer createExportedDeviceBuffer(Renderer::VulkanState& vk, VkDeviceSize size, VkBufferUsageFlags usage)
    {
        Buffer buffer = createBuffer(vk, size, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true);

        auto vkGetMemoryFdKHRFn
            = reinterpret_cast<PFN_vkGetMemoryFdKHR>(vkGetDeviceProcAddr(vk.device, "vkGetMemoryFdKHR"));
        if (!vkGetMemoryFdKHRFn) [[unlikely]]
            fatalError("vkGetMemoryFdKHR is unavailable");

        const VkMemoryGetFdInfoKHR fdInfo {
            .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
            .memory = buffer.memory,
            .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
        };
        int fd = -1;
        vkCheck(vkGetMemoryFdKHRFn(vk.device, &fdInfo, &fd), "vkGetMemoryFdKHR");

        cudaExternalMemoryHandleDesc externalMemoryDesc {};
        externalMemoryDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
        externalMemoryDesc.handle.fd = fd;
        externalMemoryDesc.size = static_cast<unsigned long long>(buffer.memorySize);
        cudaCheck(cudaImportExternalMemory(&buffer.cudaMemory, &externalMemoryDesc),
            "cudaImportExternalMemory vertex buffer");

        cudaExternalMemoryBufferDesc bufferDesc {};
        bufferDesc.offset = 0;
        bufferDesc.size = static_cast<unsigned long long>(buffer.size);
        cudaCheck(cudaExternalMemoryGetMappedBuffer(&buffer.cudaPtr, buffer.cudaMemory, &bufferDesc),
            "cudaExternalMemoryGetMappedBuffer vertex buffer");
        return buffer;
    }

    void destroyBuffer(Renderer::VulkanState& vk, Buffer& buffer)
    {
        if (buffer.cudaPtr)
            cudaCheck(cudaFree(buffer.cudaPtr), "cudaFree external buffer mapping");
        if (buffer.cudaMemory)
            cudaCheck(cudaDestroyExternalMemory(buffer.cudaMemory), "cudaDestroyExternalMemory");
        if (buffer.buffer)
            vkDestroyBuffer(vk.device, buffer.buffer, nullptr);
        if (buffer.memory)
            vkFreeMemory(vk.device, buffer.memory, nullptr);
        buffer = {};
    }

    void appendMeshFace(std::vector<MeshVertex>& vertices, Vec3 p0, Vec3 p1, Vec3 p2, Vec3 p3, Vec3 color, float shade,
        MeshMaterial material, float animationPhase)
    {
        const Vec4 packedColor { color, shade };
        const float materialId = meshMaterialId(material);
        vertices.push_back({ .position = { p0, animationPhase },
            .colorAndShade = packedColor,
            .uvMaterial = { 0.0f, 0.0f, materialId, 0.0f } });
        vertices.push_back({ .position = { p1, animationPhase },
            .colorAndShade = packedColor,
            .uvMaterial = { 1.0f, 0.0f, materialId, 0.0f } });
        vertices.push_back({ .position = { p2, animationPhase },
            .colorAndShade = packedColor,
            .uvMaterial = { 1.0f, 1.0f, materialId, 0.0f } });
        vertices.push_back({ .position = { p0, animationPhase },
            .colorAndShade = packedColor,
            .uvMaterial = { 0.0f, 0.0f, materialId, 0.0f } });
        vertices.push_back({ .position = { p2, animationPhase },
            .colorAndShade = packedColor,
            .uvMaterial = { 1.0f, 1.0f, materialId, 0.0f } });
        vertices.push_back({ .position = { p3, animationPhase },
            .colorAndShade = packedColor,
            .uvMaterial = { 0.0f, 1.0f, materialId, 0.0f } });
    }

    void appendMeshCuboid(
        std::vector<MeshVertex>& vertices, Vec3 min, Vec3 max, Vec3 color, float animationPhase = 0.0f)
    {
        const Vec3 p000 { min.x, min.y, min.z };
        const Vec3 p100 { max.x, min.y, min.z };
        const Vec3 p010 { min.x, max.y, min.z };
        const Vec3 p110 { max.x, max.y, min.z };
        const Vec3 p001 { min.x, min.y, max.z };
        const Vec3 p101 { max.x, min.y, max.z };
        const Vec3 p011 { min.x, max.y, max.z };
        const Vec3 p111 { max.x, max.y, max.z };
        const MeshMaterial material = color.g > 0.6f ? MeshMaterial::PigSnout : MeshMaterial::PigSkin;
        appendMeshFace(vertices, p010, p011, p111, p110, color, 1.0f, material, animationPhase);
        appendMeshFace(vertices, p000, p100, p101, p001, color, 0.48f, material, animationPhase);
        appendMeshFace(vertices, p100, p110, p111, p101, color, 0.78f, material, animationPhase);
        appendMeshFace(vertices, p000, p001, p011, p010, color, 0.78f, material, animationPhase);
        appendMeshFace(vertices, p001, p101, p111, p011, color, 0.68f, material, animationPhase);
        appendMeshFace(vertices, p000, p010, p110, p100, color, 0.68f, material, animationPhase);
    }

    void appendMeshPatch(std::vector<MeshVertex>& vertices, Vec3 min, Vec3 max, float z, Vec3 color)
    {
        appendMeshFace(vertices, { min.x, min.y, z }, { max.x, min.y, z }, { max.x, max.y, z }, { min.x, max.y, z },
            color, 1.0f, MeshMaterial::VertexColor, 0.0f);
    }

    std::vector<MeshVertex> createPigMesh()
    {
        std::vector<MeshVertex> vertices;
        vertices.reserve(420U);
        const Vec3 skin { 0.88f, 0.56f, 0.65f };
        const Vec3 snout { 0.96f, 0.66f, 0.73f };
        const Vec3 dark { 0.08f, 0.06f, 0.07f };
        appendMeshCuboid(vertices, { -0.36f, 0.24f, -0.58f }, { 0.36f, 0.78f, 0.58f }, skin);
        appendMeshCuboid(vertices, { -0.30f, 0.34f, 0.50f }, { 0.30f, 0.84f, 0.98f }, skin);
        appendMeshCuboid(vertices, { -0.16f, 0.48f, 0.94f }, { 0.16f, 0.66f, 1.08f }, snout);
        appendMeshPatch(vertices, { -0.23f, 0.65f, 0.0f }, { -0.13f, 0.75f, 0.0f }, 0.982f, dark);
        appendMeshPatch(vertices, { 0.13f, 0.65f, 0.0f }, { 0.23f, 0.75f, 0.0f }, 0.982f, dark);
        appendMeshPatch(vertices, { -0.10f, 0.54f, 0.0f }, { -0.04f, 0.61f, 0.0f }, 1.082f, dark);
        appendMeshPatch(vertices, { 0.04f, 0.54f, 0.0f }, { 0.10f, 0.61f, 0.0f }, 1.082f, dark);
        appendMeshCuboid(vertices, { -0.31f, 0.00f, -0.45f }, { -0.15f, 0.28f, -0.25f }, skin, 2.0f);
        appendMeshCuboid(vertices, { 0.15f, 0.00f, -0.45f }, { 0.31f, 0.28f, -0.25f }, skin, -2.0f);
        appendMeshCuboid(vertices, { -0.31f, 0.00f, 0.25f }, { -0.15f, 0.28f, 0.45f }, skin, -2.0f);
        appendMeshCuboid(vertices, { 0.15f, 0.00f, 0.25f }, { 0.31f, 0.28f, 0.45f }, skin, 2.0f);
        return vertices;
    }

    void destroyImage(Renderer::VulkanState& vk, Image& image)
    {
        if (image.view)
            vkDestroyImageView(vk.device, image.view, nullptr);
        if (image.image)
            vkDestroyImage(vk.device, image.image, nullptr);
        if (image.memory)
            vkFreeMemory(vk.device, image.memory, nullptr);
        image = {};
    }

    Image createDepthImage(Renderer::VulkanState& vk)
    {
        Image image;
        const VkImageCreateInfo imageInfo {
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = vk.depthFormat,
            .extent = { vk.renderExtent.width, vk.renderExtent.height, 1 },
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
        vkCheck(vkCreateImage(vk.device, &imageInfo, nullptr, &image.image), "vkCreateImage depth");

        VkMemoryRequirements requirements {};
        vkGetImageMemoryRequirements(vk.device, image.image, &requirements);
        const VkMemoryAllocateInfo allocInfo {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = requirements.size,
            .memoryTypeIndex
            = findMemoryType(vk.physicalDevice, requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
        };
        vkCheck(vkAllocateMemory(vk.device, &allocInfo, nullptr, &image.memory), "vkAllocateMemory depth");
        vkCheck(vkBindImageMemory(vk.device, image.image, image.memory, 0), "vkBindImageMemory depth");

        const VkImageViewCreateInfo viewInfo {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = image.image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = vk.depthFormat,
            .subresourceRange = { static_cast<VkImageAspectFlags>(VK_IMAGE_ASPECT_DEPTH_BIT
                                      | (hasStencilComponent(vk.depthFormat) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0)),
                0, 1, 0, 1 },
        };
        vkCheck(vkCreateImageView(vk.device, &viewInfo, nullptr, &image.view), "vkCreateImageView depth");
        return image;
    }

    Image createColorImage(Renderer::VulkanState& vk)
    {
        Image image;
        const VkImageCreateInfo imageInfo {
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = vk.colorFormat,
            .extent = { vk.renderExtent.width, vk.renderExtent.height, 1 },
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
        vkCheck(vkCreateImage(vk.device, &imageInfo, nullptr, &image.image), "vkCreateImage color");

        VkMemoryRequirements requirements {};
        vkGetImageMemoryRequirements(vk.device, image.image, &requirements);
        const VkMemoryAllocateInfo allocInfo {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = requirements.size,
            .memoryTypeIndex
            = findMemoryType(vk.physicalDevice, requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
        };
        vkCheck(vkAllocateMemory(vk.device, &allocInfo, nullptr, &image.memory), "vkAllocateMemory color");
        vkCheck(vkBindImageMemory(vk.device, image.image, image.memory, 0), "vkBindImageMemory color");

        const VkImageViewCreateInfo viewInfo {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = image.image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = vk.colorFormat,
            .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
        };
        vkCheck(vkCreateImageView(vk.device, &viewInfo, nullptr, &image.view), "vkCreateImageView color");
        return image;
    }

    VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& bytes)
    {
        const VkShaderModuleCreateInfo info {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = bytes.size(),
            .pCode = reinterpret_cast<const uint32_t*>(bytes.data()),
        };
        VkShaderModule module = VK_NULL_HANDLE;
        vkCheck(vkCreateShaderModule(device, &info, nullptr, &module), "vkCreateShaderModule");
        return module;
    }

    VkSurfaceFormatKHR chooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats)
    {
        for (const VkSurfaceFormatKHR& format : formats) {
            if (format.format == VK_FORMAT_B8G8R8A8_SRGB && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
                return format;
        }
        return formats.front();
    }

    VkPresentModeKHR choosePresentMode(const std::vector<VkPresentModeKHR>& modes)
    {
        for (VkPresentModeKHR mode : modes) {
            if (mode == VK_PRESENT_MODE_IMMEDIATE_KHR)
                return mode;
        }
        for (VkPresentModeKHR mode : modes) {
            if (mode == VK_PRESENT_MODE_MAILBOX_KHR)
                return mode;
        }
        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D chooseExtent(GLFWwindow* window, const VkSurfaceCapabilitiesKHR& capabilities)
    {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
            return capabilities.currentExtent;

        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        return {
            std::clamp(
                static_cast<uint32_t>(width), capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
            std::clamp(
                static_cast<uint32_t>(height), capabilities.minImageExtent.height, capabilities.maxImageExtent.height),
        };
    }

} // namespace
namespace {

    void createInstance(Renderer::VulkanState& vk)
    {
        VkApplicationInfo appInfo {
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pApplicationName = "BlockLab",
            .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
            .pEngineName = "BlockLab",
            .engineVersion = VK_MAKE_VERSION(0, 1, 0),
            .apiVersion = VK_API_VERSION_1_2,
        };

        uint32_t extensionCount = 0;
        const char** extensions = glfwGetRequiredInstanceExtensions(&extensionCount);
        VkInstanceCreateInfo info {
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pApplicationInfo = &appInfo,
            .enabledExtensionCount = extensionCount,
            .ppEnabledExtensionNames = extensions,
        };
        vkCheck(vkCreateInstance(&info, nullptr, &vk.instance), "vkCreateInstance");
    }

    void pickDevice(Renderer::VulkanState& vk)
    {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(vk.instance, &deviceCount, nullptr);
        if (deviceCount == 0) [[unlikely]]
            fatalError("No Vulkan physical devices found");
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(vk.instance, &deviceCount, devices.data());

        for (VkPhysicalDevice device : devices) {
            const QueueFamilies families = findQueueFamilies(device, vk.surface);
            const bool hasSwapchain = !vk.surface || hasDeviceExtension(device, VK_KHR_SWAPCHAIN_EXTENSION_NAME);
            const bool hasExternalMemory = hasDeviceExtension(device, VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME)
                && hasDeviceExtension(device, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
            if (families.complete(vk.surface != VK_NULL_HANDLE) && hasSwapchain && hasExternalMemory) {
                vk.physicalDevice = device;
                vk.graphicsFamily = *families.graphics;
                return;
            }
        }
        fatalError("No suitable Vulkan physical device found");
    }

    void createDevice(Renderer::VulkanState& vk)
    {
        const QueueFamilies families = findQueueFamilies(vk.physicalDevice, vk.surface);
        std::vector<VkDeviceQueueCreateInfo> queueInfos;
        std::array<uint32_t, 2> uniqueFamilies { *families.graphics, families.present.value_or(*families.graphics) };
        const float priority = 1.0f;
        for (uint32_t family : uniqueFamilies) {
            if (!queueInfos.empty() && queueInfos.front().queueFamilyIndex == family)
                continue;
            queueInfos.push_back({
                .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                .queueFamilyIndex = family,
                .queueCount = 1,
                .pQueuePriorities = &priority,
            });
        }

        std::vector<const char*> extensions {
            VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
            VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        };
        if (vk.surface)
            extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
        VkPhysicalDeviceFeatures features {};
        const VkDeviceCreateInfo info {
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .queueCreateInfoCount = static_cast<uint32_t>(queueInfos.size()),
            .pQueueCreateInfos = queueInfos.data(),
            .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
            .ppEnabledExtensionNames = extensions.data(),
            .pEnabledFeatures = &features,
        };
        vkCheck(vkCreateDevice(vk.physicalDevice, &info, nullptr, &vk.device), "vkCreateDevice");
        vkGetDeviceQueue(vk.device, *families.graphics, 0, &vk.graphicsQueue);
        if (families.present)
            vkGetDeviceQueue(vk.device, *families.present, 0, &vk.presentQueue);
    }

    void createSwapchain(Renderer::VulkanState& vk, GLFWwindow* window)
    {
        VkSurfaceCapabilitiesKHR capabilities {};
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vk.physicalDevice, vk.surface, &capabilities);

        uint32_t formatCount = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(vk.physicalDevice, vk.surface, &formatCount, nullptr);
        std::vector<VkSurfaceFormatKHR> formats(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(vk.physicalDevice, vk.surface, &formatCount, formats.data());

        uint32_t modeCount = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(vk.physicalDevice, vk.surface, &modeCount, nullptr);
        std::vector<VkPresentModeKHR> modes(modeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(vk.physicalDevice, vk.surface, &modeCount, modes.data());

        const VkSurfaceFormatKHR surfaceFormat = chooseSurfaceFormat(formats);
        const VkPresentModeKHR presentMode = choosePresentMode(modes);
        const VkExtent2D extent = chooseExtent(window, capabilities);
        uint32_t imageCount = capabilities.minImageCount + 1;
        if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount)
            imageCount = capabilities.maxImageCount;

        const QueueFamilies families = findQueueFamilies(vk.physicalDevice, vk.surface);
        const uint32_t queueFamilyIndices[] { *families.graphics, *families.present };
        const bool shared = *families.graphics != *families.present;
        const VkSwapchainCreateInfoKHR info {
            .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            .surface = vk.surface,
            .minImageCount = imageCount,
            .imageFormat = surfaceFormat.format,
            .imageColorSpace = surfaceFormat.colorSpace,
            .imageExtent = extent,
            .imageArrayLayers = 1,
            .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            .imageSharingMode = shared ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = shared ? 2U : 0U,
            .pQueueFamilyIndices = shared ? queueFamilyIndices : nullptr,
            .preTransform = capabilities.currentTransform,
            .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            .presentMode = presentMode,
            .clipped = VK_TRUE,
        };
        vkCheck(vkCreateSwapchainKHR(vk.device, &info, nullptr, &vk.swapchain), "vkCreateSwapchainKHR");
        vk.swapchainFormat = surfaceFormat.format;
        vk.swapchainExtent = extent;
        vk.colorFormat = surfaceFormat.format;
        vk.renderExtent = extent;

        vkGetSwapchainImagesKHR(vk.device, vk.swapchain, &imageCount, nullptr);
        vk.swapchainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(vk.device, vk.swapchain, &imageCount, vk.swapchainImages.data());
        vk.swapchainImageViews.resize(imageCount);
        for (std::size_t i = 0; i < vk.swapchainImages.size(); ++i) {
            const VkImageViewCreateInfo viewInfo {
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image = vk.swapchainImages[i],
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = vk.swapchainFormat,
                .components = { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                    VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY },
                .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
            };
            vkCheck(vkCreateImageView(vk.device, &viewInfo, nullptr, &vk.swapchainImageViews[i]), "vkCreateImageView");
        }
    }

    void createRenderPass(Renderer::VulkanState& vk)
    {
        const VkAttachmentDescription colorAttachment {
            .format = vk.colorFormat,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = vk.swapchain ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        };
        const VkAttachmentDescription depthAttachment {
            .format = vk.depthFormat,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };
        const VkAttachmentDescription attachments[] { colorAttachment, depthAttachment };
        const VkAttachmentReference colorRef { .attachment = 0, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
        const VkAttachmentReference depthRef {
            .attachment = 1,
            .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };
        const VkSubpassDescription subpass {
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentCount = 1,
            .pColorAttachments = &colorRef,
            .pDepthStencilAttachment = &depthRef,
        };
        const VkSubpassDependency swapchainDependency {
            .srcSubpass = VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0,
            .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
            .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        };
        const VkSubpassDependency offscreenDependencies[] {
            {
                .srcSubpass = VK_SUBPASS_EXTERNAL,
                .dstSubpass = 0,
                .srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                    | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                .srcAccessMask = 0,
                .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                    | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
            },
            {
                .srcSubpass = 0,
                .dstSubpass = VK_SUBPASS_EXTERNAL,
                .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                    | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT,
                .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                    | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
            },
        };
        const bool offscreen = vk.swapchain == VK_NULL_HANDLE;
        const VkRenderPassCreateInfo info {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .attachmentCount = 2,
            .pAttachments = attachments,
            .subpassCount = 1,
            .pSubpasses = &subpass,
            .dependencyCount = offscreen ? 2U : 1U,
            .pDependencies = offscreen ? offscreenDependencies : &swapchainDependency,
        };
        vkCheck(vkCreateRenderPass(vk.device, &info, nullptr, &vk.renderPass), "vkCreateRenderPass");
    }

} // namespace
namespace {

    void createPipeline(Renderer::VulkanState& vk)
    {
        const VkDescriptorSetLayoutBinding vertexBinding {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        };
        const VkDescriptorSetLayoutBinding instanceBinding {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        };
        const VkDescriptorSetLayoutBinding vertexBindings[] { vertexBinding, instanceBinding };
        const VkDescriptorSetLayoutCreateInfo vertexLayoutInfo {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 2,
            .pBindings = vertexBindings,
        };
        vkCheck(vkCreateDescriptorSetLayout(vk.device, &vertexLayoutInfo, nullptr, &vk.vertexSetLayout),
            "vkCreateDescriptorSetLayout vertices");

        const VkDescriptorSetLayoutBinding paramsBinding {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        };
        const VkDescriptorSetLayoutCreateInfo paramsLayoutInfo {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 1,
            .pBindings = &paramsBinding,
        };
        vkCheck(vkCreateDescriptorSetLayout(vk.device, &paramsLayoutInfo, nullptr, &vk.paramsSetLayout),
            "vkCreateDescriptorSetLayout params");

        const VkDescriptorSetLayout layouts[] { vk.vertexSetLayout, vk.paramsSetLayout };
        const VkPipelineLayoutCreateInfo pipelineLayoutInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 2,
            .pSetLayouts = layouts,
        };
        vkCheck(vkCreatePipelineLayout(vk.device, &pipelineLayoutInfo, nullptr, &vk.pipelineLayout),
            "vkCreatePipelineLayout");

        const VkShaderModule vertexShader
            = createShaderModule(vk.device, readFile(std::filesystem::path(BLOCKLAB_SHADER_DIR) / "mesh_vertex.spv"));
        const VkShaderModule fragmentShader
            = createShaderModule(vk.device, readFile(std::filesystem::path(BLOCKLAB_SHADER_DIR) / "mesh_fragment.spv"));

        const VkPipelineShaderStageCreateInfo stages[] {
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_VERTEX_BIT,
                .module = vertexShader,
                .pName = "meshVertexMain",
            },
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                .module = fragmentShader,
                .pName = "meshFragmentMain",
            },
        };
        const VkPipelineVertexInputStateCreateInfo vertexInput {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        };
        const VkPipelineInputAssemblyStateCreateInfo inputAssembly {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        };
        const VkViewport viewport {
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(vk.renderExtent.width),
            .height = static_cast<float>(vk.renderExtent.height),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        const VkRect2D scissor { .offset = { 0, 0 }, .extent = vk.renderExtent };
        const VkPipelineViewportStateCreateInfo viewportState {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1,
            .pViewports = &viewport,
            .scissorCount = 1,
            .pScissors = &scissor,
        };
        const VkPipelineRasterizationStateCreateInfo rasterizer {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_NONE,
            .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
            .lineWidth = 1.0f,
        };
        const VkPipelineMultisampleStateCreateInfo multisample {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        };
        const VkPipelineDepthStencilStateCreateInfo depthStencil {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .depthTestEnable = VK_TRUE,
            .depthWriteEnable = VK_TRUE,
            .depthCompareOp = VK_COMPARE_OP_LESS,
        };
        const VkPipelineColorBlendAttachmentState colorBlendAttachment {
            .colorWriteMask
            = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
        };
        const VkPipelineColorBlendStateCreateInfo colorBlend {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .attachmentCount = 1,
            .pAttachments = &colorBlendAttachment,
        };
        const VkGraphicsPipelineCreateInfo pipelineInfo {
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = 2,
            .pStages = stages,
            .pVertexInputState = &vertexInput,
            .pInputAssemblyState = &inputAssembly,
            .pViewportState = &viewportState,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisample,
            .pDepthStencilState = &depthStencil,
            .pColorBlendState = &colorBlend,
            .layout = vk.pipelineLayout,
            .renderPass = vk.renderPass,
            .subpass = 0,
        };
        vkCheck(vkCreateGraphicsPipelines(vk.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &vk.pipeline),
            "vkCreateGraphicsPipelines");
        vkDestroyShaderModule(vk.device, fragmentShader, nullptr);
        vkDestroyShaderModule(vk.device, vertexShader, nullptr);
    }

    void createFramebuffers(Renderer::VulkanState& vk)
    {
        if (!vk.swapchain) {
            vk.offscreenFrames.resize(OffscreenFrameCount);
            for (OffscreenFrame& frame : vk.offscreenFrames) {
                frame.color = createColorImage(vk);
                frame.depth = createDepthImage(vk);
                const VkImageView attachments[] { frame.color.view, frame.depth.view };
                const VkFramebufferCreateInfo info {
                    .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                    .renderPass = vk.renderPass,
                    .attachmentCount = 2,
                    .pAttachments = attachments,
                    .width = vk.renderExtent.width,
                    .height = vk.renderExtent.height,
                    .layers = 1,
                };
                vkCheck(vkCreateFramebuffer(vk.device, &info, nullptr, &frame.framebuffer),
                    "vkCreateFramebuffer offscreen");
            }
            return;
        }

        vk.framebuffers.resize(vk.swapchainImageViews.size());
        for (std::size_t i = 0; i < vk.swapchainImageViews.size(); ++i) {
            const VkImageView attachments[] { vk.swapchainImageViews[i], vk.depthImage.view };
            const VkFramebufferCreateInfo info {
                .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                .renderPass = vk.renderPass,
                .attachmentCount = 2,
                .pAttachments = attachments,
                .width = vk.renderExtent.width,
                .height = vk.renderExtent.height,
                .layers = 1,
            };
            vkCheck(vkCreateFramebuffer(vk.device, &info, nullptr, &vk.framebuffers[i]), "vkCreateFramebuffer");
        }
    }

    void createCommandsAndSync(Renderer::VulkanState& vk)
    {
        const VkCommandPoolCreateInfo poolInfo {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = vk.graphicsFamily,
        };
        vkCheck(vkCreateCommandPool(vk.device, &poolInfo, nullptr, &vk.commandPool), "vkCreateCommandPool");
        const uint32_t commandBufferCount = vk.swapchain ? static_cast<uint32_t>(vk.swapchainImages.size())
                                                         : static_cast<uint32_t>(vk.offscreenFrames.size());
        vk.commandBuffers.resize(commandBufferCount);
        const VkCommandBufferAllocateInfo allocInfo {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = vk.commandPool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = static_cast<uint32_t>(vk.commandBuffers.size()),
        };
        vkCheck(vkAllocateCommandBuffers(vk.device, &allocInfo, vk.commandBuffers.data()), "vkAllocateCommandBuffers");

        const VkSemaphoreCreateInfo semaphoreInfo { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        const VkFenceCreateInfo fenceInfo { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT };
        if (vk.swapchain) {
            vkCheck(
                vkCreateSemaphore(vk.device, &semaphoreInfo, nullptr, &vk.imageAvailable), "vkCreateSemaphore image");
            vkCheck(
                vkCreateSemaphore(vk.device, &semaphoreInfo, nullptr, &vk.renderFinished), "vkCreateSemaphore render");
            vkCheck(vkCreateFence(vk.device, &fenceInfo, nullptr, &vk.inFlight), "vkCreateFence");
        } else {
            for (std::size_t i = 0; i < vk.offscreenFrames.size(); ++i) {
                vk.offscreenFrames[i].commandBuffer = vk.commandBuffers[i];
                vkCheck(vkCreateFence(vk.device, &fenceInfo, nullptr, &vk.offscreenFrames[i].fence),
                    "vkCreateFence offscreen");
            }
        }
    }

} // namespace
namespace {

    void createDescriptors(Renderer::VulkanState& vk)
    {
        vk.vertexCapacityBytes = (static_cast<VkDeviceSize>(MaxTerrainVertices) + MaxPigVertices) * sizeof(MeshVertex);
        vk.vertexBuffer = createExportedDeviceBuffer(vk, vk.vertexCapacityBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        vk.instanceBuffer = createHostBuffer(
            vk, sizeof(Renderer::EntityInstance) * MaxEntityInstances, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        if (vk.swapchain)
            vk.paramsBuffer = createHostBuffer(vk, sizeof(Renderer::RenderParams), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        else {
            const VkDeviceSize observationBytes
                = static_cast<VkDeviceSize>(vk.renderExtent.width) * vk.renderExtent.height * 4U;
            vk.observationBuffer
                = createExportedDeviceBuffer(vk, observationBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            for (OffscreenFrame& frame : vk.offscreenFrames) {
                frame.paramsBuffer
                    = createHostBuffer(vk, sizeof(Renderer::RenderParams), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
            }
        }

        const uint32_t paramsSetCount = vk.swapchain ? 1U : static_cast<uint32_t>(vk.offscreenFrames.size());
        const VkDescriptorPoolSize poolSizes[] {
            { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2 },
            { .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = paramsSetCount },
        };
        const VkDescriptorPoolCreateInfo poolInfo {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1U + paramsSetCount,
            .poolSizeCount = 2,
            .pPoolSizes = poolSizes,
        };
        vkCheck(vkCreateDescriptorPool(vk.device, &poolInfo, nullptr, &vk.descriptorPool), "vkCreateDescriptorPool");

        std::vector<VkDescriptorSetLayout> layouts;
        layouts.reserve(1U + paramsSetCount);
        layouts.push_back(vk.vertexSetLayout);
        for (uint32_t i = 0; i < paramsSetCount; ++i)
            layouts.push_back(vk.paramsSetLayout);
        std::vector<VkDescriptorSet> sets(layouts.size());
        const VkDescriptorSetAllocateInfo allocInfo {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = vk.descriptorPool,
            .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
            .pSetLayouts = layouts.data(),
        };
        vkCheck(vkAllocateDescriptorSets(vk.device, &allocInfo, sets.data()), "vkAllocateDescriptorSets");
        vk.vertexDescriptorSet = sets[0];
        if (vk.swapchain)
            vk.paramsDescriptorSet = sets[1];
        else {
            for (std::size_t i = 0; i < vk.offscreenFrames.size(); ++i)
                vk.offscreenFrames[i].paramsDescriptorSet = sets[i + 1U];
        }

        const VkDescriptorBufferInfo vertexInfo {
            .buffer = vk.vertexBuffer.buffer, .offset = 0, .range = vk.vertexBuffer.size
        };
        const VkDescriptorBufferInfo instanceInfo {
            .buffer = vk.instanceBuffer.buffer, .offset = 0, .range = vk.instanceBuffer.size
        };
        const VkWriteDescriptorSet vertexWrite {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = vk.vertexDescriptorSet,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &vertexInfo,
        };
        const VkWriteDescriptorSet instanceWrite {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = vk.vertexDescriptorSet,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &instanceInfo,
        };
        const VkWriteDescriptorSet storageWrites[] { vertexWrite, instanceWrite };
        vkUpdateDescriptorSets(vk.device, 2, storageWrites, 0, nullptr);

        std::vector<VkDescriptorBufferInfo> paramsInfos(paramsSetCount);
        std::vector<VkWriteDescriptorSet> writes(paramsSetCount);
        for (uint32_t i = 0; i < paramsSetCount; ++i) {
            Buffer& paramsBuffer = vk.swapchain ? vk.paramsBuffer : vk.offscreenFrames[i].paramsBuffer;
            VkDescriptorSet paramsSet
                = vk.swapchain ? vk.paramsDescriptorSet : vk.offscreenFrames[i].paramsDescriptorSet;
            paramsInfos[i] = { .buffer = paramsBuffer.buffer, .offset = 0, .range = paramsBuffer.size };
            writes[i] = {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = paramsSet,
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pBufferInfo = &paramsInfos[i],
            };
        }
        vkUpdateDescriptorSets(vk.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    void initVulkan(Renderer::VulkanState& vk, GLFWwindow* window, RenderConfig config)
    {
        createInstance(vk);
        if (window)
            vkCheck(glfwCreateWindowSurface(vk.instance, window, nullptr, &vk.surface), "glfwCreateWindowSurface");
        pickDevice(vk);
        createDevice(vk);
        if (window)
            createSwapchain(vk, window);
        else {
            vk.renderExtent = { static_cast<uint32_t>(config.width), static_cast<uint32_t>(config.height) };
            vk.colorFormat = ColorFormat;
        }
        vk.depthFormat = findSupportedDepthFormat(vk.physicalDevice);
        createRenderPass(vk);
        if (vk.swapchain)
            vk.depthImage = createDepthImage(vk);
        createPipeline(vk);
        createFramebuffers(vk);
        createCommandsAndSync(vk);
        createDescriptors(vk);
    }

    void destroyVulkan(Renderer::VulkanState& vk)
    {
        if (!vk.device)
            return;
        vkDeviceWaitIdle(vk.device);
        for (OffscreenFrame& frame : vk.offscreenFrames) {
            destroyBuffer(vk, frame.paramsBuffer);
            if (frame.fence)
                vkDestroyFence(vk.device, frame.fence, nullptr);
            if (frame.framebuffer)
                vkDestroyFramebuffer(vk.device, frame.framebuffer, nullptr);
            destroyImage(vk, frame.depth);
            destroyImage(vk, frame.color);
        }
        destroyBuffer(vk, vk.paramsBuffer);
        destroyBuffer(vk, vk.observationBuffer);
        destroyBuffer(vk, vk.instanceBuffer);
        destroyBuffer(vk, vk.vertexBuffer);
        if (vk.descriptorPool)
            vkDestroyDescriptorPool(vk.device, vk.descriptorPool, nullptr);
        if (vk.inFlight)
            vkDestroyFence(vk.device, vk.inFlight, nullptr);
        if (vk.renderFinished)
            vkDestroySemaphore(vk.device, vk.renderFinished, nullptr);
        if (vk.imageAvailable)
            vkDestroySemaphore(vk.device, vk.imageAvailable, nullptr);
        if (vk.commandPool)
            vkDestroyCommandPool(vk.device, vk.commandPool, nullptr);
        for (VkFramebuffer framebuffer : vk.framebuffers)
            vkDestroyFramebuffer(vk.device, framebuffer, nullptr);
        if (vk.pipeline)
            vkDestroyPipeline(vk.device, vk.pipeline, nullptr);
        if (vk.pipelineLayout)
            vkDestroyPipelineLayout(vk.device, vk.pipelineLayout, nullptr);
        if (vk.paramsSetLayout)
            vkDestroyDescriptorSetLayout(vk.device, vk.paramsSetLayout, nullptr);
        if (vk.vertexSetLayout)
            vkDestroyDescriptorSetLayout(vk.device, vk.vertexSetLayout, nullptr);
        if (vk.renderPass)
            vkDestroyRenderPass(vk.device, vk.renderPass, nullptr);
        destroyImage(vk, vk.depthImage);
        for (VkImageView view : vk.swapchainImageViews)
            vkDestroyImageView(vk.device, view, nullptr);
        if (vk.swapchain)
            vkDestroySwapchainKHR(vk.device, vk.swapchain, nullptr);
        vkDestroyDevice(vk.device, nullptr);
        vk.device = VK_NULL_HANDLE;
        if (vk.surface)
            vkDestroySurfaceKHR(vk.instance, vk.surface, nullptr);
        if (vk.instance)
            vkDestroyInstance(vk.instance, nullptr);
    }

} // namespace
Renderer::Renderer(RenderConfig config)
    : m_config(config)
    , m_meshBuilder({ .halfExtent = TerrainMeshHalfExtent })
    , m_pigMesh(createPigMesh())
{
    if (m_pigMesh.size() > MaxPigVertices) [[unlikely]]
        fatalError("Pig mesh exceeds reserved Vulkan/CUDA vertex buffer capacity");
    if (!glfwInit()) [[unlikely]]
        fatalError("glfwInit failed");
    if (!glfwVulkanSupported()) [[unlikely]]
        fatalError("GLFW reports Vulkan is not supported");

    if (m_config.visible) {
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
        m_window = glfwCreateWindow(m_config.width, m_config.height, "BlockLab", nullptr, nullptr);
        if (!m_window) [[unlikely]]
            fatalError("glfwCreateWindow failed");
    }

    m_vk = new VulkanState();
    initVulkan(*m_vk, m_window, m_config);
    m_observation = {
        .width = static_cast<int32_t>(m_vk->renderExtent.width),
        .height = static_cast<int32_t>(m_vk->renderExtent.height),
        .channels = 4,
        .device = m_vk->swapchain ? ObservationDevice::VulkanSwapchain : ObservationDevice::VulkanImage,
        .format = ObservationFormat::RGBA8,
        .handle
        = reinterpret_cast<uintptr_t>(m_vk->swapchain ? reinterpret_cast<void*>(m_vk->swapchain)
                                                      : reinterpret_cast<void*>(m_vk->offscreenFrames[0].color.image)),
    };
}

Renderer::~Renderer()
{
    if (m_vk) {
        destroyVulkan(*m_vk);
        delete m_vk;
        m_vk = nullptr;
    }
    if (m_window) {
        glfwDestroyWindow(m_window);
        m_window = nullptr;
    }
    glfwTerminate();
}

bool Renderer::shouldClose() const { return m_window && glfwWindowShouldClose(m_window); }

void Renderer::pollEvents()
{
    if (m_window)
        glfwPollEvents();
}

void Renderer::resize(int32_t width, int32_t height)
{
    m_config.width = std::max(1, width);
    m_config.height = std::max(1, height);
}

void Renderer::setCudaObservationExportEnabled(bool enabled) { m_cudaObservationExportEnabled = enabled; }

void* Renderer::cudaObservationData()
{
    if (!m_vk || m_vk->swapchain || m_vk->offscreenFrames.empty())
        return nullptr;
    synchronizeObservation();
    return m_vk->observationBuffer.cudaPtr;
}

void Renderer::synchronizeObservation()
{
    if (!m_vk || m_vk->swapchain || m_vk->offscreenFrames.empty())
        return;
    OffscreenFrame& frame = m_vk->offscreenFrames[m_vk->lastSubmittedOffscreenFrame];
    vkWaitForFences(m_vk->device, 1, &frame.fence, VK_TRUE, UINT64_MAX);
}

std::size_t Renderer::cudaObservationBytes() const
{
    if (!m_vk || m_vk->swapchain)
        return 0;
    return static_cast<std::size_t>(m_vk->renderExtent.width) * static_cast<std::size_t>(m_vk->renderExtent.height) * 4U;
}

Renderer::RenderParams Renderer::buildRenderParams(const AgentState& agent, const World& world) const
{
    const Vec3 origin = agent.position + Vec3 { 0.0f, EyeHeight, 0.0f };
    const Vec3 forward = cameraForward(agent.yaw, agent.pitch);
    const Vec3 right = glm::normalize(Vec3 { std::cos(agent.yaw), 0.0f, -std::sin(agent.yaw) });
    const Vec3 up = glm::normalize(glm::cross(forward, right));
    // TODO is it practically possible for logicalTimeMs to exceed int32_t max? If so, we should probably wrap it
    // instead of clamping to max.
    assert(world.logicalTimeMs() <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max()));
    return {
        .origin = { origin.x, origin.y, origin.z, 0.0f },
        .forward = { forward.x, forward.y, forward.z, 0.0f },
        .right = { right.x, right.y, right.z, 0.0f },
        .up = { up.x, up.y, up.z, 0.0f },
        .worldOriginAndWidth = { 0, 0, 0, static_cast<int32_t>(m_vk->renderExtent.width) },
        .regionAndHeight = { 0, 0, 0, static_cast<int32_t>(m_vk->renderExtent.height) },
        .frameInfo = { .animationTimeMs = static_cast<int32_t>(world.logicalTimeMs()) },
        .tuning = { 48.0f, Pi / 2.25f, 10.0f, 28.0f },
    };
}

void Renderer::uploadInstances(const World& world)
{
    m_instances.clear();
    m_instances.reserve(world.characters().size() + 1U);
    m_instances.push_back({});
    for (const std::unique_ptr<NPC>& character : world.characters()) {
        if (!character->isAlive())
            continue;
        const CharacterSnapshot snapshot = character->snapshot();
        if (snapshot.kind != CharacterKind::Pig)
            continue;
        const float yaw = std::atan2(snapshot.forward.x, snapshot.forward.z);
        const float speed
            = std::sqrt(snapshot.velocity.x * snapshot.velocity.x + snapshot.velocity.z * snapshot.velocity.z);
        m_instances.push_back({
            .positionAndYaw = { snapshot.position.x, snapshot.position.y, snapshot.position.z, yaw },
            .velocityAndKind = { speed, 0.0f, 0.0f, renderEntityKindId(RenderEntityKind::Pig) },
        });
        if (m_instances.size() >= MaxEntityInstances)
            break;
    }
    m_instanceCount = static_cast<uint32_t>(m_instances.size() - 1U);

    VulkanState& vk = *m_vk;
    void* mapped = nullptr;
    const VkDeviceSize uploadBytes = sizeof(EntityInstance) * m_instances.size();
    vkCheck(vkMapMemory(vk.device, vk.instanceBuffer.memory, 0, uploadBytes, 0, &mapped), "vkMapMemory instances");
    std::memcpy(mapped, m_instances.data(), static_cast<std::size_t>(uploadBytes));
    vkUnmapMemory(vk.device, vk.instanceBuffer.memory);
}

void Renderer::drawFrame(const RenderParams& params)
{
    VulkanState& vk = *m_vk;
    OffscreenFrame* offscreenFrame = nullptr;
    VkFence submitFence = vk.inFlight;

    uint32_t imageIndex = 0;
    if (vk.swapchain) {
        if (!m_config.present)
            return;
        vkWaitForFences(vk.device, 1, &vk.inFlight, VK_TRUE, UINT64_MAX);
        vkResetFences(vk.device, 1, &vk.inFlight);
        VkResult acquire = vkAcquireNextImageKHR(
            vk.device, vk.swapchain, UINT64_MAX, vk.imageAvailable, VK_NULL_HANDLE, &imageIndex);
        if (acquire != VK_SUCCESS)
            return;
    } else {
        offscreenFrame = &vk.offscreenFrames[vk.nextOffscreenFrame];
        vkWaitForFences(vk.device, 1, &offscreenFrame->fence, VK_TRUE, UINT64_MAX);
        vkResetFences(vk.device, 1, &offscreenFrame->fence);
        submitFence = offscreenFrame->fence;
        vk.lastSubmittedOffscreenFrame = vk.nextOffscreenFrame;
        vk.nextOffscreenFrame = (vk.nextOffscreenFrame + 1U) % vk.offscreenFrames.size();
        m_observation.handle = reinterpret_cast<uintptr_t>(offscreenFrame->color.image);
    }

    void* mapped = nullptr;
    Buffer& paramsBuffer = offscreenFrame ? offscreenFrame->paramsBuffer : vk.paramsBuffer;
    vkCheck(vkMapMemory(vk.device, paramsBuffer.memory, 0, sizeof(params), 0, &mapped), "vkMapMemory params");
    std::memcpy(mapped, &params, sizeof(params));
    vkUnmapMemory(vk.device, paramsBuffer.memory);

    VkCommandBuffer commandBuffer = offscreenFrame ? offscreenFrame->commandBuffer : vk.commandBuffers[imageIndex];
    vkResetCommandBuffer(commandBuffer, 0);
    const VkCommandBufferBeginInfo beginInfo { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vkCheck(vkBeginCommandBuffer(commandBuffer, &beginInfo), "vkBeginCommandBuffer");
    const VkClearValue clearValues[] {
        { .color = { { 0.42f, 0.64f, 0.86f, 1.0f } } },
        { .depthStencil = { 1.0f, 0 } },
    };
    const VkRenderPassBeginInfo renderPassInfo {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = vk.renderPass,
        .framebuffer = offscreenFrame ? offscreenFrame->framebuffer : vk.framebuffers[imageIndex],
        .renderArea = { { 0, 0 }, vk.renderExtent },
        .clearValueCount = 2,
        .pClearValues = clearValues,
    };
    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk.pipeline);
    const VkDescriptorSet sets[] { vk.vertexDescriptorSet,
        offscreenFrame ? offscreenFrame->paramsDescriptorSet : vk.paramsDescriptorSet };
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk.pipelineLayout, 0, 2, sets, 0, nullptr);
    if (m_terrainVertexCount > 0)
        vkCmdDraw(commandBuffer, m_terrainVertexCount, 1, 0, 0);
    if (m_pigVertexCount > 0 && m_instanceCount > 0)
        vkCmdDraw(commandBuffer, m_pigVertexCount, m_instanceCount, m_pigVertexOffset, 1);
    vkCmdEndRenderPass(commandBuffer);
    if (offscreenFrame && m_cudaObservationExportEnabled) {
        const VkBufferImageCopy copyRegion {
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0,
                .layerCount = 1 },
            .imageOffset = { 0, 0, 0 },
            .imageExtent = { vk.renderExtent.width, vk.renderExtent.height, 1 },
        };
        vkCmdCopyImageToBuffer(commandBuffer, offscreenFrame->color.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            vk.observationBuffer.buffer, 1, &copyRegion);
    }
    vkCheck(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");

    const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    const VkSubmitInfo submitInfo {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = vk.swapchain ? 1U : 0U,
        .pWaitSemaphores = vk.swapchain ? &vk.imageAvailable : nullptr,
        .pWaitDstStageMask = vk.swapchain ? &waitStage : nullptr,
        .commandBufferCount = 1,
        .pCommandBuffers = &commandBuffer,
        .signalSemaphoreCount = vk.swapchain ? 1U : 0U,
        .pSignalSemaphores = vk.swapchain ? &vk.renderFinished : nullptr,
    };
    vkCheck(vkQueueSubmit(vk.graphicsQueue, 1, &submitInfo, submitFence), "vkQueueSubmit");

    if (vk.swapchain) {
        const VkPresentInfoKHR presentInfo {
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &vk.renderFinished,
            .swapchainCount = 1,
            .pSwapchains = &vk.swapchain,
            .pImageIndices = &imageIndex,
        };
        vkQueuePresentKHR(vk.presentQueue, &presentInfo);
    }
}

Observation Renderer::renderObservation(const World& world, const AgentState& agent)
{
    const IVec3 agentBlock { floorToInt32(agent.position.x), floorToInt32(agent.position.y), floorToInt32(agent.position.z) };
    constexpr int32_t MeshCacheStride = 12;
    const IVec3 meshDelta = glm::abs(agentBlock - m_lastMeshCenter);
    const bool agentLeftMeshCache
        = meshDelta.x > MeshCacheStride || meshDelta.y > MeshCacheStride || meshDelta.z > MeshCacheStride;
    if (m_lastWorldVersion != world.version() || agentLeftMeshCache) {
        if (!m_vk->swapchain) {
            for (OffscreenFrame& frame : m_vk->offscreenFrames)
                vkWaitForFences(m_vk->device, 1, &frame.fence, VK_TRUE, UINT64_MAX);
        } else
            vkWaitForFences(m_vk->device, 1, &m_vk->inFlight, VK_TRUE, UINT64_MAX);

        MeshVertex* const vertices = static_cast<MeshVertex*>(m_vk->vertexBuffer.cudaPtr);
        m_terrainVertexCount = m_meshBuilder.rebuild(world, agent, vertices, MaxTerrainVertices);
        m_pigVertexOffset = m_terrainVertexCount;
        m_pigVertexCount = static_cast<uint32_t>(m_pigMesh.size());
        cudaCheck(cudaMemcpy(vertices + m_pigVertexOffset, m_pigMesh.data(), sizeof(MeshVertex) * m_pigMesh.size(),
                      cudaMemcpyHostToDevice),
            "cudaMemcpy pig mesh to Vulkan vertex buffer");
        cudaCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize mesh upload");
        m_lastWorldVersion = world.version();
        m_lastMeshCenter = agentBlock;
    }
    uploadInstances(world);

    drawFrame(buildRenderParams(agent, world));
    ++m_observation.version;
    return m_observation;
}

} // namespace blocklab
