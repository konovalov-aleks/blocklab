#include "blocklab/Renderer.h"

#include "blocklab/CudaHelpers.h"
#include "blocklab/CudaObservation.h"
#include "blocklab/Error.h"
#include "blocklab/meshes/PigMesh.h"

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
#include <utility>
#include <vector>

namespace blocklab {
namespace {

    constexpr float EyeHeight = 1.62f;
    constexpr VkFormat ColorFormat = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr uint32_t OffscreenFrameCount = 2;
    constexpr uint32_t MaxEntityInstances = 256;
    constexpr int32_t TerrainMeshHalfExtent = 32;
    constexpr uint32_t TerrainMeshExtent = TerrainMeshHalfExtent * 2;
    constexpr size_t MaxTerrainVoxels = static_cast<size_t>(TerrainMeshExtent * Chunk::SizeY * TerrainMeshExtent);

    Vec3 cameraForward(float yaw, float pitch)
    {
        const float pitchCos = std::cos(pitch);
        return glm::normalize(Vec3 { std::sin(yaw) * pitchCos, std::sin(pitch), std::cos(yaw) * pitchCos });
    }

    std::vector<char> readFile(const std::filesystem::path path)
    {
        std::ifstream file(path, std::ios::ate | std::ios::binary);
        if (!file) [[unlikely]]
            fatalError("Failed to open ", path);
        const std::streamsize size = file.tellg();
        std::vector<char> bytes(static_cast<std::size_t>(size));
        file.seekg(0);
        file.read(bytes.data(), size);
        return bytes;
    }

    void vkCheck(VkResult result, const char* operation)
    {
        if (result != VK_SUCCESS) [[unlikely]]
            fatalError(operation, " failed with VkResult ", result);
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

    class ExternalSemaphore {
    public:
        ExternalSemaphore() = default;
        ExternalSemaphore(VkDevice, std::uint64_t initialValue = 0);

        ~ExternalSemaphore();

        ExternalSemaphore(const ExternalSemaphore&) = delete;
        ExternalSemaphore& operator=(const ExternalSemaphore&) = delete;

        ExternalSemaphore(ExternalSemaphore&& other) noexcept
            : m_device(std::exchange(other.m_device, VK_NULL_HANDLE))
            , m_semaphore(std::exchange(other.m_semaphore, VK_NULL_HANDLE))
            , m_cudaSemaphore(std::exchange(other.m_cudaSemaphore, nullptr))
        {
        }

        ExternalSemaphore& operator=(ExternalSemaphore&& other) noexcept
        {
            if (this == &other)
                return *this;

            reset();
            m_device = std::exchange(other.m_device, VK_NULL_HANDLE);
            m_semaphore = std::exchange(other.m_semaphore, VK_NULL_HANDLE);
            m_cudaSemaphore = std::exchange(other.m_cudaSemaphore, nullptr);
            return *this;
        }

        void wait(cudaStream_t, uint64_t value);

        VkSemaphore vulkanSemaphore() const { return m_semaphore; }
        cudaExternalSemaphore_t cudaSemaphore() const { return m_cudaSemaphore; }

    private:
        void reset();

        VkDevice m_device = VK_NULL_HANDLE;
        VkSemaphore m_semaphore = VK_NULL_HANDLE;
        cudaExternalSemaphore_t m_cudaSemaphore = nullptr;
    };

    struct OffscreenFrame {
        static constexpr std::uint64_t NO_OBSERVATION_VERSION = 0;

        Image color;
        Image depth;
        VkFramebuffer framebuffer = VK_NULL_HANDLE;
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        Buffer paramsBuffer;
        Buffer observationBuffer;
        float* observationTensor = nullptr;
        bool observationTensorValid = false;

        CudaFuture<void> conversionTaskFuture;
        ExternalSemaphore frameCompletionSemaphore;
        std::uint64_t observationVersion = NO_OBSERVATION_VERSION;
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

    VkDescriptorSetLayout drawResourceSetLayout = VK_NULL_HANDLE;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;

    VkPipeline meshPipeline = VK_NULL_HANDLE;
    VkPipeline voxelPipeline = VK_NULL_HANDLE;

    std::vector<VkFramebuffer> framebuffers;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;

    VkDescriptorSet terrainDescriptorSet = VK_NULL_HANDLE;
    VkDescriptorSet pigDescriptorSet = VK_NULL_HANDLE;

    Buffer terrainHeaderBuffer;
    Buffer terrainVoxelBuffer;
    Buffer pigVertexBuffer;
    Buffer instanceBuffer;
    Buffer paramsBuffer;

    VkSemaphore imageAvailable = VK_NULL_HANDLE;
    VkSemaphore renderFinished = VK_NULL_HANDLE;
    VkFence inFlight = VK_NULL_HANDLE;
    VkDeviceSize terrainVoxelBufferCapacityBytes = 0;
    uint32_t batchSize = 1;
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

    Buffer createDeviceBuffer(Renderer::VulkanState& vk, VkDeviceSize size, VkBufferUsageFlags usage)
    {
        return createBuffer(vk, size, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
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

    ExternalSemaphore::ExternalSemaphore(VkDevice device, std::uint64_t initialValue)
        : m_device(device)
    {
        const VkExportSemaphoreCreateInfo exportInfo {
            .sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
            .handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT,
        };
        VkSemaphoreTypeCreateInfo typeCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
            .pNext = &exportInfo,
            .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
            .initialValue = initialValue,
        };
        const VkSemaphoreCreateInfo semaphoreInfo {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .pNext = &typeCreateInfo,
        };
        vkCheck(vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &m_semaphore), "vkCreateSemaphore external");

        auto vkGetSemaphoreFdKHRFn
            = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(vkGetDeviceProcAddr(m_device, "vkGetSemaphoreFdKHR"));
        if (!vkGetSemaphoreFdKHRFn) [[unlikely]]
            fatalError("vkGetSemaphoreFdKHR is unavailable");

        const VkSemaphoreGetFdInfoKHR fdInfo {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
            .semaphore = m_semaphore,
            .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT,
        };
        int fd = -1;
        vkCheck(vkGetSemaphoreFdKHRFn(m_device, &fdInfo, &fd), "vkGetSemaphoreFdKHR");

        cudaExternalSemaphoreHandleDesc externalSemaphoreDesc {};
        externalSemaphoreDesc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
        externalSemaphoreDesc.handle.fd = fd;
        cudaCheck(cudaImportExternalSemaphore(&m_cudaSemaphore, &externalSemaphoreDesc),
            "cudaImportExternalSemaphore render");
    }

    ExternalSemaphore::~ExternalSemaphore() { reset(); }

    void ExternalSemaphore::wait(cudaStream_t stream, uint64_t value)
    {
        if (!m_cudaSemaphore) [[unlikely]]
            fatalError("External semaphore is not initialized");

        cudaExternalSemaphoreWaitParams waitParams {};
        waitParams.params.fence.value = value;
        cudaCheck(cudaWaitExternalSemaphoresAsync(&m_cudaSemaphore, &waitParams, 1, stream),
            "cudaWaitExternalSemaphoresAsync render complete");
    }

    void ExternalSemaphore::reset()
    {
        if (m_cudaSemaphore)
            cudaCheck(cudaDestroyExternalSemaphore(m_cudaSemaphore), "cudaDestroyExternalSemaphore");
        if (m_semaphore)
            vkDestroySemaphore(m_device, m_semaphore, nullptr);
        m_device = VK_NULL_HANDLE;
        m_semaphore = VK_NULL_HANDLE;
        m_cudaSemaphore = nullptr;
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

    void copyBuffer(Renderer::VulkanState& vk, VkBuffer source, VkBuffer destination, VkDeviceSize size)
    {
        const VkCommandBufferAllocateInfo allocInfo {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = vk.commandPool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        vkCheck(vkAllocateCommandBuffers(vk.device, &allocInfo, &commandBuffer), "vkAllocateCommandBuffers copy");

        const VkCommandBufferBeginInfo beginInfo {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        vkCheck(vkBeginCommandBuffer(commandBuffer, &beginInfo), "vkBeginCommandBuffer copy");
        const VkBufferCopy copyRegion {
            .srcOffset = 0,
            .dstOffset = 0,
            .size = size,
        };
        vkCmdCopyBuffer(commandBuffer, source, destination, 1, &copyRegion);
        vkCheck(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer copy");

        const VkSubmitInfo submitInfo {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &commandBuffer,
        };
        vkCheck(vkQueueSubmit(vk.graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE), "vkQueueSubmit copy");
        vkCheck(vkQueueWaitIdle(vk.graphicsQueue), "vkQueueWaitIdle copy");
        vkFreeCommandBuffers(vk.device, vk.commandPool, 1, &commandBuffer);
    }

    void uploadDeviceBuffer(Renderer::VulkanState& vk, Buffer& destination, const void* data, VkDeviceSize size)
    {
        Buffer staging = createHostBuffer(vk, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        void* mapped = nullptr;
        vkCheck(vkMapMemory(vk.device, staging.memory, 0, size, 0, &mapped), "vkMapMemory staging upload");
        std::memcpy(mapped, data, static_cast<std::size_t>(size));
        vkUnmapMemory(vk.device, staging.memory);

        copyBuffer(vk, staging.buffer, destination.buffer, size);
        destroyBuffer(vk, staging);
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
        const uint32_t layers = vk.swapchain ? 1U : vk.batchSize;
        const VkImageCreateInfo imageInfo {
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = vk.depthFormat,
            .extent = { vk.renderExtent.width, vk.renderExtent.height, 1 },
            .mipLevels = 1,
            .arrayLayers = layers,
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
            .viewType = layers == 1U ? VK_IMAGE_VIEW_TYPE_2D : VK_IMAGE_VIEW_TYPE_2D_ARRAY,
            .format = vk.depthFormat,
            .subresourceRange = { static_cast<VkImageAspectFlags>(VK_IMAGE_ASPECT_DEPTH_BIT
                                      | (hasStencilComponent(vk.depthFormat) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0)),
                0, 1, 0, layers },
        };
        vkCheck(vkCreateImageView(vk.device, &viewInfo, nullptr, &image.view), "vkCreateImageView depth");
        return image;
    }

    Image createColorImage(Renderer::VulkanState& vk)
    {
        Image image;
        const uint32_t layers = vk.swapchain ? 1U : vk.batchSize;
        const VkImageCreateInfo imageInfo {
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = vk.colorFormat,
            .extent = { vk.renderExtent.width, vk.renderExtent.height, 1 },
            .mipLevels = 1,
            .arrayLayers = layers,
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
            .viewType = layers == 1U ? VK_IMAGE_VIEW_TYPE_2D : VK_IMAGE_VIEW_TYPE_2D_ARRAY,
            .format = vk.colorFormat,
            .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers },
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
            const bool hasExternalSemaphore = hasDeviceExtension(device, VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME)
                && hasDeviceExtension(device, VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
            const bool hasShaderLayer = hasDeviceExtension(device, VK_EXT_SHADER_VIEWPORT_INDEX_LAYER_EXTENSION_NAME);
            VkPhysicalDeviceVulkan12Features features12 {
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
            };
            VkPhysicalDeviceFeatures2 features {
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                .pNext = &features12,
            };
            vkGetPhysicalDeviceFeatures2(device, &features);
            const bool hasShaderOutputLayer = features12.shaderOutputLayer == VK_TRUE;
            const bool hasShaderOutputViewportIndex = features12.shaderOutputViewportIndex == VK_TRUE;
            if (families.complete(vk.surface != VK_NULL_HANDLE) && hasSwapchain && hasExternalMemory
                && hasExternalSemaphore && hasShaderLayer && hasShaderOutputLayer && hasShaderOutputViewportIndex) {
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
            VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
            VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
            VK_EXT_SHADER_VIEWPORT_INDEX_LAYER_EXTENSION_NAME,
        };
        if (vk.surface)
            extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
        VkPhysicalDeviceVulkan12Features features12 {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
            .timelineSemaphore = VK_TRUE,
            .shaderOutputViewportIndex = VK_TRUE,
            .shaderOutputLayer = VK_TRUE,
        };
        VkPhysicalDeviceFeatures features {};
        const VkDeviceCreateInfo info {
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext = &features12,
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
        const VkDescriptorSetLayoutBinding paramsBinding {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        };
        const VkDescriptorSetLayoutBinding vertexBindings[] { vertexBinding, instanceBinding, paramsBinding };
        const VkDescriptorSetLayoutCreateInfo vertexLayoutInfo {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 3,
            .pBindings = vertexBindings,
        };
        vkCheck(vkCreateDescriptorSetLayout(vk.device, &vertexLayoutInfo, nullptr, &vk.drawResourceSetLayout),
            "vkCreateDescriptorSetLayout vertices");

        const VkPushConstantRange pushConstantRange {
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
            .offset = 0,
            .size = sizeof(Renderer::DrawPushConstants),
        };
        const VkDescriptorSetLayout layouts[] { vk.drawResourceSetLayout };
        const VkPipelineLayoutCreateInfo pipelineLayoutInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = layouts,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pushConstantRange,
        };
        vkCheck(vkCreatePipelineLayout(vk.device, &pipelineLayoutInfo, nullptr, &vk.pipelineLayout),
            "vkCreatePipelineLayout");

        const VkShaderModule voxelVertexShader
            = createShaderModule(vk.device, readFile(std::filesystem::path(BLOCKLAB_SHADER_DIR) / "voxel_vertex.spv"));
        const VkShaderModule meshVertexShader
            = createShaderModule(vk.device, readFile(std::filesystem::path(BLOCKLAB_SHADER_DIR) / "mesh_vertex.spv"));
        const VkShaderModule fragmentShader
            = createShaderModule(vk.device, readFile(std::filesystem::path(BLOCKLAB_SHADER_DIR) / "mesh_fragment.spv"));

        const VkPipelineShaderStageCreateInfo meshStages[] {
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_VERTEX_BIT,
                .module = meshVertexShader,
                .pName = "meshVertexMain",
            },
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                .module = fragmentShader,
                .pName = "meshFragmentMain",
            },
        };

        const VkPipelineShaderStageCreateInfo voxelStages[] {
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_VERTEX_BIT,
                .module = voxelVertexShader,
                .pName = "voxelVertexMain",
            },
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                .module = fragmentShader,
                .pName = "meshFragmentMain",
            },
        };

        const VkPipelineVertexInputStateCreateInfo emptyVertexInput {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        };
        const VkPipelineInputAssemblyStateCreateInfo triangleInputAssembly {
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
        const VkGraphicsPipelineCreateInfo pipelineInfo[]
            = { {
                    // mesh drawing pipeline
                    .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                    .stageCount = std::size(meshStages),
                    .pStages = meshStages,
                    .pVertexInputState = &emptyVertexInput,
                    .pInputAssemblyState = &triangleInputAssembly,
                    .pViewportState = &viewportState,
                    .pRasterizationState = &rasterizer,
                    .pMultisampleState = &multisample,
                    .pDepthStencilState = &depthStencil,
                    .pColorBlendState = &colorBlend,
                    .layout = vk.pipelineLayout,
                    .renderPass = vk.renderPass,
                    .subpass = 0,
                },
                  {
                      // voxel drawing pipeline
                      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                      .stageCount = std::size(voxelStages),
                      .pStages = voxelStages,
                      .pVertexInputState = &emptyVertexInput,
                      .pInputAssemblyState = &triangleInputAssembly,
                      .pViewportState = &viewportState,
                      .pRasterizationState = &rasterizer,
                      .pMultisampleState = &multisample,
                      .pDepthStencilState = &depthStencil,
                      .pColorBlendState = &colorBlend,
                      .layout = vk.pipelineLayout,
                      .renderPass = vk.renderPass,
                      .subpass = 0,
                  } };

        VkPipeline pipelines[2];
        vkCheck(vkCreateGraphicsPipelines(
                    vk.device, VK_NULL_HANDLE, std::size(pipelineInfo), pipelineInfo, nullptr, pipelines),
            "vkCreateGraphicsPipelines");
        vk.meshPipeline = pipelines[0];
        vk.voxelPipeline = pipelines[1];

        vkDestroyShaderModule(vk.device, voxelVertexShader, nullptr);
        vkDestroyShaderModule(vk.device, fragmentShader, nullptr);
        vkDestroyShaderModule(vk.device, meshVertexShader, nullptr);
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
                    .layers = vk.batchSize,
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

        const VkFenceCreateInfo fenceInfo { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT };
        if (vk.swapchain) {
            const VkSemaphoreCreateInfo semaphoreInfo { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
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
                vk.offscreenFrames[i].frameCompletionSemaphore = { vk.device };
            }
        }
    }

    void createDescriptors(Renderer::VulkanState& vk)
    {
        const VkDeviceSize terrainHeadersSizeBytes = sizeof(TerrainHeader) * vk.batchSize;
        vk.terrainHeaderBuffer
            = createExportedDeviceBuffer(vk, terrainHeadersSizeBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        vk.terrainVoxelBufferCapacityBytes = static_cast<VkDeviceSize>(MaxTerrainVoxels) * vk.batchSize * VoxelSize;
        vk.terrainVoxelBuffer
            = createExportedDeviceBuffer(vk, vk.terrainVoxelBufferCapacityBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        vk.pigVertexBuffer
            = createDeviceBuffer(vk, static_cast<VkDeviceSize>(PigMesh::verticesCount()) * sizeof(MeshVertex),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        vk.instanceBuffer = createHostBuffer(vk, sizeof(Renderer::EntityInstance) * MaxEntityInstances * vk.batchSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        vk.paramsBuffer
            = createHostBuffer(vk, sizeof(Renderer::RenderParams) * vk.batchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        if (!vk.swapchain) {
            const VkDeviceSize observationBytes
                = static_cast<VkDeviceSize>(vk.renderExtent.width) * vk.renderExtent.height * 4U * vk.batchSize;
            for (OffscreenFrame& frame : vk.offscreenFrames) {
                frame.observationBuffer
                    = createExportedDeviceBuffer(vk, observationBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
                const std::size_t tensorBytes = static_cast<std::size_t>(vk.renderExtent.width)
                    * static_cast<std::size_t>(vk.renderExtent.height) * 3U * vk.batchSize * sizeof(float);
                cudaCheck(cudaMalloc(&frame.observationTensor, tensorBytes), "cudaMalloc observation tensor");
            }
        }

        const VkDescriptorPoolSize poolSizes[] {
            { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 6 },
        };
        const VkDescriptorPoolCreateInfo poolInfo {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 2U,
            .poolSizeCount = 1,
            .pPoolSizes = poolSizes,
        };
        vkCheck(vkCreateDescriptorPool(vk.device, &poolInfo, nullptr, &vk.descriptorPool), "vkCreateDescriptorPool");

        const VkDescriptorSetLayout layouts[] { vk.drawResourceSetLayout, vk.drawResourceSetLayout };
        VkDescriptorSet descriptorSets[2] {};
        const VkDescriptorSetAllocateInfo allocInfo {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = vk.descriptorPool,
            .descriptorSetCount = 2,
            .pSetLayouts = layouts,
        };
        vkCheck(vkAllocateDescriptorSets(vk.device, &allocInfo, descriptorSets), "vkAllocateDescriptorSets");
        vk.terrainDescriptorSet = descriptorSets[0];
        vk.pigDescriptorSet = descriptorSets[1];

        const VkDescriptorBufferInfo terrainHeaderInfo {
            .buffer = vk.terrainHeaderBuffer.buffer, .offset = 0, .range = vk.terrainHeaderBuffer.size
        };
        const VkDescriptorBufferInfo voxelInfo {
            .buffer = vk.terrainVoxelBuffer.buffer, .offset = 0, .range = vk.terrainVoxelBuffer.size
        };
        const VkDescriptorBufferInfo pigVertexInfo {
            .buffer = vk.pigVertexBuffer.buffer, .offset = 0, .range = vk.pigVertexBuffer.size
        };
        const VkDescriptorBufferInfo instanceInfo {
            .buffer = vk.instanceBuffer.buffer, .offset = 0, .range = vk.instanceBuffer.size
        };
        const VkDescriptorBufferInfo paramsInfo {
            .buffer = vk.paramsBuffer.buffer, .offset = 0, .range = vk.paramsBuffer.size
        };

        const std::array<VkWriteDescriptorSet, 3> terrainWrites = {
            VkWriteDescriptorSet {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = vk.terrainDescriptorSet,
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &terrainHeaderInfo,
            },
            VkWriteDescriptorSet {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = vk.terrainDescriptorSet,
                .dstBinding = 1,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &voxelInfo,
            },
            VkWriteDescriptorSet {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = vk.terrainDescriptorSet,
                .dstBinding = 2,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &paramsInfo,
            },
        };

        const std::array<VkWriteDescriptorSet, 3> pigWrites = {
            VkWriteDescriptorSet {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = vk.pigDescriptorSet,
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &pigVertexInfo,
            },
            VkWriteDescriptorSet {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = vk.pigDescriptorSet,
                .dstBinding = 1,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &instanceInfo,
            },
            VkWriteDescriptorSet {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = vk.pigDescriptorSet,
                .dstBinding = 2,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &paramsInfo,
            },
        };
        vkUpdateDescriptorSets(
            vk.device, static_cast<uint32_t>(terrainWrites.size()), terrainWrites.data(), 0, nullptr);
        vkUpdateDescriptorSets(vk.device, static_cast<uint32_t>(pigWrites.size()), pigWrites.data(), 0, nullptr);
    }

    void initVulkan(Renderer::VulkanState& vk, GLFWwindow* window, RenderConfig config)
    {
        vk.batchSize = std::max(1U, config.batchSize);
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
            if (frame.observationTensor)
                cudaCheck(cudaFree(frame.observationTensor), "cudaFree observation tensor");
            frame.frameCompletionSemaphore = {};
            destroyBuffer(vk, frame.observationBuffer);
            destroyBuffer(vk, frame.paramsBuffer);
            if (frame.fence)
                vkDestroyFence(vk.device, frame.fence, nullptr);
            if (frame.framebuffer)
                vkDestroyFramebuffer(vk.device, frame.framebuffer, nullptr);
            destroyImage(vk, frame.depth);
            destroyImage(vk, frame.color);
        }
        destroyBuffer(vk, vk.paramsBuffer);
        destroyBuffer(vk, vk.instanceBuffer);
        destroyBuffer(vk, vk.pigVertexBuffer);
        destroyBuffer(vk, vk.terrainHeaderBuffer);
        destroyBuffer(vk, vk.terrainVoxelBuffer);
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

        if (vk.meshPipeline)
            vkDestroyPipeline(vk.device, vk.meshPipeline, nullptr);
        if (vk.voxelPipeline)
            vkDestroyPipeline(vk.device, vk.voxelPipeline, nullptr);

        if (vk.pipelineLayout)
            vkDestroyPipelineLayout(vk.device, vk.pipelineLayout, nullptr);
        if (vk.drawResourceSetLayout)
            vkDestroyDescriptorSetLayout(vk.device, vk.drawResourceSetLayout, nullptr);

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
    , m_worldGenerator({ .halfExtent = TerrainMeshHalfExtent })
{
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
    initializeBatchData();
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

std::size_t Renderer::lastObservationFrameIndex(std::size_t slot) const
{
    if (!m_vk || m_vk->swapchain || m_vk->offscreenFrames.empty())
        return 0;
    if (slot >= m_batchSize) [[unlikely]]
        fatalError("Invalid render slot: ", slot);
    return m_slots[slot].lastObservationFrame;
}

void* Renderer::cudaObservationTensorData(std::size_t frameIndex, uintptr_t streamHandle)
{
    if (!m_vk || m_vk->swapchain || m_vk->offscreenFrames.empty())
        return nullptr;
    if (frameIndex >= m_vk->offscreenFrames.size()) [[unlikely]]
        fatalError("Invalid observation frame index: ", frameIndex);

    OffscreenFrame& frame = m_vk->offscreenFrames[frameIndex];
    assert(frame.observationVersion != OffscreenFrame::NO_OBSERVATION_VERSION);
    auto stream = reinterpret_cast<cudaStream_t>(streamHandle);
    frame.conversionTaskFuture.enqueueGPUWait(stream);
    frame.frameCompletionSemaphore.wait(stream, frame.observationVersion);
    convertRgba8ToFloatNchw(frame.observationBuffer.cudaPtr, frame.observationTensor, m_vk->batchSize,
        m_vk->renderExtent.width, m_vk->renderExtent.height, streamHandle);
    frame.conversionTaskFuture = { stream };
    frame.observationTensorValid = true;
    return frame.observationTensor;
}

std::size_t Renderer::cudaObservationTensorBytes() const
{
    if (!m_vk || m_vk->swapchain)
        return 0;
    return static_cast<std::size_t>(m_vk->renderExtent.width) * static_cast<std::size_t>(m_vk->renderExtent.height) * 3U
        * m_vk->batchSize * sizeof(float);
}

void Renderer::initializeBatchData()
{
    if (!m_vk)
        return;
    m_batchSize = m_vk->batchSize;
    m_slots = std::make_unique<RenderSlot[]>(m_batchSize);
    m_renderParams = std::make_unique<RenderParams[]>(m_batchSize);
    m_observation.reset(m_vk->renderExtent.width, m_vk->renderExtent.height, 3U,
        m_vk->swapchain ? ObservationDevice::VulkanSwapchain : ObservationDevice::Cuda,
        m_vk->swapchain ? ObservationFormat::RGBA8 : ObservationFormat::FloatNCHW, m_batchSize);
    m_observation.setVersion(m_observationVersion);
    if (!m_pigMeshUploaded) {
        PigMesh pigMeshGenerator;
        const std::span<MeshVertex> pigMesh = pigMeshGenerator.generate();
        m_pigMeshVertexCount = static_cast<uint32_t>(pigMesh.size());
        uploadDeviceBuffer(*m_vk, m_vk->pigVertexBuffer, pigMesh.data(), sizeof(MeshVertex) * pigMesh.size());
        m_pigMeshUploaded = true;
    }
    for (uint32_t i = 0; i < m_batchSize; ++i) {
        RenderSlot& slot = m_slots[i];
        slot.terrainVoxelOffset = i * MaxTerrainVoxels;
        slot.pigVertexCount = m_pigMeshVertexCount;
        slot.instanceOffset = i * MaxEntityInstances;
        m_observation.setSlot(i,
            reinterpret_cast<uintptr_t>(m_vk->swapchain
                    ? reinterpret_cast<void*>(m_vk->swapchain)
                    : reinterpret_cast<void*>(m_vk->offscreenFrames[0].color.image)));
    }
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

void Renderer::uploadInstances(std::size_t slotIndex, const World& world)
{
    RenderSlot& slot = m_slots[slotIndex];
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
        m_instances.push_back({
            .positionAndYaw = { snapshot.position.x, snapshot.position.y, snapshot.position.z, yaw },
            .velocityAndKind = { snapshot.velocity.x, snapshot.velocity.y, snapshot.velocity.z,
                renderEntityKindId(RenderEntityKind::Pig) },
        });
        if (m_instances.size() >= MaxEntityInstances)
            break;
    }
    slot.instanceCount = static_cast<uint32_t>(m_instances.size() - 1U);

    VulkanState& vk = *m_vk;
    void* mapped = nullptr;
    const VkDeviceSize uploadBytes = sizeof(EntityInstance) * m_instances.size();
    const VkDeviceSize uploadOffset = sizeof(EntityInstance) * slot.instanceOffset;
    vkCheck(vkMapMemory(vk.device, vk.instanceBuffer.memory, uploadOffset, uploadBytes, 0, &mapped),
        "vkMapMemory instances");
    std::memcpy(mapped, m_instances.data(), static_cast<std::size_t>(uploadBytes));
    vkUnmapMemory(vk.device, vk.instanceBuffer.memory);
}

void Renderer::drawFrame()
{
    VulkanState& vk = *m_vk;
    OffscreenFrame* offscreenFrame = nullptr;
    VkFence submitFence = VK_NULL_HANDLE;

    uint32_t imageIndex = 0;
    if (vk.swapchain) {
        if (!m_config.present)
            return;
        vkWaitForFences(vk.device, 1, &vk.inFlight, VK_TRUE, UINT64_MAX);
        vkResetFences(vk.device, 1, &vk.inFlight);
        submitFence = vk.inFlight;
        VkResult acquire = vkAcquireNextImageKHR(
            vk.device, vk.swapchain, UINT64_MAX, vk.imageAvailable, VK_NULL_HANDLE, &imageIndex);
        if (acquire != VK_SUCCESS)
            return;
    } else {
        offscreenFrame = &vk.offscreenFrames[vk.nextOffscreenFrame];

        // The observation is valid until step() or reset() is called.
        // So, it's not mandatory to wait until previous observation conversion is completed
        // TODO But for better stability and API simplification it's better to implement a GPU-side fence for that.

        submitFence = offscreenFrame->fence;
        vkWaitForFences(vk.device, 1, &offscreenFrame->fence, VK_TRUE, UINT64_MAX);
        vkResetFences(vk.device, 1, &offscreenFrame->fence);

        vk.lastSubmittedOffscreenFrame = vk.nextOffscreenFrame;
        vk.nextOffscreenFrame = (vk.nextOffscreenFrame + 1U) % vk.offscreenFrames.size();
        offscreenFrame->observationTensorValid = false;
        for (uint32_t envIndex = 0; envIndex < m_batchSize; ++envIndex)
            m_slots[envIndex].lastObservationFrame = vk.lastSubmittedOffscreenFrame;
    }

    void* mapped = nullptr;
    const VkDeviceSize paramsBytes = sizeof(RenderParams) * m_batchSize;
    vkCheck(vkMapMemory(vk.device, vk.paramsBuffer.memory, 0, paramsBytes, 0, &mapped), "vkMapMemory params");
    std::memcpy(mapped, m_renderParams.get(), static_cast<std::size_t>(paramsBytes));
    vkUnmapMemory(vk.device, vk.paramsBuffer.memory);

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

    const auto pushConstants = [&vk, commandBuffer, offscreenFrame](uint32_t envIndex) {
        DrawPushConstants pushConstants {
            .envIndex = envIndex,
            .layerIndex = offscreenFrame ? envIndex : 0U,
        };
        vkCmdPushConstants(
            commandBuffer, vk.pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pushConstants), &pushConstants);
    };

    bool pipelineBound = false;
    for (uint32_t envIndex = 0; envIndex < m_batchSize; ++envIndex) {
        RenderSlot& slot = m_slots[envIndex];
        pushConstants(envIndex);
        if (slot.pigVertexCount > 0 && slot.instanceCount > 0) {
            if (!pipelineBound) {
                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk.meshPipeline);
                pipelineBound = true;
            }
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk.pipelineLayout, 0, 1,
                &vk.pigDescriptorSet, 0, nullptr);
            vkCmdDraw(commandBuffer, slot.pigVertexCount, slot.instanceCount, 0, slot.instanceOffset + 1U);
        }
    }

    // TODO implement
    // some slots may still be pending generation at this point, but we have to draw the ones that are ready, to avoid
    // stalling the GPU for too long. We will draw the pending ones in the next frame.
    // std::vector<uint32_t> postponedSlots;

    pipelineBound = false;
    for (uint32_t envIndex = 0; envIndex < m_batchSize; ++envIndex) {
        RenderSlot& slot = m_slots[envIndex];
        if (slot.pendingGeneration.valid()) {
            // TODO theoretically we could postpone this slot and switch to another slot, to reduce stalling.
            const WorldGenerationOutput& output = slot.pendingGeneration.get();
            slot.terrainVoxelCount = output.voxelCount;
            slot.lastWorldVersion = output.worldVersion;
            slot.pendingGeneration = {};
        }
        pushConstants(envIndex);

        if (slot.terrainVoxelCount > 0) {
            if (!pipelineBound) {
                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk.voxelPipeline);
                pipelineBound = true;
            }
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk.pipelineLayout, 0, 1,
                &vk.terrainDescriptorSet, 0, nullptr);
            // 6 faces, every face has 2 triangles, every triangle has 3 vertices = 36.
            constexpr uint32_t verticesPerVoxel = 36;
            vkCmdDraw(commandBuffer, verticesPerVoxel * slot.terrainVoxelCount, 1,
                verticesPerVoxel * slot.terrainVoxelOffset, 0);
        }
    }
    vkCmdEndRenderPass(commandBuffer);
    if (offscreenFrame) {
        const VkBufferImageCopy copyRegion {
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = m_batchSize },
            .imageOffset = { 0, 0, 0 },
            .imageExtent = { vk.renderExtent.width, vk.renderExtent.height, 1 },
        };
        vkCmdCopyImageToBuffer(commandBuffer, offscreenFrame->color.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            offscreenFrame->observationBuffer.buffer, 1, &copyRegion);

        offscreenFrame->observationVersion = m_observationVersion;
    }
    vkCheck(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");

    const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    const VkSemaphore signalSemaphore
        = offscreenFrame ? offscreenFrame->frameCompletionSemaphore.vulkanSemaphore() : vk.renderFinished;

    const uint64_t signalValues[] = { m_observationVersion };
    const VkTimelineSemaphoreSubmitInfo timelineSubmitInfo {
        .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
        .pNext = nullptr,
        .waitSemaphoreValueCount = 0,
        .pWaitSemaphoreValues = nullptr,
        .signalSemaphoreValueCount = std::size(signalValues),
        .pSignalSemaphoreValues = signalValues,
    };
    const VkSubmitInfo submitInfo {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = vk.swapchain ? nullptr : &timelineSubmitInfo,
        .waitSemaphoreCount = vk.swapchain ? 1U : 0U,
        .pWaitSemaphores = vk.swapchain ? &vk.imageAvailable : nullptr,
        .pWaitDstStageMask = vk.swapchain ? &waitStage : nullptr,
        .commandBufferCount = 1,
        .pCommandBuffers = &commandBuffer,
        .signalSemaphoreCount = 1U,
        .pSignalSemaphores = &signalSemaphore,
    };
    // TODO migrate to vkQueueSubmit2
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

const Observation& Renderer::renderObservations(std::span<const World> worlds, std::span<const AgentState> agents)
{
    if (worlds.size() != agents.size()) [[unlikely]]
        fatalError("renderObservations world/agent count mismatch");

    ++m_observationVersion;
    assert(worlds.size() == m_batchSize);
    for (std::size_t i = 0; i < worlds.size(); ++i) {
        renderObservationSlot(i, worlds[i], agents[i]);
        m_renderParams[i] = buildRenderParams(agents[i], worlds[i]);
    }
    drawFrame();
    m_observation.setVersion(m_observationVersion);
    for (uint32_t i = 0; i < m_batchSize; ++i) {
        RenderSlot& slot = m_slots[i];
        const uintptr_t handle = m_vk->swapchain
            ? reinterpret_cast<uintptr_t>(m_vk->swapchain)
            : reinterpret_cast<uintptr_t>(m_vk->offscreenFrames[slot.lastObservationFrame].color.image);
        m_observation.setSlot(i, handle);
    }
    return m_observation;
}

void Renderer::renderObservationSlot(std::size_t slotIndex, const World& world, const AgentState& agent)
{
    const IVec3 agentBlock { floorToInt32(agent.position.x), floorToInt32(agent.position.y),
        floorToInt32(agent.position.z) };
    constexpr int32_t MeshCacheStride = 12;
    RenderSlot& slot = m_slots[slotIndex];
    const IVec3 meshDelta = glm::abs(agentBlock - slot.lastMeshCenter);
    const bool agentLeftMeshCache
        = meshDelta.x > MeshCacheStride || meshDelta.y > MeshCacheStride || meshDelta.z > MeshCacheStride;
    if (slot.lastWorldVersion != world.version() || agentLeftMeshCache) {
        if (!m_vk->swapchain) {
            // TODO get rid of CPU synchronization. It's better to use one more semaphore instead
            for (OffscreenFrame& frame : m_vk->offscreenFrames) {
                if (frame.observationVersion != OffscreenFrame::NO_OBSERVATION_VERSION) [[likely]]
                    frame.frameCompletionSemaphore.wait(m_worldGenerator.stream(), frame.observationVersion);
            }
        } else
            vkWaitForFences(m_vk->device, 1, &m_vk->inFlight, VK_TRUE, UINT64_MAX);

        WorldGenerationBuffers buffers;
        buffers.header = &reinterpret_cast<TerrainHeader*>(m_vk->terrainHeaderBuffer.cudaPtr)[slotIndex];
        buffers.maxVoxelCount = MaxTerrainVoxels;
        buffers.voxels = reinterpret_cast<Voxel*>(
            static_cast<std::byte*>(m_vk->terrainVoxelBuffer.cudaPtr) + slot.terrainVoxelOffset * VoxelSize);
        buffers.blocks = world.borrowGenerationBuffers();

        CudaSharedFuture<WorldGenerationOutput> generation
            = m_worldGenerator.generate(world, agent, std::move(buffers)).share();
        world.updateGeneration(generation);
        slot.pendingGeneration = std::move(generation);
        slot.lastWorldVersion = world.version();
        slot.lastMeshCenter = agentBlock;
    }
    uploadInstances(slotIndex, world);
}

} // namespace blocklab
