#include <blocklab/graphics/Renderer.h>

#include <blocklab/CudaHelpers.h>
#include <blocklab/CudaObservation.h>
#include <blocklab/Error.h>
#include <blocklab/graphics/Memory.h>
#include <blocklab/graphics/Vulkan.h>
#include <blocklab/graphics/VulkanBuffer.h>
#include <blocklab/graphics/VulkanCudaInteropBuffer.h>
#include <blocklab/meshes/PigMesh.h>

#include <cuda_runtime.h>
#include <vulkan/vulkan.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <utility>
#include <vector>

namespace blocklab {
namespace {

    constexpr float EyeHeight = 1.62f;
    constexpr vk::Format ColorFormat = vk::Format::eR8G8B8A8Unorm;
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

    struct Image {
        vk::UniqueImage image;
        vk::UniqueDeviceMemory memory;
        vk::UniqueImageView view;
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
        vk::UniqueFramebuffer framebuffer;
        vk::CommandBuffer commandBuffer;
        VulkanCudaInteropBuffer observationBuffer;
        float* observationTensor = nullptr;
        CudaSharedFuture<void> conversionTaskFuture;

        ExternalSemaphore frameCompletionSemaphore;
        std::uint64_t observationVersion = NO_OBSERVATION_VERSION;
        vk::UniqueFence fence;
    };

    inline bool hasStencilComponent(vk::Format format)
    {
        return format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint;
    }

    vk::Format findSupportedDepthFormat(vk::PhysicalDevice physicalDevice)
    {
        const vk::Format candidates[] {
            vk::Format::eD32Sfloat,
            vk::Format::eD32SfloatS8Uint,
            vk::Format::eD24UnormS8Uint,
            vk::Format::eD16Unorm,
        };

        for (vk::Format format : candidates) {
            const vk::FormatProperties properties = physicalDevice.getFormatProperties(format);
            if ((properties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment)
                == vk::FormatFeatureFlagBits::eDepthStencilAttachment)
                return format;
        }

        fatalError("No supported Vulkan depth format");
    }

} // namespace

struct Renderer::VulkanState {
    VulkanState() = default;

    VulkanState(const VulkanState&) = delete;
    VulkanState& operator=(const VulkanState&) = delete;
    VulkanState(VulkanState&&) = delete;
    VulkanState& operator=(VulkanState&&) = delete;

    vk::Extent2D renderExtent;

    vk::Format colorFormat = ColorFormat;
    vk::Format depthFormat = vk::Format::eUndefined;

    Image depthImage;

    std::vector<OffscreenFrame> frames;
    std::size_t nextFrame = 0;
    std::size_t lastSubmittedFrame = 0;

    vk::UniqueRenderPass renderPass;

    vk::UniqueDescriptorSetLayout drawResourceSetLayout;
    vk::UniquePipelineLayout pipelineLayout;

    vk::UniquePipeline meshPipeline;
    vk::UniquePipeline voxelPipeline;

    vk::UniqueCommandPool commandPool;
    std::vector<vk::UniqueCommandBuffer> commandBuffers;

    // Descriptor sets are owned by the descriptor pool and are released when the pool is destroyed.
    vk::UniqueDescriptorPool descriptorPool;
    vk::DescriptorSet terrainDescriptorSet;
    vk::DescriptorSet pigDescriptorSet;

    VulkanCudaInteropBuffer terrainHeaderBuffer;
    VulkanCudaInteropBuffer terrainVoxelBuffer;
    VulkanDeviceBuffer pigVertexBuffer;
    VulkanHostBuffer instanceBuffer;
    VulkanHostBuffer paramsBuffer;

    vk::DeviceSize terrainVoxelBufferCapacityBytes = 0;
    cudaStream_t observationConversionStream = nullptr;
    std::uint32_t batchSize = 1;
};

namespace {

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

    Image createDepthImage(Vulkan& vk, Renderer::VulkanState& state)
    {
        Image result;
        const vk::ImageCreateInfo imageInfo {
            .imageType = vk::ImageType::e2D,
            .format = state.depthFormat,
            .extent = { state.renderExtent.width, state.renderExtent.height, 1 },
            .mipLevels = 1,
            .arrayLayers = state.batchSize,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eDepthStencilAttachment,
            .sharingMode = vk::SharingMode::eExclusive,
            .initialLayout = vk::ImageLayout::eUndefined,
        };
        result.image = vkCheck(vk.device().createImageUnique(imageInfo), "VkDevice::createImage depth");

        vk::MemoryRequirements requirements = vk.device().getImageMemoryRequirements(*result.image);
        const vk::MemoryAllocateInfo allocInfo {
            .allocationSize = requirements.size,
            .memoryTypeIndex = findMemoryType(
                vk.physicalDevice(), requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal),
        };
        result.memory = vkCheck(vk.device().allocateMemoryUnique(allocInfo), "VkDevice::allocateMemory depth");
        vkCheck(vk.device().bindImageMemory(*result.image, *result.memory, 0), "VkDevice::bindImageMemory depth");

        vk::ImageAspectFlags flags = vk::ImageAspectFlagBits::eDepth;
        if (hasStencilComponent(state.depthFormat))
            flags |= vk::ImageAspectFlagBits::eStencil;
        const vk::ImageViewCreateInfo viewInfo {
            .image = *result.image,
            .viewType = vk::ImageViewType::e2DArray,
            .format = state.depthFormat,
            .subresourceRange = {
                .aspectMask = flags,
                .levelCount = 1,
                .layerCount = state.batchSize,
            },
        };
        result.view = vkCheck(vk.device().createImageViewUnique(viewInfo), "VkDevice::createImageView depth");
        return result;
    }

    Image createColorImage(Vulkan& vk, Renderer::VulkanState& state)
    {
        Image result;
        const vk::ImageCreateInfo imageInfo {
            .imageType = vk::ImageType::e2D,
            .format = state.colorFormat,
            .extent = { state.renderExtent.width, state.renderExtent.height, 1 },
            .mipLevels = 1,
            .arrayLayers = state.batchSize,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled
                | vk::ImageUsageFlagBits::eTransferSrc,
            .sharingMode = vk::SharingMode::eExclusive,
            .initialLayout = vk::ImageLayout::eUndefined,
        };
        result.image = vkCheck(vk.device().createImageUnique(imageInfo), "VkDevice::createImage color");

        vk::MemoryRequirements requirements = vk.device().getImageMemoryRequirements(*result.image);
        const vk::MemoryAllocateInfo allocInfo {
            .allocationSize = requirements.size,
            .memoryTypeIndex = findMemoryType(
                vk.physicalDevice(), requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal),
        };
        result.memory = vkCheck(vk.device().allocateMemoryUnique(allocInfo), "VkDevice::allocateMemory color");
        vkCheck(vk.device().bindImageMemory(*result.image, *result.memory, 0), "VkDevice::bindImageMemory color");

        const vk::ImageViewCreateInfo viewInfo {
            .image = *result.image,
            .viewType = vk::ImageViewType::e2DArray,
            .format = state.colorFormat,
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .levelCount = 1,
                .layerCount = state.batchSize,
            },
        };
        result.view = vkCheck(vk.device().createImageViewUnique(viewInfo), "VkDevice::createImageView color");
        return result;
    }

    vk::UniqueShaderModule createShaderModule(Vulkan& vk, const std::vector<char>& bytes)
    {
        const vk::ShaderModuleCreateInfo info {
            .codeSize = bytes.size(),
            .pCode = reinterpret_cast<const uint32_t*>(bytes.data()),
        };
        return vkCheck(vk.device().createShaderModuleUnique(info), "VkDevice::createShaderModule");
    }

    void createRenderPass(Vulkan& vk, Renderer::VulkanState& state)
    {
        const vk::AttachmentDescription colorAttachment {
            .format = state.colorFormat,
            .samples = vk::SampleCountFlagBits::e1,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = vk::ImageLayout::eUndefined,
            .finalLayout = vk::ImageLayout::eTransferSrcOptimal,
        };
        const vk::AttachmentDescription depthAttachment {
            .format = state.depthFormat,
            .samples = vk::SampleCountFlagBits::e1,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eDontCare,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = vk::ImageLayout::eUndefined,
            .finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
        };
        const vk::AttachmentDescription attachments[] { colorAttachment, depthAttachment };

        const vk::AttachmentReference colorRef {
            .attachment = 0,
            .layout = vk::ImageLayout::eColorAttachmentOptimal,
        };
        const vk::AttachmentReference depthRef {
            .attachment = 1,
            .layout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
        };
        const vk::SubpassDescription subpass {
            .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
            .colorAttachmentCount = 1,
            .pColorAttachments = &colorRef,
            .pDepthStencilAttachment = &depthRef,
        };

        const vk::SubpassDependency dependencies[2] {
            {
                .srcSubpass = VK_SUBPASS_EXTERNAL,
                .dstSubpass = 0,
                .srcStageMask = vk::PipelineStageFlagBits::eFragmentShader,
                .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput
                    | vk::PipelineStageFlagBits::eEarlyFragmentTests | vk::PipelineStageFlagBits::eLateFragmentTests,
                .dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite
                    | vk::AccessFlagBits::eDepthStencilAttachmentRead
                    | vk::AccessFlagBits::eDepthStencilAttachmentWrite,
                .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            },
            {
                .srcSubpass = 0,
                .dstSubpass = VK_SUBPASS_EXTERNAL,
                .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput
                    | vk::PipelineStageFlagBits::eEarlyFragmentTests | vk::PipelineStageFlagBits::eLateFragmentTests,
                .dstStageMask = vk::PipelineStageFlagBits::eTransfer,
                .srcAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite
                    | vk::AccessFlagBits::eDepthStencilAttachmentRead
                    | vk::AccessFlagBits::eDepthStencilAttachmentWrite,
                .dstAccessMask = vk::AccessFlagBits::eTransferRead,
                .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            },
        };

        const vk::RenderPassCreateInfo info {
            .attachmentCount = std::size(attachments),
            .pAttachments = attachments,
            .subpassCount = 1,
            .pSubpasses = &subpass,
            .dependencyCount = std::size(dependencies),
            .pDependencies = dependencies,
        };
        state.renderPass = vkCheck(vk.device().createRenderPassUnique(info), "VkDevice::createRenderPass");
    }

    void createPipeline(Vulkan& vk, Renderer::VulkanState& state)
    {
        const vk::DescriptorSetLayoutBinding vertexBinding {
            .binding = 0,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eVertex,
        };
        const vk::DescriptorSetLayoutBinding instanceBinding {
            .binding = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eVertex,
        };
        const vk::DescriptorSetLayoutBinding paramsBinding {
            .binding = 2,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eVertex,
        };
        const vk::DescriptorSetLayoutBinding vertexBindings[] { vertexBinding, instanceBinding, paramsBinding };
        const vk::DescriptorSetLayoutCreateInfo vertexLayoutInfo {
            .bindingCount = std::size(vertexBindings),
            .pBindings = vertexBindings,
        };
        state.drawResourceSetLayout = vkCheck(
            vk.device().createDescriptorSetLayoutUnique(vertexLayoutInfo), "VkDevice::createDescriptorSetLayout");

        const vk::PushConstantRange pushConstantRange {
            .stageFlags = vk::ShaderStageFlagBits::eVertex,
            .offset = 0,
            .size = sizeof(Renderer::DrawPushConstants),
        };
        const vk::DescriptorSetLayout layouts[] { *state.drawResourceSetLayout };
        const vk::PipelineLayoutCreateInfo pipelineLayoutInfo {
            .setLayoutCount = std::size(layouts),
            .pSetLayouts = layouts,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pushConstantRange,
        };
        state.pipelineLayout
            = vkCheck(vk.device().createPipelineLayoutUnique(pipelineLayoutInfo), "VkDevice::createPipelineLayout");

        const vk::UniqueShaderModule voxelVertexShader
            = createShaderModule(vk, readFile(std::filesystem::path(BLOCKLAB_SHADER_DIR) / "voxel_vertex.spv"));
        const vk::UniqueShaderModule meshVertexShader
            = createShaderModule(vk, readFile(std::filesystem::path(BLOCKLAB_SHADER_DIR) / "mesh_vertex.spv"));
        const vk::UniqueShaderModule fragmentShader
            = createShaderModule(vk, readFile(std::filesystem::path(BLOCKLAB_SHADER_DIR) / "mesh_fragment.spv"));

        const vk::PipelineShaderStageCreateInfo meshStages[] {
            {
                .stage = vk::ShaderStageFlagBits::eVertex,
                .module = *meshVertexShader,
                .pName = "meshVertexMain",
            },
            {
                .stage = vk::ShaderStageFlagBits::eFragment,
                .module = *fragmentShader,
                .pName = "meshFragmentMain",
            },
        };
        const vk::PipelineShaderStageCreateInfo voxelStages[] {
            {
                .stage = vk::ShaderStageFlagBits::eVertex,
                .module = *voxelVertexShader,
                .pName = "voxelVertexMain",
            },
            {
                .stage = vk::ShaderStageFlagBits::eFragment,
                .module = *fragmentShader,
                .pName = "meshFragmentMain",
            },
        };

        const vk::PipelineVertexInputStateCreateInfo emptyVertexInput;
        const vk::PipelineInputAssemblyStateCreateInfo triangleInputAssembly {
            .topology = vk::PrimitiveTopology::eTriangleList,
        };
        const vk::Viewport viewport {
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(state.renderExtent.width),
            .height = static_cast<float>(state.renderExtent.height),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        const vk::Rect2D scissor {
            .extent = state.renderExtent,
        };
        const vk::PipelineViewportStateCreateInfo viewportState {
            .viewportCount = 1,
            .pViewports = &viewport,
            .scissorCount = 1,
            .pScissors = &scissor,
        };
        const vk::PipelineRasterizationStateCreateInfo rasterizer {
            .depthClampEnable = false,
            .rasterizerDiscardEnable = false,
            .polygonMode = vk::PolygonMode::eFill,
            .cullMode = vk::CullModeFlagBits::eNone,
            .frontFace = vk::FrontFace::eCounterClockwise,
            .lineWidth = 1.0f,
        };
        const vk::PipelineMultisampleStateCreateInfo multisample {
            .rasterizationSamples = vk::SampleCountFlagBits::e1,
        };
        const vk::PipelineDepthStencilStateCreateInfo depthStencil {
            .depthTestEnable = true,
            .depthWriteEnable = true,
            .depthCompareOp = vk::CompareOp::eLess,
        };
        const vk::PipelineColorBlendAttachmentState colorBlendAttachment {
            .blendEnable = false,
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG
                | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
        };
        const vk::PipelineColorBlendStateCreateInfo colorBlend {
            .attachmentCount = 1,
            .pAttachments = &colorBlendAttachment,
        };

        const vk::GraphicsPipelineCreateInfo pipelineInfo[2] {
            {
                .stageCount = std::size(meshStages),
                .pStages = meshStages,
                .pVertexInputState = &emptyVertexInput,
                .pInputAssemblyState = &triangleInputAssembly,
                .pViewportState = &viewportState,
                .pRasterizationState = &rasterizer,
                .pMultisampleState = &multisample,
                .pDepthStencilState = &depthStencil,
                .pColorBlendState = &colorBlend,
                .layout = *state.pipelineLayout,
                .renderPass = *state.renderPass,
                .subpass = 0,
            },
            {
                .stageCount = std::size(voxelStages),
                .pStages = voxelStages,
                .pVertexInputState = &emptyVertexInput,
                .pInputAssemblyState = &triangleInputAssembly,
                .pViewportState = &viewportState,
                .pRasterizationState = &rasterizer,
                .pMultisampleState = &multisample,
                .pDepthStencilState = &depthStencil,
                .pColorBlendState = &colorBlend,
                .layout = *state.pipelineLayout,
                .renderPass = *state.renderPass,
                .subpass = 0,
            },
        };

        state.meshPipeline = vkCheck(
            vk.device().createGraphicsPipelineUnique({}, pipelineInfo[0]), "VkDevice::createGraphicsPipeline mesh");
        state.voxelPipeline = vkCheck(
            vk.device().createGraphicsPipelineUnique({}, pipelineInfo[1]), "VkDevice::createGraphicsPipeline voxel");
    }

    void createFramebuffers(Vulkan& vk, Renderer::VulkanState& state)
    {
        state.frames.resize(OffscreenFrameCount);
        for (OffscreenFrame& frame : state.frames) {
            frame.color = createColorImage(vk, state);
            frame.depth = createDepthImage(vk, state);
            const vk::ImageView attachments[] { *frame.color.view, *frame.depth.view };
            const vk::FramebufferCreateInfo info {
                .renderPass = *state.renderPass,
                .attachmentCount = std::size(attachments),
                .pAttachments = attachments,
                .width = state.renderExtent.width,
                .height = state.renderExtent.height,
                .layers = state.batchSize,
            };
            frame.framebuffer
                = vkCheck(vk.device().createFramebufferUnique(info), "VkDevice::createFramebuffer offscreen");
        }
    }

    void createCommandsAndSync(Vulkan& vk, Renderer::VulkanState& state)
    {
        const vk::CommandPoolCreateInfo poolInfo {
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = vk.graphicsQueueFamilyIndex(),
        };
        state.commandPool = vkCheck(vk.device().createCommandPoolUnique(poolInfo), "VkDevice::createCommandPool");

        const vk::CommandBufferAllocateInfo allocInfo {
            .commandPool = *state.commandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = static_cast<uint32_t>(state.frames.size()),
        };
        state.commandBuffers
            = vkCheck(vk.device().allocateCommandBuffersUnique(allocInfo), "VkDevice::allocateCommandBuffers");

        const vk::FenceCreateInfo fenceInfo {
            .flags = vk::FenceCreateFlagBits::eSignaled,
        };
        for (std::size_t i = 0; i < state.frames.size(); ++i) {
            state.frames[i].commandBuffer = *state.commandBuffers[i];
            state.frames[i].fence
                = vkCheck(vk.device().createFenceUnique(fenceInfo), "VkDevice::createFence offscreen");
            state.frames[i].frameCompletionSemaphore = { vk.device() };
        }
    }

    void createDescriptors(Vulkan& vk, Renderer::VulkanState& state)
    {
        const vk::DeviceSize terrainHeadersSizeBytes = sizeof(TerrainHeader) * state.batchSize;
        state.terrainHeaderBuffer
            = VulkanCudaInteropBuffer(vk, terrainHeadersSizeBytes, vk::BufferUsageFlagBits::eStorageBuffer);

        state.terrainVoxelBufferCapacityBytes
            = static_cast<vk::DeviceSize>(MaxTerrainVoxels) * state.batchSize * VoxelSize;
        state.terrainVoxelBuffer = VulkanCudaInteropBuffer(
            vk, state.terrainVoxelBufferCapacityBytes, vk::BufferUsageFlagBits::eStorageBuffer);
        state.pigVertexBuffer
            = VulkanDeviceBuffer(vk, static_cast<vk::DeviceSize>(PigMesh::verticesCount()) * sizeof(MeshVertex),
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst);
        state.instanceBuffer
            = VulkanHostBuffer(vk, sizeof(Renderer::EntityInstance) * MaxEntityInstances * state.batchSize,
                vk::BufferUsageFlagBits::eStorageBuffer);
        state.paramsBuffer = VulkanHostBuffer(
            vk, sizeof(Renderer::RenderParams) * state.batchSize, vk::BufferUsageFlagBits::eStorageBuffer);

        const vk::DeviceSize observationBytes
            = static_cast<vk::DeviceSize>(state.renderExtent.width) * state.renderExtent.height * 4U * state.batchSize;
        for (OffscreenFrame& frame : state.frames) {
            frame.observationBuffer
                = VulkanCudaInteropBuffer(vk, observationBytes, vk::BufferUsageFlagBits::eTransferDst);
            const std::size_t tensorBytes = static_cast<std::size_t>(state.renderExtent.width)
                * static_cast<std::size_t>(state.renderExtent.height) * 3U * state.batchSize * sizeof(float);
            cudaCheck(cudaMalloc(&frame.observationTensor, tensorBytes), "cudaMalloc observation tensor");
        }

        const vk::DescriptorPoolSize poolSizes[] {
            {
                .type = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 6,
            },
        };
        vk::DescriptorPoolCreateInfo poolInfo;
        poolInfo.setMaxSets(2U);
        poolInfo.setPoolSizes(poolSizes);
        state.descriptorPool
            = vkCheck(vk.device().createDescriptorPoolUnique(poolInfo), "VkDevice::createDescriptorPool");

        const vk::DescriptorSetLayout layouts[] { *state.drawResourceSetLayout, *state.drawResourceSetLayout };
        vk::DescriptorSet descriptorSets[2] = {};
        const vk::DescriptorSetAllocateInfo allocInfo {
            .descriptorPool = *state.descriptorPool,
            .descriptorSetCount = 2,
            .pSetLayouts = layouts,
        };
        vkCheck(vk.device().allocateDescriptorSets(&allocInfo, descriptorSets), "VkDevice::allocateDescriptorSets");
        state.terrainDescriptorSet = descriptorSets[0];
        state.pigDescriptorSet = descriptorSets[1];

        const vk::DescriptorBufferInfo terrainHeaderInfo = state.terrainHeaderBuffer.info();
        const vk::DescriptorBufferInfo voxelInfo = state.terrainVoxelBuffer.info();
        const vk::DescriptorBufferInfo pigVertexInfo = state.pigVertexBuffer.info();
        const vk::DescriptorBufferInfo instanceInfo = state.instanceBuffer.info();
        const vk::DescriptorBufferInfo paramsInfo = state.paramsBuffer.info();

        const std::array<vk::WriteDescriptorSet, 3> terrainWrites = {
            vk::WriteDescriptorSet {
                .dstSet = state.terrainDescriptorSet,
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo = &terrainHeaderInfo,
            },
            vk::WriteDescriptorSet {
                .dstSet = state.terrainDescriptorSet,
                .dstBinding = 1,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo = &voxelInfo,
            },
            vk::WriteDescriptorSet {
                .dstSet = state.terrainDescriptorSet,
                .dstBinding = 2,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo = &paramsInfo,
            },
        };

        const std::array<vk::WriteDescriptorSet, 3> pigWrites = {
            vk::WriteDescriptorSet {
                .dstSet = state.pigDescriptorSet,
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo = &pigVertexInfo,
            },
            vk::WriteDescriptorSet {
                .dstSet = state.pigDescriptorSet,
                .dstBinding = 1,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo = &instanceInfo,
            },
            vk::WriteDescriptorSet {
                .dstSet = state.pigDescriptorSet,
                .dstBinding = 2,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo = &paramsInfo,
            },
        };
        vk.device().updateDescriptorSets(terrainWrites, {});
        vk.device().updateDescriptorSets(pigWrites, {});
    }

    void initVulkan(Vulkan& vk, Renderer::VulkanState& state, RenderConfig config)
    {
        state.batchSize = std::max(1U, config.batchSize);
        state.renderExtent = vk::Extent2D {
            .width = static_cast<uint32_t>(config.width),
            .height = static_cast<uint32_t>(config.height),
        };
        state.colorFormat = ColorFormat;
        state.depthFormat = findSupportedDepthFormat(vk.physicalDevice());
        createRenderPass(vk, state);
        createPipeline(vk, state);
        createFramebuffers(vk, state);
        createCommandsAndSync(vk, state);
        createDescriptors(vk, state);
        cudaCheck(cudaStreamCreateWithFlags(&state.observationConversionStream, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags observation");
    }

    void destroyVulkan(Vulkan& vk, Renderer::VulkanState& state)
    {
        vkCheck(vk.device().waitIdle(), "VkDevice::waitIdle");
        cudaCheck(cudaStreamSynchronize(state.observationConversionStream), "cudaStreamSynchronize observation");
        for (OffscreenFrame& frame : state.frames) {
            if (frame.observationTensor)
                cudaCheck(cudaFree(frame.observationTensor), "cudaFree observation tensor");
        }
        cudaCheck(cudaStreamDestroy(state.observationConversionStream), "cudaStreamDestroy observation");
    }

} // namespace

Renderer::Renderer(Vulkan& vk, RenderConfig config)
    : m_config(config)
    , m_vk(vk)
    , m_state(std::make_unique<VulkanState>())
    , m_worldGenerator({ .halfExtent = TerrainMeshHalfExtent })
{
    initVulkan(m_vk, *m_state, m_config);
    initializeBatchData();
}

Renderer::~Renderer() { destroyVulkan(m_vk, *m_state); }

std::size_t Renderer::cudaObservationTensorBytes() const
{
    return static_cast<std::size_t>(m_state->renderExtent.width)
        * static_cast<std::size_t>(m_state->renderExtent.height) * 3U * m_state->batchSize * sizeof(float);
}

void Renderer::initializeBatchData()
{
    m_batchSize = m_state->batchSize;
    m_slots = std::make_unique<RenderSlot[]>(m_batchSize);
    m_renderParams = std::make_unique<RenderParams[]>(m_batchSize);
    m_observation.reset(m_state->renderExtent.width, m_state->renderExtent.height, m_batchSize);
    m_observation.setVersion(m_observationVersion);
    if (!m_pigMeshUploaded) {
        PigMesh pigMeshGenerator;
        const std::span<MeshVertex> pigMesh = pigMeshGenerator.generate();
        m_pigMeshVertexCount = static_cast<uint32_t>(pigMesh.size());
        m_state->pigVertexBuffer.uploadSync(
            m_vk, *m_state->commandPool, pigMesh.data(), sizeof(MeshVertex) * pigMesh.size());
        m_pigMeshUploaded = true;
    }
    for (uint32_t i = 0; i < m_batchSize; ++i) {
        RenderSlot& slot = m_slots[i];
        slot.terrainVoxelOffset = i * MaxTerrainVoxels;
        slot.pigVertexCount = m_pigMeshVertexCount;
        slot.instanceOffset = i * MaxEntityInstances;
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
        .worldOriginAndWidth = { 0, 0, 0, static_cast<int32_t>(m_state->renderExtent.width) },
        .regionAndHeight = { 0, 0, 0, static_cast<int32_t>(m_state->renderExtent.height) },
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

    const vk::DeviceSize uploadBytes = sizeof(EntityInstance) * m_instances.size();
    const vk::DeviceSize uploadOffset = sizeof(EntityInstance) * slot.instanceOffset;
    void* mapped = vkCheck(m_vk.device().mapMemory(m_state->instanceBuffer.vkMemory(), uploadOffset, uploadBytes),
        "vkMapMemory instances");
    std::memcpy(mapped, m_instances.data(), static_cast<std::size_t>(uploadBytes));
    m_vk.device().unmapMemory(m_state->instanceBuffer.vkMemory());
}

void Renderer::drawFrame()
{
    OffscreenFrame& offscreenFrame = m_state->frames[m_state->nextFrame];

    if (offscreenFrame.observationVersion != OffscreenFrame::NO_OBSERVATION_VERSION)
        offscreenFrame.conversionTaskFuture.wait();

    vkCheck(m_vk.device().waitForFences(1, &*offscreenFrame.fence, vk::True, std::numeric_limits<std::uint64_t>::max()),
        "VkDevice::waitForFences");
    vkCheck(m_vk.device().resetFences(1, &*offscreenFrame.fence), "VkDevice::resetFences");

    m_state->lastSubmittedFrame = m_state->nextFrame;
    m_state->nextFrame = (m_state->nextFrame + 1U) % m_state->frames.size();

    m_state->paramsBuffer.upload(m_vk, 0, m_renderParams.get(), sizeof(RenderParams) * m_batchSize);

    vk::CommandBuffer commandBuffer = offscreenFrame.commandBuffer;
    commandBuffer.reset();
    const vk::CommandBufferBeginInfo beginInfo;
    vkCheck(commandBuffer.begin(beginInfo), "VkCommandBuffer::begin");
    const vk::ClearValue clearValues[] {
        vk::ClearColorValue(std::array<float, 4> { 0.42f, 0.64f, 0.86f, 1.0f }),
        vk::ClearDepthStencilValue(1.0f, 0),
    };
    const vk::Rect2D renderArea {
        .extent = m_state->renderExtent,
    };
    const vk::RenderPassBeginInfo renderPassInfo {
        .renderPass = *m_state->renderPass,
        .framebuffer = *offscreenFrame.framebuffer,
        .renderArea = renderArea,
        .clearValueCount = static_cast<uint32_t>(std::size(clearValues)),
        .pClearValues = clearValues,
    };
    commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

    const auto pushConstants = [this, commandBuffer](uint32_t envIndex) {
        DrawPushConstants pushConstants {
            .envIndex = envIndex,
            .layerIndex = envIndex,
        };
        commandBuffer.pushConstants<DrawPushConstants>(
            *m_state->pipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, pushConstants);
    };

    bool pipelineBound = false;
    for (uint32_t envIndex = 0; envIndex < m_batchSize; ++envIndex) {
        RenderSlot& slot = m_slots[envIndex];
        pushConstants(envIndex);
        if (slot.pigVertexCount > 0 && slot.instanceCount > 0) {
            if (!pipelineBound) {
                commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *m_state->meshPipeline);
                pipelineBound = true;
            }
            commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *m_state->pipelineLayout, 0, 1,
                &m_state->pigDescriptorSet, 0, nullptr);
            commandBuffer.draw(slot.pigVertexCount, slot.instanceCount, 0, slot.instanceOffset + 1U);
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
                commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *m_state->voxelPipeline);
                pipelineBound = true;
            }
            commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *m_state->pipelineLayout, 0, 1,
                &m_state->terrainDescriptorSet, 0, nullptr);
            // 6 faces, every face has 2 triangles, every triangle has 3 vertices = 36.
            constexpr uint32_t verticesPerVoxel = 36;
            commandBuffer.draw(
                verticesPerVoxel * slot.terrainVoxelCount, 1, verticesPerVoxel * slot.terrainVoxelOffset, 0);
        }
    }
    commandBuffer.endRenderPass();

    const vk::BufferImageCopy copyRegion {
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = { .aspectMask = vk::ImageAspectFlagBits::eColor, .layerCount = m_batchSize },
        .imageExtent = { m_state->renderExtent.width, m_state->renderExtent.height, 1 },
    };
    commandBuffer.copyImageToBuffer(*offscreenFrame.color.image, vk::ImageLayout::eTransferSrcOptimal,
        offscreenFrame.observationBuffer.vkBuffer(), 1, &copyRegion);
    offscreenFrame.observationVersion = m_observationVersion;

    vkCheck(commandBuffer.end(), "VkCommandBuffer::end");

    const uint64_t signalValues[] = { m_observationVersion };
    const vk::TimelineSemaphoreSubmitInfo timelineSubmitInfo {
        .signalSemaphoreValueCount = std::size(signalValues),
        .pSignalSemaphoreValues = signalValues,
    };
    const vk::Semaphore signalSemaphore = offscreenFrame.frameCompletionSemaphore.vulkanSemaphore();
    const vk::SubmitInfo submitInfo[] = { {
        .pNext = &timelineSubmitInfo,
        .commandBufferCount = 1,
        .pCommandBuffers = &commandBuffer,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &signalSemaphore,
    } };
    // TODO migrate to submit2
    vkCheck(m_vk.graphicsQueue().submit(submitInfo, *offscreenFrame.fence), "VkQueue::submit");

    cudaStream_t stream = m_state->observationConversionStream;
    offscreenFrame.frameCompletionSemaphore.wait(stream, offscreenFrame.observationVersion);
    convertRgba8ToFloatNchw(offscreenFrame.observationBuffer.cudaPtr(), offscreenFrame.observationTensor,
        m_state->batchSize, m_state->renderExtent.width, m_state->renderExtent.height,
        reinterpret_cast<std::uintptr_t>(stream));
    CudaFuture<void> conversionTask(stream);
    offscreenFrame.conversionTaskFuture = conversionTask.share();
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
    OffscreenFrame& frame = m_state->frames[m_state->lastSubmittedFrame];
    m_observation.setVersion(m_observationVersion);
    m_observation.setData(frame.observationTensor, frame.conversionTaskFuture);
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
        // TODO get rid of CPU synchronization. It's better to use one more semaphore instead
        for (OffscreenFrame& frame : m_state->frames) {
            if (frame.observationVersion != OffscreenFrame::NO_OBSERVATION_VERSION) [[likely]]
                frame.frameCompletionSemaphore.wait(m_worldGenerator.stream(), frame.observationVersion);
        }

        WorldGenerationBuffers buffers;
        buffers.header = &m_state->terrainHeaderBuffer.cudaPtr<TerrainHeader>()[slotIndex];
        buffers.maxVoxelCount = MaxTerrainVoxels;
        buffers.voxels = reinterpret_cast<Voxel*>(
            m_state->terrainVoxelBuffer.cudaPtr<std::byte>() + slot.terrainVoxelOffset * VoxelSize);
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
