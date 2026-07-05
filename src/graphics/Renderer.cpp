#include <blocklab/graphics/Renderer.h>

#include "Mesh.h"

#include <blocklab/gpu/cuda/CudaHelpers.h>
#include <blocklab/gpu/interop/VulkanCudaInteropBuffer.h>
#include <blocklab/gpu/interop/VulkanCudaInteropSemaphore.h>
#include <blocklab/gpu/vulkan/Vulkan.h>
#include <blocklab/gpu/vulkan/VulkanBuffer.h>
#include <blocklab/inventory/Item.h>
#include <blocklab/utility/Error.h>
#include <characters/meshes/PigMesh.h>
#include <environment/CudaObservation.h>
#include <gpu/vulkan/Memory.h>
#include <world/Drop.h>
#include <world/Lighting.h>
#include <world/World.h>
#include <world/WorldGenerator.h>

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
#include <optional>
#include <utility>
#include <vector>

namespace blocklab {
namespace {

    constexpr float EyeHeight = 1.62f;
    constexpr vk::Format ColorFormat = vk::Format::eR8G8B8A8Srgb;
    constexpr std::uint32_t OffscreenFrameCount = 2;
    constexpr std::uint32_t MaxEntityInstances = 256;
    constexpr std::int32_t TerrainMeshHalfExtent = 32;
    constexpr std::uint32_t TerrainMeshExtent = TerrainMeshHalfExtent * 2;
    constexpr std::uint32_t s_maxTerrainVoxels = TerrainMeshExtent * World::s_height * TerrainMeshExtent;

    std::uint32_t renderEntityKindId(CharacterKind characterKind)
    {
        switch (characterKind) {
        case CharacterKind::Pig:
            return static_cast<std::uint32_t>(RenderEntityKind::Pig);
        case CharacterKind::Agent:
            // the agent character is not intended for rendering
            return static_cast<std::uint32_t>(RenderEntityKind::None);
        }
        fatalError("corrupted CharacterKind value: ", static_cast<int>(characterKind));
    }

    std::uint32_t renderDropKindId(Item::Type itemType)
    {
        switch (itemType) {
        case Item::Type::Dirt:
            return static_cast<std::uint32_t>(RenderEntityKind::DirtDrop);
        case Item::Type::Stone:
            return static_cast<std::uint32_t>(RenderEntityKind::StoneDrop);
        case Item::Type::Torch:
            return static_cast<std::uint32_t>(RenderEntityKind::TorchDrop);
        case Item::Type::COUNT:
            break;
        }
        fatalError("corrupted Item::Type value: ", static_cast<int>(itemType));
    }

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

    struct OffscreenFrame {
        static constexpr std::uint64_t NO_OBSERVATION_VERSION = 0;

        Image color;
        Image depth;
        vk::UniqueFramebuffer framebuffer;
        vk::CommandBuffer commandBuffer;
        VulkanCudaInteropBuffer observationBuffer;
        float* observationTensor = nullptr;
        CudaSharedFuture<void> conversionTaskFuture;

        VulkanCudaInteropTimelineSemaphore frameCompletionSemaphore;
        std::uint64_t observationVersion = NO_OBSERVATION_VERSION;
        std::uint64_t worldGenerationLastWaitVersion = NO_OBSERVATION_VERSION;
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

struct Renderer::RenderSlot {
    IVec3 lastMeshCenter { std::numeric_limits<std::int32_t>::min(), std::numeric_limits<std::int32_t>::min(),
        std::numeric_limits<std::int32_t>::min() };

    std::uint32_t terrainVoxelOffset = 0;
    std::uint32_t terrainVoxelCount = 0;

    std::uint32_t pigVertexCount = 0;
    std::uint32_t dropVertexOffset = 0;
    std::uint32_t dropVertexCount = 0;

    std::uint32_t instanceOffset = 0;
    std::uint32_t characterInstanceCount = 0;
    std::uint32_t dropInstanceOffset = 0;
    std::uint32_t dropInstanceCount = 0;

    CudaSharedFuture<WorldGenerationOutput> pendingGeneration;
    std::uint64_t lastWorldVersion = 0;
};

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
    vk::DescriptorSet staticMeshDescriptorSet;

    VulkanCudaInteropBuffer terrainHeaderBuffer;
    VulkanCudaInteropBuffer terrainVoxelBuffer;
    VulkanDeviceBuffer staticMeshVertexBuffer;
    VulkanHostBuffer instanceBuffer;
    VulkanHostBuffer paramsBuffer;

    vk::DeviceSize terrainVoxelBufferCapacityBytes = 0;
    cudaStream_t observationConversionStream = nullptr;
    std::uint32_t batchSize = 1;
};

namespace {

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
            .pCode = reinterpret_cast<const std::uint32_t*>(bytes.data()),
        };
        return vkCheck(vk.device().createShaderModuleUnique(info), "VkDevice::createShaderModule");
    }

    void createRenderPass(Vulkan& vk, Renderer::VulkanState& state)
    {
        const vk::AttachmentDescription colorAttachment {
            .format = state.colorFormat,
            .samples = vk::SampleCountFlagBits::e1,
            .loadOp = vk::AttachmentLoadOp::eDontCare,
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
            .commandBufferCount = static_cast<std::uint32_t>(state.frames.size()),
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
            = static_cast<vk::DeviceSize>(s_maxTerrainVoxels) * state.batchSize * VoxelSize;
        state.terrainVoxelBuffer = VulkanCudaInteropBuffer(
            vk, state.terrainVoxelBufferCapacityBytes, vk::BufferUsageFlagBits::eStorageBuffer);
        state.staticMeshVertexBuffer = VulkanDeviceBuffer(vk,
            static_cast<vk::DeviceSize>(PigMesh::verticesCount() + Drop::s_meshVertexCount) * sizeof(MeshVertex),
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
        state.staticMeshDescriptorSet = descriptorSets[1];

        const vk::DescriptorBufferInfo terrainHeaderInfo = state.terrainHeaderBuffer.info();
        const vk::DescriptorBufferInfo voxelInfo = state.terrainVoxelBuffer.info();
        const vk::DescriptorBufferInfo pigVertexInfo = state.staticMeshVertexBuffer.info();
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
                .dstSet = state.staticMeshDescriptorSet,
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo = &pigVertexInfo,
            },
            vk::WriteDescriptorSet {
                .dstSet = state.staticMeshDescriptorSet,
                .dstBinding = 1,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo = &instanceInfo,
            },
            vk::WriteDescriptorSet {
                .dstSet = state.staticMeshDescriptorSet,
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
            .width = static_cast<std::uint32_t>(config.width),
            .height = static_cast<std::uint32_t>(config.height),
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
    , m_worldGenerator(std::make_unique<WorldGenerator>(WorldGenerationConfig { .halfExtent = TerrainMeshHalfExtent }))
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
    m_instances.reserve(MaxEntityInstances);
    m_observation.reset(m_state->renderExtent.width, m_state->renderExtent.height, m_batchSize);
    m_observation.setVersion(m_observationVersion);
    if (!m_entityMeshesUploaded) {
        PigMesh pigMeshGenerator;
        const std::span<MeshVertex> pigMesh = pigMeshGenerator.generate();
        const std::array<MeshVertex, Drop::s_meshVertexCount> dropMesh = Drop::makeMesh();

        m_pigMeshVertexCount = static_cast<std::uint32_t>(pigMesh.size());
        m_dropMeshVertexOffset = m_pigMeshVertexCount;
        m_dropMeshVertexCount = static_cast<std::uint32_t>(dropMesh.size());

        std::vector<MeshVertex> entityMesh;
        entityMesh.reserve(pigMesh.size() + dropMesh.size());
        entityMesh.insert(entityMesh.end(), pigMesh.begin(), pigMesh.end());
        entityMesh.insert(entityMesh.end(), dropMesh.begin(), dropMesh.end());

        m_state->staticMeshVertexBuffer.uploadSync(
            m_vk, *m_state->commandPool, entityMesh.data(), sizeof(MeshVertex) * entityMesh.size());
        m_entityMeshesUploaded = true;
    }
    for (std::uint32_t i = 0; i < m_batchSize; ++i) {
        RenderSlot& slot = m_slots[i];
        slot.terrainVoxelOffset = i * s_maxTerrainVoxels;
        slot.pigVertexCount = m_pigMeshVertexCount;
        slot.dropVertexOffset = m_dropMeshVertexOffset;
        slot.dropVertexCount = m_dropMeshVertexCount;
        slot.instanceOffset = i * MaxEntityInstances;
    }
}

Renderer::RenderParams Renderer::buildRenderParams(const AgentState& agent, const World& world) const
{
    const Vec3 origin = agent.position + Vec3 { 0.0f, EyeHeight, 0.0f };
    const Vec3 forward = cameraForward(agent.yaw, agent.pitch);
    const Vec3 right = glm::normalize(Vec3 { std::cos(agent.yaw), 0.0f, -std::sin(agent.yaw) });
    const Vec3 up = glm::normalize(glm::cross(forward, right));
    const Vec3 skyColor = skyColorAtTime(world.dayTime());
    const std::uint8_t skyLight = skyLightAtTime(world.dayTime());
    assert(skyLight >= 0 && skyLight <= 15);

    return {
        .origin = origin,
        .forward = forward,
        .right = right,
        .up = up,
        .worldOrigin = {},
        .viewportWidth = static_cast<std::int32_t>(m_state->renderExtent.width),
        .regionOrigin = {},
        .viewportHeight = static_cast<std::int32_t>(m_state->renderExtent.height),
        .frameInfo = {
            .animationTimeMs = static_cast<std::uint32_t>(world.logicalTimeMs()),
        },
        .projectionInfo = {
            .farPlane = 48.0f,
            .fovRadians = Pi / 2.25f,
            .fogStart = 10.0f,
            .fogEnd = 28.0f,
        },
        .skyInfo = {
            .skyColor = skyColor,
            .skyLightDimming = 15U - skyLight,
            .skyLightDirection = skyLightDirectionAtTime(world.dayTime()),
        },
    };
}

namespace {

    struct BlockLightAdapter {
        static float light(const BlockInfo& b) { return b.blockLight / 15.0f; }
    };

    struct SkyLightAdapter {
        static float light(const BlockInfo& b) { return b.skyLight / 15.0f; }
    };

    template <typename Adapter>
    float lightAtPoint(Vec3 pos, const World& world)
    {
        const Vec3 p0f = glm::floor(pos);
        const Vec3 delta = pos - p0f;
        const IVec3 p0 = p0f;

        if (!world.isInsideLoadedCache(p0) || !world.isInsideLoadedCache(p0 + IVec3(1, 1, 1))) [[unlikely]]
            return 0.0f;

        const auto airLight = [&world](IVec3 pos) -> std::optional<float> {
            const BlockInfo* b = world.block(pos);
            if (!b || isSolidBlock(b->blockType))
                return std::nullopt;
            return Adapter::light(*b);
        };

        const auto mix = [](std::optional<float> x, std::optional<float> y, float a) -> std::optional<float> {
            if (x) {
                if (y)
                    return glm::mix(*x, *y, a);
                return *x;
            }
            if (y)
                return *y;
            return std::nullopt;
        };

        const std::optional<float> l000 = airLight(p0);
        const std::optional<float> l100 = airLight({ p0.x + 1, p0.y, p0.z });
        const std::optional<float> l010 = airLight({ p0.x, p0.y + 1, p0.z });
        const std::optional<float> l110 = airLight({ p0.x + 1, p0.y + 1, p0.z });
        const std::optional<float> l001 = airLight({ p0.x, p0.y, p0.z + 1 });
        const std::optional<float> l101 = airLight({ p0.x + 1, p0.y, p0.z + 1 });
        const std::optional<float> l011 = airLight({ p0.x, p0.y + 1, p0.z + 1 });
        const std::optional<float> l111 = airLight({ p0.x + 1, p0.y + 1, p0.z + 1 });

        // interpolate along X axis

        // front 4 corners
        const std::optional<float> x00 = mix(l000, l100, delta.x);
        const std::optional<float> x10 = mix(l010, l110, delta.x);

        // back 4 corners
        const std::optional<float> x01 = mix(l001, l101, delta.x);
        const std::optional<float> x11 = mix(l011, l111, delta.x);

        // interpolate along Y axis
        const std::optional<float> y0 = mix(x00, x10, delta.y);
        const std::optional<float> y1 = mix(x01, x11, delta.y);

        // interpolate along Z axis
        std::optional<float> result = mix(y0, y1, delta.z);

        return result ? *result : 0.0f;
    }

} // namespace

void Renderer::uploadInstances(std::size_t slotIndex, const World& world)
{
    RenderSlot& slot = m_slots[slotIndex];
    m_instances.clear();
    m_instances.push_back({});
    for (const std::unique_ptr<NPC>& character : world.characters()) {
        if (!character->isAlive() || character->kind() == CharacterKind::Agent)
            continue;
        const Vec3 position = character->position();
        const Vec3 velocity = character->velocity();
        const Vec3 direction = character->direction();
        const float yaw = std::atan2(direction.x, direction.z);
        const float blockLight = lightAtPoint<BlockLightAdapter>(position, world);
        const float skyLight = lightAtPoint<SkyLightAdapter>(position, world);
        m_instances.push_back({
            .position = position,
            .yaw = yaw,
            .velocity = { velocity.x, velocity.y, velocity.z, 0.0f },
            .kind = renderEntityKindId(character->kind()),
            .blockLight = blockLight,
            .skyLight = skyLight,
        });
        if (m_instances.size() >= MaxEntityInstances)
            break;
    }
    slot.characterInstanceCount = static_cast<std::uint32_t>(m_instances.size() - 1U);

    slot.dropInstanceOffset = static_cast<std::uint32_t>(m_instances.size());
    for (const Drop& drop : world.drops()) {
        if (m_instances.size() >= MaxEntityInstances)
            break;

        const IVec3 blockPosition = drop.position();
        const Vec3 position {
            static_cast<float>(blockPosition.x) + 0.5f,
            static_cast<float>(blockPosition.y) + 0.35f,
            static_cast<float>(blockPosition.z) + 0.5f,
        };
        const float blockLight = lightAtPoint<BlockLightAdapter>(position, world);
        const float skyLight = lightAtPoint<SkyLightAdapter>(position, world);
        const float animationPhase = static_cast<float>(drop.creationTime() % 10000U) * 0.001f;
        m_instances.push_back({
            .position = position,
            .yaw = animationPhase,
            .velocity = {},
            .kind = renderDropKindId(drop.item().type()),
            .blockLight = blockLight,
            .skyLight = skyLight,
        });
    }
    slot.dropInstanceCount = static_cast<std::uint32_t>(m_instances.size()) - slot.dropInstanceOffset;

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
        {}, // color unused, loadOp = DontCare (will be cleared per env later)
        vk::ClearDepthStencilValue(1.0f, 0),
    };
    const vk::Rect2D renderArea {
        .extent = m_state->renderExtent,
    };
    const vk::RenderPassBeginInfo renderPassInfo {
        .renderPass = *m_state->renderPass,
        .framebuffer = *offscreenFrame.framebuffer,
        .renderArea = renderArea,
        .clearValueCount = static_cast<std::uint32_t>(std::size(clearValues)),
        .pClearValues = clearValues,
    };
    commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

    const auto pushConstants = [this, commandBuffer](std::uint32_t envIndex) {
        DrawPushConstants pushConstants {
            .envIndex = envIndex,
            .layerIndex = envIndex,
        };
        commandBuffer.pushConstants<DrawPushConstants>(
            *m_state->pipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, pushConstants);
    };

    for (std::uint32_t envIndex = 0; envIndex < m_batchSize; ++envIndex) {
        const auto& skyColor = m_renderParams[envIndex].skyInfo.skyColor;
        const vk::ClearAttachment clearAttachment {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .colorAttachment = 0,
            .clearValue = vk::ClearColorValue(std::array<float, 4> {
                skyColor.r,
                skyColor.g,
                skyColor.b,
                1.0f,
            }),
        };

        const vk::ClearRect clearRect {
            .rect = {
                .offset = { 0, 0 },
                .extent = m_state->renderExtent,
            },
            .baseArrayLayer = envIndex,
            .layerCount = 1,
        };
        commandBuffer.clearAttachments(clearAttachment, clearRect);
    }

    bool pipelineBound = false;
    for (std::uint32_t envIndex = 0; envIndex < m_batchSize; ++envIndex) {
        RenderSlot& slot = m_slots[envIndex];
        pushConstants(envIndex);
        if ((slot.pigVertexCount > 0 && slot.characterInstanceCount > 0)
            || (slot.dropVertexCount > 0 && slot.dropInstanceCount > 0)) {
            if (!pipelineBound) {
                commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *m_state->meshPipeline);
                pipelineBound = true;
            }
            commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *m_state->pipelineLayout, 0, 1,
                &m_state->staticMeshDescriptorSet, 0, nullptr);
            if (slot.characterInstanceCount > 0)
                commandBuffer.draw(slot.pigVertexCount, slot.characterInstanceCount, 0, slot.instanceOffset + 1U);
            if (slot.dropInstanceCount > 0)
                commandBuffer.draw(slot.dropVertexCount, slot.dropInstanceCount, slot.dropVertexOffset,
                    slot.instanceOffset + slot.dropInstanceOffset);
        }
    }

    // TODO implement
    // some slots may still be pending generation at this point, but we have to draw the ones that are ready, to avoid
    // stalling the GPU for too long. We will draw the pending ones in the next frame.
    // std::vector<std::uint32_t> postponedSlots;

    pipelineBound = false;
    for (std::uint32_t envIndex = 0; envIndex < m_batchSize; ++envIndex) {
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
            constexpr std::uint32_t verticesPerVoxel = 36;
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

    const std::uint64_t signalValues[] = { m_observationVersion };
    const vk::TimelineSemaphoreSubmitInfo timelineSubmitInfo {
        .signalSemaphoreValueCount = std::size(signalValues),
        .pSignalSemaphoreValues = signalValues,
    };
    const vk::Semaphore signalSemaphore = offscreenFrame.frameCompletionSemaphore.vkSemaphore();
    const vk::SubmitInfo submitInfo[] = { {
        .pNext = &timelineSubmitInfo,
        .commandBufferCount = 1,
        .pCommandBuffers = &commandBuffer,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &signalSemaphore,
    } };
    vkCheck(m_vk.graphicsQueue().submit(submitInfo, *offscreenFrame.fence), "VkQueue::submit");

    cudaStream_t stream = m_state->observationConversionStream;
    offscreenFrame.frameCompletionSemaphore.enqueueWait(stream, offscreenFrame.observationVersion);
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
    uploadInstances(slotIndex, world);

    const IVec3 agentBlock = floorToInt32(agent.position);
    constexpr std::int32_t MeshCacheStride = 12;
    RenderSlot& slot = m_slots[slotIndex];
    const IVec3 meshDelta = glm::abs(agentBlock - slot.lastMeshCenter);
    const bool agentLeftMeshCache
        = meshDelta.x > MeshCacheStride || meshDelta.y > MeshCacheStride || meshDelta.z > MeshCacheStride;
    if (slot.lastWorldVersion != world.version() || agentLeftMeshCache) {
        for (OffscreenFrame& frame : m_state->frames) {
            if (frame.worldGenerationLastWaitVersion >= frame.observationVersion)
                continue;
            frame.frameCompletionSemaphore.enqueueWait(m_worldGenerator->stream(), frame.observationVersion);
            frame.worldGenerationLastWaitVersion = frame.observationVersion;
        }

        WorldGenerationBuffers buffers;
        buffers.terrain.header = &m_state->terrainHeaderBuffer.cudaPtr<TerrainHeader>()[slotIndex];
        buffers.terrain.maxVoxels = s_maxTerrainVoxels;
        buffers.terrain.voxels = reinterpret_cast<Voxel*>(
            m_state->terrainVoxelBuffer.cudaPtr<std::byte>() + slot.terrainVoxelOffset * VoxelSize);
        buffers.cpuCache = world.borrowGenerationBuffers();

        CudaSharedFuture<WorldGenerationOutput> generation
            = m_worldGenerator->generate(world, agent, std::move(buffers)).share();
        world.updateGeneration(generation);
        slot.pendingGeneration = std::move(generation);
        slot.lastWorldVersion = world.version();
        slot.lastMeshCenter = agentBlock;
    }
}

} // namespace blocklab
