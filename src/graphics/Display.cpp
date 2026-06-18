#include <blocklab/graphics/Display.h>

#include <blocklab/CudaHelpers.h>
#include <blocklab/Error.h>
#include <blocklab/graphics/Vulkan.h>

#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <vector>

namespace blocklab {

void enqueueObservationConversionForDisplay(cudaStream_t, std::uint8_t* destination, float* source,
    std::uint32_t batchSize, std::uint32_t imageWidth, std::uint32_t imageHeight, std::uint32_t imageGridWidth,
    std::uint32_t imageGridHeight, bool outputBGRA /* true -> BGRA, false -> RGBA */
);

Display::Display(
    std::uint32_t batchSize, std::uint32_t imageWidth, std::uint32_t imageHeight, VulkanInstance& vkInstance)
    : m_batchSize(batchSize)
    , m_imageWidth(imageWidth)
    , m_imageHeight(imageHeight)
    , m_imageGridWidth(std::ceil(std::sqrt(static_cast<float>(batchSize))))
    , m_imageGridHeight((batchSize + m_imageGridWidth - 1) / m_imageGridWidth)
{
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    std::uint32_t wndWidth = m_imageWidth * m_imageGridWidth;
    std::uint32_t wndHeight = m_imageHeight * m_imageGridHeight;
    if (GLFWmonitor* monitor = glfwGetPrimaryMonitor()) {
        int workAreaX = 0;
        int workAreaY = 0;
        int workAreaWidth = 0;
        int workAreaHeight = 0;
        glfwGetMonitorWorkarea(monitor, &workAreaX, &workAreaY, &workAreaWidth, &workAreaHeight);
        wndWidth = std::min(wndWidth, static_cast<std::uint32_t>(workAreaWidth));
        wndHeight = std::min(wndHeight, static_cast<std::uint32_t>(workAreaHeight));
    }
    m_window = glfwCreateWindow(wndWidth, wndHeight, "BlockLab", nullptr, nullptr);
    if (!m_window) [[unlikely]]
        fatalError("glfwCreateWindow failed");

    VkSurfaceKHR surface;
    vkCheck(glfwCreateWindowSurface(vkInstance.get(), m_window, nullptr, &surface), "glfwCreateWindowSurface");
    m_surface = vk::UniqueSurfaceKHR(surface, vkInstance.get());

    cudaCheck(cudaStreamCreateWithFlags(&m_conversionStream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
}

Display::~Display()
{
    if (m_vk)
        vkCheck(m_vk->device().waitIdle(), "vk::Device::waitIdle");
    cudaCheck(cudaStreamSynchronize(m_conversionStream), "cudaStreamSynchronize display");
    cudaCheck(cudaStreamDestroy(m_conversionStream), "cudaStreamDestroy");
    glfwDestroyWindow(m_window);
}

void Display::initialize(std::shared_ptr<Vulkan> vk)
{
    m_vk = std::move(vk);

    const vk::FenceCreateInfo fenceCreateInfo {
        .flags = vk::FenceCreateFlagBits::eSignaled,
    };
    m_renderFence = vkCheck(m_vk->device().createFenceUnique(fenceCreateInfo), "vk::Device::createFence");

    constexpr int nChannels = 4; // RGBA
    const vk::DeviceSize bufferSize = nChannels * m_imageGridWidth * m_imageWidth * m_imageGridHeight * m_imageHeight;
    m_buffer = VulkanCudaInteropBuffer(*m_vk, bufferSize, vk::BufferUsageFlagBits::eTransferSrc);
    m_conversionFinishedSemaphore = { m_vk->device() };

    recreateSwapchain();
}

void Display::pollEvents() const { glfwPollEvents(); }

bool Display::shouldClose() const { return glfwWindowShouldClose(m_window); }

bool Display::show(const Observation& observation)
{
    assert(m_vk);

    const auto now = ClockT::now();
    if (now - m_lastFrameTime < s_minFrameInterval)
        return false;

    if (observation.batchSize() != m_batchSize || observation.channels() != 3 || observation.height() != m_imageHeight
        || observation.width() != m_imageWidth) [[unlikely]]
        fatalError("Display: observation tensor shape is incompatible");

    const vk::Result fenceStatus = m_vk->device().getFenceStatus(*m_renderFence);
    if (fenceStatus == vk::Result::eNotReady)
        return false; // just skip this frame instead of waiting, it's better for debug visualization
    if (fenceStatus != vk::Result::eSuccess) [[unlikely]]
        fatalError("incorrect display fence status");

    observation.enqueueReadyWait(m_conversionStream);
    enqueueObservationConversionForDisplay(m_conversionStream, m_buffer.cudaPtr<std::uint8_t>(), observation.data(),
        observation.batchSize(), observation.width(), observation.height(), m_imageGridWidth, m_imageGridHeight,
        m_surfaceColorFormat == ColorFormat::BGRA);

    std::uint32_t nextImageIndex = 0;
    const vk::Result acquireResult = m_vk->device().acquireNextImageKHR(*m_swapchain,
        std::numeric_limits<std::uint64_t>::max(), m_imageAvailableSemaphore.vkSemaphore(), nullptr, &nextImageIndex);
    if (acquireResult == vk::Result::eErrorOutOfDateKHR) {
        recreateSwapchain();
        return false;
    }
    if (acquireResult != vk::Result::eSuccess && acquireResult != vk::Result::eSuboptimalKHR) [[unlikely]]
        fatalError("vk::Device::acquireNextImageKHR failed with vk::Result ", static_cast<int>(acquireResult));

    m_conversionFinishedSemaphore.enqueueSignal(m_conversionStream);
    vkCheck(m_vk->device().resetFences(1, &*m_renderFence), "vk::Device::resetFences");

    const vk::PipelineStageFlags waitStages[]
        = { vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer };
    const vk::Semaphore waitSemaphores[]
        = { m_conversionFinishedSemaphore.vkSemaphore(), m_imageAvailableSemaphore.vkSemaphore() };
    const vk::Semaphore signalSemaphores[] = { m_imageRenderFinishedSemaphores[nextImageIndex].vkSemaphore() };
    const vk::SubmitInfo submitInfo[] = {
        {
            .waitSemaphoreCount = std::size(waitSemaphores),
            .pWaitSemaphores = waitSemaphores,
            .pWaitDstStageMask = waitStages,
            .commandBufferCount = 1,
            .pCommandBuffers = &m_staticCmdBuffers[nextImageIndex],
            .signalSemaphoreCount = std::size(signalSemaphores),
            .pSignalSemaphores = signalSemaphores,
        },
    };
    vkCheck(m_vk->graphicsQueue().submit(submitInfo, *m_renderFence), "vk::Queue::submit");

    const vk::Semaphore submitWaitSemaphores[] = { m_imageRenderFinishedSemaphores[nextImageIndex].vkSemaphore() };
    vk::PresentInfoKHR presentInfo {
        .waitSemaphoreCount = std::size(submitWaitSemaphores),
        .pWaitSemaphores = submitWaitSemaphores,
        .swapchainCount = 1,
        .pSwapchains = &*m_swapchain,
        .pImageIndices = &nextImageIndex,
    };
    const vk::Result presentResult = m_vk->graphicsQueue().presentKHR(&presentInfo);
    if (presentResult != vk::Result::eSuccess && presentResult != vk::Result::eSuboptimalKHR
        && presentResult != vk::Result::eErrorOutOfDateKHR) [[unlikely]]
        fatalError("vk::Queue::presentKHR failed with vk::Result ", static_cast<int>(presentResult));

    if (presentResult == vk::Result::eSuboptimalKHR || presentResult == vk::Result::eErrorOutOfDateKHR
        || acquireResult == vk::Result::eSuboptimalKHR)
        recreateSwapchain();

    m_lastFrameTime = now;
    return true;
}

void Display::recreateSwapchain()
{
    assert(m_vk);

    int framebufferWidth = 0;
    int framebufferHeight = 0;
    glfwGetFramebufferSize(m_window, &framebufferWidth, &framebufferHeight);
    if (framebufferWidth == 0 || framebufferHeight == 0) {
        assert(m_swapchain
            && "Initial Display framebuffer must be non-zero; zero-sized framebuffer is valid only "
               "after swapchain creation");
        return;
    }

    vkCheck(m_vk->device().waitIdle(), "vk::Device::waitIdle");

    const vk::SurfaceCapabilitiesKHR capabilities = vkCheck(
        m_vk->physicalDevice().getSurfaceCapabilitiesKHR(*m_surface), "VkPhysicalDevice::getSurfaceCapabilitiesKHR");

    constexpr vk::ImageUsageFlags imageUsage
        = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst;
    if ((capabilities.supportedUsageFlags & imageUsage) != imageUsage)
        fatalError("Swapchain does not support required image usage");

    constexpr vk::CompositeAlphaFlagBitsKHR compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    if (!(capabilities.supportedCompositeAlpha & compositeAlpha))
        fatalError("Display surface does not support opaque composite alpha");

    std::uint32_t imageCount = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount)
        imageCount = capabilities.maxImageCount;

    const auto [surfaceFormat, colorFormat] = chooseSurfaceFormat();
    m_surfaceColorFormat = colorFormat;
    m_swapchainExtent = chooseExtent(capabilities);

    const std::uint32_t queueFamilyIndices[] = {
        m_vk->graphicsQueueFamilyIndex(),
        m_vk->presentQueueFamilyIndex(),
    };
    const bool shared = queueFamilyIndices[0] != queueFamilyIndices[1];

    const vk::SwapchainCreateInfoKHR swapchainCreateInfo = {
        .surface = *m_surface,
        .minImageCount = imageCount,
        .imageFormat = surfaceFormat.format,
        .imageColorSpace = surfaceFormat.colorSpace,
        .imageExtent = m_swapchainExtent,
        .imageArrayLayers = 1,
        .imageUsage = imageUsage,
        .imageSharingMode = shared ? vk::SharingMode::eConcurrent : vk::SharingMode::eExclusive,
        .queueFamilyIndexCount = shared ? 2U : 0U,
        .pQueueFamilyIndices = shared ? queueFamilyIndices : nullptr,
        .preTransform = capabilities.currentTransform,
        .compositeAlpha = compositeAlpha,
        .presentMode = choosePresentMode(),
        .clipped = vk::True,
        .oldSwapchain = m_swapchain ? *m_swapchain : nullptr,
    };

    m_staticCmdBuffers.clear();
    m_commandPool.reset();
    m_images.clear();
    m_imageRenderFinishedSemaphores.clear();

    m_swapchain
        = vkCheck(m_vk->device().createSwapchainKHRUnique(swapchainCreateInfo), "vk::Device::createSwapchainKHR");
    m_images = vkCheck(m_vk->device().getSwapchainImagesKHR(*m_swapchain), "vk::Device::getSwapchainImages");

    m_imageAvailableSemaphore = { m_vk->device() };
    m_imageRenderFinishedSemaphores.resize(m_images.size());
    for (VulkanBinarySemaphore& semaphore : m_imageRenderFinishedSemaphores)
        semaphore = { m_vk->device() };

    initCmdBuffers();
}

std::pair<vk::SurfaceFormatKHR, Display::ColorFormat> Display::chooseSurfaceFormat() const
{
    assert(m_vk);

    const std::vector<vk::SurfaceFormatKHR> formats
        = vkCheck(m_vk->physicalDevice().getSurfaceFormatsKHR(*m_surface), "VkPhysicalDevice::getSurfaceFormatsKHR");

    const std::pair<vk::SurfaceFormatKHR, ColorFormat> preferredFormats[] {
        { { vk::Format::eR8G8B8A8Srgb, vk::ColorSpaceKHR::eSrgbNonlinear }, ColorFormat::RGBA },
        { { vk::Format::eR8G8B8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear }, ColorFormat::RGBA },
        { { vk::Format::eB8G8R8A8Srgb, vk::ColorSpaceKHR::eSrgbNonlinear }, ColorFormat::BGRA },
        { { vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear }, ColorFormat::BGRA },
    };

    for (const auto& wanted : preferredFormats) {
        for (const vk::SurfaceFormatKHR& available : formats) {
            if (available.format == wanted.first.format && available.colorSpace == wanted.first.colorSpace)
                return wanted;
        }
    }
    fatalError("Display surface does not support RGBA/BGRA 8-bit formats");
}

vk::PresentModeKHR Display::choosePresentMode() const
{
    assert(m_vk);

    const std::vector<vk::PresentModeKHR> modes = vkCheck(
        m_vk->physicalDevice().getSurfacePresentModesKHR(*m_surface), "VkPhysicalDevice::getSurfacePresentModesKHR");

    const vk::PresentModeKHR preferredFormats[]
        = { vk::PresentModeKHR::eMailbox, vk::PresentModeKHR::eImmediate, vk::PresentModeKHR::eFifoRelaxed };

    for (vk::PresentModeKHR fmt : preferredFormats) {
        auto iter = std::ranges::find(modes, fmt);
        if (iter != modes.end())
            return fmt;
    }
    return vk::PresentModeKHR::eFifo;
}

vk::Extent2D Display::chooseExtent(const vk::SurfaceCapabilitiesKHR& capabilities) const
{
    if (capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max())
        return capabilities.currentExtent;

    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(m_window, &width, &height);
    return {
        std::clamp(
            static_cast<std::uint32_t>(width), capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
        std::clamp(
            static_cast<std::uint32_t>(height), capabilities.minImageExtent.height, capabilities.maxImageExtent.height),
    };
}

void Display::initCmdBuffers()
{
    const vk::CommandPoolCreateInfo poolInfo {
        .queueFamilyIndex = m_vk->graphicsQueueFamilyIndex(),
    };
    m_commandPool = vkCheck(m_vk->device().createCommandPoolUnique(poolInfo), "vk::Device::createCommandPool");

    const vk::CommandBufferAllocateInfo allocInfo {
        .commandPool = *m_commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = static_cast<std::uint32_t>(m_images.size()),
    };
    m_staticCmdBuffers
        = vkCheck(m_vk->device().allocateCommandBuffers(allocInfo), "vk::Device::allocateCommandBuffers");
    assert(m_images.size() == m_staticCmdBuffers.size());
    for (std::size_t i = 0; i < m_images.size(); ++i)
        prepareCmdBuffer(m_staticCmdBuffers[i], m_images[i]);
}

void Display::prepareCmdBuffer(vk::CommandBuffer cmd, vk::Image image)
{
    vkCheck(cmd.begin(vk::CommandBufferBeginInfo {}), "vk::CommandBuffer::begin");

    const vk::ImageSubresourceRange colorRange {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .levelCount = 1,
        .layerCount = 1,
    };

    const vk::ImageMemoryBarrier transferDstBarrier {
        .srcAccessMask = {},
        .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eTransferDstOptimal,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = image,
        .subresourceRange = colorRange,
    };
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer, {}, nullptr,
        nullptr, transferDstBarrier);

    const vk::ClearColorValue clearColor(std::array<float, 4> { 0.0f, 0.0f, 0.0f, 1.0f });
    cmd.clearColorImage(image, vk::ImageLayout::eTransferDstOptimal, clearColor, colorRange);

    const std::uint32_t contentWidth = m_imageWidth * m_imageGridWidth;
    const std::uint32_t contentHeight = m_imageHeight * m_imageGridHeight;
    const std::uint32_t copyWidth = std::min(contentWidth, m_swapchainExtent.width);
    const std::uint32_t copyHeight = std::min(contentHeight, m_swapchainExtent.height);

    const vk::BufferImageCopy copyRegion {
        .bufferRowLength = contentWidth,
        .bufferImageHeight = contentHeight,
        .imageSubresource = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .layerCount = 1,
        },
        .imageExtent = vk::Extent3D { copyWidth, copyHeight, 1 },
    };

    if (copyWidth > 0 && copyHeight > 0)
        cmd.copyBufferToImage(m_buffer.vkBuffer(), image, vk::ImageLayout::eTransferDstOptimal, copyRegion);

    const vk::ImageMemoryBarrier presentBarrier {
        .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
        .dstAccessMask = {},
        .oldLayout = vk::ImageLayout::eTransferDstOptimal,
        .newLayout = vk::ImageLayout::ePresentSrcKHR,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = image,
        .subresourceRange = colorRange,
    };
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eBottomOfPipe, {}, nullptr,
        nullptr, presentBarrier);

    vkCheck(cmd.end(), "vk::CommandBuffer::end");
}

} // namespace blocklab
