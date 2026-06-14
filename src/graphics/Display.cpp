#include <blocklab/graphics/Display.h>

#include <blocklab/Error.h>
#include <blocklab/graphics/Vulkan.h>

#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

#include <cassert>
#include <iostream>
#include <vector>

namespace blocklab {

Display::Display(int width, int height, VulkanInstance& vkInstance)
{
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    m_window = glfwCreateWindow(width, height, "BlockLab", nullptr, nullptr);
    if (!m_window) [[unlikely]]
        fatalError("glfwCreateWindow failed");

    VkSurfaceKHR surface;
    vkCheck(glfwCreateWindowSurface(vkInstance.get(), m_window, nullptr, &surface), "glfwCreateWindowSurface");
    m_surface = vk::UniqueSurfaceKHR(surface, vkInstance.get());
}

Display::~Display()
{
    m_surface.reset();
    glfwDestroyWindow(m_window);
}

void Display::initialize(Vulkan& vk)
{
    m_vk = &vk;

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

    const vk::SurfaceFormatKHR surfaceFormat = chooseSurfaceFormat();

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
        .imageExtent = chooseExtent(capabilities),
        .imageArrayLayers = 1,
        .imageUsage = imageUsage,
        .imageSharingMode = shared ? vk::SharingMode::eConcurrent : vk::SharingMode::eExclusive,
        .queueFamilyIndexCount = shared ? 2U : 0U,
        .pQueueFamilyIndices = shared ? queueFamilyIndices : nullptr,
        .preTransform = capabilities.currentTransform,
        .compositeAlpha = compositeAlpha,
        .presentMode = choosePresentMode(),
        .clipped = vk::True,
    };
    m_swapchain = vkCheck(m_vk->device().createSwapchainKHRUnique(swapchainCreateInfo), "VkDevice::createSwapchainKHR");
    m_images = vkCheck(m_vk->device().getSwapchainImagesKHR(*m_swapchain), "VkDevice::getSwapchainImages");

    vk::SemaphoreCreateInfo semaphoreCreateInfo;
    m_imageAvailableSemaphore
        = vkCheck(m_vk->device().createSemaphoreUnique(semaphoreCreateInfo), "VkDevice::createSemaphore");
}

void Display::pollEvents() const { glfwPollEvents(); }

bool Display::shouldClose() const { return glfwWindowShouldClose(m_window); }

void Display::show(const Observation&)
{
    assert(m_vk);

    const std::uint32_t nextImageIndex
        = vkCheck(m_vk->device().acquireNextImageKHR(
                      *m_swapchain, std::numeric_limits<std::uint64_t>::max(), *m_imageAvailableSemaphore),
            "VkDevice::acquireNextImageKHR");
    (void)nextImageIndex; // TODO

    // TODO show batch
}

vk::SurfaceFormatKHR Display::chooseSurfaceFormat() const
{
    assert(m_vk);

    const std::vector<vk::SurfaceFormatKHR> formats
        = vkCheck(m_vk->physicalDevice().getSurfaceFormatsKHR(*m_surface), "VkPhysicalDevice::getSurfaceFormatsKHR");

    const vk::SurfaceFormatKHR preferredFormats[] {
        { vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear },
        { vk::Format::eR8G8B8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear },
        { vk::Format::eB8G8R8A8Srgb, vk::ColorSpaceKHR::eSrgbNonlinear },
        { vk::Format::eR8G8B8A8Srgb, vk::ColorSpaceKHR::eSrgbNonlinear },
    };

    for (const vk::SurfaceFormatKHR& wanted : preferredFormats) {
        for (const vk::SurfaceFormatKHR& available : formats) {
            if (available.format == wanted.format && available.colorSpace == wanted.colorSpace)
                return available;
        }
    }
    std::cerr << "warning: BGRA SRGB surface format is unsupported - try to continue using available one" << std::endl;
    return formats.front();
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
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
        return capabilities.currentExtent;

    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(m_window, &width, &height);
    return {
        std::clamp(static_cast<uint32_t>(width), capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
        std::clamp(
            static_cast<uint32_t>(height), capabilities.minImageExtent.height, capabilities.maxImageExtent.height),
    };
}

} // namespace blocklab
