#pragma once

#include <blocklab/environment/observation/ImageBatch.h>
#include <blocklab/gpu/interop/VulkanCudaInteropBuffer.h>
#include <blocklab/gpu/interop/VulkanCudaInteropSemaphore.h>
#include <blocklab/gpu/vulkan/Vulkan.h>
#include <blocklab/gpu/vulkan/VulkanSemaphore.h>

#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <vulkan/vulkan.hpp>

#include <chrono>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace blocklab {

class Display {
public:
    Display(std::uint32_t batchSize, std::uint32_t imageWidth, std::uint32_t imageHeight, VulkanInstance&);
    ~Display();

    Display(const Display&) = delete;
    Display& operator=(const Display&) = delete;

    void initialize(std::shared_ptr<Vulkan>);

    vk::SurfaceKHR surface() const { return *m_surface; }
    GLFWwindow* window() const { return m_window; }

    void pollEvents() const;
    bool shouldClose() const;

    // returns false if the frame was skipped
    bool show(const ImageBatch&);

private:
    using ClockT = std::chrono::steady_clock;

    static constexpr std::uint32_t s_maxFramesPerSecond = 60;
    static constexpr ClockT::duration s_minFrameInterval = std::chrono::duration_cast<ClockT::duration>(
        std::chrono::duration<double>(1.0 / static_cast<double>(s_maxFramesPerSecond)));

    enum ColorFormat {
        RGBA,
        BGRA,
    };

    std::pair<vk::SurfaceFormatKHR, Display::ColorFormat> chooseSurfaceFormat() const;
    vk::PresentModeKHR choosePresentMode() const;
    vk::Extent2D chooseExtent(const vk::SurfaceCapabilitiesKHR&) const;
    void recreateSwapchain();
    void initCmdBuffers();
    void prepareCmdBuffer(vk::CommandBuffer, vk::Image);

    GLFWwindow* m_window = nullptr;

    // Must be declared before Vulkan device resources: C++ destroys members in reverse declaration order,
    // so this keeps the Vulkan device alive until all vk::Unique* resources below are destroyed.
    std::shared_ptr<Vulkan> m_vk;

    vk::UniqueSurfaceKHR m_surface;
    vk::UniqueSwapchainKHR m_swapchain;
    vk::Extent2D m_swapchainExtent;
    std::vector<vk::Image> m_images; // the owner is the swapchain

    VulkanBinarySemaphore m_imageAvailableSemaphore;
    std::vector<VulkanBinarySemaphore> m_imageRenderFinishedSemaphores;
    vk::UniqueFence m_renderFence;

    vk::UniqueCommandPool m_commandPool;
    std::vector<vk::CommandBuffer> m_staticCmdBuffers;

    VulkanCudaInteropBuffer m_buffer;
    cudaStream_t m_conversionStream = nullptr;
    #ifndef CUDA_CPU_FALLBACK_MODE
    VulkanCudaInteropBinarySemaphore m_conversionFinishedSemaphore;
    #endif // !CUDA_CPU_FALLBACK_MODE

    const std::uint32_t m_batchSize;
    const std::uint32_t m_imageWidth;
    const std::uint32_t m_imageHeight;
    const std::uint32_t m_imageGridWidth;
    const std::uint32_t m_imageGridHeight;

    ColorFormat m_surfaceColorFormat = ColorFormat::RGBA;
    ClockT::time_point m_lastFrameTime;
};

} // namespace blocklab
