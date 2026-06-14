#pragma once

#include "Vulkan.h"
#include <blocklab/Observation.h>

#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

namespace blocklab {

class Display {
public:
    Display(int width, int height, VulkanInstance&);
    ~Display();

    Display(const Display&) = delete;
    Display& operator=(const Display&) = delete;

    void initialize(Vulkan&);

    vk::SurfaceKHR surface() const { return *m_surface; }
    GLFWwindow* window() const { return m_window; }

    void pollEvents() const;
    bool shouldClose() const;

    void show(const Observation&);

private:
    vk::SurfaceFormatKHR chooseSurfaceFormat() const;
    vk::PresentModeKHR choosePresentMode() const;
    vk::Extent2D chooseExtent(const vk::SurfaceCapabilitiesKHR&) const;

    GLFWwindow* m_window = nullptr;
    vk::UniqueSurfaceKHR m_surface;

    vk::UniqueSwapchainKHR m_swapchain;
    std::vector<vk::Image> m_images; // the owner is the swapchain

    vk::UniqueSemaphore m_imageAvailableSemaphore;

    Vulkan* m_vk = nullptr;
};

} // namespace blocklab
