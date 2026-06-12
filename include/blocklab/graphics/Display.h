#pragma once

#include "GLFWInit.h"
#include "Vulkan.h"

#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

namespace blocklab {

class Display {
public:
    Display(int width, int height, Vulkan&);
    ~Display();

    Display(const Display&) = delete;
    Display& operator=(const Display&) = delete;

    const vk::SurfaceKHR& surface() const { return *m_surface; }

private:
    GLFWInit m_init; // must be first
    GLFWwindow* m_window;
    vk::UniqueSurfaceKHR m_surface;
    Vulkan& m_vk;
};

} // namespace blocklab
