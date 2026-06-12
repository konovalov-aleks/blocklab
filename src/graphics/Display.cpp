#include <blocklab/graphics/Display.h>

#include <blocklab/Error.h>
#include <blocklab/graphics/Vulkan.h>

#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

namespace blocklab {

Display::Display(int width, int height, Vulkan& vk)
    : m_vk(vk)
{
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    m_window = glfwCreateWindow(width, height, "BlockLab", nullptr, nullptr);
    if (!m_window) [[unlikely]]
        fatalError("glfwCreateWindow failed");

    VkSurfaceKHR surface;
    vkCheck(glfwCreateWindowSurface(vk.instance(), m_window, nullptr, &surface), "glfwCreateWindowSurface");
    m_surface.reset(surface);
}

Display::~Display()
{
    m_surface.reset();
    glfwDestroyWindow(m_window);
}

} // namespace blocklab
