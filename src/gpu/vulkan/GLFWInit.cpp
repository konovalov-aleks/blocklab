#include <blocklab/gpu/vulkan/GLFWInit.h>

#include <blocklab/utility/Error.h>

#include <GLFW/glfw3.h>

namespace blocklab {

int GLFWInit::s_initCounter = 0;

GLFWInit::GLFWInit()
{
    if (s_initCounter++ == 0) {
        if (!glfwInit()) [[unlikely]]
            fatalError("glfwInit failed");
        if (!glfwVulkanSupported()) [[unlikely]]
            fatalError("GLFW reports Vulkan is not supported");
    }
}

GLFWInit::~GLFWInit()
{
    if (--s_initCounter == 0)
        glfwTerminate();
}

} // namespace blocklab