#pragma once

namespace blocklab {

class GLFWInit {
public:
    GLFWInit();
    ~GLFWInit();

    GLFWInit(const GLFWInit&) = delete;
    GLFWInit& operator=(const GLFWInit&) = delete;

private:
    static int s_initCounter;
};

} // namespace blocklab
