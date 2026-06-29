#include "CliParsing.h"

#include <blocklab/environment/Environment.h>
#include <blocklab/gpu/vulkan/GLFWInit.h>
#include <blocklab/gpu/vulkan/Vulkan.h>
#include <blocklab/graphics/Display.h>
#include <blocklab/graphics/Renderer.h>

#include <GLFW/glfw3.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string_view>

namespace {

struct AppConfig {
    blocklab::RenderConfig renderConfig;
    std::uint32_t maxSteps = 0;
    std::uint32_t seed = 1;
};

struct MouseLookState {
    bool initialized = false;
    double lastX = 0.0;
    double lastY = 0.0;
    float pendingYawDelta = 0.0f;
    float pendingPitchDelta = 0.0f;
};

struct InputState {
    MouseLookState mouse;
    bool digRequested = false;
    bool frameLimiterToggleRequested = false;
    bool placeRequested = false;
    blocklab::PlacementBlock placementBlock = blocklab::PlacementBlock::Dirt;
};

AppConfig parseAppConfig(int argc, char** argv)
{
    AppConfig config;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--resolution" || arg.starts_with("--resolution=")) {
            const auto parsed
                = blocklab::cli::parseResolution(blocklab::cli::optionValue(i, argc, argv, arg, "--resolution"));
            if (!parsed) [[unlikely]] {
                std::cerr << "Invalid --resolution value. Expected WIDTHxHEIGHT." << std::endl;
                std::exit(EXIT_FAILURE);
            }
            config.renderConfig = *parsed;
        } else if (arg == "--seed" || arg.starts_with("--seed="))
            config.seed = static_cast<std::uint32_t>(
                blocklab::cli::parseInt<std::uint64_t>(blocklab::cli::optionValue(i, argc, argv, arg, "--seed"))
                    .value_or(config.seed));
        else if (arg == "--max-steps" || arg.starts_with("--max-steps=")) {
            const auto maxSteps
                = blocklab::cli::parseInt<std::uint64_t>(blocklab::cli::optionValue(i, argc, argv, arg, "--max-steps"));
            if (!maxSteps || *maxSteps > std::numeric_limits<std::uint32_t>::max()) [[unlikely]] {
                std::cerr << "Invalid --max-steps value." << std::endl;
                std::exit(EXIT_FAILURE);
            }
            config.maxSteps = static_cast<std::uint32_t>(*maxSteps);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: blocklab [--resolution WIDTHxHEIGHT] [--seed N] [--max-steps N]" << std::endl;
            std::exit(EXIT_SUCCESS);
        } else [[unlikely]] {
            std::cerr << "Unknown argument: " << arg << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
    return config;
}

bool keyDown(GLFWwindow* window, int key) { return glfwGetKey(window, key) == GLFW_PRESS; }

void cursorPositionCallback(GLFWwindow* window, double x, double y)
{
    auto* input = static_cast<InputState*>(glfwGetWindowUserPointer(window));
    if (!input)
        return;

    MouseLookState& mouse = input->mouse;
    if (!mouse.initialized) {
        mouse.initialized = true;
        mouse.lastX = x;
        mouse.lastY = y;
        return;
    }

    constexpr float mouseSensitivity = 0.0022f;
    mouse.pendingYawDelta += static_cast<float>(x - mouse.lastX) * mouseSensitivity;
    mouse.pendingPitchDelta -= static_cast<float>(y - mouse.lastY) * mouseSensitivity;
    mouse.lastX = x;
    mouse.lastY = y;
}

void keyCallback(GLFWwindow* window, int key, int, int action, int)
{
    auto* input = static_cast<InputState*>(glfwGetWindowUserPointer(window));
    if (!input)
        return;

    if (action == GLFW_PRESS && key == GLFW_KEY_1)
        input->placementBlock = blocklab::PlacementBlock::Torch;
    else if (action == GLFW_PRESS && key == GLFW_KEY_2)
        input->placementBlock = blocklab::PlacementBlock::Dirt;
    else if (action == GLFW_PRESS && key == GLFW_KEY_3)
        input->placementBlock = blocklab::PlacementBlock::Stone;
    else if (action == GLFW_PRESS && key == GLFW_KEY_TAB)
        input->frameLimiterToggleRequested = true;
    else if (action == GLFW_RELEASE && key == GLFW_KEY_Q)
        input->digRequested = true;
    else if (action == GLFW_RELEASE && key == GLFW_KEY_E)
        input->placeRequested = true;
}

} // namespace

int main(int argc, char** argv)
{
    const AppConfig appConfig = parseAppConfig(argc, argv);

    blocklab::GLFWInit glfwInit;
    blocklab::VulkanInstance vkInstance(true);
    blocklab::Display display(1, appConfig.renderConfig.width, appConfig.renderConfig.height, vkInstance);
    auto vk = std::make_shared<blocklab::Vulkan>(vkInstance, display.surface());
    display.initialize(vk);
    blocklab::Renderer renderer(*vk, appConfig.renderConfig);
    blocklab::Environment env(renderer, 1, appConfig.maxSteps);
    env.reset(appConfig.seed);
    InputState input;
    GLFWwindow* window = display.window();
    glfwSetWindowUserPointer(window, &input);
    glfwSetCursorPosCallback(window, cursorPositionCallback);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    if (glfwRawMouseMotionSupported())
        glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);

    using Clock = std::chrono::steady_clock;
    constexpr float fixedDt = 1.0f / 60.0f;
    std::uint64_t totalSteps = 0;
    std::uint64_t statsFrames = 0;
    auto previous = Clock::now();
    auto lastStatsAt = previous;
    float accumulator = 0.0f;
    std::uint64_t statsSteps = 0;
    bool frameLimiterEnabled = true;

    while (!display.shouldClose()) {
        display.pollEvents();
        if (keyDown(window, GLFW_KEY_ESCAPE))
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        if (keyDown(window, GLFW_KEY_R))
            env.reset(appConfig.seed);
        if (input.frameLimiterToggleRequested) {
            input.frameLimiterToggleRequested = false;
            frameLimiterEnabled = !frameLimiterEnabled;
            accumulator = 0.0f;
            std::cout << "frame limiter " << (frameLimiterEnabled ? "enabled" : "disabled") << std::endl;
        }

        const auto now = Clock::now();
        const float frameDt = std::chrono::duration<float>(now - previous).count();
        previous = now;
        if (frameLimiterEnabled)
            accumulator += frameDt;

        const auto stepEnvironment = [&] {
            blocklab::AgentAction action;
            action.forward = (keyDown(window, GLFW_KEY_W) ? 1.0f : 0.0f) - (keyDown(window, GLFW_KEY_S) ? 1.0f : 0.0f);
            action.right = (keyDown(window, GLFW_KEY_D) ? 1.0f : 0.0f) - (keyDown(window, GLFW_KEY_A) ? 1.0f : 0.0f);
            action.jump = keyDown(window, GLFW_KEY_SPACE);
            action.dig = input.digRequested;
            action.place = input.placeRequested;
            action.placementBlock = input.placementBlock;
            action.yawDelta
                = (keyDown(window, GLFW_KEY_RIGHT) ? 0.045f : 0.0f) - (keyDown(window, GLFW_KEY_LEFT) ? 0.045f : 0.0f);
            action.yawDelta += input.mouse.pendingYawDelta;
            action.pitchDelta += input.mouse.pendingPitchDelta;
            input.mouse.pendingYawDelta = 0.0f;
            input.mouse.pendingPitchDelta = 0.0f;
            input.digRequested = false;
            input.placeRequested = false;
            const blocklab::AgentAction actions[] { action };
            const blocklab::StepResult result = env.step(actions).front();
            if (result.terminated || result.truncated) {
                if (result.terminated)
                    std::cout << "environment reset due to character death" << std::endl;
                else if (result.truncated)
                    std::cout << "environment reset due to step limit reached" << std::endl;
                env.reset(appConfig.seed);
            }
            ++statsSteps;
            ++totalSteps;
        };

        if (frameLimiterEnabled) {
            while (accumulator >= fixedDt) {
                stepEnvironment();
                accumulator -= fixedDt;
            }
        } else
            stepEnvironment();

        if (display.show(env.observe()))
            ++statsFrames;

        const double statsElapsed = std::chrono::duration<double>(now - lastStatsAt).count();
        if (statsElapsed >= 1.0) {
            std::cout << std::fixed << std::setprecision(1)
                      << "fps=" << static_cast<double>(statsFrames) / statsElapsed
                      << " sim_steps/s=" << static_cast<double>(statsSteps) / statsElapsed
                      << " total_steps=" << totalSteps
                      << " observation_version=" << env.observe().version()
                      << " mode=" << (frameLimiterEnabled ? "fixed" : "unlocked") << '\n';
            lastStatsAt = now;
            statsFrames = 0;
            statsSteps = 0;
        }
    }
}
