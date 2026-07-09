#include "CliParsing.h"

#if defined(BLOCKLAB_ENABLE_CLI_DISPLAY)
#include <blocklab/cli/CliDisplay.h>
#endif

#include <blocklab/environment/Environment.h>
#include <blocklab/gpu/vulkan/GLFWInit.h>
#include <blocklab/gpu/vulkan/Vulkan.h>
#include <blocklab/graphics/Display.h>
#include <blocklab/graphics/Renderer.h>
#include <blocklab/inventory/Inventory.h>

#include <GLFW/glfw3.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
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
    bool attackRequested = false;
    bool frameLimiterToggleRequested = false;
    bool mouseCaptureToggleRequested = false;
    bool mouseCaptured = true;
    bool useRequested = false;
    std::optional<blocklab::Inventory::SlotId> activeHotbarSlot;
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
    if (!input->mouseCaptured)
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

    if (action == GLFW_PRESS && key >= GLFW_KEY_1 && key <= GLFW_KEY_9)
        input->activeHotbarSlot = blocklab::Inventory::hotbarSlotId(static_cast<unsigned>(key - GLFW_KEY_1));
    else if (action == GLFW_PRESS && key == GLFW_KEY_TAB)
        input->mouseCaptureToggleRequested = true;
    else if (action == GLFW_PRESS && key == GLFW_KEY_GRAVE_ACCENT)
        input->frameLimiterToggleRequested = true;
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int)
{
    auto* input = static_cast<InputState*>(glfwGetWindowUserPointer(window));
    if (!input)
        return;

    if (action == GLFW_RELEASE && button == GLFW_MOUSE_BUTTON_LEFT)
        input->attackRequested = true;
    else if (action == GLFW_RELEASE && button == GLFW_MOUSE_BUTTON_RIGHT)
        input->useRequested = true;
}

void setMouseCaptured(GLFWwindow* window, InputState& input, bool captured)
{
    input.mouseCaptured = captured;
    input.mouse = {};
    glfwSetInputMode(window, GLFW_CURSOR, captured ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
    if (glfwRawMouseMotionSupported())
        glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, captured ? GLFW_TRUE : GLFW_FALSE);
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
#if defined(BLOCKLAB_ENABLE_CLI_DISPLAY)
    blocklab::CliDisplay cliDisplay;
    const auto logMessage = [&cliDisplay](std::string_view message) {
        cliDisplay.log(message);
    };
#else
    const auto logMessage = [](std::string_view message) {
        std::cout << message << std::endl;
    };
#endif
    InputState input;
    GLFWwindow* window = display.window();
    glfwSetWindowUserPointer(window, &input);
    glfwSetCursorPosCallback(window, cursorPositionCallback);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    setMouseCaptured(window, input, true);

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
        if (input.mouseCaptureToggleRequested) {
            input.mouseCaptureToggleRequested = false;
            setMouseCaptured(window, input, !input.mouseCaptured);
            logMessage(input.mouseCaptured ? "mouse captured" : "mouse released");
        }
        if (input.frameLimiterToggleRequested) {
            input.frameLimiterToggleRequested = false;
            frameLimiterEnabled = !frameLimiterEnabled;
            accumulator = 0.0f;
            logMessage(frameLimiterEnabled ? "frame limiter enabled" : "frame limiter disabled");
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
            action.attack = input.attackRequested;
            action.use = input.useRequested;
            action.activeHotbarSlot = input.activeHotbarSlot;
            action.yawDelta
                = (keyDown(window, GLFW_KEY_RIGHT) ? 0.045f : 0.0f) - (keyDown(window, GLFW_KEY_LEFT) ? 0.045f : 0.0f);
            action.yawDelta += input.mouse.pendingYawDelta;
            action.pitchDelta += input.mouse.pendingPitchDelta;
            input.mouse.pendingYawDelta = 0.0f;
            input.mouse.pendingPitchDelta = 0.0f;
            input.attackRequested = false;
            input.useRequested = false;
            input.activeHotbarSlot.reset();
            const blocklab::AgentAction actions[] { action };
            const blocklab::StepResult result = env.step(actions).front();
            if (result.terminated || result.truncated) {
                if (result.terminated)
                    logMessage("environment reset due to character death");
                else if (result.truncated)
                    logMessage("environment reset due to step limit reached");
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

        if (display.show(env.observe().images()))
            ++statsFrames;
#if defined(BLOCKLAB_ENABLE_CLI_DISPLAY)
        cliDisplay.show(env.observe());
#endif

        const double statsElapsed = std::chrono::duration<double>(now - lastStatsAt).count();
        if (statsElapsed >= 1.0) {
            std::ostringstream stats;
            stats << std::fixed << std::setprecision(1)
                << "fps=" << static_cast<double>(statsFrames) / statsElapsed
                << " sim_steps/s=" << static_cast<double>(statsSteps) / statsElapsed
                << " total_steps=" << totalSteps
                << " observation_version=" << env.observe().version()
                << " mode=" << (frameLimiterEnabled ? "fixed" : "unlocked");
            logMessage(stats.str());
            lastStatsAt = now;
            statsFrames = 0;
            statsSteps = 0;
        }
    }
}
