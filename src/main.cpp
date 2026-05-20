#include "blocklab/Environment.h"
#include "blocklab/Renderer.h"

#include <SDL3/SDL.h>

#include <array>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>

namespace {

struct AppConfig {
    blocklab::RenderConfig renderConfig;
    double visualFps = 60.0;
    bool unlocked = false;
};

bool keyDown(const std::array<bool, SDL_SCANCODE_COUNT>& keys, SDL_Scancode code)
{
    return keys[static_cast<std::size_t>(code)];
}

std::optional<int> parsePositiveInt(std::string_view text)
{
    const char* begin = text.data();
    const char* end = text.data() + text.size();
    int parsed = 0;
    const auto result = std::from_chars(begin, end, parsed);
    if (result.ec != std::errc { } || result.ptr != end || parsed <= 0)
        return std::nullopt;
    return parsed;
}

std::optional<double> parseDouble(std::string_view text)
{
    const std::string copy(text);
    char* end = nullptr;
    const double parsed = std::strtod(copy.c_str(), &end);
    if (end != copy.c_str() + copy.size())
        return std::nullopt;
    return parsed;
}

std::optional<blocklab::RenderConfig> parseResolution(std::string_view text)
{
    const std::size_t separator = text.find('x');
    if (separator == std::string_view::npos)
        return std::nullopt;

    const std::optional<int> width = parsePositiveInt(text.substr(0, separator));
    const std::optional<int> height = parsePositiveInt(text.substr(separator + 1));
    if (!width || !height)
        return std::nullopt;

    blocklab::RenderConfig config;
    config.width = *width;
    config.height = *height;
    return config;
}

std::string_view optionValue(int& index, int argc, char** argv, std::string_view arg, std::string_view name)
{
    if (arg.size() > name.size() && arg.starts_with(name) && arg[name.size()] == '=')
        return arg.substr(name.size() + 1);
    if (arg == name && index + 1 < argc)
        return argv[++index];
    std::fprintf(stderr, "Missing value for %.*s\n", static_cast<int>(name.size()), name.data());
    std::exit(EXIT_FAILURE);
}

AppConfig parseAppConfig(int argc, char** argv)
{
    AppConfig config;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--resolution" || arg.starts_with("--resolution=")) {
            if (auto renderConfig = parseResolution(optionValue(i, argc, argv, arg, "--resolution")); !renderConfig) {
                std::fprintf(stderr, "Invalid --resolution value. Expected WIDTHxHEIGHT, for example 640x360.\n");
                std::exit(EXIT_FAILURE);
            } else
                config.renderConfig = *renderConfig;
        } else if (arg == "--visual-fps" || arg.starts_with("--visual-fps=")) {
            if (auto visualFps = parseDouble(optionValue(i, argc, argv, arg, "--visual-fps"));
                !visualFps || *visualFps <= 0.0) {
                std::fprintf(stderr, "Invalid --visual-fps value.\n");
                std::exit(EXIT_FAILURE);
            } else
                config.visualFps = *visualFps;
        } else if (arg == "--unlocked")
            config.unlocked = true;
        else if (arg == "--help" || arg == "-h") {
            std::printf("Usage: blocklab [--resolution WIDTHxHEIGHT] [--visual-fps N] [--unlocked]\n");
            std::exit(EXIT_SUCCESS);
        } else {
            std::fprintf(stderr, "Unknown argument: %.*s\n", static_cast<int>(arg.size()), arg.data());
            std::exit(EXIT_FAILURE);
        }
    }
    return config;
}

} // namespace

int main(int argc, char** argv)
{
    const AppConfig appConfig = parseAppConfig(argc, argv);
    blocklab::RenderConfig renderConfig = appConfig.renderConfig;
    renderConfig.presentToWindow = false;

    if (!SDL_Init(SDL_INIT_VIDEO)) [[unlikely]] {
        std::fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return EXIT_FAILURE;
    }

    SDL_Window* window
        = SDL_CreateWindow("BlockLab RL Environment", renderConfig.width, renderConfig.height, SDL_WINDOW_RESIZABLE);
    if (!window) [[unlikely]] {
        std::fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        SDL_Quit();
        return EXIT_FAILURE;
    }

    int width = renderConfig.width;
    int height = renderConfig.height;
    SDL_GetWindowSize(window, &width, &height);

    {
        blocklab::Environment env(4);
        blocklab::Renderer renderer(window, renderConfig);
        renderer.resize(width, height);
        env.setObservationRenderer(&renderer);
        env.reset();
        std::array<bool, SDL_SCANCODE_COUNT> keys { };
        SDL_SetWindowRelativeMouseMode(window, true);

        using Clock = std::chrono::steady_clock;
        bool running = true;
        uint64_t previous = SDL_GetTicks();
        float accumulator = 0.0f;
        float pendingYawDelta = 0.0f;
        float pendingPitchDelta = 0.0f;
        bool digRequested = false;
        bool placeRequested = false;
        constexpr float fixedDt = 1.0f / 60.0f;
        constexpr float mouseSensitivity = 0.0022f;
        const auto visualInterval
            = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1.0 / appConfig.visualFps));
        auto lastVisualAt = Clock::now() - visualInterval;
        auto lastStatsAt = Clock::now();
        uint64_t statsSteps = 0;
        uint64_t statsFrames = 0;
        uint64_t totalSteps = 0;

        while (running) {
            SDL_Event event;
            while (SDL_PollEvent(&event)) {
                switch (event.type) {
                case SDL_EVENT_QUIT:
                    running = false;
                    break;
                case SDL_EVENT_KEY_DOWN:
                    keys[static_cast<std::size_t>(event.key.scancode)] = true;
                    if (event.key.scancode == SDL_SCANCODE_ESCAPE)
                        running = false;
                    if (event.key.scancode == SDL_SCANCODE_R)
                        env.reset();
                    break;
                case SDL_EVENT_KEY_UP:
                    keys[static_cast<std::size_t>(event.key.scancode)] = false;
                    if (event.key.scancode == SDL_SCANCODE_Q)
                        digRequested = true;
                    if (event.key.scancode == SDL_SCANCODE_E)
                        placeRequested = true;
                    break;
                case SDL_EVENT_MOUSE_MOTION:
                    pendingYawDelta += event.motion.xrel * mouseSensitivity;
                    pendingPitchDelta -= event.motion.yrel * mouseSensitivity;
                    break;
                case SDL_EVENT_WINDOW_RESIZED:
                    renderer.resize(event.window.data1, event.window.data2);
                    break;
                default:
                    break;
                }
            }

            const uint64_t nowTicks = SDL_GetTicks();
            const float frameDt = static_cast<float>(nowTicks - previous) / 1000.0f;
            previous = nowTicks;
            accumulator += frameDt;
            bool consumedPendingInputThisFrame = false;

            const auto stepEnvironment = [&](bool consumePendingInput) {
                blocklab::AgentAction action;
                action.forward
                    = (keyDown(keys, SDL_SCANCODE_W) ? 1.0f : 0.0f) - (keyDown(keys, SDL_SCANCODE_S) ? 1.0f : 0.0f);
                action.right
                    = (keyDown(keys, SDL_SCANCODE_D) ? 1.0f : 0.0f) - (keyDown(keys, SDL_SCANCODE_A) ? 1.0f : 0.0f);
                action.jump = keyDown(keys, SDL_SCANCODE_SPACE);
                action.yawDelta = (keyDown(keys, SDL_SCANCODE_RIGHT) ? 0.045f : 0.0f)
                    - (keyDown(keys, SDL_SCANCODE_LEFT) ? 0.045f : 0.0f);
                if (consumePendingInput) {
                    action.yawDelta += pendingYawDelta;
                    action.pitchDelta += pendingPitchDelta;
                    action.dig = digRequested;
                    action.place = placeRequested;
                    consumedPendingInputThisFrame = true;
                }
                const blocklab::StepResult result = env.step(action);
                if (result.terminated || result.truncated)
                    env.reset();

                ++statsSteps;
                ++totalSteps;
            };

            if (appConfig.unlocked)
                stepEnvironment(true);
            else {
                bool consumedPendingInput = false;
                while (accumulator >= fixedDt) {
                    stepEnvironment(!consumedPendingInput);
                    consumedPendingInput = true;
                    accumulator -= fixedDt;
                }
            }

            const auto now = Clock::now();
            if (now - lastVisualAt >= visualInterval) {
                renderer.present();
                lastVisualAt = now;
                ++statsFrames;
            }

            const double statsElapsed = std::chrono::duration<double>(now - lastStatsAt).count();
            if (statsElapsed >= 1.0) {
                std::printf("fps=%.1f sim_steps/s=%.1f total_steps=%llu observation_version=%llu mode=%s\n",
                    static_cast<double>(statsFrames) / statsElapsed, static_cast<double>(statsSteps) / statsElapsed,
                    static_cast<unsigned long long>(totalSteps), static_cast<unsigned long long>(env.observe().version),
                    appConfig.unlocked ? "unlocked" : "fixed");
                lastStatsAt = now;
                statsFrames = 0;
                statsSteps = 0;
            }

            if (consumedPendingInputThisFrame) {
                pendingYawDelta = 0.0f;
                pendingPitchDelta = 0.0f;
                digRequested = false;
                placeRequested = false;
            }
            if (!appConfig.unlocked)
                SDL_Delay(1);
        }

        env.setObservationRenderer(nullptr);
    }

    SDL_DestroyWindow(window);
    SDL_Quit();
}
