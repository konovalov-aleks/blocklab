#include "CliParsing.h"

#include <blocklab/environment/Environment.h>
#include <blocklab/gpu/vulkan/GLFWInit.h>
#include <blocklab/gpu/vulkan/Vulkan.h>
#include <blocklab/graphics/Display.h>
#include <blocklab/graphics/Renderer.h>
#include <environment/Agent.h>
#include <world/World.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <random>
#include <string_view>
#include <vector>

namespace blocklab {

class EnvironmentInternalAccessBenchmarkHelper {
public:
    static World& world(Environment& env, std::size_t i) { return env.m_worlds[i]; }
    static const AgentState& agentState(Environment& env, std::size_t i) { return env.m_agents[i].state(); }
};

namespace {

    struct BenchmarkConfig {
        std::uint64_t steps = 0;
        std::uint64_t warmupSteps = 128;
        double seconds = 10.0;
        double reportInterval = 1.0;
        std::uint32_t seed = 1;
        std::uint32_t batchSize = 16;
        bool visualize = false;
        std::int32_t minActionSteps = 20;
        std::int32_t maxActionSteps = 160;
        std::uint64_t initialOverrides = 1000;
        RenderConfig renderConfig { .width = 320, .height = 180 };
    };

    struct BenchmarkStats {
        std::uint64_t iterations = 0;
        std::uint64_t episodes = 0;
        std::uint64_t blocksCollected = 0;
        std::uint64_t blocksPlaced = 0;
        double reward = 0.0;
    };

    class RandomAgent {
    public:
        explicit RandomAgent(const BenchmarkConfig& config)
            : m_minActionSteps(config.minActionSteps)
            , m_maxActionSteps(config.maxActionSteps)
        {
        }

        AgentAction nextAction(std::mt19937& rng)
        {
            if (m_remainingSteps == 0) {
                std::uniform_int_distribution<std::int32_t> duration(m_minActionSteps, m_maxActionSteps);
                std::uniform_real_distribution<float> walk(0.45f, 1.0f);
                std::uniform_real_distribution<float> strafe(-0.35f, 0.35f);
                std::normal_distribution<float> turn(0.0f, 0.012f);
                std::bernoulli_distribution walkForward(0.82);
                m_remainingSteps = duration(rng);
                m_action = {};
                m_action.forward = walkForward(rng) ? walk(rng) : 0.0f;
                m_action.right = strafe(rng);
                m_action.yawDelta = turn(rng);
                m_action.pitchDelta = turn(rng) * 0.25f;
            }

            AgentAction action = m_action;
            std::bernoulli_distribution jump(0.004);
            std::bernoulli_distribution dig(0.004);
            std::bernoulli_distribution place(0.003);
            action.jump = jump(rng);
            action.dig = dig(rng);
            action.place = !action.dig && place(rng);
            --m_remainingSteps;
            return action;
        }

    private:
        std::int32_t m_minActionSteps = 20;
        std::int32_t m_maxActionSteps = 160;
        std::int32_t m_remainingSteps = 0;
        AgentAction m_action;
    };

    [[noreturn]] void usage(int exitCode)
    {
        std::printf(
            "Usage: blocklab_benchmark [options]\n"
            "\n"
            "Options:\n"
            "  --seconds N              Run for N seconds, default 10. Ignored when --steps is non-zero.\n"
            "  --steps N                Run fixed number of benchmark iterations, default 0.\n"
            "  --warmup-steps N         Run N untimed warmup iterations before measurement, default 128.\n"
            "  --batch-size N           Number of environments stepped and rendered per iteration, default 16.\n"
            "  --num-envs N             Alias for --batch-size.\n"
            "  --seed N                 RNG seed, default 1.\n"
            "  --report-interval N      Progress report interval in seconds, default 1.\n"
            "  --visualize              Open a Vulkan window and render the environment visibly.\n"
            "  --action-steps A:B       Hold sampled movement/look for A..B steps, default 20:160.\n"
            "  --initial-overrides N    Apply N clustered block overrides after each reset, default 1000.\n"
            "  --resolution WxH         Render/window resolution, default 320x180.\n"
            "  -h, --help               Show this help.\n");
        std::exit(exitCode);
    }

    BenchmarkConfig parseArgs(int argc, char** argv)
    {
        BenchmarkConfig config;
        for (int i = 1; i < argc; ++i) {
            const std::string_view arg(argv[i]);
            if (arg == "-h" || arg == "--help")
                usage(EXIT_SUCCESS);
            else if (arg == "--visualize")
                config.visualize = true;
            else if (arg == "--seconds" || arg.starts_with("--seconds="))
                config.seconds
                    = cli::parseDouble(cli::optionValue(i, argc, argv, arg, "--seconds")).value_or(config.seconds);
            else if (arg == "--steps" || arg.starts_with("--steps="))
                config.steps = cli::parseInt<std::uint64_t>(cli::optionValue(i, argc, argv, arg, "--steps"))
                                   .value_or(config.steps);
            else if (arg == "--warmup-steps" || arg.starts_with("--warmup-steps="))
                config.warmupSteps
                    = cli::parseInt<std::uint64_t>(cli::optionValue(i, argc, argv, arg, "--warmup-steps"))
                          .value_or(config.warmupSteps);
            else if (arg == "--batch-size" || arg.starts_with("--batch-size=") || arg == "--num-envs"
                || arg.starts_with("--num-envs=")) {
                const char* const optionName
                    = (arg == "--num-envs" || arg.starts_with("--num-envs=")) ? "--num-envs" : "--batch-size";
                const auto batchSize = cli::parseInt<std::uint64_t>(cli::optionValue(i, argc, argv, arg, optionName));
                if (!batchSize || *batchSize == 0 || *batchSize > static_cast<std::uint64_t>(UINT32_MAX)) [[unlikely]] {
                    std::fprintf(stderr, "Invalid %s value.\n", optionName);
                    std::exit(EXIT_FAILURE);
                }
                config.batchSize = static_cast<std::uint32_t>(*batchSize);
            } else if (arg == "--seed" || arg.starts_with("--seed="))
                config.seed = static_cast<std::uint32_t>(
                    cli::parseInt<std::uint64_t>(cli::optionValue(i, argc, argv, arg, "--seed")).value_or(config.seed));
            else if (arg == "--report-interval" || arg.starts_with("--report-interval=")) {
                const auto reportInterval = cli::parseDouble(cli::optionValue(i, argc, argv, arg, "--report-interval"));
                if (!reportInterval || *reportInterval <= 0.0) [[unlikely]] {
                    std::fprintf(stderr, "Invalid --report-interval value.\n");
                    std::exit(EXIT_FAILURE);
                }
                config.reportInterval = *reportInterval;
            } else if (arg == "--action-steps" || arg.starts_with("--action-steps=")) {
                const auto actionSteps = cli::parseActionSteps(cli::optionValue(i, argc, argv, arg, "--action-steps"));
                if (!actionSteps) [[unlikely]] {
                    std::fprintf(stderr, "Invalid --action-steps value. Expected MIN:MAX.\n");
                    std::exit(EXIT_FAILURE);
                }
                config.minActionSteps = actionSteps->first;
                config.maxActionSteps = actionSteps->second;
            } else if (arg == "--resolution" || arg.starts_with("--resolution=")) {
                auto renderConfig = cli::parseResolution(cli::optionValue(i, argc, argv, arg, "--resolution"));
                if (!renderConfig) [[unlikely]] {
                    std::fprintf(stderr, "Invalid --resolution value. Expected WIDTHxHEIGHT.\n");
                    std::exit(EXIT_FAILURE);
                }
                config.renderConfig = *renderConfig;
            } else if (arg == "--initial-overrides" || arg.starts_with("--initial-overrides="))
                config.initialOverrides
                    = cli::parseInt<std::uint64_t>(cli::optionValue(i, argc, argv, arg, "--initial-overrides"))
                          .value_or(config.initialOverrides);
            else
                usage(EXIT_FAILURE);
        }
        config.renderConfig.batchSize = config.batchSize;
        return config;
    }

    std::uint64_t applyInitialOverrides(World& world, const BenchmarkConfig& config, std::uint32_t seed)
    {
        const std::size_t before = world.overrideCount();
        std::mt19937 rng(seed ^ 0x8f3d5b79U);
        std::normal_distribution<float> horizontal(0.0f, 8.0f);
        std::uniform_int_distribution<std::int32_t> verticalOffset(-8, 4);
        for (std::uint64_t attempts = 0;
             world.overrideCount() - before < config.initialOverrides && attempts < config.initialOverrides * 64 + 1024;
             ++attempts) {
            const std::int32_t x = std::clamp(static_cast<std::int32_t>(std::lround(horizontal(rng))), -28, 28);
            const std::int32_t z = std::clamp(static_cast<std::int32_t>(std::lround(horizontal(rng))), -28, 28);
            const std::int32_t surfaceY
                = std::max(0, floorToInt32(world.groundHeight(static_cast<float>(x), static_cast<float>(z))) - 1);
            const std::int32_t y = std::clamp(surfaceY + verticalOffset(rng), 0, Chunk::SizeY - 1);
            const Block current = world.getBlock({ x, y, z });
            world.setBlock({ x, y, z }, current == Block::Air ? Block::Stone : Block::Air);
        }
        return static_cast<std::uint64_t>(world.overrideCount() - before);
    }

    std::uint64_t resetEnvironment(Environment& env, const BenchmarkConfig& config, std::uint32_t seed)
    {
        env.reset(seed);
        std::uint64_t applied = 0;
        for (std::uint32_t i = 0; i < env.batchSize(); ++i)
            applied += applyInitialOverrides(
                EnvironmentInternalAccessBenchmarkHelper::world(env, i), config, seed + i * 0x9e3779b9U);
        return applied;
    }

} // namespace
} // namespace blocklab

int main(int argc, char** argv)
{
    using namespace blocklab;

    const BenchmarkConfig config = parseArgs(argc, argv);
    std::mt19937 rng(config.seed);
    std::vector<RandomAgent> randomAgents;
    randomAgents.reserve(config.batchSize);
    for (std::uint32_t i = 0; i < config.batchSize; ++i)
        randomAgents.emplace_back(config);

    std::optional<GLFWInit> glfwInit;
    if (config.visualize)
        glfwInit.emplace();
    VulkanInstance vkInstance(config.visualize);
    std::optional<Display> display;
    vk::SurfaceKHR presentSurface;
    if (config.visualize) {
        display.emplace(config.batchSize, config.renderConfig.width, config.renderConfig.height, vkInstance);
        presentSurface = display->surface();
    }
    auto vk = std::make_shared<Vulkan>(vkInstance, presentSurface);
    if (display)
        display->initialize(vk);
    Renderer renderer(*vk, config.renderConfig);
    Environment env(renderer, config.batchSize);

    using Clock = std::chrono::steady_clock;

    static constexpr int minPollPeriodMs = 100;
    Clock::time_point lastPollTime = Clock::now();
    const auto pollEvents = [&display, &lastPollTime]() {
        if (!display)
            return;
        const Clock::time_point now = Clock::now();
        const auto timeSinceLastPoll = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastPollTime);
        if (timeSinceLastPoll.count() < minPollPeriodMs)
            return;
        display->pollEvents();
        lastPollTime = now;
    };

    std::uint64_t lastInitialOverridesApplied = resetEnvironment(env, config, config.seed);
    std::vector<AgentAction> actions(config.batchSize);
    for (std::uint64_t step = 0; step < config.warmupSteps; ++step) {
        pollEvents();
        for (std::uint32_t i = 0; i < config.batchSize; ++i)
            actions[i] = randomAgents[i].nextAction(rng);
        const std::span<const StepResult> results = env.step(actions);
        const bool resetNeeded = std::any_of(results.begin(), results.end(),
            [](const StepResult& result) { return result.terminated || result.truncated; });
        if (resetNeeded)
            lastInitialOverridesApplied
                = resetEnvironment(env, config, config.seed + static_cast<std::uint32_t>(step + 1));
    }

    const auto startedAt = Clock::now();
    auto lastReportAt = startedAt;
    std::uint64_t lastReportIterations = 0;
    BenchmarkStats stats;
    std::vector<std::int32_t> lastBlocksCollected(config.batchSize);
    std::vector<std::int32_t> lastBlocksPlaced(config.batchSize);
    for (std::uint32_t i = 0; i < config.batchSize; ++i) {
        const auto& agentState = EnvironmentInternalAccessBenchmarkHelper::agentState(env, i);
        lastBlocksCollected[i] = agentState.blocksCollected;
        lastBlocksPlaced[i] = agentState.blocksPlaced;
    }

    while (!display || !display->shouldClose()) {
        pollEvents();
        for (std::uint32_t i = 0; i < config.batchSize; ++i)
            actions[i] = randomAgents[i].nextAction(rng);
        const std::span<const StepResult> results = env.step(actions);
        ++stats.iterations;
        bool resetNeeded = false;
        for (std::uint32_t i = 0; i < config.batchSize; ++i) {
            const StepResult& result = results[i];
            stats.reward += result.reward;
            const AgentState& agent = EnvironmentInternalAccessBenchmarkHelper::agentState(env, i);
            stats.blocksCollected
                += static_cast<std::uint64_t>(std::max(0, agent.blocksCollected - lastBlocksCollected[i]));
            stats.blocksPlaced += static_cast<std::uint64_t>(std::max(0, agent.blocksPlaced - lastBlocksPlaced[i]));
            lastBlocksCollected[i] = agent.blocksCollected;
            lastBlocksPlaced[i] = agent.blocksPlaced;
            if (result.terminated || result.truncated) {
                ++stats.episodes;
                resetNeeded = true;
            }
        }

        if (resetNeeded) {
            lastInitialOverridesApplied
                = resetEnvironment(env, config, config.seed + static_cast<std::uint32_t>(stats.episodes));
            for (std::uint32_t i = 0; i < config.batchSize; ++i) {
                const auto& agentState = EnvironmentInternalAccessBenchmarkHelper::agentState(env, i);
                lastBlocksCollected[i] = agentState.blocksCollected;
                lastBlocksPlaced[i] = agentState.blocksPlaced;
            }
        }

        if (display)
            display->show(env.observe());

        const auto now = Clock::now();
        const double elapsed = std::chrono::duration<double>(now - startedAt).count();
        if (std::chrono::duration<double>(now - lastReportAt).count() >= config.reportInterval) {
            const double interval = std::chrono::duration<double>(now - lastReportAt).count();
            const std::uint64_t intervalIterations = stats.iterations - lastReportIterations;
            const std::uint64_t totalSteps = stats.iterations * config.batchSize;
            const double intervalStepsPerSecond = static_cast<double>(intervalIterations * config.batchSize) / interval;
            std::printf("iterations=%llu steps=%llu elapsed=%.2fs steps/s=%.0f avg_reward=%.4f episodes=%llu\n",
                static_cast<unsigned long long>(stats.iterations), static_cast<unsigned long long>(totalSteps), elapsed,
                intervalStepsPerSecond, stats.reward / static_cast<double>(std::max<std::uint64_t>(1, totalSteps)),
                static_cast<unsigned long long>(stats.episodes));
            lastReportAt = now;
            lastReportIterations = stats.iterations;
        }
        if ((config.steps > 0 && stats.iterations >= config.steps) || (config.steps == 0 && elapsed >= config.seconds))
            break;
    }

    const double totalSeconds = std::chrono::duration<double>(Clock::now() - startedAt).count();
    const Observation& observation = env.observe();
    std::uint64_t worldOverrides = 0;
    for (std::uint32_t i = 0; i < config.batchSize; ++i)
        worldOverrides += EnvironmentInternalAccessBenchmarkHelper::world(env, i).overrideCount();
    const std::uint64_t totalSteps = stats.iterations * config.batchSize;
    std::printf("\nBlockLab benchmark result\n"
                "  batch_size: %u\n"
                "  iterations: %llu\n"
                "  total_steps: %llu\n"
                "  warmup_steps: %llu\n"
                "  total_time_s: %.4f\n"
                "  steps_per_second: %.2f\n"
                "  avg_reward: %.6f\n"
                "  episodes: %llu\n"
                "  blocks_collected: %llu\n"
                "  blocks_placed: %llu\n"
                "  initial_overrides_requested: %llu\n"
                "  initial_overrides_applied: %llu\n"
                "  final_world_overrides: %llu\n"
                "  observation_version: %llu\n",
        config.batchSize, static_cast<unsigned long long>(stats.iterations),
        static_cast<unsigned long long>(totalSteps), static_cast<unsigned long long>(config.warmupSteps), totalSeconds,
        static_cast<double>(totalSteps) / totalSeconds,
        stats.reward / static_cast<double>(std::max<std::uint64_t>(1, totalSteps)),
        static_cast<unsigned long long>(stats.episodes), static_cast<unsigned long long>(stats.blocksCollected),
        static_cast<unsigned long long>(stats.blocksPlaced), static_cast<unsigned long long>(config.initialOverrides),
        static_cast<unsigned long long>(lastInitialOverridesApplied), static_cast<unsigned long long>(worldOverrides),
        static_cast<unsigned long long>(observation.version()));
}
