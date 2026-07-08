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
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
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
        static constexpr std::uint32_t s_defaultMaxSteps = 60 * 60 * 80; // 4 game days.

        std::uint64_t steps = 0;
        std::uint64_t warmupSteps = 128;
        double seconds = 10.0;
        double reportInterval = 1.0;
        std::uint32_t seed = 1;
        std::uint32_t maxSteps = s_defaultMaxSteps;
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
        std::cout
            << "Usage: blocklab_benchmark [options]\n"
            << "\n"
            << "Options:\n"
            << "  --seconds N              Run for N seconds, default 10. Ignored when --steps is non-zero.\n"
            << "  --steps N                Run fixed number of benchmark iterations, default 0.\n"
            << "  --warmup-steps N         Run N untimed warmup iterations before measurement, default 128.\n"
            << "  --batch-size N           Number of environments stepped and rendered per iteration, default 16.\n"
            << "  --num-envs N             Alias for --batch-size.\n"
            << "  --seed N                 RNG seed, default 1.\n"
            << "  --max-steps N            Episode step limit, default 288000 (4 game days), 0 disables.\n"
            << "  --report-interval N      Progress report interval in seconds, default 1.\n"
            << "  --visualize              Open a Vulkan window and render the environment visibly.\n"
            << "  --action-steps A:B       Hold sampled movement/look for A..B steps, default 20:160.\n"
            << "  --initial-overrides N    Apply N clustered block overrides after each reset, default 1000.\n"
            << "  --resolution WxH         Render/window resolution, default 320x180.\n"
            << "  -h, --help               Show this help.\n";
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
                if (!batchSize || *batchSize == 0 || *batchSize > std::numeric_limits<std::uint32_t>::max())
                    [[unlikely]] {
                    std::cerr << "Invalid " << optionName << " value." << std::endl;
                    std::exit(EXIT_FAILURE);
                }
                config.batchSize = static_cast<std::uint32_t>(*batchSize);
            } else if (arg == "--seed" || arg.starts_with("--seed="))
                config.seed = static_cast<std::uint32_t>(
                    cli::parseInt<std::uint64_t>(cli::optionValue(i, argc, argv, arg, "--seed")).value_or(config.seed));
            else if (arg == "--max-steps" || arg.starts_with("--max-steps=")) {
                const auto maxSteps = cli::parseInt<std::uint64_t>(cli::optionValue(i, argc, argv, arg, "--max-steps"));
                if (!maxSteps || *maxSteps > std::numeric_limits<std::uint32_t>::max()) [[unlikely]] {
                    std::cerr << "Invalid --max-steps value." << std::endl;
                    std::exit(EXIT_FAILURE);
                }
                config.maxSteps = static_cast<std::uint32_t>(*maxSteps);
            } else if (arg == "--report-interval" || arg.starts_with("--report-interval=")) {
                const auto reportInterval = cli::parseDouble(cli::optionValue(i, argc, argv, arg, "--report-interval"));
                if (!reportInterval || *reportInterval <= 0.0) [[unlikely]] {
                    std::cerr << "Invalid --report-interval value." << std::endl;
                    std::exit(EXIT_FAILURE);
                }
                config.reportInterval = *reportInterval;
            } else if (arg == "--action-steps" || arg.starts_with("--action-steps=")) {
                const auto actionSteps = cli::parseActionSteps(cli::optionValue(i, argc, argv, arg, "--action-steps"));
                if (!actionSteps) [[unlikely]] {
                    std::cerr << "Invalid --action-steps value. Expected MIN:MAX." << std::endl;
                    std::exit(EXIT_FAILURE);
                }
                config.minActionSteps = actionSteps->first;
                config.maxActionSteps = actionSteps->second;
            } else if (arg == "--resolution" || arg.starts_with("--resolution=")) {
                auto renderConfig = cli::parseResolution(cli::optionValue(i, argc, argv, arg, "--resolution"));
                if (!renderConfig) [[unlikely]] {
                    std::cerr << "Invalid --resolution value. Expected WIDTHxHEIGHT." << std::endl;
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
            const std::int32_t surfaceY = world.terrainHeight({ x, z });
            const std::int32_t y = std::clamp(surfaceY + verticalOffset(rng), World::s_minY, World::s_maxY);
            const Block current = world.blockType({ x, y, z });
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
    Environment env(renderer, config.batchSize, config.maxSteps);

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
            display->show(env.observe().images());

        const auto now = Clock::now();
        const double elapsed = std::chrono::duration<double>(now - startedAt).count();
        if (std::chrono::duration<double>(now - lastReportAt).count() >= config.reportInterval) {
            const double interval = std::chrono::duration<double>(now - lastReportAt).count();
            const std::uint64_t intervalIterations = stats.iterations - lastReportIterations;
            const std::uint64_t totalSteps = stats.iterations * config.batchSize;
            const double intervalStepsPerSecond = static_cast<double>(intervalIterations * config.batchSize) / interval;
            std::cout << "iterations=" << stats.iterations
                      << " steps=" << totalSteps
                      << " elapsed=" << std::fixed << std::setprecision(2) << elapsed << "s"
                      << " steps/s=" << std::fixed << std::setprecision(0) << intervalStepsPerSecond
                      << " avg_reward=" << std::fixed << std::setprecision(4)
                      << stats.reward / static_cast<double>(std::max<std::uint64_t>(1, totalSteps))
                      << " episodes=" << stats.episodes << '\n';
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
    std::cout << "\nBlockLab benchmark result\n"
              << "  batch_size: " << config.batchSize << '\n'
              << "  iterations: " << stats.iterations << '\n'
              << "  total_steps: " << totalSteps << '\n'
              << "  warmup_steps: " << config.warmupSteps << '\n'
              << "  total_time_s: " << std::fixed << std::setprecision(4) << totalSeconds << '\n'
              << "  steps_per_second: " << std::fixed << std::setprecision(2)
              << static_cast<double>(totalSteps) / totalSeconds << '\n'
              << "  avg_reward: " << std::fixed << std::setprecision(6)
              << stats.reward / static_cast<double>(std::max<std::uint64_t>(1, totalSteps)) << '\n'
              << "  episodes: " << stats.episodes << '\n'
              << "  blocks_collected: " << stats.blocksCollected << '\n'
              << "  blocks_placed: " << stats.blocksPlaced << '\n'
              << "  initial_overrides_requested: " << config.initialOverrides << '\n'
              << "  initial_overrides_applied: " << lastInitialOverridesApplied << '\n'
              << "  final_world_overrides: " << worldOverrides << '\n'
              << "  observation_version: " << observation.version() << '\n';
}
