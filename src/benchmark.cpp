#include "blocklab/Agent.h"
#include "blocklab/CliParsing.h"
#include "blocklab/Environment.h"
#include "blocklab/Renderer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <string_view>
#include <vector>

namespace {

struct BenchmarkConfig {
    uint64_t steps = 0;
    uint64_t warmupSteps = 128;
    double seconds = 10.0;
    double reportInterval = 1.0;
    uint32_t seed = 1;
    uint32_t batchSize = 1;
    int32_t worldRadiusChunks = 4;
    bool visualize = false;
    double visualFps = 30.0;
    int32_t minActionSteps = 20;
    int32_t maxActionSteps = 160;
    uint64_t initialOverrides = 1000;
    blocklab::RenderConfig renderConfig { .width = 320, .height = 180, .visible = false, .present = false };
    blocklab::RenderConfig visualConfig { .width = 320, .height = 180, .visible = true, .present = true };
};

struct BenchmarkStats {
    uint64_t iterations = 0;
    uint64_t episodes = 0;
    uint64_t blocksCollected = 0;
    uint64_t blocksPlaced = 0;
    double reward = 0.0;
};

class RandomAgent {
public:
    explicit RandomAgent(const BenchmarkConfig& config)
        : m_minActionSteps(config.minActionSteps)
        , m_maxActionSteps(config.maxActionSteps)
    {
    }

    blocklab::AgentAction nextAction(std::mt19937& rng)
    {
        if (m_remainingSteps == 0) {
            std::uniform_int_distribution<int32_t> duration(m_minActionSteps, m_maxActionSteps);
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

        blocklab::AgentAction action = m_action;
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
    int32_t m_minActionSteps = 20;
    int32_t m_maxActionSteps = 160;
    int32_t m_remainingSteps = 0;
    blocklab::AgentAction m_action;
};

[[noreturn]] void usage(int exitCode)
{
    std::printf("Usage: blocklab_benchmark [options]\n"
                "\n"
                "Options:\n"
                "  --seconds N              Run for N seconds, default 10. Ignored when --steps is non-zero.\n"
                "  --steps N                Run fixed number of benchmark iterations, default 0.\n"
                "  --warmup-steps N         Run N untimed warmup iterations before measurement, default 128.\n"
                "  --batch-size N           Number of environments stepped and rendered per iteration, default 1.\n"
                "  --num-envs N             Alias for --batch-size.\n"
                "  --seed N                 RNG seed, default 1.\n"
                "  --world-radius N         World radius in chunks, default 4.\n"
                "  --report-interval N      Progress report interval in seconds, default 1.\n"
                "  --visualize              Open a Vulkan window and render the environment visibly.\n"
                "  --visual-fps N           Max visible presentation rate, default 30.\n"
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
            config.seconds = blocklab::cli::parseDouble(blocklab::cli::optionValue(i, argc, argv, arg, "--seconds"))
                                 .value_or(config.seconds);
        else if (arg == "--steps" || arg.starts_with("--steps="))
            config.steps = blocklab::cli::parseInt<uint64_t>(blocklab::cli::optionValue(i, argc, argv, arg, "--steps"))
                               .value_or(config.steps);
        else if (arg == "--warmup-steps" || arg.starts_with("--warmup-steps="))
            config.warmupSteps
                = blocklab::cli::parseInt<uint64_t>(blocklab::cli::optionValue(i, argc, argv, arg, "--warmup-steps"))
                      .value_or(config.warmupSteps);
        else if (arg == "--batch-size" || arg.starts_with("--batch-size=") || arg == "--num-envs"
            || arg.starts_with("--num-envs=")) {
            const char* const optionName
                = (arg == "--num-envs" || arg.starts_with("--num-envs=")) ? "--num-envs" : "--batch-size";
            const auto batchSize
                = blocklab::cli::parseInt<uint64_t>(blocklab::cli::optionValue(i, argc, argv, arg, optionName));
            if (!batchSize || *batchSize == 0 || *batchSize > static_cast<uint64_t>(UINT32_MAX)) [[unlikely]] {
                std::fprintf(stderr, "Invalid %s value.\n", optionName);
                std::exit(EXIT_FAILURE);
            }
            config.batchSize = static_cast<uint32_t>(*batchSize);
        } else if (arg == "--seed" || arg.starts_with("--seed="))
            config.seed = static_cast<uint32_t>(
                blocklab::cli::parseInt<uint64_t>(blocklab::cli::optionValue(i, argc, argv, arg, "--seed"))
                    .value_or(config.seed));
        else if (arg == "--world-radius" || arg.starts_with("--world-radius=")) {
            const auto worldRadiusChunks
                = blocklab::cli::parseInt<int32_t>(blocklab::cli::optionValue(i, argc, argv, arg, "--world-radius"));
            if (!worldRadiusChunks || *worldRadiusChunks <= 0) [[unlikely]] {
                std::fprintf(stderr, "Invalid --world-radius value.\n");
                std::exit(EXIT_FAILURE);
            }
            config.worldRadiusChunks = *worldRadiusChunks;
        } else if (arg == "--report-interval" || arg.starts_with("--report-interval=")) {
            const auto reportInterval
                = blocklab::cli::parseDouble(blocklab::cli::optionValue(i, argc, argv, arg, "--report-interval"));
            if (!reportInterval || *reportInterval <= 0.0) [[unlikely]] {
                std::fprintf(stderr, "Invalid --report-interval value.\n");
                std::exit(EXIT_FAILURE);
            }
            config.reportInterval = *reportInterval;
        } else if (arg == "--visual-fps" || arg.starts_with("--visual-fps=")) {
            const auto visualFps
                = blocklab::cli::parseDouble(blocklab::cli::optionValue(i, argc, argv, arg, "--visual-fps"));
            if (!visualFps || *visualFps <= 0.0) [[unlikely]] {
                std::fprintf(stderr, "Invalid --visual-fps value.\n");
                std::exit(EXIT_FAILURE);
            }
            config.visualFps = *visualFps;
        } else if (arg == "--action-steps" || arg.starts_with("--action-steps=")) {
            const auto actionSteps
                = blocklab::cli::parseActionSteps(blocklab::cli::optionValue(i, argc, argv, arg, "--action-steps"));
            if (!actionSteps) [[unlikely]] {
                std::fprintf(stderr, "Invalid --action-steps value. Expected MIN:MAX.\n");
                std::exit(EXIT_FAILURE);
            }
            config.minActionSteps = actionSteps->first;
            config.maxActionSteps = actionSteps->second;
        } else if (arg == "--resolution" || arg.starts_with("--resolution=")) {
            auto renderConfig
                = blocklab::cli::parseResolution(blocklab::cli::optionValue(i, argc, argv, arg, "--resolution"));
            if (!renderConfig) [[unlikely]] {
                std::fprintf(stderr, "Invalid --resolution value. Expected WIDTHxHEIGHT.\n");
                std::exit(EXIT_FAILURE);
            }
            renderConfig->visible = config.renderConfig.visible;
            renderConfig->present = config.renderConfig.present;
            config.renderConfig = *renderConfig;
            config.visualConfig.width = renderConfig->width;
            config.visualConfig.height = renderConfig->height;
        } else if (arg == "--initial-overrides" || arg.starts_with("--initial-overrides="))
            config.initialOverrides = blocklab::cli::parseInt<uint64_t>(
                blocklab::cli::optionValue(i, argc, argv, arg, "--initial-overrides"))
                                          .value_or(config.initialOverrides);
        else
            usage(EXIT_FAILURE);
    }
    config.renderConfig.batchSize = config.batchSize;
    config.visualConfig.batchSize = 1;
    return config;
}

uint64_t applyInitialOverrides(blocklab::World& world, const BenchmarkConfig& config, uint32_t seed)
{
    const std::size_t before = world.overrideCount();
    std::mt19937 rng(seed ^ 0x8f3d5b79U);
    std::normal_distribution<float> horizontal(0.0f, 8.0f);
    std::uniform_int_distribution<int32_t> verticalOffset(-8, 4);
    for (uint64_t attempts = 0;
         world.overrideCount() - before < config.initialOverrides && attempts < config.initialOverrides * 64 + 1024;
         ++attempts) {
        const int32_t x = std::clamp(static_cast<int32_t>(std::lround(horizontal(rng))), -28, 28);
        const int32_t z = std::clamp(static_cast<int32_t>(std::lround(horizontal(rng))), -28, 28);
        const int32_t surfaceY
            = std::max(0, blocklab::floorToInt32(world.groundHeight(static_cast<float>(x), static_cast<float>(z))) - 1);
        const int32_t y = std::clamp(surfaceY + verticalOffset(rng), 0, blocklab::Chunk::SizeY - 1);
        const blocklab::Block current = world.getBlock(x, y, z);
        world.setBlock(x, y, z, current == blocklab::Block::Air ? blocklab::Block::Stone : blocklab::Block::Air);
    }
    return static_cast<uint64_t>(world.overrideCount() - before);
}

uint64_t resetEnvironment(blocklab::Environment& env, const BenchmarkConfig& config, uint32_t seed)
{
    env.reset(seed);
    uint64_t applied = 0;
    for (uint32_t i = 0; i < env.batchSize(); ++i)
        applied += applyInitialOverrides(env.mutableWorld(i), config, seed + i * 0x9e3779b9U);
    return applied;
}

} // namespace

int main(int argc, char** argv)
{
    const BenchmarkConfig config = parseArgs(argc, argv);
    std::mt19937 rng(config.seed);
    std::vector<RandomAgent> randomAgents;
    randomAgents.reserve(config.batchSize);
    for (uint32_t i = 0; i < config.batchSize; ++i)
        randomAgents.emplace_back(config);
    std::vector<blocklab::AgentAction> actions(config.batchSize);
    blocklab::Renderer renderer(config.renderConfig);
    blocklab::Environment env(renderer, config.batchSize, config.worldRadiusChunks);
    std::unique_ptr<blocklab::Renderer> visualRenderer;
    if (config.visualize)
        visualRenderer = std::make_unique<blocklab::Renderer>(config.visualConfig);
    uint64_t lastInitialOverridesApplied = resetEnvironment(env, config, config.seed);

    for (uint64_t step = 0; step < config.warmupSteps; ++step) {
        renderer.pollEvents();
        for (uint32_t i = 0; i < config.batchSize; ++i)
            actions[i] = randomAgents[i].nextAction(rng);
        const std::span<const blocklab::StepResult> results = env.step(actions);
        const bool resetNeeded = std::any_of(results.begin(), results.end(),
            [](const blocklab::StepResult& result) { return result.terminated || result.truncated; });
        if (resetNeeded)
            lastInitialOverridesApplied = resetEnvironment(env, config, config.seed + static_cast<uint32_t>(step + 1));
    }

    using Clock = std::chrono::steady_clock;
    const auto startedAt = Clock::now();
    auto lastReportAt = startedAt;
    const auto visualInterval
        = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1.0 / config.visualFps));
    auto lastVisualAt = startedAt - visualInterval;
    uint64_t lastReportIterations = 0;
    BenchmarkStats stats;
    std::vector<int32_t> lastBlocksCollected(config.batchSize);
    std::vector<int32_t> lastBlocksPlaced(config.batchSize);
    for (uint32_t i = 0; i < config.batchSize; ++i) {
        lastBlocksCollected[i] = env.agent(i).state().blocksCollected;
        lastBlocksPlaced[i] = env.agent(i).state().blocksPlaced;
    }

    while (true) {
        renderer.pollEvents();

        if (visualRenderer) {
            visualRenderer->pollEvents();
            if (visualRenderer->shouldClose())
                break;
        }
        for (uint32_t i = 0; i < config.batchSize; ++i)
            actions[i] = randomAgents[i].nextAction(rng);
        const std::span<const blocklab::StepResult> results = env.step(actions);
        ++stats.iterations;
        bool resetNeeded = false;
        for (uint32_t i = 0; i < config.batchSize; ++i) {
            const blocklab::StepResult& result = results[i];
            stats.reward += result.reward;
            const blocklab::AgentState& agent = env.agent(i).state();
            stats.blocksCollected += static_cast<uint64_t>(std::max(0, agent.blocksCollected - lastBlocksCollected[i]));
            stats.blocksPlaced += static_cast<uint64_t>(std::max(0, agent.blocksPlaced - lastBlocksPlaced[i]));
            lastBlocksCollected[i] = agent.blocksCollected;
            lastBlocksPlaced[i] = agent.blocksPlaced;
            if (result.terminated || result.truncated) {
                ++stats.episodes;
                resetNeeded = true;
            }
        }

        if (resetNeeded) {
            lastInitialOverridesApplied
                = resetEnvironment(env, config, config.seed + static_cast<uint32_t>(stats.episodes));
            for (uint32_t i = 0; i < config.batchSize; ++i) {
                lastBlocksCollected[i] = env.agent(i).state().blocksCollected;
                lastBlocksPlaced[i] = env.agent(i).state().blocksPlaced;
            }
        }

        const auto now = Clock::now();
        if (visualRenderer && now - lastVisualAt >= visualInterval) {
            const blocklab::AgentState agents[] = { env.agent(0).state() };
            visualRenderer->renderObservations({ &env.world(0), 1 }, agents);
            lastVisualAt = now;
        }
        const double elapsed = std::chrono::duration<double>(now - startedAt).count();
        if (std::chrono::duration<double>(now - lastReportAt).count() >= config.reportInterval) {
            const double interval = std::chrono::duration<double>(now - lastReportAt).count();
            const uint64_t intervalIterations = stats.iterations - lastReportIterations;
            const uint64_t totalSteps = stats.iterations * config.batchSize;
            const double intervalStepsPerSecond = static_cast<double>(intervalIterations * config.batchSize) / interval;
            std::printf("iterations=%llu steps=%llu elapsed=%.2fs steps/s=%.0f avg_reward=%.4f episodes=%llu\n",
                static_cast<unsigned long long>(stats.iterations), static_cast<unsigned long long>(totalSteps), elapsed,
                intervalStepsPerSecond, stats.reward / static_cast<double>(std::max<uint64_t>(1, totalSteps)),
                static_cast<unsigned long long>(stats.episodes));
            lastReportAt = now;
            lastReportIterations = stats.iterations;
        }
        if ((config.steps > 0 && stats.iterations >= config.steps) || (config.steps == 0 && elapsed >= config.seconds))
            break;
    }

    const double totalSeconds = std::chrono::duration<double>(Clock::now() - startedAt).count();
    const blocklab::Observation& observation = env.observe();
    uint64_t worldOverrides = 0;
    for (uint32_t i = 0; i < config.batchSize; ++i)
        worldOverrides += env.world(i).overrideCount();
    const uint64_t totalSteps = stats.iterations * config.batchSize;
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
                "  world_overrides: %llu\n"
                "  observation_version: %llu\n"
                "  observation_handle: 0x%llx\n",
        config.batchSize, static_cast<unsigned long long>(stats.iterations),
        static_cast<unsigned long long>(totalSteps), static_cast<unsigned long long>(config.warmupSteps), totalSeconds,
        static_cast<double>(totalSteps) / totalSeconds,
        stats.reward / static_cast<double>(std::max<uint64_t>(1, totalSteps)),
        static_cast<unsigned long long>(stats.episodes), static_cast<unsigned long long>(stats.blocksCollected),
        static_cast<unsigned long long>(stats.blocksPlaced), static_cast<unsigned long long>(config.initialOverrides),
        static_cast<unsigned long long>(lastInitialOverridesApplied), static_cast<unsigned long long>(worldOverrides),
        static_cast<unsigned long long>(observation.version()), static_cast<unsigned long long>(observation.handle(0)));
}
