#pragma once

#include <blocklab/environment/observation/Observation.h>

#include <chrono>
#include <cstddef>
#include <deque>
#include <optional>
#include <string>
#include <string_view>

namespace blocklab {

class CliDisplay {
public:
    CliDisplay();
    ~CliDisplay();

    void log(std::string_view);
    void show(const Observation&);

private:
    using ClockT = std::chrono::steady_clock;

    static constexpr std::size_t s_maxLogLines = 8;
    static constexpr ClockT::duration s_minFrameInterval = std::chrono::milliseconds(500);

    std::deque<std::string> m_logs;
    ClockT::time_point m_lastFrameTime;
    std::optional<std::size_t> m_lastFrameHeight;
};

} // namespace blocklab
