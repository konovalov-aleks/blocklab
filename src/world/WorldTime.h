#pragma once

#include <cstdint>

namespace blocklab {

using WorldTime = std::uint16_t;

inline constexpr WorldTime s_ticksPerGameDay = 24000;

inline constexpr WorldTime s_ticksPerSec = 20;
inline constexpr unsigned s_tickPeriodMs = 1000 / s_ticksPerSec;

inline constexpr WorldTime s_sunriseStart = 22300;
inline constexpr WorldTime s_sunriseEnd = 23961;
inline constexpr WorldTime s_sunsetStart = 12000;
inline constexpr WorldTime s_sunsetEnd = 13702;

} // namespace blocklab
