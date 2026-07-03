#include "Lighting.h"

#include <cassert>
#include <cmath>
#include <cstdint>

namespace blocklab {

struct FloatFactorTrait {
    using LightT = float;
    static constexpr LightT s_min = 4.0f / 15.0f;
    static constexpr LightT s_max = 1.0f;
};

struct IntegerLightingTrait {
    using LightT = std::uint8_t;
    static constexpr LightT s_min = 4;
    static constexpr LightT s_max = 15;
};

template <typename Trait>
typename Trait::LightT implSkyLightAtTime(WorldTime t)
{
    assert(t >= 0 && t < s_ticksPerGameDay);

    static_assert(s_sunsetStart < s_sunsetEnd && s_sunsetEnd < s_sunriseStart && s_sunriseStart < s_sunriseEnd,
        "the invariant is broken: logic relies to these relations");

    if (t < s_sunsetStart)
        return Trait::s_max;

    const auto interpolate = [t](WorldTime start, WorldTime end) {
        const auto range = Trait::s_max - Trait::s_min;
        return range * (t - start) / (end - start);
    };

    if (t <= s_sunsetEnd)
        return Trait::s_max - interpolate(s_sunsetStart, s_sunsetEnd);

    if (t <= s_sunriseStart)
        return Trait::s_min;

    if (t <= s_sunriseEnd)
        return Trait::s_min + interpolate(s_sunriseStart, s_sunriseEnd);

    return Trait::s_max;
}

std::uint8_t skyLightAtTime(WorldTime t) { return implSkyLightAtTime<IntegerLightingTrait>(t); }

float skyLightFactorAtTime(WorldTime t) { return implSkyLightAtTime<FloatFactorTrait>(t); }

static Vec3 skyColorAtTime(WorldTime t, float lightFactor)
{
    static_assert(s_sunsetStart < s_sunsetEnd && s_sunriseStart < s_sunriseEnd,
        "the invariant is broken: logic relies to these relations");

    static constexpr Vec3 dayColor = { 0.42f, 0.64f, 0.86f };
    static constexpr Vec3 nightColor = { 0.015f, 0.025f, 0.070f };
    static constexpr Vec3 sunriseColor = { 0.72f, 0.42f, 0.24f };
    static constexpr Vec3 sunsetColor = { 0.75f, 0.34f, 0.20f };

    const float brightness
        = (lightFactor - FloatFactorTrait::s_min) / (FloatFactorTrait::s_max - FloatFactorTrait::s_min);
    Vec3 color = glm::mix(nightColor, dayColor, brightness);

    const auto addTransitionEffect = [&color, t](WorldTime start, WorldTime end, Vec3 targetColor) {
        if (t >= start && t <= end) {
            const float progress = (t - start) / static_cast<float>(end - start);
            const float triangle = 1.0f - std::abs(2.0f * progress - 1.0f);
            color = glm::mix(color, targetColor, glm::smoothstep(0.0f, 1.0f, triangle));
        }
    };
    addTransitionEffect(s_sunsetStart, s_sunsetEnd, sunsetColor);
    addTransitionEffect(s_sunriseStart, s_sunriseEnd, sunriseColor);

    return color;
}

Vec3 skyColorAtTime(WorldTime t) { return skyColorAtTime(t, skyLightFactorAtTime(t)); }

std::pair<Vec3, float> skyColorAndLightFactorAtTime(WorldTime t)
{
    const float lightFactor = skyLightFactorAtTime(t);
    return { skyColorAtTime(t, lightFactor), lightFactor };
}

Vec3 skyLightDirectionAtTime(WorldTime time)
{
    assert(time >= 0 && time < s_ticksPerGameDay);

    constexpr std::int32_t halfDay = s_ticksPerGameDay / 2;

    std::int32_t offset =
        static_cast<std::int32_t>(time) - static_cast<std::int32_t>(s_dayNoon);
    if (offset < -halfDay)
        offset += s_ticksPerGameDay;
    else if (offset >= halfDay)
        offset -= s_ticksPerGameDay;

    const float angle = 2.0f * Pi * static_cast<float>(offset) / static_cast<float>(s_ticksPerGameDay);

    Vec3 sunDirection {
        -std::sin(angle),
        std::cos(angle),
        0.3f,
    };

    if (sunDirection.y < 0.0f)
        sunDirection = -sunDirection;

    return glm::normalize(sunDirection);
}

} // namespace blocklab
