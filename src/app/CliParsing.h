#pragma once

#include <blocklab/graphics/Renderer.h>

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace blocklab::cli {

template <class T>
std::optional<T> parseInt(std::string_view text)
{
    T parsed = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), parsed);
    if (result.ec != std::errc {} || result.ptr != text.data() + text.size()) [[unlikely]]
        return std::nullopt;
    return parsed;
}

inline std::optional<double> parseDouble(std::string_view text)
{
    const std::string copy(text);
    char* end = nullptr;
    const double parsed = std::strtod(copy.c_str(), &end);
    if (end != copy.c_str() + copy.size()) [[unlikely]]
        return std::nullopt;
    return parsed;
}

inline std::optional<RenderConfig> parseResolution(std::string_view text)
{
    const std::size_t separator = text.find('x');
    if (separator == std::string_view::npos) [[unlikely]]
        return std::nullopt;

    const auto width = parseInt<std::int32_t>(text.substr(0, separator));
    const auto height = parseInt<std::int32_t>(text.substr(separator + 1));
    if (!width || !height || *width <= 0 || *height <= 0) [[unlikely]]
        return std::nullopt;

    RenderConfig config;
    config.width = *width;
    config.height = *height;
    return config;
}

inline std::optional<std::pair<std::int32_t, std::int32_t>> parseActionSteps(std::string_view text)
{
    const std::size_t separator = text.find(':');
    if (separator == std::string_view::npos) [[unlikely]]
        return std::nullopt;

    const auto minSteps = parseInt<std::int32_t>(text.substr(0, separator));
    const auto maxSteps = parseInt<std::int32_t>(text.substr(separator + 1));
    if (!minSteps || !maxSteps || *minSteps <= 0 || *maxSteps < *minSteps) [[unlikely]]
        return std::nullopt;

    return std::pair { *minSteps, *maxSteps };
}

inline std::string_view optionValue(int& index, int argc, char** argv, std::string_view arg, std::string_view name)
{
    if (arg.size() > name.size() && arg.starts_with(name) && arg[name.size()] == '=')
        return arg.substr(name.size() + 1);
    if (arg == name && index + 1 < argc)
        return argv[++index];

    [[unlikely]] std::fprintf(stderr, "Missing value for %.*s\n", static_cast<int>(name.size()), name.data());
    std::exit(EXIT_FAILURE);
}

} // namespace blocklab::cli
