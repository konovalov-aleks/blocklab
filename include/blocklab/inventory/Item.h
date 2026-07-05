#pragma once

#include <cstdint>

namespace blocklab {

class Item {
public:

    enum class Type {
        Dirt,
        Stone,
        Torch,

        COUNT,
    };

    Item(Type type, std::uint8_t count = 1)
        : m_type(type)
        , m_count(count)
    {
        // TODO handle max stuck size
    }

    Item(const Item&) = default;
    Item& operator=(const Item&) = default;

    Type type() const { return m_type; }
    std::uint8_t count() const { return m_count; }

private:
    Type m_type;
    std::uint8_t m_count;
};

} // namespace blocklab
