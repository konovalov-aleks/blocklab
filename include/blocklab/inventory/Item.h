#pragma once

#include <algorithm>
#include <cassert>
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

    Item() = default;

    Item(Type type, std::uint8_t count = 1)
        : m_type(type)
        , m_count(count)
    {
        assert(count <= stackSize(type));
    }

    Item(const Item&) = default;
    Item& operator=(const Item&) = default;

    bool empty() const { return m_count == 0; }

    Type type() const { assert(!empty()); return m_type; }
    std::uint8_t count() const { return m_count; }

    // TODO implement properly
    static std::uint8_t stackSize(Type) { return 64; }

    std::uint8_t addFrom(Item& item, std::uint8_t maxCount = 0xFF)
    {
        if (item.empty() || item.type() != type())
            return 0;

        maxCount = std::min(item.count(), maxCount);
        assert(maxCount <= stackSize(type()));
        std::uint8_t count = std::min<std::uint8_t>(stackSize(type()) - m_count, maxCount);
        m_count += count;
        item.m_count -= count;
        return count;
    }

    bool remove(unsigned count)
    {
        if (m_count < count)
            return false;
        m_count = static_cast<std::uint8_t>(m_count - count);
        return true;
    }

private:
    Type m_type = Type::Dirt;
    std::uint8_t m_count = 0;
};

} // namespace blocklab
