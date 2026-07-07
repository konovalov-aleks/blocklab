#pragma once

#include "Item.h"

#include <array>
#include <optional>
#include <span>

namespace blocklab {

class Inventory {
public:
    using SlotId = unsigned;

    static constexpr int s_hotbarSlotsCount = 9;
    static constexpr int s_storageSlotsCount = 27;

    std::span<Item> hotbarSlots();
    std::span<const Item> hotbarSlots() const;
    std::span<Item> storageSlots();
    std::span<const Item> storageSlots() const;

    std::optional<SlotId> findSlotForItem(Item::Type) const;
    std::uint8_t put(Item&, SlotId);

private:
    std::array<Item, s_hotbarSlotsCount + s_storageSlotsCount> m_slots;
};

} // namespace blocklab
