#pragma once

#include "Item.h"

#include <array>
#include <optional>
#include <span>

namespace blocklab {

class Inventory {
public:
    using OptItem = std::optional<Item>;
    using SlotId = int;

    static constexpr int s_inventorySlotsCount = 9;
    static constexpr int s_storageSlotsCount = 27;

    std::span<const OptItem> inventorySlots() const;
    std::span<const OptItem> storageSlots() const;

    std::optional<SlotId> findSlotForItem(Item::Type) const;
    const OptItem& operator[](SlotId) const;

private:
    std::array<OptItem, s_inventorySlotsCount + s_storageSlotsCount> m_slots;
};

} // namespace blocklab
