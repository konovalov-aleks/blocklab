#pragma once

#include "Item.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace blocklab {

class Inventory {
public:
    enum class SlotId : std::uint8_t {};

    static constexpr std::size_t s_hotbarSlotsCount = 9;
    static constexpr std::size_t s_storageSlotsCount = 27;

    static SlotId hotbarSlotId(unsigned slotIndex);
    static SlotId storageSlotId(unsigned slotIndex);
    static bool isHotbarSlot(SlotId);
    static bool isStorageSlot(SlotId);
    static std::optional<unsigned> hotbarSlotIndex(SlotId);
    static std::optional<unsigned> storageSlotIndex(SlotId);

    std::span<Item> hotbarSlots();
    std::span<const Item> hotbarSlots() const;
    std::span<Item> storageSlots();
    std::span<const Item> storageSlots() const;

    Item& operator[](SlotId);
    const Item& operator[](SlotId) const;

    SlotId activeHotbarSlot() const { return m_activeHotbarSlot; }
    unsigned activeHotbarSlotIndex() const { return *hotbarSlotIndex(m_activeHotbarSlot); }
    void setActiveHotbarSlot(SlotId);

    std::optional<SlotId> findSlotForItem(Item::Type) const;
    std::uint8_t put(Item&, SlotId);

private:
    static std::uint8_t slotLinearIndex(SlotId);

    std::array<Item, s_hotbarSlotsCount + s_storageSlotsCount> m_slots;
    SlotId m_activeHotbarSlot {};
};

} // namespace blocklab
