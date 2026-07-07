#include <blocklab/inventory/Inventory.h>

#include <cassert>
#include <utility>

namespace blocklab {

std::span<Item> Inventory::hotbarSlots()
{
    return { m_slots.data(), s_hotbarSlotsCount };
}

std::span<const Item> Inventory::hotbarSlots() const
{
    return { m_slots.data(), s_hotbarSlotsCount };
}

std::span<Item> Inventory::storageSlots()
{
    return { m_slots.data() + s_hotbarSlotsCount, s_storageSlotsCount };
}

std::span<const Item> Inventory::storageSlots() const
{
    return { m_slots.data() + s_hotbarSlotsCount, s_storageSlotsCount };
}

std::optional<Inventory::SlotId> Inventory::findSlotForItem(Item::Type type) const
{
    const std::uint8_t stackSize = Item::stackSize(type);
    std::optional<SlotId> firstEmptySlot;
    for (SlotId sId = 0; sId < m_slots.size(); ++sId) {
        const Item& slot = m_slots[sId];
        if (slot.empty()) {
            if (!firstEmptySlot)
                firstEmptySlot = sId;
            continue;
        }
        if (slot.type() == type && slot.count() < stackSize)
            return sId;
    }
    return firstEmptySlot;
}

std::uint8_t Inventory::put(Item& item, SlotId sId)
{
    assert(sId < m_slots.size());

    Item& slot = m_slots[sId];
    if (slot.empty()) {
        std::swap(slot, item);
        return slot.count();
    }

    assert(slot.type() == item.type());
    return slot.addFrom(item);
}

} // namespace blocklab
