#include <blocklab/inventory/Inventory.h>

#include <cassert>
#include <utility>

namespace blocklab {

Inventory::SlotId Inventory::hotbarSlotId(unsigned slotIndex)
{
    assert(slotIndex < s_hotbarSlotsCount);
    return SlotId { static_cast<std::uint8_t>(slotIndex) };
}

Inventory::SlotId Inventory::storageSlotId(unsigned slotIndex)
{
    assert(slotIndex < s_storageSlotsCount);
    return SlotId { static_cast<std::uint8_t>(s_hotbarSlotsCount + slotIndex) };
}

std::uint8_t Inventory::slotLinearIndex(SlotId slot)
{
    return static_cast<std::uint8_t>(slot);
}

bool Inventory::isHotbarSlot(SlotId slot)
{
    return hotbarSlotIndex(slot).has_value();
}

bool Inventory::isStorageSlot(SlotId slot)
{
    return storageSlotIndex(slot).has_value();
}

std::optional<unsigned> Inventory::hotbarSlotIndex(SlotId slot)
{
    const std::uint8_t rawSlot = slotLinearIndex(slot);
    if (rawSlot >= s_hotbarSlotsCount)
        return std::nullopt;
    return rawSlot;
}

std::optional<unsigned> Inventory::storageSlotIndex(SlotId slot)
{
    const std::uint8_t rawSlot = slotLinearIndex(slot);
    if (rawSlot < s_hotbarSlotsCount || rawSlot >= s_hotbarSlotsCount + s_storageSlotsCount)
        return std::nullopt;
    return static_cast<unsigned>(rawSlot - s_hotbarSlotsCount);
}

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

Item& Inventory::operator[](SlotId slot)
{
    const std::uint8_t slotIndex = slotLinearIndex(slot);
    assert(slotIndex < m_slots.size());
    return m_slots[slotIndex];
}

const Item& Inventory::operator[](SlotId slot) const
{
    const std::uint8_t slotIndex = slotLinearIndex(slot);
    assert(slotIndex < m_slots.size());
    return m_slots[slotIndex];
}

void Inventory::setActiveHotbarSlot(SlotId slot)
{
    assert(isHotbarSlot(slot));
    m_activeHotbarSlot = slot;
}

std::optional<Inventory::SlotId> Inventory::findSlotForItem(Item::Type type) const
{
    const std::uint8_t stackSize = Item::stackSize(type);
    std::optional<SlotId> firstEmptySlot;
    for (std::size_t index = 0; index < m_slots.size(); ++index) {
        const SlotId slotId { static_cast<std::uint8_t>(index) };
        const Item& slot = m_slots[index];
        if (slot.empty()) {
            if (!firstEmptySlot)
                firstEmptySlot = slotId;
            continue;
        }
        if (slot.type() == type && slot.count() < stackSize)
            return slotId;
    }
    return firstEmptySlot;
}

std::uint8_t Inventory::put(Item& item, SlotId sId)
{
    Item& slot = (*this)[sId];
    if (slot.empty()) {
        std::swap(slot, item);
        return slot.count();
    }

    assert(slot.type() == item.type());
    return slot.addFrom(item);
}

} // namespace blocklab
