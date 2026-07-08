#pragma once

#include <blocklab/inventory/Inventory.h>

#include <cassert>
#include <cstdint>
#include <memory>

namespace blocklab {

class InventoryBatch {
public:
    InventoryBatch(std::uint32_t batchSize)
        : m_data(std::make_unique<const Inventory*[]>(batchSize))
        , m_batchSize(batchSize)
    {}

    std::uint32_t batchSize() const { return m_batchSize; }

    const Inventory& operator[](std::uint32_t batchId) const
    {
        assert(batchId < m_batchSize);
        assert(m_data[batchId]);
        return *m_data[batchId];
    }

    void set(std::uint32_t batchId, const Inventory& inventory)
    {
        assert(batchId < m_batchSize);
        m_data[batchId] = &inventory;
    }

private:
    std::unique_ptr<const Inventory*[]> m_data;
    std::uint32_t m_batchSize;
};

} // namespace blocklab
