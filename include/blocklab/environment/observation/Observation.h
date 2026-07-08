#pragma once

#include "ImageBatch.h"
#include "InventoryBatch.h"

#include <cassert>
#include <cstdint>

namespace blocklab {

class Observation {
public:
    Observation(std::uint32_t batchSize)
        : m_inventories(batchSize)
        , m_batchSize(batchSize)
    {}

    std::uint32_t batchSize() const { return m_batchSize; }
    std::uint64_t version() const { return m_version; }

    const ImageBatch& images() const { assert(m_images); return *m_images; }
    const InventoryBatch& inventories() const { return m_inventories; }

    InventoryBatch& inventories() { return m_inventories; }
    void setImageBatchRef(const ImageBatch& imageBatch) { m_images = &imageBatch; }
    void setVersion(std::uint64_t version) { m_version = version; }

private:
    std::uint64_t m_version = 0;
    const ImageBatch* m_images = nullptr;
    InventoryBatch m_inventories;
    const std::uint32_t m_batchSize;
};

} // namespace blocklab
