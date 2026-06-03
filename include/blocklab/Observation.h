#pragma once

#include <cstdint>
#include <vector>

namespace blocklab {

enum class ObservationDevice {
    None,
    Cpu,
    VulkanSwapchain,
    VulkanImage,
    Cuda,
};

enum class ObservationFormat {
    RGBA8,
    FloatNCHW,
};

class Observation {
public:
    using HandleT = uintptr_t;

    uint32_t width() const { return m_width; }
    uint32_t height() const { return m_height; }
    uint32_t channels() const { return m_channels; }
    ObservationDevice device() const { return m_device; }
    ObservationFormat format() const { return m_format; }
    uint64_t version() const { return m_version; }
    uint32_t batchSize() const { return static_cast<uint32_t>(m_slots.size()); }

    HandleT handle(uint32_t batchIndex = 0) const { return m_slots[batchIndex].handle; }

    void reset(uint32_t width, uint32_t height, uint32_t channels, ObservationDevice device, ObservationFormat format,
        uint32_t batchSize)
    {
        m_width = width;
        m_height = height;
        m_channels = channels;
        m_device = device;
        m_format = format;
        m_slots.resize(batchSize);
    }

    void setVersion(uint64_t version) { m_version = version; }

    void setSlot(uint32_t batchIndex, HandleT handle) { m_slots[batchIndex] = { .handle = handle }; }

private:
    struct Slot {
        HandleT handle = 0;
    };

    uint32_t m_width = 0;
    uint32_t m_height = 0;
    uint32_t m_channels = 4;
    ObservationDevice m_device = ObservationDevice::None;
    ObservationFormat m_format = ObservationFormat::RGBA8;
    uint64_t m_version = 0;
    std::vector<Slot> m_slots;
};

} // namespace blocklab
