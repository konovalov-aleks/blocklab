#include <blocklab/cli/CliDisplay.h>

#include <blocklab/inventory/Inventory.h>
#include <blocklab/inventory/Item.h>

#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/screen.hpp>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <deque>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace blocklab {
namespace {

    using namespace std::string_view_literals;

    void hideCursor()
    {
        std::cout << "\x1b[?25l";
    }

    void showCursor()
    {
        std::cout << "\x1b[?25h";
    }

    void moveCursorTo(std::size_t row, std::size_t column)
    {
        std::cout << "\x1b[" << row << ';' << column << 'H';
    }

    void moveCursorHome()
    {
        std::cout << "\x1b[H";
    }

    std::string_view itemLabel(Item::Type type)
    {
        switch (type) {
        case Item::Type::Dirt:
            return "Dirt"sv;
        case Item::Type::Stone:
            return "Stone"sv;
        case Item::Type::Torch:
            return "Torch"sv;
        case Item::Type::COUNT:
            break;
        }
        [[unlikely]]
        return "Unknown"sv;
    }

    std::string slotText(std::size_t index, const Item& item)
    {
        std::string result = std::to_string(index + 1) + ": ";
        if (item.empty())
            return result + "-";

        result += itemLabel(item.type());
        result += " x";
        result += std::to_string(item.count());
        return result;
    }

    ftxui::Element renderSlots(
        std::span<const Item> slots, std::uint32_t columns, std::optional<unsigned> activeSlot = std::nullopt)
    {
        std::vector<ftxui::Element> rows;
        rows.reserve((slots.size() + columns - 1) / columns);
        for (std::size_t first = 0; first < slots.size(); first += columns) {
            std::vector<ftxui::Element> cells;
            cells.reserve(std::min<std::size_t>(columns, slots.size() - first));
            for (std::size_t offset = 0; offset < columns && first + offset < slots.size(); ++offset) {
                const std::size_t index = first + offset;
                ftxui::Element cell = ftxui::text(slotText(index, slots[index]));
                if (activeSlot && index == *activeSlot)
                    cell = cell | ftxui::bold;
                cells.push_back(cell | ftxui::size(ftxui::WIDTH, ftxui::EQUAL, 18));
            }
            rows.push_back(ftxui::hbox(std::move(cells)));
        }
        return ftxui::vbox(std::move(rows));
    }

    ftxui::Element renderInventory(const Inventory& inventory)
    {
        const unsigned activeHotbarIndex = inventory.activeHotbarSlotIndex();
        return ftxui::vbox({
            ftxui::text("Hotbar") | ftxui::bold,
            renderSlots(inventory.hotbarSlots(), 3, activeHotbarIndex),
            ftxui::separator(),
            ftxui::text("Storage") | ftxui::bold,
            renderSlots(inventory.storageSlots(), 3),
        }) | ftxui::border;
    }

    ftxui::Element renderObservationInfo(const Observation& observation)
    {
        const ImageBatch& images = observation.images();
        return ftxui::vbox({
            ftxui::text("Observation") | ftxui::bold,
            ftxui::separator(),
            ftxui::text("version: " + std::to_string(observation.version())),
            ftxui::text("batch: " + std::to_string(observation.batchSize())),
            ftxui::text("image: " + std::to_string(images.width()) + "x" + std::to_string(images.height())),
            ftxui::text("channels: " + std::to_string(images.channels())),
            ftxui::text(std::string("image ready: ") + (images.ready() ? "yes" : "no")),
        }) | ftxui::border;
    }

    ftxui::Element renderLogs(const std::deque<std::string>& logs)
    {
        std::vector<ftxui::Element> lines;
        lines.reserve(std::max<std::size_t>(logs.size(), 1));
        if (logs.empty())
            lines.push_back(ftxui::text("-"));
        else {
            for (const std::string& line : logs)
                lines.push_back(ftxui::text(line));
        }
        return ftxui::vbox({
            ftxui::text("Logs") | ftxui::bold,
            ftxui::separator(),
            ftxui::vbox(std::move(lines)),
        }) | ftxui::border;
    }

} // namespace

CliDisplay::CliDisplay()
{
    hideCursor();
    std::cout << std::flush;
}

CliDisplay::~CliDisplay()
{
    if (m_lastFrameHeight)
        moveCursorTo(*m_lastFrameHeight + 1, 1);
    showCursor();
    std::cout << std::endl;
}

void CliDisplay::log(std::string_view message)
{
    m_logs.emplace_back(message);
    while (m_logs.size() > s_maxLogLines)
        m_logs.pop_front();
}

void CliDisplay::show(const Observation& observation)
{
    const ClockT::time_point now = ClockT::now();
    if (now - m_lastFrameTime < s_minFrameInterval)
        return;
    m_lastFrameTime = now;

    std::vector<ftxui::Element> inventoryPanels;
    const std::uint32_t visibleInventories = std::min<std::uint32_t>(observation.batchSize(), 2);
    inventoryPanels.reserve(visibleInventories);
    for (std::uint32_t batchId = 0; batchId < visibleInventories; ++batchId) {
        inventoryPanels.push_back(ftxui::vbox({
            ftxui::text("Environment " + std::to_string(batchId)) | ftxui::bold,
            renderInventory(observation.inventories()[batchId]),
        }));
    }

    ftxui::Element document = ftxui::vbox({
        ftxui::hbox({
            renderObservationInfo(observation) | ftxui::flex,
            ftxui::vbox(std::move(inventoryPanels)) | ftxui::flex,
        }),
        renderLogs(m_logs),
    });

    ftxui::Screen screen = ftxui::Screen::Create(ftxui::Dimension::Full(), ftxui::Dimension::Fit(document));
    ftxui::Render(screen, document);
    m_lastFrameHeight = screen.dimy();

    moveCursorHome();
    std::cout << screen.ToString() << std::flush;
}

} // namespace blocklab
