#include "combo_books.hpp"

#include "quest_player.hpp"

#include <cstring>

namespace mhf::mhfu {
namespace {

// Verified identical across EU (ULES01213), US (ULUS10391), JP (ULJM05500) and FUComplete.
// Pouch: GameSys + 0x6AEA6 (24 slots x 4 bytes {u16 id, u16 count}).
// Item Box: GameSys + 0x4A0 + 0x2F88 = GameSys + 0x3428 (up to 1000 slots x 4 bytes {u16 id, u16 count}).
constexpr std::uint32_t kGameSysPouchBase = 0x6AEA6u;
constexpr std::size_t kPouchSlots = 24u;
constexpr std::uint32_t kGameSysBoxBase = 0x3428u;
constexpr std::size_t kBoxSlots = 1000u;
constexpr std::uint32_t kSlotSize = 4u;
constexpr std::uint8_t kAllFiveBooksMask = 0x1Fu; // bits 0..4 for item IDs 1..5

void scan_slots(const psprecomp::GuestMemory &memory, std::uint32_t base, std::size_t count, std::uint8_t &mask) noexcept {
    const std::uint32_t bytes = static_cast<std::uint32_t>(count) * kSlotSize;
    if (const std::uint8_t *raw = memory.raw_pointer(base, bytes)) {
        for (std::size_t i = 0; i < count && mask != kAllFiveBooksMask; ++i) {
            std::uint16_t id = 0, qty = 0;
            std::memcpy(&id, raw + i * kSlotSize, 2);
            if (id >= kComboBookFirstId && id <= kComboBookLastId) {
                std::memcpy(&qty, raw + i * kSlotSize + 2u, 2);
                if (qty > 0u) mask |= static_cast<std::uint8_t>(1u << (id - kComboBookFirstId));
            }
        }
        return;
    }
    if (!memory.contains(base, bytes)) return;
    for (std::size_t i = 0; i < count && mask != kAllFiveBooksMask; ++i) {
        const std::uint32_t addr = base + static_cast<std::uint32_t>(i) * kSlotSize;
        const std::uint16_t id = memory.load16(addr);
        if (id >= kComboBookFirstId && id <= kComboBookLastId && memory.load16(addr + 2u) > 0u) {
            mask |= static_cast<std::uint8_t>(1u << (id - kComboBookFirstId));
        }
    }
}

} // namespace

bool owns_all_five_combo_books(const psprecomp::GuestMemory &memory, ComboBooksSymbols symbols) noexcept {
    if (!symbols.complete() || !readable_word_address(memory, *symbols.game_sys)) return false;
    const std::uint32_t game_sys = memory.load32(*symbols.game_sys);
    if (game_sys == 0u) return false;

    std::uint8_t mask = 0u;
    scan_slots(memory, game_sys + kGameSysPouchBase, kPouchSlots, mask);
    if (mask != kAllFiveBooksMask) {
        scan_slots(memory, game_sys + kGameSysBoxBase, kBoxSlots, mask);
    }
    return mask == kAllFiveBooksMask;
}

std::uint32_t apply_combo_books_rate(const psprecomp::GuestMemory &memory, ComboBooksSymbols symbols,
                                     std::uint32_t current_rate, bool enabled) noexcept {
    if (!enabled) return current_rate;
    const auto signed_rate = static_cast<std::int32_t>(current_rate);
    if (signed_rate < 0 || signed_rate >= 100) return current_rate;
    return owns_all_five_combo_books(memory, symbols) ? 100u : current_rate;
}

} // namespace mhf::mhfu
