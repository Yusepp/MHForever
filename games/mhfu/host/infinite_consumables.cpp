#include "infinite_consumables.hpp"

#include "quest_player.hpp"

namespace mhf::mhfu {
namespace {

// Verified identical across EU (ULES01213), US (ULUS10391), JP (ULJM05500) and FUComplete.
constexpr std::uint32_t kGameSysQuestPhase = 0x2Eu;
constexpr std::uint32_t kGameSysPouchBase = 0x6AEA6u;
constexpr std::uint32_t kSlotSize = 4u;
constexpr std::uint8_t kQuestPhaseActive = 2u;
constexpr std::uint8_t kQuestPhasePostClear = 3u;

std::uint32_t pouch_slot_addr(std::uint32_t game_sys, std::size_t slot) noexcept {
    return game_sys + kGameSysPouchBase + static_cast<std::uint32_t>(slot) * kSlotSize;
}

} // namespace

InfiniteConsumables::InfiniteConsumables(InfiniteConsumablesSymbols symbols) noexcept : symbols_(symbols) {}

void InfiniteConsumables::remove_temp_item(psprecomp::GuestMemory &memory, std::uint32_t game_sys,
                                           std::optional<std::size_t> &slot, std::uint16_t item_id) noexcept {
    if (!slot) return;
    const std::uint32_t addr = pouch_slot_addr(game_sys, *slot);
    if (memory.contains(addr, kSlotSize) && memory.load16(addr) == item_id) {
        memory.store16(addr, 0u);
        memory.store16(addr + 2u, 0u);
    }
    slot.reset();
}

void InfiniteConsumables::clear(psprecomp::GuestMemory *memory) noexcept {
    if (memory != nullptr && symbols_.complete() && readable_word_address(*memory, *symbols_.game_sys)) {
        const std::uint32_t game_sys = memory->load32(*symbols_.game_sys);
        if (game_sys != 0u) {
            remove_temp_item(*memory, game_sys, temp_whetstone_slot_, kItemWhetstone);
            remove_temp_item(*memory, game_sys, temp_paintball_slot_, kItemPaintball);
        }
    } else {
        temp_whetstone_slot_.reset();
        temp_paintball_slot_.reset();
    }
    prev_slots_.fill({});
    in_quest_ = false;
}

void InfiniteConsumables::poll(psprecomp::GuestMemory &memory, bool infinite_whetstone, bool infinite_paintballs) {
    if (!symbols_.complete() || !readable_word_address(memory, *symbols_.game_sys)) {
        clear();
        return;
    }
    const std::uint32_t game_sys = memory.load32(*symbols_.game_sys);
    if (game_sys == 0u || !memory.contains(game_sys + kGameSysQuestPhase, 1u) ||
        !memory.contains(game_sys + kGameSysPouchBase, kPouchSlots * kSlotSize)) {
        clear();
        return;
    }

    const std::uint8_t phase = memory.load8(game_sys + kGameSysQuestPhase);
    const bool quest_running = phase == kQuestPhaseActive || phase == kQuestPhasePostClear;
    if (!quest_running || (!infinite_whetstone && !infinite_paintballs)) {
        remove_temp_item(memory, game_sys, temp_whetstone_slot_, kItemWhetstone);
        remove_temp_item(memory, game_sys, temp_paintball_slot_, kItemPaintball);
        prev_slots_.fill({});
        in_quest_ = false;
        return;
    }

    if (!infinite_whetstone) remove_temp_item(memory, game_sys, temp_whetstone_slot_, kItemWhetstone);
    if (!infinite_paintballs) remove_temp_item(memory, game_sys, temp_paintball_slot_, kItemPaintball);

    bool has_whetstone = false;
    bool has_paintball = false;

    for (std::size_t i = 0; i < kPouchSlots; ++i) {
        const std::uint32_t addr = pouch_slot_addr(game_sys, i);
        std::uint16_t id = memory.load16(addr);
        std::uint16_t count = memory.load16(addr + 2u);

        if (in_quest_) {
            const SlotSnapshot prev = prev_slots_[i];
            const bool is_whet = prev.id == kItemWhetstone || prev.id == kItemMiniWhetstone;
            const bool restore_whet = infinite_whetstone && is_whet && prev.count > 0u;
            const bool restore_paint = infinite_paintballs && prev.id == kItemPaintball && prev.count > 0u;
            if ((restore_whet || restore_paint) && ((id == prev.id && count < prev.count) || (id == 0u && count == 0u))) {
                id = prev.id;
                count = prev.count;
                memory.store16(addr, id);
                memory.store16(addr + 2u, count);
            }
        }

        if ((id == kItemWhetstone || id == kItemMiniWhetstone) && count > 0u) has_whetstone = true;
        if (id == kItemPaintball && count > 0u) has_paintball = true;
        prev_slots_[i] = {id, count};
    }

    auto find_last_empty = [&]() -> std::optional<std::size_t> {
        for (std::size_t i = kPouchSlots; i > 0u; --i) {
            if (prev_slots_[i - 1u].id == 0u && prev_slots_[i - 1u].count == 0u) return i - 1u;
        }
        return std::nullopt;
    };

    // World/Rise/Wilds style: supply a temporary Whetstone / Paintball if none were packed.
    if (infinite_whetstone && !has_whetstone) {
        if (const auto slot = find_last_empty()) {
            const std::uint32_t addr = pouch_slot_addr(game_sys, *slot);
            memory.store16(addr, kItemWhetstone);
            memory.store16(addr + 2u, 1u);
            prev_slots_[*slot] = {kItemWhetstone, 1u};
            temp_whetstone_slot_ = *slot;
        }
    }
    if (infinite_paintballs && !has_paintball) {
        if (const auto slot = find_last_empty()) {
            const std::uint32_t addr = pouch_slot_addr(game_sys, *slot);
            memory.store16(addr, kItemPaintball);
            memory.store16(addr + 2u, 1u);
            prev_slots_[*slot] = {kItemPaintball, 1u};
            temp_paintball_slot_ = *slot;
        }
    }

    in_quest_ = true;
}

} // namespace mhf::mhfu
