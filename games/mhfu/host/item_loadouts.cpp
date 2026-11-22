#include "item_loadouts.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>

namespace mhf::mhfu {
namespace {

// Verified identical across EU (ULES01213), US (ULUS10391), JP (ULJM05500) and FUComplete.
constexpr std::uint32_t kGameSysQuestPhase = 0x2Eu;
constexpr std::uint32_t kGameSysPouchBase = 0x6AEA6u;
constexpr std::uint32_t kGameSysBoxBase = 0x3428u;
constexpr std::uint32_t kSlotSize = 4u;
constexpr std::uint16_t kBoxStackLimit = 99u;
constexpr std::uint8_t kQuestPhaseActive = 2u;
constexpr std::uint8_t kQuestPhasePostClear = 3u;

bool readable_word_address(const psprecomp::GuestMemory &memory, std::uint32_t address) noexcept {
    return (address & 3u) == 0u && memory.contains(address, 4u);
}

ItemSlot read_slot(const psprecomp::GuestMemory &memory, std::uint32_t base, std::size_t idx) noexcept {
    const std::uint32_t addr = base + static_cast<std::uint32_t>(idx) * kSlotSize;
    return {memory.load16(addr), memory.load16(addr + 2u)};
}

void write_slot(psprecomp::GuestMemory &memory, std::uint32_t base, std::size_t idx, ItemSlot s) noexcept {
    const std::uint32_t addr = base + static_cast<std::uint32_t>(idx) * kSlotSize;
    memory.store16(addr, s.count > 0u ? s.id : 0u);
    memory.store16(addr + 2u, s.id > 0u ? s.count : 0u);
}

std::uint32_t valid_game_sys(const psprecomp::GuestMemory &memory, ItemLoadoutsSymbols symbols) noexcept {
    if (!symbols.complete() || !readable_word_address(memory, *symbols.game_sys)) return 0u;
    const std::uint32_t gs = memory.load32(*symbols.game_sys);
    if (gs == 0u || !memory.contains(gs + kGameSysQuestPhase, 1u) ||
        !memory.contains(gs + kGameSysPouchBase, static_cast<std::uint32_t>(kItemPouchSlots) * kSlotSize) ||
        !memory.contains(gs + kGameSysBoxBase, static_cast<std::uint32_t>(kItemBoxSlots) * kSlotSize)) {
        return 0u;
    }
    return gs;
}

} // namespace

ItemLoadouts::ItemLoadouts(ItemLoadoutsSymbols symbols, std::string storage_path) noexcept
    : symbols_(symbols), storage_path_(std::move(storage_path)) {}

void ItemLoadouts::set_storage_path(std::string path) {
    storage_path_ = std::move(path);
    load_from_disk();
}

void ItemLoadouts::load_from_disk() {
    presets_.fill({});
    if (storage_path_.empty()) return;
    std::ifstream in(storage_path_);
    if (!in) return;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::size_t idx = 0;
        if (!(iss >> idx) || idx >= kItemLoadoutCount) continue;
        ItemLoadoutPreset p;
        p.saved = true;
        for (std::size_t s = 0; s < kItemPouchSlots; ++s) {
            std::uint16_t id = 0, count = 0;
            char colon = 0;
            if (iss >> id >> colon >> count) p.slots[s] = {id, count};
        }
        presets_[idx] = p;
    }
}

void ItemLoadouts::save_to_disk() const {
    if (storage_path_.empty()) return;
    std::ofstream out(storage_path_, std::ios::trunc);
    if (!out) return;
    out << "# MHForever item pouch loadouts (slot_index id:count ...)\n";
    for (std::size_t i = 0; i < kItemLoadoutCount; ++i) {
        if (!presets_[i].saved) continue;
        out << i;
        for (const ItemSlot &s : presets_[i].slots) {
            out << ' ' << s.id << ':' << s.count;
        }
        out << '\n';
    }
}

bool ItemLoadouts::can_use_loadouts(const psprecomp::GuestMemory &memory) const noexcept {
    const std::uint32_t gs = valid_game_sys(memory, symbols_);
    if (gs == 0u) return false;
    const std::uint8_t phase = memory.load8(gs + kGameSysQuestPhase);
    return phase != kQuestPhaseActive && phase != kQuestPhasePostClear;
}

bool ItemLoadouts::has_preset(std::size_t index) const noexcept {
    return index < kItemLoadoutCount && presets_[index].saved;
}

const ItemLoadoutPreset &ItemLoadouts::preset(std::size_t index) const noexcept {
    return presets_[std::min(index, kItemLoadoutCount - 1u)];
}

LoadoutResult ItemLoadouts::save_preset(const psprecomp::GuestMemory &memory, std::size_t index) {
    if (index >= kItemLoadoutCount) return {false, "Invalid loadout slot."};
    if (!can_use_loadouts(memory)) return {false, "Loadouts are only available in Village or Hall."};
    const std::uint32_t gs = valid_game_sys(memory, symbols_);
    ItemLoadoutPreset p;
    p.saved = true;
    for (std::size_t s = 0; s < kItemPouchSlots; ++s) {
        p.slots[s] = read_slot(memory, gs + kGameSysPouchBase, s);
    }
    presets_[index] = p;
    save_to_disk();
    return {true, "Saved Loadout " + std::to_string(index + 1)};
}

LoadoutResult ItemLoadouts::apply_preset(psprecomp::GuestMemory &memory, std::size_t index) const {
    if (index >= kItemLoadoutCount || !presets_[index].saved) return {false, "Loadout slot is empty."};
    if (!can_use_loadouts(memory)) return {false, "Loadouts are only available in Village or Hall."};
    const std::uint32_t gs = valid_game_sys(memory, symbols_);
    const std::uint32_t pouch_base = gs + kGameSysPouchBase;
    const std::uint32_t box_base = gs + kGameSysBoxBase;

    // 1. Deposit current pouch items into the Item Box (stacking onto existing stacks first, then empty slots).
    for (std::size_t p = 0; p < kItemPouchSlots; ++p) {
        ItemSlot cur = read_slot(memory, pouch_base, p);
        if (cur.id == 0u || cur.count == 0u) continue;
        std::uint16_t rem = cur.count;

        for (std::size_t b = 0; b < kItemBoxSlots && rem > 0u; ++b) {
            ItemSlot box = read_slot(memory, box_base, b);
            if (box.id == cur.id && box.count < kBoxStackLimit) {
                const std::uint16_t add = static_cast<std::uint16_t>(std::min<std::uint32_t>(rem, kBoxStackLimit - box.count));
                box.count = static_cast<std::uint16_t>(box.count + add);
                rem = static_cast<std::uint16_t>(rem - add);
                write_slot(memory, box_base, b, box);
            }
        }
        for (std::size_t b = 0; b < kItemBoxSlots && rem > 0u; ++b) {
            ItemSlot box = read_slot(memory, box_base, b);
            if (box.id == 0u || box.count == 0u) {
                const std::uint16_t add = static_cast<std::uint16_t>(std::min<std::uint32_t>(rem, kBoxStackLimit));
                write_slot(memory, box_base, b, {cur.id, add});
                rem = static_cast<std::uint16_t>(rem - add);
            }
        }
        write_slot(memory, pouch_base, p, rem > 0u ? ItemSlot{cur.id, rem} : ItemSlot{0u, 0u});
    }

    // 2. Withdraw preset items from the Item Box into the Pouch.
    std::size_t missing = 0;
    const auto &target = presets_[index].slots;
    for (std::size_t p = 0; p < kItemPouchSlots; ++p) {
        const ItemSlot want = target[p];
        if (want.id == 0u || want.count == 0u) continue;
        // Pick target slot if empty, otherwise first empty pouch slot.
        std::optional<std::size_t> dest;
        if (read_slot(memory, pouch_base, p).id == 0u) {
            dest = p;
        } else {
            for (std::size_t alt = 0; alt < kItemPouchSlots; ++alt) {
                if (read_slot(memory, pouch_base, alt).id == 0u) {
                    dest = alt;
                    break;
                }
            }
        }
        if (!dest) {
            ++missing;
            continue;
        }
        std::uint16_t got = 0u;
        for (std::size_t b = 0; b < kItemBoxSlots && got < want.count; ++b) {
            ItemSlot box = read_slot(memory, box_base, b);
            if (box.id == want.id && box.count > 0u) {
                const std::uint16_t take = static_cast<std::uint16_t>(std::min<std::uint32_t>(want.count - got, box.count));
                box.count = static_cast<std::uint16_t>(box.count - take);
                got = static_cast<std::uint16_t>(got + take);
                write_slot(memory, box_base, b, box.count > 0u ? box : ItemSlot{0u, 0u});
            }
        }
        if (got > 0u) write_slot(memory, pouch_base, *dest, {want.id, got});
        if (got < want.count) ++missing;
    }

    if (missing > 0u) {
        return {true, "Loaded Loadout " + std::to_string(index + 1) + " (" + std::to_string(missing) + " short)"};
    }
    return {true, "Loaded Loadout " + std::to_string(index + 1)};
}

} // namespace mhf::mhfu
