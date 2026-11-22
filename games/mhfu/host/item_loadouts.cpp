#include "item_loadouts.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace mhf::mhfu {
namespace {

// Verified identical across EU (ULES01213), US (ULUS10391), JP (ULJM05500) and FUComplete.
constexpr std::uint32_t kGameSysQuestPhase = 0x2Eu;
constexpr std::uint32_t kGameSysSaveBase = 0x4A0u;
constexpr std::uint32_t kGameSysEquippedBase = kGameSysSaveBase + 0x20u;    // 6 x 12 bytes (Weapon, Legs, Head, Chest, Arms, Waist)
constexpr std::uint32_t kGameSysEquipBoxBase = kGameSysSaveBase + 0xA8u;    // 1000 x 12 bytes
constexpr std::uint32_t kGameSysEquipIdxBase = kGameSysSaveBase + 16418u;   // u16[6] indices into Equipment Box
constexpr std::uint32_t kGameSysBoxBase = 0x3428u;                          // 1000 x 4 bytes
constexpr std::uint32_t kGameSysPouchBase = 0x6AEA6u;                       // 24 x 4 bytes
constexpr std::uint32_t kSlotSize = 4u;
constexpr std::uint32_t kEquipPieceSize = 12u;
constexpr std::uint16_t kBoxStackLimit = 99u;
constexpr std::uint16_t kEquipIndexNone = 0xFFFFu;
constexpr std::uint8_t kQuestPhaseActive = 2u;
constexpr std::uint8_t kQuestPhasePostClear = 3u;
constexpr std::uint64_t kBoxHeartbeatUs = 80'000u; // disappears promptly (~2-3 frames) when closing the box

// Monster Hunter warm leather-slate & antique-gold palette.
constexpr mhf::overlay::Color kMhTop{0.125f, 0.106f, 0.086f};
constexpr mhf::overlay::Color kMhBot{0.059f, 0.051f, 0.043f};
constexpr mhf::overlay::Color kMhHeader{0.184f, 0.149f, 0.102f};
constexpr mhf::overlay::Color kMhGold{0.847f, 0.706f, 0.392f};
constexpr mhf::overlay::Color kMhGoldBright{0.988f, 0.843f, 0.451f};
constexpr mhf::overlay::Color kMhGoldDim{0.443f, 0.357f, 0.204f};
constexpr mhf::overlay::Color kMhParchment{0.969f, 0.937f, 0.855f};
constexpr mhf::overlay::Color kMhMuted{0.667f, 0.620f, 0.525f};
constexpr mhf::overlay::Color kMhRow{0.118f, 0.106f, 0.090f};
constexpr mhf::overlay::Color kMhSelectTop{0.784f, 0.569f, 0.176f};
constexpr mhf::overlay::Color kMhSelectBot{0.588f, 0.400f, 0.106f};
constexpr mhf::overlay::Color kMhPillTop{0.216f, 0.184f, 0.141f};
constexpr mhf::overlay::Color kMhPillBot{0.118f, 0.098f, 0.075f};
constexpr mhf::overlay::Color kMhOkBg{0.118f, 0.235f, 0.145f};
constexpr mhf::overlay::Color kMhOkText{0.765f, 0.965f, 0.745f};
constexpr mhf::overlay::Color kMhErrBg{0.275f, 0.110f, 0.110f};
constexpr mhf::overlay::Color kMhErrText{1.0f, 0.725f, 0.705f};

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

EquipPiece read_piece(const psprecomp::GuestMemory &memory, std::uint32_t addr) noexcept {
    EquipPiece p{};
    for (std::uint32_t i = 0; i < kEquipPieceSize; ++i) p[i] = memory.load8(addr + i);
    return p;
}

void write_piece(psprecomp::GuestMemory &memory, std::uint32_t addr, const EquipPiece &p) noexcept {
    for (std::uint32_t i = 0; i < kEquipPieceSize; ++i) memory.store8(addr + i, p[i]);
}

bool piece_non_empty(const EquipPiece &p) noexcept {
    return p[0] != 0u;
}

std::uint32_t valid_game_sys(const psprecomp::GuestMemory &memory, ItemLoadoutsSymbols symbols) noexcept {
    if (!symbols.complete() || !readable_word_address(memory, *symbols.game_sys)) return 0u;
    const std::uint32_t gs = memory.load32(*symbols.game_sys);
    if (gs == 0u || !memory.contains(gs + kGameSysQuestPhase, 1u) ||
        !memory.contains(gs + kGameSysEquippedBase, static_cast<std::uint32_t>(kEquipSlots) * kEquipPieceSize) ||
        !memory.contains(gs + kGameSysEquipBoxBase, static_cast<std::uint32_t>(kEquipBoxSlots) * kEquipPieceSize) ||
        !memory.contains(gs + kGameSysEquipIdxBase, static_cast<std::uint32_t>(kEquipSlots) * 2u) ||
        !memory.contains(gs + kGameSysPouchBase, static_cast<std::uint32_t>(kItemPouchSlots) * kSlotSize) ||
        !memory.contains(gs + kGameSysBoxBase, static_cast<std::uint32_t>(kItemBoxSlots) * kSlotSize)) {
        return 0u;
    }
    return gs;
}

std::size_t count_non_empty_items(const ItemLoadoutPreset &p) noexcept {
    std::size_t n = 0;
    for (const auto &s : p.slots) {
        if (s.id != 0u && s.count > 0u) ++n;
    }
    return n;
}

std::size_t count_non_empty_armor(const ArmorLoadoutPreset &p) noexcept {
    std::size_t n = 0;
    for (const auto &piece : p.pieces) {
        if (piece_non_empty(piece)) ++n;
    }
    return n;
}

void draw_button_pill(mhf::overlay::Painter &painter, float x, float y, float w, float h, const std::string &label,
                      bool highlight = false) {
    painter.rrect(x, y, w, h, h * 0.5f, highlight ? kMhGold : kMhPillTop, 0.96f, 0.0f,
                  highlight ? &kMhGoldBright : &kMhPillBot);
    painter.rrect(x, y, w, h, h * 0.5f, highlight ? kMhGoldBright : kMhGold, 0.92f, 0.85f);
    const float fsize = 6.5f;
    const float tw = painter.font_ok() ? painter.measure(label, fsize, true, 0.25f) : w * 0.6f;
    painter.text(x + (w - tw) * 0.5f, y + h * 0.70f, label, fsize,
                 highlight ? mhf::overlay::kInk : kMhParchment, 0.98f, true, 0.25f);
}

} // namespace

ItemLoadouts::ItemLoadouts(ItemLoadoutsSymbols symbols, std::string storage_path) noexcept
    : symbols_(symbols), storage_path_(std::move(storage_path)) {}

std::string ItemLoadouts::armor_storage_path() const {
    if (storage_path_.empty()) return {};
    return storage_path_ + ".armor";
}

void ItemLoadouts::set_storage_path(std::string path) {
    storage_path_ = std::move(path);
    load_from_disk();
}

void ItemLoadouts::load_from_disk() {
    presets_.fill({});
    armor_presets_.fill({});
    if (storage_path_.empty()) return;

    if (std::ifstream in(storage_path_); in) {
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

    if (std::ifstream in(armor_storage_path()); in) {
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            std::size_t idx = 0;
            if (!(iss >> idx) || idx >= kItemLoadoutCount) continue;
            ArmorLoadoutPreset a;
            a.saved = true;
            for (std::size_t s = 0; s < kEquipSlots; ++s) {
                std::uint16_t box_idx = kEquipIndexNone;
                std::string hex;
                if (!(iss >> box_idx >> hex) || hex.size() != kEquipPieceSize * 2u) break;
                a.box_indices[s] = box_idx;
                for (std::size_t b = 0; b < kEquipPieceSize; ++b) {
                    a.pieces[s][b] = static_cast<std::uint8_t>(std::stoul(hex.substr(b * 2u, 2u), nullptr, 16));
                }
            }
            armor_presets_[idx] = a;
        }
    }
}

void ItemLoadouts::save_to_disk() const {
    if (storage_path_.empty()) return;
    if (std::ofstream out(storage_path_, std::ios::trunc); out) {
        out << "# MHForever item pouch loadouts (slot_index id:count ...)\n";
        for (std::size_t i = 0; i < kItemLoadoutCount; ++i) {
            if (!presets_[i].saved) continue;
            out << i;
            for (const ItemSlot &s : presets_[i].slots) out << ' ' << s.id << ':' << s.count;
            out << '\n';
        }
    }
    if (std::ofstream out(armor_storage_path(), std::ios::trunc); out) {
        out << "# MHForever armor loadouts (slot_index box_idx hex12 ...)\n";
        for (std::size_t i = 0; i < kItemLoadoutCount; ++i) {
            if (!armor_presets_[i].saved) continue;
            out << i;
            for (std::size_t s = 0; s < kEquipSlots; ++s) {
                out << ' ' << armor_presets_[i].box_indices[s] << ' ';
                for (std::uint8_t byte : armor_presets_[i].pieces[s]) {
                    out << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(byte);
                }
                out << std::dec;
            }
            out << '\n';
        }
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

bool ItemLoadouts::has_armor_preset(std::size_t index) const noexcept {
    return index < kItemLoadoutCount && armor_presets_[index].saved;
}

const ItemLoadoutPreset &ItemLoadouts::preset(std::size_t index) const noexcept {
    return presets_[std::min(index, kItemLoadoutCount - 1u)];
}

const ArmorLoadoutPreset &ItemLoadouts::armor_preset(std::size_t index) const noexcept {
    return armor_presets_[std::min(index, kItemLoadoutCount - 1u)];
}

LoadoutResult ItemLoadouts::save_preset(const psprecomp::GuestMemory &memory, std::size_t index) {
    if (index >= kItemLoadoutCount) return {false, "Invalid loadout slot."};
    if (!can_use_loadouts(memory)) return {false, "Only available in Village or Hall."};
    const std::uint32_t gs = valid_game_sys(memory, symbols_);
    ItemLoadoutPreset p;
    p.saved = true;
    for (std::size_t s = 0; s < kItemPouchSlots; ++s) {
        p.slots[s] = read_slot(memory, gs + kGameSysPouchBase, s);
    }
    presets_[index] = p;
    save_to_disk();
    return {true, "Saved Item Set " + std::to_string(index + 1)};
}

LoadoutResult ItemLoadouts::apply_preset(psprecomp::GuestMemory &memory, std::size_t index) const {
    if (index >= kItemLoadoutCount || !presets_[index].saved) return {false, "Item Set is empty."};
    if (!can_use_loadouts(memory)) return {false, "Only available in Village or Hall."};
    const std::uint32_t gs = valid_game_sys(memory, symbols_);
    const std::uint32_t pouch_base = gs + kGameSysPouchBase;
    const std::uint32_t box_base = gs + kGameSysBoxBase;

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

    std::size_t missing = 0;
    const auto &target = presets_[index].slots;
    for (std::size_t p = 0; p < kItemPouchSlots; ++p) {
        const ItemSlot want = target[p];
        if (want.id == 0u || want.count == 0u) continue;
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
        return {true, "Loaded Item Set " + std::to_string(index + 1) + " (" + std::to_string(missing) + " short)"};
    }
    return {true, "Loaded Item Set " + std::to_string(index + 1)};
}

LoadoutResult ItemLoadouts::save_armor_preset(const psprecomp::GuestMemory &memory, std::size_t index) {
    if (index >= kItemLoadoutCount) return {false, "Invalid armor set slot."};
    if (!can_use_loadouts(memory)) return {false, "Only available in Village or Hall."};
    const std::uint32_t gs = valid_game_sys(memory, symbols_);
    ArmorLoadoutPreset a;
    a.saved = true;
    for (std::size_t s = 0; s < kEquipSlots; ++s) {
        a.pieces[s] = read_piece(memory, gs + kGameSysEquippedBase + static_cast<std::uint32_t>(s) * kEquipPieceSize);
        a.box_indices[s] = memory.load16(gs + kGameSysEquipIdxBase + static_cast<std::uint32_t>(s) * 2u);
    }
    armor_presets_[index] = a;
    save_to_disk();
    return {true, "Saved Armor Set " + std::to_string(index + 1)};
}

LoadoutResult ItemLoadouts::apply_armor_preset(psprecomp::GuestMemory &memory, std::size_t index) const {
    if (index >= kItemLoadoutCount || !armor_presets_[index].saved) return {false, "Armor Set is empty."};
    if (!can_use_loadouts(memory)) return {false, "Only available in Village or Hall."};
    const std::uint32_t gs = valid_game_sys(memory, symbols_);
    const auto &target = armor_presets_[index];

    std::size_t missing = 0;
    for (std::size_t s = 0; s < kEquipSlots; ++s) {
        const std::uint32_t eq_addr = gs + kGameSysEquippedBase + static_cast<std::uint32_t>(s) * kEquipPieceSize;
        const std::uint32_t idx_addr = gs + kGameSysEquipIdxBase + static_cast<std::uint32_t>(s) * 2u;

        if (!piece_non_empty(target.pieces[s])) {
            // Unequip armor slot (never unequip weapon slot 0).
            if (s != 0u) {
                write_piece(memory, eq_addr, EquipPiece{});
                memory.store16(idx_addr, kEquipIndexNone);
            }
            continue;
        }

        // First check the remembered Equipment Box index; if the box was sorted, locate the matching piece.
        std::optional<std::uint16_t> found_idx;
        if (target.box_indices[s] < kEquipBoxSlots) {
            const EquipPiece at_idx = read_piece(memory, gs + kGameSysEquipBoxBase +
                                                             static_cast<std::uint32_t>(target.box_indices[s]) * kEquipPieceSize);
            if (at_idx == target.pieces[s]) found_idx = target.box_indices[s];
        }
        if (!found_idx) {
            for (std::uint16_t b = 0; b < kEquipBoxSlots; ++b) {
                const EquipPiece cand = read_piece(memory, gs + kGameSysEquipBoxBase + static_cast<std::uint32_t>(b) * kEquipPieceSize);
                if (cand == target.pieces[s]) {
                    found_idx = b;
                    break;
                }
            }
        }
        if (!found_idx) {
            ++missing;
            continue;
        }

        const EquipPiece box_piece = read_piece(memory, gs + kGameSysEquipBoxBase + static_cast<std::uint32_t>(*found_idx) * kEquipPieceSize);
        write_piece(memory, eq_addr, box_piece);
        memory.store16(idx_addr, *found_idx);
    }

    if (missing > 0u) {
        return {true, "Equipped Armor Set " + std::to_string(index + 1) + " (" + std::to_string(missing) + " missing)"};
    }
    return {true, "Equipped Armor Set " + std::to_string(index + 1)};
}

void ItemLoadouts::mark_box_open(std::uint64_t now_us) noexcept {
    box_seen_us_ = now_us;
}

bool ItemLoadouts::box_open(std::uint64_t now_us) const noexcept {
    return box_seen_us_ != 0u && now_us >= box_seen_us_ && (now_us - box_seen_us_) <= kBoxHeartbeatUs;
}

void ItemLoadouts::poll_box_ui(psprecomp::GuestMemory &memory, std::uint32_t buttons, std::uint64_t now_us, bool enabled) {
    const std::uint32_t pressed = buttons & ~prev_buttons_;
    prev_buttons_ = buttons;

    if (!enabled || !box_open(now_us) || !can_use_loadouts(memory)) {
        panel_open_ = false;
        return;
    }

    using namespace mhf::psp::buttons;
    if (!panel_open_) {
        if ((pressed & kSelect) != 0u) panel_open_ = true;
        return;
    }

    if ((pressed & (kLTrigger | kRTrigger | kLeft | kRight)) != 0u) {
        tab_ = tab_ == Tab::Items ? Tab::Armor : Tab::Items;
    } else if ((pressed & kUp) != 0u) {
        selected_ = (selected_ + kItemLoadoutCount - 1u) % kItemLoadoutCount;
    } else if ((pressed & kDown) != 0u) {
        selected_ = (selected_ + 1u) % kItemLoadoutCount;
    } else if ((pressed & kCircle) != 0u) {
        const auto res = tab_ == Tab::Items ? apply_preset(memory, selected_) : apply_armor_preset(memory, selected_);
        toast_ = res.message;
        toast_ok_ = res.ok;
        toast_until_us_ = now_us + 2'200'000u;
    } else if ((pressed & kTriangle) != 0u) {
        const auto res = tab_ == Tab::Items ? save_preset(memory, selected_) : save_armor_preset(memory, selected_);
        toast_ = res.message;
        toast_ok_ = res.ok;
        toast_until_us_ = now_us + 2'200'000u;
    } else if ((pressed & (kCross | kSelect | kStart)) != 0u) {
        panel_open_ = false;
    }
}

bool ItemLoadouts::render_box_ui(mhf::overlay::Painter &painter, std::uint64_t now_us) const {
    if (!box_open(now_us)) return false;

    if (!panel_open_) {
        // Sleek MH-styled bottom-right prompt card inside House & Guild Hall Item Box.
        const std::string lbl = "Item & Armor Sets";
        const float fsize = 8.5f;
        const float tw = painter.font_ok() ? painter.measure(lbl, fsize, true) : 78.0f;
        const float pill_w = 36.0f;
        const float cw = 8.0f * 2.0f + pill_w + 6.0f + tw;
        const float ch = 22.0f;
        const float cx = 480.0f - cw - 8.0f;
        const float cy = 272.0f - ch - 8.0f;

        painter.rrect(cx + 1.5f, cy + 1.5f, cw, ch, 4.5f, mhf::overlay::kBlack, 0.45f);
        painter.rrect(cx, cy, cw, ch, 4.0f, kMhTop, 0.90f, 0.0f, &kMhBot);
        painter.rrect(cx, cy, cw, ch, 4.0f, kMhGold, 0.92f, 1.0f);
        draw_button_pill(painter, cx + 7.0f, cy + 4.5f, pill_w, 13.0f, "SELECT");
        painter.text(cx + 7.0f + pill_w + 6.0f, cy + 14.0f, lbl, fsize, kMhParchment, 0.98f, true);
        return true;
    }

    // Centered Monster Hunter-styled Hunter Loadouts window (Item Sets + Armor Sets).
    const float px = 104.0f, py = 38.0f, pw = 272.0f, ph = 194.0f;
    painter.rrect(px + 2.5f, py + 3.0f, pw, ph, 6.5f, mhf::overlay::kBlack, 0.55f);
    painter.rrect(px, py, pw, ph, 6.0f, kMhTop, 0.94f, 0.0f, &kMhBot);
    painter.rrect(px, py, pw, ph, 6.0f, kMhGold, 0.95f, 1.2f);
    painter.rrect(px + 2.5f, py + 2.5f, pw - 5.0f, ph - 5.0f, 4.0f, kMhGoldDim, 0.60f, 0.75f);

    // Ornamental Header Bar
    painter.rrect(px + 3.0f, py + 3.0f, pw - 6.0f, 22.0f, 3.5f, kMhHeader, 0.95f);
    painter.rect(px + 4.0f, py + 25.0f, pw - 8.0f, 1.0f, kMhGoldDim, 0.85f);
    painter.text(px + 12.0f, py + 17.5f, "Hunter Loadouts", 11.0f, kMhGoldBright, 0.98f, true);

    // Tab switcher pills (L / R): Item Sets | Armor Sets
    const bool on_items = tab_ == Tab::Items;
    const float tab_y = py + 31.0f;
    const float tab_w = (pw - 24.0f) * 0.5f;
    auto draw_tab = [&](float x, const std::string &title, bool active) {
        painter.rrect(x, tab_y, tab_w, 17.0f, 3.5f, active ? kMhSelectTop : kMhRow, active ? 0.95f : 0.78f, 0.0f,
                      active ? &kMhSelectBot : nullptr);
        painter.rrect(x, tab_y, tab_w, 17.0f, 3.5f, active ? kMhGoldBright : kMhGoldDim, active ? 0.95f : 0.55f, 0.85f);
        const float tw = painter.font_ok() ? painter.measure(title, 8.5f, true) : 48.0f;
        painter.text(x + (tab_w - tw) * 0.5f, tab_y + 11.5f, title, 8.5f,
                     active ? mhf::overlay::kInk : kMhMuted, 0.98f, true);
    };
    draw_tab(px + 10.0f, "Item Sets", on_items);
    draw_tab(px + 14.0f + tab_w, "Armor Sets", !on_items);

    // 5 Loadout Rows
    for (std::size_t i = 0; i < kItemLoadoutCount; ++i) {
        const float ry = py + 54.0f + static_cast<float>(i) * 20.5f;
        const bool sel = i == selected_;
        painter.rrect(px + 10.0f, ry, pw - 20.0f, 17.5f, 3.5f, sel ? kMhSelectTop : kMhRow, sel ? 0.92f : 0.80f, 0.0f,
                      sel ? &kMhSelectBot : nullptr);
        if (sel) {
            painter.rrect(px + 10.0f, ry, pw - 20.0f, 17.5f, 3.5f, kMhGoldBright, 0.98f, 1.0f);
        }

        const std::string name = (on_items ? "Item Set " : "Armor Set ") + std::to_string(i + 1);
        painter.text(px + 18.0f, ry + 11.8f, name, 8.8f, sel ? mhf::overlay::kInk : kMhParchment, 0.98f, true);

        const bool saved = on_items ? presets_[i].saved : armor_presets_[i].saved;
        const std::string detail = !saved ? "Empty"
                                          : (on_items ? std::to_string(count_non_empty_items(presets_[i])) + " items"
                                                      : std::to_string(count_non_empty_armor(armor_presets_[i])) + "/6 pieces");
        const float dw = painter.font_ok() ? painter.measure(detail, 8.0f, false) : 36.0f;
        painter.text(px + pw - 18.0f - dw, ry + 11.6f, detail, 8.0f,
                     sel ? mhf::overlay::kInk : (saved ? kMhGold : kMhMuted), 0.95f, sel);
    }

    // Status Toast
    if (!toast_.empty() && now_us < toast_until_us_) {
        const float ty = py + ph - 36.0f;
        painter.rrect(px + 10.0f, ty, pw - 20.0f, 15.0f, 3.0f, toast_ok_ ? kMhOkBg : kMhErrBg, 0.92f);
        painter.text(px + 16.0f, ty + 10.5f, toast_, 8.0f, toast_ok_ ? kMhOkText : kMhErrText, 0.98f, true);
    }

    // Footer Controls Bar with PSP-style button pills
    const float fy = py + ph - 17.5f;
    painter.rect(px + 8.0f, fy - 2.5f, pw - 16.0f, 0.75f, kMhGoldDim, 0.65f);
    float fx = px + 10.0f;
    auto add_hint = [&](const std::string &btn, float bw, const std::string &act) {
        draw_button_pill(painter, fx, fy, bw, 11.5f, btn);
        fx += bw + 3.5f;
        fx += painter.text(fx, fy + 8.5f, act, 7.5f, kMhParchment, 0.92f, false) + 7.0f;
    };
    add_hint("O", 13.0f, on_items ? "Load" : "Equip");
    add_hint("△", 13.0f, "Save");
    add_hint("L/R", 20.0f, "Tab");
    add_hint("X", 13.0f, "Close");
    return true;
}

} // namespace mhf::mhfu
