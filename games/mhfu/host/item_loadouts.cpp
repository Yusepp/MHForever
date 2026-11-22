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
    // MHFU 12-byte equipment record: byte 0 = level/flags, byte 1 = equip type (1..7), bytes 2..3 = u16 id.
    return p[1] != 0u || p[2] != 0u || p[3] != 0u;
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
    const float fsize = 6.2f;
    const float tw = painter.font_ok() ? painter.measure(label, fsize, true, 0.25f) : w * 0.6f;
    painter.text(x + (w - tw) * 0.5f, y + h * 0.71f, label, fsize,
                 highlight ? mhf::overlay::kInk : kMhParchment, 0.98f, true, 0.25f);
}

// Master localized text archive layout in ResourceManager (same format across EU, US, JP, FUComplete):
// header: u32 num_groups (~23), u32 table_offset (8), u32 group_offsets[num_groups].
// Each group starts with an array of (possibly unaligned) u32 offsets from the group base to UTF-8 strings.
constexpr std::uint32_t kResourceTable = 0x83C030u;
constexpr std::uint32_t kResourceEntry = 12u;
constexpr std::uint32_t kResources = 512u;
constexpr std::uint32_t kGroupItems = 1u;
constexpr std::uint32_t kGroupBlademaster = 3u;
constexpr std::uint32_t kGroupGunner = 5u;
constexpr std::uint32_t kGroupHead = 7u;
constexpr std::uint32_t kGroupChest = 9u;
constexpr std::uint32_t kGroupArms = 11u;
constexpr std::uint32_t kGroupWaist = 13u;
constexpr std::uint32_t kGroupLegs = 15u;
constexpr std::uint8_t kEquipTypeBlademaster = 5u;

std::uint32_t load_u32_unaligned(const psprecomp::GuestMemory &memory, std::uint32_t addr) noexcept {
    return static_cast<std::uint32_t>(memory.load8(addr)) |
           (static_cast<std::uint32_t>(memory.load8(addr + 1u)) << 8u) |
           (static_cast<std::uint32_t>(memory.load8(addr + 2u)) << 16u) |
           (static_cast<std::uint32_t>(memory.load8(addr + 3u)) << 24u);
}

bool printable_utf8(std::string_view text) noexcept {
    if (text.empty()) return false;
    for (std::size_t i = 0; i < text.size();) {
        const auto c = static_cast<unsigned char>(text[i]);
        const std::size_t len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 0;
        if (len == 0 || i + len > text.size() || (len == 1 && c < 0x20)) return false;
        for (std::size_t k = 1; k < len; ++k) {
            if ((static_cast<unsigned char>(text[i + k]) & 0xC0) != 0x80) return false;
        }
        i += len;
    }
    return true;
}

std::optional<std::string> read_text_entry(const psprecomp::GuestMemory &memory, std::uint32_t res,
                                           std::uint32_t group_idx, std::uint32_t entry_id) {
    if (res == 0u || !memory.contains(res, 16u)) return std::nullopt;
    const std::uint32_t num_groups = load_u32_unaligned(memory, res);
    if (group_idx >= num_groups || !memory.contains(res + (group_idx + 2u) * 4u, 4u)) return std::nullopt;
    const std::uint32_t group_off = load_u32_unaligned(memory, res + (group_idx + 2u) * 4u);
    if (group_off == 0u) return std::nullopt;
    const std::uint32_t group = res + group_off;
    if (!memory.contains(group + entry_id * 4u, 4u)) return std::nullopt;
    const std::uint32_t str_off = load_u32_unaligned(memory, group + entry_id * 4u);
    if (str_off == 0xFFFFFFFFu || str_off == 0u || str_off > 0x200000u) return std::nullopt;
    const std::uint32_t str_addr = group + str_off;
    if (!memory.contains(str_addr, 2u)) return std::nullopt;

    std::string out;
    out.reserve(32);
    for (std::uint32_t i = 0; i < 64u && memory.contains(str_addr + i, 1u); ++i) {
        const char ch = static_cast<char>(memory.load8(str_addr + i));
        if (ch == '\0' || ch == '\n' || ch == '\r') break;
        out.push_back(ch);
    }
    return printable_utf8(out) ? std::optional(std::move(out)) : std::nullopt;
}

// Finds the resident master text archive (Items, Weapons, Armor) in ResourceManager.
std::uint32_t find_master_text_resource(const psprecomp::GuestMemory &memory, ItemLoadoutsSymbols symbols) noexcept {
    if (!symbols.resource_manager || !readable_word_address(memory, *symbols.resource_manager)) return 0u;
    const std::uint32_t mgr = memory.load32(*symbols.resource_manager);
    if (mgr == 0u || !memory.contains(mgr + kResourceTable, kResources * kResourceEntry)) return 0u;
    for (std::uint32_t i = 0; i < kResources; ++i) {
        const std::uint32_t id = memory.load32(mgr + kResourceTable + i * kResourceEntry);
        if (id == 0xFFFFFFFFu) continue;
        const std::uint32_t data = memory.load32(mgr + kResourceTable + i * kResourceEntry + 4u);
        if (data == 0u || !memory.contains(data, 72u)) continue;
        const std::uint32_t num_groups = load_u32_unaligned(memory, data);
        const std::uint32_t table_off = load_u32_unaligned(memory, data + 4u);
        if (num_groups >= 16u && num_groups <= 40u && table_off == 8u) {
            if (read_text_entry(memory, data, kGroupItems, 1u).has_value() &&
                read_text_entry(memory, data, kGroupBlademaster, 1u).has_value()) {
                return data;
            }
        }
    }
    return 0u;
}

std::string fit_text(mhf::overlay::Painter &painter, std::string text, float fsize, float max_w) {
    if (!painter.font_ok() || max_w <= 8.0f || painter.measure(text, fsize, false) <= max_w) return text;
    while (text.size() > 2u) {
        // Pop one UTF-8 codepoint at a time so we never split a multibyte character.
        do {
            text.pop_back();
        } while (!text.empty() && (static_cast<unsigned char>(text.back()) & 0xC0u) == 0x80u);
        if (painter.measure(text + "..", fsize, false) <= max_w) return text + "..";
    }
    return text;
}

mhf::overlay::Color item_accent_color(std::uint16_t id) noexcept {
    constexpr mhf::overlay::Color kHeal{0.34f, 0.78f, 0.45f};
    constexpr mhf::overlay::Color kTool{0.44f, 0.72f, 0.92f};
    constexpr mhf::overlay::Color kBomb{0.92f, 0.55f, 0.24f};
    constexpr mhf::overlay::Color kBook{0.89f, 0.74f, 0.36f};
    constexpr mhf::overlay::Color kMisc{0.76f, 0.70f, 0.58f};

    if (id >= 1u && id <= 5u) return kBook;
    if (id >= 7u && id <= 25u) return kHeal;
    if (id >= 60u && id <= 80u) return kBomb;
    if ((id >= 135u && id <= 160u) || id == 867u) return kTool;
    return kMisc;
}

std::optional<std::string> read_text_entry_from_bytes(const std::vector<std::uint8_t> &buf,
                                                      std::uint32_t group_idx, std::uint32_t entry_id) {
    if (buf.size() < 16u) return std::nullopt;
    auto u32 = [&](std::size_t off) -> std::uint32_t {
        return static_cast<std::uint32_t>(buf[off]) | (static_cast<std::uint32_t>(buf[off + 1u]) << 8u) |
               (static_cast<std::uint32_t>(buf[off + 2u]) << 16u) | (static_cast<std::uint32_t>(buf[off + 3u]) << 24u);
    };
    const std::uint32_t num_groups = u32(0u);
    if (group_idx >= num_groups || (group_idx + 3u) * 4u > buf.size()) return std::nullopt;
    const std::uint32_t group_off = u32((group_idx + 2u) * 4u);
    if (group_off == 0u || static_cast<std::size_t>(group_off) + static_cast<std::size_t>(entry_id) * 4u + 4u > buf.size()) {
        return std::nullopt;
    }
    const std::uint32_t str_off = u32(static_cast<std::size_t>(group_off) + static_cast<std::size_t>(entry_id) * 4u);
    if (str_off == 0xFFFFFFFFu || str_off == 0u) return std::nullopt;
    const std::size_t str_pos = static_cast<std::size_t>(group_off) + static_cast<std::size_t>(str_off);
    if (str_pos >= buf.size()) return std::nullopt;

    std::string out;
    out.reserve(32);
    for (std::size_t i = 0; i < 64u && str_pos + i < buf.size(); ++i) {
        const char ch = static_cast<char>(buf[str_pos + i]);
        if (ch == '\0' || ch == '\n' || ch == '\r') break;
        out.push_back(ch);
    }
    return printable_utf8(out) ? std::optional(std::move(out)) : std::nullopt;
}

std::string resolve_item_name(const psprecomp::GuestMemory &memory, std::uint32_t text_res,
                              const std::vector<std::uint8_t> &fallback, std::uint16_t id) {
    if (auto live = read_text_entry(memory, text_res, kGroupItems, id)) return *live;
    if (auto disk = read_text_entry_from_bytes(fallback, kGroupItems, id)) return *disk;
    return "Item #" + std::to_string(id);
}

std::string resolve_equip_name(const psprecomp::GuestMemory &memory, std::uint32_t text_res,
                               const std::vector<std::uint8_t> &fallback, std::size_t raw_slot,
                               const EquipPiece &piece) {
    const std::uint16_t piece_id = static_cast<std::uint16_t>(piece[2]) | (static_cast<std::uint16_t>(piece[3]) << 8u);
    std::uint32_t group = kGroupBlademaster;
    switch (raw_slot) {
        case 0u: group = (piece[1] == kEquipTypeBlademaster) ? kGroupBlademaster : kGroupGunner; break;
        case 1u: group = kGroupLegs; break;
        case 2u: group = kGroupHead; break;
        case 3u: group = kGroupChest; break;
        case 4u: group = kGroupArms; break;
        case 5u: group = kGroupWaist; break;
        default: break;
    }
    if (auto live = read_text_entry(memory, text_res, group, piece_id)) return *live;
    if (auto disk = read_text_entry_from_bytes(fallback, group, piece_id)) return *disk;
    return (raw_slot == 0u ? "Weapon #" : "Armor #") + std::to_string(piece_id);
}

std::size_t count_decorations(const EquipPiece &p) noexcept {
    std::size_t n = 0;
    for (std::size_t off = 6u; off + 1u < kEquipPieceSize; off += 2u) {
        const std::uint16_t deco = static_cast<std::uint16_t>(p[off]) | (static_cast<std::uint16_t>(p[off + 1u]) << 8u);
        if (deco != 0u) ++n;
    }
    return n;
}

bool piece_in_equip_box(const psprecomp::GuestMemory &memory, std::uint32_t gs, const EquipPiece &piece,
                        std::uint16_t hint_idx) noexcept {
    if (!piece_non_empty(piece) || gs == 0u) return false;
    if (hint_idx < kEquipBoxSlots) {
        if (read_piece(memory, gs + kGameSysEquipBoxBase + static_cast<std::uint32_t>(hint_idx) * kEquipPieceSize) == piece) {
            return true;
        }
    }
    for (std::uint16_t b = 0; b < kEquipBoxSlots; ++b) {
        if (read_piece(memory, gs + kGameSysEquipBoxBase + static_cast<std::uint32_t>(b) * kEquipPieceSize) == piece) {
            return true;
        }
    }
    return false;
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

void ItemLoadouts::set_text_archive(std::vector<std::uint8_t> archive) {
    text_archive_ = std::move(archive);
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

    using namespace mhf::psp::buttons;
    if (release_guard_ && (buttons & (kCross | kSelect | kStart | kCircle | kTriangle)) == 0u) {
        release_guard_ = false;
    }

    if (!enabled || !box_open(now_us) || !can_use_loadouts(memory)) {
        panel_open_ = false;
        release_guard_ = false;
        return;
    }

    if (!panel_open_) {
        if ((pressed & kSelect) != 0u) {
            panel_open_ = true;
            release_guard_ = false;
        }
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
        release_guard_ = true;
    }
}

bool ItemLoadouts::render_box_ui(mhf::overlay::Painter &painter, const psprecomp::GuestMemory &memory,
                                 std::uint64_t now_us) const {
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

    // Modern MH (World/Rise/Wilds) two-column Hunter Loadouts window rendered at native HD.
    const float px = 24.0f, py = 24.0f, pw = 432.0f, ph = 224.0f;
    painter.rrect(px + 2.5f, py + 3.0f, pw, ph, 6.5f, mhf::overlay::kBlack, 0.58f);
    painter.rrect(px, py, pw, ph, 6.0f, kMhTop, 0.95f, 0.0f, &kMhBot);
    painter.rrect(px, py, pw, ph, 6.0f, kMhGold, 0.95f, 1.2f);
    painter.rrect(px + 2.5f, py + 2.5f, pw - 5.0f, ph - 5.0f, 4.0f, kMhGoldDim, 0.58f, 0.75f);

    // Ornamental Header Bar
    painter.rrect(px + 3.0f, py + 3.0f, pw - 6.0f, 22.0f, 3.5f, kMhHeader, 0.96f);
    painter.rect(px + 4.0f, py + 25.0f, pw - 8.0f, 1.0f, kMhGoldDim, 0.85f);
    painter.text(px + 12.0f, py + 17.5f, "Hunter Loadouts", 10.5f, kMhGoldBright, 0.98f, true);
    const std::string sub = "Box Controls Locked";
    const float sub_w = painter.font_ok() ? painter.measure(sub, 7.2f, false) : 64.0f;
    painter.text(px + pw - 12.0f - sub_w, py + 16.8f, sub, 7.2f, kMhMuted, 0.90f, false);

    // Left Column: Tabs + 5 Loadout Slots
    const float lx = px + 10.0f;
    const float lw = 148.0f;
    const bool on_items = tab_ == Tab::Items;
    const float tab_y = py + 31.0f;
    const float tab_w = (lw - 4.0f) * 0.5f;
    auto draw_tab = [&](float x, const std::string &title, bool active) {
        painter.rrect(x, tab_y, tab_w, 16.5f, 3.5f, active ? kMhSelectTop : kMhRow, active ? 0.95f : 0.80f, 0.0f,
                      active ? &kMhSelectBot : nullptr);
        painter.rrect(x, tab_y, tab_w, 16.5f, 3.5f, active ? kMhGoldBright : kMhGoldDim, active ? 0.95f : 0.55f, 0.85f);
        const float tw = painter.font_ok() ? painter.measure(title, 7.8f, true) : 42.0f;
        painter.text(x + (tab_w - tw) * 0.5f, tab_y + 11.2f, title, 7.8f,
                     active ? mhf::overlay::kInk : kMhMuted, 0.98f, true);
    };
    draw_tab(lx, "Item Sets", on_items);
    draw_tab(lx + tab_w + 4.0f, "Armor Sets", !on_items);

    for (std::size_t i = 0; i < kItemLoadoutCount; ++i) {
        const float ry = py + 54.0f + static_cast<float>(i) * 23.0f;
        const bool sel = i == selected_;
        painter.rrect(lx, ry, lw, 19.5f, 3.5f, sel ? kMhSelectTop : kMhRow, sel ? 0.94f : 0.82f, 0.0f,
                      sel ? &kMhSelectBot : nullptr);
        if (sel) painter.rrect(lx, ry, lw, 19.5f, 3.5f, kMhGoldBright, 0.98f, 1.0f);

        const std::string name = (on_items ? "Item Set " : "Armor Set ") + std::to_string(i + 1);
        painter.text(lx + 8.0f, ry + 12.8f, name, 8.2f, sel ? mhf::overlay::kInk : kMhParchment, 0.98f, true);

        const bool saved = on_items ? presets_[i].saved : armor_presets_[i].saved;
        const std::string detail = !saved ? "Empty"
                                          : (on_items ? std::to_string(count_non_empty_items(presets_[i])) + "/24"
                                                      : std::to_string(count_non_empty_armor(armor_presets_[i])) + "/6");
        const float dw = painter.font_ok() ? painter.measure(detail, 7.6f, false) : 26.0f;
        painter.text(lx + lw - 8.0f - dw, ry + 12.6f, detail, 7.6f,
                     sel ? mhf::overlay::kInk : (saved ? kMhGold : kMhMuted), 0.95f, sel);
    }

    // Status Toast inside Left Column below slots
    if (!toast_.empty() && now_us < toast_until_us_) {
        const float ty = py + 174.0f;
        painter.rrect(lx, ty, lw, 17.0f, 3.0f, toast_ok_ ? kMhOkBg : kMhErrBg, 0.94f);
        painter.rrect(lx, ty, lw, 17.0f, 3.0f, toast_ok_ ? kMhOkText : kMhErrText, 0.55f, 0.7f);
        painter.text(lx + 6.0f, ty + 11.4f, toast_, 7.2f, toast_ok_ ? kMhOkText : kMhErrText, 0.98f, true);
    }

    // Vertical Divider between Set List and Live Visualizer
    const float div_x = lx + lw + 8.0f;
    painter.rect(div_x, py + 31.0f, 0.8f, ph - 55.0f, kMhGoldDim, 0.70f);

    // Right Column: Live Modern MH Visualizer
    const float rx = div_x + 8.0f;
    const float rw = (px + pw - 10.0f) - rx;
    const std::uint32_t gs = valid_game_sys(memory, symbols_);
    const std::uint32_t text_res = find_master_text_resource(memory, symbols_);

    if (on_items) {
        const bool saved = presets_[selected_].saved;
        std::array<ItemSlot, kItemPouchSlots> view_slots{};
        if (saved) {
            view_slots = presets_[selected_].slots;
        } else if (gs != 0u) {
            for (std::size_t s = 0; s < kItemPouchSlots; ++s) {
                view_slots[s] = read_slot(memory, gs + kGameSysPouchBase, s);
            }
        }
        std::size_t used = 0;
        for (const auto &s : view_slots) {
            if (s.id != 0u && s.count > 0u) ++used;
        }
        const std::string hdr = saved ? ("Item Set " + std::to_string(selected_ + 1) + " — Pouch Preview")
                                      : "Current Pouch (Press △ to Save)";
        painter.text(rx, py + 40.5f, hdr, 8.2f, saved ? kMhGoldBright : kMhParchment, 0.96f, true);
        const std::string cnt = std::to_string(used) + " / 24";
        const float cw = painter.font_ok() ? painter.measure(cnt, 7.5f, false) : 24.0f;
        painter.text(rx + rw - cw, py + 40.5f, cnt, 7.5f, kMhMuted, 0.92f, false);

        // 3 x 8 (24-slot) visual pouch grid
        constexpr std::size_t kCols = 3u;
        constexpr std::size_t kRows = 8u;
        const float gap_x = 4.0f;
        const float gap_y = 2.5f;
        const float cell_w = (rw - gap_x * static_cast<float>(kCols - 1u)) / static_cast<float>(kCols);
        const float cell_h = 16.0f;
        const float grid_y = py + 46.5f;

        for (std::size_t idx = 0; idx < kItemPouchSlots; ++idx) {
            const std::size_t col = idx / kRows;
            const std::size_t row = idx % kRows;
            const float cx = rx + static_cast<float>(col) * (cell_w + gap_x);
            const float cy = grid_y + static_cast<float>(row) * (cell_h + gap_y);
            const ItemSlot s = view_slots[idx];
            const bool occupied = s.id != 0u && s.count > 0u;

            painter.rrect(cx, cy, cell_w, cell_h, 2.5f, kMhRow, occupied ? 0.88f : 0.38f);
            painter.rrect(cx, cy, cell_w, cell_h, 2.5f, kMhGoldDim, occupied ? 0.55f : 0.22f, 0.65f);
            if (!occupied) continue;

            painter.rrect(cx + 2.5f, cy + 3.0f, 3.2f, cell_h - 6.0f, 1.5f, item_accent_color(s.id), 0.96f);

            const std::string qty = "x" + std::to_string(s.count);
            const float qw = painter.font_ok() ? painter.measure(qty, 6.4f, true) : 14.0f;
            const float max_name_w = cell_w - 8.0f - qw - 6.5f;
            const std::string name = fit_text(painter, resolve_item_name(memory, text_res, text_archive_, s.id), 6.4f, max_name_w);
            painter.text(cx + 8.0f, cy + 10.7f, name, 6.4f, kMhParchment, 0.96f, false);
            painter.text(cx + cell_w - 3.5f - qw, cy + 10.7f, qty, 6.4f, kMhGoldBright, 0.96f, true);
        }
    } else {
        const bool saved = armor_presets_[selected_].saved;
        ArmorLoadoutPreset view_armor{};
        if (saved) {
            view_armor = armor_presets_[selected_];
        } else if (gs != 0u) {
            for (std::size_t s = 0; s < kEquipSlots; ++s) {
                view_armor.pieces[s] = read_piece(memory, gs + kGameSysEquippedBase + static_cast<std::uint32_t>(s) * kEquipPieceSize);
                view_armor.box_indices[s] = memory.load16(gs + kGameSysEquipIdxBase + static_cast<std::uint32_t>(s) * 2u);
            }
        }
        const std::string hdr = saved ? ("Armor Set " + std::to_string(selected_ + 1) + " — Equipment Inspector")
                                      : "Current Equipment (Press △ to Save)";
        painter.text(rx, py + 40.5f, hdr, 8.2f, saved ? kMhGoldBright : kMhParchment, 0.96f, true);

        // Display in modern MH vertical order: Weapon, Head, Chest, Arms, Waist, Legs.
        struct SlotView {
            std::size_t raw_slot;
            const char *tag;
            mhf::overlay::Color badge;
        };
        constexpr std::array<SlotView, kEquipSlots> kOrder{{
            {0u, "WPN",   {0.89f, 0.56f, 0.24f}},
            {2u, "HEAD",  {0.44f, 0.72f, 0.92f}},
            {3u, "CHEST", {0.42f, 0.78f, 0.55f}},
            {4u, "ARMS",  {0.78f, 0.62f, 0.88f}},
            {5u, "WAIST", {0.86f, 0.72f, 0.36f}},
            {1u, "LEGS",  {0.72f, 0.64f, 0.52f}},
        }};

        const float list_y = py + 48.0f;
        const float row_h = 21.5f;
        const float row_stride = 24.0f;
        for (std::size_t r = 0; r < kEquipSlots; ++r) {
            const auto &sv = kOrder[r];
            const float cy = list_y + static_cast<float>(r) * row_stride;
            const EquipPiece &piece = view_armor.pieces[sv.raw_slot];
            const bool occ = piece_non_empty(piece);

            painter.rrect(rx, cy, rw, row_h, 3.2f, kMhRow, occ ? 0.88f : 0.42f);
            painter.rrect(rx, cy, rw, row_h, 3.2f, kMhGoldDim, occ ? 0.58f : 0.25f, 0.7f);

            // Slot tag pill
            painter.rrect(rx + 4.0f, cy + 4.0f, 34.0f, row_h - 8.0f, 2.5f, sv.badge, occ ? 0.88f : 0.35f);
            const float tw = painter.font_ok() ? painter.measure(sv.tag, 6.4f, true) : 20.0f;
            painter.text(rx + 4.0f + (34.0f - tw) * 0.5f, cy + 13.0f, sv.tag, 6.4f,
                         occ ? mhf::overlay::kInk : kMhMuted, 0.96f, true);

            if (!occ) {
                painter.text(rx + 44.0f, cy + 13.5f, "Empty", 7.6f, kMhMuted, 0.75f, false);
                continue;
            }

            std::string desc = resolve_equip_name(memory, text_res, text_archive_, sv.raw_slot, piece);
            if (sv.raw_slot != 0u && piece[0] > 0u) desc += " Lv." + std::to_string(static_cast<unsigned>(piece[0]) + 1u);
            const std::size_t decos = count_decorations(piece);
            if (decos > 0u) desc += " [" + std::to_string(decos) + " deco]";
            const float st_w = 42.0f;
            const std::string fitted = fit_text(painter, std::move(desc), 7.4f, rw - 44.0f - st_w - 10.0f);
            painter.text(rx + 44.0f, cy + 13.4f, fitted, 7.4f, kMhParchment, 0.96f, false);

            // Live ownership status badge against current equipped / Equipment Box
            bool equipped_now = false;
            if (gs != 0u) {
                const EquipPiece cur = read_piece(memory, gs + kGameSysEquippedBase +
                                                              static_cast<std::uint32_t>(sv.raw_slot) * kEquipPieceSize);
                equipped_now = (cur == piece);
            }
            const bool in_box = equipped_now || piece_in_equip_box(memory, gs, piece, view_armor.box_indices[sv.raw_slot]);
            const std::string status = equipped_now ? "Equipped" : (in_box ? "In Box" : "Missing");
            const mhf::overlay::Color st_bg = equipped_now ? kMhHeader : (in_box ? kMhOkBg : kMhErrBg);
            const mhf::overlay::Color st_fg = equipped_now ? kMhGoldBright : (in_box ? kMhOkText : kMhErrText);
            painter.rrect(rx + rw - st_w - 5.0f, cy + 4.5f, st_w, row_h - 9.0f, 2.5f, st_bg, 0.92f);
            const float st_tw = painter.font_ok() ? painter.measure(status, 6.4f, true) : 26.0f;
            painter.text(rx + rw - st_w - 5.0f + (st_w - st_tw) * 0.5f, cy + 13.0f, status, 6.4f, st_fg, 0.96f, true);
        }
    }

    // Footer Controls Bar with PSP-style button pills
    const float fy = py + ph - 17.5f;
    painter.rect(px + 8.0f, fy - 2.5f, pw - 16.0f, 0.75f, kMhGoldDim, 0.65f);
    float fx = px + 10.0f;
    auto add_hint = [&](const std::string &btn, float bw, const std::string &act) {
        draw_button_pill(painter, fx, fy, bw, 11.5f, btn);
        fx += bw + 3.5f;
        fx += painter.text(fx, fy + 8.5f, act, 7.5f, kMhParchment, 0.92f, false) + 10.0f;
    };
    add_hint("O", 13.0f, on_items ? "Load Item Set" : "Equip Armor Set");
    add_hint("△", 13.0f, "Save Current");
    add_hint("L/R", 20.0f, "Switch Tab");
    add_hint("X", 13.0f, "Close");
    return true;
}

} // namespace mhf::mhfu
