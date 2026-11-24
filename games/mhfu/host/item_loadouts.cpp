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
constexpr std::uint64_t kBoxHeartbeatUs = 80'000u;   // disappears promptly (~2-3 frames) when closing the box
constexpr std::uint64_t kSelectHoldOpenUs = 350'000u; // hold SELECT 0.35s to open Loadouts; tap SELECT to Sort
constexpr std::uint64_t kSortPulseUs = 50'000u;       // brief SELECT pulse forwarded to MHFU for native Sort

// Native MHFU menu palette (matched to Pokke House Item Box, Equipment Box, and Smithy UI).
constexpr mhf::overlay::Color kMhPanelBg{0.075f, 0.070f, 0.062f};
constexpr mhf::overlay::Color kMhCellBg{0.032f, 0.032f, 0.032f};
constexpr mhf::overlay::Color kMhBrassLight{0.804f, 0.651f, 0.373f};
constexpr mhf::overlay::Color kMhBrassDark{0.463f, 0.349f, 0.188f};
constexpr mhf::overlay::Color kMhFrameInk{0.055f, 0.047f, 0.039f};
constexpr mhf::overlay::Color kMhSelectGreen{0.110f, 0.635f, 0.220f};
constexpr mhf::overlay::Color kMhArrowGreen{0.180f, 0.880f, 0.240f};
constexpr mhf::overlay::Color kMhHeaderYellow{0.925f, 0.871f, 0.341f};
constexpr mhf::overlay::Color kMhRuleGreen{0.455f, 0.678f, 0.420f};
constexpr mhf::overlay::Color kMhRibbonBg{0.114f, 0.251f, 0.078f};
constexpr mhf::overlay::Color kMhRibbonEdge{0.278f, 0.420f, 0.180f};
constexpr mhf::overlay::Color kMhRibbonTribal{0.235f, 0.392f, 0.145f};
constexpr mhf::overlay::Color kMhWhite{0.960f, 0.955f, 0.935f};
constexpr mhf::overlay::Color kMhDimText{0.620f, 0.612f, 0.576f};
constexpr mhf::overlay::Color kMhEquippedBlue{0.118f, 0.361f, 0.620f};
constexpr mhf::overlay::Color kMhOkText{0.640f, 0.940f, 0.580f};
constexpr mhf::overlay::Color kMhErrBg{0.340f, 0.110f, 0.100f};
constexpr mhf::overlay::Color kMhErrText{1.0f, 0.680f, 0.640f};
constexpr mhf::overlay::Color kBtnCapBg{0.130f, 0.130f, 0.130f};
constexpr mhf::overlay::Color kBtnCapBorder{0.580f, 0.575f, 0.545f};
constexpr mhf::overlay::Color kBtnCircle{0.930f, 0.350f, 0.350f};
constexpr mhf::overlay::Color kBtnTriangle{0.280f, 0.860f, 0.480f};
constexpr mhf::overlay::Color kBtnCross{0.430f, 0.670f, 0.960f};

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

// Draws MHFU's signature translucent charcoal window with notched antique-brass borders and 4 corner clasps.
void draw_mh_window(mhf::overlay::Painter &painter, float x, float y, float w, float h, float bg_alpha = 0.80f) {
    // Smoked translucent interior
    painter.rect(x + 2.0f, y + 2.0f, w - 4.0f, h - 4.0f, kMhPanelBg, bg_alpha);

    // Outer dark ink outline
    painter.rect(x, y, w, 1.0f, kMhFrameInk, 0.96f);
    painter.rect(x, y + h - 1.0f, w, 1.0f, kMhFrameInk, 0.96f);
    painter.rect(x, y, 1.0f, h, kMhFrameInk, 0.96f);
    painter.rect(x + w - 1.0f, y, 1.0f, h, kMhFrameInk, 0.96f);

    // Antique-brass rails (2px thick)
    painter.rect(x + 1.0f, y + 1.0f, w - 2.0f, 2.0f, kMhBrassLight, 0.96f);
    painter.rect(x + 1.0f, y + h - 3.0f, w - 2.0f, 2.0f, kMhBrassLight, 0.96f);
    painter.rect(x + 1.0f, y + 1.0f, 2.0f, h - 2.0f, kMhBrassLight, 0.96f);
    painter.rect(x + w - 3.0f, y + 1.0f, 2.0f, h - 2.0f, kMhBrassLight, 0.96f);

    // Inner dark hairline
    painter.rect(x + 3.0f, y + 3.0f, w - 6.0f, 0.75f, kMhFrameInk, 0.92f);
    painter.rect(x + 3.0f, y + h - 3.75f, w - 6.0f, 0.75f, kMhFrameInk, 0.92f);
    painter.rect(x + 3.0f, y + 3.0f, 0.75f, h - 6.0f, kMhFrameInk, 0.92f);
    painter.rect(x + w - 3.75f, y + 3.0f, 0.75f, h - 6.0f, kMhFrameInk, 0.92f);

    // Repeating dark square notches along all 4 rails (iconic MHFU riveted border)
    constexpr float kStep = 6.5f;
    for (float nx = x + 6.5f; nx < x + w - 8.5f; nx += kStep) {
        painter.rect(nx, y + 1.0f, 2.2f, 2.0f, kMhFrameInk, 0.90f);
        painter.rect(nx, y + h - 3.0f, 2.2f, 2.0f, kMhFrameInk, 0.90f);
    }
    for (float ny = y + 6.5f; ny < y + h - 8.5f; ny += kStep) {
        painter.rect(x + 1.0f, ny, 2.0f, 2.2f, kMhFrameInk, 0.90f);
        painter.rect(x + w - 3.0f, ny, 2.0f, 2.2f, kMhFrameInk, 0.90f);
    }

    // 4-corner square brass clasps
    auto draw_clasp = [&](float cx, float cy) {
        painter.rect(cx, cy, 4.6f, 4.6f, kMhFrameInk, 0.98f);
        painter.rect(cx + 0.7f, cy + 0.7f, 3.2f, 3.2f, kMhBrassLight, 0.98f);
        painter.rect(cx + 1.6f, cy + 1.6f, 1.4f, 1.4f, kMhBrassDark, 0.98f);
    };
    draw_clasp(x - 0.5f, y - 0.5f);
    draw_clasp(x + w - 4.1f, y - 0.5f);
    draw_clasp(x - 0.5f, y + h - 4.1f);
    draw_clasp(x + w - 4.1f, y + h - 4.1f);
}

// Draws MHFU's dark forest-green prompt ribbon with geometric tribal right-end cap and optional hold fill.
void draw_mh_ribbon(mhf::overlay::Painter &painter, float x, float y, float w, float h, float progress = 0.0f) {
    painter.rect(x, y, w, h, kMhRibbonBg, 0.92f);
    painter.rect(x, y, w, 0.85f, kMhRibbonEdge, 0.92f);
    painter.rect(x, y + h - 0.85f, w, 0.85f, kMhRibbonEdge, 0.92f);
    painter.rect(x, y, 0.85f, h, kMhRibbonEdge, 0.92f);

    if (progress > 0.001f) {
        const float fw = std::clamp(progress, 0.0f, 1.0f) * (w - 14.0f);
        painter.rect(x + 1.0f, y + 1.0f, fw, h - 2.0f, kMhSelectGreen, 0.88f);
    }

    // Geometric tribal motif on the right edge (matching Pokke House / Smithy green bars)
    const float tx = x + w - 12.0f;
    painter.rect(tx, y + 1.5f, 1.2f, h - 3.0f, kMhRibbonTribal, 0.90f);
    painter.rect(tx + 2.5f, y + 2.2f, 7.0f, 1.2f, kMhRibbonTribal, 0.90f);
    painter.rect(tx + 2.5f, y + h - 3.4f, 7.0f, 1.2f, kMhRibbonTribal, 0.90f);
    painter.rect(tx + 5.2f, y + 4.5f, 3.2f, h - 9.0f, kMhRibbonTribal, 0.85f);
    painter.rect(x + w - 1.5f, y + 3.0f, 1.5f, h - 6.0f, kMhRibbonEdge, 0.78f);
}

// Draws MHFU's iconic emerald-green horizontal selection glow (strong on the left, fading right).
void draw_mh_select_bar(mhf::overlay::Painter &painter, float x, float y, float w, float h) {
    constexpr int kBands = 14;
    const float bw = w / static_cast<float>(kBands);
    for (int i = 0; i < kBands; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(kBands - 1);
        const float alpha = 0.86f * (1.0f - 0.72f * t * t);
        painter.rect(x + static_cast<float>(i) * bw, y, bw + 0.35f, h, kMhSelectGreen, alpha);
    }
    painter.rect(x, y, w, 0.65f, kMhArrowGreen, 0.55f);
    painter.rect(x, y + h - 0.65f, w, 0.65f, kMhArrowGreen, 0.45f);
}

// Draws a PSP-style dark keycap ("SELECT", "L1", "R1") or colored PlayStation face-button badge ("O", "△", "X").
void draw_ps_button(mhf::overlay::Painter &painter, float x, float y, float w, float h, const std::string &label,
                    mhf::overlay::Color fg, bool round_badge) {
    const float rad = round_badge ? h * 0.5f : 1.8f;
    painter.rrect(x, y, w, h, rad, kBtnCapBg, 0.96f);
    painter.rrect(x, y, w, h, rad, round_badge ? fg : kBtnCapBorder, 0.92f, 0.8f);
    const float fsize = round_badge ? 6.8f : 5.8f;
    const float tw = painter.font_ok() ? painter.measure(label, fsize, true, 0.15f) : w * 0.6f;
    painter.text(x + (w - tw) * 0.5f, y + h * 0.74f, label, fsize, fg, 0.98f, true, 0.15f);
}

// MHFU authentic rarity color palette (0-based rarity 0..9 -> RARE 1..10).
mhf::overlay::Color rarity_color(std::uint8_t rarity) noexcept {
    switch (rarity) {
        case 0u:
        case 1u:
        case 2u: return {0.95f, 0.95f, 0.93f}; // RARE 1-3: White
        case 3u: return {0.38f, 0.86f, 0.48f}; // RARE 4: Green
        case 4u: return {0.95f, 0.36f, 0.36f}; // RARE 5: Red
        case 5u: return {0.38f, 0.72f, 0.96f}; // RARE 6: Sky Blue
        case 6u: return {0.96f, 0.86f, 0.34f}; // RARE 7: Gold
        case 7u: return {0.96f, 0.58f, 0.26f}; // RARE 8: Orange
        case 8u: return {0.78f, 0.48f, 0.96f}; // RARE 9: Violet
        default: return {0.96f, 0.42f, 0.74f}; // RARE 10: Magenta
    }
}

// Draws a premultiplied RGBA sprite tinted by `tint` while keeping alpha and dark outlines crisp.
void draw_tinted_icon(mhf::overlay::Painter &painter, float x, float y, float draw_size,
                      const LoadoutSprite &sprite, mhf::overlay::Color tint, float opacity = 1.0f) {
    if (!sprite.valid()) return;
    if (tint.r >= 0.94f && tint.g >= 0.94f && tint.b >= 0.92f) {
        painter.image(x, y, draw_size, sprite.pixels->data(), sprite.size, opacity);
        return;
    }
    const auto &src = *sprite.pixels;
    std::vector<std::uint32_t> tinted(src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        const std::uint32_t px = src[i];
        const std::uint32_t a = (px >> 24u) & 0xFFu;
        if (a == 0u) continue;
        const std::uint32_t r = static_cast<std::uint32_t>(std::clamp(static_cast<float>(px & 0xFFu) * tint.r, 0.0f, 255.0f));
        const std::uint32_t g = static_cast<std::uint32_t>(std::clamp(static_cast<float>((px >> 8u) & 0xFFu) * tint.g, 0.0f, 255.0f));
        const std::uint32_t b = static_cast<std::uint32_t>(std::clamp(static_cast<float>((px >> 16u) & 0xFFu) * tint.b, 0.0f, 255.0f));
        tinted[i] = r | (g << 8u) | (b << 16u) | (a << 24u);
    }
    painter.image(x, y, draw_size, tinted.data(), sprite.size, opacity);
}

struct EquipMeta {
    std::optional<std::uint8_t> weapon_class;
    std::uint8_t rarity{0u};
};

EquipMeta resolve_equip_meta(const psprecomp::GuestMemory &memory, const ItemLoadoutsSymbols &symbols,
                             std::size_t raw_slot, const EquipPiece &piece) noexcept {
    EquipMeta meta{};
    if (!piece_non_empty(piece)) return meta;
    const std::uint16_t piece_id = static_cast<std::uint16_t>(piece[2]) | (static_cast<std::uint16_t>(piece[3]) << 8u);
    constexpr std::uint32_t kBlademasterCount = 1150u;
    constexpr std::uint32_t kBlademasterStride = 24u;
    constexpr std::uint32_t kGunnerStride = 28u;
    constexpr std::uint32_t kWeaponModels = 553u;
    constexpr std::uint32_t kArmorStride = 40u;

    if (raw_slot == 0u) {
        if (!symbols.blademaster_weapons || !symbols.weapon_model_classes) return meta;
        std::uint32_t entry = 0u;
        if (piece[1] == 5u && piece_id < kBlademasterCount) {
            entry = *symbols.blademaster_weapons + static_cast<std::uint32_t>(piece_id) * kBlademasterStride;
        } else if (piece[1] == 6u && piece_id < 512u) {
            entry = *symbols.blademaster_weapons + kBlademasterCount * kBlademasterStride +
                    static_cast<std::uint32_t>(piece_id) * kGunnerStride;
        }
        if (entry != 0u && memory.contains(entry, 4u)) {
            const std::uint16_t model = memory.load16(entry);
            meta.rarity = std::min<std::uint8_t>(memory.load8(entry + 2u), 9u);
            if (model < kWeaponModels && memory.contains(*symbols.weapon_model_classes + model, 1u)) {
                const std::uint8_t cls = memory.load8(*symbols.weapon_model_classes + model);
                if (cls < 11u) meta.weapon_class = cls;
            }
        }
        return meta;
    }

    // Armor tables (Legs=0, Head=1, Chest=2, Arms=3, Waist=4) are pointed to by u32[5] immediately before mh_weapon_model_classes.
    if (raw_slot >= 1u && raw_slot <= 5u && symbols.weapon_model_classes && *symbols.weapon_model_classes >= 20u) {
        const std::uint32_t ptr_addr = *symbols.weapon_model_classes - 20u + static_cast<std::uint32_t>(raw_slot - 1u) * 4u;
        if (readable_word_address(memory, ptr_addr)) {
            const std::uint32_t table = memory.load32(ptr_addr);
            const std::uint32_t entry = table + static_cast<std::uint32_t>(piece_id) * kArmorStride;
            if (table != 0u && piece_id < 512u && memory.contains(entry + 5u, 1u)) {
                meta.rarity = std::min<std::uint8_t>(memory.load8(entry + 5u), 9u);
            }
        }
    }
    return meta;
}

const LoadoutSprite *pick_item_icon(const LoadoutIcons &icons, std::uint16_t id) noexcept {
    auto use = [](const LoadoutSprite &s) -> const LoadoutSprite * { return s.valid() ? &s : nullptr; };
    if (id >= 1u && id <= 5u) return use(icons.book);
    if (id >= 7u && id <= 28u) return use(icons.potion);
    if (id == 64u) return use(icons.paintball);
    if (id >= 60u && id <= 76u) return use(icons.bomb);
    if (id >= 77u && id <= 84u) return use(icons.trap);
    if (id == 135u || id == 867u) return use(icons.whetstone);
    if (id >= 136u && id <= 165u) return use(icons.pickaxe);
    if (id >= 85u && id <= 134u) return use(icons.ammo);
    return use(icons.pouch);
}

mhf::overlay::Color item_icon_tint(std::uint16_t id) noexcept {
    if (id >= 1u && id <= 5u) return {0.94f, 0.80f, 0.38f};   // Books of Combos: Gold
    if (id == 7u || id == 8u) return {0.38f, 0.88f, 0.46f};   // Potion / Mega Potion: Green
    if (id == 11u || id == 12u) return {0.96f, 0.86f, 0.34f}; // Max / Ancient Potion: Gold
    if (id == 13u || id == 15u || id == 21u) return {0.38f, 0.74f, 0.96f}; // Antidote / Cool Drink: Sky Blue
    if (id == 14u || id == 16u || id == 17u || id == 20u) return {0.95f, 0.38f, 0.38f}; // Hot Drink / Demondrug: Red
    if (id >= 7u && id <= 28u) return {0.46f, 0.86f, 0.56f};  // Other consumables: Green
    if (id == 64u) return {0.96f, 0.44f, 0.78f};              // Paintball: Magenta
    if (id >= 60u && id <= 76u) return {0.96f, 0.56f, 0.26f}; // Bombs: Orange
    if (id >= 77u && id <= 84u) return {0.94f, 0.84f, 0.36f}; // Traps: Yellow
    if (id == 135u || id == 867u) return {0.58f, 0.82f, 0.96f}; // Whetstone: Steel Blue
    if (id >= 136u && id <= 165u) return {0.68f, 0.84f, 0.96f}; // Gathering Tools: Light Blue
    if (id >= 85u && id <= 134u) return {0.90f, 0.68f, 0.38f};  // Ammo / Coatings: Amber
    return kMhWhite;
}

const LoadoutSprite *pick_equip_icon(const LoadoutIcons &icons, std::size_t raw_slot, const EquipPiece &piece,
                                     std::optional<std::uint8_t> weapon_class) noexcept {
    auto use = [](const LoadoutSprite &s) -> const LoadoutSprite * { return s.valid() ? &s : nullptr; };
    switch (raw_slot) {
        case 0u:
            if (weapon_class && *weapon_class < icons.weapon_classes.size() && icons.weapon_classes[*weapon_class].valid()) {
                return &icons.weapon_classes[*weapon_class];
            }
            return use(piece[1] == 5u ? icons.weapon_blade : icons.weapon_gun);
        case 1u: return use(icons.legs);
        case 2u: return use(icons.head);
        case 3u: return use(icons.chest_armor);
        case 4u: return use(icons.arms);
        case 5u: return use(icons.waist);
        default: return nullptr;
    }
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

void ItemLoadouts::set_icons(LoadoutIcons icons) {
    icons_ = std::move(icons);
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

std::uint32_t ItemLoadouts::game_button_clear_mask() const noexcept {
    if (!box_active_ || panel_open_ || sort_pulse_active_) return 0u;
    return mhf::psp::buttons::kSelect;
}

std::uint32_t ItemLoadouts::game_button_force_mask() const noexcept {
    if (!box_active_ || panel_open_ || !sort_pulse_active_) return 0u;
    return mhf::psp::buttons::kSelect;
}

float ItemLoadouts::select_hold_progress(std::uint64_t now_us) const noexcept {
    if (!select_holding_ || now_us < select_hold_start_us_) return 0.0f;
    return std::clamp(static_cast<float>(now_us - select_hold_start_us_) / static_cast<float>(kSelectHoldOpenUs), 0.0f, 1.0f);
}

void ItemLoadouts::poll_box_ui(psprecomp::GuestMemory &memory, std::uint32_t buttons, std::uint64_t now_us, bool enabled) {
    const std::uint32_t pressed = buttons & ~prev_buttons_;
    prev_buttons_ = buttons;

    using namespace mhf::psp::buttons;
    if (release_guard_ && (buttons & (kCross | kSelect | kStart | kCircle | kTriangle)) == 0u) {
        release_guard_ = false;
    }

    if (!enabled || !box_open(now_us) || !can_use_loadouts(memory)) {
        box_active_ = false;
        panel_open_ = false;
        release_guard_ = false;
        select_holding_ = false;
        sort_pulse_active_ = false;
        sort_pulse_until_us_ = 0u;
        return;
    }
    box_active_ = true;

    if (!panel_open_) {
        if ((buttons & kSelect) != 0u) {
            if (!select_holding_ && !release_guard_) {
                select_holding_ = true;
                select_hold_start_us_ = now_us;
            } else if (select_holding_ && now_us >= select_hold_start_us_ &&
                       (now_us - select_hold_start_us_) >= kSelectHoldOpenUs) {
                select_holding_ = false;
                sort_pulse_active_ = false;
                sort_pulse_until_us_ = 0u;
                panel_open_ = true;
                release_guard_ = false;
            }
        } else if (select_holding_) {
            // Released before the hold threshold: forward a clean SELECT tap so MHFU sorts normally.
            select_holding_ = false;
            sort_pulse_until_us_ = now_us + kSortPulseUs;
        }
        sort_pulse_active_ = !panel_open_ && now_us < sort_pulse_until_us_;
        return;
    }
    select_holding_ = false;
    sort_pulse_active_ = false;
    sort_pulse_until_us_ = 0u;

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
        // Cohesive continuation of the Item Box's left-side green hint strip directly beneath
        // "(L) Rotate" and "[START] Menu OFF" (soft translucent green fading right, no harsh box).
        const float hold = select_hold_progress(now_us);
        const float rx = 0.0f;
        const float ry = 171.5f;
        const float rw = 162.0f;
        const float rh = 15.5f;

        constexpr int kBands = 16;
        const float bw = rw / static_cast<float>(kBands);
        for (int i = 0; i < kBands; ++i) {
            const float t = static_cast<float>(i) / static_cast<float>(kBands - 1);
            const float alpha = 0.72f * (1.0f - 0.85f * t * t);
            painter.rect(rx + static_cast<float>(i) * bw, ry, bw + 0.35f, rh, kMhRibbonBg, alpha);
        }
        if (hold > 0.001f) {
            const float fw = std::clamp(hold, 0.0f, 1.0f) * (rw - 12.0f);
            painter.rect(rx, ry, fw, rh, kMhSelectGreen, 0.78f);
        }

        // Align the SELECT cap and label with MHFU's "(L) Rotate" and "[START] Menu OFF" column.
        draw_ps_button(painter, rx + 10.5f, ry + 2.8f, 22.0f, 9.5f, "SELECT", kMhWhite, false);
        const std::string lbl = hold > 0.01f ? "Opening..." : "Hold  Loadouts";
        painter.text(rx + 36.5f, ry + 11.4f, lbl, 8.6f, kMhFrameInk, 0.85f, false, 0.6f);
        painter.text(rx + 35.8f, ry + 10.7f, lbl, 8.6f, hold > 0.01f ? kMhHeaderYellow : kMhWhite, 0.98f, false, 0.6f);
        return true;
    }

    const bool on_items = tab_ == Tab::Items;
    const std::uint32_t gs = valid_game_sys(memory, symbols_);
    const std::uint32_t text_res = find_master_text_resource(memory, symbols_);

    // Two-window native MHFU layout (matching Equipment Box & Smithy screens).
    const float win_y = 14.0f;
    const float win_h = 234.0f;
    const float lx = 10.0f, lw = 152.0f;
    const float rx = 168.0f, rw = 302.0f;

    draw_mh_window(painter, lx, win_y, lw, win_h, 0.82f);
    draw_mh_window(painter, rx, win_y, rw, win_h, 0.82f);

    // --- Left Window: Set Category Header & 5 Loadout Slots ---
    const float l_in_x = lx + 8.0f;
    const float l_in_w = lw - 16.0f;
    painter.text(l_in_x, win_y + 16.5f, on_items ? "[Item Sets]" : "[Armor Sets]", 8.6f, kMhHeaderYellow, 0.98f, false);
    // Page/Tab indicator matching MHFU's green << 1 / 2 >> style
    const std::string tab_num = on_items ? "1/2" : "2/2";
    const float tn_w = painter.font_ok() ? painter.measure(tab_num, 7.4f, false) : 16.0f;
    const float tab_cx = lx + lw - 24.0f;
    painter.text(tab_cx - tn_w * 0.5f - 11.0f, win_y + 16.2f, "<<", 7.4f, kMhArrowGreen, 0.96f, true);
    painter.text(tab_cx - tn_w * 0.5f, win_y + 16.2f, tab_num, 7.4f, kMhWhite, 0.96f, false);
    painter.text(tab_cx + tn_w * 0.5f + 3.0f, win_y + 16.2f, ">>", 7.4f, kMhArrowGreen, 0.96f, true);

    // Sage-green MHFU section divider
    painter.rect(l_in_x, win_y + 22.0f, l_in_w, 1.3f, kMhRuleGreen, 0.92f);

    // Compact tab indicator pills
    const float sub_y = win_y + 27.0f;
    const float sub_w = (l_in_w - 4.0f) * 0.5f;
    auto draw_mode_tab = [&](float tx, const std::string &title, bool active) {
        painter.rect(tx, sub_y, sub_w, 13.5f, kMhCellBg, active ? 0.90f : 0.55f);
        if (active) {
            painter.rect(tx, sub_y, sub_w, 1.0f, kMhWhite, 0.92f);
            painter.rect(tx, sub_y + 12.5f, sub_w, 1.0f, kMhWhite, 0.92f);
            painter.rect(tx, sub_y, 1.0f, 13.5f, kMhWhite, 0.92f);
            painter.rect(tx + sub_w - 1.0f, sub_y, 1.0f, 13.5f, kMhWhite, 0.92f);
        } else {
            painter.rect(tx, sub_y + 12.5f, sub_w, 0.7f, kMhBrassDark, 0.65f);
        }
        const float tw = painter.font_ok() ? painter.measure(title, 7.0f, active) : 34.0f;
        painter.text(tx + (sub_w - tw) * 0.5f, sub_y + 9.4f, title, 7.0f,
                     active ? kMhHeaderYellow : kMhDimText, 0.98f, active);
    };
    draw_mode_tab(l_in_x, "Items", on_items);
    draw_mode_tab(l_in_x + sub_w + 4.0f, "Armor", !on_items);

    const LoadoutSprite *list_icon = on_items
        ? (icons_.pouch.valid() ? &icons_.pouch : nullptr)
        : (icons_.chest_armor.valid() ? &icons_.chest_armor : nullptr);

    for (std::size_t i = 0; i < kItemLoadoutCount; ++i) {
        const float ry = win_y + 47.0f + static_cast<float>(i) * 24.5f;
        const bool sel = i == selected_;
        if (sel) {
            draw_mh_select_bar(painter, l_in_x - 2.0f, ry, l_in_w + 4.0f, 20.0f);
            painter.text(l_in_x + 1.0f, ry + 13.2f, ">>", 7.2f, kMhArrowGreen, 0.98f, true);
        } else {
            painter.rect(l_in_x, ry, l_in_w, 20.0f, kMhCellBg, 0.50f);
        }

        float text_x = l_in_x + 14.0f;
        if (list_icon) {
            painter.image(l_in_x + 12.5f, ry + 4.0f, 12.0f, list_icon->pixels->data(), list_icon->size, sel ? 1.0f : 0.85f);
            text_x = l_in_x + 27.0f;
        }
        const std::string name = (on_items ? "Item Set " : "Armor Set ") + std::to_string(i + 1);
        painter.text(text_x, ry + 13.2f, name, 8.0f, sel ? kMhWhite : kMhWhite, sel ? 1.0f : 0.88f, sel);

        const bool saved = on_items ? presets_[i].saved : armor_presets_[i].saved;
        const std::string detail = !saved ? "---"
                                          : (on_items ? std::to_string(count_non_empty_items(presets_[i])) + "/24"
                                                      : std::to_string(count_non_empty_armor(armor_presets_[i])) + "/6");
        const float dw = painter.font_ok() ? painter.measure(detail, 7.4f, false) : 20.0f;
        painter.text(l_in_x + l_in_w - 4.0f - dw, ry + 13.0f, detail, 7.4f,
                     sel ? kMhHeaderYellow : (saved ? kMhWhite : kMhDimText), 0.95f, false);
    }

    // Bottom info / toast box inside Left Window
    painter.rect(l_in_x, win_y + win_h - 52.0f, l_in_w, 1.1f, kMhRuleGreen, 0.78f);
    if (!toast_.empty() && now_us < toast_until_us_) {
        painter.rect(l_in_x, win_y + win_h - 46.0f, l_in_w, 18.0f, toast_ok_ ? kMhRibbonBg : kMhErrBg, 0.92f);
        painter.text(l_in_x + 4.0f, win_y + win_h - 34.0f, fit_text(painter, toast_, 7.0f, l_in_w - 8.0f), 7.0f,
                     toast_ok_ ? kMhOkText : kMhErrText, 0.98f, true);
    } else {
        painter.text(l_in_x + 2.0f, win_y + win_h - 38.0f, "Tap SELECT outside", 6.8f, kMhDimText, 0.92f, false);
        painter.text(l_in_x + 2.0f, win_y + win_h - 28.0f, "to Sort box normally.", 6.8f, kMhDimText, 0.92f, false);
        painter.text(l_in_x + 2.0f, win_y + win_h - 14.0f, "Box Controls Locked", 6.8f, kMhHeaderYellow, 0.85f, false);
    }

    // --- Right Window: Native MHFU Pouch Grid or Equipment Inspector ---
    const float r_in_x = rx + 8.0f;
    const float r_in_w = rw - 16.0f;

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
        const std::string hdr = saved ? ("[Item Set " + std::to_string(selected_ + 1) + "]")
                                      : "[Current Pouch — Press △ to Save]";
        painter.text(r_in_x, win_y + 16.5f, hdr, 8.6f, kMhHeaderYellow, 0.98f, false);
        const std::string cnt = std::to_string(used) + " / 24";
        const float cw = painter.font_ok() ? painter.measure(cnt, 7.6f, false) : 24.0f;
        painter.text(r_in_x + r_in_w - cw, win_y + 16.5f, cnt, 7.6f, kMhWhite, 0.95f, false);
        painter.rect(r_in_x, win_y + 22.0f, r_in_w, 1.3f, kMhRuleGreen, 0.92f);

        // 3 x 8 (24-slot) black-framed MHFU item pouch grid with real native MHFU item icons
        constexpr std::size_t kCols = 3u;
        constexpr std::size_t kRows = 8u;
        const float gap_x = 3.5f;
        const float gap_y = 2.5f;
        const float cell_w = (r_in_w - gap_x * static_cast<float>(kCols - 1u)) / static_cast<float>(kCols);
        const float cell_h = 22.0f;
        const float grid_y = win_y + 28.0f;

        for (std::size_t idx = 0; idx < kItemPouchSlots; ++idx) {
            const std::size_t col = idx / kRows;
            const std::size_t row = idx % kRows;
            const float cx = r_in_x + static_cast<float>(col) * (cell_w + gap_x);
            const float cy = grid_y + static_cast<float>(row) * (cell_h + gap_y);
            const ItemSlot s = view_slots[idx];
            const bool occupied = s.id != 0u && s.count > 0u;

            painter.rect(cx, cy, cell_w, cell_h, kMhCellBg, occupied ? 0.88f : 0.52f);
            painter.rect(cx, cy, cell_w, 0.7f, kMhBrassDark, occupied ? 0.75f : 0.35f);
            painter.rect(cx, cy + cell_h - 0.7f, cell_w, 0.7f, kMhBrassDark, occupied ? 0.75f : 0.35f);
            painter.rect(cx, cy, 0.7f, cell_h, kMhBrassDark, occupied ? 0.75f : 0.35f);
            painter.rect(cx + cell_w - 0.7f, cy, 0.7f, cell_h, kMhBrassDark, occupied ? 0.75f : 0.35f);
            if (!occupied) continue;

            float name_x = cx + 4.0f;
            if (const LoadoutSprite *ico = pick_item_icon(icons_, s.id)) {
                draw_tinted_icon(painter, cx + 2.5f, cy + 4.0f, 14.0f, *ico, item_icon_tint(s.id), 0.98f);
                name_x = cx + 18.0f;
            } else {
                painter.rect(cx + 2.5f, cy + 4.0f, 2.5f, cell_h - 8.0f, item_icon_tint(s.id), 0.96f);
                name_x = cx + 7.5f;
            }

            const std::string qty = "x" + std::to_string(s.count);
            const float qw = painter.font_ok() ? painter.measure(qty, 6.4f, false) : 13.0f;
            const float max_name_w = (cx + cell_w - 3.5f - qw - 2.5f) - name_x;
            const std::string name = fit_text(painter, resolve_item_name(memory, text_res, text_archive_, s.id), 6.5f, max_name_w);
            painter.text(name_x, cy + 13.5f, name, 6.5f, kMhWhite, 0.98f, false);
            painter.text(cx + cell_w - 3.0f - qw, cy + 13.5f, qty, 6.4f, kMhHeaderYellow, 0.98f, false);
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
        const std::string hdr = saved ? ("[Armor Set " + std::to_string(selected_ + 1) + "]")
                                      : "[Current Equipment — Press △ to Save]";
        painter.text(r_in_x, win_y + 16.5f, hdr, 8.6f, kMhHeaderYellow, 0.98f, false);
        painter.rect(r_in_x, win_y + 22.0f, r_in_w, 1.3f, kMhRuleGreen, 0.92f);

        struct SlotView {
            std::size_t raw_slot;
            const char *tag;
        };
        constexpr std::array<SlotView, kEquipSlots> kOrder{{
            {0u, "Weapon"},
            {2u, "Head"},
            {3u, "Chest"},
            {4u, "Arms"},
            {5u, "Waist"},
            {1u, "Legs"},
        }};

        const float list_y = win_y + 29.0f;
        const float row_h = 28.0f;
        const float row_stride = 32.0f;
        for (std::size_t r = 0; r < kEquipSlots; ++r) {
            const auto &sv = kOrder[r];
            const float cy = list_y + static_cast<float>(r) * row_stride;
            const EquipPiece &piece = view_armor.pieces[sv.raw_slot];
            const bool occ = piece_non_empty(piece);

            painter.rect(r_in_x, cy, r_in_w, row_h, kMhCellBg, occ ? 0.88f : 0.52f);
            painter.rect(r_in_x, cy, r_in_w, 0.75f, kMhBrassDark, occ ? 0.80f : 0.35f);
            painter.rect(r_in_x, cy + row_h - 0.75f, r_in_w, 0.75f, kMhBrassDark, occ ? 0.80f : 0.35f);
            painter.rect(r_in_x, cy, 0.75f, row_h, kMhBrassDark, occ ? 0.80f : 0.35f);
            painter.rect(r_in_x + r_in_w - 0.75f, cy, 0.75f, row_h, kMhBrassDark, occ ? 0.80f : 0.35f);

            // Live ownership status
            bool equipped_now = false;
            if (occ && gs != 0u) {
                const EquipPiece cur = read_piece(memory, gs + kGameSysEquippedBase +
                                                              static_cast<std::uint32_t>(sv.raw_slot) * kEquipPieceSize);
                equipped_now = (cur == piece);
            }
            const bool in_box = occ && (equipped_now || piece_in_equip_box(memory, gs, piece, view_armor.box_indices[sv.raw_slot]));

            const EquipMeta meta = resolve_equip_meta(memory, symbols_, sv.raw_slot, piece);
            const mhf::overlay::Color rare_col = occ ? rarity_color(meta.rarity) : kMhDimText;

            // Square equipment icon box (blue highlight + yellow 'E' badge when equipped, matching Equipment Box screenshot!)
            const float box_s = 22.0f;
            const float bx = r_in_x + 3.0f;
            const float by = cy + 3.0f;
            painter.rect(bx, by, box_s, box_s, equipped_now ? kMhEquippedBlue : kMhFrameInk, 0.94f);
            if (const LoadoutSprite *ico = pick_equip_icon(icons_, sv.raw_slot, piece, meta.weapon_class)) {
                draw_tinted_icon(painter, bx + 2.0f, by + 2.0f, 18.0f, *ico, rare_col, occ ? 1.0f : 0.35f);
            } else {
                painter.text(bx + 3.0f, by + 14.0f, sv.tag, 5.5f, rare_col, 0.92f, true);
            }
            if (equipped_now) {
                painter.rect(bx + box_s - 7.5f, by + box_s - 8.5f, 7.5f, 8.5f, kMhFrameInk, 0.88f);
                painter.text(bx + box_s - 6.0f, by + box_s - 1.5f, "E", 6.8f, kMhHeaderYellow, 1.0f, true);
            }

            if (!occ) {
                painter.text(r_in_x + 31.0f, cy + 16.5f, std::string(sv.tag) + " : ---", 7.6f, kMhDimText, 0.80f, false);
                continue;
            }

            std::string desc = resolve_equip_name(memory, text_res, text_archive_, sv.raw_slot, piece);
            if (sv.raw_slot != 0u && piece[0] > 0u) desc += " Lv." + std::to_string(static_cast<unsigned>(piece[0]) + 1u);
            const std::size_t decos = count_decorations(piece);
            if (decos > 0u) desc += " [" + std::to_string(decos) + " deco]";

            const std::string status = equipped_now ? "Equipped" : (in_box ? "In Box" : "Missing");
            const float st_w = painter.font_ok() ? painter.measure(status, 6.8f, false) + 8.0f : 38.0f;
            const std::string fitted = fit_text(painter, std::move(desc), 7.8f, r_in_w - 31.0f - st_w - 8.0f);
            painter.text(r_in_x + 31.0f, cy + 12.2f, fitted, 7.8f, kMhWhite, 0.98f, false);
            const std::string sub = std::string(sv.tag) + "  RARE-" + std::to_string(static_cast<unsigned>(meta.rarity) + 1u);
            painter.text(r_in_x + 31.0f, cy + 22.8f, sub, 6.4f, rare_col, 0.92f, false);

            const mhf::overlay::Color st_fg = equipped_now ? kMhHeaderYellow : (in_box ? kMhOkText : kMhErrText);
            const float st_tw = painter.font_ok() ? painter.measure(status, 6.8f, false) : 26.0f;
            painter.text(r_in_x + r_in_w - 6.0f - st_tw, cy + 16.8f, status, 6.8f, st_fg, 0.98f, false);
        }
    }

    // Native MHFU Dark Forest-Green Footer Controls Ribbon
    const float fy = 252.5f;
    const float fh = 15.0f;
    draw_mh_ribbon(painter, lx, fy, (rx + rw) - lx, fh);
    float fx = lx + 8.0f;
    auto add_hint = [&](const std::string &btn, float bw, mhf::overlay::Color fg, bool round_badge, const std::string &act) {
        draw_ps_button(painter, fx, fy + 2.3f, bw, 10.4f, btn, fg, round_badge);
        fx += bw + 4.0f;
        fx += painter.text(fx, fy + 10.2f, act, 7.5f, kMhWhite, 0.96f, false) + 12.0f;
    };
    add_hint("O", 11.0f, kBtnCircle, true, on_items ? "Load Item Set" : "Equip Armor Set");
    add_hint("△", 11.0f, kBtnTriangle, true, "Save Current");
    add_hint("L1/R1", 23.0f, kMhWhite, false, "Switch Tab");
    add_hint("X", 11.0f, kBtnCross, true, "Close");
    return true;
}

} // namespace mhf::mhfu
