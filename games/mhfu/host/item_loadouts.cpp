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

namespace {

constexpr int kOverlayW = 480;
constexpr int kOverlayH = 272;
constexpr std::uint64_t kBoxHeartbeatUs = 160'000u;

void blend_px(std::vector<std::uint32_t> &rgba, int x, int y, std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a) noexcept {
    if (x < 0 || x >= kOverlayW || y < 0 || y >= kOverlayH || a == 0u) return;
    const std::size_t idx = static_cast<std::size_t>(y * kOverlayW + x);
    const std::uint32_t dst = rgba[idx];
    const std::uint32_t dr = dst & 0xFFu;
    const std::uint32_t dg = (dst >> 8u) & 0xFFu;
    const std::uint32_t db = (dst >> 16u) & 0xFFu;
    const std::uint32_t da = (dst >> 24u) & 0xFFu;
    const std::uint32_t sr = (static_cast<std::uint32_t>(r) * a + 127u) / 255u;
    const std::uint32_t sg = (static_cast<std::uint32_t>(g) * a + 127u) / 255u;
    const std::uint32_t sb = (static_cast<std::uint32_t>(b) * a + 127u) / 255u;
    const std::uint32_t inv = 255u - a;
    rgba[idx] = std::min<std::uint32_t>(255u, sr + (dr * inv + 127u) / 255u) |
                (std::min<std::uint32_t>(255u, sg + (dg * inv + 127u) / 255u) << 8u) |
                (std::min<std::uint32_t>(255u, sb + (db * inv + 127u) / 255u) << 16u) |
                (std::min<std::uint32_t>(255u, a + (da * inv + 127u) / 255u) << 24u);
}

void fill_rect(std::vector<std::uint32_t> &rgba, int x0, int y0, int w, int h,
               std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a) noexcept {
    for (int y = y0; y < y0 + h; ++y) {
        for (int x = x0; x < x0 + w; ++x) blend_px(rgba, x, y, r, g, b, a);
    }
}

void stroke_rect(std::vector<std::uint32_t> &rgba, int x0, int y0, int w, int h,
                 std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a) noexcept {
    fill_rect(rgba, x0, y0, w, 1, r, g, b, a);
    fill_rect(rgba, x0, y0 + h - 1, w, 1, r, g, b, a);
    fill_rect(rgba, x0, y0, 1, h, r, g, b, a);
    fill_rect(rgba, x0 + w - 1, y0, 1, h, r, g, b, a);
}

// Minimal crisp 3x5 uppercase + digit bitmap font for native-looking 480x272 MH panels.
const char *const *glyph_3x5(char c) noexcept {
    static constexpr const char *kSpace[5] = {"000", "000", "000", "000", "000"};
    static constexpr const char *kColon[5] = {"000", "010", "000", "010", "000"};
    static constexpr const char *kCaret[5] = {"010", "101", "000", "000", "000"};
    static constexpr const char *kDash[5] = {"000", "000", "111", "000", "000"};
    static constexpr const char *kLBrk[5] = {"110", "100", "100", "100", "110"};
    static constexpr const char *kRBrk[5] = {"011", "001", "001", "001", "011"};
    static constexpr const char *kDigits[10][5] = {
        {"111", "101", "101", "101", "111"}, {"010", "110", "010", "010", "111"},
        {"111", "001", "111", "100", "111"}, {"111", "001", "111", "001", "111"},
        {"101", "101", "111", "001", "001"}, {"111", "100", "111", "001", "111"},
        {"111", "100", "111", "101", "111"}, {"111", "001", "001", "010", "010"},
        {"111", "101", "111", "101", "111"}, {"111", "101", "111", "001", "111"},
    };
    static constexpr const char *kAlpha[26][5] = {
        {"010", "101", "111", "101", "101"}, {"110", "101", "110", "101", "110"},
        {"111", "100", "100", "100", "111"}, {"110", "101", "101", "101", "110"},
        {"111", "100", "110", "100", "111"}, {"111", "100", "110", "100", "100"},
        {"111", "100", "101", "101", "111"}, {"101", "101", "111", "101", "101"},
        {"111", "010", "010", "010", "111"}, {"001", "001", "001", "101", "111"},
        {"101", "101", "110", "101", "101"}, {"100", "100", "100", "100", "111"},
        {"101", "111", "111", "101", "101"}, {"111", "101", "101", "101", "101"},
        {"111", "101", "101", "101", "111"}, {"111", "101", "111", "100", "100"},
        {"111", "101", "101", "111", "001"}, {"110", "101", "110", "101", "101"},
        {"111", "100", "111", "001", "111"}, {"111", "010", "010", "010", "010"},
        {"101", "101", "101", "101", "111"}, {"101", "101", "101", "101", "010"},
        {"101", "101", "111", "111", "101"}, {"101", "101", "010", "101", "101"},
        {"101", "101", "010", "010", "010"}, {"111", "001", "010", "100", "111"},
    };
    if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
    if (c >= '0' && c <= '9') return kDigits[c - '0'];
    if (c >= 'A' && c <= 'Z') return kAlpha[c - 'A'];
    if (c == ':') return kColon;
    if (c == '^') return kCaret;
    if (c == '-') return kDash;
    if (c == '[') return kLBrk;
    if (c == ']') return kRBrk;
    return kSpace;
}

void draw_text(std::vector<std::uint32_t> &rgba, int x0, int y0, std::string_view text, int scale,
               std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a) noexcept {
    int x = x0;
    for (char c : text) {
        const char *const *g3 = glyph_3x5(c);
        for (int row = 0; row < 5; ++row) {
            for (int col = 0; col < 3; ++col) {
                if (g3[row][col] == '1') {
                    fill_rect(rgba, x + col * scale + 1, y0 + row * scale + 1, scale, scale, 10, 8, 6, static_cast<std::uint8_t>(a * 3 / 4));
                    fill_rect(rgba, x + col * scale, y0 + row * scale, scale, scale, r, g, b, a);
                }
            }
        }
        x += 4 * scale;
    }
}

std::size_t count_non_empty(const ItemLoadoutPreset &p) noexcept {
    std::size_t n = 0;
    for (const auto &s : p.slots) {
        if (s.id != 0u && s.count > 0u) ++n;
    }
    return n;
}

} // namespace

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

    if ((pressed & kUp) != 0u) {
        selected_ = (selected_ + kItemLoadoutCount - 1u) % kItemLoadoutCount;
    } else if ((pressed & kDown) != 0u) {
        selected_ = (selected_ + 1u) % kItemLoadoutCount;
    } else if ((pressed & kCircle) != 0u) {
        const auto res = apply_preset(memory, selected_);
        toast_ = res.message;
        toast_until_us_ = now_us + 2'000'000u;
    } else if ((pressed & kTriangle) != 0u) {
        const auto res = save_preset(memory, selected_);
        toast_ = res.message;
        toast_until_us_ = now_us + 2'000'000u;
    } else if ((pressed & (kCross | kSelect | kStart)) != 0u) {
        panel_open_ = false;
    }
}

bool ItemLoadouts::render_box_ui(std::vector<std::uint32_t> &rgba, std::uint64_t now_us) const noexcept {
    if (!box_open(now_us)) return false;

    if (!panel_open_) {
        // Sleek bottom-right MH prompt pill inside the House / Guild Item Box.
        const int bx = 342, by = 246, bw = 128, bh = 18;
        fill_rect(rgba, bx, by, bw, bh, 18, 16, 14, 215);
        stroke_rect(rgba, bx, by, bw, bh, 206, 174, 104, 235);
        draw_text(rgba, bx + 7, by + 6, "[SELECT] ITEM SETS", 1, 248, 236, 200, 245);
        return true;
    }

    // Centered MH-styled Item Sets window right over the Item Box.
    const int px = 124, py = 54, pw = 232, ph = 158;
    fill_rect(rgba, px, py, pw, ph, 16, 15, 13, 232);
    stroke_rect(rgba, px, py, pw, ph, 214, 180, 108, 245);
    stroke_rect(rgba, px + 2, py + 2, pw - 4, ph - 4, 118, 96, 56, 210);

    // Header bar
    fill_rect(rgba, px + 3, py + 3, pw - 6, 20, 42, 34, 22, 235);
    draw_text(rgba, px + 14, py + 8, "ITEM POUCH SETS", 2, 252, 224, 136, 250);

    // 5 Loadout rows
    for (std::size_t i = 0; i < kItemLoadoutCount; ++i) {
        const int ry = py + 28 + static_cast<int>(i) * 19;
        const bool sel = i == selected_;
        if (sel) {
            fill_rect(rgba, px + 8, ry, pw - 16, 16, 198, 142, 42, 170);
            stroke_rect(rgba, px + 8, ry, pw - 16, 16, 255, 220, 118, 235);
        } else {
            fill_rect(rgba, px + 8, ry, pw - 16, 16, 28, 26, 22, 180);
        }
        const std::string label = "SET " + std::to_string(i + 1);
        draw_text(rgba, px + 16, ry + 5, label, 1, sel ? 255 : 230, sel ? 248 : 224, sel ? 225 : 210, 250);

        const std::string info = presets_[i].saved
                                     ? std::to_string(count_non_empty(presets_[i])) + " ITEMS"
                                     : "EMPTY";
        draw_text(rgba, px + 150, ry + 5, info, 1,
                  presets_[i].saved ? 175 : 130, presets_[i].saved ? 235 : 130, presets_[i].saved ? 165 : 130, 245);
    }

    // Status toast or footer legend
    if (!toast_.empty() && now_us < toast_until_us_) {
        fill_rect(rgba, px + 8, py + ph - 32, pw - 16, 14, 34, 58, 36, 225);
        draw_text(rgba, px + 14, py + ph - 28, toast_, 1, 210, 255, 190, 250);
    }
    draw_text(rgba, px + 12, py + ph - 14, "O:LOAD  ^:SAVE  X:CLOSE", 1, 224, 206, 158, 240);
    return true;
}

} // namespace mhf::mhfu
