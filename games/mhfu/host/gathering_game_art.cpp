#include "gathering_game_art.hpp"
#include "data_bin.hpp"
#include "iso9660.hpp"
#include "texture_images.hpp"
#include "texture_pack.hpp"

#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace mhf::mhfu {
namespace {
using Bytes = std::span<const std::uint8_t>;
std::uint32_t word(Bytes bytes, std::size_t offset) {
    return std::uint32_t(bytes[offset]) | (std::uint32_t(bytes[offset + 1]) << 8) |
           (std::uint32_t(bytes[offset + 2]) << 16) | (std::uint32_t(bytes[offset + 3]) << 24);
}
std::uint32_t premultiply(std::uint32_t color) {
    const unsigned alpha = color >> 24;
    std::uint32_t pixel = alpha << 24;
    for (unsigned channel = 0; channel < 3; ++channel)
        pixel |= ((((color >> (channel * 8)) & 255u) * alpha + 127) / 255) << (channel * 8);
    return pixel;
}
struct Atlas {
    Bytes pixels, palette;
};
// Native TMH images are swizzled. The two atlases we use are 256x256 CLUT8 with a 256-entry RGBA8888 palette.
// Other images in the same container are skipped by their declared size, without decoding them.
std::optional<Atlas> atlas(Bytes bytes, unsigned index) {
    if (bytes.size() < 16 || !std::equal(bytes.begin(), bytes.begin() + 8, ".TMH0.14")) return {};
    const auto count = word(bytes, 8);
    if (count > 16 || index >= count) return {};
    std::size_t offset = 16;
    for (unsigned i = 0; i <= index; ++i) {
        if (offset > bytes.size() || bytes.size() - offset < 32) return {};
        const auto size = word(bytes, offset);
        if (size < 32 || size > bytes.size() - offset) return {};
        if (i == index) {
            if (word(bytes, offset + 12) != 1 || word(bytes, offset + 16) != 65552 || word(bytes, offset + 24) != 5 ||
                word(bytes, offset + 28) != 0x01000100 || size != 66608)
                return {};
            const auto palette = offset + 32 + 65536;
            if (word(bytes, palette) != 1040 || word(bytes, palette + 8) != 3 || word(bytes, palette + 12) != 256)
                return {};
            return Atlas{bytes.subspan(offset + 32, 65536), bytes.subspan(palette + 16, 1024)};
        }
        offset += size;
    }
    return {};
}
// Match the original atlas using the selected pack's own hash rules and palette, never a pack-specific filename.
// No live guest address is needed for content/palette keys. Address-only mappings keep the original atlas.
std::optional<psp::textures::Image> replacement(const Atlas &image, const psp::textures::Pack *pack) {
    if (!pack) return {};
    psp::textures::TextureFacts facts;
    facts.width = facts.height = facts.buffer_width = 256;
    facts.format = 5;
    facts.dimension = 0x0808;
    facts.clut_format_word = 0xc500ff03;
    facts.clut = image.palette.data();
    facts.clut_last_bytes = facts.clut_max_bytes = 1024;
    psp::textures::Key key;
    std::uint32_t width, height;
    if (!pack->key_of(
            facts,
            [&](std::uint32_t address, std::uint32_t size) {
                return address == 0 && size <= image.pixels.size() ? image.pixels.data() : nullptr;
            },
            key, width, height) ||
        width != 256 || height != 256)
        return {};
    const auto match = pack->find(key);
    if (!match.entry || match.entry->keeps_original()) return {};
    const auto level = match.entry->levels.find(0);
    if (level == match.entry->levels.end() || level->second.empty()) return {};
    psp::textures::Image decoded;
    std::string error;
    if (!psp::textures::decode_image(pack->path_of(level->second), decoded, error) || decoded.width < 256 ||
        decoded.width > 8192 || decoded.width != decoded.height || decoded.width % 256 != 0 || decoded.levels.empty() ||
        decoded.levels.front().size() != std::size_t(decoded.width) * decoded.height * 4)
        return {};
    return decoded;
}
GatheringImage crop(const psp::textures::Image &image, unsigned x, unsigned y, unsigned size) {
    const unsigned scale = image.width / 256, side = size * scale;
    auto pixels = std::make_shared<std::vector<std::uint32_t>>(side * side);
    const auto &rgba = image.levels.front();
    for (unsigned row = 0; row < side; ++row) {
        for (unsigned column = 0; column < side; ++column) {
            const auto offset = ((y * scale + row) * image.width + x * scale + column) * 4;
            (*pixels)[row * side + column] = premultiply(word(rgba, offset));
        }
    }
    return {static_cast<int>(side), std::move(pixels)};
}
GatheringImage crop(const Atlas &image, unsigned x, unsigned y, unsigned size) {
    auto pixels = std::make_shared<std::vector<std::uint32_t>>(size * size);
    for (unsigned row = 0; row < size; ++row) {
        for (unsigned column = 0; column < size; ++column) {
            const unsigned u = x + column, v = y + row;
            const unsigned swizzled = (v / 8) * 2048 + (u / 16) * 128 + (v % 8) * 16 + u % 16;
            (*pixels)[row * size + column] = premultiply(word(image.palette, 4 * image.pixels[swizzled]));
        }
    }
    return {static_cast<int>(size), std::move(pixels)};
}
} // namespace

std::optional<GatheringArt> decode_game_gathering_art(Bytes hud, Bytes items, const psp::textures::Pack *pack,
                                                      unsigned *replaced) {
    if (replaced) *replaced = 0;
    const auto background = atlas(hud, 1), actions = atlas(items, 0);
    if (!background || !actions) return {};
    const auto hud_image = replacement(*background, pack), item_image = replacement(*actions, pack);
    if (replaced) *replaced = unsigned(hud_image.has_value()) + unsigned(item_image.has_value());
    const auto hud_crop = [&](unsigned x, unsigned y, unsigned size) {
        return hud_image ? crop(*hud_image, x, y, size) : crop(*background, x, y, size);
    };
    const auto item_crop = [&](unsigned x, unsigned y) {
        return item_image ? crop(*item_image, x, y, 24) : crop(*actions, x, y, 24);
    };
    return GatheringArt{hud_crop(220, 224, 32), hud_crop(0, 0, 16), item_crop(24, 48), item_crop(144, 24),
                        item_crop(48, 96), item_crop(120, 0)};
}
std::optional<GatheringArt> load_game_gathering_art(const std::filesystem::path &iso_path,
                                                    const std::filesystem::path &executable, std::uint32_t hud_file,
                                                    std::uint32_t items_file, const psp::textures::Pack *pack,
                                                    unsigned *replaced) {
    if (replaced) *replaced = 0;
    try {
        // Use the existing game-specific DATA.BIN reader, with the player's own prepared executable and ISO.
        std::ifstream file(executable, std::ios::binary | std::ios::ate);
        if (!file || file.tellg() <= 0 || file.tellg() > 16 * 1024 * 1024) return {};
        std::vector<std::uint8_t> exe(static_cast<std::size_t>(file.tellg()));
        file.seekg(0);
        if (!file.read(reinterpret_cast<char *>(exe.data()), static_cast<std::streamsize>(exe.size()))) return {};
        psp::IsoImage iso(iso_path);
        const auto entry = iso.find("/PSP_GAME/USRDIR/DATA.BIN");
        if (!entry) return {};
        const auto data = DataBin::open(
            [&](std::uint64_t offset, std::size_t size) {
                if (size > 1024 * 1024) throw std::runtime_error("icon archive read exceeds limit");
                return iso.read_file(*entry, offset, size);
            },
            entry->size, find_substitution_table_candidates(exe));
        if (!data) return {};
        for (const auto id : {hud_file, items_file}) {
            if (id >= data->files().size() || data->files()[id].size > 1024 * 1024) return {};
        }
        return decode_game_gathering_art(data->read(hud_file), data->read(items_file), pack, replaced);
    } catch (const std::exception &) {
        return {}; // Unsupported/missing input keeps the text fallback.
    }
}

std::vector<std::uint8_t> load_game_text_archive(const std::filesystem::path &iso_path,
                                                 const std::filesystem::path &executable) {
    try {
        std::ifstream file(executable, std::ios::binary | std::ios::ate);
        if (!file || file.tellg() <= 0 || file.tellg() > 16 * 1024 * 1024) return {};
        std::vector<std::uint8_t> exe(static_cast<std::size_t>(file.tellg()));
        file.seekg(0);
        if (!file.read(reinterpret_cast<char *>(exe.data()), static_cast<std::streamsize>(exe.size()))) return {};
        psp::IsoImage iso(iso_path);
        const auto entry = iso.find("/PSP_GAME/USRDIR/DATA.BIN");
        if (!entry) return {};
        const auto data = DataBin::open(
            [&](std::uint64_t offset, std::size_t size) {
                if (size > 2 * 1024 * 1024) throw std::runtime_error("text archive read exceeds limit");
                return iso.read_file(*entry, offset, size);
            },
            entry->size, find_substitution_table_candidates(exe));
        if (!data) return {};
        // Master text archive lives at file 15 on JP/FUComplete (and nearby in EU/US); validate header & size.
        for (std::uint32_t id = 15; id <= 25 && id < data->files().size(); ++id) {
            const auto sz = data->files()[id].size;
            if (sz < 65536u || sz > 2 * 1024 * 1024) continue;
            auto bytes = data->read(id);
            if (bytes.size() < 72u) continue;
            const std::uint32_t num_groups = word(bytes, 0);
            const std::uint32_t table_off = word(bytes, 4);
            if (num_groups >= 16u && num_groups <= 40u && table_off == 8u) return bytes;
        }
        return {};
    } catch (const std::exception &) {
        return {};
    }
}

std::optional<LoadoutIcons> decode_game_loadout_icons(Bytes items, const psp::textures::Pack *pack) {
    const auto actions = atlas(items, 0);
    if (!actions) return {};
    const auto item_image = replacement(*actions, pack);
    const auto pick = [&](unsigned x, unsigned y, unsigned size = 24u) -> LoadoutSprite {
        auto img = item_image ? crop(*item_image, x, y, size) : crop(*actions, x, y, size);
        return {img.size, std::move(img.pixels)};
    };
    LoadoutIcons icons{
        .pouch = pick(96, 96),
        .chest = pick(168, 0),
        .potion = pick(24, 72),
        .book = pick(120, 96),
        .bomb = pick(48, 48),
        .trap = pick(48, 24),
        .whetstone = pick(0, 24),
        .pickaxe = pick(24, 48),
        .paintball = pick(0, 72),
        .ammo = pick(72, 72),
        .weapon_blade = pick(24, 120),
        .weapon_gun = pick(120, 120),
        .weapon_classes = {{
            pick(208, 144, 16u), // 0: Great Sword
            pick(208, 192, 16u), // 1: Heavy Bowgun
            pick(144, 160, 16u), // 2: Hammer
            pick(240, 144, 16u), // 3: Lance
            pick(224, 144, 16u), // 4: Sword & Shield
            pick(160, 160, 16u), // 5: Light Bowgun
            pick(176, 160, 16u), // 6: Dual Blades
            pick(240, 192, 16u), // 7: Long Sword
            pick(224, 192, 16u), // 8: Hunting Horn
            pick(224, 208, 16u), // 9: Gunlance
            pick(192, 160, 16u), // 10: Bow
        }},
        .item_sprites = {},
        .head = pick(0, 144),
        .chest_armor = pick(24, 144),
        .arms = pick(72, 144),
        .waist = pick(48, 144),
        .legs = pick(96, 144),
    };
    for (unsigned icon_id = 0; icon_id < icons.item_sprites.size(); ++icon_id) {
        const unsigned slot = icon_id + 1u;
        icons.item_sprites[icon_id] = pick((slot % 8u) * 24u, (slot / 8u) * 24u, 24u);
    }
    return icons;
}

std::optional<LoadoutIcons> load_game_loadout_icons(const std::filesystem::path &iso_path,
                                                    const std::filesystem::path &executable, std::uint32_t items_file,
                                                    const psp::textures::Pack *pack) {
    try {
        std::ifstream file(executable, std::ios::binary | std::ios::ate);
        if (!file || file.tellg() <= 0 || file.tellg() > 16 * 1024 * 1024) return {};
        std::vector<std::uint8_t> exe(static_cast<std::size_t>(file.tellg()));
        file.seekg(0);
        if (!file.read(reinterpret_cast<char *>(exe.data()), static_cast<std::streamsize>(exe.size()))) return {};
        psp::IsoImage iso(iso_path);
        const auto entry = iso.find("/PSP_GAME/USRDIR/DATA.BIN");
        if (!entry) return {};
        const auto data = DataBin::open(
            [&](std::uint64_t offset, std::size_t size) {
                if (size > 1024 * 1024) throw std::runtime_error("icon archive read exceeds limit");
                return iso.read_file(*entry, offset, size);
            },
            entry->size, find_substitution_table_candidates(exe));
        if (!data || items_file >= data->files().size() || data->files()[items_file].size > 1024 * 1024) return {};
        return decode_game_loadout_icons(data->read(items_file), pack);
    } catch (const std::exception &) {
        return {};
    }
}
} // namespace mhf::mhfu
