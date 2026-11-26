#pragma once

// Connects a PSP runtime to the desktop host: shows the displayed framebuffer, feeds controller input and plays the
// mixed audio. Used by the game executables and the homebrew runner when a window is wanted.

#include "mhf/psp/runtime.hpp"
#include "overlay/hud_bar.hpp"
#include "overlay/song_panel.hpp"
#include "semantic_scene.hpp"

// host/keymap.hpp and host/padmap.hpp are not included here on purpose: the PSP runtime and the homebrew runners include this
// header without the SDL host's include directories. The keyboard and controller bindings travel as templates instead.
#include <functional>
#include <cstdint>
#include <memory>
#include <array>
#include <utility>
#include <vector>
#include <optional>
#include <string>
#include <string_view>

namespace mhf::psp {

enum class GraphicsBackend { OpenGL, Vulkan, Software };

[[nodiscard]] inline std::optional<GraphicsBackend> parse_graphics_backend(std::string_view name) noexcept {
    if (name == "opengl" || name == "gl") return GraphicsBackend::OpenGL;
    if (name == "vulkan" || name == "vk") return GraphicsBackend::Vulkan;
    if (name == "software" || name == "sw") return GraphicsBackend::Software;
    return std::nullopt;
}

class Frontend {
public:
    Frontend();
    ~Frontend();

    // Chooses the rendering backend before open() (default: OpenGL when gpu=true, Software when gpu=false).
    void set_graphics_backend(GraphicsBackend backend);

    // Opens the window (hidden = no visible window, for tests). Returns false when no display is available.
    // With `gpu`, the GE draws through OpenGL at `render_scale` times the PSP resolution (software rendering is the
    // fallback when OpenGL 3.3 is missing). `fullscreen` starts in borderless fullscreen; `smooth_textures` is the
    // bilinear texture enhancement and `target_fps` the output rate: multiples of the game's 30 fps up to 180, made with in-between pictures.
    bool open(const char *title, bool hidden = false, int scale = 2, bool gpu = true, int render_scale = 4,
              bool fullscreen = false, bool smooth_textures = false, int target_fps = 30);

    // The same, with the keyboard and controller bindings the game was given (host::KeyBindings, host::PadBindings). A
    // template, so this header never has to know those types; the frontend instantiates it for the game.
    template <typename KeyBindings, typename PadBindings>
    bool open(const char *title, bool hidden, int scale, bool gpu, int render_scale, bool fullscreen,
              bool smooth_textures, int target_fps, const KeyBindings &keys, const PadBindings &pad) {
        const bool shown = open(title, hidden, scale, gpu, render_scale, fullscreen, smooth_textures, target_fps);
        set_bindings(keys, pad);
        return shown;
    }

    // Where the bindings changed in the in-game menu are written ("keymap=...\npadmap=...\n"), for the launcher to keep.
    // Empty (the default): changes last for the session.
    void set_controls_file(const std::string &path);
    // Friends in the Online tab (docs/14-rooms-and-friends.md section 4.3): the launcher keeps `friends` current while the
    // game runs; what the player does with them (invite, dismiss an invite) goes to `actions`.
    void set_friends_files(const std::string &friends, const std::string &actions);
    // A texture pack (frontend/texture_pack.hpp): its folder, and the release being played ("ULES01213"), which
    // chooses the pack's ini; `mipmaps` makes mip levels for its images that ship none, and `anisotropy` (1: off, 2, 4,
    // 8 or 16) filters them (gl_backend.hpp, LiveSettings). Before attach(). The Video tab of the in-game menu then
    // turns the pack on and off.
    void set_texture_pack(const std::string &folder, const std::string &game_id, bool mipmaps = false, int anisotropy = 16);
    // What a game knows about its textures, for the projected shadows (LiveSettings). Before attach().
    void set_shadow_textures(std::vector<std::uint64_t> air, std::vector<std::uint64_t> solid, std::vector<std::uint64_t> cave);
    void set_shadow_cave_zones(std::vector<std::string> zones);
    void set_shadow_haze_textures(std::vector<std::uint64_t> textures); // the shadows are worked out before these
    void set_shadow_zone_suns(std::vector<std::pair<std::string, std::array<float, 3>>> suns); // zone: height in degrees, darkness (< 0: 0.4), turn in degrees
    void set_shadow_zones(std::vector<std::string> names, std::vector<std::pair<std::uint32_t, int>> textures);
    void set_shadow_zone_biomes(std::vector<std::pair<std::uint8_t, bool>> biomes); // (biome_id, is_cave) aligned with shadow_zone_names
    void set_shadow_zone_mist_profiles(std::vector<GroundMistProfile> profiles);  // aligned with shadow_zone_names
    // The quest HUD's size in percent, 50..100 (gl_backend.hpp, LiveSettings). The Video tab of the in-game menu
    // changes it while playing.
    void set_hud(int size);
    // Widescreen (gl_backend.hpp, LiveSettings): the 3D fills a window wider than the PSP's screen, 2D keeps its shape.
    // Before attach(); the Video tab turns it on and off while playing.
    void set_widescreen(bool on);
    // The shape the game should draw its 3D at (width over height), 0 while widescreen does not apply. Read by the
    // game program's camera hooks.
    [[nodiscard]] float game_aspect() const;
    // The projected shadows as the launcher keeps them (gl_backend.hpp, LiveSettings): on, whole map, quality 0..2,
    // darkness 0..100. The Video tab changes them while playing, and the controls file keeps them for the launcher.
    void set_shadows(bool on, bool whole_map, int quality, int darkness);
    // Vulkan compute lighting & atmosphere (GTAO, volumetric fog/sun shafts, emissive bloom, rim light, etc.).
    void set_enhanced_lighting(bool on);
    void set_vulkan_effects_mask(std::uint32_t mask);
    void set_aa_mode(int mode);
    // Publishes game-semantic world-space dynamic lights (e.g., elemental hit flashes, monster auras)
    // for the Vulkan renderer; safe to call from the host poll thread.
    void set_scene_camera(const SceneCamera &camera);
    void set_semantic_frame(const SemanticSceneFrame &frame);
    // Turns the hunting horn's songs on (an enhancement, off unless called) at the place the player chose; the Game tab
    // of the in-game menu changes the place or lets the player drag them anywhere, and the controls file keeps the new
    // spot for the launcher.
    void set_song_spot(overlay::SongSpot spot);
    // The songs can be shown at all (the game maps them): the Game tab then turns them on and off, and the controls file
    // keeps the choice (`horn-songs`). Without a call, set_song_spot turns them on with no switch.
    void offer_songs(bool on, overlay::SongSpot spot);

    // An option of the game the Game tab changes while playing (an enhancement whose hook reads it every time). The
    // controls file keeps its value for the launcher as "option.<setting>=true|false" (a switch) or "=<number>" (a
    // slider), where <setting> is the launcher's name for it.
    struct GameOption {
        std::string setting; // "fastSupplyBox"
        std::string label, detail;
        bool slider = false; // a switch (value 0 or 1) or a slider (min..max by step, in percent)
        int value = 0, min = 0, max = 1, step = 1;
        std::string shown_with; // the setting of a switch that must be on for this row to show ("analogCamera")
        std::function<void(int)> changed; // runs on the game's thread when the player changes it
        std::string category; // optional section heading inside the QoL tab
    };
    void add_game_option(GameOption option);
    // An interactive action or choice row in the QoL tab (e.g. Item Pouch Loadouts in Village/Hall).
    struct GameAction {
        std::string label, detail;
        std::string shown_with; // optional GameOption::setting that must be on
        std::string category;
        std::vector<std::string> choices; // non-empty: Choice row; empty: Button row
        int value = 0;
        std::function<bool()> enabled;     // optional predicate (default: always enabled)
        std::function<std::string(int)> triggered; // runs on game thread; optional status note returned
    };
    void add_game_action(GameAction action);
    // The hunting horn's songs (overlay/song_panel.hpp), from the `extra` poll: shown at their place while the game draws
    // its quest HUD and no menu, system dialog or notice is up; std::nullopt (no horn, not in a quest) hides them. Only
    // the OpenGL renderer finds the quest HUD, so the software renderer never shows them.
    void set_song_panel(std::optional<overlay::SongPanel> panel);
    // A game-supplied bar over the quest HUD, composed with the other quest overlays.
    void set_hud_bar(std::optional<overlay::HudBar> bar);
    // True once the worker has published the replacement over the quest HUD.
    [[nodiscard]] bool hud_bar_visible() const;
    // A short game-provided action prompt; hidden with the quest HUD and while menus are open.
    void set_game_hint(std::string hint, std::vector<overlay::HintSprite> sprites = {});
    [[nodiscard]] bool game_input_blocked() const;
    // Suppresses guest PSP controller input (e.g. while an in-game QoL modal panel is open) while keeping
    // raw host buttons readable via raw_buttons().
    void set_game_input_suppressed(bool suppressed);
    // Masks off `clear_mask` and forces on `force_mask` in the guest PSP button state (when not fully suppressed).
    void set_game_button_overrides(std::uint32_t clear_mask, std::uint32_t force_mask);
    // Enables physical keyboard text capture for an active in-game overlay (e.g. loadout rename) when no dialog is open.
    struct GameTextEvent {
        enum class Kind { Text, Backspace, Enter, Escape };
        Kind kind{Kind::Text};
        std::string text;
    };
    void set_game_text_input(bool active);
    [[nodiscard]] std::vector<GameTextEvent> take_game_key_events();
    [[nodiscard]] std::uint32_t raw_buttons() const;
    // Native pixel size of the 16:9 overlay canvas (matches window/internal resolution, capped at 1920x1088).
    [[nodiscard]] std::pair<int, int> overlay_size() const;
    // A game's own transparent premultiplied RGBA image (default 480x272, or native HD width x height).
    // A null image hides it. Reuse the same shared image until its pixels change.
    void set_game_overlay(std::shared_ptr<const std::vector<std::uint32_t>> image, int width = 480, int height = 272);

    // Mouse look: while `wanted` says so and the game has the input (no menu, system dialog or song placing), the mouse
    // is held in the window and its motion reaches the game's hooks (ControllerState::mouse_x and mouse_y). Asked on the
    // game's thread at every poll.
    void set_mouse_look(std::function<bool()> wanted);

    // Installs the runtime's host hook. `extra` runs on every poll too (frame dumps, time limits...).
    void attach(Runtime &rt, std::function<void()> extra = {});

    struct Impl;

private:
    // Which physical key and which controller button stand for which pad button; defined in the frontend, which does know
    // host::KeyBindings and host::PadBindings.
    template <typename KeyBindings, typename PadBindings>
    void set_bindings(const KeyBindings &keys, const PadBindings &pad);

    std::unique_ptr<Impl> impl_;
};

} // namespace mhf::psp
