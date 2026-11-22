#pragma once

// OpenGL 3.3 drawing for the GE (docs/05-runtime-hle.md §6, "hardware renderer").
//
// Runs on the GE thread: it owns the OpenGL context of the game window, draws every framebuffer the game renders
// into an upscaled render target, and presents the displayed one to the window.

#include "../gpu_common.hpp"

#include <memory>

struct SDL_Window;

namespace mhf::psp {

// Creates the context on the calling thread. Returns nullptr (after logging why) when OpenGL 3.3 is unavailable.
// `scale` is the internal resolution multiplier (1 = the original 480x272). `smooth_textures` (enhancement) forces
// bilinear filtering on 3D textures. `target_fps` (enhancement) is the output rate: a whole multiple of the game's 30 fps up to 180 (rounded), with
// in-between pictures drawn for every flip; 30 shows the game as it is.
// `live` (optional) is read every draw and starts with the caller's value; without it `smooth_textures` is fixed.
std::unique_ptr<gpu::Backend> make_gl_backend(SDL_Window *window, int scale, psprecomp::GuestMemory &memory,
                                              bool smooth_textures = false, OverlayImage *overlay = nullptr,
                                              int target_fps = 30, LiveSettings *live = nullptr);

} // namespace mhf::psp
