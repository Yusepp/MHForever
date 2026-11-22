#pragma once

// Modern Vulkan 1.3+ drawing backend for the GE (see VULKAN.md).
//
// Runs on the GE thread (with a decoupled presentation thread in Phase 5), owns the Vulkan instance/device/swapchain
// for the game window, and renders PSP framebuffers into upscaled VkImages using dynamic rendering, bindless textures,
// Buffer Device Address, and VMA.

#include "gpu_common.hpp"

#include <memory>

struct SDL_Window;

namespace mhf::psp {

// Creates the Vulkan 1.3+ backend on the calling thread. Returns nullptr (after logging why) when Vulkan 1.3 is
// unavailable on the host system, allowing the frontend to fall back cleanly.
std::unique_ptr<gpu::Backend> make_vk_backend(SDL_Window *window, int scale, psprecomp::GuestMemory &memory,
                                              bool smooth_textures = false, OverlayImage *overlay = nullptr,
                                              int target_fps = 30, LiveSettings *live = nullptr);

} // namespace mhf::psp
