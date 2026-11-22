#include "vk_backend.hpp"

#include <cstdio>

namespace mhf::psp {

std::unique_ptr<gpu::Backend> make_vk_backend(SDL_Window * /*window*/, int /*scale*/, psprecomp::GuestMemory & /*memory*/,
                                              bool /*smooth_textures*/, OverlayImage * /*overlay*/, int /*target_fps*/,
                                              LiveSettings * /*live*/) {
    std::fprintf(stderr, "[vk] Vulkan backend scaffolded (Phase 0); runtime implementation arrives in Phase 1\n");
    return nullptr;
}

} // namespace mhf::psp
