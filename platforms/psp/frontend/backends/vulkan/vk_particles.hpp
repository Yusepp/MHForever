#pragma once

// 120/180 Hz GPU Compute Particle Engine & VORTEXEL-Lite Living Environments.
// Simulates up to 4096 persistent world-space particles on the GPU (snow, volcanic embers/ash,
// forest pollen, swamp fireflies, desert sand streaks, plus semantic/VFX campfire/fireball/hit sparks)
// and renders them depth-tested into the 3D HDR target strictly before post-processing & 2D HUD.

#include "vk_context.hpp"
#include "vk_lighting.hpp"

#include <array>
#include <cstdint>

namespace mhf::psp::vk {

struct alignas(16) GpuParticle {
    float pos_life[4]{};
    float vel_size[4]{};
    float color_kind[4]{};
    float aux[4]{};
};
static_assert(sizeof(GpuParticle) == 64);

struct alignas(16) ParticleEmitter {
    float pos_radius[4]{};
    float color_rate[4]{};
};
static_assert(sizeof(ParticleEmitter) == 32);

struct alignas(16) ParticleSimParams {
    float view_proj[16]{};
    float cam_pos_dt[4]{};
    float wind_time[4]{};
    float vz_and_size[4]{};
    float biome_cfg[4]{};
    float ambient_col_a[4]{};
    float ambient_col_b[4]{};
    std::uint32_t counts_slots[4]{};
    std::uint32_t flags_seed[4]{};
    ParticleEmitter emitters[8]{};
    VkDeviceAddress particles_bda{0};
    std::uint64_t pad_{0};
};
static_assert(sizeof(ParticleSimParams) == 464);

struct alignas(16) ParticleDrawParams {
    float view[16]{};
    float proj[16]{};
    float cam_right[4]{};
    float cam_up[4]{};
    float screen_info[4]{};
    std::uint32_t slots[4]{};
    VkDeviceAddress particles_bda{0};
    std::uint64_t pad_{0};
};
static_assert(sizeof(ParticleDrawParams) == 208);

struct ParticleFrameInput {
    float view[16]{};        // column-major world -> PSP view matrix
    float proj[16]{};        // column-major PSP view -> Vulkan clip matrix
    float vz_scale = 32767.5f;
    float vz_center = 32767.5f;
    std::uint32_t depth_w = 0;
    std::uint32_t depth_h = 0;
    std::uint32_t depth_slot = 0;
    std::uint32_t soft_depth_slot = 0;
    BiomeKind biome = BiomeKind::DefaultOutdoor;
    bool is_cave = false;
    float dt = 1.0f / 60.0f;
    float elapsed_time = 0.0f;
    std::uint32_t frame_seed = 1;
    std::array<ParticleEmitter, 8> emitters{};
    std::uint32_t emitter_count = 0;
};

class ParticleSystem {
public:
    static constexpr std::uint32_t kMaxParticles = 4096u;
    static constexpr std::uint32_t kAmbientCount = 3072u;

    bool init(const Context &ctx, VkDescriptorSetLayout bindless_layout);
    void shutdown(const Context &ctx);

    [[nodiscard]] bool ready() const noexcept {
        return sim_pipe_ != VK_NULL_HANDLE && draw_pipe_hdr_ != VK_NULL_HANDLE &&
               particle_buf_ != VK_NULL_HANDLE;
    }

    void reset() noexcept { need_reset_ = true; }

    // Dispatches GPU compute simulation (must be called OUTSIDE an active VkRendering pass).
    // Uses `sim_params_cpu` / `sim_params_bda` (sizeof(ParticleSimParams)).
    void simulate(VkCommandBuffer cmd, VkDescriptorSet bindless_set,
                  const ParticleFrameInput &in,
                  void *sim_params_cpu, VkDeviceAddress sim_params_bda);

    // Draws all live particles (must be called INSIDE the active 3D VkRendering pass).
    // Uses `draw_params_cpu` / `draw_params_bda` (sizeof(ParticleDrawParams)).
    void draw(VkCommandBuffer cmd, VkDescriptorSet bindless_set,
              const ParticleFrameInput &in, bool hdr_target, bool msaa_4x,
              void *draw_params_cpu, VkDeviceAddress draw_params_bda);

private:
    VkPipelineLayout sim_layout_ = VK_NULL_HANDLE;
    VkPipeline sim_pipe_ = VK_NULL_HANDLE;

    VkPipelineLayout draw_layout_ = VK_NULL_HANDLE;
    VkPipeline draw_pipe_hdr_ = VK_NULL_HANDLE;
    VkPipeline draw_pipe_hdr_msaa_ = VK_NULL_HANDLE;
    VkPipeline draw_pipe_ldr_ = VK_NULL_HANDLE;
    VkPipeline draw_pipe_ldr_msaa_ = VK_NULL_HANDLE;

    VkBuffer particle_buf_ = VK_NULL_HANDLE;
    VmaAllocation particle_alloc_ = VK_NULL_HANDLE;
    VkDeviceAddress particle_bda_ = 0;

    BiomeKind last_biome_ = BiomeKind::DefaultOutdoor;
    bool last_cave_ = false;
    bool need_reset_ = true;
};

} // namespace mhf::psp::vk
