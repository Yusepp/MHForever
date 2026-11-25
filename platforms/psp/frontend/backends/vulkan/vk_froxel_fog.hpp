#pragma once

// Hybrid True 3D Volumetric Fog + Ocarina-of-Time-Remake Interactive Ground Mist:
// - Pass 0 (froxel_fluid.comp): 128x128 world-snapped RGBA16F semi-Lagrangian fluid advection
//   driven by skinned actors (hunter + monsters) parting & swirling the ground mist!
// - Pass 1 (froxel_inject.comp): 160x90x64 VK_IMAGE_TYPE_3D (RGBA16F) frustum volume for upper-air
//   valley atmosphere, RT/shadow-map god-ray shafts & dynamic point lights + temporal reprojection.
// - Pass 2 (froxel_integrate.comp): Frostbite closed-form analytical Z-integration along 64 slices.
// - Set 2 exposes both u_froxel_vol (binding 0, sampler3D) and u_ground_fluid (binding 1, sampler2D)
//   to lighting_composite.comp for full-res ground-mist slab ray-marching + 1-tap 3D froxel lookup!

#include "vk_context.hpp"
#include "vk_descriptors.hpp"

#include <cstdint>

namespace mhf::psp::vk {

struct FogActor {
    float x = 0.0f;
    float foot_y = 0.0f;
    float z = 0.0f;
    float radius = 75.0f;
};

struct alignas(16) FroxelInjectParams {
    float inv_view[16]{};
    float prev_view[16]{};
    float shadow_from_view[16]{};
    float proj_info[4]{};
    float prev_proj_info[4]{};
    float sun_dir_view[4]{};
    float fog_color[4]{};
    float scatter_color[4]{};
    float world_env[4]{};
    std::uint32_t slots[4]{};
    std::uint32_t rt_and_flags[4]{};
    VkDeviceAddress lights_bda = 0;
    std::uint32_t pad_bda[2]{};
};
static_assert(sizeof(FroxelInjectParams) == 336);

struct alignas(16) FroxelIntegrateParams {
    float slice_info[4]{};
};
static_assert(sizeof(FroxelIntegrateParams) == 16);

struct alignas(16) FroxelFluidParams {
    struct ActorGpu {
        float pos_rad[4]{};
        float vel_pad[4]{};
    };
    float grid_info[4]{};   // origin_x, origin_z, world_size, dt
    float shift_and_t[4]{}; // shift_x, shift_z, elapsed_sec, reset_flag
    std::uint32_t counts[4]{};
    ActorGpu actors[8]{};
};
static_assert(sizeof(FroxelFluidParams) == 304);

class FroxelFog {
public:
    static constexpr std::uint32_t kWidth = 160u;
    static constexpr std::uint32_t kHeight = 90u;
    static constexpr std::uint32_t kDepth = 64u;
    static constexpr float kNearZ = 70.0f;
    static constexpr float kFarZ = 2200.0f;

    static constexpr std::uint32_t kFluidSize = 128u;
    static constexpr float kFluidWorldSize = 2400.0f; // 24m x 24m world-locked interactive mist grid (18.75 cm/texel)
    static constexpr std::size_t kMaxActors = 8u;

    bool init(const Context &ctx, VkDescriptorSetLayout bindless_layout);
    void shutdown(const Context &ctx);

    [[nodiscard]] bool ready() const noexcept {
        return inject_pipe_ != VK_NULL_HANDLE && integrate_pipe_ != VK_NULL_HANDLE &&
               fluid_pipe_ != VK_NULL_HANDLE && integ_vol_.view != VK_NULL_HANDLE &&
               sampler3d_ != VK_NULL_HANDLE && sampler2d_ != VK_NULL_HANDLE;
    }
    [[nodiscard]] VkDescriptorSetLayout sample_set_layout() const noexcept { return sample_set_layout_; }
    [[nodiscard]] VkDescriptorSet sample_set() const noexcept { return sample_sets_[last_fluid_idx_]; }
    [[nodiscard]] VkDescriptorSet fallback_sample_set() const noexcept { return fallback_sample_set_; }

    void reset_history() noexcept {
        history_valid_ = false;
        zone_floor_valid_ = false;
        fluid_valid_ = false;
        prev_actor_count_ = 0u;
    }

    // Ensures fallback 3D + 2D fluid textures are cleared & in SHADER_READ_ONLY_OPTIMAL even when fog is OFF.
    void ensure_fallback(VkCommandBuffer cmd);

    // Dispatches Pass 0 (128x128 world-locked fluid advection), Pass 1 (3D froxel inject), and Pass 2 (Z-integrate).
    // Writes `out_ground_info[4]` = {fluid_origin_x, ground_floor_y, fluid_origin_z, kFluidWorldSize}.
    bool dispatch(VkCommandBuffer cmd, VkDescriptorSet bindless_set,
                  const float view_mat[16], const float inv_view[16],
                  const float proj[16], const float sun_dir_view[4],
                  const float fog_color_lin[4], const float scatter_color_lin[4],
                  float elapsed_sec, float vapor_strength, std::uint64_t frame_idx,
                  std::uint32_t noise3d_slot, std::uint32_t linear_sampler,
                  std::uint32_t shadow_map_slot, std::uint32_t nearest_sampler,
                  const float *shadow_from_view, std::uint32_t rt_shadow_slot,
                  VkDeviceAddress lights_bda, std::uint32_t light_count,
                  const FogActor *actors, std::uint32_t actor_count,
                  void *inject_cpu, VkDeviceAddress inject_bda,
                  void *integ_cpu, VkDeviceAddress integ_bda,
                  void *fluid_cpu, VkDeviceAddress fluid_bda,
                  float out_ground_info[4]);

private:
    struct VolTex {
        VkImage image = VK_NULL_HANDLE;
        VmaAllocation alloc = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
        void destroy(const Context &ctx) noexcept;
    };

    bool create_vol3d(const Context &ctx, VolTex &v, std::uint32_t w, std::uint32_t h, std::uint32_t d,
                      VkImageUsageFlags usage);
    bool create_tex2d(const Context &ctx, VolTex &v, std::uint32_t w, std::uint32_t h,
                      VkImageUsageFlags usage);

    VkSampler sampler3d_ = VK_NULL_HANDLE;
    VkSampler sampler2d_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout work_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout fluid_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout sample_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkDescriptorSet inject_sets_[2]{};
    VkDescriptorSet integ_sets_[2]{};
    VkDescriptorSet fluid_sets_[2]{};
    VkDescriptorSet sample_sets_[2]{};
    VkDescriptorSet fallback_sample_set_ = VK_NULL_HANDLE;

    VkPipelineLayout inject_pipe_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout integ_pipe_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout fluid_pipe_layout_ = VK_NULL_HANDLE;
    VkPipeline inject_pipe_ = VK_NULL_HANDLE;
    VkPipeline integrate_pipe_ = VK_NULL_HANDLE;
    VkPipeline fluid_pipe_ = VK_NULL_HANDLE;

    VolTex scatter_vol_[2]{};
    VolTex integ_vol_{};
    VolTex fallback_vol_{};
    VolTex fluid_vol_[2]{};
    VolTex fallback_fluid_{};

    int ping_ = 0;
    int fluid_ping_ = 0;
    int last_fluid_idx_ = 0;
    bool history_valid_ = false;
    bool fallback_cleared_ = false;
    bool fluid_valid_ = false;
    float prev_view_[16]{};
    float prev_p00_ = 1.0f;
    float prev_p11_ = 1.76f;
    float zone_floor_y_ = 0.0f;
    bool zone_floor_valid_ = false;
    float last_cam_world_[3]{};
    int fluid_cell_origin_[2]{0, 0};
    float last_elapsed_sec_ = 0.0f;
    std::uint64_t last_fluid_frame_idx_ = ~0ull;
    FogActor prev_actors_[kMaxActors]{};
    std::uint32_t prev_actor_count_ = 0u;
};

} // namespace mhf::psp::vk
