#pragma once

// Hybrid True 3D Volumetric Fog + True 2D GPU Navier-Stokes Interactive Ground Mist:
// - 256x256 world-snapped GPU Navier-Stokes solver (32m x 32m, 12.5 cm/texel, stepped at 30 Hz):
//     1. fluid_advect.comp   : Selle MacCormack advection + Bridson curl-noise + actor wakes
//     2. fluid_vort_div.comp : Fedkiw vorticity confinement + velocity divergence
//     3. fluid_jacobi.comp   : Harris GPU Gems Jacobi pressure Poisson solver (12 iterations)
//     4. fluid_project.comp  : Stam Helmholtz-Hodge incompressible projection (div u == 0)
// - 160x90x64 3D Froxel Volume (froxel_inject.comp + froxel_integrate.comp) for upper-air valley
//   atmosphere, RT/shadow-map god-ray shafts & dynamic point lights.
// - Set 2 exposes u_froxel_vol (binding 0, sampler3D) and u_ground_fluid (binding 1, sampler2D)
//   to lighting_composite.comp for 8-plane stacked static world-space ground-mist rendering!

#include "vk_context.hpp"
#include "vk_descriptors.hpp"
#include "semantic_scene.hpp"

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
    float wind_info[4]{};   // wind_x, wind_z, 0, 0
    std::uint32_t counts[4]{};
    ActorGpu actors[8]{};
};
static_assert(sizeof(FroxelFluidParams) == 320);

class FroxelFog {
public:
    static constexpr std::uint32_t kWidth = 160u;
    static constexpr std::uint32_t kHeight = 90u;
    static constexpr std::uint32_t kDepth = 64u;
    static constexpr float kNearZ = 70.0f;
    static constexpr float kFarZ = 2200.0f;

    static constexpr std::uint32_t kFluidSize = 256u;
    static constexpr float kFluidWorldSize = 3200.0f; // 32m x 32m world-locked Navier-Stokes grid (12.5 cm/texel)
    static constexpr int kJacobiIters = 12;
    static constexpr std::size_t kMaxActors = 8u;

    bool init(const Context &ctx, VkDescriptorSetLayout bindless_layout);
    void shutdown(const Context &ctx);

    [[nodiscard]] bool ready() const noexcept {
        return inject_pipe_ != VK_NULL_HANDLE && integrate_pipe_ != VK_NULL_HANDLE &&
               advect_pipe_ != VK_NULL_HANDLE && vort_div_pipe_ != VK_NULL_HANDLE &&
               jacobi_pipe_ != VK_NULL_HANDLE && project_pipe_ != VK_NULL_HANDLE &&
               integ_vol_.view != VK_NULL_HANDLE && sampler3d_ != VK_NULL_HANDLE &&
               sampler2d_ != VK_NULL_HANDLE;
    }
    [[nodiscard]] VkDescriptorSetLayout sample_set_layout() const noexcept { return sample_set_layout_; }
    [[nodiscard]] VkDescriptorSet sample_set() const noexcept { return sample_sets_[last_fluid_idx_]; }
    [[nodiscard]] VkDescriptorSet fallback_sample_set() const noexcept { return fallback_sample_set_; }

    void reset_history() noexcept {
        history_valid_ = false;
        zone_floor_valid_ = false;
        zone_floor_frames_ = 0u;
        fluid_valid_ = false;
        prev_actor_count_ = 0u;
        uploaded_hf_rev_ = 0u;
        uploaded_hf_area_ = 0u;
    }

    // Ensures fallback 3D + 2D fluid + 2D heightfield textures are cleared & in SHADER_READ_ONLY_OPTIMAL.
    void ensure_fallback(VkCommandBuffer cmd);

    // Dispatches 256x256 Navier-Stokes solver (at 30 Hz), optional 64x64 heightfield upload,
    // 3D froxel inject, and Frostbite Z-integration.
    // Writes `out_ground_info[4]` = {fluid_origin_x, ground_floor_y, fluid_origin_z, kFluidWorldSize}
    // and `out_hf_info[4]` = {hf_origin_x, hf_origin_z, hf_cell_size, hf_grid_n_or_0}.
    bool dispatch(VkCommandBuffer cmd, VkDescriptorSet bindless_set,
                  const float view_mat[16], const float inv_view[16],
                  const float proj[16], const float sun_dir_view[4],
                  const float fog_color_lin[4], const float scatter_color_lin[4],
                  float elapsed_sec, float vapor_strength, std::uint64_t frame_idx,
                  bool ground_mist_only, std::uint16_t ground_area,
                  float wind_x, float wind_z,
                  const GroundHeightField *heightfield,
                  std::uint32_t noise3d_slot, std::uint32_t linear_sampler,
                  std::uint32_t shadow_map_slot, std::uint32_t nearest_sampler,
                  const float *shadow_from_view, std::uint32_t rt_shadow_slot,
                  VkDeviceAddress lights_bda, std::uint32_t light_count,
                  const FogActor *actors, std::uint32_t actor_count,
                  void *inject_cpu, VkDeviceAddress inject_bda,
                  void *integ_cpu, VkDeviceAddress integ_bda,
                  void *fluid_cpu, VkDeviceAddress fluid_bda,
                  float out_ground_info[4], float out_hf_info[4]);

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
                      VkFormat fmt, VkImageUsageFlags usage);

    VkSampler sampler3d_ = VK_NULL_HANDLE;
    VkSampler sampler2d_ = VK_NULL_HANDLE;
    VkSampler sampler_nearest_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout work_set_layout_ = VK_NULL_HANDLE;     // 1 storage + 1 sampler
    VkDescriptorSetLayout vort_div_layout_ = VK_NULL_HANDLE;     // 2 storage + 1 sampler
    VkDescriptorSetLayout project_layout_ = VK_NULL_HANDLE;      // 1 storage + 2 samplers
    VkDescriptorSetLayout sample_set_layout_ = VK_NULL_HANDLE;   // 3 samplers (3D froxel + 2D fluid + 2D heightfield)
    VkDescriptorPool pool_ = VK_NULL_HANDLE;

    VkDescriptorSet inject_sets_[2]{};
    VkDescriptorSet integ_sets_[2]{};
    VkDescriptorSet advect_sets_[2]{};
    VkDescriptorSet vort_div_sets_[2]{};
    VkDescriptorSet jacobi_sets_[2]{};
    VkDescriptorSet project_sets_[2]{};
    VkDescriptorSet sample_sets_[2]{};
    VkDescriptorSet fallback_sample_set_ = VK_NULL_HANDLE;

    VkPipelineLayout inject_pipe_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout integ_pipe_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout advect_pipe_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout vort_div_pipe_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout jacobi_pipe_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout project_pipe_layout_ = VK_NULL_HANDLE;

    VkPipeline inject_pipe_ = VK_NULL_HANDLE;
    VkPipeline integrate_pipe_ = VK_NULL_HANDLE;
    VkPipeline advect_pipe_ = VK_NULL_HANDLE;
    VkPipeline vort_div_pipe_ = VK_NULL_HANDLE;
    VkPipeline jacobi_pipe_ = VK_NULL_HANDLE;
    VkPipeline project_pipe_ = VK_NULL_HANDLE;

    VolTex scatter_vol_[2]{};
    VolTex integ_vol_{};
    VolTex fallback_vol_{};
    VolTex fluid_state_[2]{}; // 256x256 RGBA16F: RG=vel, B=smoke density, A=|vorticity|
    VolTex fluid_press_[2]{}; // 256x256 RG16F:   R=pressure, G=divergence
    VolTex fallback_fluid_{};
    VolTex heightfield_tex_{}; // 64x64 R32F world-locked terrain collision heightfield
    VolTex fallback_hf_{};
    VkBuffer hf_stage_buf_ = VK_NULL_HANDLE;
    VmaAllocation hf_stage_alloc_ = VK_NULL_HANDLE;
    void *hf_stage_mapped_ = nullptr;
    std::uint32_t uploaded_hf_rev_ = 0u;
    std::uint16_t uploaded_hf_area_ = 0u;

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
    std::uint16_t ground_area_ = 0;
    std::uint32_t zone_floor_frames_ = 0u;
    float last_cam_world_[3]{};
    int fluid_cell_origin_[2]{0, 0};
    float last_elapsed_sec_ = 0.0f;
    std::uint64_t last_fluid_frame_idx_ = ~0ull;
    FogActor prev_actors_[kMaxActors]{};
    std::uint32_t prev_actor_count_ = 0u;
};

} // namespace mhf::psp::vk
