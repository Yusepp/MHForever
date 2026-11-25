#pragma once

// Frostbite / Wronski True 3D Froxel Volumetric Fog (160x90x64 VK_IMAGE_TYPE_3D RGBA16F):
// - Pass 1 (froxel_inject.comp): world-locked altitude-banded billowing 3D vapor + directional
//   self-shadowing + RT/shadow-map god-rays + dynamic point lights + temporal reprojection.
// - Pass 2 (froxel_integrate.comp): Frostbite closed-form analytical Z-integration along 64 slices.
// - Sampled via 1 hardware trilinear 3D texture fetch per full-res pixel in lighting_composite.comp.

#include "vk_context.hpp"
#include "vk_descriptors.hpp"

#include <cstdint>

namespace mhf::psp::vk {

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

class FroxelFog {
public:
    static constexpr std::uint32_t kWidth = 160u;
    static constexpr std::uint32_t kHeight = 90u;
    static constexpr std::uint32_t kDepth = 64u;
    static constexpr float kNearZ = 70.0f;
    static constexpr float kFarZ = 2200.0f;

    bool init(const Context &ctx, VkDescriptorSetLayout bindless_layout);
    void shutdown(const Context &ctx);

    [[nodiscard]] bool ready() const noexcept {
        return inject_pipe_ != VK_NULL_HANDLE && integrate_pipe_ != VK_NULL_HANDLE &&
               integ_vol_.view != VK_NULL_HANDLE && sampler3d_ != VK_NULL_HANDLE;
    }
    [[nodiscard]] VkDescriptorSetLayout sample_set_layout() const noexcept { return sample_set_layout_; }
    [[nodiscard]] VkDescriptorSet sample_set() const noexcept { return sample_set_; }
    [[nodiscard]] VkDescriptorSet fallback_sample_set() const noexcept { return fallback_sample_set_; }

    void reset_history() noexcept {
        history_valid_ = false;
        zone_floor_valid_ = false;
    }

    // Ensures the 1x1x1 fallback 3D volume is cleared & in SHADER_READ_ONLY_OPTIMAL even when Froxel Fog is OFF.
    void ensure_fallback(VkCommandBuffer cmd);

    // Dispatches Pass 1 (inject + temporal reproject) and Pass 2 (Frostbite analytical integration).
    // Requires `inject_cpu`/`inject_bda` (336B) and `integ_cpu`/`integ_bda` (16B).
    bool dispatch(VkCommandBuffer cmd, VkDescriptorSet bindless_set,
                  const float view_mat[16], const float inv_view[16],
                  const float proj[16], const float sun_dir_view[4],
                  const float fog_color_lin[4], const float scatter_color_lin[4],
                  float elapsed_sec, float vapor_strength, std::uint64_t frame_idx,
                  std::uint32_t noise3d_slot, std::uint32_t linear_sampler,
                  std::uint32_t shadow_map_slot, std::uint32_t nearest_sampler,
                  const float *shadow_from_view, std::uint32_t rt_shadow_slot,
                  VkDeviceAddress lights_bda, std::uint32_t light_count,
                  void *inject_cpu, VkDeviceAddress inject_bda,
                  void *integ_cpu, VkDeviceAddress integ_bda);

private:
    struct Vol3D {
        VkImage image = VK_NULL_HANDLE;
        VmaAllocation alloc = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
        void destroy(const Context &ctx) noexcept;
    };

    bool create_vol3d(const Context &ctx, Vol3D &v, std::uint32_t w, std::uint32_t h, std::uint32_t d,
                      VkImageUsageFlags usage);

    VkSampler sampler3d_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout work_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout sample_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkDescriptorSet inject_sets_[2]{};
    VkDescriptorSet integ_sets_[2]{};
    VkDescriptorSet sample_set_ = VK_NULL_HANDLE;
    VkDescriptorSet fallback_sample_set_ = VK_NULL_HANDLE;

    VkPipelineLayout inject_pipe_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout integ_pipe_layout_ = VK_NULL_HANDLE;
    VkPipeline inject_pipe_ = VK_NULL_HANDLE;
    VkPipeline integrate_pipe_ = VK_NULL_HANDLE;

    Vol3D scatter_vol_[2]{};
    Vol3D integ_vol_{};
    Vol3D fallback_vol_{}; // 1x1x1 (0,0,0,1) transparent 3D fallback when Froxel Fog is disabled

    int ping_ = 0;
    bool history_valid_ = false;
    bool fallback_cleared_ = false;
    float prev_view_[16]{};
    float prev_p00_ = 1.0f;
    float prev_p11_ = 1.76f;
    float zone_floor_y_ = 0.0f;
    bool zone_floor_valid_ = false;
    float last_cam_world_[3]{};
};

} // namespace mhf::psp::vk
