#pragma once

// Phase 4: Compute-accelerated projected shadows (shadow_map graphics pass + shadow_factor.comp + LDS shadow_apply.comp).

#include "vk_context.hpp"
#include "vk_descriptors.hpp"

#include <cstdint>

namespace mhf::psp::vk {

struct alignas(16) ShadowComputeParams {
    float proj[4]{};
    float proj_z_vz[4]{};
    float vp[4]{};
    float center_pixel[4]{};
    float right_dark[4]{};
    float up_bias[4]{};
    float dir_drop[4]{};
    float half_soft[4]{};
    float player_box[4]{};
    float fog_down[4]{};
    float rc0_pad[4]{};
    float rc1_pad[4]{};
    float rc2_pad[4]{};
    float t_pad[4]{};
    std::uint32_t tex_indices[4]{};  // depth, map, kind, player_map
    std::uint32_t size_sampler[4]{}; // out_w, out_h, nearest_sampler, linear_sampler
};

struct ShadowApplyPush {
    std::uint32_t factor_tex{0};
    std::uint32_t depth_tex{0};
    std::uint32_t width{0};
    std::uint32_t height{0};
    std::int32_t step_px{1};
    std::int32_t down{1};
};

class ShadowCompute {
public:
    bool init(const Context &ctx, VkDescriptorSetLayout bindless_layout);
    void shutdown(const Context &ctx, BindlessDescriptors &desc);

    bool ensure_factor_image(const Context &ctx, BindlessDescriptors &desc, std::uint32_t w, std::uint32_t h);

    // Runs shadow_factor.comp into the internal factor image, then shadow_apply.comp (with 10x10 LDS tile cache)
    // to multiply the scene color image in-place.
    void dispatch_factor_and_apply(VkCommandBuffer cmd, const Context &ctx, const BindlessDescriptors &desc,
                                   VkDeviceAddress params_bda, VkImage scene_color, VkImageView scene_color_view,
                                   std::uint32_t scene_depth_slot, std::uint32_t scene_w, std::uint32_t scene_h,
                                   int down);

    [[nodiscard]] bool ready() const noexcept { return factor_pipe_ != VK_NULL_HANDLE && apply_pipe_ != VK_NULL_HANDLE; }

private:
    VkDescriptorSetLayout storage_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool storage_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet factor_storage_set_ = VK_NULL_HANDLE;
    VkDescriptorSet scene_storage_set_ = VK_NULL_HANDLE;

    VkPipelineLayout factor_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout apply_layout_ = VK_NULL_HANDLE;
    VkPipeline factor_pipe_ = VK_NULL_HANDLE;
    VkPipeline apply_pipe_ = VK_NULL_HANDLE;

    VkImage factor_image_ = VK_NULL_HANDLE;
    VmaAllocation factor_alloc_ = VK_NULL_HANDLE;
    VkImageView factor_view_ = VK_NULL_HANDLE;
    std::uint32_t factor_slot_ = 0;
    std::uint32_t factor_w_ = 0, factor_h_ = 0;
};

} // namespace mhf::psp::vk
