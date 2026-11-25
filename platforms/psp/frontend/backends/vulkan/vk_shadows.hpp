#pragma once

// Vulkan 1.3+ Projected Orthographic Sun Shadow Map (2048x2048 D16_UNORM, depth-only).
// Rendered once per 30 Hz game frame from world-space opaque/alpha-tested 3D geometry,
// texel-snapped around the camera focus to eliminate shimmering, and sampled via
// 4-tap IGN-rotated PCF directly inside lighting_composite.comp (zero extra full-res passes).

#include "vk_context.hpp"
#include "vk_descriptors.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace mhf::psp::vk {

struct ShadowVertex {
    float x, y, z;
    float u, v;
};
static_assert(sizeof(ShadowVertex) == 20);

struct ShadowBatch {
    std::uint32_t v_offset = 0;
    std::uint32_t v_count = 0;
    std::uint32_t tex_slot = 0;
    std::uint32_t sampler_idx = 0;
    std::uint32_t alpha_bits = 1u;
    std::uint32_t textured = 0u;
    float uv_xform[4]{1.0f, 1.0f, 0.0f, 0.0f};
};

struct alignas(16) ShadowDrawParams {
    float shadow_vp[16]{};
    float uv_xform[4]{1.0f, 1.0f, 0.0f, 0.0f};
    std::uint32_t tex_alpha[4]{};
};
static_assert(sizeof(ShadowDrawParams) == 96);

class SunShadowMap {
public:
    static constexpr std::uint32_t kMapSize = 2048u;

    bool init(const Context &ctx, VkDescriptorSetLayout bindless_layout, BindlessDescriptors &desc);
    void shutdown(const Context &ctx, BindlessDescriptors &desc);

    [[nodiscard]] bool ready() const noexcept {
        return pipe_ != VK_NULL_HANDLE && depth_img_ != VK_NULL_HANDLE && depth_slot_ != 0u;
    }
    [[nodiscard]] std::uint32_t map_slot() const noexcept { return valid_ ? depth_slot_ : 0u; }
    [[nodiscard]] bool valid() const noexcept { return valid_; }
    [[nodiscard]] const float *shadow_vp() const noexcept { return shadow_vp_; }
    [[nodiscard]] float world_span() const noexcept { return world_span_; }

    void invalidate() noexcept { valid_ = false; }

    // Renders the depth-only sun shadow map for the current 30 Hz frame.
    // Must be called OUTSIDE an active VkRendering pass.
    // Leaves the shadow depth image in VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL.
    bool render(VkCommandBuffer cmd, VkDescriptorSet bindless_set,
                const float view_mat[16], const float sun_dir_world[3],
                const std::vector<ShadowVertex> &verts, const std::vector<ShadowBatch> &batches,
                VkBuffer vb_handle, VkDeviceSize vb_offset,
                void *params_cpu, VkDeviceAddress params_bda, std::size_t max_batches);

    // Computes column-major `out_shadow_from_view = shadow_vp * inverse(view_mat)` for sampling in lighting_composite.
    void compute_shadow_from_view(const float view_mat[16], float out_shadow_from_view[16]) const noexcept;

private:
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipe_ = VK_NULL_HANDLE;

    VkImage depth_img_ = VK_NULL_HANDLE;
    VmaAllocation depth_alloc_ = VK_NULL_HANDLE;
    VkImageView depth_view_ = VK_NULL_HANDLE;
    std::uint32_t depth_slot_ = 0u;
    VkImageLayout depth_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;

    float shadow_vp_[16]{};
    float world_span_ = 3200.0f;
    bool valid_ = false;
};

} // namespace mhf::psp::vk
