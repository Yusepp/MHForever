#pragma once

// Dynamic-rendering graphics pipelines for Present, Overlay, Blur, and specialized GE Draw passes.
// Uses VK_EXT_extended_dynamic_state3 (blend enable/equations/color mask) when available, with a hashed
// pipeline cache fallback when EDS3 is absent.

#include "vk_context.hpp"

#include <cstdint>
#include <unordered_map>

namespace mhf::psp::vk {

// Matches GLSL DrawParams in ge_draw.vert / ge_draw.frag (80 bytes, scalar layout).
struct alignas(16) DrawParams {
    float uv_xform[4]{1.0f, 1.0f, 0.0f, 0.0f};
    float env_and_flags[4]{0.0f, 0.0f, 0.0f, 0.0f};
    float fog_and_zmin[4]{0.0f, 0.0f, 0.0f, 0.0f};
    float target_size[2]{480.0f, 272.0f};
    float zmax_pad[2]{65535.0f, 0.0f};
    std::uint32_t tex_idx{0};
    std::uint32_t sampler_idx{0};
    std::uint32_t tex_func_bits{0};
    std::uint32_t alpha_bits{1};
};
static_assert(sizeof(DrawParams) == 80);

struct PresentPush {
    float uv_max[2]{1.0f, 1.0f};
    std::uint32_t tex_idx{0};
    std::uint32_t sampler_idx{0};
    float alpha{1.0f};
};

struct BlurPush {
    float uv_off[2]{0.0f, 0.0f};
    float uv_scale[2]{1.0f, 1.0f};
    float dir[2]{0.0f, 0.0f};
    float dim{1.0f};
    std::uint32_t tex_idx{0};
    std::uint32_t sampler_idx{0};
};

struct DrawPush {
    VkDeviceAddress params_addr{0};
    std::uint32_t draw_idx{0};
};

// Key for GE pipeline lookup. With EDS3 enabled, blend/mask fields are zeroed so at most ~6 pipelines exist total.
struct GePipelineKey {
    std::uint8_t spec_clear{0};
    std::uint8_t spec_textured{1};
    std::uint8_t spec_alpha_test{1};
    std::uint8_t blend_enable{0};
    std::uint8_t color_op{0};
    std::uint8_t src_factor{1};
    std::uint8_t dst_factor{0};
    std::uint8_t color_mask{0xFu};

    bool operator==(const GePipelineKey &) const = default;
    [[nodiscard]] std::uint64_t hash() const noexcept {
        return static_cast<std::uint64_t>(spec_clear) | (static_cast<std::uint64_t>(spec_textured) << 8u) |
               (static_cast<std::uint64_t>(spec_alpha_test) << 16u) | (static_cast<std::uint64_t>(blend_enable) << 24u) |
               (static_cast<std::uint64_t>(color_op) << 32u) | (static_cast<std::uint64_t>(src_factor) << 40u) |
               (static_cast<std::uint64_t>(dst_factor) << 48u) | (static_cast<std::uint64_t>(color_mask) << 56u);
    }
};

struct Pipelines {
    VkPipelineLayout present_layout = VK_NULL_HANDLE;
    VkPipelineLayout blur_layout = VK_NULL_HANDLE;
    VkPipelineLayout ge_layout = VK_NULL_HANDLE;

    VkPipeline present_pipe = VK_NULL_HANDLE;
    VkPipeline overlay_pipe = VK_NULL_HANDLE;
    VkPipeline blur_pipe = VK_NULL_HANDLE;

    VkShaderModule ge_vert = VK_NULL_HANDLE;
    VkShaderModule ge_frag = VK_NULL_HANDLE;
    VkFormat color_format = VK_FORMAT_R8G8B8A8_UNORM;
    VkFormat depth_format = VK_FORMAT_D32_SFLOAT;

    std::unordered_map<std::uint64_t, VkPipeline> ge_pipes;

    bool init(const Context &ctx, VkDescriptorSetLayout bindless_layout, VkFormat swap_fmt);
    void shutdown(const Context &ctx);
    bool rebuild_swapchain_pipelines(const Context &ctx, VkFormat swap_fmt);

    [[nodiscard]] VkPipeline get_ge_pipeline(const Context &ctx, GePipelineKey key);
};

} // namespace mhf::psp::vk
