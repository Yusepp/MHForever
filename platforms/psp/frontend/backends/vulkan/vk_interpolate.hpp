#pragma once

// Next-Gen GPU Compute Frame Interpolation (place_v2.comp) + Matcher integration.
// Re-projects raw 3D world-space vertices directly into a device-local GpuVertex buffer on the GPU for every
// intermediate 60/120/180 Hz subframe — supporting dual-stream per-vertex skinned/deforming interpolation,
// UV/color/fog blending, and deterministic Sutherland-Hodgman near-plane clipping (2 output tris per input tri).

#include "vk_buffers.hpp"
#include "vk_context.hpp"
#include "frame_matching.hpp"

#include <cstdint>
#include <vector>

namespace mhf::psp::vk {

struct alignas(16) RawGpuVertex {
    float world_pos[3]{};
    std::uint32_t draw_idx{0};
    float uv[2]{};
    std::uint32_t color_rgba{0xFFFFFFFFu};
    std::uint32_t spec_rgba{0xFF000000u};
    float fog{1.0f};
    float pad[3]{};
};
static_assert(sizeof(RawGpuVertex) == 48);

struct alignas(16) DrawTransform {
    float mv_old[12]{};   // 3x4 affine rows: eye_blend * inverse(older_world)
    float mv_new[12]{};   // 3x4 affine rows: eye_blend * inverse(newer_world)
    float proj[16]{};     // 4x4 column-major projection matrix
    float vp_scale[4]{};  // x_scale, y_scale, z_scale, unused
    float vp_center[4]{}; // x_center - off_x, y_center - off_y, z_center, overlay_stretch
    std::uint32_t pair_info[4]{}; // older_raw_start, newer_raw_start, flags, float_bits(max_deform_dist)
};
static_assert(sizeof(DrawTransform) == 208);

struct PlacePush {
    VkDeviceAddress raw_verts_old{0};
    VkDeviceAddress raw_verts_new{0};
    VkDeviceAddress transforms{0};
    VkDeviceAddress out_verts{0};
    std::uint32_t vertex_count{0};
    float phase{0.0f};
};

class GpuInterpolator {
public:
    bool init(const Context &ctx);
    void shutdown(const Context &ctx);

    [[nodiscard]] bool ready() const noexcept { return pipe_ != VK_NULL_HANDLE; }

    // Dispatches place_v2.comp to transform & clip `vertex_count` raw vertices into `out_vertex_buffer()`
    // (2x output vertices per input triangle for deterministic Sutherland-Hodgman near-plane clipping),
    // inserting a Synchronization2 COMPUTE -> VERTEX_ATTRIBUTE_READ barrier.
    void dispatch_place_v2(VkCommandBuffer cmd, VkDeviceAddress raw_verts_old_bda, VkDeviceAddress raw_verts_new_bda,
                           VkDeviceAddress transforms_bda, std::uint32_t vertex_count, float phase);

    [[nodiscard]] VkBuffer out_vertex_buffer() const noexcept { return out_buf_; }
    [[nodiscard]] VkDeviceAddress out_vertex_address() const noexcept { return out_addr_; }
    [[nodiscard]] std::uint32_t max_in_vertices() const noexcept { return max_in_vertices_; }

    interpolation::Matcher &matcher() noexcept { return matcher_; }

private:
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipe_ = VK_NULL_HANDLE;
    VkBuffer out_buf_ = VK_NULL_HANDLE;
    VmaAllocation out_alloc_ = VK_NULL_HANDLE;
    VkDeviceAddress out_addr_ = 0;
    std::uint32_t max_in_vertices_ = 262144u; // 256K input verts -> 512K output verts (~18.8 MB device-local)
    interpolation::Matcher matcher_;
};

} // namespace mhf::psp::vk
