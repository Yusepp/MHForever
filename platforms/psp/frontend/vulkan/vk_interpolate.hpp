#pragma once

// Phase 5: GPU Compute-Accelerated Frame Interpolation (place_v2.comp) + Matcher integration.
// Re-projects raw 3D world-space vertices directly into a device-local GpuVertex buffer on the GPU for every
// intermediate 60/120/180 Hz subframe — zero CPU vertex re-projection per subframe!

#include "vk_buffers.hpp"
#include "vk_context.hpp"
#include "../frame_matching.hpp"

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
    float mvp[16]{};
    float vp_scale[4]{};
    float vp_center[4]{};
};
static_assert(sizeof(DrawTransform) == 96);

struct PlacePush {
    VkDeviceAddress raw_verts{0};
    VkDeviceAddress transforms{0};
    VkDeviceAddress out_verts{0};
    std::uint32_t vertex_count{0};
};

class GpuInterpolator {
public:
    bool init(const Context &ctx);
    void shutdown(const Context &ctx);

    [[nodiscard]] bool ready() const noexcept { return pipe_ != VK_NULL_HANDLE; }

    // Dispatches place_v2.comp to transform `vertex_count` raw vertices via `transforms_bda` into `out_vertex_buffer()`,
    // inserting a Synchronization2 COMPUTE -> VERTEX_ATTRIBUTE_READ barrier so graphics can draw directly from it.
    void dispatch_place_v2(VkCommandBuffer cmd, VkDeviceAddress raw_verts_bda, VkDeviceAddress transforms_bda,
                           std::uint32_t vertex_count);

    [[nodiscard]] VkBuffer out_vertex_buffer() const noexcept { return out_buf_; }
    [[nodiscard]] VkDeviceAddress out_vertex_address() const noexcept { return out_addr_; }

    interpolation::Matcher &matcher() noexcept { return matcher_; }

private:
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipe_ = VK_NULL_HANDLE;
    VkBuffer out_buf_ = VK_NULL_HANDLE;
    VmaAllocation out_alloc_ = VK_NULL_HANDLE;
    VkDeviceAddress out_addr_ = 0;
    std::uint32_t max_vertices_ = 262144u; // 256K vertices (~9.4 MB device-local)
    interpolation::Matcher matcher_;
};

} // namespace mhf::psp::vk
