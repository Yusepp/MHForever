#pragma once

// Ultra-Efficient Hardware Ray-Traced Sun Shadows via VK_KHR_ray_query + Partitioned Static/Dynamic BLAS.
// Inspired by Clustered/Partitioned TLAS design:
// - Static zone terrain BLAS is built ONCE per zone (cached by vertex count & content hash, PREFER_FAST_TRACE).
// - Dynamic character BLAS (skinned hunter/monsters only) is rebuilt once per 30 Hz frame (PREFER_FAST_BUILD).
// - 2-instance TLAS traces 1 terminate-on-first-hit ray per half-res pixel (< 0.5 ms on RDNA 3.5+/RTX).
// - Zero VRAM allocated and zero GPU commands recorded when disabled or unsupported!

#include "vk_context.hpp"
#include "vk_descriptors.hpp"
#include "vk_lighting.hpp"

#include <cstdint>
#include <unordered_set>
#include <vector>

namespace mhf::psp::vk {

struct RtVertex {
    float x, y, z;
};
static_assert(sizeof(RtVertex) == 12);

struct RtStaticBatch {
    std::uint32_t v_offset = 0u;
    std::uint32_t v_count = 0u;
};

struct alignas(16) RtShadowParams {
    float inv_view[16]{};
    float proj_info[4]{};
    float vz_and_size[4]{};
    float sun_dir_world[4]{};
    std::uint32_t out_slots[4]{};
};
static_assert(sizeof(RtShadowParams) == 128);

class RayTracedShadows {
public:
    static constexpr int kRingSlots = 3; // matches VkBackend::kFramesInFlight

    bool init(const Context &ctx, VkDescriptorSetLayout bindless_layout);
    void shutdown(const Context &ctx, BindlessDescriptors &desc);

    [[nodiscard]] bool ready() const noexcept { return pipe_ != VK_NULL_HANDLE; }
    [[nodiscard]] bool valid() const noexcept { return tlas_valid_[last_slot_] && out_slot_ != 0u; }
    [[nodiscard]] std::uint32_t shadow_slot() const noexcept { return valid() ? out_slot_ : 0u; }

    void invalidate() noexcept {
        for (int i = 0; i < kRingSlots; ++i) tlas_valid_[i] = false;
    }

    void reset_zone_cache() noexcept {
        zone_chunks_.clear();
        zone_static_verts_.clear();
        ++zone_static_gen_;
        empty_static_frames_ = 0u;
        has_last_cam_ = false;
    }

    // Updates Partitioned Static + Dynamic BLAS and 2-instance TLAS for `frame_slot` (0..2, guaranteed GPU-idle).
    // Static zone chunks are accumulated persistently per zone so PSP CPU frustum culling never pops off-screen shadows!
    bool update_bvh(VkCommandBuffer cmd, const Context &ctx, int frame_slot,
                    const std::vector<RtVertex> &static_verts,
                    const std::vector<RtStaticBatch> &static_batches,
                    const std::vector<RtVertex> &dyn_verts,
                    const float view_mat[16]);

    // Traces half-res hardware ray-traced sun shadows for `frame_slot` before lighting_composite.
    bool trace(VkCommandBuffer cmd, const Context &ctx, BindlessDescriptors &desc,
               VkDescriptorSet bindless_set, int frame_slot, std::uint32_t full_w, std::uint32_t full_h,
               std::uint32_t depth_slot, const float view_mat[16], const LightingFrameInput &in,
               const float sun_dir_world[3], void *params_cpu, VkDeviceAddress params_bda);

private:
    struct GpuBuf {
        VkBuffer buf = VK_NULL_HANDLE;
        VmaAllocation alloc = VK_NULL_HANDLE;
        VkDeviceAddress addr = 0;
        void *mapped = nullptr;
        VkDeviceSize size = 0;
        void destroy(const Context &ctx) noexcept;
    };

    struct BlasSlot {
        VkAccelerationStructureKHR as = VK_NULL_HANDLE;
        GpuBuf as_buf;
        GpuBuf vtx_buf;
        VkDeviceAddress device_addr = 0;
        std::uint64_t cached_gen = 0;
        std::uint64_t last_used_seq = 0;
        std::uint32_t cached_tris = 0;
        void destroy(const Context &ctx) noexcept;
    };

    bool ensure_buf(const Context &ctx, GpuBuf &b, VkDeviceSize needed, VkBufferUsageFlags usage, bool host_visible,
                    VkDeviceSize min_alloc = 256u);
    bool build_blas(VkCommandBuffer cmd, const Context &ctx, BlasSlot &slot, GpuBuf &scratch,
                    const std::vector<RtVertex> &verts, VkBuildAccelerationStructureFlagsKHR flags,
                    VkDeviceSize min_vtx_alloc, VkDeviceSize min_as_alloc);
    bool ensure_output(const Context &ctx, BindlessDescriptors &desc, std::uint32_t hw, std::uint32_t hh);

    VkDescriptorSetLayout rt_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool rt_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet rt_sets_[kRingSlots]{};
    VkPipelineLayout pipe_layout_ = VK_NULL_HANDLE;
    VkPipeline pipe_ = VK_NULL_HANDLE;

    // Persistent per-zone static chunk cache (prevents PSP frustum culling from popping off-screen building/tree shadows).
    std::unordered_set<std::uint64_t> zone_chunks_;
    std::vector<RtVertex> zone_static_verts_;
    std::uint64_t zone_static_gen_ = 1u;
    std::uint32_t empty_static_frames_ = 0u;
    float last_cam_world_[3]{};
    bool has_last_cam_ = false;

    BlasSlot static_blas_[kRingSlots]{};
    int active_static_slot_ = -1;
    std::uint64_t use_seq_ = 10u;

    BlasSlot dyn_blas_[kRingSlots]{};
    VkAccelerationStructureKHR tlas_[kRingSlots]{};
    GpuBuf tlas_buf_[kRingSlots]{};
    GpuBuf inst_buf_[kRingSlots]{};
    GpuBuf scratch_buf_[kRingSlots]{};
    bool tlas_valid_[kRingSlots]{};
    bool rt_desc_dirty_[kRingSlots]{true, true, true};
    int last_slot_ = 0;

    VkImage out_img_ = VK_NULL_HANDLE;
    VmaAllocation out_alloc_ = VK_NULL_HANDLE;
    VkImageView out_view_ = VK_NULL_HANDLE;
    std::uint32_t out_slot_ = 0u;
    std::uint32_t out_w_ = 0u;
    std::uint32_t out_h_ = 0u;
    VkImageLayout out_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
};

} // namespace mhf::psp::vk
