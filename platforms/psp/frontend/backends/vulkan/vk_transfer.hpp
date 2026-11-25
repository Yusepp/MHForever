#pragma once

// Phase 3: Async texture-pack streaming + GPU mipmap generation + VMA VRAM budget telemetry.
// Integrates with textures::Pack and textures::Loader so KTX2/PNG packs upload without stalling the GE thread.

#include "vk_context.hpp"
#include "vk_descriptors.hpp"
#include "texture_images.hpp"
#include "texture_pack.hpp"

#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace mhf::psp::vk {

struct PackPbrSlots {
    std::uint32_t normal_slot = 0;      // _n tangent-space normal map
    std::uint32_t rough_metal_slot = 0; // _r roughness (G) / metallic (B) map
    std::uint32_t emissive_slot = 0;    // _e emissive glow map
};

struct PackCompanionGpu {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation alloc = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    std::uint32_t bindless_slot = 0;
    std::size_t vram_bytes = 0;
    std::string file_path; // empty if absent on disk
    bool probed = false;
};

struct PackGpuImage {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation alloc = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    std::uint32_t bindless_slot = 0;
    std::uint32_t mip_levels = 1;
    std::size_t vram_bytes = 0;
    std::uint64_t last_used = 0;
    PackCompanionGpu pbr[3]{}; // 0 = _n, 1 = _r, 2 = _e
};

class PackStreamer {
public:
    bool init(const Context &ctx, const std::string &pack_folder, const std::string &game_id,
              const std::string &builtin_folder, bool generate_mips);
    void shutdown(const Context &ctx, BindlessDescriptors &desc);

    [[nodiscard]] bool active() const noexcept { return pack_ != nullptr || builtin_pack_ != nullptr; }

    // Non-blocking lookup: returns bindless slot > 0 when the replacement image is resident on GPU, or 0 while
    // loading/missing (so the caller draws the original PSP texture until ready).
    // When a decoded image is ready to upload, `begin_upload_cmd()` is called to obtain the active frame's
    // recording command buffer (outside any render pass), avoiding any synchronous vkWaitForFences stall.
    [[nodiscard]] std::uint32_t lookup_or_queue(const Context &ctx, BindlessDescriptors &desc,
                                                const textures::TextureFacts &facts,
                                                const textures::MemoryReader &reader,
                                                const std::function<VkCommandBuffer()> &begin_upload_cmd,
                                                std::uint64_t retire_timeline,
                                                std::uint64_t frame_num, bool &has_mips,
                                                textures::Filter &filter_out,
                                                bool &pending_out,
                                                PackPbrSlots *pbr_out = nullptr);

    // Updates LRU timestamp for a pack texture hit via an external slot cache, and optionally
    // polls/returns its companion PBR slots (_n, _r, _e) once resident.
    void touch(std::uint32_t bindless_slot, std::uint64_t frame_num,
               PackPbrSlots *pbr_out = nullptr) noexcept;

    // Frees staging buffers and evicted pack images whose GPU submissions have completed.
    void collect_staging(const Context &ctx, std::uint64_t completed_timeline);

    // Checks VMA heap budgets (VK_EXT_memory_budget when available) and evicts LRU pack images under pressure.
    // Returns true if any pack textures were evicted (so callers can invalidate cached slot lookups).
    bool trim_vram(const Context &ctx, BindlessDescriptors &desc, std::uint64_t retire_timeline,
                   std::uint64_t current_frame);

private:
    struct RetiredStaging {
        VkBuffer buffer = VK_NULL_HANDLE;
        VmaAllocation alloc = VK_NULL_HANDLE;
        std::uint64_t retire_value = 0;
    };
    struct RetiredImage {
        VkImage image = VK_NULL_HANDLE;
        VmaAllocation alloc = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        std::uint64_t retire_value = 0;
    };

    bool upload_image_async(const Context &ctx, BindlessDescriptors &desc, VkCommandBuffer cmd,
                            std::uint64_t retire_timeline, const textures::Image &img, PackGpuImage &out);
    void probe_and_stream_pbr(const Context &ctx, BindlessDescriptors &desc,
                              const std::string &base_id, const std::filesystem::path &base_file,
                              const std::function<VkCommandBuffer()> &begin_upload_cmd,
                              std::uint64_t retire_timeline, PackGpuImage &entry);

    std::unique_ptr<textures::Pack> pack_;
    std::unique_ptr<textures::Pack> builtin_pack_;
    std::unique_ptr<textures::Loader> loader_;
    bool generate_mips_ = false;

    std::uint64_t last_upload_frame_ = UINT64_MAX;
    std::uint32_t uploads_this_frame_ = 0;
    std::deque<RetiredStaging> retired_staging_;
    std::deque<RetiredImage> retired_images_;

    std::unordered_map<std::string, PackGpuImage> uploaded_;
    std::unordered_map<std::uint32_t, PackGpuImage *> slot_map_;
    std::size_t total_vram_bytes_ = 0;
};

} // namespace mhf::psp::vk
