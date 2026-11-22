#pragma once

// Global bindless descriptor table (Set 0):
//   binding 0: 16384 SAMPLED_IMAGE slots (UPDATE_AFTER_BIND | PARTIALLY_BOUND), slot 0 = 1x1 white fallback
//   binding 1: Immutable VkSampler array covering all (linear, mipmap, clamp_s, clamp_t, aniso) combinations

#include "vk_context.hpp"

#include <cstdint>
#include <deque>
#include <vector>

namespace mhf::psp::vk {

class BindlessDescriptors {
public:
    static constexpr std::uint32_t kMaxTextures = 16384u;
    // Sampler index encoding: bit 0 = linear, bit 1 = mipmap, bit 2 = clamp_s, bit 3 = clamp_t, bits 4..5 = aniso tier (0=1x, 1=4x, 2=8x, 3=16x)
    static constexpr std::uint32_t kSamplerCount = 64u;

    bool init(const Context &ctx);
    void shutdown(const Context &ctx);

    [[nodiscard]] VkDescriptorSetLayout layout() const noexcept { return layout_; }
    [[nodiscard]] VkDescriptorSet set() const noexcept { return set_; }

    // Allocates a bindless texture slot and writes `view` into it (safe to call mid-frame via UPDATE_AFTER_BIND).
    [[nodiscard]] std::uint32_t allocate_texture(const Context &ctx, VkImageView view);
    // Updates an already-allocated slot with a new view.
    void update_texture(const Context &ctx, std::uint32_t slot, VkImageView view);
    // Schedules a slot for reuse once the graphics timeline reaches `retire_value`.
    void free_texture(std::uint32_t slot, std::uint64_t retire_value);
    void collect_retired(std::uint64_t completed_timeline);

    [[nodiscard]] static std::uint32_t sampler_index(bool linear, bool mips, bool clamp_s, bool clamp_t, int anisotropy) noexcept;

private:
    VkDescriptorSetLayout layout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkDescriptorSet set_ = VK_NULL_HANDLE;
    std::vector<VkSampler> samplers_;

    VkImage fallback_image_ = VK_NULL_HANDLE;
    VmaAllocation fallback_alloc_ = VK_NULL_HANDLE;
    VkImageView fallback_view_ = VK_NULL_HANDLE;

    std::uint32_t next_slot_ = 1u; // slot 0 reserved for fallback 1x1 white
    std::vector<std::uint32_t> free_slots_;
    struct PendingFree {
        std::uint32_t slot;
        std::uint64_t retire_value;
    };
    std::deque<PendingFree> pending_frees_;
};

} // namespace mhf::psp::vk
