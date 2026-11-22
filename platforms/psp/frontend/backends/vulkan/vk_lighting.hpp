#pragma once

// Next-Gen Vulkan 1.3+ Lighting & Atmosphere Pipeline (branch feat/lightning-improv):
// - Half-res Multi-Bounce Horizon GTAO + discontinuity-aware normal reconstruction (gtao.comp)
// - Quarter-res Zone-Aware Volumetric Sun Shafts & Atmospheric Height Fog (volumetric_fog.comp)
// - Quarter-res Selective Emissive Bloom (bloom_extract.comp)
// - Unified Full-Res Composite: multi-bounce sky-tinted AO, per-pixel sun key & Fresnel rim,
//   volumetric fog/shafts, emissive bloom, and soft filmic highlight shoulder (lighting_composite.comp)
//
// Runs strictly after 3D draws and BEFORE 2D HUD draws (both on 30 Hz frames and 120/180 Hz subframes),
// so HUD/UI/menus are never affected and the 4x HUD-free killcam captures the enhanced lighting too!

#include "vk_context.hpp"
#include "vk_descriptors.hpp"
#include "../gpu_common.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace mhf::psp::vk {

enum class BiomeKind : std::uint8_t {
    DefaultOutdoor = 0,
    Snowy,
    Desert,
    JungleForest,
    Swamp,
    Volcano,
    CaveIndoor,
};

struct alignas(16) LightingComputeParams {
    float proj_info[4]{};     // x = 1/proj00, y = 1/proj11, z = proj10, w = proj14
    float vz_and_size[4]{};   // x = vz_scale, y = vz_center, z = full_w, w = full_h
    float sun_dir_view[4]{};  // xyz = normalized sun dir in view space, w = is_cave (0 or 1)
    float sun_screen[4]{};    // xy = sun UV in [0,1], z = sun_facing [0,1], w = frame_jitter
    float shadow_tint[4]{};   // rgb = biome/sky shadow tint, a = ao_strength
    float fog_color[4]{};     // rgb = biome/PSP fog color, a = fog_density
    float scatter_color[4]{}; // rgb = volumetric sun scatter color, a = scatter_strength
    float bloom_rim[4]{};     // x = bloom_threshold, y = bloom_strength, z = rim_strength, w = key_strength
    std::uint32_t tex_slots[4]{}; // x = color_slot, y = depth_slot, z = gtao_slot, w = vol_slot
    std::uint32_t aux_slots[4]{}; // x = bloom_slot, y = linear_sampler, z = out_w, w = out_h
};
static_assert(sizeof(LightingComputeParams) == 160);

struct LightingFrameInput {
    float proj[16]{};         // active 3D perspective projection matrix (column-major)
    float vz_scale = 32767.5f;
    float vz_center = 32767.5f;
    float light0_dir[3]{0.35f, 0.78f, 0.52f}; // live PSP GE directional light 0
    bool has_light0 = false;
    float psp_fog_rgb[3]{0.72f, 0.78f, 0.86f};
    bool has_psp_fog = false;
    std::uint64_t frame_index = 0;
};

class LightingPipeline {
public:
    bool init(const Context &ctx, VkDescriptorSetLayout bindless_layout, const LiveSettings *live);
    void shutdown(const Context &ctx, BindlessDescriptors &desc);

    [[nodiscard]] bool ready() const noexcept {
        return gtao_pipe_ != VK_NULL_HANDLE && vol_pipe_ != VK_NULL_HANDLE &&
               bloom_pipe_ != VK_NULL_HANDLE && comp_pipe_ != VK_NULL_HANDLE;
    }

    // Zone detection: call with the 32-bit texture hash during 3D draws (zero-cost lookup).
    void note_texture_hash(std::uint64_t tex_hash) noexcept;
    void end_frame_zone_vote() noexcept;

    [[nodiscard]] BiomeKind active_biome() const noexcept { return active_biome_; }
    [[nodiscard]] bool active_is_cave() const noexcept { return active_is_cave_; }

    // Runs GTAO (1/2 res) -> Volumetric Fog/Shafts (1/4 res) -> Emissive Bloom (1/4 res) ->
    // Unified Full-Res Composite in-place on `scene_color`.
    // `params_bda` must hold 4 contiguous LightingComputeParams slots (or 1 reused — we pass 3 BDA offsets
    // from the caller's mapped param ring so each pass has its exact `out_w/out_h`!).
    void dispatch(VkCommandBuffer cmd, const Context &ctx, const BindlessDescriptors &desc,
                  VkImage scene_color, VkImageView scene_color_view, std::uint32_t scene_color_slot,
                  std::uint32_t scene_depth_slot, std::uint32_t scene_w, std::uint32_t scene_h,
                  const LightingFrameInput &in,
                  void *params_cpu_ptr, VkDeviceAddress params_bda);

    // Number of LightingComputeParams structs needed per dispatch (gtao, vol, bloom, composite).
    static constexpr std::size_t kParamSlotsPerDispatch = 4;

private:
    bool ensure_targets(const Context &ctx, BindlessDescriptors &desc, std::uint32_t full_w, std::uint32_t full_h);
    void fill_biome_params(LightingComputeParams &p, const LightingFrameInput &in) const noexcept;

    struct Target2D {
        VkImage image = VK_NULL_HANDLE;
        VmaAllocation alloc = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        std::uint32_t slot = 0;
        std::uint32_t w = 0, h = 0;
        void destroy(const Context &ctx, BindlessDescriptors &desc);
    };

    VkDescriptorSetLayout storage_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool storage_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet gtao_set_ = VK_NULL_HANDLE;
    VkDescriptorSet vol_set_ = VK_NULL_HANDLE;
    VkDescriptorSet bloom_set_ = VK_NULL_HANDLE;
    VkDescriptorSet scene_set_ = VK_NULL_HANDLE;

    VkPipelineLayout pipe_layout_ = VK_NULL_HANDLE;
    VkPipeline gtao_pipe_ = VK_NULL_HANDLE;
    VkPipeline vol_pipe_ = VK_NULL_HANDLE;
    VkPipeline bloom_pipe_ = VK_NULL_HANDLE;
    VkPipeline comp_pipe_ = VK_NULL_HANDLE;

    Target2D gtao_tex_;  // half-res RGBA8 (ao, nx, ny, valid)
    Target2D vol_tex_;   // quarter-res RGBA16F (in_scatter.rgb, transmittance)
    Target2D bloom_tex_; // quarter-res RGBA16F (emissive bloom.rgb)

    // Zone & biome classification tables (populated once from LiveSettings).
    struct ZoneInfo {
        BiomeKind biome = BiomeKind::DefaultOutdoor;
        bool is_cave = false;
        float sun_height_deg = -1.0f;
        float sun_turn = 0.0f;
    };
    std::vector<ZoneInfo> zones_;
    std::unordered_map<std::uint32_t, int> tex_to_zone_;
    std::vector<std::uint32_t> zone_votes_;
    int active_zone_ = -1;
    BiomeKind active_biome_ = BiomeKind::DefaultOutdoor;
    bool active_is_cave_ = false;
};

} // namespace mhf::psp::vk
