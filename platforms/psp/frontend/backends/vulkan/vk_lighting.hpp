#pragma once

// Vulkan 1.3+ Compute Lighting & Atmosphere Pipeline:
// - Half-res multi-bounce horizon GTAO + two-scale contact/projected sun shadows + bilateral denoise
// - Quarter-res zone-aware volumetric height fog & sun/doorway shafts (volumetric_fog.comp)
// - Quarter-res 13-tap Karis emissive bloom, SSGI-lite indirect bounce, Kawase blur & 4x2 dynamic flash
// - Full-res unified composite (AO, hemisphere bounce, specular/rim, fog, bloom, vignette, debanding)
// - Full-res pre-HUD FXAA 3.11 edge anti-aliasing (fxaa_prehud.comp)
//
// Runs strictly after 3D draws and BEFORE 2D HUD draws (both on 30 Hz frames and 120/180 Hz subframes),
// so HUD/UI/menus stay crisp and the 4x HUD-free killcam captures the enhanced lighting too.

#include "vk_context.hpp"
#include "vk_descriptors.hpp"
#include "../gpu_common.hpp"
#include "texture_pack.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

namespace mhf::psp::vk {

enum class BiomeKind : std::uint8_t {
    DefaultOutdoor = 0,
    PokkeVillage,
    SnowyPeak,
    IceCave,
    CozyIndoor,
    DarkCave,
    Desert,
    DesertSandstorm,
    ForestHills,
    GreatForest,
    Jungle,
    Swamp,
    Volcano,
    Tower,
    FortressSiege,
    Arena,
};

struct alignas(16) LightingComputeParams {
    float proj_info[4]{};         // x = 1/proj00, y = 1/proj11, z = proj10, w = proj14
    float vz_and_size[4]{};       // x = vz_scale, y = vz_center, z = full_w, w = full_h
    float sun_dir_view[4]{};      // xyz = normalized sun dir in view space, w = is_cave (0 or 1)
    float sun_screen[4]{};        // xy = sun UV in [0,1], z = sun_facing [0,1], w = frame_jitter
    float shadow_tint[4]{};       // rgb = biome/sky shadow tint, a = ao_strength
    float fog_color[4]{};         // rgb = biome/PSP fog color, a = fog_density
    float scatter_color[4]{};     // rgb = volumetric sun scatter color, a = scatter_strength
    float bloom_rim[4]{};         // x = bloom_threshold, y = bloom_strength, z = rim_strength, w = key_strength
    float world_up_view[4]{};     // xyz = normalized world-up in view space, w = ssgi_strength
    float ground_bounce[4]{};     // rgb = biome ground/magma bounce tint, a = hemi_strength
    float spec_contact[4]{};      // x = spec_strength, y = spec_gloss, z = contact_shadow_str, w = vibrance
    std::uint32_t tex_slots[4]{}; // x = color_slot, y = depth_slot, z = gtao_slot, w = vol_slot
    std::uint32_t aux_slots[4]{}; // x = bloom_slot, y = linear_sampler, z = out_w, w = out_h
    std::uint32_t ext_slots[4]{}; // x = ssgi_slot, y = gtao_raw/flash_slot, z = biome_id, w = flags (bit0=cave, bit1=vig, bit2=fxaa)
};
static_assert(sizeof(LightingComputeParams) == 224);

struct LightingFrameInput {
    float proj[16]{};         // active 3D perspective projection matrix (column-major)
    float view_rot[9]{1.0f, 0.0f, 0.0f,
                      0.0f, 1.0f, 0.0f,
                      0.0f, 0.0f, 1.0f}; // row-major 3x3 world -> view rotation
    bool has_view_rot = false;
    float vz_scale = 32767.5f;
    float vz_center = 32767.5f;
    float light0_dir[3]{0.35f, 0.78f, 0.52f}; // live PSP GE directional light 0 (world space)
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
        return gtao_pipe_ != VK_NULL_HANDLE && gtao_denoise_pipe_ != VK_NULL_HANDLE &&
               vol_pipe_ != VK_NULL_HANDLE && bloom_pipe_ != VK_NULL_HANDLE &&
               ssgi_pipe_ != VK_NULL_HANDLE && glow_blur_pipe_ != VK_NULL_HANDLE &&
               flash_pipe_ != VK_NULL_HANDLE && comp_pipe_ != VK_NULL_HANDLE &&
               comp_inplace_pipe_ != VK_NULL_HANDLE && fxaa_pipe_ != VK_NULL_HANDLE;
    }

    // Zone detection: resolves PPSSPP data hash once per unique content hash `content_hash` (cached O(1)).
    void note_texture(std::uint64_t content_hash, const textures::TextureFacts &facts,
                      const textures::MemoryReader &reader);
    void end_frame_zone_vote() noexcept;

    // Runs enabled lighting/atmosphere compute passes on `scene_color` using `kParamSlotsPerDispatch`
    // contiguous `LightingComputeParams` slots starting at `params_cpu_ptr` / `params_bda`.
    void dispatch(VkCommandBuffer cmd, const Context &ctx, BindlessDescriptors &desc,
                  VkImage scene_color, VkImageView scene_color_view, std::uint32_t scene_color_slot,
                  std::uint32_t scene_depth_slot, std::uint32_t scene_w, std::uint32_t scene_h,
                  const LightingFrameInput &in,
                  void *params_cpu_ptr, VkDeviceAddress params_bda);

    // Number of LightingComputeParams structs needed per dispatch
    // (gtao, vol, bloom, ssgi, gtao_denoise, bloom_blur, ssgi_blur, flash, composite, fxaa).
    static constexpr std::size_t kParamSlotsPerDispatch = 10;

private:
    bool ensure_targets(const Context &ctx, BindlessDescriptors &desc, std::uint32_t full_w, std::uint32_t full_h);
    void fill_biome_params(LightingComputeParams &p, const LightingFrameInput &in) noexcept;

    struct Target2D {
        VkImage image = VK_NULL_HANDLE;
        VmaAllocation alloc = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        std::uint32_t slot = 0;
        std::uint32_t w = 0, h = 0;
        void destroy(const Context &ctx, BindlessDescriptors &desc);
    };

    static constexpr std::size_t kSceneSetCount = 16;

    VkDescriptorSetLayout storage_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool storage_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet gtao_set_ = VK_NULL_HANDLE;
    VkDescriptorSet gtao_blur_set_ = VK_NULL_HANDLE;
    VkDescriptorSet vol_set_ = VK_NULL_HANDLE;
    VkDescriptorSet bloom_set_ = VK_NULL_HANDLE;
    VkDescriptorSet bloom_blur_set_ = VK_NULL_HANDLE;
    VkDescriptorSet ssgi_set_ = VK_NULL_HANDLE;
    VkDescriptorSet ssgi_blur_set_ = VK_NULL_HANDLE;
    VkDescriptorSet flash_set_ = VK_NULL_HANDLE;
    VkDescriptorSet lit_tmp_set_ = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, kSceneSetCount> scene_sets_{};
    std::array<VkImageView, kSceneSetCount> scene_set_views_{};
    std::size_t scene_set_cursor_ = 0;

    VkPipelineLayout pipe_layout_ = VK_NULL_HANDLE;
    VkPipeline gtao_pipe_ = VK_NULL_HANDLE;
    VkPipeline gtao_denoise_pipe_ = VK_NULL_HANDLE;
    VkPipeline vol_pipe_ = VK_NULL_HANDLE;
    VkPipeline bloom_pipe_ = VK_NULL_HANDLE;
    VkPipeline ssgi_pipe_ = VK_NULL_HANDLE;
    VkPipeline glow_blur_pipe_ = VK_NULL_HANDLE;
    VkPipeline flash_pipe_ = VK_NULL_HANDLE;
    VkPipeline comp_pipe_ = VK_NULL_HANDLE;
    VkPipeline comp_inplace_pipe_ = VK_NULL_HANDLE;
    VkPipeline fxaa_pipe_ = VK_NULL_HANDLE;

    Target2D gtao_tex_;       // half-res RGBA8 (ao, nx, ny, contact_shadow)
    Target2D gtao_blur_tex_;  // half-res RGBA8 (denoised ao, crisp nx, ny, denoised contact_shadow)
    Target2D vol_tex_;        // quarter-res RGBA16F (in_scatter.rgb, transmittance)
    Target2D bloom_tex_;      // quarter-res RGBA16F (Karis-extracted emissive bloom.rgb)
    Target2D bloom_blur_tex_; // quarter-res RGBA16F (Kawase-smoothed bloom.rgb)
    Target2D ssgi_tex_;       // quarter-res RGBA16F (colored indirect bounce.rgb)
    Target2D ssgi_blur_tex_;  // quarter-res RGBA16F (Kawase-smoothed indirect bounce.rgb)
    Target2D flash_tex_;      // 4x2 RGBA16F (coarse dynamic emissive flash light-field)
    Target2D lit_tmp_tex_;    // full-res RGBA8 (composite output -> FXAA input)

    const LiveSettings *live_ = nullptr;
    // Zone & biome classification tables (populated once from LiveSettings).
    struct ZoneInfo {
        BiomeKind biome = BiomeKind::DefaultOutdoor;
        bool is_cave = false;
        float sun_height_deg = -1.0f;
        float sun_darkness = -1.0f;
        float sun_turn = 0.0f;
    };
    std::vector<ZoneInfo> zones_;
    std::unordered_map<std::uint32_t, int> tex_to_zone_;
    std::unordered_map<std::uint64_t, int> content_hash_to_zone_;
    std::unique_ptr<textures::Pack> zone_keys_;
    std::vector<std::uint32_t> zone_votes_;
    int active_zone_ = -1;
    int pending_zone_ = -1;
    std::uint32_t pending_zone_streak_ = 0;
    BiomeKind active_biome_ = BiomeKind::DefaultOutdoor;
    bool active_is_cave_ = false;

    // Smoothly interpolated mood state (prevents abrupt lighting pops on outdoor <-> cave transitions).
    struct SmoothedMood {
        bool valid = false;
        std::uint64_t last_frame = UINT64_MAX;
        float shadow_tint[4]{};
        float fog_color[4]{};
        float scatter_color[4]{};
        float bloom_rim[4]{};
        float ground_bounce[4]{};
        float spec_contact[4]{};
        float ssgi_str = 0.0f;
    } mood_{};
};

} // namespace mhf::psp::vk
