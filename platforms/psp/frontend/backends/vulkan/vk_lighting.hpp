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
#include "vk_froxel_fog.hpp"
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

struct alignas(16) ViewPointLight {
    float pos_radius[4]{};      // xyz = view-space pos (+Y down, -Z forward), w = radius
    float color_intensity[4]{}; // rgb = linear HDR color, w = intensity
};
static_assert(sizeof(ViewPointLight) == 32);

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
    std::uint32_t phase1_slots[4]{}; // x = gbuf_normal_slot, y = exposure_slot, z = hdr_flags (bit0=hdr, bit1=exp_reset), w = ground weather (0 absent, 1..1025)
    VkDeviceAddress lights_bda{0};
    std::uint32_t light_count{0};
    std::uint32_t lights_pad{0};
    float shadow_from_view[16]{}; // column-major: PSP view (+Y up) -> shadow clip space
    std::uint32_t shadow_info[4]{}; // x = shadow_map_slot, y = nearest_sampler, z = rt_shadow_slot, w = noise3d_slot
    float inv_view[16]{};           // column-major: PSP view (+Y up) -> world space (cm)
    float env_time[4]{};            // x = elapsed_sec, y = cam_world_y, z = vapor_strength, w = has_inv_view
    float ground_fog_info[4]{};     // x = fluid_origin_x, y = ground_floor_y, z = fluid_origin_z, w = fluid_world_size
    float mist_shape[4]{};          // x = base_h_cm, y = surge_h_cm, z = base_dens, w = surge_dens
    float mist_tint[4]{};           // rgb = multiplicative mist tint, w = mist_enabled (1.0 or 0.0)
    float ground_hf_info[4]{};      // x = hf_origin_x, y = hf_origin_z, z = hf_cell_size, w = hf_grid_n (64.0 or 0.0)
};
static_assert(sizeof(LightingComputeParams) == 480);

struct alignas(16) TaaComputeParams {
    float proj_info[4]{};       // x = 1/p00, y = 1/p11, z = p10, w = p14
    float vz_and_size[4]{};     // x = vz_scale, y = vz_center, z = scene_w, w = scene_h
    float prev_from_curr0[4]{}; // row 0 of rigid prev_from_curr view transform
    float prev_from_curr1[4]{}; // row 1
    float prev_from_curr2[4]{}; // row 2
    float jitter_info[4]{};     // xy = curr_jitter_uv, zw = prev_jitter_uv
    std::uint32_t slots[4]{};   // x = curr_color_slot, y = history_slot, z = depth_slot, w = linear_sampler
    std::uint32_t flags[4]{};   // x = out_w, y = out_h, z = reset_history, w = valid_reproj
};
static_assert(sizeof(TaaComputeParams) == 128);

struct alignas(16) HiZComputeParams {
    std::uint32_t info[4]{}; // x = src_slot, y = dst_w, z = dst_h, w = 0
};
static_assert(sizeof(HiZComputeParams) == 16);

struct alignas(16) WaterSsrParams {
    float proj_info[4]{};
    float vz_and_size[4]{};
    float inv_view[16]{};
    float sun_dir_view[4]{};
    float world_up_view[4]{};
    float fog_color[4]{};
    std::uint32_t slots0[4]{};      // x = color_slot, y = depth_slot, z = noise3d_slot, w = linear_sampler
    std::uint32_t hiz_slots[4]{};   // x..w = mip0..mip3
    std::uint32_t extra_slots[4]{}; // x = rt_shadow_slot, y = gtao_slot, z = ripple_count, w = flags (bit0 = hdr)
    float ripples[24][4]{};         // x = world_x, y = world_z, z = age_sec, w = radius_cm
};
static_assert(sizeof(WaterSsrParams) == 576);

struct LightingFrameInput {
    SceneCamera fog_camera{};
    static constexpr std::size_t kMaxViewLights = 64;
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
    float psp_fog_intensity = 0.0f; // live PSP fog density at mid-range (0 = clear area, 1 = dense/surging fog)
    bool has_psp_fog = false;
    float translucent_fog_signal = 0.0f;
    std::uint64_t frame_index = 0;
    FogActor fog_actors[FroxelFog::kMaxActors]{};
    std::uint32_t fog_actor_count = 0u;
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
               comp_inplace_pipe_ != VK_NULL_HANDLE && exposure_pipe_ != VK_NULL_HANDLE &&
               fxaa_pipe_ != VK_NULL_HANDLE && taa_pipe_ != VK_NULL_HANDLE;
    }

    // Zone & material detection: resolves PPSSPP (data, clut) key once per unique `content_hash` (cached O(1)).
    MaterialClass note_texture(std::uint64_t content_hash, const textures::TextureFacts &facts,
                               const textures::MemoryReader &reader, bool is_blended = false,
                               std::uint32_t *out_data_hash = nullptr,
                               std::uint32_t *out_clut_hash = nullptr);
    void end_frame_zone_vote() noexcept;

    // Runs enabled lighting/atmosphere compute passes on `scene_color` using `kParamSlotsPerDispatch`
    // contiguous `LightingComputeParams` slots starting at `params_cpu_ptr` / `params_bda`.
    // When `hdr_input_slot != 0`, reads 3D linear HDR color from `hdr_input_slot` and optional R8_UINT
    // MaterialClass IDs from `mat_id_view`, tonemapping into `scene_color` (left in VK_IMAGE_LAYOUT_GENERAL).
    bool dispatch(VkCommandBuffer cmd, const Context &ctx, BindlessDescriptors &desc,
                  VkImage scene_color, VkImageView scene_color_view, std::uint32_t scene_color_slot,
                  std::uint32_t scene_depth_slot, std::uint32_t scene_w, std::uint32_t scene_h,
                  const LightingFrameInput &in,
                  void *params_cpu_ptr, VkDeviceAddress params_bda, bool run_fxaa = true,
                  std::uint32_t hdr_input_slot = 0, std::uint32_t gbuf_normal_slot = 0,
                  VkDeviceAddress lights_bda = 0, std::uint32_t light_count = 0,
                  std::uint32_t shadow_map_slot = 0, const float *shadow_from_view = nullptr,
                  std::uint32_t rt_shadow_slot = 0, const float *view_mat = nullptr,
                  float elapsed_sec = 0.0f, VkImageView mat_id_view = VK_NULL_HANDLE,
                  std::uint32_t soft_depth_slot = 0, bool has_water_draws = false);

    // Standalone pre-HUD FXAA 3.11 pass (used when Enhanced Lighting is OFF and AA Mode == FXAA).
    // Expects `scene_color` in VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; leaves it in VK_IMAGE_LAYOUT_GENERAL.
    bool dispatch_fxaa_only(VkCommandBuffer cmd, const Context &ctx, BindlessDescriptors &desc,
                            VkImage scene_color, VkImageView scene_color_view,
                            std::uint32_t scene_w, std::uint32_t scene_h,
                            void *params_cpu_ptr, VkDeviceAddress params_bda);

    // Pre-HUD Temporal Anti-Aliasing (TAA) resolve pass (works with or without Enhanced Lighting).
    // Expects `scene_color` and `scene_depth` in VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    // leaves `scene_color` in VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL.
    bool dispatch_taa(VkCommandBuffer cmd, const Context &ctx, BindlessDescriptors &desc,
                      VkImage scene_color, std::uint32_t scene_color_slot, std::uint32_t scene_depth_slot,
                      std::uint32_t scene_w, std::uint32_t scene_h, const LightingFrameInput &in,
                      const float prev_from_curr_3x4[12], bool valid_reproj, bool reset_history,
                      const float curr_jitter_uv[2], const float prev_jitter_uv[2],
                      void *params_cpu_ptr, VkDeviceAddress params_bda);

    void reset_taa_history() noexcept { taa_hist_valid_ = false; }
    void reset_exposure() noexcept { exposure_reset_ = true; }
    void reset_froxel_history() noexcept { froxel_.reset_history(); }
    [[nodiscard]] bool active_is_cave() const noexcept { return active_is_cave_; }
    [[nodiscard]] int active_zone() const noexcept { return active_zone_; }
    [[nodiscard]] std::uint32_t noise3d_slot() const noexcept { return noise3d_tex_.slot; }
    void fill_water_frame_params(struct WaterFrameParams &out, const LightingFrameInput &in,
                                 const float *view_mat, float vp_w, float vp_h, float elapsed_sec,
                                 std::uint32_t bg_color_slot, std::uint32_t bg_depth_slot) const noexcept;
    [[nodiscard]] const char *active_zone_name() const noexcept {
        if (live_ && active_zone_ >= 0 && static_cast<std::size_t>(active_zone_) < live_->shadow_zone_names.size()) {
            return live_->shadow_zone_names[static_cast<std::size_t>(active_zone_)].c_str();
        }
        return "unknown";
    }
    void resolve_sun_dir_world(const LightingFrameInput &in, float out[3]) const noexcept;

    // Invalidates any cached VkImageView -> VkDescriptorSet entry matching a retired view.
    void invalidate_scene_view(VkImageView view) noexcept {
        if (view == VK_NULL_HANDLE) return;
        for (auto &v : scene_set_views_) {
            if (v == view) v = VK_NULL_HANDLE;
        }
        for (auto &v : mat_set_views_) {
            if (v == view) v = VK_NULL_HANDLE;
        }
    }

    // Number of LightingComputeParams structs needed per dispatch
    // (11 lighting passes + 3 froxel passes + 2 slots for WaterSsrParams + 1 slot for 4x HiZComputeParams).
    static constexpr std::size_t kParamSlotsPerDispatch = 18;

private:
    bool ensure_targets(const Context &ctx, BindlessDescriptors &desc, std::uint32_t full_w, std::uint32_t full_h);
    bool ensure_noise3d(VkCommandBuffer cmd, const Context &ctx, BindlessDescriptors &desc);
    bool ensure_taa_targets(const Context &ctx, BindlessDescriptors &desc, std::uint32_t full_w, std::uint32_t full_h);
    VkDescriptorSet get_or_bind_scene_set(const Context &ctx, VkImageView scene_color_view);
    VkDescriptorSet get_or_bind_mat_set(const Context &ctx, VkImageView mat_id_view);
    void ensure_fallback_mat(VkCommandBuffer cmd);
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
    VkDescriptorSet exposure_set_ = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, 2> taa_hist_sets_{};
    std::array<VkDescriptorSet, 4> hiz_sets_{};
    VkDescriptorSet water_ssr_set_ = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, kSceneSetCount> scene_sets_{};
    std::array<VkImageView, kSceneSetCount> scene_set_views_{};
    std::size_t scene_set_cursor_ = 0;

    VkPipelineLayout pipe_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout comp_pipe_layout_ = VK_NULL_HANDLE;
    FroxelFog froxel_;
    VkPipeline gtao_pipe_ = VK_NULL_HANDLE;
    VkPipeline gtao_denoise_pipe_ = VK_NULL_HANDLE;
    VkPipeline vol_pipe_ = VK_NULL_HANDLE;
    VkPipeline bloom_pipe_ = VK_NULL_HANDLE;
    VkPipeline ssgi_pipe_ = VK_NULL_HANDLE;
    VkPipeline glow_blur_pipe_ = VK_NULL_HANDLE;
    VkPipeline flash_pipe_ = VK_NULL_HANDLE;
    VkPipeline comp_pipe_ = VK_NULL_HANDLE;
    VkPipeline comp_inplace_pipe_ = VK_NULL_HANDLE;
    VkPipeline exposure_pipe_ = VK_NULL_HANDLE;
    VkPipeline fxaa_pipe_ = VK_NULL_HANDLE;
    VkPipeline taa_pipe_ = VK_NULL_HANDLE;
    VkPipeline hiz_pipe_ = VK_NULL_HANDLE;
    VkPipeline water_ssr_pipe_ = VK_NULL_HANDLE;

    Target2D gtao_tex_;       // half-res RGBA8 (ao, nx, ny, contact_shadow)
    Target2D gtao_blur_tex_;  // half-res RGBA8 (denoised ao, crisp nx, ny, denoised contact_shadow)
    std::array<Target2D, 4> hiz_tex_{}; // 4-mip Min-Z pyramid (half, quarter, eighth, sixteenth R32F)
    Target2D water_ssr_tex_;  // half-res RGBA16F (water reflection rgb + hit confidence a)
    Target2D vol_tex_;        // quarter-res RGBA16F (in_scatter.rgb, transmittance)
    Target2D bloom_tex_;      // quarter-res RGBA16F (Karis-extracted emissive bloom.rgb)
    Target2D bloom_blur_tex_; // quarter-res RGBA16F (Kawase-smoothed bloom.rgb)
    Target2D ssgi_tex_;       // quarter-res RGBA16F (colored indirect bounce.rgb)
    Target2D ssgi_blur_tex_;  // quarter-res RGBA16F (Kawase-smoothed indirect bounce.rgb)
    Target2D flash_tex_;      // 4x2 RGBA16F (coarse dynamic emissive flash light-field)
    Target2D lit_tmp_tex_;    // full-res RGBA8 (composite output -> FXAA input)
    Target2D exposure_tex_;   // 1x1 R16F persistent adaptive eye exposure
    Target2D noise3d_tex_;    // 256x128 R8_UNORM (8x4 tile atlas of 32x32x32 Perlin-Worley vapor noise, 32 KB)
    VkBuffer noise3d_stage_buf_ = VK_NULL_HANDLE;
    VmaAllocation noise3d_stage_alloc_ = VK_NULL_HANDLE;
    bool exposure_reset_ = true;
    bool exposure_initialized_ = false;
    std::array<Target2D, 2> taa_hist_{}; // full-res RGBA8 ping-pong TAA history
    int taa_hist_idx_ = 0;
    bool taa_hist_valid_ = false;

    const LiveSettings *live_ = nullptr;
    // Zone & biome classification tables (populated once from LiveSettings).
    struct ZoneInfo {
        BiomeKind biome = BiomeKind::DefaultOutdoor;
        bool is_cave = false;
        float sun_height_deg = -1.0f;
        float sun_darkness = -1.0f;
        float sun_turn = 0.0f;
        GroundMistProfile mist{};
    };
    std::vector<ZoneInfo> zones_;
    std::unordered_map<std::uint32_t, int> tex_to_zone_;
    std::unordered_map<std::uint64_t, int> content_hash_to_zone_;
    // Clean-room MaterialClass lookup keyed by (data_hash, clut_hash) + optional zone_hint.
    struct MatKey {
        std::uint32_t data = 0u;
        std::uint32_t clut = 0u;
        bool operator==(const MatKey &) const = default;
    };
    struct MatKeyHash {
        std::size_t operator()(const MatKey &k) const noexcept {
            return (static_cast<std::size_t>(k.data) << 1u) ^ (static_cast<std::size_t>(k.clut) * 0x9E3779B9u);
        }
    };
    struct MatEntry {
        int zone_hint = -1;
        MaterialClass mat = MaterialClass::Unknown;
    };
    std::unordered_map<MatKey, std::vector<MatEntry>, MatKeyHash> tex_to_mat_;
    struct CachedTexSem {
        int zone = -1;
        std::uint32_t data_hash = 0u;
        std::uint32_t clut_hash = 0u;
        bool has_key = false;
    };
    std::unordered_map<std::uint64_t, CachedTexSem> content_hash_sem_;
    std::unordered_map<std::uint64_t, std::uint8_t> logged_mat_keys_;
    // Dedicated Set 3 for R8_UINT MaterialClass surface attachment (utexture2D u_mat_id).
    static constexpr std::size_t kMatSetCount = 4;
    VkDescriptorSetLayout mat_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool mat_pool_ = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, kMatSetCount> mat_sets_{};
    std::array<VkImageView, kMatSetCount> mat_set_views_{};
    std::size_t mat_set_cursor_ = 0;
    Target2D fallback_mat_tex_;
    bool fallback_mat_cleared_ = false;
    // Analytic interactive water ripple ring buffer (continuous at 120/180 Hz, 0 VRAM).
    struct RippleRing {
        float x = 0.0f;
        float z = 0.0f;
        float birth_sec = -100.0f;
        float radius = 80.0f;
    };
    struct PrevRippleActor {
        float x = 0.0f;
        float z = 0.0f;
        bool valid = false;
    };
    static constexpr std::size_t kMaxRipples = 24u;
    std::array<RippleRing, kMaxRipples> ripples_{};
    std::size_t ripple_cursor_ = 0u;
    std::array<PrevRippleActor, SceneCamera::kMaxWakeActors> prev_ripple_actors_{};
    std::uint16_t ripple_area_ = 0u;
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
