// Modern Vulkan 1.3+ drawing backend for the GE (see VULKAN.md).
//
// - Vulkan 1.3 dynamic rendering, Synchronization2, timeline semaphores, VMA, volk via SDL3.
// - Global bindless texture array (16384 UPDATE_AFTER_BIND slots) + 64 immutable samplers.
// - Persistently mapped per-frame ring buffers for GpuVertex, BDA DrawParams, and texture staging.
// - Specialized SPIR-V GE pipelines with Extended Dynamic State 1/2/3 + on-disk VkPipelineCache.
// - Upscaled render target pool, RTT + self-sampling copies, block transfers, overlays, compute
//   lighting/VFX, GPU frame interpolation, and HUD-free internal-res killcam capture.

#include "vk_backend.hpp"
#include "vk_buffers.hpp"
#include "vk_context.hpp"
#include "vk_descriptors.hpp"
#include "vk_interpolate.hpp"
#include "vk_lighting.hpp"
#include "vk_pipelines.hpp"
#include "vk_raytracing.hpp"
#include "vk_shadows.hpp"
#include "vk_swapchain.hpp"
#include "vk_transfer.hpp"

#include "ge.hpp"

#include <SDL3/SDL.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <unordered_map>
#include <vector>

namespace mhf::psp {

namespace {

using namespace gpu_common;

constexpr int kFramesInFlight = 2;
constexpr std::size_t kVertexRingBytes = 32u * 1024u * 1024u;
constexpr std::size_t kParamsRingBytes = 8u * 1024u * 1024u;
constexpr std::size_t kStagingRingBytes = 32u * 1024u * 1024u;
constexpr std::size_t kMaxCachedTextures = 8192u;

VkBlendFactor psp_blend_factor(std::uint32_t f, bool is_src, std::uint32_t fix, bool &uses_constant) noexcept {
    switch (f) {
    case 0: return is_src ? VK_BLEND_FACTOR_DST_COLOR : VK_BLEND_FACTOR_SRC_COLOR;
    case 1: return is_src ? VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR : VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
    case 2: return VK_BLEND_FACTOR_SRC_ALPHA;
    case 3: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    case 4: return VK_BLEND_FACTOR_DST_ALPHA;
    case 5: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
    case 6: return VK_BLEND_FACTOR_SRC_ALPHA;
    case 7: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    case 8: return VK_BLEND_FACTOR_DST_ALPHA;
    case 9: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
    case 10:
        if ((fix & 0xFFFFFFu) == 0u) return VK_BLEND_FACTOR_ZERO;
        if ((fix & 0xFFFFFFu) == 0xFFFFFFu) return VK_BLEND_FACTOR_ONE;
        uses_constant = true;
        return VK_BLEND_FACTOR_CONSTANT_COLOR;
    default: return VK_BLEND_FACTOR_ONE;
    }
}

struct VkImageResource {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation alloc = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    std::uint32_t width = 0, height = 0, mip_levels = 1;
    std::uint32_t bindless_slot = 0;

    bool create(const vk::Context &ctx, vk::BindlessDescriptors *desc, std::uint32_t w, std::uint32_t h,
                VkFormat fmt, VkImageUsageFlags usage, VkImageAspectFlags asp, std::uint32_t mips = 1,
                VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT) {
        width = w;
        height = h;
        format = fmt;
        aspect = asp;
        mip_levels = mips;
        layout = VK_IMAGE_LAYOUT_UNDEFINED;

        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = fmt;
        ici.extent = {w, h, 1};
        ici.mipLevels = mips;
        ici.arrayLayers = 1;
        ici.samples = samples;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = usage;

        VmaAllocationCreateInfo vaci{};
        vaci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        if (vmaCreateImage(ctx.allocator, &ici, &vaci, &image, &alloc, nullptr) != VK_SUCCESS) return false;

        VkImageViewCreateInfo ivci{};
        ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ivci.image = image;
        ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ivci.format = fmt;
        ivci.subresourceRange = {asp, 0, mips, 0, 1};
        if (vkCreateImageView(ctx.device, &ivci, nullptr, &view) != VK_SUCCESS) {
            vmaDestroyImage(ctx.allocator, image, alloc);
            image = VK_NULL_HANDLE;
            alloc = VK_NULL_HANDLE;
            return false;
        }
        if (desc && (usage & VK_IMAGE_USAGE_SAMPLED_BIT)) {
            bindless_slot = desc->allocate_texture(ctx, view);
        }
        return true;
    }

    void destroy_immediate(const vk::Context &ctx, vk::BindlessDescriptors *desc) {
        if (desc && bindless_slot != 0) {
            desc->free_texture(bindless_slot, 0);
            bindless_slot = 0;
        }
        if (view != VK_NULL_HANDLE) {
            vkDestroyImageView(ctx.device, view, nullptr);
            view = VK_NULL_HANDLE;
        }
        if (image != VK_NULL_HANDLE) {
            vmaDestroyImage(ctx.allocator, image, alloc);
            image = VK_NULL_HANDLE;
            alloc = VK_NULL_HANDLE;
        }
        layout = VK_IMAGE_LAYOUT_UNDEFINED;
    }
};

struct RetiredImage {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation alloc = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    std::uint64_t retire_value = 0;
};

struct RenderTarget {
    std::uint32_t address = 0;
    std::uint32_t stride = 512;
    std::uint32_t width = 512;
    std::uint32_t height = 272;
    std::uint32_t format = 3;
    std::uint64_t last_draw = 0;
    std::uint64_t last_vram_copy = 0;
    VkImageResource color;
    VkImageResource depth;

    [[nodiscard]] std::uint32_t bytes_per_pixel() const noexcept { return format == 3u ? 4u : 2u; }
};

struct CachedTexture {
    VkImageResource image;
    std::uint64_t last_used = 0;
};

struct DrawSetup {
    DrawKey key{};
    std::uint32_t tex_slot = 0;
    vk::PackPbrSlots pbr_slots{};
    float uv_xform[4]{1.0f, 1.0f, 0.0f, 0.0f};
    bool through = false;
    bool linear_override = false;
    bool texture_mips = false;
    float tex_width = 1.0f, tex_height = 1.0f;
    std::uint32_t target = 0;
    bool self_sampled = false;
    bool is_killcam = false;
    bool is_skinned = false;
    bool is_additive_vfx = false;
};

struct FrameSlice {
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkSemaphore image_available = VK_NULL_HANDLE;
    std::uint64_t timeline_value = 0;
    vk::MappedRingBuffer vertices;
    vk::MappedRingBuffer params;
    vk::MappedRingBuffer staging;
};

class VkBackend final : public gpu::Backend {
public:
    VkBackend(SDL_Window *window, int scale, psprecomp::GuestMemory &memory, bool smooth_textures,
              OverlayImage *overlay, int target_fps, LiveSettings *live)
        : window_(window), scale_(std::clamp(scale, 1, 10)), memory_(memory), overlay_(overlay),
          target_fps_(target_fps), live_(live ? live : &own_live_) {
        if (!live) own_live_.smooth_textures = smooth_textures;
    }

    ~VkBackend() override { shutdown(); }

    bool init();

    void triangle(const gpu::Ge &ge, const gpu::Vertex &a, const gpu::Vertex &b, const gpu::Vertex &c) override;
    void triangle_unclipped(const gpu::Ge &ge, const gpu::Vertex &a, const gpu::Vertex &b, const gpu::Vertex &c) override;
    void draw_started(const gpu::DrawInfo &draw) override;
    void sprite(const gpu::Ge &ge, const gpu::Vertex &a, const gpu::Vertex &b) override;
    void block_transfer(std::uint32_t src, std::uint32_t src_stride, std::uint32_t sx, std::uint32_t sy,
                        std::uint32_t dst, std::uint32_t dst_stride, std::uint32_t dx, std::uint32_t dy,
                        std::uint32_t width, std::uint32_t height, std::uint32_t bytes_per_pixel) override;
    void flush() override;
    void present(std::uint32_t address, std::uint32_t stride, std::uint32_t pixel_format) override;

    [[nodiscard]] std::optional<std::chrono::steady_clock::time_point> next_wakeup() const override;
    void tick() override;

private:
    void shutdown();
    void retire_image(VkImageResource &res);
    void collect_retired(std::uint64_t completed);

    void ensure_recording();
    void end_render_pass();
    void begin_target_pass(RenderTarget &rt, bool use_msaa = false, bool use_hdr = false);
    void transition_resource(VkImageResource &res, VkImageLayout new_layout,
                             VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access);
    [[nodiscard]] int effective_aa_mode() const noexcept {
        int m = live_->aa_mode.load(std::memory_order_relaxed);
        if (m == LiveSettings::kAaMsaa4x && !ctx_.caps.msaa_4x) m = LiveSettings::kAaFxaa;
        return std::clamp(m, static_cast<int>(LiveSettings::kAaOff), static_cast<int>(LiveSettings::kAaTaa));
    }
    [[nodiscard]] bool want_enhanced_3d_hdr() const noexcept {
        if (!lighting_.ready() || !live_->enhanced_lighting.load(std::memory_order_relaxed)) return false;
        constexpr std::uint32_t kNonFxaaMask = LiveSettings::kVkEffectComputeMask & ~LiveSettings::kVkEffectFxaa;
        return (live_->vulkan_effects_mask.load(std::memory_order_relaxed) & kNonFxaaMask) != 0u;
    }
    bool ensure_msaa_targets(std::uint32_t pw, std::uint32_t ph);
    bool ensure_hdr_targets(std::uint32_t pw, std::uint32_t ph, bool need_msaa);
    static void get_taa_jitter(std::uint64_t idx, float &jx, float &jy) noexcept {
        static constexpr float kHalton23[8][2] = {
            { 0.000000f, -0.166667f},
            {-0.250000f,  0.166667f},
            { 0.250000f, -0.388889f},
            {-0.375000f, -0.055556f},
            { 0.125000f,  0.277778f},
            {-0.125000f, -0.277778f},
            { 0.375000f,  0.055556f},
            {-0.437500f,  0.388889f},
        };
        const std::size_t s = static_cast<std::size_t>(idx & 7u);
        jx = kHalton23[s][0];
        jy = kHalton23[s][1];
    }

    RenderTarget &ensure_target(std::uint32_t address, std::uint32_t stride, std::uint32_t format, std::uint32_t min_h);
    void apply_state(const gpu::Ge &ge);
    void draw_batch();
    void upload_texture_2d(VkImageResource &dst, std::uint32_t w, std::uint32_t h, const void *rgba_pixels);
    void upload_overlay_if_needed();
    void submit_recorded_without_present();
    void show_to_swapchain(std::uint32_t tex_slot, float uv_max_x, float uv_max_y, bool linear_filter);
    void submit_and_present(std::uint32_t swap_img_idx);
    void apply_live_scale();
    void apply_live_fps();
    void prepare_interp_draws();
    void render_interpolated_subframe(float phase);
    void apply_post_3d(RenderTarget &rt, const vk::LightingFrameInput &in,
                       const float *taa_prev_from_curr = nullptr, bool taa_valid_reproj = false,
                       bool taa_reset = false, const float *curr_jitter_uv = nullptr,
                       bool use_hdr = false,
                       const vk::ViewPointLight *view_lights = nullptr, std::uint32_t view_light_count = 0,
                       const interpolation::Matrix *view_mat = nullptr);
    void render_sun_shadow_pass(const interpolation::Matrix &view_mat, const vk::LightingFrameInput &light_in,
                                const std::vector<vk::ShadowVertex> &verts,
                                const std::vector<vk::ShadowBatch> &batches);
    std::uint32_t build_view_lights(const interpolation::Matrix &view_mat,
                                    const std::vector<SemanticPointLight> &vfx_lights,
                                    std::array<vk::ViewPointLight, vk::LightingFrameInput::kMaxViewLights> &out_lights) const;
    void snapshot_soft_depth(RenderTarget &rt);
    [[nodiscard]] bool want_vk_effect(std::uint32_t bit) const noexcept {
        return live_->enhanced_lighting.load(std::memory_order_relaxed) &&
               (live_->vulkan_effects_mask.load(std::memory_order_relaxed) & bit) != 0u;
    }
    void apply_draw_fx_bits(vk::DrawParams &dp, bool is_soft_particle, bool is_killcam,
                            std::uint32_t blend_mode) const noexcept {
        if (is_killcam && want_vk_effect(LiveSettings::kVkEffectKillcamDof)) {
            dp.tex_func_bits |= 1024u;
        }
        if (is_soft_particle && soft_depth_copy_.bindless_slot != 0) {
            const bool is_additive_one =
                ((blend_mode & 0xFu) == 10u && ((blend_mode >> 4u) & 0xFu) == 10u && ((blend_mode >> 8u) & 0xFu) == 0u);
            dp.tex_func_bits |= (soft_depth_copy_.bindless_slot << 12u) | (is_additive_one ? 512u : 0u);
            dp.zmax_pad[1] = 0.0014f;
        }
    }
    struct PassStateCache {
        VkPipeline pipe = VK_NULL_HANDLE;
        bool has_scissor = false;
        VkRect2D scissor{};
        bool has_depth = false;
        VkBool32 depth_test = VK_FALSE;
        VkBool32 depth_write = VK_FALSE;
        VkCompareOp depth_op = VK_COMPARE_OP_ALWAYS;
        bool has_blend_const = false;
        float blend_const[4]{};
        bool has_eds3 = false;
        VkBool32 blend_en = VK_FALSE;
        VkColorBlendEquationEXT blend_eq{};
        VkColorComponentFlags write_mask = 0;
        void reset() noexcept { *this = PassStateCache{}; }
    };
    PassStateCache main_pass_state_{};

    [[nodiscard]] bool bind_ge_draw_state(VkCommandBuffer cmd, const DrawSetup &setup, std::uint32_t rt_w,
                                          std::uint32_t rt_h, bool is_soft_particle, bool msaa_4x, bool hdr_3d,
                                          float jitter_ndc_x, float jitter_ndc_y, float mip_lod_bias,
                                          VkDeviceAddress &out_params_addr, PassStateCache &cache);
    void reset_interp_state() noexcept {
        interp_active_ = false;
        interp_can_blend_ = false;
        have_newer_snap_ = false;
        cur_snap_.clear();
        blend_older_.clear();
        blend_newer_.clear();
        cur_raw_batch_start_ = 0;
        raw_old_slot_ = -1;
        raw_new_slot_ = -1;
    }
    void read_back();
    void capture_clean_3d_snapshot(RenderTarget &rt);
    void copy_shown_to_vram(RenderTarget *shown);
    void copy_target_to_ram(RenderTarget &from, std::uint32_t src, std::uint32_t sx, std::uint32_t sy,
                            std::uint32_t dst, std::uint32_t dst_stride, std::uint32_t dx, std::uint32_t dy,
                            std::uint32_t width, std::uint32_t height);
    [[nodiscard]] std::uint32_t match_killcam_hires(std::uint32_t tex_addr, std::uint32_t tex_stride,
                                                    std::uint32_t tex_w, std::uint32_t tex_h, std::uint32_t tex_fmt,
                                                    bool swizzled, const std::uint32_t *rgba_pixels,
                                                    std::uint32_t &out_ox, std::uint32_t &out_oy) noexcept;

    struct RecordedBatch {
        DrawSetup setup{};
        std::uint32_t v_offset = 0;
        std::uint32_t v_count = 0;
        bool use_gpu_raw = false;
    };
    struct InterpSnapshot {
        std::uint32_t shown_addr = 0;
        std::uint32_t width = 512, height = 272;
        bool unblendable = false;
        std::vector<interpolation::DrawSummary> summaries;
        std::vector<std::array<float, 8>> viewports;
        std::vector<std::uint32_t> draw_raw_start;
        std::vector<std::uint32_t> draw_raw_count;
        std::vector<vk::RawGpuVertex> raw_verts;
        std::vector<GpuVertex> flat_verts;
        std::vector<RecordedBatch> batches;
        std::vector<vk::ShadowVertex> shadow_verts;
        std::vector<vk::ShadowBatch> shadow_batches;
        std::vector<vk::RtVertex> rt_static_verts;
        std::vector<vk::RtVertex> rt_dyn_verts;
        void clear() noexcept {
            shown_addr = 0;
            width = 512;
            height = 272;
            unblendable = false;
            summaries.clear();
            viewports.clear();
            draw_raw_start.clear();
            draw_raw_count.clear();
            raw_verts.clear();
            flat_verts.clear();
            batches.clear();
            shadow_verts.clear();
            shadow_batches.clear();
            rt_static_verts.clear();
            rt_dyn_verts.clear();
        }
    };
    struct SnapRawBuf {
        VkBuffer buf = VK_NULL_HANDLE;
        VmaAllocation alloc = VK_NULL_HANDLE;
        VkDeviceAddress addr = 0;
        void *mapped = nullptr;
        std::size_t cap = 0;
        std::uint64_t last_gpu_timeline = 0;
    };
    bool upload_snap_raw(SnapRawBuf &dst, const std::vector<vk::RawGpuVertex> &verts);

    SDL_Window *window_ = nullptr;
    int scale_ = 4;
    psprecomp::GuestMemory &memory_;
    OverlayImage *overlay_ = nullptr;
    int target_fps_ = 30;
    int subframes_ = 1;
    LiveSettings own_live_;
    LiveSettings *live_ = nullptr;

    vk::Context ctx_;
    vk::Swapchain swapchain_;
    vk::BindlessDescriptors descriptors_;
    vk::Pipelines pipelines_;
    vk::PackStreamer pack_streamer_;
    vk::LightingPipeline lighting_;
    vk::SunShadowMap sun_shadows_;
    vk::RayTracedShadows rt_shadows_;
    vk::GpuInterpolator interpolator_;
    std::vector<vk::ShadowVertex> cur_shadow_verts_;
    std::vector<vk::ShadowBatch> cur_shadow_batches_;
    std::vector<vk::RtVertex> cur_rt_static_verts_;
    std::vector<vk::RtVertex> cur_rt_dyn_verts_;
    std::uint32_t cur_shadow_batch_start_ = 0u;
    bool setup_is_shadow_caster_ = false;
    bool setup_is_rt_caster_ = false;

    VkImageResource blur_a_;
    VkImageResource blur_b_;
    std::uint32_t blur_src_slot_ = 0;

    void build_menu_blur(std::uint32_t src_slot, float uv_max_x, float uv_max_y);

    std::array<FrameSlice, kFramesInFlight> frames_{};
    int frame_slot_ = 0;
    std::uint64_t frame_number_ = 1;
    bool recording_ = false;
    bool in_render_pass_ = false;
    bool swapchain_dirty_ = false;
    std::uint32_t active_target_addr_ = UINT32_MAX;

    std::unordered_map<std::uint32_t, RenderTarget> targets_;
    std::unordered_map<std::uint64_t, CachedTexture> textures_;
    struct FrameTexBinding {
        std::uint32_t tex_slot = 0;
        vk::PackPbrSlots pbr_slots{};
        bool texture_mips = false;
        bool linear_override = false;
        bool is_killcam = false;
        float uv_xform[4]{1.0f, 1.0f, 0.0f, 0.0f};
    };
    struct PackCacheEntry {
        std::uint32_t slot = 0;
        vk::PackPbrSlots pbr_slots{};
        bool has_mips = false;
        bool linear_override = false;
        bool pending = false;
        std::uint64_t last_try_frame = 0;
    };
    // Per-frame fast texture-parameter cache (eliminates redundant full-texture hashing when many monsters
    // share textures across hundreds of skinned draw calls) + persistent pack-lookup cache keyed by content hash.
    std::unordered_map<std::uint64_t, FrameTexBinding> frame_tex_cache_;
    std::unordered_map<std::uint64_t, PackCacheEntry> pack_cache_;
    std::deque<RetiredImage> retired_images_;

    VkImageResource cpu_display_tex_;
    VkImageResource copy_tex_;
    VkImageResource soft_depth_copy_;
    VkImageResource msaa_color_;
    VkImageResource msaa_depth_;
    VkImageResource hdr_color_;
    VkImageResource msaa_hdr_color_;
    VkImageResource overlay_tex_;
    bool main_depth_snapped_ = false;
    bool active_pass_msaa_ = false;
    bool active_pass_hdr_ = false;
    bool main_msaa_started_ = false;
    bool main_hdr_started_ = false;
    bool main_3d_finished_ = false;
    std::uint64_t taa_sample_idx_ = 0;
    float taa_prev_phase_ = 0.0f;
    bool taa_have_prev_ = false;
    float taa_prev_jitter_uv_[2]{0.0f, 0.0f};
    float main_taa_jitter_uv_[2]{0.0f, 0.0f};
    float main_taa_jitter_ndc_[2]{0.0f, 0.0f};
    bool main_taa_jitter_set_ = false;
    int last_aa_mode_ = -1;
    std::uint64_t overlay_gen_ = 0;
    std::uint64_t overlay_shown_at_ms_ = 0;
    bool overlay_visible_ = false;
    bool overlay_continuous_ = false;
    float overlay_aspect_ = 0.0f;
    std::chrono::steady_clock::time_point last_menu_draw_{};

    // Quest HUD detection + HUD sizing + widescreen 2D aspect preservation (drives Better Sharpness,
    // Gathering Assistance, Hunting Horn Songs, and pixel-aligned Key Quest Markers).
    std::uint32_t cur_layer_ = gpu::DrawInfo::kNoLayer;
    std::uint32_t hud_layer_ = gpu::DrawInfo::kNoLayer;
    bool quest_hud_this_flip_ = false;
    std::array<std::uint32_t, 2> shown_addrs_{UINT32_MAX, UINT32_MAX};
    float frame_stretch_ = 1.0f;

    [[nodiscard]] bool is_shown_target(std::uint32_t addr) const noexcept {
        if (shown_addrs_[0] == UINT32_MAX) return true;
        return addr == shown_addrs_[0] || addr == shown_addrs_[1];
    }

    // Clean HUD-free internal-res killcam capture + async VRAM readback (fixes black quest-end killcam and
    // upgrades it to full internal resolution without HUD baked in, supporting both full-frame and tiled/striped draws).
    static constexpr std::size_t kKillcamSlots = 12;
    struct KillcamSlot {
        VkImageResource image;
        std::vector<std::uint32_t> thumb; // 480x272 RGBA8 reference for full-frame or sub-tile matching
        bool valid = false;
        std::uint64_t frame = 0;
    };
    struct PendingVramReadback {
        bool active = false;
        std::uint64_t timeline_value = 0;
        std::uint32_t address = 0;
        std::uint32_t stride = 512;
        std::uint32_t format = 3;
        std::uint32_t width = 480;
        std::uint32_t height = 272;
        bool has_killcam_hires = false;
    };
    VkImageResource killcam_clean_;
    VkImageResource killcam_staging_;
    std::uint32_t killcam_clean_target_ = UINT32_MAX;
    bool killcam_clean_valid_ = false;
    KillcamSlot killcam_pinned_{};
    std::array<KillcamSlot, kKillcamSlots> killcam_slots_{};
    std::size_t killcam_next_slot_ = 0;
    std::uint32_t killcam_ram_addr_ = 0;
    std::uint32_t killcam_ram_bpp_ = 4;
    std::uint32_t killcam_ram_stride_ = 512;
    std::uint64_t killcam_ram_verified_frame_ = 0;
    bool frame_had_3d_ = false;
    bool captured_clean_3d_ = false;
    bool lit_3d_this_frame_ = false;
    vk::LightingFrameInput cur_light_in_{};
    vk::LightingFrameInput last_light_in_{};
    interpolation::Matrix cur_view_mat_{interpolation::identity()};
    interpolation::Matrix last_view_mat_{interpolation::identity()};
    bool has_cur_view_mat_ = false;
    bool has_last_view_mat_ = false;
    std::vector<SemanticPointLight> cur_vfx_lights_;
    std::vector<SemanticPointLight> last_vfx_lights_;
    float batch_world_sum_[3]{0.0f, 0.0f, 0.0f};
    float batch_col_sum_[3]{0.0f, 0.0f, 0.0f};
    std::uint32_t batch_world_cnt_ = 0;
    VkImageResource vram_rb_img_;
    VkBuffer vram_rb_buf_ = VK_NULL_HANDLE;
    VmaAllocation vram_rb_alloc_ = VK_NULL_HANDLE;
    void *vram_rb_mapped_ = nullptr;
    PendingVramReadback pending_vram_rb_{};

    std::uint32_t last_shown_slot_ = 0;
    float last_shown_uv_[2]{1.0f, 1.0f};
    bool last_shown_linear_ = true;

    bool have_state_ = false;
    std::uint64_t last_state_gen_ = 0;
    DrawKey state_key_{};
    DrawSetup setup_{};
    std::vector<GpuVertex> batch_;
    std::vector<std::uint32_t> decode_buf_;

    // Phase 5: GPU compute frame interpolation state (60/120/180 FPS).
    InterpSnapshot cur_snap_;
    InterpSnapshot blend_older_;
    InterpSnapshot blend_newer_;
    bool have_newer_snap_ = false;
    std::uint32_t cur_raw_batch_start_ = 0;
    bool cur_draw_eligible_ = false;
    std::uint32_t cur_draw_index_ = 0;

    // 3-slot ring so each 30 Hz frame uploads raw_verts ONCE (older simply reuses the previous frame's slot),
    // and never overwrites a buffer still referenced by an in-flight subframe's place_v2.comp.
    SnapRawBuf raw_ring_[3];
    int raw_old_slot_ = -1;
    int raw_new_slot_ = -1;
    int raw_next_slot_ = 0;

    struct PreparedDraw {
        bool active = false;
        bool has_partner = false;
        bool has_rigid = false;
        bool follow_cam = false;
        bool has_inv_world_old = false;
        bool has_vertex_pair = false;
        bool proj_differs = false;
        std::size_t partner = 0;
        std::uint32_t old_start = 0;
        std::uint32_t new_start = 0;
        std::uint32_t max_deform_bits = 0;
        interpolation::Matrix eye_old{};
        interpolation::Matrix eye_new{};
        interpolation::Matrix inv_world_old{};
        interpolation::Matrix inv_world_new{};
        interpolation::RigidMotion rm{};
    };
    std::vector<PreparedDraw> prepared_draws_;
    std::unordered_map<std::uint64_t, std::uint32_t> interp_key_counts_;

    RenderTarget interp_rt_;
    bool interp_active_ = false;
    bool interp_can_blend_ = false;
    interpolation::Matching last_matching_{};
    interpolation::RigidMotion last_cam_motion_{};
    interpolation::RigidMotion prev_cam_motion_{};
    bool has_prev_cam_ = false;
    int next_subframe_ = 0;
    std::chrono::steady_clock::time_point flip_time_{};
    std::chrono::steady_clock::duration flip_period_{std::chrono::microseconds(33333)};
    std::chrono::steady_clock::time_point next_subframe_at_{};

    // Optional per-stage profiler enabled via MHF_VK_PROFILE=1 (zero overhead when off).
    struct ScopedTimer {
        bool enabled;
        std::uint64_t &accum_ns;
        std::chrono::steady_clock::time_point start;
        ScopedTimer(bool on, std::uint64_t &ns)
            : enabled(on), accum_ns(ns),
              start(on ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}) {}
        ~ScopedTimer() {
            if (enabled) {
                accum_ns += static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
            }
        }
    };

    struct ProfStats {
        bool enabled = false;
        std::uint64_t frames = 0;
        std::uint64_t draws = 0;
        std::uint64_t batches = 0;
        std::uint64_t raw_verts = 0;
        std::uint64_t flat_verts = 0;
        std::uint64_t tex_hits = 0;
        std::uint64_t tex_misses = 0;
        std::uint64_t pack_queries = 0;
        std::uint64_t interp_calls = 0;
        std::uint64_t apply_ns = 0;
        std::uint64_t batch_ns = 0;
        std::uint64_t draw_start_ns = 0;
        std::uint64_t match_ns = 0;
        std::uint64_t up_raw_ns = 0;
        std::uint64_t interp_ns = 0;
        std::uint64_t light_ns = 0;
        std::uint64_t wait_gpu_ns = 0;
        std::uint64_t acq_ns = 0;
        std::uint64_t pres_ns = 0;
        std::chrono::steady_clock::time_point window_start{};
    } prof_{};
};

bool VkBackend::init() {
    if (const char *v = std::getenv("MHF_VK_PROFILE"); v != nullptr && v[0] == '1') {
        prof_.enabled = true;
        prof_.window_start = std::chrono::steady_clock::now();
    }
    if (!ctx_.init(window_)) return false;
    live_->msaa_4x_supported.store(ctx_.caps.msaa_4x, std::memory_order_relaxed);
    live_->ray_query_supported.store(ctx_.caps.ray_query, std::memory_order_relaxed);

    int pw = 0, ph = 0;
    SDL_GetWindowSizeInPixels(window_, &pw, &ph);
    if (pw <= 0 || ph <= 0) {
        pw = 480 * scale_;
        ph = 272 * scale_;
    }
    if (!swapchain_.create(ctx_, pw, ph)) return false;
    if (!descriptors_.init(ctx_)) return false;
    if (!pipelines_.init(ctx_, descriptors_.layout(), swapchain_.format)) return false;
    pack_streamer_.init(ctx_, live_->texture_pack_folder, live_->texture_pack_game,
                        live_->builtin_textures_folder, live_->texture_pack_mipmaps.load());
    lighting_.init(ctx_, descriptors_.layout(), live_);
    sun_shadows_.init(ctx_, descriptors_.layout(), descriptors_);
    rt_shadows_.init(ctx_, descriptors_.layout());
    interpolator_.init(ctx_);

    for (int i = 0; i < kFramesInFlight; ++i) {
        VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr,
                                     VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, ctx_.families.graphics};
        if (vkCreateCommandPool(ctx_.device, &cpci, nullptr, &frames_[i].pool) != VK_SUCCESS) return false;

        VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr,
                                         frames_[i].pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        if (vkAllocateCommandBuffers(ctx_.device, &cbai, &frames_[i].cmd) != VK_SUCCESS) return false;

        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, nullptr, 0};
        if (vkCreateSemaphore(ctx_.device, &sci, nullptr, &frames_[i].image_available) != VK_SUCCESS) return false;

        if (!frames_[i].vertices.create(ctx_, kVertexRingBytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) ||
            !frames_[i].params.create(ctx_, kParamsRingBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) ||
            !frames_[i].staging.create(ctx_, kStagingRingBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) {
            return false;
        }
    }

    batch_.reserve(65536);
    subframes_ = std::clamp((target_fps_ + 15) / 30, 1, 6);
    std::fprintf(stderr, "[vk] ready, internal resolution %dx%d, target %d fps (%dx subframes)\n",
                 480 * scale_, 272 * scale_, subframes_ * 30, subframes_);
    return true;
}

void VkBackend::shutdown() {
    if (ctx_.device == VK_NULL_HANDLE) return;
    ctx_.wait_idle();

    for (auto &[addr, rt] : targets_) {
        rt.color.destroy_immediate(ctx_, &descriptors_);
        rt.depth.destroy_immediate(ctx_, &descriptors_);
    }
    targets_.clear();

    for (auto &[hash, tex] : textures_) {
        tex.image.destroy_immediate(ctx_, &descriptors_);
    }
    textures_.clear();

    cpu_display_tex_.destroy_immediate(ctx_, &descriptors_);
    copy_tex_.destroy_immediate(ctx_, &descriptors_);
    soft_depth_copy_.destroy_immediate(ctx_, &descriptors_);
    msaa_color_.destroy_immediate(ctx_, nullptr);
    msaa_depth_.destroy_immediate(ctx_, nullptr);
    hdr_color_.destroy_immediate(ctx_, &descriptors_);
    msaa_hdr_color_.destroy_immediate(ctx_, nullptr);
    overlay_tex_.destroy_immediate(ctx_, &descriptors_);
    blur_a_.destroy_immediate(ctx_, &descriptors_);
    blur_b_.destroy_immediate(ctx_, &descriptors_);
    killcam_clean_.destroy_immediate(ctx_, &descriptors_);
    killcam_staging_.destroy_immediate(ctx_, &descriptors_);
    killcam_pinned_.image.destroy_immediate(ctx_, &descriptors_);
    for (auto &ks : killcam_slots_) ks.image.destroy_immediate(ctx_, &descriptors_);
    vram_rb_img_.destroy_immediate(ctx_, nullptr);
    if (vram_rb_buf_ != VK_NULL_HANDLE) {
        vmaDestroyBuffer(ctx_.allocator, vram_rb_buf_, vram_rb_alloc_);
        vram_rb_buf_ = VK_NULL_HANDLE;
        vram_rb_alloc_ = VK_NULL_HANDLE;
        vram_rb_mapped_ = nullptr;
    }
    interp_rt_.color.destroy_immediate(ctx_, &descriptors_);
    interp_rt_.depth.destroy_immediate(ctx_, &descriptors_);
    auto destroy_snap_buf = [&](SnapRawBuf &sb) {
        if (sb.buf != VK_NULL_HANDLE) {
            vmaDestroyBuffer(ctx_.allocator, sb.buf, sb.alloc);
            sb = SnapRawBuf{};
        }
    };
    for (int i = 0; i < 3; ++i) {
        destroy_snap_buf(raw_ring_[i]);
    }

    pack_streamer_.shutdown(ctx_, descriptors_);
    rt_shadows_.shutdown(ctx_, descriptors_);
    sun_shadows_.shutdown(ctx_, descriptors_);
    lighting_.shutdown(ctx_, descriptors_);
    interpolator_.shutdown(ctx_);

    collect_retired(UINT64_MAX);

    for (auto &f : frames_) {
        f.vertices.destroy(ctx_);
        f.params.destroy(ctx_);
        f.staging.destroy(ctx_);
        if (f.image_available != VK_NULL_HANDLE) {
            vkDestroySemaphore(ctx_.device, f.image_available, nullptr);
            f.image_available = VK_NULL_HANDLE;
        }
        if (f.pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(ctx_.device, f.pool, nullptr);
            f.pool = VK_NULL_HANDLE;
            f.cmd = VK_NULL_HANDLE;
        }
    }

    pipelines_.shutdown(ctx_);
    descriptors_.shutdown(ctx_);
    swapchain_.destroy(ctx_);
    ctx_.shutdown();
}

void VkBackend::retire_image(VkImageResource &res) {
    if (res.image == VK_NULL_HANDLE && res.view == VK_NULL_HANDLE) return;
    if (res.view != VK_NULL_HANDLE) lighting_.invalidate_scene_view(res.view);
    // Frame-gen snapshots (blend_older_ / blend_newer_) hold bindless texture slots across up to
    // 2 full 30 Hz frames (2 * subframes_ submissions), so keep retired images alive past that horizon.
    const std::uint64_t margin = static_cast<std::uint64_t>(std::max(subframes_, 1)) * 3u + 4u;
    const std::uint64_t retire_at = ctx_.graphics_timeline_value + margin;
    if (res.bindless_slot != 0) {
        if (res.bindless_slot == last_shown_slot_) last_shown_slot_ = 0;
        if (res.bindless_slot == blur_src_slot_) blur_src_slot_ = 0;
        descriptors_.free_texture(res.bindless_slot, retire_at);
        res.bindless_slot = 0;
    }
    retired_images_.push_back({res.image, res.alloc, res.view, retire_at});
    res.image = VK_NULL_HANDLE;
    res.alloc = VK_NULL_HANDLE;
    res.view = VK_NULL_HANDLE;
    res.layout = VK_IMAGE_LAYOUT_UNDEFINED;
}

void VkBackend::collect_retired(std::uint64_t completed) {
    descriptors_.collect_retired(completed);
    pack_streamer_.collect_staging(ctx_, completed);
    while (!retired_images_.empty() && retired_images_.front().retire_value <= completed) {
        const auto &r = retired_images_.front();
        if (r.view != VK_NULL_HANDLE) vkDestroyImageView(ctx_.device, r.view, nullptr);
        if (r.image != VK_NULL_HANDLE) vmaDestroyImage(ctx_.allocator, r.image, r.alloc);
        retired_images_.pop_front();
    }
}

void VkBackend::ensure_recording() {
    if (recording_) return;
    FrameSlice &f = frames_[frame_slot_];
    const auto t0 = prof_.enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    ctx_.wait_timeline(ctx_.graphics_timeline, f.timeline_value);
    if (prof_.enabled) {
        prof_.wait_gpu_ns += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
    }
    collect_retired(ctx_.query_timeline(ctx_.graphics_timeline));

    vkResetCommandPool(ctx_.device, f.pool, 0);
    f.vertices.reset();
    f.params.reset();
    f.staging.reset();

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
                                VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
    vkBeginCommandBuffer(f.cmd, &bi);
    recording_ = true;
    in_render_pass_ = false;
    active_target_addr_ = UINT32_MAX;
}

void VkBackend::end_render_pass() {
    if (!in_render_pass_) return;
    vkCmdEndRendering(frames_[frame_slot_].cmd);
    in_render_pass_ = false;
    active_pass_msaa_ = false;
    active_pass_hdr_ = false;
    active_target_addr_ = UINT32_MAX;
}

bool VkBackend::ensure_msaa_targets(std::uint32_t pw, std::uint32_t ph) {
    if (!ctx_.caps.msaa_4x || pw == 0 || ph == 0) return false;
    if (msaa_color_.image != VK_NULL_HANDLE && msaa_color_.width == pw && msaa_color_.height == ph &&
        msaa_depth_.image != VK_NULL_HANDLE && msaa_depth_.width == pw && msaa_depth_.height == ph) {
        return true;
    }
    retire_image(msaa_color_);
    retire_image(msaa_depth_);
    return msaa_color_.create(ctx_, nullptr, pw, ph, VK_FORMAT_R8G8B8A8_UNORM,
                              VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 1,
                              VK_SAMPLE_COUNT_4_BIT) &&
           msaa_depth_.create(ctx_, nullptr, pw, ph, VK_FORMAT_D32_SFLOAT,
                              VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, 1,
                              VK_SAMPLE_COUNT_4_BIT);
}

bool VkBackend::ensure_hdr_targets(std::uint32_t pw, std::uint32_t ph, bool need_msaa) {
    if (pw == 0 || ph == 0) return false;
    if (hdr_color_.image == VK_NULL_HANDLE || hdr_color_.width != pw || hdr_color_.height != ph) {
        retire_image(hdr_color_);
        if (!hdr_color_.create(ctx_, &descriptors_, pw, ph, VK_FORMAT_B10G11R11_UFLOAT_PACK32,
                               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                               VK_IMAGE_ASPECT_COLOR_BIT)) {
            return false;
        }
    }
    if (need_msaa) {
        if (!ensure_msaa_targets(pw, ph)) return false;
        if (msaa_hdr_color_.image == VK_NULL_HANDLE || msaa_hdr_color_.width != pw || msaa_hdr_color_.height != ph) {
            retire_image(msaa_hdr_color_);
            if (!msaa_hdr_color_.create(ctx_, nullptr, pw, ph, VK_FORMAT_B10G11R11_UFLOAT_PACK32,
                                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 1,
                                        VK_SAMPLE_COUNT_4_BIT)) {
                return false;
            }
        }
    }
    return hdr_color_.bindless_slot != 0;
}

void VkBackend::transition_resource(VkImageResource &res, VkImageLayout new_layout,
                                    VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access) {
    if (res.image == VK_NULL_HANDLE || res.layout == new_layout) return;
    VkPipelineStageFlags2 src_stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkAccessFlags2 src_access = VK_ACCESS_2_MEMORY_WRITE_BIT;
    if (res.layout == VK_IMAGE_LAYOUT_UNDEFINED) {
        src_stage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        src_access = 0;
    } else if (res.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        src_stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        src_access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    } else if (res.layout == VK_IMAGE_LAYOUT_GENERAL) {
        src_stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        src_access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    } else if (res.layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL) {
        src_stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        src_access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    } else if (res.layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
        src_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        src_access = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    } else if (res.layout == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL) {
        src_stage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        src_access = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    } else if (res.layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        src_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        src_access = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    } else if (res.layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
        src_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        src_access = VK_ACCESS_2_TRANSFER_READ_BIT;
    }
    vk::Context::cmd_transition_image(frames_[frame_slot_].cmd, res.image, res.aspect, res.layout, new_layout,
                                      src_stage, src_access, dst_stage, dst_access, 0, res.mip_levels);
    res.layout = new_layout;
}

RenderTarget &VkBackend::ensure_target(std::uint32_t address, std::uint32_t stride, std::uint32_t format,
                                       std::uint32_t min_h) {
    const std::uint32_t w = std::max(stride, 64u);
    const std::uint32_t h = std::max(min_h, 272u);
    auto &rt = targets_[address];
    const std::uint32_t pw = w * static_cast<std::uint32_t>(scale_);
    const std::uint32_t ph = h * static_cast<std::uint32_t>(scale_);
    if (rt.color.image == VK_NULL_HANDLE || rt.color.width != pw || rt.color.height != ph) {
        retire_image(rt.color);
        retire_image(rt.depth);
        rt.address = address;
        rt.stride = w;
        rt.width = w;
        rt.height = h;
        rt.format = format;
        rt.color.create(ctx_, &descriptors_, pw, ph, VK_FORMAT_R8G8B8A8_UNORM,
                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                            VK_IMAGE_USAGE_STORAGE_BIT |
                            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                        VK_IMAGE_ASPECT_COLOR_BIT);
        rt.depth.create(ctx_, &descriptors_, pw, ph, VK_FORMAT_D32_SFLOAT,
                        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                        VK_IMAGE_ASPECT_DEPTH_BIT);
    } else {
        rt.stride = w;
        rt.format = format;
    }
    return rt;
}

void VkBackend::begin_target_pass(RenderTarget &rt, bool use_msaa, bool use_hdr) {
    ensure_recording();
    if (use_msaa && !ensure_msaa_targets(rt.color.width, rt.color.height)) {
        use_msaa = false;
    }
    if (use_hdr && !ensure_hdr_targets(rt.color.width, rt.color.height, use_msaa)) {
        use_hdr = false;
    }
    if (in_render_pass_ && active_target_addr_ == rt.address && active_pass_msaa_ == use_msaa &&
        active_pass_hdr_ == use_hdr) {
        return;
    }
    end_render_pass();

    const bool first_use_color = (rt.color.layout == VK_IMAGE_LAYOUT_UNDEFINED);
    const bool first_use_depth = (rt.depth.layout == VK_IMAGE_LAYOUT_UNDEFINED);
    transition_resource(rt.color, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    transition_resource(rt.depth, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

    VkRenderingAttachmentInfo color_att{};
    color_att.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    color_att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color_att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_att.clearValue.color = {{0.0f, 0.0f, 0.0f, 1.0f}};

    VkRenderingAttachmentInfo depth_att{};
    depth_att.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depth_att.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depth_att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depth_att.clearValue.depthStencil = {1.0f, 0u};

    if (use_hdr) {
        transition_resource(hdr_color_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                            VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        const VkAttachmentLoadOp hdr_load = main_hdr_started_ ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
        if (use_msaa) {
            transition_resource(msaa_hdr_color_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
            transition_resource(msaa_depth_, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
            color_att.imageView = msaa_hdr_color_.view;
            color_att.loadOp = hdr_load;
            color_att.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
            color_att.resolveImageView = hdr_color_.view;
            color_att.resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            depth_att.imageView = msaa_depth_.view;
            depth_att.loadOp = main_msaa_started_ ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
            depth_att.resolveMode = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT;
            depth_att.resolveImageView = rt.depth.view;
            depth_att.resolveImageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
            main_msaa_started_ = true;
        } else {
            color_att.imageView = hdr_color_.view;
            color_att.loadOp = hdr_load;
            depth_att.imageView = rt.depth.view;
            depth_att.loadOp = first_use_depth ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
        }
        main_hdr_started_ = true;
    } else if (use_msaa) {
        transition_resource(msaa_color_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                            VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        transition_resource(msaa_depth_, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
        color_att.imageView = msaa_color_.view;
        color_att.loadOp = main_msaa_started_ ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
        color_att.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
        color_att.resolveImageView = rt.color.view;
        color_att.resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        depth_att.imageView = msaa_depth_.view;
        depth_att.loadOp = main_msaa_started_ ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth_att.resolveMode = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT;
        depth_att.resolveImageView = rt.depth.view;
        depth_att.resolveImageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        main_msaa_started_ = true;
    } else {
        color_att.imageView = rt.color.view;
        color_att.loadOp = first_use_color ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
        depth_att.imageView = rt.depth.view;
        depth_att.loadOp = first_use_depth ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
    }

    VkRenderingInfo ri{};
    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea = {{0, 0}, {rt.color.width, rt.color.height}};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1u;
    ri.pColorAttachments = &color_att;
    ri.pDepthAttachment = &depth_att;

    vkCmdBeginRendering(frames_[frame_slot_].cmd, &ri);
    main_pass_state_.reset();
    VkDescriptorSet set0 = descriptors_.set();
    vkCmdBindDescriptorSets(frames_[frame_slot_].cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines_.ge_layout, 0, 1, &set0, 0, nullptr);
    VkViewport vp{0.0f, 0.0f, static_cast<float>(rt.color.width), static_cast<float>(rt.color.height), 0.0f, 1.0f};
    vkCmdSetViewport(frames_[frame_slot_].cmd, 0, 1, &vp);
    in_render_pass_ = true;
    active_pass_msaa_ = use_msaa;
    active_pass_hdr_ = use_hdr;
    active_target_addr_ = rt.address;
    rt.last_draw = frame_number_;
}

void VkBackend::upload_texture_2d(VkImageResource &dst, std::uint32_t w, std::uint32_t h, const void *rgba_pixels) {
    ensure_recording();
    end_render_pass();

    if (dst.image == VK_NULL_HANDLE || dst.width != w || dst.height != h) {
        retire_image(dst);
        if (!dst.create(ctx_, &descriptors_, w, h, VK_FORMAT_R8G8B8A8_UNORM,
                        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                        VK_IMAGE_ASPECT_COLOR_BIT)) {
            return;
        }
    }

    const std::size_t bytes = static_cast<std::size_t>(w) * h * 4u;
    auto slice = frames_[frame_slot_].staging.allocate(bytes, 16);
    if (!slice.ptr) {
        // Never leave a newly created image in UNDEFINED layout while bound to a bindless slot.
        if (dst.layout == VK_IMAGE_LAYOUT_UNDEFINED) {
            transition_resource(dst, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }
        return;
    }
    std::memcpy(slice.ptr, rgba_pixels, bytes);

    transition_resource(dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

    VkBufferImageCopy region{};
    region.bufferOffset = slice.offset;
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {w, h, 1};
    vkCmdCopyBufferToImage(frames_[frame_slot_].cmd, frames_[frame_slot_].staging.buffer, dst.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    transition_resource(dst, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

void VkBackend::apply_state(const gpu::Ge &ge) {
    const std::uint64_t sgen = ge.state_generation();
    if (have_state_ && sgen == last_state_gen_) return;

    DrawKey key{};
    for (std::size_t i = 0; i < kStateRegs.size(); ++i) key[i] = ge.reg(kStateRegs[i]);
    key[kStateRegs.size()] = static_cast<std::uint32_t>(ge.clut_generation() & 0xFFFFFFFFu);
    key[kStateRegs.size() + 1] = ge.reg(kMaskAlpha);

    last_state_gen_ = sgen;
    if (have_state_ && key == state_key_) return;
    draw_batch();
    ScopedTimer prof_scope(prof_.enabled, prof_.apply_ns);

    state_key_ = key;
    have_state_ = true;
    setup_ = DrawSetup{};
    setup_.key = key;
    setup_.through = (ge.reg(kVertexType) & 0x800000u) != 0;
    setup_.is_skinned = !setup_.through && (((ge.reg(kVertexType) >> 9u) & 3u) != 0);

    const std::uint32_t fb_addr = ((ge.reg(kFramebufPtr) & 0xFFFFFFu) | ((ge.reg(kFramebufWidth) & 0xFF0000u) << 8u)) & 0x001FFFFFu;
    const std::uint32_t fb_stride = ge.reg(kFramebufWidth) & 0xFFFFu;
    const std::uint32_t fb_fmt = ge.reg(kFramebufPixFormat) & 3u;
    setup_.target = fb_addr;
    ensure_target(fb_addr, fb_stride, fb_fmt, 272u);

    const bool clear = (ge.reg(kClearMode) & 1u) != 0;
    const bool textured = !clear && (ge.reg(kTextureMapEnable) & 1u) != 0;
    {
        const bool ab = (ge.reg(kAlphaBlendEnable) & 1u) != 0;
        const bool zw_off = (ge.reg(kZWriteDisable) & 1u) != 0;
        const std::uint32_t bm = ge.reg(kBlendMode);
        const std::uint32_t bop = (bm >> 8u) & 0xFu;
        const std::uint32_t bdst = (bm >> 4u) & 0xFu;
        const bool dst_one = (bdst == 10u && (ge.reg(kBlendFixedB) & 0xFFFFFFu) == 0xFFFFFFu);
        setup_.is_additive_vfx = !clear && !setup_.through && ab && zw_off && bop == 0u && dst_one;
    }
    if (!setup_.through) {
        if ((ge.reg(0x18u) & 1u) != 0) {
            auto f24 = [](std::uint32_t r) {
                const std::uint32_t bits = r << 8u;
                float f = 0.0f;
                std::memcpy(&f, &bits, 4);
                return f;
            };
            cur_light_in_.light0_dir[0] = f24(ge.reg(0x63u));
            cur_light_in_.light0_dir[1] = f24(ge.reg(0x64u));
            cur_light_in_.light0_dir[2] = f24(ge.reg(0x65u));
            cur_light_in_.has_light0 = true;
        }
        if ((ge.reg(kFogEnable) & 1u) != 0 && (ge.reg(kAlphaBlendEnable) & 1u) == 0 && !cur_light_in_.has_psp_fog) {
            unpack_rgb(ge.reg(kFogColor), cur_light_in_.psp_fog_rgb);
            cur_light_in_.has_psp_fog = true;
        }
    }

    if (textured) {
        const auto info = ge.texture_info();
        setup_.tex_width = static_cast<float>(std::max(info.width, 1u));
        setup_.tex_height = static_cast<float>(std::max(info.height, 1u));

        // Check if texture lives inside an existing RenderTarget (render-to-texture).
        const std::uint32_t tex_vram = info.address & 0x1FFFFFFFu;
        bool rtt_hit = false;
        if (tex_vram >= 0x04000000u && tex_vram < 0x04800000u) {
            const std::uint32_t off = tex_vram & 0x001FFFFFu;
            for (auto &[rt_addr, rt] : targets_) {
                const std::uint32_t bpp = rt.bytes_per_pixel();
                const std::uint32_t rt_bytes = rt.stride * rt.height * bpp;
                if (off >= rt_addr && off < rt_addr + rt_bytes) {
                    const std::uint32_t rel_px = (off - rt_addr) / bpp;
                    const std::uint32_t oy = rel_px / rt.stride;
                    const std::uint32_t ox = rel_px % rt.stride;
                    setup_.uv_xform[0] = static_cast<float>(info.width) / static_cast<float>(rt.width);
                    setup_.uv_xform[1] = static_cast<float>(info.height) / static_cast<float>(rt.height);
                    setup_.uv_xform[2] = static_cast<float>(ox) / static_cast<float>(rt.width);
                    setup_.uv_xform[3] = static_cast<float>(oy) / static_cast<float>(rt.height);

                    if (rt_addr == setup_.target) {
                        // Self-sampling: copy active target into copy_tex_ to avoid read/write hazard.
                        ensure_recording();
                        end_render_pass();
                        if (copy_tex_.image == VK_NULL_HANDLE || copy_tex_.width != rt.color.width || copy_tex_.height != rt.color.height) {
                            retire_image(copy_tex_);
                            copy_tex_.create(ctx_, &descriptors_, rt.color.width, rt.color.height, VK_FORMAT_R8G8B8A8_UNORM,
                                             VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                             VK_IMAGE_ASPECT_COLOR_BIT);
                        }
                        transition_resource(rt.color, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
                        transition_resource(copy_tex_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
                        VkImageCopy cp{};
                        cp.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                        cp.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                        cp.extent = {rt.color.width, rt.color.height, 1};
                        vkCmdCopyImage(frames_[frame_slot_].cmd, rt.color.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                       copy_tex_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);
                        transition_resource(copy_tex_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                        setup_.tex_slot = copy_tex_.bindless_slot;
                        setup_.self_sampled = true;
                    } else {
                        ensure_recording();
                        end_render_pass();
                        transition_resource(rt.color, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                        setup_.tex_slot = rt.color.bindless_slot;
                    }
                    rtt_hit = true;
                    break;
                }
            }
        }

        if (!rtt_hit && info.width > 0 && info.height > 0) {
            // Fast per-frame parameter cache: avoids re-hashing the same monster/terrain texture bytes
            // hundreds of times per frame across skinned multi-part draw calls!
            const bool indexed = info.format >= 4u && info.format <= 7u;
            std::uint64_t param_key = static_cast<std::uint64_t>(info.address) * 1315423911ull ^
                                      (static_cast<std::uint64_t>(info.format) << 40u) ^
                                      (static_cast<std::uint64_t>(info.width) << 20u) ^
                                      static_cast<std::uint64_t>(info.height) ^
                                      (static_cast<std::uint64_t>(info.stride) << 48u) ^
                                      (info.swizzled ? 0x8000000000000000ull : 0u);
            if (indexed) {
                param_key ^= (static_cast<std::uint64_t>(ge.reg(kClutFormat)) * 0x9E3779B97F4A7C15ull) ^
                             (ge.clut_generation() * 0xBF58476D1CE4E5B9ull);
            }
            if (const auto fit = frame_tex_cache_.find(param_key); fit != frame_tex_cache_.end()) {
                if (prof_.enabled) ++prof_.tex_hits;
                setup_.tex_slot = fit->second.tex_slot;
                setup_.pbr_slots = fit->second.pbr_slots;
                setup_.texture_mips = fit->second.texture_mips;
                setup_.linear_override = fit->second.linear_override;
                setup_.is_killcam = fit->second.is_killcam;
                std::copy_n(fit->second.uv_xform, 4, setup_.uv_xform);
            } else {
                if (prof_.enabled) ++prof_.tex_misses;
                // Compute content hash ONCE per unique texture per frame.
                const std::uint64_t h = ge.texture_hash();
                auto build_facts = [&]() {
                    textures::TextureFacts facts{};
                    facts.address = info.address;
                    facts.buffer_width = info.stride;
                    facts.format = info.format;
                    facts.width = info.width;
                    facts.height = info.height;
                    facts.dimension = static_cast<std::uint16_t>(ge.reg(kTexSize0) & 0xFFFFu);
                    facts.clut_format_word = (kClutFormat << 24u) | (ge.reg(kClutFormat) & 0xFFFFFFu);
                    facts.clut = ge.clut_bytes();
                    facts.clut_last_bytes = ge.clut_last_bytes();
                    facts.clut_max_bytes = ge.clut_max_bytes();
                    return facts;
                };
                if (!setup_.through && live_->enhanced_lighting.load(std::memory_order_relaxed)) {
                    lighting_.note_texture(h, build_facts(),
                                           [&ge](std::uint32_t addr, std::uint32_t sz) { return ge.guest_bytes(addr, sz); });
                }

                // Phase 3: check texture pack (cached by content hash so XXHash never runs on repeated draws).
                std::uint32_t pack_slot = 0;
                const bool want_pack = pack_streamer_.active() && live_->texture_pack.load(std::memory_order_relaxed);
                if (want_pack) {
                    auto pit = pack_cache_.find(h);
                    const bool need_query = (pit == pack_cache_.end()) ||
                                            (pit->second.pending && pit->second.last_try_frame != frame_number_);
                    if (need_query) {
                        if (prof_.enabled) ++prof_.pack_queries;
                        const textures::TextureFacts facts = build_facts();
                        textures::Filter pfilter = textures::Filter::Auto;
                        bool has_mips = false;
                        bool pending = false;
                        vk::PackPbrSlots pbr_slots{};
                        const std::uint32_t slot = pack_streamer_.lookup_or_queue(
                            ctx_, descriptors_, facts,
                            [&ge](std::uint32_t addr, std::uint32_t sz) { return ge.guest_bytes(addr, sz); },
                            [this]() -> VkCommandBuffer {
                                ensure_recording();
                                end_render_pass();
                                return frames_[frame_slot_].cmd;
                            },
                            ctx_.graphics_timeline_value + 1u,
                            frame_number_, has_mips, pfilter, pending, &pbr_slots);
                        PackCacheEntry entry{};
                        entry.slot = slot;
                        entry.pbr_slots = pbr_slots;
                        entry.has_mips = has_mips;
                        entry.linear_override = (pfilter == textures::Filter::Linear);
                        entry.pending = pending;
                        entry.last_try_frame = frame_number_;
                        pack_cache_[h] = entry;
                        pit = pack_cache_.find(h);
                    }
                    if (pit != pack_cache_.end() && pit->second.slot != 0) {
                        pack_slot = pit->second.slot;
                        if (!need_query) pack_streamer_.touch(pack_slot, frame_number_, &pit->second.pbr_slots);
                        setup_.tex_slot = pack_slot;
                        setup_.pbr_slots = pit->second.pbr_slots;
                        setup_.texture_mips = pit->second.has_mips;
                        setup_.linear_override = pit->second.linear_override;
                    }
                }
                if (pack_slot == 0) {
                    std::uint32_t kc_slot = 0;
                    std::uint32_t kc_ox = 0, kc_oy = 0;
                    const bool kc_candidate = setup_.through && info.format <= 3u &&
                                              ((info.stride >= 480u && info.width >= 32u && info.height >= 16u) ||
                                               (info.width >= 480u && info.height >= 256u));
                    if (kc_candidate) {
                        kc_slot = match_killcam_hires(info.address, info.stride, info.width, info.height, info.format,
                                                      info.swizzled, nullptr, kc_ox, kc_oy);
                    }
                    if (kc_slot != 0) {
                        setup_.tex_slot = kc_slot;
                        setup_.uv_xform[0] = static_cast<float>(info.width) / 480.0f;
                        setup_.uv_xform[1] = static_cast<float>(info.height) / 272.0f;
                        setup_.uv_xform[2] = static_cast<float>(kc_ox) / 480.0f;
                        setup_.uv_xform[3] = static_cast<float>(kc_oy) / 272.0f;
                        setup_.linear_override = true;
                        setup_.is_killcam = true;
                    } else {
                        auto it = textures_.find(h);
                        if (it == textures_.end()) {
                            if (textures_.size() >= kMaxCachedTextures) {
                                auto oldest = textures_.begin();
                                for (auto cur = textures_.begin(); cur != textures_.end(); ++cur) {
                                    if (cur->second.last_used < oldest->second.last_used) oldest = cur;
                                }
                                retire_image(oldest->second.image);
                                textures_.erase(oldest);
                            }
                            decode_buf_.resize(static_cast<std::size_t>(info.width) * info.height);
                            ge.decode_texture(decode_buf_.data());
                            if (kc_candidate) {
                                kc_slot = match_killcam_hires(info.address, info.stride, info.width, info.height, info.format,
                                                              info.swizzled, decode_buf_.data(), kc_ox, kc_oy);
                            }
                            if (kc_slot != 0) {
                                setup_.tex_slot = kc_slot;
                                setup_.uv_xform[0] = static_cast<float>(info.width) / 480.0f;
                                setup_.uv_xform[1] = static_cast<float>(info.height) / 272.0f;
                                setup_.uv_xform[2] = static_cast<float>(kc_ox) / 480.0f;
                                setup_.uv_xform[3] = static_cast<float>(kc_oy) / 272.0f;
                                setup_.linear_override = true;
                                setup_.is_killcam = true;
                            } else {
                                CachedTexture ct{};
                                ct.last_used = frame_number_;
                                upload_texture_2d(ct.image, info.width, info.height, decode_buf_.data());
                                it = textures_.emplace(h, std::move(ct)).first;
                                setup_.tex_slot = it->second.image.bindless_slot;
                            }
                        } else {
                            it->second.last_used = frame_number_;
                            setup_.tex_slot = it->second.image.bindless_slot;
                        }
                    }
                }
                FrameTexBinding bind{};
                bind.tex_slot = setup_.tex_slot;
                bind.pbr_slots = setup_.pbr_slots;
                bind.texture_mips = setup_.texture_mips;
                bind.linear_override = setup_.linear_override;
                bind.is_killcam = setup_.is_killcam;
                std::copy_n(setup_.uv_xform, 4, bind.uv_xform);
                frame_tex_cache_.emplace(param_key, bind);
            }
        }
    }
    const auto RS = [&](std::uint32_t reg) { return key_reg(setup_.key, reg); };
    const bool opaque_3d = !setup_.through && (RS(kClearMode) & 1u) == 0 && (RS(kZTestEnable) & 1u) != 0 &&
                           (RS(kAlphaBlendEnable) & 1u) == 0 && (RS(kZWriteDisable) & 1u) == 0 &&
                           is_shown_target(setup_.target);
    const bool want_rt = rt_shadows_.ready() && want_vk_effect(LiveSettings::kVkEffectRayTracing) &&
                         !lighting_.active_is_cave();
    setup_is_rt_caster_ = want_rt && opaque_3d;
    setup_is_shadow_caster_ = !want_rt && sun_shadows_.ready() && want_vk_effect(LiveSettings::kVkEffectSunShadows) &&
                              !lighting_.active_is_cave() && opaque_3d;
}

bool VkBackend::bind_ge_draw_state(VkCommandBuffer cmd, const DrawSetup &setup, std::uint32_t rt_w, std::uint32_t rt_h,
                                   bool is_soft_particle, bool msaa_4x, bool hdr_3d, float jitter_ndc_x,
                                   float jitter_ndc_y, float mip_lod_bias, VkDeviceAddress &out_params_addr,
                                   PassStateCache &cache) {
    FrameSlice &f = frames_[frame_slot_];
    auto pslice = f.params.allocate(sizeof(vk::DrawParams), 16);
    if (!pslice.ptr) return false;
    out_params_addr = pslice.address;

    const auto R = [&](std::uint32_t reg) { return key_reg(setup.key, reg); };
    const bool clear = (R(kClearMode) & 1u) != 0;
    const bool textured = !clear && (R(kTextureMapEnable) & 1u) != 0;
    const bool fog = !clear && !setup.through && (R(kFogEnable) & 1u) != 0;
    const bool alpha_test = !clear && (R(kAlphaTestEnable) & 1u) != 0;
    const std::uint32_t minz = R(kMinZ) & 0xFFFFu, maxz = R(kMaxZ) & 0xFFFFu;
    const bool zrange = !clear && !setup.through && maxz != 0 && (minz != 0 || maxz != 0xFFFFu);

    const bool linear = ((R(kTexFilter) & 0x100u) != 0) ||
                        (live_->smooth_textures.load(std::memory_order_relaxed) && !setup.through) ||
                        setup.linear_override;
    const bool clamp_s = setup.is_killcam || ((R(kTexWrap) & 1u) != 0);
    const bool clamp_t = setup.is_killcam || ((R(kTexWrap) & 0x100u) != 0);
    const int aniso = std::clamp(live_->texture_pack_anisotropy.load(std::memory_order_relaxed), 1, 16);

    vk::DrawParams dp{};
    std::copy_n(setup.uv_xform, 4, dp.uv_xform);
    unpack_rgb(R(kTexEnvColor), dp.env_and_flags);
    unpack_rgb(R(kFogColor), dp.fog_and_zmin);
    dp.fog_and_zmin[3] = static_cast<float>(minz);
    dp.target_size[0] = static_cast<float>(rt_w);
    dp.target_size[1] = static_cast<float>(rt_h);
    dp.zmax_pad[0] = static_cast<float>(maxz);
    dp.tex_idx = textured ? setup.tex_slot : 0u;
    dp.sampler_idx = vk::BindlessDescriptors::sampler_index(linear, setup.texture_mips, clamp_s, clamp_t, aniso);

    const std::uint32_t func = R(kTexFunc);
    dp.tex_func_bits = (func & 7u) | ((func & 0x100u) ? 8u : 0u) | ((func & 0x10000u) ? 16u : 0u) |
                       (clear ? 32u : 0u) | (textured ? 64u : 0u) | (fog ? 128u : 0u) | (zrange ? 256u : 0u) |
                       (setup.is_skinned ? 2048u : 0u);
    apply_draw_fx_bits(dp, is_soft_particle, setup.is_killcam, R(kBlendMode));
    dp.alpha_bits = alpha_test ? (R(kAlphaTest) & 0xFFFFFFu) : 1u;
    dp.jitter_and_lod[0] = jitter_ndc_x;
    dp.jitter_and_lod[1] = jitter_ndc_y;
    dp.jitter_and_lod[2] = mip_lod_bias;
    dp.jitter_and_lod[3] = 0.0f;
    if (textured && !clear && !setup.through && want_vk_effect(LiveSettings::kVkEffectPbrMaterials) &&
        (setup.pbr_slots.normal_slot != 0u || setup.pbr_slots.rough_metal_slot != 0u ||
         setup.pbr_slots.emissive_slot != 0u)) {
        dp.pbr_slots[0] = setup.pbr_slots.normal_slot;
        dp.pbr_slots[1] = setup.pbr_slots.rough_metal_slot;
        dp.pbr_slots[2] = setup.pbr_slots.emissive_slot;
        dp.pbr_slots[3] = 0u;
        float sw[3]{0.34f, 0.78f, 0.52f};
        const vk::LightingFrameInput &lin = has_cur_view_mat_ ? cur_light_in_ : last_light_in_;
        lighting_.resolve_sun_dir_world(lin, sw);
        if (lin.has_view_rot) {
            const float *r = lin.view_rot;
            dp.sun_dir_view[0] = r[0] * sw[0] + r[1] * sw[1] + r[2] * sw[2];
            dp.sun_dir_view[1] = r[3] * sw[0] + r[4] * sw[1] + r[5] * sw[2];
            dp.sun_dir_view[2] = r[6] * sw[0] + r[7] * sw[1] + r[8] * sw[2];
        } else {
            dp.sun_dir_view[0] = sw[0];
            dp.sun_dir_view[1] = sw[1];
            dp.sun_dir_view[2] = sw[2];
        }
        dp.sun_dir_view[3] = lighting_.active_is_cave() ? 0.0f : 1.0f;
    }
    std::memcpy(pslice.ptr, &dp, sizeof(dp));

    vk::GePipelineKey pkey{};
    pkey.spec_clear = clear ? 1u : 0u;
    pkey.spec_textured = textured ? 1u : 0u;
    pkey.spec_alpha_test = alpha_test ? 1u : 0u;
    pkey.spec_zrange = zrange ? 1u : 0u;
    pkey.msaa_4x = msaa_4x ? 1u : 0u;
    pkey.hdr_3d = hdr_3d ? 1u : 0u;

    VkBool32 blend_en = VK_FALSE;
    VkColorBlendEquationEXT blend_eq{VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD,
                                     VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD};
    float blend_const[4]{1.0f, 1.0f, 1.0f, 1.0f};
    VkColorComponentFlags write_mask = 0xFu;

    if (clear) {
        const std::uint32_t cm = R(kClearMode);
        const bool rgb = (cm & 0x100u) != 0, alpha = (cm & 0x200u) != 0;
        write_mask = (rgb ? (VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT) : 0u) |
                     (alpha ? VK_COLOR_COMPONENT_A_BIT : 0u);
    } else {
        if ((R(kAlphaBlendEnable) & 1u) != 0) {
            blend_en = VK_TRUE;
            const std::uint32_t mode = R(kBlendMode);
            bool const_a = false, const_b = false;
            const std::uint32_t fix_a = R(kBlendFixedA), fix_b = R(kBlendFixedB);
            VkBlendFactor src = psp_blend_factor(mode & 0xFu, true, fix_a, const_a);
            VkBlendFactor dst = psp_blend_factor((mode >> 4u) & 0xFu, false, fix_b, const_b);
            std::uint32_t constant = const_a ? fix_a : fix_b;
            if (const_a && const_b && (fix_a & 0xFFFFFFu) != (fix_b & 0xFFFFFFu)) {
                dst = VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
                constant = fix_a;
            }
            if (const_a || const_b) unpack_rgb(constant, blend_const);
            static constexpr VkBlendOp kOps[6] = {VK_BLEND_OP_ADD, VK_BLEND_OP_SUBTRACT, VK_BLEND_OP_REVERSE_SUBTRACT,
                                                  VK_BLEND_OP_MIN, VK_BLEND_OP_MAX, VK_BLEND_OP_MAX};
            blend_eq.srcColorBlendFactor = src;
            blend_eq.dstColorBlendFactor = dst;
            blend_eq.colorBlendOp = kOps[std::min<std::uint32_t>((mode >> 8u) & 0xFu, 5u)];
        }
        const std::uint32_t mask = R(kMaskRgb);
        const bool keep_alpha = (R(kStencilTestEnable) & 1u) != 0 || (R(kMaskAlpha) & 0xFFu) == 0xFFu;
        write_mask = ((mask & 0xFFu) != 0xFFu ? VK_COLOR_COMPONENT_R_BIT : 0u) |
                     (((mask >> 8u) & 0xFFu) != 0xFFu ? VK_COLOR_COMPONENT_G_BIT : 0u) |
                     (((mask >> 16u) & 0xFFu) != 0xFFu ? VK_COLOR_COMPONENT_B_BIT : 0u) |
                     (!keep_alpha ? VK_COLOR_COMPONENT_A_BIT : 0u);
    }

    pkey.blend_enable = blend_en ? 1u : 0u;
    pkey.color_op = static_cast<std::uint8_t>(blend_eq.colorBlendOp);
    pkey.src_factor = static_cast<std::uint8_t>(blend_eq.srcColorBlendFactor);
    pkey.dst_factor = static_cast<std::uint8_t>(blend_eq.dstColorBlendFactor);
    pkey.color_mask = static_cast<std::uint8_t>(write_mask);

    VkPipeline pipe = pipelines_.get_ge_pipeline(ctx_, pkey);
    if (!pipe) return false;
    if (pipe != cache.pipe) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
        cache.pipe = pipe;
    }

    const std::uint32_t sx1 = R(kScissor1) & 0x3FFu, sy1 = (R(kScissor1) >> 10u) & 0x3FFu;
    const std::uint32_t sx2 = std::min(R(kScissor2) & 0x3FFu, rt_w - 1u);
    const std::uint32_t sy2 = std::min((R(kScissor2) >> 10u) & 0x3FFu, rt_h - 1u);
    VkRect2D scissor{{0, 0}, {0, 0}};
    if (sx1 <= sx2 && sy1 <= sy2) {
        scissor.offset = {static_cast<std::int32_t>(sx1) * scale_, static_cast<std::int32_t>(sy1) * scale_};
        scissor.extent = {(sx2 - sx1 + 1u) * static_cast<std::uint32_t>(scale_),
                          (sy2 - sy1 + 1u) * static_cast<std::uint32_t>(scale_)};
    }
    if (!cache.has_scissor || scissor.offset.x != cache.scissor.offset.x || scissor.offset.y != cache.scissor.offset.y ||
        scissor.extent.width != cache.scissor.extent.width || scissor.extent.height != cache.scissor.extent.height) {
        vkCmdSetScissor(cmd, 0, 1, &scissor);
        cache.scissor = scissor;
        cache.has_scissor = true;
    }

    VkBool32 depth_test = VK_FALSE;
    VkBool32 depth_write = VK_FALSE;
    VkCompareOp depth_op = VK_COMPARE_OP_ALWAYS;
    if (clear) {
        depth_test = VK_TRUE;
        depth_op = VK_COMPARE_OP_ALWAYS;
        depth_write = (R(kClearMode) & 0x400u) != 0 ? VK_TRUE : VK_FALSE;
    } else if ((R(kZTestEnable) & 1u) != 0) {
        static constexpr VkCompareOp kOps[8] = {VK_COMPARE_OP_NEVER, VK_COMPARE_OP_ALWAYS, VK_COMPARE_OP_EQUAL,
                                                VK_COMPARE_OP_NOT_EQUAL, VK_COMPARE_OP_LESS, VK_COMPARE_OP_LESS_OR_EQUAL,
                                                VK_COMPARE_OP_GREATER, VK_COMPARE_OP_GREATER_OR_EQUAL};
        depth_test = VK_TRUE;
        depth_op = kOps[R(kZTest) & 7u];
        depth_write = (R(kZWriteDisable) & 1u) == 0 ? VK_TRUE : VK_FALSE;
    }
    if (!cache.has_depth || depth_test != cache.depth_test) {
        vkCmdSetDepthTestEnable(cmd, depth_test);
        cache.depth_test = depth_test;
    }
    if (!cache.has_depth || depth_op != cache.depth_op) {
        vkCmdSetDepthCompareOp(cmd, depth_op);
        cache.depth_op = depth_op;
    }
    if (!cache.has_depth || depth_write != cache.depth_write) {
        vkCmdSetDepthWriteEnable(cmd, depth_write);
        cache.depth_write = depth_write;
    }
    cache.has_depth = true;

    if (!cache.has_blend_const || std::memcmp(blend_const, cache.blend_const, sizeof(blend_const)) != 0) {
        vkCmdSetBlendConstants(cmd, blend_const);
        std::memcpy(cache.blend_const, blend_const, sizeof(blend_const));
        cache.has_blend_const = true;
    }
    if (ctx_.caps.extended_dynamic_state3) {
        if (!cache.has_eds3 || blend_en != cache.blend_en) {
            vkCmdSetColorBlendEnableEXT(cmd, 0, 1, &blend_en);
            cache.blend_en = blend_en;
        }
        if (!cache.has_eds3 || std::memcmp(&blend_eq, &cache.blend_eq, sizeof(blend_eq)) != 0) {
            vkCmdSetColorBlendEquationEXT(cmd, 0, 1, &blend_eq);
            cache.blend_eq = blend_eq;
        }
        if (!cache.has_eds3 || write_mask != cache.write_mask) {
            vkCmdSetColorWriteMaskEXT(cmd, 0, 1, &write_mask);
            cache.write_mask = write_mask;
        }
        cache.has_eds3 = true;
    }
    return true;
}

void VkBackend::draw_batch() {
    if (batch_.empty()) return;
    if (prof_.enabled) ++prof_.batches;
    ScopedTimer prof_scope(prof_.enabled, prof_.batch_ns);
    auto rt_it = targets_.find(setup_.target);
    if (rt_it == targets_.end()) {
        batch_.clear();
        return;
    }
    RenderTarget &rt = rt_it->second;
    const auto R = [&](std::uint32_t reg) { return key_reg(setup_.key, reg); };
    const bool clear = (R(kClearMode) & 1u) != 0;

    const bool main_rt_skipped = (subframes_ > 1 && interp_can_blend_);
    const int aa = effective_aa_mode();

    if (!clear && is_shown_target(setup_.target)) {
        if (!setup_.through) {
            frame_had_3d_ = true;
            if (!main_rt_skipped && aa == LiveSettings::kAaTaa && !main_taa_jitter_set_) {
                float jx = 0.0f, jy = 0.0f;
                get_taa_jitter(taa_sample_idx_++, jx, jy);
                const std::uint32_t sw = std::min(480u, rt.width) * static_cast<std::uint32_t>(scale_);
                const std::uint32_t sh = std::min(272u, rt.height) * static_cast<std::uint32_t>(scale_);
                main_taa_jitter_ndc_[0] = jx * 2.0f / static_cast<float>(rt.color.width);
                main_taa_jitter_ndc_[1] = jy * 2.0f / static_cast<float>(rt.color.height);
                main_taa_jitter_uv_[0] = jx / static_cast<float>(std::max(sw, 1u));
                main_taa_jitter_uv_[1] = jy / static_cast<float>(std::max(sh, 1u));
                main_taa_jitter_set_ = true;
            }
        } else if (frame_had_3d_ && !setup_.self_sampled) {
            // Skip redundant 5th post-3D pass on main_rt when frame-gen is actively presenting interp_rt_!
            if (!lit_3d_this_frame_ && !main_rt_skipped) {
                const bool had_hdr = main_hdr_started_;
                end_render_pass();
                std::array<vk::ViewPointLight, vk::LightingFrameInput::kMaxViewLights> vlights{};
                if (has_cur_view_mat_ && subframes_ <= 1) {
                    if (rt_shadows_.ready() && want_vk_effect(LiveSettings::kVkEffectRayTracing) && !lighting_.active_is_cave()) {
                        ensure_recording();
                        end_render_pass();
                        rt_shadows_.update_bvh(frames_[frame_slot_].cmd, ctx_, cur_rt_static_verts_, cur_rt_dyn_verts_);
                        sun_shadows_.invalidate();
                    } else {
                        rt_shadows_.invalidate();
                        render_sun_shadow_pass(cur_view_mat_, cur_light_in_, cur_shadow_verts_, cur_shadow_batches_);
                    }
                }
                const std::uint32_t vl_count = has_cur_view_mat_
                                                   ? build_view_lights(cur_view_mat_, cur_vfx_lights_, vlights)
                                                   : 0u;
                apply_post_3d(rt, cur_light_in_, nullptr, false, false, main_taa_jitter_uv_, had_hdr,
                              vlights.data(), vl_count, has_cur_view_mat_ ? &cur_view_mat_ : nullptr);
                lit_3d_this_frame_ = true;
            }
            main_3d_finished_ = true;
            if (!captured_clean_3d_) {
                capture_clean_3d_snapshot(rt);
            }
        }
    }

    const bool is_soft_particle = !clear && !setup_.through && is_shown_target(setup_.target) &&
                                  (R(kAlphaBlendEnable) & 1u) != 0 && (R(kZTestEnable) & 1u) != 0 &&
                                  (R(kZWriteDisable) & 1u) != 0 &&
                                  want_vk_effect(LiveSettings::kVkEffectSoftParticles);
    if (is_soft_particle && !main_depth_snapped_) {
        snapshot_soft_depth(rt);
        main_depth_snapped_ = true;
    }

    const bool use_main_msaa = !main_rt_skipped && (aa == LiveSettings::kAaMsaa4x) &&
                               is_shown_target(setup_.target) && !main_3d_finished_ &&
                               (!setup_.through || clear);
    const bool use_main_hdr = !main_rt_skipped && want_enhanced_3d_hdr() &&
                              is_shown_target(setup_.target) && !main_3d_finished_ &&
                              (!setup_.through || clear);
    begin_target_pass(rt, use_main_msaa, use_main_hdr);

    // Quest HUD detection, HUD scaling, and widescreen 2D aspect preservation (single tight pass).
    float hud_scale = 1.0f, hud_ax = 0.0f, hud_ay = 0.0f;
    float wide_scale = 1.0f, wide_ax = 240.0f;
    if (setup_.through && !clear && !setup_.self_sampled && is_shown_target(setup_.target)) {
        float x1 = 1e9f, y1 = 1e9f, x2 = -1e9f, y2 = -1e9f;
        for (const GpuVertex &v : batch_) {
            x1 = std::min(x1, v.x);
            y1 = std::min(y1, v.y);
            x2 = std::max(x2, v.x);
            y2 = std::max(y2, v.y);
        }
        const bool whole_screen = setup_.is_killcam || (x2 - x1 >= 448.0f && y2 - y1 >= 16.0f) || (y2 - y1 >= 256.0f);
        if (!whole_screen) {
            constexpr float kHudClock[4] = {8.0f, 8.0f, 60.0f, 56.0f};
            const bool is_clock = std::fabs(x1 - kHudClock[0]) < 1.5f && std::fabs(y1 - kHudClock[1]) < 1.5f &&
                                  x2 >= kHudClock[2] - 1.5f && y2 >= kHudClock[3] - 1.5f && y2 <= 160.0f;
            if (hud_layer_ == gpu::DrawInfo::kNoLayer && is_clock) {
                hud_layer_ = cur_layer_;
                quest_hud_this_flip_ = true;
            }
            const int hud_pct = std::clamp(live_->hud_size.load(std::memory_order_relaxed), 50, 100);
            if (hud_layer_ != gpu::DrawInfo::kNoLayer) {
                if (cur_layer_ == hud_layer_) {
                    hud_scale = static_cast<float>(hud_pct) / 100.0f;
                    hud_ax = 0.0f;
                    hud_ay = 0.0f;
                    if (frame_stretch_ > 1.0f) { wide_scale = 1.0f / frame_stretch_; wide_ax = 0.0f; }
                } else if (cur_layer_ == hud_layer_ + 1u && x1 >= 240.0f) {
                    hud_scale = static_cast<float>(hud_pct) / 100.0f;
                    hud_ax = 480.0f;
                    hud_ay = y1 < 180.0f ? 0.0f : 272.0f;
                    if (frame_stretch_ > 1.0f) { wide_scale = 1.0f / frame_stretch_; wide_ax = 480.0f; }
                } else if (frame_stretch_ > 1.0f) {
                    wide_scale = 1.0f / frame_stretch_;
                    wide_ax = 240.0f;
                }
            } else if (frame_stretch_ > 1.0f) {
                wide_scale = 1.0f / frame_stretch_;
                wide_ax = 240.0f;
            }
            if (hud_scale != 1.0f || wide_scale != 1.0f) {
                for (GpuVertex &v : batch_) {
                    const float hx = hud_ax + (v.x - hud_ax) * hud_scale;
                    const float hy = hud_ay + (v.y - hud_ay) * hud_scale;
                    v.x = wide_ax + (hx - wide_ax) * wide_scale;
                    v.y = hy;
                }
            }
        }
    }

    FrameSlice &f = frames_[frame_slot_];
    const std::size_t vbytes = batch_.size() * sizeof(GpuVertex);
    auto vslice = f.vertices.allocate(vbytes, 16);
    if (!vslice.ptr) {
        batch_.clear();
        return;
    }
    std::memcpy(vslice.ptr, batch_.data(), vbytes);

    VkCommandBuffer cmd = f.cmd;
    VkDeviceAddress params_addr = 0;
    const bool apply_main_taa_jitter = main_taa_jitter_set_ && !clear && !setup_.through && is_shown_target(setup_.target);
    if (!bind_ge_draw_state(cmd, setup_, rt.width, rt.height, is_soft_particle, active_pass_msaa_, active_pass_hdr_,
                            apply_main_taa_jitter ? main_taa_jitter_ndc_[0] : 0.0f,
                            apply_main_taa_jitter ? main_taa_jitter_ndc_[1] : 0.0f,
                            apply_main_taa_jitter ? -0.5f : 0.0f,
                            params_addr, main_pass_state_)) {
        batch_.clear();
        return;
    }

    const VkDeviceSize voff = vslice.offset;
    vkCmdBindVertexBuffers(cmd, 0, 1, &f.vertices.buffer, &voff);

    vk::DrawPush push{params_addr, 0u};
    vkCmdPushConstants(cmd, pipelines_.ge_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(push), &push);
    vkCmdDraw(cmd, static_cast<std::uint32_t>(batch_.size()), 1, 0, 0);

    if (subframes_ > 1) {
        if (setup_.self_sampled) cur_snap_.unblendable = true;
        RecordedBatch rb{};
        rb.setup = setup_;
        const std::uint32_t raw_cnt = static_cast<std::uint32_t>(cur_snap_.raw_verts.size()) - cur_raw_batch_start_;
        if (cur_draw_eligible_ && !clear && !setup_.through && raw_cnt > 0) {
            rb.use_gpu_raw = true;
            rb.v_offset = cur_raw_batch_start_;
            rb.v_count = raw_cnt;
        } else {
            rb.use_gpu_raw = false;
            rb.v_offset = static_cast<std::uint32_t>(cur_snap_.flat_verts.size());
            rb.v_count = static_cast<std::uint32_t>(batch_.size());
            cur_snap_.flat_verts.insert(cur_snap_.flat_verts.end(), batch_.begin(), batch_.end());
        }
        cur_raw_batch_start_ = static_cast<std::uint32_t>(cur_snap_.raw_verts.size());
        cur_snap_.batches.push_back(rb);
    }

    if (cur_shadow_verts_.size() > cur_shadow_batch_start_) {
        vk::ShadowBatch sb{};
        sb.v_offset = cur_shadow_batch_start_;
        sb.v_count = static_cast<std::uint32_t>(cur_shadow_verts_.size()) - cur_shadow_batch_start_;
        const bool textured = (R(kTextureMapEnable) & 1u) != 0;
        const bool alpha_test = (R(kAlphaTestEnable) & 1u) != 0;
        const bool clamp_s = (R(kTexWrap) & 1u) != 0;
        const bool clamp_t = (R(kTexWrap) & 0x100u) != 0;
        sb.tex_slot = textured ? setup_.tex_slot : 0u;
        sb.sampler_idx = vk::BindlessDescriptors::sampler_index(false, false, clamp_s, clamp_t, 1);
        sb.alpha_bits = alpha_test ? (R(kAlphaTest) & 0xFFFFFFu) : 1u;
        sb.textured = textured ? 1u : 0u;
        std::copy_n(setup_.uv_xform, 4, sb.uv_xform);
        cur_shadow_batches_.push_back(sb);
        cur_shadow_batch_start_ = static_cast<std::uint32_t>(cur_shadow_verts_.size());
    }

    if (setup_.is_additive_vfx && batch_world_cnt_ >= 12u && is_shown_target(setup_.target) &&
        cur_vfx_lights_.size() < 32u) {
        const float inv_n = 1.0f / static_cast<float>(batch_world_cnt_);
        const float cx = batch_world_sum_[0] * inv_n;
        const float cy = batch_world_sum_[1] * inv_n;
        const float cz = batch_world_sum_[2] * inv_n;
        float cr = std::clamp(batch_col_sum_[0] * inv_n / 255.0f, 0.0f, 1.0f);
        float cg = std::clamp(batch_col_sum_[1] * inv_n / 255.0f, 0.0f, 1.0f);
        float cb = std::clamp(batch_col_sum_[2] * inv_n / 255.0f, 0.0f, 1.0f);
        const float cmax = std::max({cr, cg, cb});
        const float cmin = std::min({cr, cg, cb});
        const float sat = cmax > 1e-4f ? (cmax - cmin) / cmax : 0.0f;
        // Require either vivid chromatic VFX or a multi-particle bright burst (catches white-vertex-color fireballs
        // while ignoring tiny 2-triangle NPC speech bubbles / floating indicators!).
        if ((cmax > 0.32f && sat > 0.16f) || (cmax > 0.45f && batch_world_cnt_ >= 24u)) {
            if (sat < 0.10f) {
                // Neutral-white additive bursts (e.g. textured fireballs/explosions) get a natural warm amber tint.
                cr = 1.00f; cg = 0.72f; cb = 0.42f;
            }
            const float intensity = std::clamp(0.30f + cmax * 0.28f + sat * 0.20f, 0.30f, 0.72f);
            const float radius = 250.0f + std::min<float>(static_cast<float>(batch_world_cnt_) * 1.0f, 200.0f);
            bool merged = false;
            for (auto &ex : cur_vfx_lights_) {
                const float dx = ex.world_pos[0] - cx;
                const float dy = ex.world_pos[1] - cy;
                const float dz = ex.world_pos[2] - cz;
                if (dx * dx + dy * dy + dz * dz < 160.0f * 160.0f) {
                    ex.world_pos[0] = 0.5f * (ex.world_pos[0] + cx);
                    ex.world_pos[1] = 0.5f * (ex.world_pos[1] + cy);
                    ex.world_pos[2] = 0.5f * (ex.world_pos[2] + cz);
                    ex.color[0] = std::max(ex.color[0], cr);
                    ex.color[1] = std::max(ex.color[1], cg);
                    ex.color[2] = std::max(ex.color[2], cb);
                    ex.intensity = std::min(1.30f, std::max(ex.intensity, intensity) + 0.08f);
                    ex.radius = std::max(ex.radius, radius);
                    merged = true;
                    break;
                }
            }
            if (!merged) {
                SemanticPointLight pl{};
                pl.world_pos[0] = cx;
                pl.world_pos[1] = cy;
                pl.world_pos[2] = cz;
                pl.radius = radius;
                pl.color[0] = cr;
                pl.color[1] = cg;
                pl.color[2] = cb;
                pl.intensity = intensity;
                cur_vfx_lights_.push_back(pl);
            }
        }
    }
    batch_world_sum_[0] = batch_world_sum_[1] = batch_world_sum_[2] = 0.0f;
    batch_col_sum_[0] = batch_col_sum_[1] = batch_col_sum_[2] = 0.0f;
    batch_world_cnt_ = 0u;
    batch_.clear();
}

void VkBackend::draw_started(const gpu::DrawInfo &draw) {
    if (prof_.enabled) ++prof_.draws;
    ScopedTimer prof_scope(prof_.enabled, prof_.draw_start_ns);
    cur_layer_ = draw.layer;
    const auto sum = (!draw.through && !draw.clear) || subframes_ > 1 ? interpolation::summarize(draw)
                                                                      : interpolation::DrawSummary{};
    if (!draw.through && !draw.clear && sum.perspective &&
        std::fabs(draw.viewport[0]) >= 100.0f && std::fabs(draw.viewport[1]) >= 50.0f &&
        cur_light_in_.proj[0] == 0.0f) {
        std::copy_n(sum.projection.data(), 16, cur_light_in_.proj);
        cur_light_in_.vz_scale = draw.viewport[2];
        cur_light_in_.vz_center = draw.viewport[5];
        cur_view_mat_ = sum.view;
        has_cur_view_mat_ = true;
        // Capture orthonormalized 3x3 world -> view rotation from the first terrain/camera draw so sun direction,
        // contact shadows, and sky/ground hemisphere bounce stay 100% world-locked as the camera rotates.
        const auto eye = interpolation::multiply(sum.view, sum.world);
        float c0[3]{eye[0], eye[1], eye[2]};
        float c1[3]{eye[4], eye[5], eye[6]};
        float c2[3]{eye[8], eye[9], eye[10]};
        auto norm3 = [](float v[3]) {
            const float l = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            if (l > 1e-4f) { v[0] /= l; v[1] /= l; v[2] /= l; return true; }
            return false;
        };
        if (norm3(c0) && norm3(c1) && norm3(c2)) {
            cur_light_in_.view_rot[0] = c0[0]; cur_light_in_.view_rot[1] = c1[0]; cur_light_in_.view_rot[2] = c2[0];
            cur_light_in_.view_rot[3] = c0[1]; cur_light_in_.view_rot[4] = c1[1]; cur_light_in_.view_rot[5] = c2[1];
            cur_light_in_.view_rot[6] = c0[2]; cur_light_in_.view_rot[7] = c1[2]; cur_light_in_.view_rot[8] = c2[2];
            cur_light_in_.has_view_rot = true;
        }
    }
    if (subframes_ <= 1) return;

    const bool next_eligible = !draw.through && !draw.clear && sum.perspective;
    // Only flush batch_ on eligibility transitions (keeps full draw-call batching across identical-state draws!),
    // and discard any orphaned raw_verts from fully frustum-clipped draws when batch_ is empty.
    if (next_eligible != cur_draw_eligible_) {
        draw_batch();
        cur_raw_batch_start_ = static_cast<std::uint32_t>(cur_snap_.raw_verts.size());
    } else if (batch_.empty()) {
        cur_raw_batch_start_ = static_cast<std::uint32_t>(cur_snap_.raw_verts.size());
    }
    cur_draw_index_ = static_cast<std::uint32_t>(cur_snap_.summaries.size());
    cur_snap_.summaries.push_back(sum);
    cur_snap_.viewports.push_back(draw.viewport);
    cur_snap_.draw_raw_start.push_back(static_cast<std::uint32_t>(cur_snap_.raw_verts.size()));
    cur_snap_.draw_raw_count.push_back(0u);
    cur_draw_eligible_ = next_eligible;
}

void VkBackend::triangle_unclipped(const gpu::Ge &ge, const gpu::Vertex &a, const gpu::Vertex &b, const gpu::Vertex &c) {
    if (subframes_ <= 1 || !cur_draw_eligible_ || cur_snap_.summaries.empty()) return;
    apply_state(ge);
    auto &sum = cur_snap_.summaries.back();
    auto push_raw = [&](const gpu::Vertex &v) {
        if (v.has_world) {
            sum.world_sum[0] += v.world[0];
            sum.world_sum[1] += v.world[1];
            sum.world_sum[2] += v.world[2];
            ++sum.world_points;
        }
        vk::RawGpuVertex rv{};
        rv.world_pos[0] = v.world[0];
        rv.world_pos[1] = v.world[1];
        rv.world_pos[2] = v.world[2];
        rv.draw_idx = cur_draw_index_;
        rv.uv[0] = v.u;
        rv.uv[1] = v.v;
        const std::uint32_t r = static_cast<std::uint32_t>(std::clamp(v.r, 0.0f, 255.0f));
        const std::uint32_t g = static_cast<std::uint32_t>(std::clamp(v.g, 0.0f, 255.0f));
        const std::uint32_t bl = static_cast<std::uint32_t>(std::clamp(v.b, 0.0f, 255.0f));
        const std::uint32_t al = static_cast<std::uint32_t>(std::clamp(v.a, 0.0f, 255.0f));
        rv.color_rgba = r | (g << 8u) | (bl << 16u) | (al << 24u);
        const std::uint32_t sr = static_cast<std::uint32_t>(std::clamp(v.sr, 0.0f, 255.0f));
        const std::uint32_t sg = static_cast<std::uint32_t>(std::clamp(v.sg, 0.0f, 255.0f));
        const std::uint32_t sb = static_cast<std::uint32_t>(std::clamp(v.sb, 0.0f, 255.0f));
        rv.spec_rgba = sr | (sg << 8u) | (sb << 16u) | (255u << 24u);
        rv.fog = v.fog;
        if (v.has_normal) {
            rv.world_normal[0] = v.normal[0];
            rv.world_normal[1] = v.normal[1];
            rv.world_normal[2] = v.normal[2];
        }
        cur_snap_.raw_verts.push_back(rv);
    };
    push_raw(a);
    push_raw(b);
    push_raw(c);
    cur_snap_.draw_raw_count.back() += 3u;
}

void VkBackend::triangle(const gpu::Ge &ge, const gpu::Vertex &a, const gpu::Vertex &b, const gpu::Vertex &c) {
    apply_state(ge);
    auto push_v = [&](const gpu::Vertex &v) {
        GpuVertex g{};
        g.x = v.x;
        g.y = v.y;
        g.z = v.z;
        g.inv_w = setup_.through ? 1.0f : v.w;
        g.u = setup_.through ? v.u / setup_.tex_width : v.u;
        g.v = setup_.through ? v.v / setup_.tex_height : v.v;
        g.color[0] = static_cast<std::uint8_t>(std::clamp(v.r, 0.0f, 255.0f));
        g.color[1] = static_cast<std::uint8_t>(std::clamp(v.g, 0.0f, 255.0f));
        g.color[2] = static_cast<std::uint8_t>(std::clamp(v.b, 0.0f, 255.0f));
        g.color[3] = static_cast<std::uint8_t>(std::clamp(v.a, 0.0f, 255.0f));
        g.spec[0] = static_cast<std::uint8_t>(std::clamp(v.sr, 0.0f, 255.0f));
        g.spec[1] = static_cast<std::uint8_t>(std::clamp(v.sg, 0.0f, 255.0f));
        g.spec[2] = static_cast<std::uint8_t>(std::clamp(v.sb, 0.0f, 255.0f));
        g.spec[3] = 255u;
        g.fog = v.fog;
        g.normal_oct = 0u;
        if (!setup_.through && v.has_normal) {
            const auto &vm = ge.view_matrix().m;
            const float vnx = vm[0] * v.normal[0] + vm[3] * v.normal[1] + vm[6] * v.normal[2];
            const float vny = vm[1] * v.normal[0] + vm[4] * v.normal[1] + vm[7] * v.normal[2];
            const float vnz = vm[2] * v.normal[0] + vm[5] * v.normal[1] + vm[8] * v.normal[2];
            g.normal_oct = pack_oct_normal(vnx, vny, vnz);
        }
        if (setup_.is_additive_vfx && v.has_world) {
            batch_world_sum_[0] += v.world[0];
            batch_world_sum_[1] += v.world[1];
            batch_world_sum_[2] += v.world[2];
            batch_col_sum_[0] += std::clamp(v.r, 0.0f, 255.0f);
            batch_col_sum_[1] += std::clamp(v.g, 0.0f, 255.0f);
            batch_col_sum_[2] += std::clamp(v.b, 0.0f, 255.0f);
            ++batch_world_cnt_;
        }
        if (setup_is_shadow_caster_ && v.has_world && cur_shadow_verts_.size() < 180000u) {
            cur_shadow_verts_.push_back({v.world[0], v.world[1], v.world[2], v.u, v.v});
        }
        if (setup_is_rt_caster_ && v.has_world) {
            auto &dst = setup_.is_skinned ? cur_rt_dyn_verts_ : cur_rt_static_verts_;
            if (dst.size() < 150000u) dst.push_back({v.world[0], v.world[1], v.world[2]});
        }
        batch_.push_back(g);
    };
    push_v(a);
    push_v(b);
    push_v(c);
}

void VkBackend::sprite(const gpu::Ge &ge, const gpu::Vertex &a, const gpu::Vertex &b) {
    gpu::Vertex v0 = a, v1 = b, v2 = b, v3 = b;
    v0.z = b.z; v0.w = b.w; v0.r = b.r; v0.g = b.g; v0.b = b.b; v0.a = b.a;
    v0.sr = b.sr; v0.sg = b.sg; v0.sb = b.sb; v0.fog = b.fog;
    v1.x = b.x; v1.y = a.y; v1.u = b.u; v1.v = a.v;
    v2.x = a.x; v2.y = b.y; v2.u = a.u; v2.v = b.v;
    triangle(ge, v0, v1, v3);
    triangle(ge, v0, v3, v2);
}

void VkBackend::block_transfer(std::uint32_t src, std::uint32_t /*src_stride*/, std::uint32_t sx, std::uint32_t sy,
                               std::uint32_t dst, std::uint32_t dst_stride, std::uint32_t dx, std::uint32_t dy,
                               std::uint32_t width, std::uint32_t height, std::uint32_t bytes_per_pixel) {
    draw_batch();
    frame_tex_cache_.clear();
    have_state_ = false;
    last_state_gen_ = 0;
    if (subframes_ > 1) cur_snap_.unblendable = true;
    const std::uint32_t src_vram = src & 0x1FFFFFFFu;
    const std::uint32_t dst_vram = dst & 0x1FFFFFFFu;
    const bool src_in_vram = src_vram >= 0x04000000u && src_vram < 0x04800000u;
    const bool dst_in_vram = dst_vram >= 0x04000000u && dst_vram < 0x04800000u;
    if (!src_in_vram) return;

    const std::uint32_t src_off = src_vram & 0x001FFFFFu;
    auto sit = targets_.find(src_off);
    if (sit == targets_.end()) {
        // Also check if `src_off` lies inside an existing render target (sub-rect transfer).
        for (auto it = targets_.begin(); it != targets_.end(); ++it) {
            const std::uint32_t span = it->second.stride * it->second.height * it->second.bytes_per_pixel();
            if (src_off >= it->first && src_off < it->first + span) {
                sit = it;
                break;
            }
        }
    }
    if (sit == targets_.end()) return;

    if (!dst_in_vram) {
        // VRAM -> RAM readback (used by quest-end killcam capture when the game copies out of VRAM).
        // Games often transfer a 16-bit framebuffer using 32-bit GE block_transfer (bpp=4, half width/stride),
        // so normalize coordinates into the render target's native pixel units.
        const std::uint32_t rt_bpp = sit->second.bytes_per_pixel();
        if (rt_bpp == bytes_per_pixel) {
            copy_target_to_ram(sit->second, src, sx, sy, dst, dst_stride, dx, dy, width, height);
        } else if (rt_bpp != 0u) {
            const std::uint32_t px_sx = (sx * bytes_per_pixel) / rt_bpp;
            const std::uint32_t px_w = (width * bytes_per_pixel) / rt_bpp;
            const std::uint32_t px_dx = (dx * bytes_per_pixel) / rt_bpp;
            const std::uint32_t px_dst_stride = (dst_stride * bytes_per_pixel) / rt_bpp;
            copy_target_to_ram(sit->second, src, px_sx, sy, dst, px_dst_stride, px_dx, dy, px_w, height);
        }
        return;
    }

    const std::uint32_t dst_off = dst_vram & 0x001FFFFFu;
    const std::uint32_t fmt = bytes_per_pixel == 4u ? 3u : sit->second.format;
    RenderTarget &dst_rt = ensure_target(dst_off, dst_stride > 0 ? dst_stride : sit->second.stride, fmt, dy + height);
    RenderTarget &src_rt = sit->second;

    ensure_recording();
    end_render_pass();
    transition_resource(src_rt.color, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    transition_resource(dst_rt.color, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

    VkImageBlit region{};
    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.srcOffsets[0] = {static_cast<std::int32_t>(sx) * scale_, static_cast<std::int32_t>(sy) * scale_, 0};
    region.srcOffsets[1] = {static_cast<std::int32_t>(sx + width) * scale_, static_cast<std::int32_t>(sy + height) * scale_, 1};
    region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.dstOffsets[0] = {static_cast<std::int32_t>(dx) * scale_, static_cast<std::int32_t>(dy) * scale_, 0};
    region.dstOffsets[1] = {static_cast<std::int32_t>(dx + width) * scale_, static_cast<std::int32_t>(dy + height) * scale_, 1};
    vkCmdBlitImage(frames_[frame_slot_].cmd, src_rt.color.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   dst_rt.color.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region, VK_FILTER_NEAREST);
    dst_rt.last_draw = frame_number_;
}

void VkBackend::capture_clean_3d_snapshot(RenderTarget &rt) {
    if (rt.color.image == VK_NULL_HANDLE || rt.width < 480u || rt.height < 272u) return;
    const std::uint32_t cw = 480u * static_cast<std::uint32_t>(scale_);
    const std::uint32_t ch = 272u * static_cast<std::uint32_t>(scale_);
    if (cw == 0 || ch == 0 || cw > rt.color.width || ch > rt.color.height) return;

    ensure_recording();
    end_render_pass();

    if (killcam_clean_.image == VK_NULL_HANDLE || killcam_clean_.width != cw || killcam_clean_.height != ch) {
        retire_image(killcam_clean_);
        killcam_clean_valid_ = false;
        killcam_clean_target_ = UINT32_MAX;
        killcam_clean_.create(ctx_, &descriptors_, cw, ch, VK_FORMAT_R8G8B8A8_UNORM,
                              VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                              VK_IMAGE_ASPECT_COLOR_BIT);
    }
    if (killcam_clean_.image == VK_NULL_HANDLE) return;

    transition_resource(rt.color, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    transition_resource(killcam_clean_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

    VkImageCopy cp{};
    cp.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    cp.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    cp.extent = {cw, ch, 1};
    vkCmdCopyImage(frames_[frame_slot_].cmd, rt.color.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   killcam_clean_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);
    killcam_clean_target_ = rt.address;
    killcam_clean_valid_ = true;
    captured_clean_3d_ = true;
}

void VkBackend::copy_shown_to_vram(RenderTarget *shown) {
    constexpr std::uint32_t kW = 480u, kH = 272u;

    // 1. Harvest previous async readback if the GPU has finished (zero stalls).
    if (pending_vram_rb_.active && ctx_.query_timeline(ctx_.graphics_timeline) >= pending_vram_rb_.timeline_value) {
        const PendingVramReadback p = pending_vram_rb_;
        pending_vram_rb_.active = false;
        if (vram_rb_mapped_ != nullptr) {
            vmaInvalidateAllocation(ctx_.allocator, vram_rb_alloc_, 0, VK_WHOLE_SIZE);
            const auto *rgba = static_cast<const std::uint8_t *>(vram_rb_mapped_);
            const std::uint32_t bpp = p.format == 3u ? 4u : 2u;
            for (std::uint32_t y = 0; y < p.height; ++y) {
                std::uint8_t *row = memory_.raw_pointer(0x04000000u + p.address + y * p.stride * bpp, p.width * bpp);
                if (!row) continue;
                const std::uint8_t *line = rgba + static_cast<std::size_t>(y) * kW * 4u;
                if (p.format == 3u) {
                    std::memcpy(row, line, p.width * 4u);
                } else {
                    for (std::uint32_t x = 0; x < p.width; ++x) pack_pixel(line + x * 4u, p.format, row + x * bpp);
                }
            }
            if (p.has_killcam_hires && killcam_staging_.image != VK_NULL_HANDLE) {
                // Only promote bright frames into the rolling ring; dark fade-to-black / loading frames
                // leave all 12 bright 3D snapshots completely intact!
                std::uint64_t luma_sum = 0;
                for (std::uint32_t sy = 16; sy < kH; sy += 32) {
                    for (std::uint32_t sx = 16; sx < kW; sx += 32) {
                        const std::uint8_t *px = rgba + (static_cast<std::size_t>(sy) * kW + sx) * 4u;
                        luma_sum += static_cast<std::uint64_t>(px[0]) + px[1] + px[2];
                    }
                }
                const std::uint64_t avg_ch = luma_sum / (8u * 15u * 3u);
                if (avg_ch >= 18u) {
                    auto &ks = killcam_slots_[killcam_next_slot_];
                    killcam_next_slot_ = (killcam_next_slot_ + 1u) % kKillcamSlots;
                    std::swap(ks.image, killcam_staging_);
                    ks.thumb.resize(static_cast<std::size_t>(kW) * kH);
                    std::memcpy(ks.thumb.data(), rgba, static_cast<std::size_t>(kW) * kH * 4u);
                    ks.valid = true;
                    ks.frame = frame_number_;
                }
            }
        }
    }

    // 2. Schedule a new async 480x272 readback at most every 4th flip per render target.
    if (pending_vram_rb_.active || !shown || shown->color.image == VK_NULL_HANDLE) return;
    if (shown->last_vram_copy != 0 && frame_number_ - shown->last_vram_copy < 4u) return;
    shown->last_vram_copy = frame_number_;

    if (vram_rb_img_.image == VK_NULL_HANDLE) {
        vram_rb_img_.create(ctx_, nullptr, kW, kH, VK_FORMAT_R8G8B8A8_UNORM,
                            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                            VK_IMAGE_ASPECT_COLOR_BIT);
    }
    if (vram_rb_buf_ == VK_NULL_HANDLE) {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0,
                               static_cast<VkDeviceSize>(kW) * kH * 4u,
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
        VmaAllocationCreateInfo vaci{};
        vaci.usage = VMA_MEMORY_USAGE_AUTO;
        vaci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo info{};
        if (vmaCreateBuffer(ctx_.allocator, &bci, &vaci, &vram_rb_buf_, &vram_rb_alloc_, &info) == VK_SUCCESS) {
            vram_rb_mapped_ = info.pMappedData;
        }
    }
    if (vram_rb_img_.image == VK_NULL_HANDLE || vram_rb_buf_ == VK_NULL_HANDLE) return;

    // Prefer the HUD-free clean 3D snapshot when available so the killcam never bakes in the HUD!
    const bool use_clean = captured_clean_3d_ && killcam_clean_.image != VK_NULL_HANDLE;
    VkImageResource &src_img = use_clean ? killcam_clean_ : shown->color;
    const std::uint32_t src_w = use_clean ? killcam_clean_.width : std::min(kW, shown->width) * static_cast<std::uint32_t>(scale_);
    const std::uint32_t src_h = use_clean ? killcam_clean_.height : std::min(kH, shown->height) * static_cast<std::uint32_t>(scale_);

    ensure_recording();
    end_render_pass();

    bool saved_hires = false;
    if (use_clean) {
        if (killcam_staging_.image == VK_NULL_HANDLE || killcam_staging_.width != src_w || killcam_staging_.height != src_h) {
            retire_image(killcam_staging_);
            killcam_staging_.create(ctx_, &descriptors_, src_w, src_h, VK_FORMAT_R8G8B8A8_UNORM,
                                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                    VK_IMAGE_ASPECT_COLOR_BIT);
        }
        if (killcam_staging_.image != VK_NULL_HANDLE) {
            transition_resource(killcam_clean_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
            transition_resource(killcam_staging_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            VkImageCopy cp{};
            cp.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            cp.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            cp.extent = {src_w, src_h, 1};
            vkCmdCopyImage(frames_[frame_slot_].cmd, killcam_clean_.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           killcam_staging_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);
            transition_resource(killcam_staging_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            saved_hires = true;
        }
    }

    transition_resource(src_img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    transition_resource(vram_rb_img_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

    VkImageBlit blit{};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.srcOffsets[0] = {0, 0, 0};
    blit.srcOffsets[1] = {static_cast<std::int32_t>(src_w), static_cast<std::int32_t>(src_h), 1};
    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.dstOffsets[0] = {0, 0, 0};
    blit.dstOffsets[1] = {static_cast<std::int32_t>(kW), static_cast<std::int32_t>(kH), 1};
    vkCmdBlitImage(frames_[frame_slot_].cmd, src_img.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   vram_rb_img_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

    transition_resource(vram_rb_img_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {kW, kH, 1};
    vkCmdCopyImageToBuffer(frames_[frame_slot_].cmd, vram_rb_img_.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           vram_rb_buf_, 1, &region);

    // Restore shown->color back to shader-read layout if we transitioned it.
    if (!use_clean) {
        transition_resource(shown->color, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }

    pending_vram_rb_.active = true;
    pending_vram_rb_.timeline_value = ctx_.graphics_timeline_value + 1u;
    pending_vram_rb_.address = shown->address;
    pending_vram_rb_.stride = shown->stride;
    pending_vram_rb_.format = shown->format;
    pending_vram_rb_.width = std::min(kW, shown->width);
    pending_vram_rb_.height = std::min(kH, shown->height);
    pending_vram_rb_.has_killcam_hires = saved_hires;
}

void VkBackend::copy_target_to_ram(RenderTarget &from, std::uint32_t src, std::uint32_t sx, std::uint32_t sy,
                                   std::uint32_t dst, std::uint32_t dst_stride, std::uint32_t dx, std::uint32_t dy,
                                   std::uint32_t width, std::uint32_t height) {
    const std::uint32_t bpp = from.bytes_per_pixel();
    const std::uint32_t src_offset = ((src & 0x001FFFFFu) - from.address) / bpp;
    const std::uint32_t fx = src_offset % from.stride + sx, fy = src_offset / from.stride + sy;
    if (fx >= from.width || fy >= from.height) return;
    width = std::min(width, from.width - fx);
    height = std::min(height, from.height - fy);
    if (width == 0 || height == 0) return;

    const bool have_clean = killcam_clean_valid_ && killcam_clean_.image != VK_NULL_HANDLE &&
                            (captured_clean_3d_ || killcam_clean_target_ == from.address);
    const bool use_clean = is_shown_target(from.address) && have_clean && fx < 480u && fy < 272u;
    if (use_clean) {
        width = std::min(width, 480u - fx);
        height = std::min(height, 272u - fy);
    }
    VkImageResource &src_img = use_clean ? killcam_clean_ : from.color;

    VkImageResource tmp_img;
    tmp_img.create(ctx_, nullptr, width, height, VK_FORMAT_R8G8B8A8_UNORM,
                   VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    VkBuffer rb_buf = VK_NULL_HANDLE;
    VmaAllocation rb_alloc = VK_NULL_HANDLE;
    VmaAllocationInfo rb_info{};
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0,
                           static_cast<VkDeviceSize>(width) * height * 4u,
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
    VmaAllocationCreateInfo vaci{};
    vaci.usage = VMA_MEMORY_USAGE_AUTO;
    vaci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    if (tmp_img.image == VK_NULL_HANDLE || vmaCreateBuffer(ctx_.allocator, &bci, &vaci, &rb_buf, &rb_alloc, &rb_info) != VK_SUCCESS) {
        tmp_img.destroy_immediate(ctx_, nullptr);
        return;
    }

    ensure_recording();
    end_render_pass();

    // If this is a full-screen or top-left strip killcam readback, pin the full internal-res viewport snapshot
    // in a dedicated slot that the rolling VRAM readback ring can never evict during long post-quest countdowns.
    bool pinned_killcam = false;
    const std::uint32_t pin_w = use_clean ? killcam_clean_.width : (480u * static_cast<std::uint32_t>(scale_));
    const std::uint32_t pin_h = use_clean ? killcam_clean_.height : (272u * static_cast<std::uint32_t>(scale_));
    if (is_shown_target(from.address) && fx == 0u && fy == 0u && (width >= 480u || height >= 256u) &&
        pin_w <= src_img.width && pin_h <= src_img.height) {
        auto &ks = killcam_pinned_;
        if (ks.image.image == VK_NULL_HANDLE || ks.image.width != pin_w || ks.image.height != pin_h) {
            retire_image(ks.image);
            ks.image.create(ctx_, &descriptors_, pin_w, pin_h, VK_FORMAT_R8G8B8A8_UNORM,
                            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        }
        if (ks.image.image != VK_NULL_HANDLE) {
            transition_resource(src_img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
            transition_resource(ks.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            VkImageCopy cp{};
            cp.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            cp.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            cp.extent = {pin_w, pin_h, 1};
            vkCmdCopyImage(frames_[frame_slot_].cmd, src_img.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           ks.image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);
            transition_resource(ks.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            killcam_ram_stride_ = dst_stride >= 480u ? dst_stride : 512u;
            killcam_ram_addr_ = (dst + (dy * killcam_ram_stride_ + dx) * bpp) & 0x1FFFFFFFu;
            killcam_ram_bpp_ = bpp;
            ks.valid = true;
            ks.frame = frame_number_;
            pinned_killcam = true;
        }
    }

    transition_resource(src_img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    transition_resource(tmp_img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkImageBlit blit{};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.srcOffsets[0] = {static_cast<std::int32_t>(fx) * scale_, static_cast<std::int32_t>(fy) * scale_, 0};
    blit.srcOffsets[1] = {static_cast<std::int32_t>(fx + width) * scale_, static_cast<std::int32_t>(fy + height) * scale_, 1};
    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.dstOffsets[0] = {0, 0, 0};
    blit.dstOffsets[1] = {static_cast<std::int32_t>(width), static_cast<std::int32_t>(height), 1};
    vkCmdBlitImage(frames_[frame_slot_].cmd, src_img.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   tmp_img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
    transition_resource(tmp_img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {width, height, 1};
    vkCmdCopyImageToBuffer(frames_[frame_slot_].cmd, tmp_img.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           rb_buf, 1, &region);

    vkEndCommandBuffer(frames_[frame_slot_].cmd);
    recording_ = false;
    const std::uint64_t sig = ++ctx_.graphics_timeline_value;
    VkCommandBufferSubmitInfo cbsi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, nullptr, frames_[frame_slot_].cmd, 0};
    VkSemaphoreSubmitInfo ssi{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, nullptr, ctx_.graphics_timeline, sig,
                              VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0};
    VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2, nullptr, 0, 0, nullptr, 1, &cbsi, 1, &ssi};
    {
        std::lock_guard lock(ctx_.graphics_queue_mutex);
        vkQueueSubmit2(ctx_.graphics_queue, 1, &si, VK_NULL_HANDLE);
    }
    ctx_.wait_timeline(ctx_.graphics_timeline, sig);
    frames_[frame_slot_].timeline_value = sig;

    vmaInvalidateAllocation(ctx_.allocator, rb_alloc, 0, VK_WHOLE_SIZE);
    const auto *rgba = static_cast<const std::uint8_t *>(rb_info.pMappedData);
    for (std::uint32_t y = 0; y < height; ++y) {
        std::uint8_t *row = memory_.raw_pointer(dst + ((dy + y) * dst_stride + dx) * bpp, width * bpp);
        if (!row) continue;
        const std::uint8_t *line = rgba + static_cast<std::size_t>(y) * width * 4u;
        if (from.format == 3u) {
            std::memcpy(row, line, width * 4u);
        } else {
            for (std::uint32_t x = 0; x < width; ++x) pack_pixel(line + x * 4u, from.format, row + x * bpp);
        }
    }
    if (pinned_killcam) {
        auto &ks = killcam_pinned_;
        ks.thumb.assign(480u * 272u, 0xFF000000u);
        const std::uint32_t copy_w = std::min(width, 480u);
        const std::uint32_t copy_h = std::min(height, 272u);
        for (std::uint32_t y = 0; y < copy_h; ++y) {
            std::memcpy(ks.thumb.data() + static_cast<std::size_t>(y) * 480u,
                        rgba + static_cast<std::size_t>(y) * width * 4u, copy_w * 4u);
        }
    } else if (is_shown_target(from.address) && killcam_pinned_.valid && killcam_pinned_.thumb.size() >= 480u * 272u &&
               frame_number_ <= killcam_pinned_.frame + 2u && fx < 480u && fy < 272u) {
        const std::uint32_t copy_w = std::min(width, 480u - fx);
        const std::uint32_t copy_h = std::min(height, 272u - fy);
        for (std::uint32_t y = 0; y < copy_h; ++y) {
            std::memcpy(killcam_pinned_.thumb.data() + static_cast<std::size_t>(fy + y) * 480u + fx,
                        rgba + static_cast<std::size_t>(y) * width * 4u, copy_w * 4u);
        }
    }
    vmaDestroyBuffer(ctx_.allocator, rb_buf, rb_alloc);
    tmp_img.destroy_immediate(ctx_, nullptr);
    ensure_recording();
}

std::uint32_t VkBackend::match_killcam_hires(std::uint32_t tex_addr, std::uint32_t tex_stride,
                                             std::uint32_t tex_w, std::uint32_t tex_h, std::uint32_t tex_fmt,
                                             bool swizzled, const std::uint32_t *rgba_pixels,
                                             std::uint32_t &out_ox, std::uint32_t &out_oy) noexcept {
    out_ox = 0;
    out_oy = 0;
    if (tex_fmt > 3u) return 0u;
    const std::uint32_t addr = tex_addr & 0x1FFFFFFFu;
    const std::uint32_t bpp = (tex_fmt == 3u) ? 4u : 2u;

    // 1. Direct base/strip address match against the pinned killcam slot, verified with an O(1) 8-pixel
    //    content check so reused RAM buffers can never falsely hijack an unrelated texture.
    if (killcam_ram_addr_ != 0 && bpp == killcam_ram_bpp_ && killcam_pinned_.valid &&
        killcam_pinned_.image.bindless_slot != 0) {
        const std::uint32_t row_stride = killcam_ram_stride_ >= 480u ? killcam_ram_stride_ : 512u;
        const std::uint32_t span = row_stride * 272u * killcam_ram_bpp_;
        if (addr >= killcam_ram_addr_ && addr < killcam_ram_addr_ + span &&
            (tex_stride == row_stride || addr == killcam_ram_addr_)) {
            const std::uint32_t px_off = (addr - killcam_ram_addr_) / killcam_ram_bpp_;
            const std::uint32_t ox = px_off % row_stride;
            const std::uint32_t oy = px_off / row_stride;
            if (ox < 480u && oy < 272u) {
                bool verified = (killcam_ram_verified_frame_ == frame_number_);
                if (!verified) {
                    if (swizzled && !rgba_pixels) {
                        // Defer swizzled verification until decoded pixels are available on the second call.
                        return 0u;
                    }
                    if (killcam_pinned_.thumb.size() >= 480u * 272u) {
                        const std::uint32_t cw = std::min(tex_w, 480u - ox);
                        const std::uint32_t ch = std::min(tex_h, 272u - oy);
                        const std::uint32_t eff_stride = tex_stride >= cw ? tex_stride : row_stride;
                        std::uint32_t err = 0u;
                        std::uint32_t samples = 0u;
                        for (std::uint32_t gy = 0; gy < 2u; ++gy) {
                            const std::uint32_t sy = (ch * (gy * 2u + 1u)) / 4u;
                            for (std::uint32_t gx = 0; gx < 4u; ++gx) {
                                const std::uint32_t sx = (cw * (gx * 2u + 1u)) / 8u;
                                std::uint32_t a = 0u;
                                if (rgba_pixels) {
                                    a = rgba_pixels[static_cast<std::size_t>(sy) * tex_w + sx];
                                } else {
                                    const std::uint32_t paddr = addr + (sy * eff_stride + sx) * bpp;
                                    const std::uint8_t *raw = memory_.raw_pointer(paddr, bpp);
                                    if (!raw) continue;
                                    a = framebuffer_pixel(raw, tex_fmt);
                                }
                                const std::uint32_t b = killcam_pinned_.thumb[static_cast<std::size_t>(oy + sy) * 480u + (ox + sx)];
                                const int ar = static_cast<int>(a & 0xFFu), ag = static_cast<int>((a >> 8u) & 0xFFu),
                                          ab = static_cast<int>((a >> 16u) & 0xFFu);
                                const int br = static_cast<int>(b & 0xFFu), bg = static_cast<int>((b >> 8u) & 0xFFu),
                                          bb = static_cast<int>((b >> 16u) & 0xFFu);
                                err += static_cast<std::uint32_t>(std::abs(ar - br) + std::abs(ag - bg) + std::abs(ab - bb));
                                ++samples;
                            }
                        }
                        verified = (samples > 0u) && (err <= 24u * 3u * samples);
                    } else {
                        verified = true;
                    }
                }
                if (verified) {
                    killcam_ram_verified_frame_ = frame_number_;
                    out_ox = ox;
                    out_oy = oy;
                    return killcam_pinned_.image.bindless_slot;
                }
                if (addr == killcam_ram_addr_) {
                    killcam_ram_addr_ = 0;
                }
            }
        }
    }
    if (!rgba_pixels || tex_w < 32u || tex_h < 64u) return 0u;

    // 2. Perceptual 16-sample match against saved 480x272 HUD-free captures (checks pinned slot first, then
    //    rolling ring slots for games that snapshot VRAM via CPU memcpy instead of block_transfer).
    const std::uint32_t cmp_w = std::min(tex_w, 480u);
    const std::uint32_t cmp_h = std::min(tex_h, 272u);
    auto score_slot = [&](const KillcamSlot &ks, std::uint32_t &out_err) -> bool {
        if (!ks.valid || ks.image.bindless_slot == 0 || ks.thumb.size() < 480u * 272u) return false;
        std::uint32_t err = 0u;
        std::uint32_t non_black = 0u;
        for (std::uint32_t gy = 0; gy < 4u; ++gy) {
            const std::uint32_t sy = (cmp_h * (gy * 2u + 1u)) / 8u;
            for (std::uint32_t gx = 0; gx < 4u; ++gx) {
                const std::uint32_t sx = (cmp_w * (gx * 2u + 1u)) / 8u;
                const std::uint32_t a = rgba_pixels[static_cast<std::size_t>(sy) * tex_w + sx];
                const std::uint32_t b = ks.thumb[static_cast<std::size_t>(sy) * 480u + sx];
                const int ar = static_cast<int>(a & 0xFFu), ag = static_cast<int>((a >> 8u) & 0xFFu),
                          ab = static_cast<int>((a >> 16u) & 0xFFu);
                const int br = static_cast<int>(b & 0xFFu), bg = static_cast<int>((b >> 8u) & 0xFFu),
                          bb = static_cast<int>((b >> 16u) & 0xFFu);
                if (ar > 10 || ag > 10 || ab > 10) ++non_black;
                err += static_cast<std::uint32_t>(std::abs(ar - br) + std::abs(ag - bg) + std::abs(ab - bb));
            }
        }
        if (non_black < 6u) return false;
        out_err = err;
        return true;
    };

    std::size_t best_idx = kKillcamSlots + 1u; // kKillcamSlots == pinned slot, < kKillcamSlots == rolling slot
    std::uint32_t best_err = 18u * 3u * 16u;
    std::uint64_t best_frame = 0u;
    std::uint32_t pin_err = 0u;
    if (score_slot(killcam_pinned_, pin_err) && pin_err < best_err) {
        best_err = pin_err;
        best_frame = killcam_pinned_.frame;
        best_idx = kKillcamSlots;
    }
    for (std::size_t idx = 0; idx < kKillcamSlots; ++idx) {
        std::uint32_t err = 0u;
        if (score_slot(killcam_slots_[idx], err) &&
            (err < best_err || (err == best_err && killcam_slots_[idx].frame > best_frame))) {
            best_err = err;
            best_frame = killcam_slots_[idx].frame;
            best_idx = idx;
        }
    }
    if (best_idx <= kKillcamSlots) {
        if (best_idx < kKillcamSlots) {
            std::swap(killcam_pinned_, killcam_slots_[best_idx]);
            killcam_slots_[best_idx].valid = false;
        }
        // Lock this RAM base address so all remaining strips/tiles of the killcam hit in O(1)!
        killcam_ram_addr_ = addr;
        killcam_ram_bpp_ = bpp;
        killcam_ram_stride_ = tex_stride >= 480u ? tex_stride : 512u;
        killcam_ram_verified_frame_ = frame_number_;
        return killcam_pinned_.image.bindless_slot;
    }
    return 0u;
}

void VkBackend::flush() {
    draw_batch();
    frame_tex_cache_.clear();
    static const bool readback = [] {
        const char *v = std::getenv("MHF_VK_READBACK");
        return v != nullptr && v[0] == '1';
    }();
    if (readback) read_back();
}

void VkBackend::read_back() {
    if (!recording_) return;
    end_render_pass();
    // Synchronous diagnostic readback when MHF_VK_READBACK=1 (matches MHF_GL_READBACK=1 for pixel tests).
    for (auto &[addr, rt] : targets_) {
        if (rt.last_draw != frame_number_) continue;
        const std::size_t bytes = static_cast<std::size_t>(rt.color.width) * rt.color.height * 4u;
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, bytes,
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
        VmaAllocationCreateInfo vaci{};
        vaci.usage = VMA_MEMORY_USAGE_AUTO;
        vaci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VkBuffer rb_buf = VK_NULL_HANDLE;
        VmaAllocation rb_alloc = VK_NULL_HANDLE;
        VmaAllocationInfo rb_info{};
        if (vmaCreateBuffer(ctx_.allocator, &bci, &vaci, &rb_buf, &rb_alloc, &rb_info) != VK_SUCCESS) continue;

        transition_resource(rt.color, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {rt.color.width, rt.color.height, 1};
        vkCmdCopyImageToBuffer(frames_[frame_slot_].cmd, rt.color.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               rb_buf, 1, &region);

        vkEndCommandBuffer(frames_[frame_slot_].cmd);
        recording_ = false;

        const std::uint64_t sig = ++ctx_.graphics_timeline_value;
        VkCommandBufferSubmitInfo cbsi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, nullptr, frames_[frame_slot_].cmd, 0};
        VkSemaphoreSubmitInfo ssi{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, nullptr, ctx_.graphics_timeline, sig,
                                  VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0};
        VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2, nullptr, 0, 0, nullptr, 1, &cbsi, 1, &ssi};
        {
            std::lock_guard lock(ctx_.graphics_queue_mutex);
            vkQueueSubmit2(ctx_.graphics_queue, 1, &si, VK_NULL_HANDLE);
        }
        ctx_.wait_timeline(ctx_.graphics_timeline, sig);
        frames_[frame_slot_].timeline_value = sig;

        vmaInvalidateAllocation(ctx_.allocator, rb_alloc, 0, VK_WHOLE_SIZE);
    const auto *rgba = static_cast<const std::uint8_t *>(rb_info.pMappedData);
        const std::uint32_t bpp = rt.bytes_per_pixel();
        for (std::uint32_t y = 0; y < rt.height; ++y) {
            std::uint8_t *row = memory_.raw_pointer(0x04000000u + rt.address + y * rt.stride * bpp, rt.width * bpp);
            if (!row) continue;
            for (std::uint32_t x = 0; x < rt.width; ++x) {
                const std::uint8_t *p = rgba + (static_cast<std::size_t>(y) * scale_ * rt.color.width +
                                                static_cast<std::size_t>(x) * scale_) * 4u;
                pack_pixel(p, rt.format, row + x * bpp);
            }
        }
        vmaDestroyBuffer(ctx_.allocator, rb_buf, rb_alloc);
        ensure_recording();
    }
    have_state_ = false;
    last_state_gen_ = 0;
}

void VkBackend::upload_overlay_if_needed() {
    if (!overlay_) return;
    std::vector<std::uint32_t> pixels;
    int w = 0, h = 0;
    bool upload = false;
    {
        std::lock_guard lock(overlay_->mutex);
        const bool now_visible = overlay_->visible && overlay_->width > 0 && overlay_->height > 0;
        if (now_visible && !overlay_visible_) overlay_shown_at_ms_ = SDL_GetTicks();
        overlay_visible_ = now_visible;
        overlay_continuous_ = overlay_->continuous;
        overlay_aspect_ = overlay_->aspect;
        if (now_visible && overlay_->generation != overlay_gen_) {
            w = overlay_->width;
            h = overlay_->height;
            pixels = overlay_->pixels;
            overlay_gen_ = overlay_->generation;
            upload = true;
        }
    }
    if (upload && !pixels.empty()) {
        upload_texture_2d(overlay_tex_, static_cast<std::uint32_t>(w), static_cast<std::uint32_t>(h), pixels.data());
    }
}

void VkBackend::build_menu_blur(std::uint32_t src_slot, float uv_max_x, float uv_max_y) {
    constexpr std::uint32_t kBw = 480u, kBh = 272u;
    if (blur_a_.image == VK_NULL_HANDLE) {
        blur_a_.create(ctx_, &descriptors_, kBw, kBh, VK_FORMAT_R8G8B8A8_UNORM,
                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        blur_b_.create(ctx_, &descriptors_, kBw, kBh, VK_FORMAT_R8G8B8A8_UNORM,
                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    }
    if (blur_a_.image == VK_NULL_HANDLE || blur_b_.image == VK_NULL_HANDLE) return;

    auto run_pass = [&](VkImageResource &dst, std::uint32_t in_slot, float u_scale, float v_scale,
                        float dx, float dy, float dim) {
        transition_resource(dst, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        VkRenderingAttachmentInfo att{};
        att.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        att.imageView = dst.view;
        att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        att.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0, 0}, {kBw, kBh}}, 1, 0, 1, &att, nullptr, nullptr};
        VkCommandBuffer cmd = frames_[frame_slot_].cmd;
        vkCmdBeginRendering(cmd, &ri);
        VkViewport vp{0.0f, 0.0f, static_cast<float>(kBw), static_cast<float>(kBh), 0.0f, 1.0f};
        VkRect2D sc{{0, 0}, {kBw, kBh}};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);
        VkDescriptorSet set0 = descriptors_.set();
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines_.blur_pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines_.blur_layout, 0, 1, &set0, 0, nullptr);
        vk::BlurPush bp{{0.0f, 0.0f}, {u_scale, v_scale}, {dx, dy}, dim, in_slot,
                        vk::BindlessDescriptors::sampler_index(true, false, true, true, 1)};
        vkCmdPushConstants(cmd, pipelines_.blur_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(bp), &bp);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRendering(cmd);
        transition_resource(dst, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    };

    run_pass(blur_a_, src_slot, uv_max_x, uv_max_y, uv_max_x / static_cast<float>(kBw), 0.0f, 1.0f);
    run_pass(blur_b_, blur_a_.bindless_slot, 1.0f, 1.0f, 0.0f, 1.0f / static_cast<float>(kBh), 0.55f);
    blur_src_slot_ = blur_b_.bindless_slot;
}

void VkBackend::submit_recorded_without_present() {
    if (!recording_) return;
    end_render_pass();
    FrameSlice &f = frames_[frame_slot_];
    vkEndCommandBuffer(f.cmd);
    recording_ = false;

    const std::uint64_t sig = ++ctx_.graphics_timeline_value;
    f.timeline_value = sig;

    VkCommandBufferSubmitInfo cmd_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, nullptr, f.cmd, 0};
    VkSemaphoreSubmitInfo sig_info{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, nullptr, ctx_.graphics_timeline, sig,
                                   VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0};
    VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2, nullptr, 0, 0, nullptr, 1, &cmd_info, 1, &sig_info};
    {
        std::lock_guard lock(ctx_.graphics_queue_mutex);
        vkQueueSubmit2(ctx_.graphics_queue, 1, &si, VK_NULL_HANDLE);
    }
    frame_slot_ = (frame_slot_ + 1) % kFramesInFlight;
}

void VkBackend::show_to_swapchain(std::uint32_t tex_slot, float uv_max_x, float uv_max_y, bool linear_filter) {
    ensure_recording();
    end_render_pass();

    int pw = 0, ph = 0;
    SDL_GetWindowSizeInPixels(window_, &pw, &ph);
    if (pw <= 0 || ph <= 0) {
        submit_recorded_without_present();
        return;
    }

    if (swapchain_.handle == VK_NULL_HANDLE || swapchain_dirty_ ||
        static_cast<std::uint32_t>(pw) != swapchain_.extent.width ||
        static_cast<std::uint32_t>(ph) != swapchain_.extent.height) {
        // Submit already-recorded 3D work & layout transitions before recreating the swapchain
        // so tracked VkImageResource::layout values stay 100% synchronized with GPU state!
        submit_recorded_without_present();
        if (swapchain_.recreate(ctx_, pw, ph)) {
            pipelines_.rebuild_swapchain_pipelines(ctx_, swapchain_.format);
            swapchain_dirty_ = false;
        } else {
            swapchain_dirty_ = true;
            return;
        }
        ensure_recording();
    }

    upload_overlay_if_needed();
    if (overlay_visible_ && overlay_continuous_) {
        build_menu_blur(tex_slot, uv_max_x, uv_max_y);
        if (blur_src_slot_ != 0) {
            tex_slot = blur_src_slot_;
            uv_max_x = 1.0f;
            uv_max_y = 1.0f;
            linear_filter = true;
        }
    }

    std::uint32_t img_idx = 0;
    const auto acq_t0 = prof_.enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    VkResult acq = vkAcquireNextImageKHR(ctx_.device, swapchain_.handle, UINT64_MAX,
                                         frames_[frame_slot_].image_available, VK_NULL_HANDLE, &img_idx);
    if (prof_.enabled) {
        prof_.acq_ns += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - acq_t0).count());
    }
    if (acq == VK_ERROR_OUT_OF_DATE_KHR) {
        submit_recorded_without_present();
        if (swapchain_.recreate(ctx_, pw, ph)) {
            pipelines_.rebuild_swapchain_pipelines(ctx_, swapchain_.format);
            swapchain_dirty_ = false;
        } else {
            swapchain_dirty_ = true;
        }
        return;
    }
    if (acq == VK_SUBOPTIMAL_KHR) {
        swapchain_dirty_ = true;
    } else if (acq != VK_SUCCESS) {
        submit_recorded_without_present();
        return;
    }

    VkCommandBuffer cmd = frames_[frame_slot_].cmd;
    VkImage swap_img = swapchain_.images[img_idx];
    vk::Context::cmd_transition_image(cmd, swap_img, VK_IMAGE_ASPECT_COLOR_BIT,
                                      VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                                      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                      VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

    VkRenderingAttachmentInfo att{};
    att.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    att.imageView = swapchain_.views[img_idx];
    att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att.clearValue.color = {{0.0f, 0.0f, 0.0f, 1.0f}};

    VkRenderingInfo ri{};
    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea = {{0, 0}, swapchain_.extent};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &att;

    vkCmdBeginRendering(cmd, &ri);

    // Letterboxed viewport (or widescreen when active).
    const float win_w = static_cast<float>(swapchain_.extent.width);
    const float win_h = static_cast<float>(swapchain_.extent.height);
    const float wide = live_->wide_aspect.load(std::memory_order_relaxed);
    const float target_aspect = wide > 0.0f ? wide : (480.0f / 272.0f);
    float draw_w = win_w, draw_h = win_h;
    if (win_w / win_h > target_aspect) draw_w = win_h * target_aspect;
    else draw_h = win_w / target_aspect;
    const float vx = (win_w - draw_w) * 0.5f;
    const float vy = (win_h - draw_h) * 0.5f;

    VkViewport vp{vx, vy, draw_w, draw_h, 0.0f, 1.0f};
    VkRect2D sc{{static_cast<std::int32_t>(vx), static_cast<std::int32_t>(vy)},
                {static_cast<std::uint32_t>(draw_w), static_cast<std::uint32_t>(draw_h)}};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);

    VkDescriptorSet set0 = descriptors_.set();
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines_.present_pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines_.present_layout, 0, 1, &set0, 0, nullptr);

    vk::PresentPush pp{{uv_max_x, uv_max_y}, tex_slot,
                       vk::BindlessDescriptors::sampler_index(linear_filter, false, true, true, 1), 1.0f};
    vkCmdPushConstants(cmd, pipelines_.present_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(pp), &pp);
    vkCmdDraw(cmd, 3, 1, 0, 0);

    // Draw overlay (system dialogs, in-game menu, HUD overlays).
    if (overlay_visible_ && overlay_tex_.bindless_slot != 0) {
        const float ov_aspect = overlay_aspect_ > 0.0f ? overlay_aspect_ : (480.0f / 272.0f);
        float ow = win_w, oh = win_h;
        if (win_w / win_h > ov_aspect) ow = win_h * ov_aspect;
        else oh = win_w / ov_aspect;
        const float ox = (win_w - ow) * 0.5f;
        const float oy = (win_h - oh) * 0.5f;
        VkViewport ovp{ox, oy, ow, oh, 0.0f, 1.0f};
        VkRect2D osc{{static_cast<std::int32_t>(ox), static_cast<std::int32_t>(oy)},
                     {static_cast<std::uint32_t>(ow), static_cast<std::uint32_t>(oh)}};
        vkCmdSetViewport(cmd, 0, 1, &ovp);
        vkCmdSetScissor(cmd, 0, 1, &osc);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines_.overlay_pipe);
        const float ft = std::clamp(static_cast<float>(SDL_GetTicks() - overlay_shown_at_ms_) / 150.0f, 0.0f, 1.0f);
        const float alpha = ft * (2.0f - ft); // smooth ease-out fade-in
        vk::PresentPush op{{1.0f, 1.0f}, overlay_tex_.bindless_slot,
                           vk::BindlessDescriptors::sampler_index(true, false, true, true, 1), alpha};
        vkCmdPushConstants(cmd, pipelines_.present_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(op), &op);
        vkCmdDraw(cmd, 3, 1, 0, 0);
    }

    vkCmdEndRendering(cmd);

    vk::Context::cmd_transition_image(cmd, swap_img, VK_IMAGE_ASPECT_COLOR_BIT,
                                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                      VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0);

    submit_and_present(img_idx);
}

void VkBackend::submit_and_present(std::uint32_t swap_img_idx) {
    FrameSlice &f = frames_[frame_slot_];
    vkEndCommandBuffer(f.cmd);
    recording_ = false;

    const std::uint64_t sig = ++ctx_.graphics_timeline_value;
    f.timeline_value = sig;

    VkSemaphoreSubmitInfo wait_info{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, nullptr,
                                    f.image_available, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0};
    VkCommandBufferSubmitInfo cmd_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, nullptr, f.cmd, 0};
    std::array<VkSemaphoreSubmitInfo, 2> signals = {{
        {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, nullptr, swapchain_.render_finished[swap_img_idx], 0,
         VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, 0},
        {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, nullptr, ctx_.graphics_timeline, sig,
         VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0},
    }};
    VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2, nullptr, 0, 1, &wait_info, 1, &cmd_info,
                     static_cast<std::uint32_t>(signals.size()), signals.data()};

    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, nullptr,
                        1, &swapchain_.render_finished[swap_img_idx],
                        1, &swapchain_.handle, &swap_img_idx, nullptr};
#ifdef VK_KHR_PRESENT_ID_EXTENSION_NAME
    std::uint64_t pid = 0;
    VkPresentIdKHR present_id_info{VK_STRUCTURE_TYPE_PRESENT_ID_KHR, nullptr, 1, &pid};
    if (ctx_.caps.present_id) {
        pid = swapchain_.next_present_id++;
        pi.pNext = &present_id_info;
    }
#endif
    VkResult pres = VK_SUCCESS;
    {
        std::lock_guard lock(ctx_.graphics_queue_mutex);
        vkQueueSubmit2(ctx_.graphics_queue, 1, &si, VK_NULL_HANDLE);
        pres = vkQueuePresentKHR(ctx_.graphics_queue, &pi);
    }
    if (pres == VK_ERROR_OUT_OF_DATE_KHR || pres == VK_SUBOPTIMAL_KHR) {
        swapchain_dirty_ = true;
    }

#ifdef VK_KHR_PRESENT_WAIT_EXTENSION_NAME
    // Hardware scanout backpressure (keeps presentation queue depth bounded to 1 without busy-wait).
    if (ctx_.caps.present_wait && vkWaitForPresentKHR && swapchain_.next_present_id > 3u) {
        const auto pw_t0 = prof_.enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        vkWaitForPresentKHR(ctx_.device, swapchain_.handle, swapchain_.next_present_id - 3u, 8'000'000ULL);
        if (prof_.enabled) {
            prof_.pres_ns += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - pw_t0).count());
        }
    }
#endif

    frame_slot_ = (frame_slot_ + 1) % kFramesInFlight;
}

void VkBackend::apply_live_scale() {
    const int req = live_->scale.exchange(0, std::memory_order_relaxed);
    if (req >= 1 && req <= 10 && req != scale_) {
        scale_ = req;
        reset_interp_state();
        for (auto &[addr, rt] : targets_) {
            retire_image(rt.color);
            retire_image(rt.depth);
        }
        targets_.clear();
        retire_image(interp_rt_.color);
        retire_image(interp_rt_.depth);
        retire_image(copy_tex_);
        retire_image(soft_depth_copy_);
        retire_image(msaa_color_);
        retire_image(msaa_depth_);
        retire_image(hdr_color_);
        retire_image(msaa_hdr_color_);
        retire_image(killcam_clean_);
        retire_image(killcam_staging_);
        lighting_.reset_taa_history();
        lighting_.reset_exposure();
        taa_have_prev_ = false;
        killcam_clean_valid_ = false;
        killcam_clean_target_ = UINT32_MAX;
    }
}

void VkBackend::apply_live_fps() {
    const int req = live_->fps.exchange(0, std::memory_order_relaxed);
    if (req >= 30 && req <= 180) {
        target_fps_ = req;
        subframes_ = std::clamp((target_fps_ + 15) / 30, 1, 6);
        if (subframes_ <= 1) {
            reset_interp_state();
        }
        lighting_.reset_taa_history();
        taa_have_prev_ = false;
    }
}

bool VkBackend::upload_snap_raw(SnapRawBuf &dst, const std::vector<vk::RawGpuVertex> &verts) {
    if (verts.empty()) return false;
    const std::size_t rbytes = verts.size() * sizeof(vk::RawGpuVertex);
    if (dst.last_gpu_timeline != 0) {
        ctx_.wait_timeline(ctx_.graphics_timeline, dst.last_gpu_timeline);
    }
    if (dst.cap < rbytes || dst.buf == VK_NULL_HANDLE) {
        ctx_.wait_idle();
        if (dst.buf != VK_NULL_HANDLE) {
            vmaDestroyBuffer(ctx_.allocator, dst.buf, dst.alloc);
            dst = SnapRawBuf{};
        }
        dst.cap = std::max(rbytes, std::size_t{4u << 20});
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, dst.cap,
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                               VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
        VmaAllocationCreateInfo vaci{};
        vaci.usage = VMA_MEMORY_USAGE_AUTO;
        vaci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo rinfo{};
        if (vmaCreateBuffer(ctx_.allocator, &bci, &vaci, &dst.buf, &dst.alloc, &rinfo) != VK_SUCCESS) {
            dst = SnapRawBuf{};
            return false;
        }
        VkBufferDeviceAddressInfo bdai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, dst.buf};
        dst.addr = vkGetBufferDeviceAddress(ctx_.device, &bdai);
        dst.mapped = rinfo.pMappedData;
    }
    if (!dst.mapped || dst.addr == 0) return false;
    std::memcpy(dst.mapped, verts.data(), rbytes);
    return true;
}

void VkBackend::snapshot_soft_depth(RenderTarget &rt) {
    if (rt.depth.image == VK_NULL_HANDLE) return;
    ensure_recording();
    end_render_pass();
    if (soft_depth_copy_.image == VK_NULL_HANDLE ||
        soft_depth_copy_.width != rt.depth.width || soft_depth_copy_.height != rt.depth.height) {
        retire_image(soft_depth_copy_);
        if (!soft_depth_copy_.create(ctx_, &descriptors_, rt.depth.width, rt.depth.height, VK_FORMAT_D32_SFLOAT,
                                     VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                     VK_IMAGE_ASPECT_DEPTH_BIT)) {
            return;
        }
    }
    transition_resource(rt.depth, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    transition_resource(soft_depth_copy_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkImageCopy cp{};
    cp.srcSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
    cp.dstSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
    cp.extent = {rt.depth.width, rt.depth.height, 1};
    vkCmdCopyImage(frames_[frame_slot_].cmd, rt.depth.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   soft_depth_copy_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);
    transition_resource(soft_depth_copy_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    transition_resource(rt.depth, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
}

std::uint32_t VkBackend::build_view_lights(
    const interpolation::Matrix &view_mat, const std::vector<SemanticPointLight> &vfx_lights,
    std::array<vk::ViewPointLight, vk::LightingFrameInput::kMaxViewLights> &out_lights) const {
    if (!want_vk_effect(LiveSettings::kVkEffectDynamicLights)) return 0u;
    std::uint32_t count = 0u;
    auto push_world_light = [&](const SemanticPointLight &src) {
        if (count >= vk::LightingFrameInput::kMaxViewLights || src.intensity <= 0.01f || src.radius <= 1.0f) return;
        const float wx = src.world_pos[0], wy = src.world_pos[1], wz = src.world_pos[2];
        const float vx = view_mat[0] * wx + view_mat[4] * wy + view_mat[8] * wz + view_mat[12];
        const float vy = view_mat[1] * wx + view_mat[5] * wy + view_mat[9] * wz + view_mat[13];
        const float vz = view_mat[2] * wx + view_mat[6] * wy + view_mat[10] * wz + view_mat[14];
        // Cull lights well behind the camera or far beyond visible range.
        if (vz > src.radius || vz < -2500.0f) return;
        vk::ViewPointLight &dst = out_lights[count++];
        // Negate Y to match lighting_composite.comp's +Y-down screen-aligned view space.
        dst.pos_radius[0] = vx;
        dst.pos_radius[1] = -vy;
        dst.pos_radius[2] = vz;
        dst.pos_radius[3] = src.radius;
        dst.color_intensity[0] = std::pow(std::clamp(src.color[0], 0.0f, 1.0f), 2.2f);
        dst.color_intensity[1] = std::pow(std::clamp(src.color[1], 0.0f, 1.0f), 2.2f);
        dst.color_intensity[2] = std::pow(std::clamp(src.color[2], 0.0f, 1.0f), 2.2f);
        dst.color_intensity[3] = src.intensity;
    };
    for (const auto &l : vfx_lights) push_world_light(l);
    SemanticSceneFrame sem{};
    {
        std::lock_guard<std::mutex> lock(live_->semantic_mutex);
        sem = live_->semantic_frame;
    }
    const std::uint32_t sn = std::min<std::uint32_t>(sem.light_count, static_cast<std::uint32_t>(SemanticSceneFrame::kMaxLights));
    for (std::uint32_t i = 0u; i < sn; ++i) push_world_light(sem.lights[i]);
    return count;
}

void VkBackend::render_sun_shadow_pass(const interpolation::Matrix &view_mat, const vk::LightingFrameInput &light_in,
                                       const std::vector<vk::ShadowVertex> &verts,
                                       const std::vector<vk::ShadowBatch> &batches) {
    if (!sun_shadows_.ready() || !want_vk_effect(LiveSettings::kVkEffectSunShadows) || lighting_.active_is_cave() ||
        verts.empty() || batches.empty()) {
        sun_shadows_.invalidate();
        return;
    }
    ensure_recording();
    end_render_pass();
    FrameSlice &f = frames_[frame_slot_];
    const std::size_t vbytes = verts.size() * sizeof(vk::ShadowVertex);
    auto vsl = f.vertices.allocate(vbytes, 16);
    if (!vsl.ptr) {
        sun_shadows_.invalidate();
        return;
    }
    std::memcpy(vsl.ptr, verts.data(), vbytes);
    const std::size_t nb = std::min<std::size_t>(batches.size(), 512u);
    auto psl = f.params.allocate(nb * sizeof(vk::ShadowDrawParams), 16);
    if (!psl.ptr) {
        sun_shadows_.invalidate();
        return;
    }
    float sun_dir[3]{};
    lighting_.resolve_sun_dir_world(light_in, sun_dir);
    sun_shadows_.render(f.cmd, descriptors_.set(), view_mat.data(), sun_dir, verts, batches,
                        f.vertices.buffer, vsl.offset, psl.ptr, psl.address, nb);
}

void VkBackend::apply_post_3d(RenderTarget &rt, const vk::LightingFrameInput &in,
                              const float *taa_prev_from_curr, bool taa_valid_reproj,
                              bool taa_reset, const float *curr_jitter_uv, bool use_hdr,
                              const vk::ViewPointLight *view_lights, std::uint32_t view_light_count,
                              const interpolation::Matrix *view_mat) {
    if (!lighting_.ready()) return;
    if (rt.color.image == VK_NULL_HANDLE || rt.depth.image == VK_NULL_HANDLE ||
        rt.color.bindless_slot == 0 || rt.depth.bindless_slot == 0) return;

    const int aa = effective_aa_mode();
    if (aa != last_aa_mode_) {
        lighting_.reset_taa_history();
        taa_have_prev_ = false;
        last_aa_mode_ = aa;
    }

    const std::uint32_t mask = live_->vulkan_effects_mask.load(std::memory_order_relaxed);
    constexpr std::uint32_t kNonFxaaLightingMask = LiveSettings::kVkEffectComputeMask & ~LiveSettings::kVkEffectFxaa;
    const bool want_lighting = live_->enhanced_lighting.load(std::memory_order_relaxed) &&
                               ((mask & kNonFxaaLightingMask) != 0u);
    const bool want_fxaa = (aa == LiveSettings::kAaFxaa);
    const bool want_taa = (aa == LiveSettings::kAaTaa);
    if (!want_lighting && !want_fxaa && !want_taa) return;

    ScopedTimer prof_scope(prof_.enabled, prof_.light_ns);

    const std::uint32_t w = std::min(480u, rt.width) * static_cast<std::uint32_t>(scale_);
    const std::uint32_t h = std::min(272u, rt.height) * static_cast<std::uint32_t>(scale_);
    if (w < 64u || h < 64u) return;

    ensure_recording();
    end_render_pass();

    FrameSlice &f = frames_[frame_slot_];
    if (want_lighting) {
        auto pslice = f.params.allocate(sizeof(vk::LightingComputeParams) * vk::LightingPipeline::kParamSlotsPerDispatch, 16);
        if (!pslice.ptr) return;

        std::uint32_t hdr_slot = 0u;
        if (use_hdr && hdr_color_.image != VK_NULL_HANDLE && hdr_color_.bindless_slot != 0) {
            transition_resource(hdr_color_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            hdr_slot = hdr_color_.bindless_slot;
        } else {
            transition_resource(rt.color, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }
        transition_resource(rt.depth, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

        VkDeviceAddress lights_bda = 0;
        std::uint32_t light_cnt = 0;
        if (view_lights && view_light_count > 0u) {
            const std::uint32_t clamped = std::min<std::uint32_t>(
                view_light_count, static_cast<std::uint32_t>(vk::LightingFrameInput::kMaxViewLights));
            auto lslice = f.params.allocate(clamped * sizeof(vk::ViewPointLight), 16);
            if (lslice.ptr) {
                std::memcpy(lslice.ptr, view_lights, clamped * sizeof(vk::ViewPointLight));
                lights_bda = lslice.address;
                light_cnt = clamped;
            }
        }

        vk::LightingFrameInput fin = in;
        fin.frame_index = frame_number_;
        float shadow_from_view[16]{};
        std::uint32_t shadow_slot = 0u;
        std::uint32_t rt_slot = 0u;
        if (view_mat != nullptr && !lighting_.active_is_cave()) {
            if (rt_shadows_.ready() && want_vk_effect(LiveSettings::kVkEffectRayTracing)) {
                auto rtslice = f.params.allocate(sizeof(vk::RtShadowParams), 16);
                if (rtslice.ptr) {
                    float sun_dir[3]{};
                    lighting_.resolve_sun_dir_world(fin, sun_dir);
                    if (rt_shadows_.trace(f.cmd, ctx_, descriptors_, descriptors_.set(), w, h,
                                          rt.depth.bindless_slot, view_mat->data(), fin, sun_dir,
                                          rtslice.ptr, rtslice.address)) {
                        rt_slot = rt_shadows_.shadow_slot();
                    }
                }
            }
            if (rt_slot == 0u && sun_shadows_.valid()) {
                sun_shadows_.compute_shadow_from_view(view_mat->data(), shadow_from_view);
                shadow_slot = sun_shadows_.map_slot();
            }
        }
        if (lighting_.dispatch(f.cmd, ctx_, descriptors_, rt.color.image, rt.color.view, rt.color.bindless_slot,
                               rt.depth.bindless_slot, w, h, fin, pslice.ptr, pslice.address, want_fxaa,
                               hdr_slot, 0u, lights_bda, light_cnt, shadow_slot,
                               shadow_slot != 0u ? shadow_from_view : nullptr, rt_slot)) {
            rt.color.layout = VK_IMAGE_LAYOUT_GENERAL;
        }
    } else if (want_fxaa) {
        auto pslice = f.params.allocate(sizeof(vk::LightingComputeParams), 16);
        if (!pslice.ptr) return;

        transition_resource(rt.color, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        if (lighting_.dispatch_fxaa_only(f.cmd, ctx_, descriptors_, rt.color.image, rt.color.view, w, h,
                                         pslice.ptr, pslice.address)) {
            rt.color.layout = VK_IMAGE_LAYOUT_GENERAL;
        }
    }

    if (want_taa) {
        auto tslice = f.params.allocate(sizeof(vk::TaaComputeParams), 16);
        if (!tslice.ptr) return;

        transition_resource(rt.color, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        transition_resource(rt.depth, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

        if (lighting_.dispatch_taa(f.cmd, ctx_, descriptors_, rt.color.image, rt.color.bindless_slot,
                                   rt.depth.bindless_slot, w, h, in, taa_prev_from_curr, taa_valid_reproj,
                                   taa_reset, curr_jitter_uv, taa_prev_jitter_uv_, tslice.ptr, tslice.address)) {
            rt.color.layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            if (curr_jitter_uv) {
                taa_prev_jitter_uv_[0] = curr_jitter_uv[0];
                taa_prev_jitter_uv_[1] = curr_jitter_uv[1];
            }
        }
    }
}

void VkBackend::prepare_interp_draws() {
    const std::size_t ndraws = blend_older_.summaries.size();
    prepared_draws_.resize(ndraws);
    const bool new_buf_ok = raw_new_slot_ >= 0 && raw_ring_[raw_new_slot_].buf != VK_NULL_HANDLE &&
                            raw_ring_[raw_new_slot_].cap >= blend_newer_.raw_verts.size() * sizeof(vk::RawGpuVertex);

    // Count skinned draw occurrences per key_hash (low 16 bits = older frame, high 16 bits = newer frame)
    // so unique skinned entities (boss monsters, hunter, weapons) get a generous deformation limit for
    // fast animations while instanced herds keep a tighter guard against cross-instance stretching.
    interp_key_counts_.clear();
    if (new_buf_ok) {
        for (const auto &s : blend_older_.summaries) {
            if (s.skinned) interp_key_counts_[s.key_hash] += 1u;
        }
        for (const auto &s : blend_newer_.summaries) {
            if (s.skinned) interp_key_counts_[s.key_hash] += 0x10000u;
        }
    }
    constexpr float kMaxDeformUnique = 600.0f;
    constexpr float kMaxDeformShared = 180.0f;
    std::uint32_t unique_deform_bits = 0;
    std::uint32_t shared_deform_bits = 0;
    std::memcpy(&unique_deform_bits, &kMaxDeformUnique, sizeof(float));
    std::memcpy(&shared_deform_bits, &kMaxDeformShared, sizeof(float));

    for (std::size_t i = 0; i < ndraws; ++i) {
        PreparedDraw &pd = prepared_draws_[i];
        const std::uint32_t old_cnt = (i < blend_older_.draw_raw_count.size()) ? blend_older_.draw_raw_count[i] : 0u;
        if (old_cnt == 0u) {
            pd.active = false;
            continue;
        }
        pd.active = true;
        const auto &older = blend_older_.summaries[i];
        pd.eye_old = interpolation::multiply(older.view, older.world);
        pd.has_inv_world_old = interpolation::affine_inverse(older.world, pd.inv_world_old);
        pd.old_start = (i < blend_older_.draw_raw_start.size()) ? blend_older_.draw_raw_start[i] : 0u;
        pd.new_start = 0u;
        pd.max_deform_bits = shared_deform_bits;
        pd.has_rigid = false;
        pd.follow_cam = false;
        pd.has_vertex_pair = false;
        pd.proj_differs = false;

        const std::int32_t partner = (i < last_matching_.newer_of.size()) ? last_matching_.newer_of[i] : interpolation::Matching::kNoPartner;
        pd.has_partner = (partner >= 0 && static_cast<std::size_t>(partner) < blend_newer_.summaries.size());
        if (pd.has_partner) {
            const std::size_t pi = static_cast<std::size_t>(partner);
            pd.partner = pi;
            const auto &newer = blend_newer_.summaries[pi];
            pd.eye_new = interpolation::multiply(newer.view, newer.world);
            pd.has_rigid = interpolation::rigid_between(pd.eye_old, pd.eye_new, pd.rm);
            pd.proj_differs = (older.projection != newer.projection);
            if (older.skinned && new_buf_ok) {
                const std::uint32_t new_cnt = (pi < blend_newer_.draw_raw_count.size()) ? blend_newer_.draw_raw_count[pi] : 0u;
                const std::uint32_t cand_new_start = (pi < blend_newer_.draw_raw_start.size()) ? blend_newer_.draw_raw_start[pi] : 0u;
                if (old_cnt == new_cnt &&
                    static_cast<std::size_t>(cand_new_start) + new_cnt <= blend_newer_.raw_verts.size() &&
                    interpolation::affine_inverse(newer.world, pd.inv_world_new)) {
                    pd.new_start = cand_new_start;
                    pd.has_vertex_pair = true;
                    const auto kc = interp_key_counts_.find(older.key_hash);
                    pd.max_deform_bits = (kc != interp_key_counts_.end() && kc->second == 0x00010001u)
                                             ? unique_deform_bits
                                             : shared_deform_bits;
                }
            }
        } else if (older.eligible && last_cam_motion_.valid) {
            pd.follow_cam = true;
        }
    }
}

void VkBackend::render_interpolated_subframe(float phase) {
    if (prof_.enabled) ++prof_.interp_calls;
    ScopedTimer prof_scope(prof_.enabled, prof_.interp_ns);
    if (!interp_can_blend_ || blend_older_.raw_verts.empty() || raw_old_slot_ < 0 ||
        raw_ring_[raw_old_slot_].buf == VK_NULL_HANDLE) {
        // Cut / 2D / movie screen: present latest rendered frame at target cadence so pacing & MangoHUD stay locked.
        if (last_shown_slot_ != 0) {
            show_to_swapchain(last_shown_slot_, last_shown_uv_[0], last_shown_uv_[1], last_shown_linear_);
        }
        return;
    }
    SnapRawBuf &cur_old_buf = raw_ring_[raw_old_slot_];
    SnapRawBuf *cur_new_buf_ptr = (raw_new_slot_ >= 0) ? &raw_ring_[raw_new_slot_] : nullptr;

    ensure_recording();
    end_render_pass();

    const std::uint32_t pw = blend_older_.width * static_cast<std::uint32_t>(scale_);
    const std::uint32_t ph = blend_older_.height * static_cast<std::uint32_t>(scale_);
    if (interp_rt_.color.image == VK_NULL_HANDLE || interp_rt_.color.width != pw || interp_rt_.color.height != ph) {
        retire_image(interp_rt_.color);
        retire_image(interp_rt_.depth);
        interp_rt_.width = blend_older_.width;
        interp_rt_.height = blend_older_.height;
        interp_rt_.stride = blend_older_.width;
        interp_rt_.color.create(ctx_, &descriptors_, pw, ph, VK_FORMAT_R8G8B8A8_UNORM,
                                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                    VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                VK_IMAGE_ASPECT_COLOR_BIT);
        interp_rt_.depth.create(ctx_, &descriptors_, pw, ph, VK_FORMAT_D32_SFLOAT,
                                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                VK_IMAGE_ASPECT_DEPTH_BIT);
    }
    interp_rt_.address = blend_older_.shown_addr;

    // Velocity-matched cubic Hermite phase smoothing (zero extra frame latency):
    // Applied uniformly to the entire scene (characters + terrain) so skinned models never desync from the ground.
    float scene_phase = phase;
    if (has_prev_cam_ && last_cam_motion_.valid && prev_cam_motion_.valid && phase > 0.0f && phase < 1.0f) {
        const float a_curr = last_cam_motion_.angle;
        const float a_prev = prev_cam_motion_.angle;
        if (a_curr > 0.01f && a_prev > 0.01f) {
            const float ratio = std::clamp(a_prev / a_curr, 0.7f, 1.3f);
            const float m0 = 0.5f * (ratio + 1.0f);
            const float t = phase;
            const float t2 = t * t;
            const float t3 = t2 * t;
            scene_phase = std::clamp((t3 - 2.0f * t2 + t) * m0 + (-2.0f * t3 + 3.0f * t2) + (t3 - t2), 0.0f, 1.0f);
        }
    }

    // 1. Build per-draw interpolated DrawTransform array from precomputed per-frame invariants.
    FrameSlice &f = frames_[frame_slot_];
    const std::size_t ndraws = blend_older_.summaries.size();
    auto tslice = f.params.allocate(ndraws * sizeof(vk::DrawTransform), 16);
    if (!tslice.ptr) {
        show_to_swapchain(last_shown_slot_, last_shown_uv_[0], last_shown_uv_[1], last_shown_linear_);
        return;
    }
    auto pack_3x4 = [](const interpolation::Matrix &m, float out[12]) {
        // Column-major 4x4 -> 3 vec4 rows for dot(row, vec4(p, 1.0)).
        out[0] = m[0]; out[1] = m[4]; out[2] = m[8];  out[3] = m[12];
        out[4] = m[1]; out[5] = m[5]; out[6] = m[9];  out[7] = m[13];
        out[8] = m[2]; out[9] = m[6]; out[10] = m[10]; out[11] = m[14];
    };
    auto *transforms = reinterpret_cast<vk::DrawTransform *>(tslice.ptr);
    const bool zero_phase = (scene_phase <= 0.0f);
    const bool has_cam_rt = !zero_phase && last_cam_motion_.valid;
    const interpolation::Matrix cam_rt =
        has_cam_rt ? interpolation::rigid_at(last_cam_motion_, scene_phase) : interpolation::identity();
    bool has_cached_cam_view = false;
    interpolation::Matrix cached_cam_view{};
    float cached_cam_mv[12]{};

    for (std::size_t i = 0; i < ndraws; ++i) {
        const PreparedDraw &pd = prepared_draws_[i];
        if (!pd.active) continue; // zero vertices in raw_verts -> never read by place_v2.comp!

        const auto &older = blend_older_.summaries[i];
        vk::DrawTransform &tr = transforms[i];
        if (zero_phase) {
            pack_3x4(older.view, tr.mv_old);
            pack_3x4(older.view, tr.mv_new);
            std::copy_n(older.projection.data(), 16, tr.proj);
            tr.pair_info[0] = pd.old_start;
            tr.pair_info[1] = pd.new_start;
            tr.pair_info[2] = 0u;
            tr.pair_info[3] = pd.max_deform_bits;
        } else {
            if (pd.has_partner && pd.has_rigid) {
                // Algebraically: (R(t) * view * world) * inv(world) == R(t) * view (avoids a 4x4 multiply).
                const interpolation::Matrix r_t = interpolation::rigid_at(pd.rm, scene_phase);
                const interpolation::Matrix world_to_eye_old =
                    pd.has_inv_world_old ? interpolation::multiply(r_t, older.view) : older.view;
                pack_3x4(world_to_eye_old, tr.mv_old);
                if (pd.has_vertex_pair) {
                    const interpolation::Matrix eye_blend = interpolation::multiply(r_t, pd.eye_old);
                    const auto world_to_eye_new = interpolation::multiply(eye_blend, pd.inv_world_new);
                    pack_3x4(world_to_eye_new, tr.mv_new);
                } else {
                    std::copy_n(tr.mv_old, 12, tr.mv_new);
                }
            } else if (!pd.has_partner && pd.follow_cam && has_cam_rt && pd.has_inv_world_old) {
                // Hoisted cam_rt * older.view, cached across consecutive draws sharing the same view matrix.
                if (!has_cached_cam_view || cached_cam_view != older.view) {
                    cached_cam_view = older.view;
                    pack_3x4(interpolation::multiply(cam_rt, older.view), cached_cam_mv);
                    has_cached_cam_view = true;
                }
                std::copy_n(cached_cam_mv, 12, tr.mv_old);
                std::copy_n(cached_cam_mv, 12, tr.mv_new);
            } else {
                interpolation::Matrix eye_blend = pd.eye_old;
                if (pd.has_partner) {
                    if (last_cam_motion_.valid) {
                        eye_blend = interpolation::blend_eye(pd.eye_old, pd.eye_new, last_cam_motion_, scene_phase);
                    } else {
                        eye_blend = interpolation::blend_affine(pd.eye_old, pd.eye_new, scene_phase);
                    }
                } else if (pd.follow_cam) {
                    eye_blend = interpolation::multiply(cam_rt, pd.eye_old);
                }

                const interpolation::Matrix world_to_eye_old =
                    pd.has_inv_world_old ? interpolation::multiply(eye_blend, pd.inv_world_old) : older.view;
                pack_3x4(world_to_eye_old, tr.mv_old);

                if (pd.has_vertex_pair) {
                    const auto world_to_eye_new = interpolation::multiply(eye_blend, pd.inv_world_new);
                    pack_3x4(world_to_eye_new, tr.mv_new);
                } else {
                    std::copy_n(tr.mv_old, 12, tr.mv_new);
                }
            }

            if (pd.proj_differs) {
                const auto proj_blend = interpolation::blend_linear(older.projection, blend_newer_.summaries[pd.partner].projection, scene_phase);
                std::copy_n(proj_blend.data(), 16, tr.proj);
            } else {
                std::copy_n(older.projection.data(), 16, tr.proj);
            }
            tr.pair_info[0] = pd.old_start;
            tr.pair_info[1] = pd.new_start;
            tr.pair_info[2] = pd.has_vertex_pair ? 1u : 0u;
            tr.pair_info[3] = pd.max_deform_bits;
        }

        const auto &vp = blend_older_.viewports[i];
        tr.vp_scale[0] = vp[0];
        tr.vp_scale[1] = vp[1];
        tr.vp_scale[2] = vp[2];
        tr.vp_scale[3] = 0.0f;
        tr.vp_center[0] = vp[3] - vp[6];
        tr.vp_center[1] = vp[4] - vp[7];
        tr.vp_center[2] = vp[5];
        tr.vp_center[3] = 1.0f;
    }

    // 2. GPU Compute: dual-stream per-vertex interpolation + Sutherland-Hodgman near-plane clipping!
    const std::uint32_t raw_count = static_cast<std::uint32_t>(blend_older_.raw_verts.size());
    const VkDeviceAddress new_addr = cur_new_buf_ptr ? cur_new_buf_ptr->addr : 0;
    interpolator_.dispatch_place_v2(f.cmd, frame_slot_, cur_old_buf.addr, new_addr, tslice.address, raw_count, scene_phase);
    cur_old_buf.last_gpu_timeline = ctx_.graphics_timeline_value + 1u;
    if (cur_new_buf_ptr) cur_new_buf_ptr->last_gpu_timeline = ctx_.graphics_timeline_value + 1u;

    // 3. Render interpolated 3D + 2D batches into interp_rt_ via Dynamic Rendering.
    const int aa = effective_aa_mode();
    bool pass_msaa = (aa == LiveSettings::kAaMsaa4x) && ensure_msaa_targets(pw, ph);
    bool pass_hdr = want_enhanced_3d_hdr() && ensure_hdr_targets(pw, ph, pass_msaa);
    const bool use_taa = (aa == LiveSettings::kAaTaa);

    float taa_jitter_ndc[2]{0.0f, 0.0f};
    float taa_jitter_uv[2]{0.0f, 0.0f};
    float taa_prev_from_curr[12]{1.0f, 0.0f, 0.0f, 0.0f,
                                 0.0f, 1.0f, 0.0f, 0.0f,
                                 0.0f, 0.0f, 1.0f, 0.0f};
    bool taa_valid_reproj = false;
    if (use_taa) {
        float jx = 0.0f, jy = 0.0f;
        get_taa_jitter(taa_sample_idx_++, jx, jy);
        const std::uint32_t sw = std::min(480u, blend_older_.width) * static_cast<std::uint32_t>(scale_);
        const std::uint32_t sh = std::min(272u, blend_older_.height) * static_cast<std::uint32_t>(scale_);
        taa_jitter_ndc[0] = jx * 2.0f / static_cast<float>(pw);
        taa_jitter_ndc[1] = jy * 2.0f / static_cast<float>(ph);
        taa_jitter_uv[0] = jx / static_cast<float>(std::max(sw, 1u));
        taa_jitter_uv[1] = jy / static_cast<float>(std::max(sh, 1u));
        if (taa_have_prev_) {
            if (phase > 0.0f && last_cam_motion_.valid) {
                const float dt = scene_phase - taa_prev_phase_;
                const interpolation::Matrix pfc = interpolation::rigid_at(last_cam_motion_, -dt);
                pack_3x4(pfc, taa_prev_from_curr);
                taa_valid_reproj = true;
            } else if (phase <= 0.0f && has_prev_cam_ && prev_cam_motion_.valid) {
                const float dt = 1.0f - taa_prev_phase_;
                const interpolation::Matrix pfc = interpolation::rigid_at(prev_cam_motion_, -dt);
                pack_3x4(pfc, taa_prev_from_curr);
                taa_valid_reproj = true;
            }
        }
        taa_prev_phase_ = scene_phase;
        taa_have_prev_ = true;
    }

    // Update Partitioned RT BVH or 2048x2048 Sun Shadow Map ONCE per 30 Hz game frame (phase <= 0).
    const interpolation::Matrix interp_view =
        (has_last_view_mat_ && has_cam_rt) ? interpolation::multiply(cam_rt, last_view_mat_) : last_view_mat_;
    if (phase <= 0.0f && has_last_view_mat_) {
        if (rt_shadows_.ready() && want_vk_effect(LiveSettings::kVkEffectRayTracing) && !lighting_.active_is_cave()) {
            rt_shadows_.update_bvh(f.cmd, ctx_, blend_older_.rt_static_verts, blend_older_.rt_dyn_verts);
            sun_shadows_.invalidate();
        } else {
            rt_shadows_.invalidate();
            render_sun_shadow_pass(interp_view, last_light_in_, blend_older_.shadow_verts, blend_older_.shadow_batches);
        }
    }

    transition_resource(interp_rt_.color, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    transition_resource(interp_rt_.depth, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

    VkRenderingAttachmentInfo color_att{};
    color_att.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    color_att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color_att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_att.clearValue.color = {{0.0f, 0.0f, 0.0f, 1.0f}};

    VkRenderingAttachmentInfo depth_att{};
    depth_att.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depth_att.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depth_att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depth_att.clearValue.depthStencil = {1.0f, 0u};

    if (pass_hdr) {
        transition_resource(hdr_color_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                            VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        if (pass_msaa) {
            transition_resource(msaa_hdr_color_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
            transition_resource(msaa_depth_, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
            color_att.imageView = msaa_hdr_color_.view;
            color_att.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
            color_att.resolveImageView = hdr_color_.view;
            color_att.resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            depth_att.imageView = msaa_depth_.view;
            depth_att.resolveMode = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT;
            depth_att.resolveImageView = interp_rt_.depth.view;
            depth_att.resolveImageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        } else {
            color_att.imageView = hdr_color_.view;
            depth_att.imageView = interp_rt_.depth.view;
        }
    } else if (pass_msaa) {
        transition_resource(msaa_color_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                            VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        transition_resource(msaa_depth_, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
        color_att.imageView = msaa_color_.view;
        color_att.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
        color_att.resolveImageView = interp_rt_.color.view;
        color_att.resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        depth_att.imageView = msaa_depth_.view;
        depth_att.resolveMode = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT;
        depth_att.resolveImageView = interp_rt_.depth.view;
        depth_att.resolveImageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    } else {
        color_att.imageView = interp_rt_.color.view;
        depth_att.imageView = interp_rt_.depth.view;
    }

    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0, 0}, {pw, ph}}, 1, 0,
                       1u, &color_att, &depth_att, nullptr};
    vkCmdBeginRendering(f.cmd, &ri);

    // Upload 2D/flat vertices once for this subframe.
    VkDeviceSize flat_base_off = 0;
    bool flat_uploaded = false;
    if (!blend_older_.flat_verts.empty()) {
        const std::size_t fb = blend_older_.flat_verts.size() * sizeof(GpuVertex);
        auto vsl = f.vertices.allocate(fb, 16);
        if (vsl.ptr) {
            std::memcpy(vsl.ptr, blend_older_.flat_verts.data(), fb);
            flat_base_off = vsl.offset;
            flat_uploaded = true;
        }
    }

    VkDescriptorSet set0 = descriptors_.set();
    vkCmdBindDescriptorSets(f.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines_.ge_layout, 0, 1, &set0, 0, nullptr);
    VkViewport vp_full{0.0f, 0.0f, static_cast<float>(pw), static_cast<float>(ph), 0.0f, 1.0f};
    vkCmdSetViewport(f.cmd, 0, 1, &vp_full);
    PassStateCache interp_state{};
    VkBuffer last_vb = VK_NULL_HANDLE;
    VkDeviceSize last_vb_off = UINT64_MAX;
    bool interp_saw_3d = false;
    bool interp_lit = false;
    bool interp_depth_snapped = false;

    std::array<vk::ViewPointLight, vk::LightingFrameInput::kMaxViewLights> interp_vlights{};
    std::uint32_t interp_vl_count = 0u;
    if (has_last_view_mat_) {
        interp_vl_count = build_view_lights(interp_view, last_vfx_lights_, interp_vlights);
    }

    for (const auto &rb : blend_older_.batches) {
        if (rb.v_count == 0 || rb.setup.target != blend_older_.shown_addr) continue;
        const auto R = [&](std::uint32_t reg) { return key_reg(rb.setup.key, reg); };
        const bool clear = (R(kClearMode) & 1u) != 0;

        // Snapshot opaque 3D depth once before depth-tested non-depth-writing blended particles (Soft Particles!).
        const bool is_soft_particle = !clear && !rb.setup.through &&
                                      (R(kAlphaBlendEnable) & 1u) != 0 && (R(kZTestEnable) & 1u) != 0 &&
                                      (R(kZWriteDisable) & 1u) != 0 &&
                                      want_vk_effect(LiveSettings::kVkEffectSoftParticles);
        if (is_soft_particle && !interp_depth_snapped) {
            vkCmdEndRendering(f.cmd);
            snapshot_soft_depth(interp_rt_);
            interp_depth_snapped = true;
            color_att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            depth_att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            vkCmdBeginRendering(f.cmd, &ri);
            vkCmdBindDescriptorSets(f.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines_.ge_layout, 0, 1, &set0, 0, nullptr);
            vkCmdSetViewport(f.cmd, 0, 1, &vp_full);
            interp_state.reset();
            last_vb = VK_NULL_HANDLE;
        }

        // Run enhanced lighting & AA strictly after 3D draws and before 2D HUD draws on 120/180 Hz subframes too!
        if (!clear) {
            if (!rb.setup.through) {
                interp_saw_3d = true;
            } else if (interp_saw_3d && !interp_lit && !rb.setup.self_sampled) {
                const bool need_boundary = pass_msaa || pass_hdr ||
                                           (lighting_.ready() &&
                                            (live_->enhanced_lighting.load(std::memory_order_relaxed) ||
                                             aa == LiveSettings::kAaFxaa || aa == LiveSettings::kAaTaa));
                if (need_boundary) {
                    vkCmdEndRendering(f.cmd);
                    apply_post_3d(interp_rt_, last_light_in_, taa_prev_from_curr, taa_valid_reproj, false,
                                  taa_jitter_uv, pass_hdr, interp_vlights.data(), interp_vl_count,
                                  has_last_view_mat_ ? &interp_view : nullptr);
                    interp_lit = true;
                    if (phase <= 0.0f) {
                        capture_clean_3d_snapshot(interp_rt_);
                    }
                    // Switch to 1x single-sample RGBA8 attachments for crisp 2D HUD draws!
                    pass_msaa = false;
                    pass_hdr = false;
                    color_att.imageView = interp_rt_.color.view;
                    color_att.resolveMode = VK_RESOLVE_MODE_NONE;
                    color_att.resolveImageView = VK_NULL_HANDLE;
                    depth_att.imageView = interp_rt_.depth.view;
                    depth_att.resolveMode = VK_RESOLVE_MODE_NONE;
                    depth_att.resolveImageView = VK_NULL_HANDLE;
                    transition_resource(interp_rt_.color, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                        VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
                    transition_resource(interp_rt_.depth, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                                        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
                    color_att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
                    depth_att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
                    vkCmdBeginRendering(f.cmd, &ri);
                    vkCmdBindDescriptorSets(f.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines_.ge_layout, 0, 1, &set0, 0, nullptr);
                    vkCmdSetViewport(f.cmd, 0, 1, &vp_full);
                    interp_state.reset();
                    last_vb = VK_NULL_HANDLE;
                }
            }
        }

        VkDeviceAddress params_addr = 0;
        const bool apply_taa_jitter = use_taa && !clear && !rb.setup.through;
        if (!bind_ge_draw_state(f.cmd, rb.setup, blend_older_.width, blend_older_.height, is_soft_particle, pass_msaa,
                                pass_hdr,
                                apply_taa_jitter ? taa_jitter_ndc[0] : 0.0f,
                                apply_taa_jitter ? taa_jitter_ndc[1] : 0.0f,
                                apply_taa_jitter ? -0.5f : 0.0f,
                                params_addr, interp_state)) {
            continue;
        }

        std::uint32_t draw_vcount = rb.v_count;
        std::uint32_t first_vertex = 0;
        if (rb.use_gpu_raw) {
            if (static_cast<std::uint64_t>(rb.v_offset) + rb.v_count > interpolator_.max_in_vertices()) continue;
            VkBuffer gpu_vb = interpolator_.out_vertex_buffer(frame_slot_);
            if (last_vb != gpu_vb || last_vb_off != 0) {
                const VkDeviceSize zero = 0;
                vkCmdBindVertexBuffers(f.cmd, 0, 1, &gpu_vb, &zero);
                last_vb = gpu_vb;
                last_vb_off = 0;
            }
            // place_v2.comp emits 2 clipped triangles (6 verts) per input triangle (3 verts).
            first_vertex = rb.v_offset * 2u;
            draw_vcount = rb.v_count * 2u;
        } else {
            if (!flat_uploaded) continue;
            if (last_vb != f.vertices.buffer || last_vb_off != flat_base_off) {
                vkCmdBindVertexBuffers(f.cmd, 0, 1, &f.vertices.buffer, &flat_base_off);
                last_vb = f.vertices.buffer;
                last_vb_off = flat_base_off;
            }
            first_vertex = rb.v_offset;
        }
        vk::DrawPush push{params_addr, 0u};
        vkCmdPushConstants(f.cmd, pipelines_.ge_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(push), &push);
        vkCmdDraw(f.cmd, draw_vcount, 1, first_vertex, 0);
    }

    vkCmdEndRendering(f.cmd);
    if (interp_saw_3d && !interp_lit) {
        apply_post_3d(interp_rt_, last_light_in_, taa_prev_from_curr, taa_valid_reproj, false, taa_jitter_uv,
                      pass_hdr, interp_vlights.data(), interp_vl_count,
                      has_last_view_mat_ ? &interp_view : nullptr);
    }
    transition_resource(interp_rt_.color, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

    const float uv_x = std::min(1.0f, 480.0f / static_cast<float>(blend_older_.width));
    const float uv_y = std::min(1.0f, 272.0f / static_cast<float>(blend_older_.height));
    show_to_swapchain(interp_rt_.color.bindless_slot, uv_x, uv_y, true);
}

void VkBackend::present(std::uint32_t address, std::uint32_t stride, std::uint32_t pixel_format) {
    draw_batch();
    have_state_ = false;
    last_state_gen_ = 0;
    ++frame_number_;

    // Publish quest HUD detection for Better Sharpness, Gathering Assistance, and Hunting Horn Songs.
    live_->quest_hud.store(quest_hud_this_flip_, std::memory_order_relaxed);
    quest_hud_this_flip_ = false;
    hud_layer_ = gpu::DrawInfo::kNoLayer;
    const float wide = live_->wide_aspect.load(std::memory_order_relaxed);
    frame_stretch_ = wide > 0.0f ? (wide / (480.0f / 272.0f)) : 1.0f;

    const bool is_vram_addr = (address & 0x0F000000u) == 0x04000000u;
    const std::uint32_t off = (address & 0x1FFFFFFFu) & 0x001FFFFFu;
    if (is_vram_addr && off != shown_addrs_[0]) {
        shown_addrs_[1] = shown_addrs_[0];
        shown_addrs_[0] = off;
    }
    auto it = is_vram_addr ? targets_.find(off) : targets_.end();
    bool have_gpu_frame = (it != targets_.end() && it->second.color.image != VK_NULL_HANDLE && it->second.last_draw + 4 >= frame_number_);
    if (have_gpu_frame) {
        RenderTarget &rt = it->second;
        const bool main_rt_skipped = (subframes_ > 1 && interp_can_blend_);
        if (frame_had_3d_ && !lit_3d_this_frame_ && !main_rt_skipped) {
            const bool had_hdr = main_hdr_started_;
            end_render_pass();
            std::array<vk::ViewPointLight, vk::LightingFrameInput::kMaxViewLights> vlights{};
            const std::uint32_t vl_count = has_cur_view_mat_
                                               ? build_view_lights(cur_view_mat_, cur_vfx_lights_, vlights)
                                               : 0u;
            apply_post_3d(rt, cur_light_in_, nullptr, false, false, main_taa_jitter_uv_, had_hdr,
                          vlights.data(), vl_count);
            lit_3d_this_frame_ = true;
        }
        if (frame_had_3d_ && !captured_clean_3d_) {
            capture_clean_3d_snapshot(rt);
        }
        ensure_recording();
        end_render_pass();
        copy_shown_to_vram(&rt);
        transition_resource(rt.color, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        const float uv_x = std::min(1.0f, 480.0f / static_cast<float>(rt.width));
        const float uv_y = std::min(1.0f, 272.0f / static_cast<float>(rt.height));
        last_shown_slot_ = rt.color.bindless_slot;
        last_shown_uv_[0] = uv_x;
        last_shown_uv_[1] = uv_y;
        last_shown_linear_ = true;
    } else {
        copy_shown_to_vram(nullptr);
        // CPU-drawn screen (boot logos, PSMF intro videos in VRAM 0x04xxxxxx OR main RAM 0x08xxxxxx).
        const std::uint32_t bpp = pixel_format == 3u ? 4u : 2u;
        const std::uint32_t min_stride =
            ((address & 0x0F000000u) == 0x04000000u || (address & 0x0F000000u) == 0x08000000u) ? 480u : 512u;
        const std::uint32_t s = std::max(stride, min_stride);
        const std::size_t needed = static_cast<std::size_t>(s) * 271u * bpp + 480u * bpp;
        decode_buf_.assign(480u * 272u, 0xFF000000u);
        if (memory_.contains(address, needed)) {
            const std::uint8_t *base = memory_.raw_pointer(address, needed);
            if (base) {
                for (std::uint32_t y = 0; y < 272u; ++y) {
                    const std::uint8_t *row = base + static_cast<std::size_t>(y) * s * bpp;
                    for (std::uint32_t x = 0; x < 480u; ++x) {
                        decode_buf_[y * 480u + x] = framebuffer_pixel(row + x * bpp, pixel_format);
                    }
                }
            }
        }
        upload_texture_2d(cpu_display_tex_, 480u, 272u, decode_buf_.data());
        last_shown_slot_ = cpu_display_tex_.bindless_slot;
        last_shown_uv_[0] = 1.0f;
        last_shown_uv_[1] = 1.0f;
        last_shown_linear_ = true;
    }

    const auto now = std::chrono::steady_clock::now();
    if (flip_time_.time_since_epoch().count() != 0) {
        const auto dt = now - flip_time_;
        if (dt >= std::chrono::milliseconds(16) && dt <= std::chrono::milliseconds(60)) {
            // EMA-smoothed flip period eliminates OS wakeup jitter from subframe cadence.
            const auto prev_us = std::chrono::duration_cast<std::chrono::microseconds>(flip_period_).count();
            const auto dt_us = std::chrono::duration_cast<std::chrono::microseconds>(dt).count();
            const auto ema_us = (prev_us * 85 + dt_us * 15) / 100;
            flip_period_ = std::chrono::microseconds(std::clamp<long long>(ema_us, 16000, 50000));
        }
    }
    flip_time_ = now;

    if (subframes_ > 1) {
        cur_snap_.shown_addr = off;
        std::swap(cur_snap_.shadow_verts, cur_shadow_verts_);
        std::swap(cur_snap_.shadow_batches, cur_shadow_batches_);
        std::swap(cur_snap_.rt_static_verts, cur_rt_static_verts_);
        std::swap(cur_snap_.rt_dyn_verts, cur_rt_dyn_verts_);
        cur_shadow_verts_.clear();
        cur_shadow_batches_.clear();
        cur_rt_static_verts_.clear();
        cur_rt_dyn_verts_.clear();
        cur_shadow_batch_start_ = 0u;
        if (it != targets_.end()) {
            cur_snap_.width = it->second.width;
            cur_snap_.height = it->second.height;
        }
        interpolation::mark_eligible(cur_snap_.summaries, off);
        const bool was_blending = interp_can_blend_ && last_cam_motion_.valid;
        prev_cam_motion_ = last_cam_motion_;
        has_prev_cam_ = was_blending;
        interp_can_blend_ = false;
        if (have_newer_snap_) {
            // Rotate snapshots in-place via swap + clear to retain multi-MB vector capacities (zero heap churn).
            std::swap(blend_older_, blend_newer_);
            std::swap(blend_newer_, cur_snap_);
            cur_snap_.clear();
            cur_raw_batch_start_ = 0;

            raw_old_slot_ = raw_new_slot_;
            raw_new_slot_ = -1;
            if (!blend_newer_.raw_verts.empty()) {
                ScopedTimer up_scope(prof_.enabled, prof_.up_raw_ns);
                const int slot = raw_next_slot_;
                raw_next_slot_ = (raw_next_slot_ + 1) % 3;
                if (upload_snap_raw(raw_ring_[slot], blend_newer_.raw_verts)) {
                    raw_new_slot_ = slot;
                }
            }

            if (have_gpu_frame && !blend_older_.unblendable && !blend_newer_.unblendable &&
                !blend_older_.raw_verts.empty() && interpolator_.ready()) {
                interpolation::CutThresholds thr{};
                const auto mt0 = prof_.enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                last_matching_ = interpolator_.matcher().match(blend_older_.summaries, blend_newer_.summaries, thr);
                if (prof_.enabled) {
                    prof_.match_ns += static_cast<std::uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - mt0).count());
                }
                if (last_matching_.cut == nullptr) {
                    last_cam_motion_ = last_matching_.camera_found ? interpolation::rigid_motion(last_matching_.camera)
                                                                   : interpolation::RigidMotion{};
                    bool ok_old = (raw_old_slot_ >= 0 && raw_ring_[raw_old_slot_].buf != VK_NULL_HANDLE);
                    if (!ok_old) {
                        ScopedTimer up_scope(prof_.enabled, prof_.up_raw_ns);
                        const int slot = raw_next_slot_;
                        raw_next_slot_ = (raw_next_slot_ + 1) % 3;
                        if (upload_snap_raw(raw_ring_[slot], blend_older_.raw_verts)) {
                            raw_old_slot_ = slot;
                            ok_old = true;
                        }
                    }
                    if (prof_.enabled) {
                        prof_.raw_verts += blend_older_.raw_verts.size();
                        prof_.flat_verts += blend_older_.flat_verts.size();
                    }
                    interp_can_blend_ = ok_old;
                    if (interp_can_blend_) {
                        prepare_interp_draws();
                    }
                } else {
                    interpolator_.matcher().forget_motion();
                    has_prev_cam_ = false;
                    lighting_.reset_taa_history();
                    taa_have_prev_ = false;
                }
            } else {
                interpolator_.matcher().forget_motion();
                has_prev_cam_ = false;
                lighting_.reset_taa_history();
                taa_have_prev_ = false;
            }
        } else {
            std::swap(blend_newer_, cur_snap_);
            cur_snap_.clear();
            cur_raw_batch_start_ = 0;
            raw_old_slot_ = -1;
            raw_new_slot_ = -1;
            if (!blend_newer_.raw_verts.empty()) {
                const int slot = raw_next_slot_;
                raw_next_slot_ = (raw_next_slot_ + 1) % 3;
                if (upload_snap_raw(raw_ring_[slot], blend_newer_.raw_verts)) {
                    raw_new_slot_ = slot;
                }
            }
            have_newer_snap_ = true;
        }
        // Monotonic forward motion: show t = 0.0 (older frame) now, then t = 1/N .. (N-1)/N in tick()!
        if (interp_can_blend_) {
            render_interpolated_subframe(0.0f);
        } else {
            // If lighting on main_rt was skipped assuming frame-gen would present interp_rt_,
            // run it now before falling back to main_rt on a camera cut (prevents 1-frame unlit flash).
            if (have_gpu_frame && frame_had_3d_ && !lit_3d_this_frame_ && it != targets_.end()) {
                const bool had_hdr = main_hdr_started_;
                end_render_pass();
                std::array<vk::ViewPointLight, vk::LightingFrameInput::kMaxViewLights> vlights{};
                const std::uint32_t vl_count = has_cur_view_mat_
                                                   ? build_view_lights(cur_view_mat_, cur_vfx_lights_, vlights)
                                                   : 0u;
                apply_post_3d(it->second, cur_light_in_, nullptr, false, true, main_taa_jitter_uv_, had_hdr,
                              vlights.data(), vl_count);
                transition_resource(it->second.color, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                lit_3d_this_frame_ = true;
            }
            show_to_swapchain(last_shown_slot_, last_shown_uv_[0], last_shown_uv_[1], last_shown_linear_);
        }
        interp_active_ = true;
        next_subframe_ = 1;
        next_subframe_at_ = now + flip_period_ / subframes_;
    } else {
        show_to_swapchain(last_shown_slot_, last_shown_uv_[0], last_shown_uv_[1], last_shown_linear_);
    }

    last_menu_draw_ = now;
    if (frame_had_3d_) {
        last_light_in_ = cur_light_in_;
        if (has_cur_view_mat_) {
            last_view_mat_ = cur_view_mat_;
            has_last_view_mat_ = true;
        }
        last_vfx_lights_ = cur_vfx_lights_;
    }
    cur_vfx_lights_.clear();
    cur_shadow_verts_.clear();
    cur_shadow_batches_.clear();
    cur_rt_static_verts_.clear();
    cur_rt_dyn_verts_.clear();
    cur_shadow_batch_start_ = 0u;
    has_cur_view_mat_ = false;
    lighting_.end_frame_zone_vote();
    frame_had_3d_ = false;
    captured_clean_3d_ = false;
    lit_3d_this_frame_ = false;
    main_depth_snapped_ = false;
    main_msaa_started_ = false;
    main_hdr_started_ = false;
    main_3d_finished_ = false;
    main_taa_jitter_set_ = false;
    cur_light_in_ = vk::LightingFrameInput{};
    frame_tex_cache_.clear();
    if ((frame_number_ % 60u) == 0u) {
        const std::uint64_t margin = static_cast<std::uint64_t>(std::max(subframes_, 1)) * 3u + 4u;
        if (pack_streamer_.trim_vram(ctx_, descriptors_, ctx_.graphics_timeline_value + margin, frame_number_)) {
            pack_cache_.clear();
        }
    }
    apply_live_scale();
    apply_live_fps();

    if (prof_.enabled) {
        ++prof_.frames;
        if (prof_.frames >= 30u) {
            const auto pnow = std::chrono::steady_clock::now();
            const double sec = std::chrono::duration<double>(pnow - prof_.window_start).count();
            const double inv_f = 1.0 / static_cast<double>(prof_.frames);
            const double ms = 1e-6 * inv_f;
            std::fprintf(stderr,
                "[vk][prof] gameFPS=%.1f draws=%.0f batches=%.0f rawV=%.0f flatV=%.0f texHit=%.0f texMiss=%.0f packQ=%.0f | "
                "apply=%.2fms batch=%.2fms drawStart=%.2fms match=%.2fms upRaw=%.2fms interp=%.2fms(%.1fx) light=%.2fms waitGPU=%.2fms acq=%.2fms pres=%.2fms\n",
                sec > 0.0 ? static_cast<double>(prof_.frames) / sec : 0.0,
                static_cast<double>(prof_.draws) * inv_f,
                static_cast<double>(prof_.batches) * inv_f,
                static_cast<double>(prof_.raw_verts) * inv_f,
                static_cast<double>(prof_.flat_verts) * inv_f,
                static_cast<double>(prof_.tex_hits) * inv_f,
                static_cast<double>(prof_.tex_misses) * inv_f,
                static_cast<double>(prof_.pack_queries) * inv_f,
                static_cast<double>(prof_.apply_ns) * ms,
                static_cast<double>(prof_.batch_ns) * ms,
                static_cast<double>(prof_.draw_start_ns) * ms,
                static_cast<double>(prof_.match_ns) * ms,
                static_cast<double>(prof_.up_raw_ns) * ms,
                static_cast<double>(prof_.interp_ns) * ms,
                static_cast<double>(prof_.interp_calls) * inv_f,
                static_cast<double>(prof_.light_ns) * ms,
                static_cast<double>(prof_.wait_gpu_ns) * ms,
                static_cast<double>(prof_.acq_ns) * ms,
                static_cast<double>(prof_.pres_ns) * ms);
            std::fflush(stderr);
            prof_ = ProfStats{};
            prof_.enabled = true;
            prof_.window_start = pnow;
        }
    }
}

std::optional<std::chrono::steady_clock::time_point> VkBackend::next_wakeup() const {
    if (interp_active_ && subframes_ > 1 && next_subframe_ < subframes_) {
        return next_subframe_at_;
    }
    if (overlay_visible_ && overlay_continuous_) {
        return last_menu_draw_ + std::chrono::milliseconds(16);
    }
    return std::nullopt;
}

void VkBackend::tick() {
    const auto now = std::chrono::steady_clock::now();
    if (interp_active_ && subframes_ > 1 && next_subframe_ < subframes_ && now >= next_subframe_at_) {
        const float phase = static_cast<float>(next_subframe_) / static_cast<float>(subframes_);
        render_interpolated_subframe(phase);
        ++next_subframe_;
        if (next_subframe_ < subframes_) {
            next_subframe_at_ = flip_time_ + (flip_period_ * next_subframe_) / subframes_;
        } else {
            interp_active_ = false;
        }
        last_menu_draw_ = now;
        return;
    }
    if (overlay_visible_ && overlay_continuous_ && last_shown_slot_ != 0) {
        show_to_swapchain(last_shown_slot_, last_shown_uv_[0], last_shown_uv_[1], last_shown_linear_);
        last_menu_draw_ = now;
    }
}

} // namespace

std::unique_ptr<gpu::Backend> make_vk_backend(SDL_Window *window, int scale, psprecomp::GuestMemory &memory,
                                              bool smooth_textures, OverlayImage *overlay, int target_fps,
                                              LiveSettings *live) {
    auto backend = std::make_unique<VkBackend>(window, scale, memory, smooth_textures, overlay, target_fps, live);
    if (!backend->init()) return nullptr;
    return backend;
}

} // namespace mhf::psp
