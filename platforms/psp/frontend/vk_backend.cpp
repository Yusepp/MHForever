// Modern Vulkan 1.3+ drawing backend for the GE (see VULKAN.md).
//
// Phase 1 + Phase 2 implementation:
// - Vulkan 1.3 dynamic rendering, Synchronization2, timeline semaphores, VMA, volk via SDL3.
// - Global bindless texture array (16384 UPDATE_AFTER_BIND slots) + 64 immutable samplers.
// - Persistently mapped per-frame ring buffers for GpuVertex, BDA DrawParams, and texture staging.
// - Specialized SPIR-V GE pipelines with Extended Dynamic State 1/2/3 + on-disk VkPipelineCache.
// - Upscaled VkRenderTarget pool, RTT + self-sampling copies, block transfers, overlays, and menu blur.

#include "vk_backend.hpp"
#include "vulkan/vk_buffers.hpp"
#include "vulkan/vk_context.hpp"
#include "vulkan/vk_descriptors.hpp"
#include "vulkan/vk_pipelines.hpp"
#include "vulkan/vk_swapchain.hpp"

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
constexpr std::size_t kVertexRingBytes = 16u * 1024u * 1024u;
constexpr std::size_t kParamsRingBytes = 2u * 1024u * 1024u;
constexpr std::size_t kStagingRingBytes = 32u * 1024u * 1024u;
constexpr std::size_t kMaxCachedTextures = 4096u;

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
                VkFormat fmt, VkImageUsageFlags usage, VkImageAspectFlags asp, std::uint32_t mips = 1) {
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
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
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
    VkImageResource color;
    VkImageResource depth;

    [[nodiscard]] std::uint32_t bytes_per_pixel() const noexcept { return format == 3u ? 4u : 2u; }
};

struct CachedTexture {
    VkImageResource image;
    std::uint64_t last_used = 0;
    std::uint32_t width = 0, height = 0;
};

struct DrawSetup {
    DrawKey key{};
    std::uint32_t tex_slot = 0;
    float uv_xform[4]{1.0f, 1.0f, 0.0f, 0.0f};
    bool through = false;
    bool linear_override = false;
    bool texture_mips = false;
    float tex_width = 1.0f, tex_height = 1.0f;
    std::uint32_t target = 0;
    bool self_sampled = false;
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
    void begin_target_pass(RenderTarget &rt);
    void transition_resource(VkImageResource &res, VkImageLayout new_layout,
                             VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access);

    RenderTarget &ensure_target(std::uint32_t address, std::uint32_t stride, std::uint32_t format, std::uint32_t min_h);
    void apply_state(const gpu::Ge &ge);
    void draw_batch();
    void upload_texture_2d(VkImageResource &dst, std::uint32_t w, std::uint32_t h, const void *rgba_pixels);
    void upload_overlay_if_needed();
    void show_to_swapchain(std::uint32_t tex_slot, float uv_max_x, float uv_max_y, bool linear_filter);
    void submit_and_present(std::uint32_t swap_img_idx);
    void apply_live_scale();
    void read_back();

    SDL_Window *window_ = nullptr;
    int scale_ = 4;
    psprecomp::GuestMemory &memory_;
    OverlayImage *overlay_ = nullptr;
    int target_fps_ = 30;
    LiveSettings own_live_;
    LiveSettings *live_ = nullptr;

    vk::Context ctx_;
    vk::Swapchain swapchain_;
    vk::BindlessDescriptors descriptors_;
    vk::Pipelines pipelines_;

    std::array<FrameSlice, kFramesInFlight> frames_{};
    int frame_slot_ = 0;
    std::uint64_t frame_number_ = 1;
    bool recording_ = false;
    bool in_render_pass_ = false;
    std::uint32_t active_target_addr_ = UINT32_MAX;

    std::unordered_map<std::uint32_t, RenderTarget> targets_;
    std::unordered_map<std::uint64_t, CachedTexture> textures_;
    std::deque<RetiredImage> retired_images_;

    VkImageResource cpu_display_tex_;
    VkImageResource copy_tex_;
    VkImageResource overlay_tex_;
    std::uint64_t overlay_gen_ = 0;
    bool overlay_visible_ = false;
    bool overlay_continuous_ = false;
    float overlay_aspect_ = 0.0f;
    std::chrono::steady_clock::time_point last_menu_draw_{};

    std::uint32_t last_shown_slot_ = 0;
    float last_shown_uv_[2]{1.0f, 1.0f};
    bool last_shown_linear_ = true;

    bool have_state_ = false;
    DrawKey state_key_{};
    DrawSetup setup_{};
    std::vector<GpuVertex> batch_;
    std::vector<std::uint32_t> decode_buf_;
};

bool VkBackend::init() {
    if (!ctx_.init(window_)) return false;

    int pw = 0, ph = 0;
    SDL_GetWindowSizeInPixels(window_, &pw, &ph);
    if (pw <= 0 || ph <= 0) {
        pw = 480 * scale_;
        ph = 272 * scale_;
    }
    if (!swapchain_.create(ctx_, pw, ph)) return false;
    if (!descriptors_.init(ctx_)) return false;
    if (!pipelines_.init(ctx_, descriptors_.layout(), swapchain_.format)) return false;

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

    // Initialize fallback 1x1 white image pixels on slot 0.
    batch_.reserve(65536);
    std::fprintf(stderr, "[vk] ready, internal resolution %dx%d\n", 480 * scale_, 272 * scale_);
    return true;
}

void VkBackend::shutdown() {
    if (ctx_.device == VK_NULL_HANDLE) return;
    ctx_.wait_idle();

    for (auto &[addr, rt] : targets_) {
        rt.color.destroy_immediate(ctx_, &descriptors_);
        rt.depth.destroy_immediate(ctx_, nullptr);
    }
    targets_.clear();

    for (auto &[hash, tex] : textures_) {
        tex.image.destroy_immediate(ctx_, &descriptors_);
    }
    textures_.clear();

    cpu_display_tex_.destroy_immediate(ctx_, &descriptors_);
    copy_tex_.destroy_immediate(ctx_, &descriptors_);
    overlay_tex_.destroy_immediate(ctx_, &descriptors_);

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
    const std::uint64_t retire_at = ctx_.graphics_timeline_value + 1u;
    if (res.bindless_slot != 0) {
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
    ctx_.wait_timeline(ctx_.graphics_timeline, f.timeline_value);
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
    active_target_addr_ = UINT32_MAX;
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
        src_stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
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
                            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                        VK_IMAGE_ASPECT_COLOR_BIT);
        rt.depth.create(ctx_, nullptr, pw, ph, VK_FORMAT_D32_SFLOAT,
                        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                        VK_IMAGE_ASPECT_DEPTH_BIT);
    } else {
        rt.stride = w;
        rt.format = format;
    }
    return rt;
}

void VkBackend::begin_target_pass(RenderTarget &rt) {
    ensure_recording();
    if (in_render_pass_ && active_target_addr_ == rt.address) return;
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
    color_att.imageView = rt.color.view;
    color_att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color_att.loadOp = first_use_color ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
    color_att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_att.clearValue.color = {{0.0f, 0.0f, 0.0f, 1.0f}};

    VkRenderingAttachmentInfo depth_att{};
    depth_att.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depth_att.imageView = rt.depth.view;
    depth_att.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depth_att.loadOp = first_use_depth ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
    depth_att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depth_att.clearValue.depthStencil = {1.0f, 0u};

    VkRenderingInfo ri{};
    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea = {{0, 0}, {rt.color.width, rt.color.height}};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &color_att;
    ri.pDepthAttachment = &depth_att;

    vkCmdBeginRendering(frames_[frame_slot_].cmd, &ri);
    in_render_pass_ = true;
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
    if (!slice.ptr) return; // staging ring full for this frame; skips gracefully
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
    DrawKey key{};
    for (std::size_t i = 0; i < kStateRegs.size(); ++i) key[i] = ge.reg(kStateRegs[i]);
    key[kStateRegs.size()] = static_cast<std::uint32_t>(ge.clut_generation() & 0xFFFFFFFFu);
    key[kStateRegs.size() + 1] = ge.reg(kMaskAlpha);

    if (have_state_ && key == state_key_) return;
    draw_batch();

    state_key_ = key;
    have_state_ = true;
    setup_ = DrawSetup{};
    setup_.key = key;
    setup_.through = (ge.reg(kVertexType) & 0x800000u) != 0;

    const std::uint32_t fb_addr = ((ge.reg(kFramebufPtr) & 0xFFFFFFu) | ((ge.reg(kFramebufWidth) & 0xFF0000u) << 8u)) & 0x001FFFFFu;
    const std::uint32_t fb_stride = ge.reg(kFramebufWidth) & 0xFFFFu;
    const std::uint32_t fb_fmt = ge.reg(kFramebufPixFormat) & 3u;
    setup_.target = fb_addr;
    ensure_target(fb_addr, fb_stride, fb_fmt, 272u);

    const bool clear = (ge.reg(kClearMode) & 1u) != 0;
    const bool textured = !clear && (ge.reg(kTextureMapEnable) & 1u) != 0;
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
            const std::uint64_t h = ge.texture_hash();
            auto it = textures_.find(h);
            if (it == textures_.end()) {
                if (textures_.size() >= kMaxCachedTextures) {
                    // Evict oldest LRU texture.
                    auto oldest = textures_.begin();
                    for (auto cur = textures_.begin(); cur != textures_.end(); ++cur) {
                        if (cur->second.last_used < oldest->second.last_used) oldest = cur;
                    }
                    retire_image(oldest->second.image);
                    textures_.erase(oldest);
                }
                decode_buf_.resize(static_cast<std::size_t>(info.width) * info.height);
                ge.decode_texture(decode_buf_.data());
                CachedTexture ct{};
                ct.width = info.width;
                ct.height = info.height;
                ct.last_used = frame_number_;
                upload_texture_2d(ct.image, info.width, info.height, decode_buf_.data());
                it = textures_.emplace(h, std::move(ct)).first;
            } else {
                it->second.last_used = frame_number_;
            }
            setup_.tex_slot = it->second.image.bindless_slot;
        }
    }
}

void VkBackend::draw_batch() {
    if (batch_.empty()) return;
    auto rt_it = targets_.find(setup_.target);
    if (rt_it == targets_.end()) {
        batch_.clear();
        return;
    }
    RenderTarget &rt = rt_it->second;
    begin_target_pass(rt);

    FrameSlice &f = frames_[frame_slot_];
    const std::size_t vbytes = batch_.size() * sizeof(GpuVertex);
    auto vslice = f.vertices.allocate(vbytes, 16);
    auto pslice = f.params.allocate(sizeof(vk::DrawParams), 16);
    if (!vslice.ptr || !pslice.ptr) {
        batch_.clear();
        return;
    }
    std::memcpy(vslice.ptr, batch_.data(), vbytes);

    const auto R = [&](std::uint32_t reg) { return key_reg(setup_.key, reg); };
    const bool clear = (R(kClearMode) & 1u) != 0;
    const bool textured = !clear && (R(kTextureMapEnable) & 1u) != 0;
    const bool fog = !clear && !setup_.through && (R(kFogEnable) & 1u) != 0;
    const bool alpha_test = !clear && (R(kAlphaTestEnable) & 1u) != 0;
    const std::uint32_t minz = R(kMinZ) & 0xFFFFu, maxz = R(kMaxZ) & 0xFFFFu;
    const bool zrange = !clear && !setup_.through && maxz != 0 && (minz != 0 || maxz != 0xFFFFu);

    const bool linear = ((R(kTexFilter) & 0x100u) != 0) ||
                        (live_->smooth_textures.load(std::memory_order_relaxed) && !setup_.through);
    const bool clamp_s = (R(kTexWrap) & 1u) != 0;
    const bool clamp_t = (R(kTexWrap) & 0x100u) != 0;
    const int aniso = std::clamp(live_->texture_pack_anisotropy.load(std::memory_order_relaxed), 1, 16);

    vk::DrawParams dp{};
    std::copy_n(setup_.uv_xform, 4, dp.uv_xform);
    unpack_rgb(R(kTexEnvColor), dp.env_and_flags);
    unpack_rgb(R(kFogColor), dp.fog_and_zmin);
    dp.fog_and_zmin[3] = static_cast<float>(minz);
    dp.target_size[0] = static_cast<float>(rt.width);
    dp.target_size[1] = static_cast<float>(rt.height);
    dp.zmax_pad[0] = static_cast<float>(maxz);
    dp.tex_idx = textured ? setup_.tex_slot : 0u;
    dp.sampler_idx = vk::BindlessDescriptors::sampler_index(linear, setup_.texture_mips, clamp_s, clamp_t, aniso);

    const std::uint32_t func = R(kTexFunc);
    dp.tex_func_bits = (func & 7u) | ((func & 0x100u) ? 8u : 0u) | ((func & 0x10000u) ? 16u : 0u) |
                       (clear ? 32u : 0u) | (textured ? 64u : 0u) | (fog ? 128u : 0u) | (zrange ? 256u : 0u);
    if (alpha_test) {
        dp.alpha_bits = R(kAlphaTest) & 0xFFFFFFu;
    } else {
        dp.alpha_bits = 1u; // ALWAYS
    }
    std::memcpy(pslice.ptr, &dp, sizeof(dp));

    // Blend & color write mask state.
    vk::GePipelineKey pkey{};
    pkey.spec_clear = clear ? 1u : 0u;
    pkey.spec_textured = textured ? 1u : 0u;
    pkey.spec_alpha_test = alpha_test ? 1u : 0u;

    VkBool32 blend_en = VK_FALSE;
    VkColorBlendEquationEXT blend_eq{};
    blend_eq.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    blend_eq.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    blend_eq.colorBlendOp = VK_BLEND_OP_ADD;
    blend_eq.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blend_eq.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    blend_eq.alphaBlendOp = VK_BLEND_OP_ADD;
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
    if (!pipe) {
        batch_.clear();
        return;
    }

    VkCommandBuffer cmd = f.cmd;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    VkDescriptorSet set0 = descriptors_.set();
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines_.ge_layout, 0, 1, &set0, 0, nullptr);

    VkViewport vp{0.0f, 0.0f, static_cast<float>(rt.color.width), static_cast<float>(rt.color.height), 0.0f, 1.0f};
    vkCmdSetViewport(cmd, 0, 1, &vp);

    const std::uint32_t sx1 = R(kScissor1) & 0x3FFu, sy1 = (R(kScissor1) >> 10u) & 0x3FFu;
    const std::uint32_t sx2 = std::min(R(kScissor2) & 0x3FFu, rt.width - 1u);
    const std::uint32_t sy2 = std::min((R(kScissor2) >> 10u) & 0x3FFu, rt.height - 1u);
    VkRect2D scissor{{0, 0}, {0, 0}};
    if (sx1 <= sx2 && sy1 <= sy2) {
        scissor.offset = {static_cast<std::int32_t>(sx1) * scale_, static_cast<std::int32_t>(sy1) * scale_};
        scissor.extent = {(sx2 - sx1 + 1u) * static_cast<std::uint32_t>(scale_),
                          (sy2 - sy1 + 1u) * static_cast<std::uint32_t>(scale_)};
    }
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // Depth dynamic state (Vulkan 1.3 core).
    if (clear) {
        const bool depth_clear = (R(kClearMode) & 0x400u) != 0;
        vkCmdSetDepthTestEnable(cmd, VK_TRUE);
        vkCmdSetDepthCompareOp(cmd, VK_COMPARE_OP_ALWAYS);
        vkCmdSetDepthWriteEnable(cmd, depth_clear ? VK_TRUE : VK_FALSE);
    } else if ((R(kZTestEnable) & 1u) != 0) {
        static constexpr VkCompareOp kOps[8] = {VK_COMPARE_OP_NEVER, VK_COMPARE_OP_ALWAYS, VK_COMPARE_OP_EQUAL,
                                                VK_COMPARE_OP_NOT_EQUAL, VK_COMPARE_OP_LESS, VK_COMPARE_OP_LESS_OR_EQUAL,
                                                VK_COMPARE_OP_GREATER, VK_COMPARE_OP_GREATER_OR_EQUAL};
        vkCmdSetDepthTestEnable(cmd, VK_TRUE);
        vkCmdSetDepthCompareOp(cmd, kOps[R(kZTest) & 7u]);
        vkCmdSetDepthWriteEnable(cmd, (R(kZWriteDisable) & 1u) == 0 ? VK_TRUE : VK_FALSE);
    } else {
        vkCmdSetDepthTestEnable(cmd, VK_FALSE);
        vkCmdSetDepthWriteEnable(cmd, VK_FALSE);
        vkCmdSetDepthCompareOp(cmd, VK_COMPARE_OP_ALWAYS);
    }

    vkCmdSetBlendConstants(cmd, blend_const);
    if (ctx_.caps.extended_dynamic_state3) {
        vkCmdSetColorBlendEnableEXT(cmd, 0, 1, &blend_en);
        vkCmdSetColorBlendEquationEXT(cmd, 0, 1, &blend_eq);
        vkCmdSetColorWriteMaskEXT(cmd, 0, 1, &write_mask);
    }

    const VkDeviceSize voff = vslice.offset;
    vkCmdBindVertexBuffers(cmd, 0, 1, &f.vertices.buffer, &voff);

    vk::DrawPush push{pslice.address, 0u};
    vkCmdPushConstants(cmd, pipelines_.ge_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(push), &push);
    vkCmdDraw(cmd, static_cast<std::uint32_t>(batch_.size()), 1, 0, 0);

    batch_.clear();
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
    const std::uint32_t src_vram = src & 0x1FFFFFFFu;
    const std::uint32_t dst_vram = dst & 0x1FFFFFFFu;
    const bool src_in_vram = src_vram >= 0x04000000u && src_vram < 0x04800000u;
    const bool dst_in_vram = dst_vram >= 0x04000000u && dst_vram < 0x04800000u;
    if (!src_in_vram || !dst_in_vram) return;

    const std::uint32_t src_off = src_vram & 0x001FFFFFu;
    const std::uint32_t dst_off = dst_vram & 0x001FFFFFu;
    auto sit = targets_.find(src_off);
    if (sit == targets_.end()) return;

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

void VkBackend::flush() {
    draw_batch();
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
}

void VkBackend::upload_overlay_if_needed() {
    if (!overlay_) return;
    std::vector<std::uint32_t> pixels;
    int w = 0, h = 0;
    bool upload = false;
    {
        std::lock_guard lock(overlay_->mutex);
        overlay_visible_ = overlay_->visible;
        overlay_continuous_ = overlay_->continuous;
        overlay_aspect_ = overlay_->aspect;
        if (overlay_->visible && overlay_->generation != overlay_gen_ && overlay_->width > 0 && overlay_->height > 0) {
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

void VkBackend::show_to_swapchain(std::uint32_t tex_slot, float uv_max_x, float uv_max_y, bool linear_filter) {
    ensure_recording();
    end_render_pass();
    upload_overlay_if_needed();

    int pw = 0, ph = 0;
    SDL_GetWindowSizeInPixels(window_, &pw, &ph);
    if (pw <= 0 || ph <= 0) return;

    if (static_cast<std::uint32_t>(pw) != swapchain_.extent.width ||
        static_cast<std::uint32_t>(ph) != swapchain_.extent.height) {
        vkEndCommandBuffer(frames_[frame_slot_].cmd);
        recording_ = false;
        swapchain_.recreate(ctx_, pw, ph);
        pipelines_.rebuild_swapchain_pipelines(ctx_, swapchain_.format);
        ensure_recording();
        upload_overlay_if_needed();
    }

    std::uint32_t img_idx = 0;
    VkResult acq = vkAcquireNextImageKHR(ctx_.device, swapchain_.handle, UINT64_MAX,
                                         frames_[frame_slot_].image_available, VK_NULL_HANDLE, &img_idx);
    if (acq == VK_ERROR_OUT_OF_DATE_KHR) {
        vkEndCommandBuffer(frames_[frame_slot_].cmd);
        recording_ = false;
        swapchain_.recreate(ctx_, pw, ph);
        pipelines_.rebuild_swapchain_pipelines(ctx_, swapchain_.format);
        return;
    }
    if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) return;

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
        vk::PresentPush op{{1.0f, 1.0f}, overlay_tex_.bindless_slot,
                           vk::BindlessDescriptors::sampler_index(true, false, true, true, 1), 1.0f};
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
    {
        std::lock_guard lock(ctx_.graphics_queue_mutex);
        vkQueueSubmit2(ctx_.graphics_queue, 1, &si, VK_NULL_HANDLE);
        vkQueuePresentKHR(ctx_.graphics_queue, &pi);
    }

    frame_slot_ = (frame_slot_ + 1) % kFramesInFlight;
}

void VkBackend::apply_live_scale() {
    const int req = live_->scale.exchange(0, std::memory_order_relaxed);
    if (req >= 1 && req <= 10 && req != scale_) {
        scale_ = req;
        for (auto &[addr, rt] : targets_) {
            retire_image(rt.color);
            retire_image(rt.depth);
        }
        targets_.clear();
        retire_image(copy_tex_);
    }
}

void VkBackend::present(std::uint32_t address, std::uint32_t stride, std::uint32_t pixel_format) {
    draw_batch();
    have_state_ = false;
    ++frame_number_;

    const std::uint32_t off = (address & 0x1FFFFFFFu) & 0x001FFFFFu;
    auto it = targets_.find(off);
    if (it != targets_.end() && it->second.color.image != VK_NULL_HANDLE && it->second.last_draw + 4 >= frame_number_) {
        RenderTarget &rt = it->second;
        ensure_recording();
        end_render_pass();
        transition_resource(rt.color, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        const float uv_x = std::min(1.0f, 480.0f / static_cast<float>(rt.width));
        const float uv_y = std::min(1.0f, 272.0f / static_cast<float>(rt.height));
        last_shown_slot_ = rt.color.bindless_slot;
        last_shown_uv_[0] = uv_x;
        last_shown_uv_[1] = uv_y;
        last_shown_linear_ = true;
        show_to_swapchain(last_shown_slot_, uv_x, uv_y, true);
    } else {
        // CPU-drawn screen (boot logos, PSMF intro videos): upload from guest VRAM.
        const std::uint32_t s = stride > 0 ? stride : 512u;
        const std::uint32_t bpp = pixel_format == 3u ? 4u : 2u;
        decode_buf_.assign(480u * 272u, 0xFF000000u);
        for (std::uint32_t y = 0; y < 272u; ++y) {
            const std::uint8_t *row = memory_.raw_pointer(0x04000000u + off + y * s * bpp, 480u * bpp);
            if (!row) continue;
            for (std::uint32_t x = 0; x < 480u; ++x) {
                decode_buf_[y * 480u + x] = framebuffer_pixel(row + x * bpp, pixel_format);
            }
        }
        upload_texture_2d(cpu_display_tex_, 480u, 272u, decode_buf_.data());
        last_shown_slot_ = cpu_display_tex_.bindless_slot;
        last_shown_uv_[0] = 1.0f;
        last_shown_uv_[1] = 1.0f;
        last_shown_linear_ = true;
        show_to_swapchain(last_shown_slot_, 1.0f, 1.0f, true);
    }

    last_menu_draw_ = std::chrono::steady_clock::now();
    apply_live_scale();
}

std::optional<std::chrono::steady_clock::time_point> VkBackend::next_wakeup() const {
    if (overlay_visible_ && overlay_continuous_) {
        return last_menu_draw_ + std::chrono::milliseconds(16);
    }
    return std::nullopt;
}

void VkBackend::tick() {
    if (overlay_visible_ && overlay_continuous_ && last_shown_slot_ != 0) {
        show_to_swapchain(last_shown_slot_, last_shown_uv_[0], last_shown_uv_[1], last_shown_linear_);
        last_menu_draw_ = std::chrono::steady_clock::now();
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
