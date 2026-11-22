#include "vk_lighting.hpp"
#include "vk_shaders_spv.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace mhf::psp::vk {

namespace {

template <std::size_t N>
VkShaderModule make_mod(const Context &ctx, const std::array<std::uint32_t, N> &words) {
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
                                words.size() * sizeof(std::uint32_t), words.data()};
    VkShaderModule m = VK_NULL_HANDLE;
    vkCreateShaderModule(ctx.device, &ci, nullptr, &m);
    return m;
}

BiomeKind classify_zone_name(const std::string &name, bool is_cave) {
    if (is_cave) return BiomeKind::CaveIndoor;
    if (name.find("snowy") != std::string::npos || name.find("pokke/village") != std::string::npos) {
        return BiomeKind::Snowy;
    }
    if (name.find("desert") != std::string::npos) return BiomeKind::Desert;
    if (name.find("volcano") != std::string::npos || name.find("battleground") != std::string::npos) {
        return BiomeKind::Volcano;
    }
    if (name.find("swamp") != std::string::npos) return BiomeKind::Swamp;
    if (name.find("jungle") != std::string::npos || name.find("forest") != std::string::npos) {
        return BiomeKind::JungleForest;
    }
    return BiomeKind::DefaultOutdoor;
}

} // namespace

void LightingPipeline::Target2D::destroy(const Context &ctx, BindlessDescriptors &desc) {
    if (slot != 0) {
        desc.free_texture(slot, 0);
        slot = 0;
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
    w = 0;
    h = 0;
}

bool LightingPipeline::init(const Context &ctx, VkDescriptorSetLayout bindless_layout, const LiveSettings *live) {
    VkDescriptorSetLayoutBinding b{};
    b.binding = 0;
    b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 1, &b};
    if (vkCreateDescriptorSetLayout(ctx.device, &dslci, nullptr, &storage_layout_) != VK_SUCCESS) return false;

    constexpr std::uint32_t kTotalSets = 3u + static_cast<std::uint32_t>(kSceneSetCount);
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kTotalSets};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, kTotalSets, 1, &ps};
    if (vkCreateDescriptorPool(ctx.device, &dpci, nullptr, &storage_pool_) != VK_SUCCESS) return false;

    std::array<VkDescriptorSetLayout, kTotalSets> dsls{};
    dsls.fill(storage_layout_);
    std::array<VkDescriptorSet, kTotalSets> sets{};
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, storage_pool_, kTotalSets, dsls.data()};
    if (vkAllocateDescriptorSets(ctx.device, &dsai, sets.data()) != VK_SUCCESS) return false;
    gtao_set_ = sets[0];
    vol_set_ = sets[1];
    bloom_set_ = sets[2];
    for (std::size_t i = 0; i < kSceneSetCount; ++i) {
        scene_sets_[i] = sets[3 + i];
    }
    scene_set_cursor_ = 0;

    std::array<VkDescriptorSetLayout, 2> pipe_sets = {bindless_layout, storage_layout_};
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkDeviceAddress)};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 2, pipe_sets.data(), 1, &pcr};
    if (vkCreatePipelineLayout(ctx.device, &plci, nullptr, &pipe_layout_) != VK_SUCCESS) return false;

    auto make_pipe = [&](auto &spv) -> VkPipeline {
        VkShaderModule mod = make_mod(ctx, spv);
        if (!mod) return VK_NULL_HANDLE;
        VkComputePipelineCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, mod, "main", nullptr};
        ci.layout = pipe_layout_;
        VkPipeline pipe = VK_NULL_HANDLE;
        vkCreateComputePipelines(ctx.device, ctx.pipeline_cache, 1, &ci, nullptr, &pipe);
        vkDestroyShaderModule(ctx.device, mod, nullptr);
        return pipe;
    };

    gtao_pipe_ = make_pipe(shaders::kGtaoCompSpv);
    vol_pipe_ = make_pipe(shaders::kVolumetricFogCompSpv);
    bloom_pipe_ = make_pipe(shaders::kBloomExtractCompSpv);
    comp_pipe_ = make_pipe(shaders::kLightingCompositeCompSpv);

    // Build fast zone & biome lookup tables from LiveSettings (supports MHFU zones + graceful MHF1 fallback).
    if (live && !live->shadow_zone_names.empty()) {
        std::unordered_set<std::string> caves(live->shadow_cave_zones.begin(), live->shadow_cave_zones.end());
        std::unordered_map<std::string, std::array<float, 3>> suns;
        for (const auto &[zname, s] : live->shadow_zone_suns) {
            suns[zname] = s;
        }
        zones_.resize(live->shadow_zone_names.size());
        zone_votes_.assign(zones_.size(), 0u);
        for (std::size_t i = 0; i < zones_.size(); ++i) {
            const std::string &zname = live->shadow_zone_names[i];
            const bool is_cave = caves.count(zname) > 0;
            zones_[i].is_cave = is_cave;
            zones_[i].biome = classify_zone_name(zname, is_cave);
            if (auto it = suns.find(zname); it != suns.end()) {
                zones_[i].sun_height_deg = it->second[0];
                zones_[i].sun_turn = it->second[2];
            }
        }
        tex_to_zone_.reserve(live->shadow_zone_textures.size());
        for (const auto &[h32, zidx] : live->shadow_zone_textures) {
            if (zidx >= 0 && static_cast<std::size_t>(zidx) < zones_.size()) {
                tex_to_zone_[h32] = zidx;
            }
        }
    }

    return ready();
}

void LightingPipeline::shutdown(const Context &ctx, BindlessDescriptors &desc) {
    gtao_tex_.destroy(ctx, desc);
    vol_tex_.destroy(ctx, desc);
    bloom_tex_.destroy(ctx, desc);
    auto kill_pipe = [&](VkPipeline &p) {
        if (p != VK_NULL_HANDLE) {
            vkDestroyPipeline(ctx.device, p, nullptr);
            p = VK_NULL_HANDLE;
        }
    };
    kill_pipe(gtao_pipe_);
    kill_pipe(vol_pipe_);
    kill_pipe(bloom_pipe_);
    kill_pipe(comp_pipe_);
    if (pipe_layout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(ctx.device, pipe_layout_, nullptr);
        pipe_layout_ = VK_NULL_HANDLE;
    }
    if (storage_pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(ctx.device, storage_pool_, nullptr);
        storage_pool_ = VK_NULL_HANDLE;
    }
    if (storage_layout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(ctx.device, storage_layout_, nullptr);
        storage_layout_ = VK_NULL_HANDLE;
    }
}

void LightingPipeline::note_texture_hash(std::uint64_t tex_hash) noexcept {
    if (tex_to_zone_.empty()) return;
    const auto it = tex_to_zone_.find(static_cast<std::uint32_t>(tex_hash));
    if (it != tex_to_zone_.end()) {
        ++zone_votes_[static_cast<std::size_t>(it->second)];
    }
}

void LightingPipeline::end_frame_zone_vote() noexcept {
    if (zone_votes_.empty()) return;
    std::uint32_t best_votes = 0;
    int best_zone = -1;
    for (std::size_t i = 0; i < zone_votes_.size(); ++i) {
        if (zone_votes_[i] > best_votes) {
            best_votes = zone_votes_[i];
            best_zone = static_cast<int>(i);
        }
        zone_votes_[i] = 0u;
    }
    if (best_zone >= 0 && best_votes >= 2u) {
        if (active_zone_ < 0 || best_zone == active_zone_) {
            active_zone_ = best_zone;
            active_biome_ = zones_[static_cast<std::size_t>(best_zone)].biome;
            active_is_cave_ = zones_[static_cast<std::size_t>(best_zone)].is_cave;
            pending_zone_ = -1;
            pending_zone_streak_ = 0;
        } else if (best_zone == pending_zone_) {
            if (++pending_zone_streak_ >= 3u) {
                active_zone_ = best_zone;
                active_biome_ = zones_[static_cast<std::size_t>(best_zone)].biome;
                active_is_cave_ = zones_[static_cast<std::size_t>(best_zone)].is_cave;
                pending_zone_ = -1;
                pending_zone_streak_ = 0;
            }
        } else {
            pending_zone_ = best_zone;
            pending_zone_streak_ = 1u;
        }
    }
}

bool LightingPipeline::ensure_targets(const Context &ctx, BindlessDescriptors &desc,
                                      std::uint32_t full_w, std::uint32_t full_h) {
    const std::uint32_t hw = std::max(full_w / 2u, 1u);
    const std::uint32_t hh = std::max(full_h / 2u, 1u);
    const std::uint32_t qw = std::max(full_w / 4u, 1u);
    const std::uint32_t qh = std::max(full_h / 4u, 1u);

    auto make_target = [&](Target2D &t, std::uint32_t w, std::uint32_t h, VkFormat fmt, VkDescriptorSet dset) -> bool {
        if (t.image != VK_NULL_HANDLE && t.w == w && t.h == h) return true;
        ctx.wait_idle();
        t.destroy(ctx, desc);
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = fmt;
        ici.extent = {w, h, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        VmaAllocationCreateInfo vaci{};
        vaci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        if (vmaCreateImage(ctx.allocator, &ici, &vaci, &t.image, &t.alloc, nullptr) != VK_SUCCESS) return false;

        VkImageViewCreateInfo ivci{};
        ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ivci.image = t.image;
        ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ivci.format = fmt;
        ivci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(ctx.device, &ivci, nullptr, &t.view) != VK_SUCCESS) return false;

        t.w = w;
        t.h = h;
        t.slot = desc.allocate_texture(ctx, t.view);
        VkDescriptorImageInfo dii{VK_NULL_HANDLE, t.view, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet wds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dset, 0, 0, 1,
                                 VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &dii, nullptr, nullptr};
        vkUpdateDescriptorSets(ctx.device, 1, &wds, 0, nullptr);
        return t.slot != 0;
    };

    return make_target(gtao_tex_, hw, hh, VK_FORMAT_R8G8B8A8_UNORM, gtao_set_) &&
           make_target(vol_tex_, qw, qh, VK_FORMAT_R16G16B16A16_SFLOAT, vol_set_) &&
           make_target(bloom_tex_, qw, qh, VK_FORMAT_R16G16B16A16_SFLOAT, bloom_set_);
}

void LightingPipeline::fill_biome_params(LightingComputeParams &p, const LightingFrameInput &in) const noexcept {
    // Projection info for view-space reconstruction.
    const float p00 = std::fabs(in.proj[0]) > 1e-4f ? in.proj[0] : 1.0f;
    const float p11 = std::fabs(in.proj[5]) > 1e-4f ? in.proj[5] : 1.76f;
    p.proj_info[0] = 1.0f / p00;
    p.proj_info[1] = 1.0f / p11;
    p.proj_info[2] = in.proj[10];
    p.proj_info[3] = std::fabs(in.proj[14]) > 1e-4f ? in.proj[14] : -10.0f;
    p.vz_and_size[0] = in.vz_scale;
    p.vz_and_size[1] = in.vz_center;

    // Normalized view-space sun direction (from live PSP GE light 0, with sensible upper-sky fallback).
    float lx = in.has_light0 ? in.light0_dir[0] : 0.32f;
    float ly = in.has_light0 ? in.light0_dir[1] : 0.78f;
    float lz = in.has_light0 ? in.light0_dir[2] : -0.54f;
    const float llen = std::sqrt(lx * lx + ly * ly + lz * lz);
    if (llen > 1e-4f) {
        lx /= llen;
        ly /= llen;
        lz /= llen;
    } else {
        lx = 0.32f;
        ly = 0.78f;
        lz = -0.54f;
    }
    p.sun_dir_view[0] = lx;
    p.sun_dir_view[1] = ly;
    p.sun_dir_view[2] = lz;
    p.sun_dir_view[3] = active_is_cave_ ? 1.0f : 0.0f;

    // Project sun direction onto screen UV for volumetric god-rays (smoothly fades when sun is behind camera).
    const float sun_facing = std::clamp((-lz + 0.25f) / 1.15f, 0.0f, 1.0f);
    const float inv_z = 1.0f / std::max(-lz, 0.25f);
    p.sun_screen[0] = std::clamp(0.5f + 0.5f * (lx * p00 * inv_z), -0.35f, 1.35f);
    p.sun_screen[1] = std::clamp(0.5f - 0.5f * (ly * p11 * inv_z), -0.35f, 1.35f);
    p.sun_screen[2] = active_is_cave_ ? 0.0f : sun_facing;
    p.sun_screen[3] = static_cast<float>(in.frame_index & 63u);

    // Biome-tuned lighting, AO, shadow tint, volumetric fog, and bloom presets.
    float shadow_rgb[3]{0.78f, 0.82f, 0.90f};
    float ao_str = 0.62f;
    float fog_rgb[3]{0.70f, 0.78f, 0.88f};
    float fog_dens = 0.00042f;
    float scat_rgb[3]{1.00f, 0.94f, 0.82f};
    float scat_str = 0.22f;
    float bloom_th = 0.80f;
    float bloom_str = 0.26f;
    float rim_str = 0.14f;
    float key_str = 0.14f;

    switch (active_biome_) {
    case BiomeKind::Snowy:
        // Pokke Village & Snowy Mountains: crisp cool alpine sky bounce + gentle snow-glare scatter.
        shadow_rgb[0] = 0.74f; shadow_rgb[1] = 0.82f; shadow_rgb[2] = 0.96f;
        ao_str = 0.64f;
        fog_rgb[0] = 0.78f; fog_rgb[1] = 0.86f; fog_rgb[2] = 0.96f;
        fog_dens = 0.00046f;
        scat_rgb[0] = 0.94f; scat_rgb[1] = 0.97f; scat_rgb[2] = 1.00f;
        scat_str = 0.20f;
        bloom_th = 0.82f; bloom_str = 0.28f;
        rim_str = 0.15f; key_str = 0.14f;
        break;
    case BiomeKind::Desert:
        // Desert / Old Desert: golden sunlight, warm sand shadow bounce, subtle heat haze.
        shadow_rgb[0] = 0.84f; shadow_rgb[1] = 0.78f; shadow_rgb[2] = 0.72f;
        ao_str = 0.58f;
        fog_rgb[0] = 0.90f; fog_rgb[1] = 0.82f; fog_rgb[2] = 0.68f;
        fog_dens = 0.00038f;
        scat_rgb[0] = 1.00f; scat_rgb[1] = 0.90f; scat_rgb[2] = 0.70f;
        scat_str = 0.24f;
        bloom_th = 0.82f; bloom_str = 0.24f;
        rim_str = 0.15f; key_str = 0.16f;
        break;
    case BiomeKind::JungleForest:
        // Jungle / Forest & Hills / Great Forest: lush green canopy sun shafts & humid valley mist.
        shadow_rgb[0] = 0.72f; shadow_rgb[1] = 0.82f; shadow_rgb[2] = 0.76f;
        ao_str = 0.68f;
        fog_rgb[0] = 0.70f; fog_rgb[1] = 0.80f; fog_rgb[2] = 0.74f;
        fog_dens = 0.00050f;
        scat_rgb[0] = 0.96f; scat_rgb[1] = 0.98f; scat_rgb[2] = 0.80f;
        scat_str = 0.28f;
        bloom_th = 0.80f; bloom_str = 0.25f;
        rim_str = 0.15f; key_str = 0.14f;
        break;
    case BiomeKind::Swamp:
        // Swamp / Old Swamp: atmospheric misty teal/violet fog with moody low sun scatter.
        shadow_rgb[0] = 0.72f; shadow_rgb[1] = 0.78f; shadow_rgb[2] = 0.86f;
        ao_str = 0.68f;
        fog_rgb[0] = 0.64f; fog_rgb[1] = 0.72f; fog_rgb[2] = 0.78f;
        fog_dens = 0.00068f;
        scat_rgb[0] = 0.82f; scat_rgb[1] = 0.90f; scat_rgb[2] = 0.94f;
        scat_str = 0.22f;
        bloom_th = 0.78f; bloom_str = 0.24f;
        rim_str = 0.13f; key_str = 0.12f;
        break;
    case BiomeKind::Volcano:
        // Volcano / Battleground: warm ember scatter, glowing lava/fire bloom, smoky haze.
        shadow_rgb[0] = 0.80f; shadow_rgb[1] = 0.68f; shadow_rgb[2] = 0.64f;
        ao_str = 0.66f;
        fog_rgb[0] = 0.62f; fog_rgb[1] = 0.36f; fog_rgb[2] = 0.24f;
        fog_dens = 0.00052f;
        scat_rgb[0] = 1.00f; scat_rgb[1] = 0.62f; scat_rgb[2] = 0.32f;
        scat_str = 0.26f;
        bloom_th = 0.70f; bloom_str = 0.38f;
        rim_str = 0.18f; key_str = 0.14f;
        break;
    case BiomeKind::CaveIndoor:
        // Caves & Pokke indoors (House, Hall, Kitchen, Farm): ZERO sun shafts, stronger architectural AO, warm torch bloom.
        shadow_rgb[0] = 0.72f; shadow_rgb[1] = 0.74f; shadow_rgb[2] = 0.80f;
        ao_str = 0.74f;
        fog_rgb[0] = 0.28f; fog_rgb[1] = 0.30f; fog_rgb[2] = 0.36f;
        fog_dens = 0.00032f;
        scat_rgb[0] = 0.95f; scat_rgb[1] = 0.82f; scat_rgb[2] = 0.66f;
        scat_str = 0.0f;
        bloom_th = 0.76f; bloom_str = 0.30f;
        rim_str = 0.08f; key_str = 0.10f;
        break;
    case BiomeKind::DefaultOutdoor:
        break;
    }

    // Blend with the game's live PSP kFogColor so day vs. night quests adapt automatically!
    if (in.has_psp_fog) {
        const float psp_luma = in.psp_fog_rgb[0] * 0.2126f + in.psp_fog_rgb[1] * 0.7152f + in.psp_fog_rgb[2] * 0.0722f;
        for (int c = 0; c < 3; ++c) {
            fog_rgb[c] = fog_rgb[c] * 0.45f + in.psp_fog_rgb[c] * 0.55f;
        }
        // Dark night sky/fog: tone down sun shafts gently and cool down the scatter tint into moonlight.
        if (psp_luma < 0.22f && active_biome_ != BiomeKind::Volcano && !active_is_cave_) {
            scat_str *= 0.55f;
            scat_rgb[0] = 0.72f;
            scat_rgb[1] = 0.82f;
            scat_rgb[2] = 1.00f;
        }
    }

    p.shadow_tint[0] = shadow_rgb[0]; p.shadow_tint[1] = shadow_rgb[1]; p.shadow_tint[2] = shadow_rgb[2];
    p.shadow_tint[3] = ao_str;
    p.fog_color[0] = fog_rgb[0]; p.fog_color[1] = fog_rgb[1]; p.fog_color[2] = fog_rgb[2];
    p.fog_color[3] = fog_dens;
    p.scatter_color[0] = scat_rgb[0]; p.scatter_color[1] = scat_rgb[1]; p.scatter_color[2] = scat_rgb[2];
    p.scatter_color[3] = scat_str;
    p.bloom_rim[0] = bloom_th;
    p.bloom_rim[1] = bloom_str;
    p.bloom_rim[2] = rim_str;
    p.bloom_rim[3] = key_str;
}

void LightingPipeline::dispatch(VkCommandBuffer cmd, const Context &ctx, const BindlessDescriptors &desc,
                                VkImage scene_color, VkImageView scene_color_view, std::uint32_t scene_color_slot,
                                std::uint32_t scene_depth_slot, std::uint32_t scene_w, std::uint32_t scene_h,
                                const LightingFrameInput &in, void *params_cpu_ptr, VkDeviceAddress params_bda) {
    if (!ready() || !params_cpu_ptr || params_bda == 0 || scene_depth_slot == 0 || scene_color_slot == 0) return;
    if (!ensure_targets(ctx, const_cast<BindlessDescriptors &>(desc), scene_w, scene_h)) return;

    // Cycle through a ring of storage descriptor sets so we NEVER mutate a descriptor set
    // referenced by an earlier dispatch in the same command buffer or an in-flight frame!
    const VkDescriptorSet scene_set = scene_sets_[scene_set_cursor_];
    scene_set_cursor_ = (scene_set_cursor_ + 1u) % kSceneSetCount;
    VkDescriptorImageInfo sii{VK_NULL_HANDLE, scene_color_view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet swds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, scene_set, 0, 0, 1,
                              VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &sii, nullptr, nullptr};
    vkUpdateDescriptorSets(ctx.device, 1, &swds, 0, nullptr);

    LightingComputeParams base{};
    fill_biome_params(base, in);
    base.vz_and_size[2] = static_cast<float>(scene_w);
    base.vz_and_size[3] = static_cast<float>(scene_h);
    base.tex_slots[0] = scene_color_slot;
    base.tex_slots[1] = scene_depth_slot;
    base.tex_slots[2] = gtao_tex_.slot;
    base.tex_slots[3] = vol_tex_.slot;
    base.aux_slots[0] = bloom_tex_.slot;
    base.aux_slots[1] = BindlessDescriptors::sampler_index(true, false, true, true, 1);

    auto *slots = reinterpret_cast<LightingComputeParams *>(params_cpu_ptr);
    slots[0] = base; slots[0].aux_slots[2] = gtao_tex_.w;  slots[0].aux_slots[3] = gtao_tex_.h;
    slots[1] = base; slots[1].aux_slots[2] = vol_tex_.w;   slots[1].aux_slots[3] = vol_tex_.h;
    slots[2] = base; slots[2].aux_slots[2] = bloom_tex_.w; slots[2].aux_slots[3] = bloom_tex_.h;
    slots[3] = base; slots[3].aux_slots[2] = scene_w;      slots[3].aux_slots[3] = scene_h;

    // Transition intermediate targets -> GENERAL for compute writes, waiting for any prior
    // composite pass's sampled reads to finish so back-to-back dispatches never race!
    Context::cmd_transition_image(cmd, gtao_tex_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                                  VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                  VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    Context::cmd_transition_image(cmd, vol_tex_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                                  VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                  VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    Context::cmd_transition_image(cmd, bloom_tex_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                                  VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                  VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    // Passes 1, 2, 3 are independent (all read scene_depth / scene_color in SHADER_READ_ONLY_OPTIMAL) ->
    // the GPU executes GTAO, Volumetric Fog, and Bloom in parallel across CUs with ZERO inter-pass barriers!
    auto run_pass = [&](VkPipeline pipe, VkDescriptorSet out_set, std::uint32_t w, std::uint32_t h, std::size_t idx) {
        std::array<VkDescriptorSet, 2> dsets = {desc.set(), out_set};
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_, 0, 2, dsets.data(), 0, nullptr);
        const VkDeviceAddress addr = params_bda + idx * sizeof(LightingComputeParams);
        vkCmdPushConstants(cmd, pipe_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(addr), &addr);
        vkCmdDispatch(cmd, (w + 7u) / 8u, (h + 7u) / 8u, 1u);
    };

    run_pass(gtao_pipe_, gtao_set_, gtao_tex_.w, gtao_tex_.h, 0);
    run_pass(vol_pipe_, vol_set_, vol_tex_.w, vol_tex_.h, 1);
    run_pass(bloom_pipe_, bloom_set_, bloom_tex_.w, bloom_tex_.h, 2);

    // Single barrier batch: transition gtao/vol/bloom -> SHADER_READ_ONLY, scene_color -> GENERAL.
    Context::cmd_transition_image(cmd, gtao_tex_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                                  VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    Context::cmd_transition_image(cmd, vol_tex_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                                  VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    Context::cmd_transition_image(cmd, bloom_tex_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                                  VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    Context::cmd_transition_image(cmd, scene_color, VK_IMAGE_ASPECT_COLOR_BIT,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                  VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    // Pass 4: Unified full-res composite in-place on scene_color.
    run_pass(comp_pipe_, scene_set, scene_w, scene_h, 3);
}

} // namespace mhf::psp::vk
