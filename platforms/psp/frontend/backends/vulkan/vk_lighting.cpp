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

    constexpr std::uint32_t kTotalSets = 7u + static_cast<std::uint32_t>(kSceneSetCount);
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kTotalSets};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, kTotalSets, 1, &ps};
    if (vkCreateDescriptorPool(ctx.device, &dpci, nullptr, &storage_pool_) != VK_SUCCESS) return false;

    std::array<VkDescriptorSetLayout, kTotalSets> dsls{};
    dsls.fill(storage_layout_);
    std::array<VkDescriptorSet, kTotalSets> sets{};
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, storage_pool_, kTotalSets, dsls.data()};
    if (vkAllocateDescriptorSets(ctx.device, &dsai, sets.data()) != VK_SUCCESS) return false;
    gtao_set_ = sets[0];
    gtao_blur_set_ = sets[1];
    vol_set_ = sets[2];
    bloom_set_ = sets[3];
    bloom_blur_set_ = sets[4];
    ssgi_set_ = sets[5];
    ssgi_blur_set_ = sets[6];
    for (std::size_t i = 0; i < kSceneSetCount; ++i) {
        scene_sets_[i] = sets[7 + i];
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
    gtao_denoise_pipe_ = make_pipe(shaders::kGtaoDenoiseCompSpv);
    vol_pipe_ = make_pipe(shaders::kVolumetricFogCompSpv);
    bloom_pipe_ = make_pipe(shaders::kBloomExtractCompSpv);
    ssgi_pipe_ = make_pipe(shaders::kSsgiBounceCompSpv);
    glow_blur_pipe_ = make_pipe(shaders::kGlowBlurCompSpv);
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
        if (!tex_to_zone_.empty()) {
            zone_keys_ = textures::Pack::keys_only();
        }
    }

    return ready();
}

void LightingPipeline::shutdown(const Context &ctx, BindlessDescriptors &desc) {
    gtao_tex_.destroy(ctx, desc);
    gtao_blur_tex_.destroy(ctx, desc);
    vol_tex_.destroy(ctx, desc);
    bloom_tex_.destroy(ctx, desc);
    bloom_blur_tex_.destroy(ctx, desc);
    ssgi_tex_.destroy(ctx, desc);
    ssgi_blur_tex_.destroy(ctx, desc);
    auto kill_pipe = [&](VkPipeline &p) {
        if (p != VK_NULL_HANDLE) {
            vkDestroyPipeline(ctx.device, p, nullptr);
            p = VK_NULL_HANDLE;
        }
    };
    kill_pipe(gtao_pipe_);
    kill_pipe(gtao_denoise_pipe_);
    kill_pipe(vol_pipe_);
    kill_pipe(bloom_pipe_);
    kill_pipe(ssgi_pipe_);
    kill_pipe(glow_blur_pipe_);
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

void LightingPipeline::note_texture(std::uint64_t content_hash, const textures::TextureFacts &facts,
                                    const textures::MemoryReader &reader) {
    if (tex_to_zone_.empty() || !zone_keys_) return;
    int zone = -1;
    if (const auto cit = content_hash_to_zone_.find(content_hash); cit != content_hash_to_zone_.end()) {
        zone = cit->second;
    } else {
        textures::Key key{};
        std::uint32_t cw = facts.width, ch = facts.height;
        if (zone_keys_->key_of(facts, reader, key, cw, ch)) {
            if (const auto zit = tex_to_zone_.find(key.data); zit != tex_to_zone_.end()) {
                zone = zit->second;
            }
        }
        if (content_hash_to_zone_.size() > 16384u) content_hash_to_zone_.clear();
        content_hash_to_zone_.emplace(content_hash, zone);
    }
    if (zone >= 0 && static_cast<std::size_t>(zone) < zone_votes_.size()) {
        ++zone_votes_[static_cast<std::size_t>(zone)];
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
           make_target(gtao_blur_tex_, hw, hh, VK_FORMAT_R8G8B8A8_UNORM, gtao_blur_set_) &&
           make_target(vol_tex_, qw, qh, VK_FORMAT_R16G16B16A16_SFLOAT, vol_set_) &&
           make_target(bloom_tex_, qw, qh, VK_FORMAT_R16G16B16A16_SFLOAT, bloom_set_) &&
           make_target(bloom_blur_tex_, qw, qh, VK_FORMAT_R16G16B16A16_SFLOAT, bloom_blur_set_) &&
           make_target(ssgi_tex_, qw, qh, VK_FORMAT_R16G16B16A16_SFLOAT, ssgi_set_) &&
           make_target(ssgi_blur_tex_, qw, qh, VK_FORMAT_R16G16B16A16_SFLOAT, ssgi_blur_set_);
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

    // World-space sun direction: prefer live PSP GE directional light 0, or fall back to zone sun elevation/turn.
    float wx = 0.34f, wy = 0.78f, wz = 0.52f;
    if (in.has_light0) {
        wx = in.light0_dir[0];
        wy = in.light0_dir[1];
        wz = in.light0_dir[2];
    } else if (active_zone_ >= 0 && static_cast<std::size_t>(active_zone_) < zones_.size() &&
               zones_[static_cast<std::size_t>(active_zone_)].sun_height_deg > 0.0f) {
        const auto &z = zones_[static_cast<std::size_t>(active_zone_)];
        const float elev = std::clamp(z.sun_height_deg, 18.0f, 75.0f) * (3.14159265f / 180.0f);
        const float azim = z.sun_turn * (3.14159265f / 180.0f);
        const float ce = std::cos(elev);
        wx = ce * std::sin(azim);
        wy = std::sin(elev);
        wz = ce * std::cos(azim);
    }
    const float wlen = std::sqrt(wx * wx + wy * wy + wz * wz);
    if (wlen > 1e-4f) { wx /= wlen; wy /= wlen; wz /= wlen; }
    else { wx = 0.34f; wy = 0.78f; wz = 0.52f; }
    // Ensure outdoor sun points upward so contact shadows & key light always come from the sky.
    if (wy < 0.15f) {
        wy = std::fabs(wy) + 0.25f;
        const float l2 = std::sqrt(wx * wx + wy * wy + wz * wz);
        wx /= l2; wy /= l2; wz /= l2;
    }

    // Transform world sun & world up (0,1,0) into view space using active camera rotation so sun shafts,
    // contact shadows, key light, rim light, and hemisphere bounce stay 100% world-locked as camera rotates!
    float lx = wx, ly = wy, lz = wz;
    float ux = 0.0f, uy = 1.0f, uz = 0.0f;
    if (in.has_view_rot) {
        const float *R = in.view_rot;
        lx = R[0] * wx + R[1] * wy + R[2] * wz;
        ly = R[3] * wx + R[4] * wy + R[5] * wz;
        lz = R[6] * wx + R[7] * wy + R[8] * wz;
        ux = R[1];
        uy = R[4];
        uz = R[7];
    }
    const float llen = std::sqrt(lx * lx + ly * ly + lz * lz);
    if (llen > 1e-4f) { lx /= llen; ly /= llen; lz /= llen; }
    const float ulen = std::sqrt(ux * ux + uy * uy + uz * uz);
    if (ulen > 1e-4f) { ux /= ulen; uy /= ulen; uz /= ulen; }

    p.sun_dir_view[0] = lx;
    p.sun_dir_view[1] = ly;
    p.sun_dir_view[2] = lz;
    p.sun_dir_view[3] = active_is_cave_ ? 1.0f : 0.0f;
    p.world_up_view[0] = ux;
    p.world_up_view[1] = uy;
    p.world_up_view[2] = uz;

    // Project sun direction onto screen UV for volumetric god-rays (smoothly fades when sun is behind camera).
    const float sun_facing = std::clamp((-lz + 0.25f) / 1.15f, 0.0f, 1.0f);
    const float inv_z = 1.0f / std::max(-lz, 0.25f);
    p.sun_screen[0] = std::clamp(0.5f + 0.5f * (lx * p00 * inv_z), -0.35f, 1.35f);
    p.sun_screen[1] = std::clamp(0.5f - 0.5f * (ly * p11 * inv_z), -0.35f, 1.35f);
    p.sun_screen[2] = active_is_cave_ ? 0.0f : sun_facing;
    p.sun_screen[3] = static_cast<float>(in.frame_index & 63u);

    // Biome-tuned lighting, AO, contact shadows, hemisphere ground/magma bounce, specular sheen, SSGI, fog & bloom.
    float shadow_rgb[3]{0.78f, 0.82f, 0.90f};
    float ao_str = 0.60f;
    float fog_rgb[3]{0.70f, 0.78f, 0.88f};
    float fog_dens = 0.00040f;
    float scat_rgb[3]{1.00f, 0.94f, 0.82f};
    float scat_str = 0.22f;
    float bloom_th = 0.80f;
    float bloom_str = 0.26f;
    float rim_str = 0.14f;
    float key_str = 0.14f;
    float ground_rgb[3]{0.38f, 0.40f, 0.36f};
    float hemi_str = 0.09f;
    float spec_str = 0.12f;
    float spec_gloss = 28.0f;
    float contact_str = 0.42f;
    float ssgi_str = 0.22f;
    float vibrance = 0.06f;

    switch (active_biome_) {
    case BiomeKind::Snowy:
        // Pokke Village & Snowy Mountains: crisp alpine sky bounce, bright snow ground bounce, icy specular glint.
        shadow_rgb[0] = 0.72f; shadow_rgb[1] = 0.82f; shadow_rgb[2] = 0.96f;
        ao_str = 0.58f;
        fog_rgb[0] = 0.80f; fog_rgb[1] = 0.88f; fog_rgb[2] = 0.96f;
        fog_dens = 0.00042f;
        scat_rgb[0] = 0.94f; scat_rgb[1] = 0.97f; scat_rgb[2] = 1.00f;
        scat_str = 0.20f;
        bloom_th = 0.84f; bloom_str = 0.26f;
        rim_str = 0.15f; key_str = 0.14f;
        ground_rgb[0] = 0.80f; ground_rgb[1] = 0.88f; ground_rgb[2] = 0.98f;
        hemi_str = 0.10f;
        spec_str = 0.16f; spec_gloss = 38.0f;
        contact_str = 0.45f;
        ssgi_str = 0.18f;
        vibrance = 0.05f;
        break;
    case BiomeKind::Desert:
        // Desert / Old Desert: golden sunlight, warm sand ground bounce, crisp sunlit contact shadows.
        shadow_rgb[0] = 0.84f; shadow_rgb[1] = 0.78f; shadow_rgb[2] = 0.70f;
        ao_str = 0.56f;
        fog_rgb[0] = 0.90f; fog_rgb[1] = 0.82f; fog_rgb[2] = 0.68f;
        fog_dens = 0.00036f;
        scat_rgb[0] = 1.00f; scat_rgb[1] = 0.90f; scat_rgb[2] = 0.70f;
        scat_str = 0.24f;
        bloom_th = 0.82f; bloom_str = 0.24f;
        rim_str = 0.15f; key_str = 0.16f;
        ground_rgb[0] = 0.86f; ground_rgb[1] = 0.72f; ground_rgb[2] = 0.50f;
        hemi_str = 0.11f;
        spec_str = 0.13f; spec_gloss = 26.0f;
        contact_str = 0.48f;
        ssgi_str = 0.20f;
        vibrance = 0.07f;
        break;
    case BiomeKind::JungleForest:
        // Jungle / Forest & Hills / Great Forest: lush canopy sun shafts, humid wet-rock specular sheen.
        shadow_rgb[0] = 0.72f; shadow_rgb[1] = 0.82f; shadow_rgb[2] = 0.76f;
        ao_str = 0.64f;
        fog_rgb[0] = 0.70f; fog_rgb[1] = 0.80f; fog_rgb[2] = 0.74f;
        fog_dens = 0.00048f;
        scat_rgb[0] = 0.96f; scat_rgb[1] = 0.98f; scat_rgb[2] = 0.80f;
        scat_str = 0.28f;
        bloom_th = 0.80f; bloom_str = 0.25f;
        rim_str = 0.15f; key_str = 0.14f;
        ground_rgb[0] = 0.36f; ground_rgb[1] = 0.54f; ground_rgb[2] = 0.34f;
        hemi_str = 0.10f;
        spec_str = 0.18f; spec_gloss = 42.0f; // rain-slick wet sheen
        contact_str = 0.44f;
        ssgi_str = 0.24f;
        vibrance = 0.08f;
        break;
    case BiomeKind::Swamp:
        // Swamp / Old Swamp: misty teal/violet fog, wet sheen on stone & monster scales.
        shadow_rgb[0] = 0.72f; shadow_rgb[1] = 0.78f; shadow_rgb[2] = 0.86f;
        ao_str = 0.64f;
        fog_rgb[0] = 0.64f; fog_rgb[1] = 0.72f; fog_rgb[2] = 0.78f;
        fog_dens = 0.00064f;
        scat_rgb[0] = 0.82f; scat_rgb[1] = 0.90f; scat_rgb[2] = 0.94f;
        scat_str = 0.22f;
        bloom_th = 0.78f; bloom_str = 0.24f;
        rim_str = 0.14f; key_str = 0.12f;
        ground_rgb[0] = 0.32f; ground_rgb[1] = 0.42f; ground_rgb[2] = 0.46f;
        hemi_str = 0.09f;
        spec_str = 0.19f; spec_gloss = 44.0f; // damp/rainy specular
        contact_str = 0.40f;
        ssgi_str = 0.24f;
        vibrance = 0.05f;
        break;
    case BiomeKind::Volcano:
        // Volcano / Battleground: warm magma under-glow from below, fiery ember bloom & strong SSGI lava bounce.
        shadow_rgb[0] = 0.80f; shadow_rgb[1] = 0.68f; shadow_rgb[2] = 0.64f;
        ao_str = 0.62f;
        fog_rgb[0] = 0.62f; fog_rgb[1] = 0.36f; fog_rgb[2] = 0.24f;
        fog_dens = 0.00050f;
        scat_rgb[0] = 1.00f; scat_rgb[1] = 0.62f; scat_rgb[2] = 0.32f;
        scat_str = 0.26f;
        bloom_th = 0.70f; bloom_str = 0.38f;
        rim_str = 0.18f; key_str = 0.14f;
        ground_rgb[0] = 1.00f; ground_rgb[1] = 0.38f; ground_rgb[2] = 0.12f; // magma under-glow!
        hemi_str = 0.16f;
        spec_str = 0.16f; spec_gloss = 30.0f;
        contact_str = 0.38f;
        ssgi_str = 0.38f; // strong lava & fireball bounce!
        vibrance = 0.08f;
        break;
    case BiomeKind::CaveIndoor:
        // Caves & Pokke indoors (House, Hall, Kitchen, Farm): ZERO sun shafts/sun contact shadows,
        // rich architectural AO, warm torch/campfire SSGI bounce!
        shadow_rgb[0] = 0.72f; shadow_rgb[1] = 0.74f; shadow_rgb[2] = 0.80f;
        ao_str = 0.70f;
        fog_rgb[0] = 0.28f; fog_rgb[1] = 0.30f; fog_rgb[2] = 0.36f;
        fog_dens = 0.00028f;
        scat_rgb[0] = 0.95f; scat_rgb[1] = 0.82f; scat_rgb[2] = 0.66f;
        scat_str = 0.0f;
        bloom_th = 0.76f; bloom_str = 0.30f;
        rim_str = 0.08f; key_str = 0.08f;
        ground_rgb[0] = 0.42f; ground_rgb[1] = 0.34f; ground_rgb[2] = 0.28f;
        hemi_str = 0.07f;
        spec_str = 0.13f; spec_gloss = 34.0f; // damp cave rock / warm wood
        contact_str = 0.0f; // no directional sun inside caves/indoors
        ssgi_str = 0.36f; // campfires, torches, Khezu lightning illuminate walls & hunter!
        vibrance = 0.05f;
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
        // Dark night sky/fog: tone down sun shafts gently and cool down scatter into moonlight.
        if (psp_luma < 0.22f && active_biome_ != BiomeKind::Volcano && !active_is_cave_) {
            scat_str *= 0.55f;
            scat_rgb[0] = 0.72f;
            scat_rgb[1] = 0.82f;
            scat_rgb[2] = 1.00f;
            contact_str *= 0.75f;
            ssgi_str = std::max(ssgi_str, 0.30f); // campfires/fireballs pop at night!
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
    p.world_up_view[3] = ssgi_str;
    p.ground_bounce[0] = ground_rgb[0]; p.ground_bounce[1] = ground_rgb[1]; p.ground_bounce[2] = ground_rgb[2];
    p.ground_bounce[3] = hemi_str;
    p.spec_contact[0] = spec_str;
    p.spec_contact[1] = spec_gloss;
    p.spec_contact[2] = contact_str;
    p.spec_contact[3] = vibrance;
    p.ext_slots[2] = static_cast<std::uint32_t>(active_biome_);
    p.ext_slots[3] = active_is_cave_ ? 1u : 0u;
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
    base.tex_slots[2] = gtao_blur_tex_.slot;
    base.tex_slots[3] = vol_tex_.slot;
    base.aux_slots[0] = bloom_blur_tex_.slot;
    base.aux_slots[1] = BindlessDescriptors::sampler_index(true, false, true, true, 1);
    base.ext_slots[0] = ssgi_blur_tex_.slot;
    base.ext_slots[1] = gtao_tex_.slot;

    auto *slots = reinterpret_cast<LightingComputeParams *>(params_cpu_ptr);
    slots[0] = base; slots[0].aux_slots[2] = gtao_tex_.w;       slots[0].aux_slots[3] = gtao_tex_.h;
    slots[1] = base; slots[1].aux_slots[2] = vol_tex_.w;        slots[1].aux_slots[3] = vol_tex_.h;
    slots[2] = base; slots[2].aux_slots[2] = bloom_tex_.w;      slots[2].aux_slots[3] = bloom_tex_.h;
    slots[3] = base; slots[3].aux_slots[2] = ssgi_tex_.w;       slots[3].aux_slots[3] = ssgi_tex_.h;
    slots[4] = base; slots[4].aux_slots[2] = gtao_blur_tex_.w;  slots[4].aux_slots[3] = gtao_blur_tex_.h;
    slots[5] = base; slots[5].aux_slots[0] = bloom_tex_.slot;   slots[5].aux_slots[2] = bloom_blur_tex_.w; slots[5].aux_slots[3] = bloom_blur_tex_.h;
    slots[6] = base; slots[6].aux_slots[0] = ssgi_tex_.slot;    slots[6].aux_slots[2] = ssgi_blur_tex_.w;  slots[6].aux_slots[3] = ssgi_blur_tex_.h;
    slots[7] = base; slots[7].aux_slots[2] = scene_w;           slots[7].aux_slots[3] = scene_h;

    // Transition intermediate targets -> GENERAL for compute writes, waiting for any prior
    // composite/denoise pass's sampled reads to finish so back-to-back dispatches never race!
    auto to_general = [&](VkImage img) {
        Context::cmd_transition_image(cmd, img, VK_IMAGE_ASPECT_COLOR_BIT,
                                      VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                      VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    };
    auto to_read = [&](VkImage img) {
        Context::cmd_transition_image(cmd, img, VK_IMAGE_ASPECT_COLOR_BIT,
                                      VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    };

    to_general(gtao_tex_.image);
    to_general(gtao_blur_tex_.image);
    to_general(vol_tex_.image);
    to_general(bloom_tex_.image);
    to_general(bloom_blur_tex_.image);
    to_general(ssgi_tex_.image);
    to_general(ssgi_blur_tex_.image);

    // Stage 1: GTAO+ContactShadows (1/2), Volumetric Fog/Shafts (1/4), Bloom (1/4), SSGI-lite (1/4)
    // all read scene_depth/scene_color in SHADER_READ_ONLY -> execute in parallel across GPU CUs!
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
    run_pass(ssgi_pipe_, ssgi_set_, ssgi_tex_.w, ssgi_tex_.h, 3);

    // Stage 2: Bilateral GTAO/ContactShadow denoise (1/2) + Kawase Bloom blur (1/4) + Kawase SSGI blur (1/4)
    // all execute in parallel!
    to_read(gtao_tex_.image);
    to_read(bloom_tex_.image);
    to_read(ssgi_tex_.image);
    run_pass(gtao_denoise_pipe_, gtao_blur_set_, gtao_blur_tex_.w, gtao_blur_tex_.h, 4);
    run_pass(glow_blur_pipe_, bloom_blur_set_, bloom_blur_tex_.w, bloom_blur_tex_.h, 5);
    run_pass(glow_blur_pipe_, ssgi_blur_set_, ssgi_blur_tex_.w, ssgi_blur_tex_.h, 6);

    // Barrier batch: transition smoothed gtao/vol/bloom/ssgi -> SHADER_READ_ONLY, scene_color -> GENERAL.
    to_read(gtao_blur_tex_.image);
    to_read(vol_tex_.image);
    to_read(bloom_blur_tex_.image);
    to_read(ssgi_blur_tex_.image);
    Context::cmd_transition_image(cmd, scene_color, VK_IMAGE_ASPECT_COLOR_BIT,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                  VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    // Stage 3: Unified full-res composite in-place on scene_color.
    run_pass(comp_pipe_, scene_set, scene_w, scene_h, 7);
}

} // namespace mhf::psp::vk
