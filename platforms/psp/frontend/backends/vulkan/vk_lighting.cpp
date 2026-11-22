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

BiomeKind classify_zone_name(const std::string &name, bool &is_cave) {
    // Pokke Farm is open-air alpine outdoors (unlike Hall/House/Kitchen).
    if (name.find("pokke/farm") != std::string::npos || name.find("pokke/village") != std::string::npos) {
        is_cave = false;
        return BiomeKind::PokkeVillage;
    }
    if (name.find("pokke/hall") != std::string::npos || name.find("pokke/house") != std::string::npos ||
        name.find("pokke/kitchen") != std::string::npos) {
        is_cave = true;
        return BiomeKind::CozyIndoor;
    }
    // Volcano & Battleground (even volcano/area5 is a glowing magma cavern, not a dark cold cave!).
    if (name.find("volcano") != std::string::npos || name.find("battleground") != std::string::npos) {
        return BiomeKind::Volcano;
    }
    // Snowy Mountains: distinguish Ice Caves (areas 3-5), high peaks (6-8 & Peak), and lower valley (base/1/2).
    if (name.find("snowy_mountains") != std::string::npos) {
        if (is_cave) return BiomeKind::IceCave;
        if (name.find("peak") != std::string::npos || name.find("area6") != std::string::npos ||
            name.find("area7") != std::string::npos || name.find("area8") != std::string::npos) {
            return BiomeKind::SnowyPeak;
        }
        return BiomeKind::PokkeVillage;
    }
    // Remaining caves (Swamp, Old Swamp, Desert area10, Great Forest 7/8, Old Jungle 4/9/10).
    if (is_cave && name.find("fortress") == std::string::npos &&
        name.find("castle_schrade") == std::string::npos && name.find("tower") == std::string::npos) {
        return BiomeKind::DarkCave;
    }
    if (name.find("desert/area2") != std::string::npos) return BiomeKind::DesertSandstorm;
    if (name.find("desert") != std::string::npos) return BiomeKind::Desert;
    if (name.find("great_forest") != std::string::npos) return BiomeKind::GreatForest;
    if (name.find("forestandhills") != std::string::npos) return BiomeKind::ForestHills;
    if (name.find("jungle") != std::string::npos) return BiomeKind::Jungle;
    if (name.find("swamp") != std::string::npos) return BiomeKind::Swamp;
    if (name.find("tower") != std::string::npos) {
        is_cave = false; // Tower is high ancient ruins above the clouds!
        return BiomeKind::Tower;
    }
    if (name.find("castle_schrade") != std::string::npos || name.find("fortress") != std::string::npos ||
        name.find("town") != std::string::npos) {
        is_cave = false; // Open-air dramatic siege / fortress skies
        return BiomeKind::FortressSiege;
    }
    if (name.find("arena") != std::string::npos) return BiomeKind::Arena;
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

    constexpr std::uint32_t kTotalSets = 9u + static_cast<std::uint32_t>(kSceneSetCount);
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
    flash_set_ = sets[7];
    lit_tmp_set_ = sets[8];
    for (std::size_t i = 0; i < kSceneSetCount; ++i) {
        scene_sets_[i] = sets[9 + i];
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
    flash_pipe_ = make_pipe(shaders::kEmissiveFlashCompSpv);
    comp_pipe_ = make_pipe(shaders::kLightingCompositeCompSpv);
    fxaa_pipe_ = make_pipe(shaders::kFxaaPrehudCompSpv);

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
            bool is_cave = caves.count(zname) > 0;
            zones_[i].biome = classify_zone_name(zname, is_cave);
            zones_[i].is_cave = is_cave;
            if (auto it = suns.find(zname); it != suns.end()) {
                zones_[i].sun_height_deg = it->second[0];
                zones_[i].sun_darkness = it->second[1];
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

    live_ = live;
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
    flash_tex_.destroy(ctx, desc);
    lit_tmp_tex_.destroy(ctx, desc);
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
    kill_pipe(flash_pipe_);
    kill_pipe(comp_pipe_);
    kill_pipe(fxaa_pipe_);
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
           make_target(ssgi_blur_tex_, qw, qh, VK_FORMAT_R16G16B16A16_SFLOAT, ssgi_blur_set_) &&
           make_target(flash_tex_, 4u, 2u, VK_FORMAT_R16G16B16A16_SFLOAT, flash_set_) &&
           make_target(lit_tmp_tex_, full_w, full_h, VK_FORMAT_R8G8B8A8_UNORM, lit_tmp_set_);
}

void LightingPipeline::fill_biome_params(LightingComputeParams &p, const LightingFrameInput &in) noexcept {
    // Projection info for view-space reconstruction.
    const float p00 = std::fabs(in.proj[0]) > 1e-4f ? in.proj[0] : 1.0f;
    const float p11 = std::fabs(in.proj[5]) > 1e-4f ? in.proj[5] : 1.76f;
    p.proj_info[0] = 1.0f / p00;
    p.proj_info[1] = 1.0f / p11;
    p.proj_info[2] = in.proj[10];
    p.proj_info[3] = std::fabs(in.proj[14]) > 1e-4f ? in.proj[14] : -10.0f;
    p.vz_and_size[0] = in.vz_scale;
    p.vz_and_size[1] = in.vz_center;

    // World-space light direction: overhead indoor light for caves/halls (grounds hunter & props onto floor),
    // otherwise live PSP GE directional light 0 or zone sun elevation/azimuth.
    float wx = 0.34f, wy = 0.78f, wz = 0.52f;
    if (active_is_cave_) {
        wx = 0.14f; wy = 0.95f; wz = 0.28f;
    } else if (in.has_light0) {
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
    float contact_str = 0.52f;
    float ssgi_str = 0.22f;
    float vibrance = 0.06f;

    switch (active_biome_) {
    case BiomeKind::PokkeVillage:
        // Pokke Village, Pokke Farm & Lower Snowy Valley: crisp alpine sky, clean visibility, subtle warm hearth glow.
        shadow_rgb[0] = 0.74f; shadow_rgb[1] = 0.83f; shadow_rgb[2] = 0.96f;
        ao_str = 0.56f;
        fog_rgb[0] = 0.80f; fog_rgb[1] = 0.88f; fog_rgb[2] = 0.96f;
        fog_dens = 0.00032f;
        scat_rgb[0] = 0.96f; scat_rgb[1] = 0.97f; scat_rgb[2] = 1.00f;
        scat_str = 0.22f;
        bloom_th = 0.84f; bloom_str = 0.26f;
        rim_str = 0.14f; key_str = 0.14f;
        ground_rgb[0] = 0.82f; ground_rgb[1] = 0.88f; ground_rgb[2] = 0.96f;
        hemi_str = 0.10f;
        spec_str = 0.15f; spec_gloss = 36.0f;
        contact_str = 0.52f;
        ssgi_str = 0.20f;
        vibrance = 0.05f;
        break;
    case BiomeKind::SnowyPeak:
        // Snowy Mountains high slopes (Areas 6-8 & Peak): crisp blizzard mist, dazzling snow bounce, icy specular glint.
        shadow_rgb[0] = 0.68f; shadow_rgb[1] = 0.80f; shadow_rgb[2] = 0.98f;
        ao_str = 0.56f;
        fog_rgb[0] = 0.82f; fog_rgb[1] = 0.89f; fog_rgb[2] = 0.98f;
        fog_dens = 0.00046f;
        scat_rgb[0] = 0.92f; scat_rgb[1] = 0.97f; scat_rgb[2] = 1.00f;
        scat_str = 0.24f;
        bloom_th = 0.85f; bloom_str = 0.24f;
        rim_str = 0.17f; key_str = 0.14f;
        ground_rgb[0] = 0.84f; ground_rgb[1] = 0.91f; ground_rgb[2] = 1.00f;
        hemi_str = 0.11f;
        spec_str = 0.19f; spec_gloss = 44.0f; // crisp alpine ice glint
        contact_str = 0.54f;
        ssgi_str = 0.16f;
        vibrance = 0.04f;
        break;
    case BiomeKind::IceCave:
        // Snowy Mountains Ice Caves (Areas 3-5, Khezu): glacial cyan-blue shadows, icy wall sheen, strong lightning bounce.
        shadow_rgb[0] = 0.62f; shadow_rgb[1] = 0.78f; shadow_rgb[2] = 0.96f;
        ao_str = 0.68f;
        fog_rgb[0] = 0.34f; fog_rgb[1] = 0.46f; fog_rgb[2] = 0.58f;
        fog_dens = 0.00034f;
        scat_rgb[0] = 0.78f; scat_rgb[1] = 0.92f; scat_rgb[2] = 1.00f;
        scat_str = 0.0f;
        bloom_th = 0.76f; bloom_str = 0.32f;
        rim_str = 0.12f; key_str = 0.09f;
        ground_rgb[0] = 0.46f; ground_rgb[1] = 0.66f; ground_rgb[2] = 0.82f; // ice floor bounce
        hemi_str = 0.10f;
        spec_str = 0.22f; spec_gloss = 48.0f; // crystalline ice walls
        contact_str = 0.36f;
        ssgi_str = 0.38f; // Khezu electric-blue lightning illuminates ice walls!
        vibrance = 0.06f;
        break;
    case BiomeKind::CozyIndoor:
        // Pokke Gathering Hall, House, Kitchen: warm timber & hearth amber bounce, cozy fire SSGI, doorway light.
        shadow_rgb[0] = 0.76f; shadow_rgb[1] = 0.70f; shadow_rgb[2] = 0.66f;
        ao_str = 0.68f;
        fog_rgb[0] = 0.32f; fog_rgb[1] = 0.26f; fog_rgb[2] = 0.22f;
        fog_dens = 0.00022f;
        scat_rgb[0] = 1.00f; scat_rgb[1] = 0.84f; scat_rgb[2] = 0.62f;
        scat_str = 0.0f;
        bloom_th = 0.75f; bloom_str = 0.30f;
        rim_str = 0.09f; key_str = 0.08f;
        ground_rgb[0] = 0.56f; ground_rgb[1] = 0.38f; ground_rgb[2] = 0.22f; // warm wood/hearth bounce
        hemi_str = 0.09f;
        spec_str = 0.13f; spec_gloss = 32.0f;
        contact_str = 0.34f;
        ssgi_str = 0.36f;
        vibrance = 0.05f;
        break;
    case BiomeKind::DarkCave:
        // Damp rock caves (Swamp, Desert Area 10, Old Jungle, Great Forest 7/8): rich stone AO, torch & glow bounce.
        shadow_rgb[0] = 0.68f; shadow_rgb[1] = 0.72f; shadow_rgb[2] = 0.78f;
        ao_str = 0.72f;
        fog_rgb[0] = 0.24f; fog_rgb[1] = 0.28f; fog_rgb[2] = 0.34f;
        fog_dens = 0.00032f;
        scat_rgb[0] = 0.90f; scat_rgb[1] = 0.84f; scat_rgb[2] = 0.72f;
        scat_str = 0.0f;
        bloom_th = 0.74f; bloom_str = 0.32f;
        rim_str = 0.09f; key_str = 0.08f;
        ground_rgb[0] = 0.34f; ground_rgb[1] = 0.34f; ground_rgb[2] = 0.38f;
        hemi_str = 0.07f;
        spec_str = 0.16f; spec_gloss = 38.0f; // damp cavern rock
        contact_str = 0.36f;
        ssgi_str = 0.38f;
        vibrance = 0.05f;
        break;
    case BiomeKind::Desert:
        // Desert & Old Desert: harsh golden sun, warm sand ground bounce, crisp sunlit contact shadows.
        shadow_rgb[0] = 0.82f; shadow_rgb[1] = 0.76f; shadow_rgb[2] = 0.70f;
        ao_str = 0.56f;
        fog_rgb[0] = 0.92f; fog_rgb[1] = 0.83f; fog_rgb[2] = 0.66f;
        fog_dens = 0.00034f;
        scat_rgb[0] = 1.00f; scat_rgb[1] = 0.90f; scat_rgb[2] = 0.68f;
        scat_str = 0.26f;
        bloom_th = 0.82f; bloom_str = 0.24f;
        rim_str = 0.16f; key_str = 0.17f;
        ground_rgb[0] = 0.88f; ground_rgb[1] = 0.72f; ground_rgb[2] = 0.48f;
        hemi_str = 0.12f;
        spec_str = 0.13f; spec_gloss = 26.0f;
        contact_str = 0.58f;
        ssgi_str = 0.20f;
        vibrance = 0.07f;
        break;
    case BiomeKind::DesertSandstorm:
        // Desert Area 2 Sandstorm: thicker warm ochre dust haze, diffused soft shadows, glowing dust scatter.
        shadow_rgb[0] = 0.86f; shadow_rgb[1] = 0.78f; shadow_rgb[2] = 0.64f;
        ao_str = 0.52f;
        fog_rgb[0] = 0.88f; fog_rgb[1] = 0.76f; fog_rgb[2] = 0.56f;
        fog_dens = 0.00072f;
        scat_rgb[0] = 1.00f; scat_rgb[1] = 0.86f; scat_rgb[2] = 0.62f;
        scat_str = 0.30f;
        bloom_th = 0.82f; bloom_str = 0.22f;
        rim_str = 0.13f; key_str = 0.11f;
        ground_rgb[0] = 0.86f; ground_rgb[1] = 0.72f; ground_rgb[2] = 0.50f;
        hemi_str = 0.12f;
        spec_str = 0.09f; spec_gloss = 20.0f;
        contact_str = 0.34f; // diffused sandstorm light
        ssgi_str = 0.18f;
        vibrance = 0.04f;
        break;
    case BiomeKind::ForestHills:
        // Forest & Hills: classic sunlit meadows & cliffs, clear distance, warm golden-green sunbeams.
        shadow_rgb[0] = 0.72f; shadow_rgb[1] = 0.80f; shadow_rgb[2] = 0.86f;
        ao_str = 0.60f;
        fog_rgb[0] = 0.74f; fog_rgb[1] = 0.84f; fog_rgb[2] = 0.88f;
        fog_dens = 0.00034f;
        scat_rgb[0] = 1.00f; scat_rgb[1] = 0.96f; scat_rgb[2] = 0.80f;
        scat_str = 0.26f;
        bloom_th = 0.82f; bloom_str = 0.24f;
        rim_str = 0.15f; key_str = 0.16f;
        ground_rgb[0] = 0.44f; ground_rgb[1] = 0.62f; ground_rgb[2] = 0.34f;
        hemi_str = 0.10f;
        spec_str = 0.13f; spec_gloss = 30.0f;
        contact_str = 0.56f;
        ssgi_str = 0.20f;
        vibrance = 0.08f;
        break;
    case BiomeKind::GreatForest:
        // Great Forest (Nargacuga / Chameleos): dense ancient canopy, strong cathedral god-rays, mossy AO & humid mist.
        shadow_rgb[0] = 0.68f; shadow_rgb[1] = 0.80f; shadow_rgb[2] = 0.72f;
        ao_str = 0.66f;
        fog_rgb[0] = 0.66f; fog_rgb[1] = 0.78f; fog_rgb[2] = 0.68f;
        fog_dens = 0.00054f;
        scat_rgb[0] = 0.94f; scat_rgb[1] = 1.00f; scat_rgb[2] = 0.76f;
        scat_str = 0.36f; // signature cathedral canopy sun shafts!
        bloom_th = 0.80f; bloom_str = 0.26f;
        rim_str = 0.16f; key_str = 0.15f;
        ground_rgb[0] = 0.34f; ground_rgb[1] = 0.56f; ground_rgb[2] = 0.32f;
        hemi_str = 0.11f;
        spec_str = 0.17f; spec_gloss = 40.0f;
        contact_str = 0.54f;
        ssgi_str = 0.24f;
        vibrance = 0.09f;
        break;
    case BiomeKind::Jungle:
        // Jungle & Old Jungle: humid tropical palms & coastal cliffs, wet-rock specular sheen, bright canopy shafts.
        shadow_rgb[0] = 0.70f; shadow_rgb[1] = 0.82f; shadow_rgb[2] = 0.78f;
        ao_str = 0.64f;
        fog_rgb[0] = 0.70f; fog_rgb[1] = 0.82f; fog_rgb[2] = 0.78f;
        fog_dens = 0.00046f;
        scat_rgb[0] = 0.96f; scat_rgb[1] = 0.98f; scat_rgb[2] = 0.82f;
        scat_str = 0.30f;
        bloom_th = 0.80f; bloom_str = 0.25f;
        rim_str = 0.15f; key_str = 0.15f;
        ground_rgb[0] = 0.38f; ground_rgb[1] = 0.56f; ground_rgb[2] = 0.38f;
        hemi_str = 0.10f;
        spec_str = 0.20f; spec_gloss = 44.0f; // rain-slick coastal stone & monster scales
        contact_str = 0.54f;
        ssgi_str = 0.22f;
        vibrance = 0.08f;
        break;
    case BiomeKind::Swamp:
        // Swamp & Old Swamp: moody teal/violet marsh mist, damp specular on stone & scales, eerie low shafts.
        shadow_rgb[0] = 0.70f; shadow_rgb[1] = 0.76f; shadow_rgb[2] = 0.88f;
        ao_str = 0.66f;
        fog_rgb[0] = 0.60f; fog_rgb[1] = 0.68f; fog_rgb[2] = 0.78f;
        fog_dens = 0.00066f;
        scat_rgb[0] = 0.80f; scat_rgb[1] = 0.88f; scat_rgb[2] = 0.96f;
        scat_str = 0.24f;
        bloom_th = 0.78f; bloom_str = 0.26f;
        rim_str = 0.14f; key_str = 0.12f;
        ground_rgb[0] = 0.30f; ground_rgb[1] = 0.40f; ground_rgb[2] = 0.48f;
        hemi_str = 0.09f;
        spec_str = 0.21f; spec_gloss = 46.0f; // wet marsh sheen
        contact_str = 0.46f;
        ssgi_str = 0.26f;
        vibrance = 0.05f;
        break;
    case BiomeKind::Volcano:
        // Volcano, Old Volcano & Battleground: intense magma under-glow from below, fiery ember bloom & strong lava SSGI.
        shadow_rgb[0] = 0.80f; shadow_rgb[1] = 0.66f; shadow_rgb[2] = 0.62f;
        ao_str = 0.64f;
        fog_rgb[0] = 0.60f; fog_rgb[1] = 0.34f; fog_rgb[2] = 0.22f;
        fog_dens = 0.00048f;
        scat_rgb[0] = 1.00f; scat_rgb[1] = 0.60f; scat_rgb[2] = 0.28f;
        scat_str = active_is_cave_ ? 0.0f : 0.28f;
        bloom_th = 0.70f; bloom_str = 0.38f;
        rim_str = 0.19f; key_str = 0.14f;
        ground_rgb[0] = 1.00f; ground_rgb[1] = 0.36f; ground_rgb[2] = 0.10f; // glowing magma under-light!
        hemi_str = 0.18f;
        spec_str = 0.16f; spec_gloss = 30.0f;
        contact_str = active_is_cave_ ? 0.38f : 0.48f;
        ssgi_str = 0.40f; // rich lava & fireball indirect glow!
        vibrance = 0.08f;
        break;
    case BiomeKind::Tower:
        // Ancient Tower (1/2/3): golden sandstone ruins above the clouds, dramatic sunlit god-rays & cloud-sea mist.
        shadow_rgb[0] = 0.78f; shadow_rgb[1] = 0.76f; shadow_rgb[2] = 0.86f;
        ao_str = 0.64f;
        fog_rgb[0] = 0.88f; fog_rgb[1] = 0.86f; fog_rgb[2] = 0.82f;
        fog_dens = 0.00044f;
        scat_rgb[0] = 1.00f; scat_rgb[1] = 0.92f; scat_rgb[2] = 0.72f;
        scat_str = 0.36f; // epic golden temple god-rays!
        bloom_th = 0.78f; bloom_str = 0.28f;
        rim_str = 0.18f; key_str = 0.16f;
        ground_rgb[0] = 0.78f; ground_rgb[1] = 0.70f; ground_rgb[2] = 0.56f; // warm ancient sandstone
        hemi_str = 0.11f;
        spec_str = 0.15f; spec_gloss = 32.0f;
        contact_str = 0.56f;
        ssgi_str = 0.24f;
        vibrance = 0.07f;
        break;
    case BiomeKind::FortressSiege:
        // Castle Schrade, Fortress & Town (Fatalis / Lao-Shan / Shen Gaoren): dramatic stormy stone AO & siege mist.
        shadow_rgb[0] = 0.68f; shadow_rgb[1] = 0.72f; shadow_rgb[2] = 0.80f;
        ao_str = 0.68f;
        fog_rgb[0] = 0.58f; fog_rgb[1] = 0.62f; fog_rgb[2] = 0.68f;
        fog_dens = 0.00050f;
        scat_rgb[0] = 0.94f; scat_rgb[1] = 0.88f; scat_rgb[2] = 0.80f;
        scat_str = 0.24f;
        bloom_th = 0.76f; bloom_str = 0.28f;
        rim_str = 0.17f; key_str = 0.14f;
        ground_rgb[0] = 0.44f; ground_rgb[1] = 0.42f; ground_rgb[2] = 0.40f;
        hemi_str = 0.09f;
        spec_str = 0.16f; spec_gloss = 34.0f;
        contact_str = 0.52f;
        ssgi_str = 0.28f; // Fatalis fire & cannon flashes light up castle walls!
        vibrance = 0.05f;
        break;
    case BiomeKind::Arena:
        // Arena & Great Arena: clean competitive sunlit colosseum, crisp contact shadows, minimal fog.
        shadow_rgb[0] = 0.80f; shadow_rgb[1] = 0.78f; shadow_rgb[2] = 0.76f;
        ao_str = 0.56f;
        fog_rgb[0] = 0.86f; fog_rgb[1] = 0.84f; fog_rgb[2] = 0.78f;
        fog_dens = 0.00018f; // ultra-clear combat readability
        scat_rgb[0] = 1.00f; scat_rgb[1] = 0.94f; scat_rgb[2] = 0.80f;
        scat_str = 0.16f;
        bloom_th = 0.82f; bloom_str = 0.22f;
        rim_str = 0.15f; key_str = 0.15f;
        ground_rgb[0] = 0.78f; ground_rgb[1] = 0.70f; ground_rgb[2] = 0.56f;
        hemi_str = 0.10f;
        spec_str = 0.14f; spec_gloss = 30.0f;
        contact_str = 0.56f;
        ssgi_str = 0.18f;
        vibrance = 0.06f;
        break;
    case BiomeKind::DefaultOutdoor:
        break;
    }

    // Modulate contact shadow & AO contrast gently using MHFU's hand-tuned per-zone sun darkness (when present).
    if (active_zone_ >= 0 && static_cast<std::size_t>(active_zone_) < zones_.size()) {
        const float zd = zones_[static_cast<std::size_t>(active_zone_)].sun_darkness;
        if (zd > 0.05f) {
            // 0.40 is MHFU's baseline shadow darkness; scale smoothly in [0.75, 1.20].
            const float scale = std::clamp(0.65f + zd * 0.85f, 0.75f, 1.20f);
            contact_str = std::clamp(contact_str * scale, 0.0f, 0.72f);
            ao_str = std::clamp(ao_str * (0.88f + zd * 0.28f), 0.45f, 0.78f);
        }
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

    // Smooth temporal mood blending (~0.3s transition on outdoor <-> cave / biome changes,
    // advancing once per 30 Hz game frame so all 120/180 Hz subframes stay 100% identical).
    float tgt_shadow[4]{shadow_rgb[0], shadow_rgb[1], shadow_rgb[2], ao_str};
    float tgt_fog[4]{fog_rgb[0], fog_rgb[1], fog_rgb[2], fog_dens};
    float tgt_scat[4]{scat_rgb[0], scat_rgb[1], scat_rgb[2], scat_str};
    float tgt_bloom[4]{bloom_th, bloom_str, rim_str, key_str};
    float tgt_ground[4]{ground_rgb[0], ground_rgb[1], ground_rgb[2], hemi_str};
    float tgt_spec[4]{spec_str, spec_gloss, contact_str, vibrance};

    if (!mood_.valid) {
        std::memcpy(mood_.shadow_tint, tgt_shadow, sizeof(tgt_shadow));
        std::memcpy(mood_.fog_color, tgt_fog, sizeof(tgt_fog));
        std::memcpy(mood_.scatter_color, tgt_scat, sizeof(tgt_scat));
        std::memcpy(mood_.bloom_rim, tgt_bloom, sizeof(tgt_bloom));
        std::memcpy(mood_.ground_bounce, tgt_ground, sizeof(tgt_ground));
        std::memcpy(mood_.spec_contact, tgt_spec, sizeof(tgt_spec));
        mood_.ssgi_str = ssgi_str;
        mood_.last_frame = in.frame_index;
        mood_.valid = true;
    } else if (in.frame_index != mood_.last_frame) {
        constexpr float kAlpha = 0.16f;
        auto blend4 = [&](float cur[4], const float tgt[4]) {
            for (int c = 0; c < 4; ++c) cur[c] += (tgt[c] - cur[c]) * kAlpha;
        };
        blend4(mood_.shadow_tint, tgt_shadow);
        blend4(mood_.fog_color, tgt_fog);
        blend4(mood_.scatter_color, tgt_scat);
        blend4(mood_.bloom_rim, tgt_bloom);
        blend4(mood_.ground_bounce, tgt_ground);
        blend4(mood_.spec_contact, tgt_spec);
        mood_.ssgi_str += (ssgi_str - mood_.ssgi_str) * kAlpha;
        mood_.last_frame = in.frame_index;
    }

    std::memcpy(p.shadow_tint, mood_.shadow_tint, sizeof(p.shadow_tint));
    std::memcpy(p.fog_color, mood_.fog_color, sizeof(p.fog_color));
    std::memcpy(p.scatter_color, mood_.scatter_color, sizeof(p.scatter_color));
    std::memcpy(p.bloom_rim, mood_.bloom_rim, sizeof(p.bloom_rim));
    p.world_up_view[3] = mood_.ssgi_str;
    std::memcpy(p.ground_bounce, mood_.ground_bounce, sizeof(p.ground_bounce));
    std::memcpy(p.spec_contact, mood_.spec_contact, sizeof(p.spec_contact));
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

    const std::uint32_t mask = live_ ? live_->vulkan_effects_mask.load(std::memory_order_relaxed) : LiveSettings::kVkEffectAll;
    const bool want_ao      = (mask & LiveSettings::kVkEffectAo) != 0u;
    const bool want_contact = (mask & LiveSettings::kVkEffectContactShadows) != 0u;
    const bool want_vol     = (mask & LiveSettings::kVkEffectVolumetricFog) != 0u;
    const bool want_bloom   = (mask & LiveSettings::kVkEffectBloom) != 0u;
    const bool want_ssgi    = (mask & LiveSettings::kVkEffectSsgi) != 0u;
    const bool want_spec    = (mask & LiveSettings::kVkEffectSpecular) != 0u;
    const bool want_fxaa    = (mask & LiveSettings::kVkEffectFxaa) != 0u;
    const bool want_vig     = (mask & LiveSettings::kVkEffectVignette) != 0u;

    const bool need_gtao_pass  = want_ao || want_contact || want_spec || want_ssgi;
    const bool need_bloom_src  = want_bloom || want_ssgi;

    LightingComputeParams base{};
    fill_biome_params(base, in);
    if (!want_ao)      base.shadow_tint[3] = 0.0f;
    if (!want_contact) base.spec_contact[2] = 0.0f;
    if (!want_bloom)   base.bloom_rim[1] = 0.0f;
    if (!want_ssgi)    base.world_up_view[3] = 0.0f;
    if (!want_spec) {
        base.ground_bounce[3] = 0.0f;
        base.bloom_rim[2] = 0.0f;
        base.bloom_rim[3] = 0.0f;
        base.spec_contact[0] = 0.0f;
        base.spec_contact[3] = 0.0f;
    }
    base.ext_slots[3] = (active_is_cave_ ? 1u : 0u) | (want_vig ? 2u : 0u) | (want_fxaa ? 4u : 0u);
    base.vz_and_size[2] = static_cast<float>(scene_w);
    base.vz_and_size[3] = static_cast<float>(scene_h);
    base.tex_slots[0] = scene_color_slot;
    base.tex_slots[1] = scene_depth_slot;
    base.tex_slots[2] = need_gtao_pass ? gtao_blur_tex_.slot : 0u;
    base.tex_slots[3] = want_vol ? vol_tex_.slot : 0u;
    base.aux_slots[0] = want_bloom ? bloom_blur_tex_.slot : 0u;
    base.aux_slots[1] = BindlessDescriptors::sampler_index(true, false, true, true, 1);
    base.ext_slots[0] = want_ssgi ? ssgi_blur_tex_.slot : 0u;
    base.ext_slots[1] = need_gtao_pass ? gtao_tex_.slot : 0u;

    auto *slots = reinterpret_cast<LightingComputeParams *>(params_cpu_ptr);
    slots[0] = base; slots[0].aux_slots[2] = gtao_tex_.w;       slots[0].aux_slots[3] = gtao_tex_.h;
    slots[1] = base; slots[1].aux_slots[2] = vol_tex_.w;        slots[1].aux_slots[3] = vol_tex_.h;
    slots[2] = base; slots[2].aux_slots[2] = bloom_tex_.w;      slots[2].aux_slots[3] = bloom_tex_.h;
    slots[3] = base; slots[3].aux_slots[2] = ssgi_tex_.w;       slots[3].aux_slots[3] = ssgi_tex_.h;
    slots[4] = base; slots[4].aux_slots[2] = gtao_blur_tex_.w;  slots[4].aux_slots[3] = gtao_blur_tex_.h;
    slots[5] = base; slots[5].aux_slots[0] = bloom_tex_.slot;   slots[5].aux_slots[2] = bloom_blur_tex_.w; slots[5].aux_slots[3] = bloom_blur_tex_.h;
    slots[6] = base; slots[6].aux_slots[0] = ssgi_tex_.slot;    slots[6].aux_slots[2] = ssgi_blur_tex_.w;  slots[6].aux_slots[3] = ssgi_blur_tex_.h;
    slots[7] = base; slots[7].aux_slots[0] = bloom_tex_.slot;   slots[7].aux_slots[2] = 4u; slots[7].aux_slots[3] = 2u;
    slots[8] = base; slots[8].ext_slots[1] = want_ssgi ? flash_tex_.slot : 0u; slots[8].aux_slots[2] = scene_w; slots[8].aux_slots[3] = scene_h;
    slots[9] = base; slots[9].aux_slots[0] = lit_tmp_tex_.slot; slots[9].aux_slots[2] = scene_w; slots[9].aux_slots[3] = scene_h;

    // Transition only the intermediate targets needed for enabled effects -> GENERAL for compute writes.
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

    if (need_gtao_pass) {
        to_general(gtao_tex_.image);
        to_general(gtao_blur_tex_.image);
    }
    if (want_vol)       to_general(vol_tex_.image);
    if (need_bloom_src) to_general(bloom_tex_.image);
    if (want_bloom)     to_general(bloom_blur_tex_.image);
    if (want_ssgi) {
        to_general(ssgi_tex_.image);
        to_general(ssgi_blur_tex_.image);
        to_general(flash_tex_.image);
    }
    to_general(lit_tmp_tex_.image);

    // Stage 1: Enabled half/quarter-res passes read scene_depth/scene_color in parallel across GPU CUs.
    auto run_pass = [&](VkPipeline pipe, VkDescriptorSet out_set, std::uint32_t w, std::uint32_t h, std::size_t idx) {
        std::array<VkDescriptorSet, 2> dsets = {desc.set(), out_set};
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_, 0, 2, dsets.data(), 0, nullptr);
        const VkDeviceAddress addr = params_bda + idx * sizeof(LightingComputeParams);
        vkCmdPushConstants(cmd, pipe_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(addr), &addr);
        vkCmdDispatch(cmd, (w + 7u) / 8u, (h + 7u) / 8u, 1u);
    };

    if (need_gtao_pass) run_pass(gtao_pipe_, gtao_set_, gtao_tex_.w, gtao_tex_.h, 0);
    if (want_vol)       run_pass(vol_pipe_, vol_set_, vol_tex_.w, vol_tex_.h, 1);
    if (need_bloom_src) run_pass(bloom_pipe_, bloom_set_, bloom_tex_.w, bloom_tex_.h, 2);
    if (want_ssgi)      run_pass(ssgi_pipe_, ssgi_set_, ssgi_tex_.w, ssgi_tex_.h, 3);

    // Stage 2: Bilateral GTAO/ContactShadow denoise + Kawase Bloom/SSGI blur + 4x2 dynamic flash.
    if (need_gtao_pass) to_read(gtao_tex_.image);
    if (need_bloom_src) to_read(bloom_tex_.image);
    if (want_ssgi)      to_read(ssgi_tex_.image);
    if (need_gtao_pass) run_pass(gtao_denoise_pipe_, gtao_blur_set_, gtao_blur_tex_.w, gtao_blur_tex_.h, 4);
    if (want_bloom)     run_pass(glow_blur_pipe_, bloom_blur_set_, bloom_blur_tex_.w, bloom_blur_tex_.h, 5);
    if (want_ssgi) {
        run_pass(glow_blur_pipe_, ssgi_blur_set_, ssgi_blur_tex_.w, ssgi_blur_tex_.h, 6);
        run_pass(flash_pipe_, flash_set_, 4u, 2u, 7);
    }

    // Barrier batch: transition produced targets -> SHADER_READ_ONLY (scene_color stays SHADER_READ_ONLY).
    if (need_gtao_pass) to_read(gtao_blur_tex_.image);
    if (want_vol)       to_read(vol_tex_.image);
    if (want_bloom)     to_read(bloom_blur_tex_.image);
    if (want_ssgi) {
        to_read(ssgi_blur_tex_.image);
        to_read(flash_tex_.image);
    }

    // Stage 3: Unified full-res composite reading scene_color -> writing lit_tmp_tex_.
    run_pass(comp_pipe_, lit_tmp_set_, scene_w, scene_h, 8);

    // Stage 4: Pre-HUD Edge Anti-Aliasing (FXAA 3.11, or fast pass-through copy when disabled) -> scene_color.
    to_read(lit_tmp_tex_.image);
    Context::cmd_transition_image(cmd, scene_color, VK_IMAGE_ASPECT_COLOR_BIT,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    run_pass(fxaa_pipe_, scene_set, scene_w, scene_h, 9);
}

} // namespace mhf::psp::vk
