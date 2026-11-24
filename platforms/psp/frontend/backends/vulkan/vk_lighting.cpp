#include "vk_lighting.hpp"
#include "vk_shaders_spv.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_set>

namespace mhf::psp::vk {

namespace {

struct BiomePreset {
    float shadow_rgb[3];
    float ao_str;
    float fog_rgb[3];
    float fog_dens;
    float scat_rgb[3];
    float scat_str;
    float bloom_th, bloom_str, rim_str, key_str;
    float ground_rgb[3];
    float hemi_str;
    float spec_str, spec_gloss, contact_str, ssgi_str, vibrance;
};

// Authentic MHFU biome lighting/atmosphere presets indexed by `BiomeKind`.
constexpr BiomePreset kBiomePresets[] = {
    // DefaultOutdoor
    {{0.78f, 0.82f, 0.90f}, 0.60f, {0.70f, 0.78f, 0.88f}, 0.00040f, {1.00f, 0.94f, 0.82f}, 0.22f,
     0.80f, 0.26f, 0.14f, 0.14f, {0.38f, 0.40f, 0.36f}, 0.09f, 0.12f, 28.0f, 0.52f, 0.22f, 0.06f},
    // PokkeVillage (Pokke Village, Farm & lower snowy valley)
    {{0.74f, 0.83f, 0.96f}, 0.56f, {0.80f, 0.88f, 0.96f}, 0.00032f, {0.96f, 0.97f, 1.00f}, 0.22f,
     0.84f, 0.26f, 0.14f, 0.14f, {0.82f, 0.88f, 0.96f}, 0.10f, 0.15f, 36.0f, 0.52f, 0.20f, 0.05f},
    // SnowyPeak (high slopes Areas 6-8 & Peak)
    {{0.68f, 0.80f, 0.98f}, 0.56f, {0.82f, 0.89f, 0.98f}, 0.00046f, {0.92f, 0.97f, 1.00f}, 0.24f,
     0.85f, 0.24f, 0.17f, 0.14f, {0.84f, 0.91f, 1.00f}, 0.11f, 0.19f, 44.0f, 0.54f, 0.16f, 0.04f},
    // IceCave (Snowy Mountains Areas 3-5)
    {{0.62f, 0.78f, 0.96f}, 0.68f, {0.34f, 0.46f, 0.58f}, 0.00034f, {0.78f, 0.92f, 1.00f}, 0.00f,
     0.76f, 0.32f, 0.12f, 0.09f, {0.46f, 0.66f, 0.82f}, 0.10f, 0.22f, 48.0f, 0.36f, 0.38f, 0.06f},
    // CozyIndoor (Pokke Hall, House, Kitchen)
    {{0.76f, 0.70f, 0.66f}, 0.68f, {0.32f, 0.26f, 0.22f}, 0.00022f, {1.00f, 0.84f, 0.62f}, 0.00f,
     0.75f, 0.30f, 0.09f, 0.08f, {0.56f, 0.38f, 0.22f}, 0.09f, 0.13f, 32.0f, 0.34f, 0.36f, 0.05f},
    // DarkCave (Swamp, Desert Area 10, Old Jungle, Great Forest 7/8)
    {{0.68f, 0.72f, 0.78f}, 0.72f, {0.24f, 0.28f, 0.34f}, 0.00032f, {0.90f, 0.84f, 0.72f}, 0.00f,
     0.74f, 0.32f, 0.09f, 0.08f, {0.34f, 0.34f, 0.38f}, 0.07f, 0.16f, 38.0f, 0.36f, 0.38f, 0.05f},
    // Desert (Desert & Old Desert)
    {{0.82f, 0.76f, 0.70f}, 0.56f, {0.92f, 0.83f, 0.66f}, 0.00034f, {1.00f, 0.90f, 0.68f}, 0.26f,
     0.82f, 0.24f, 0.16f, 0.17f, {0.88f, 0.72f, 0.48f}, 0.12f, 0.13f, 26.0f, 0.58f, 0.20f, 0.07f},
    // DesertSandstorm (Desert Area 2)
    {{0.86f, 0.78f, 0.64f}, 0.52f, {0.88f, 0.76f, 0.56f}, 0.00072f, {1.00f, 0.86f, 0.62f}, 0.30f,
     0.82f, 0.22f, 0.13f, 0.11f, {0.86f, 0.72f, 0.50f}, 0.12f, 0.09f, 20.0f, 0.34f, 0.18f, 0.04f},
    // ForestHills
    {{0.72f, 0.80f, 0.86f}, 0.60f, {0.74f, 0.84f, 0.88f}, 0.00034f, {1.00f, 0.96f, 0.80f}, 0.26f,
     0.82f, 0.24f, 0.15f, 0.16f, {0.44f, 0.62f, 0.34f}, 0.10f, 0.13f, 30.0f, 0.56f, 0.20f, 0.08f},
    // GreatForest
    {{0.68f, 0.80f, 0.72f}, 0.66f, {0.66f, 0.78f, 0.68f}, 0.00054f, {0.94f, 1.00f, 0.76f}, 0.36f,
     0.80f, 0.26f, 0.16f, 0.15f, {0.34f, 0.56f, 0.32f}, 0.11f, 0.17f, 40.0f, 0.54f, 0.24f, 0.09f},
    // Jungle (Jungle & Old Jungle)
    {{0.70f, 0.82f, 0.78f}, 0.64f, {0.70f, 0.82f, 0.78f}, 0.00046f, {0.96f, 0.98f, 0.82f}, 0.30f,
     0.80f, 0.25f, 0.15f, 0.15f, {0.38f, 0.56f, 0.38f}, 0.10f, 0.20f, 44.0f, 0.54f, 0.22f, 0.08f},
    // Swamp (Swamp & Old Swamp)
    {{0.70f, 0.76f, 0.88f}, 0.66f, {0.60f, 0.68f, 0.78f}, 0.00066f, {0.80f, 0.88f, 0.96f}, 0.24f,
     0.78f, 0.26f, 0.14f, 0.12f, {0.30f, 0.40f, 0.48f}, 0.09f, 0.21f, 46.0f, 0.46f, 0.26f, 0.05f},
    // Volcano (Volcano, Old Volcano & Battleground)
    {{0.80f, 0.66f, 0.62f}, 0.64f, {0.60f, 0.34f, 0.22f}, 0.00048f, {1.00f, 0.60f, 0.28f}, 0.28f,
     0.70f, 0.38f, 0.19f, 0.14f, {1.00f, 0.36f, 0.10f}, 0.18f, 0.16f, 30.0f, 0.48f, 0.40f, 0.08f},
    // Tower (Ancient Tower 1/2/3)
    {{0.78f, 0.76f, 0.86f}, 0.64f, {0.88f, 0.86f, 0.82f}, 0.00044f, {1.00f, 0.92f, 0.72f}, 0.36f,
     0.78f, 0.28f, 0.18f, 0.16f, {0.78f, 0.70f, 0.56f}, 0.11f, 0.15f, 32.0f, 0.56f, 0.24f, 0.07f},
    // FortressSiege (Castle Schrade, Fortress & Town)
    {{0.68f, 0.72f, 0.80f}, 0.68f, {0.58f, 0.62f, 0.68f}, 0.00050f, {0.94f, 0.88f, 0.80f}, 0.24f,
     0.76f, 0.28f, 0.17f, 0.14f, {0.44f, 0.42f, 0.40f}, 0.09f, 0.16f, 34.0f, 0.52f, 0.28f, 0.05f},
    // Arena (Arena & Great Arena)
    {{0.80f, 0.78f, 0.76f}, 0.56f, {0.86f, 0.84f, 0.78f}, 0.00018f, {1.00f, 0.94f, 0.80f}, 0.16f,
     0.82f, 0.22f, 0.15f, 0.15f, {0.78f, 0.70f, 0.56f}, 0.10f, 0.14f, 30.0f, 0.56f, 0.18f, 0.06f},
};
static_assert(std::size(kBiomePresets) == static_cast<std::size_t>(BiomeKind::Arena) + 1u);

enum PassSlot : std::size_t {
    kPassGtao = 0,
    kPassVol = 1,
    kPassBloom = 2,
    kPassSsgi = 3,
    kPassGtaoDenoise = 4,
    kPassBloomBlur = 5,
    kPassSsgiBlur = 6,
    kPassFlash = 7,
    kPassComposite = 8,
    kPassFxaa = 9,
};

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

    auto make_pipe_spec = [&](const auto &spv, const VkSpecializationInfo *spec) -> VkPipeline {
        VkShaderModule mod = ctx.create_shader_module(spv);
        if (!mod) return VK_NULL_HANDLE;
        VkComputePipelineCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, mod, "main", spec};
        ci.layout = pipe_layout_;
        VkPipeline pipe = VK_NULL_HANDLE;
        vkCreateComputePipelines(ctx.device, ctx.pipeline_cache, 1, &ci, nullptr, &pipe);
        vkDestroyShaderModule(ctx.device, mod, nullptr);
        return pipe;
    };
    auto make_pipe = [&](const auto &spv) -> VkPipeline { return make_pipe_spec(spv, nullptr); };

    gtao_pipe_ = make_pipe(shaders::kGtaoCompSpv);
    gtao_denoise_pipe_ = make_pipe(shaders::kGtaoDenoiseCompSpv);
    vol_pipe_ = make_pipe(shaders::kVolumetricFogCompSpv);
    bloom_pipe_ = make_pipe(shaders::kBloomExtractCompSpv);
    ssgi_pipe_ = make_pipe(shaders::kSsgiBounceCompSpv);
    glow_blur_pipe_ = make_pipe(shaders::kGlowBlurCompSpv);
    flash_pipe_ = make_pipe(shaders::kEmissiveFlashCompSpv);
    comp_pipe_ = make_pipe(shaders::kLightingCompositeCompSpv);
    const std::uint32_t inplace_val = 1u;
    const VkSpecializationMapEntry inplace_entry{0, 0, sizeof(std::uint32_t)};
    const VkSpecializationInfo inplace_spec{1, &inplace_entry, sizeof(inplace_val), &inplace_val};
    comp_inplace_pipe_ = make_pipe_spec(shaders::kLightingCompositeCompSpv, &inplace_spec);
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
            if (i < live->shadow_zone_biomes.size()) {
                const auto [biome_id, cave] = live->shadow_zone_biomes[i];
                zones_[i].biome = biome_id <= static_cast<std::uint8_t>(BiomeKind::Arena)
                                      ? static_cast<BiomeKind>(biome_id)
                                      : BiomeKind::DefaultOutdoor;
                zones_[i].is_cave = cave;
            } else {
                const bool is_cave = caves.count(zname) > 0;
                zones_[i].biome = is_cave ? BiomeKind::DarkCave : BiomeKind::DefaultOutdoor;
                zones_[i].is_cave = is_cave;
            }
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
    kill_pipe(comp_inplace_pipe_);
    kill_pipe(fxaa_pipe_);
    scene_set_views_.fill(VK_NULL_HANDLE);
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

    auto needs_recreate = [](const Target2D &t, std::uint32_t w, std::uint32_t h) {
        return t.image == VK_NULL_HANDLE || t.w != w || t.h != h;
    };
    const bool any_recreate =
        needs_recreate(gtao_tex_, hw, hh) || needs_recreate(gtao_blur_tex_, hw, hh) ||
        needs_recreate(vol_tex_, qw, qh) || needs_recreate(bloom_tex_, qw, qh) ||
        needs_recreate(bloom_blur_tex_, qw, qh) || needs_recreate(ssgi_tex_, qw, qh) ||
        needs_recreate(ssgi_blur_tex_, qw, qh) || needs_recreate(flash_tex_, 4u, 2u) ||
        needs_recreate(lit_tmp_tex_, full_w, full_h);
    if (!any_recreate) return true;

    ctx.wait_idle();
    scene_set_views_.fill(VK_NULL_HANDLE);

    auto make_target = [&](Target2D &t, std::uint32_t w, std::uint32_t h, VkFormat fmt, VkDescriptorSet dset) -> bool {
        if (t.image != VK_NULL_HANDLE && t.w == w && t.h == h) return true;
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

    // Load biome-tuned lighting, AO, contact shadows, hemisphere bounce, specular, SSGI, fog & bloom preset.
    const std::size_t bidx = std::min(static_cast<std::size_t>(active_biome_), std::size(kBiomePresets) - 1u);
    BiomePreset m = kBiomePresets[bidx];
    if (active_biome_ == BiomeKind::Volcano && active_is_cave_) {
        m.scat_str = 0.0f;
        m.contact_str = 0.38f;
    }

    // Modulate contact shadow & AO contrast gently using MHFU's hand-tuned per-zone sun darkness (when present).
    if (active_zone_ >= 0 && static_cast<std::size_t>(active_zone_) < zones_.size()) {
        const float zd = zones_[static_cast<std::size_t>(active_zone_)].sun_darkness;
        if (zd > 0.05f) {
            const float scale = std::clamp(0.65f + zd * 0.85f, 0.75f, 1.20f);
            m.contact_str = std::clamp(m.contact_str * scale, 0.0f, 0.72f);
            m.ao_str = std::clamp(m.ao_str * (0.88f + zd * 0.28f), 0.45f, 0.78f);
        }
    }

    // Blend with the game's live PSP kFogColor so day vs. night quests adapt automatically.
    if (in.has_psp_fog) {
        const float psp_luma = in.psp_fog_rgb[0] * 0.2126f + in.psp_fog_rgb[1] * 0.7152f + in.psp_fog_rgb[2] * 0.0722f;
        for (int c = 0; c < 3; ++c) {
            m.fog_rgb[c] = m.fog_rgb[c] * 0.45f + in.psp_fog_rgb[c] * 0.55f;
        }
        // Dark night sky/fog: tone down sun shafts gently and cool down scatter into moonlight.
        if (psp_luma < 0.22f && active_biome_ != BiomeKind::Volcano && !active_is_cave_) {
            m.scat_str *= 0.55f;
            m.scat_rgb[0] = 0.72f;
            m.scat_rgb[1] = 0.82f;
            m.scat_rgb[2] = 1.00f;
            m.contact_str *= 0.75f;
            m.ssgi_str = std::max(m.ssgi_str, 0.30f);
        }
    }

    // Smooth temporal mood blending (~0.3s transition on outdoor <-> cave / biome changes,
    // advancing once per 30 Hz game frame so all 120/180 Hz subframes stay 100% identical).
    const float tgt_shadow[4]{m.shadow_rgb[0], m.shadow_rgb[1], m.shadow_rgb[2], m.ao_str};
    const float tgt_fog[4]{m.fog_rgb[0], m.fog_rgb[1], m.fog_rgb[2], m.fog_dens};
    const float tgt_scat[4]{m.scat_rgb[0], m.scat_rgb[1], m.scat_rgb[2], m.scat_str};
    const float tgt_bloom[4]{m.bloom_th, m.bloom_str, m.rim_str, m.key_str};
    const float tgt_ground[4]{m.ground_rgb[0], m.ground_rgb[1], m.ground_rgb[2], m.hemi_str};
    const float tgt_spec[4]{m.spec_str, m.spec_gloss, m.contact_str, m.vibrance};

    if (!mood_.valid) {
        std::memcpy(mood_.shadow_tint, tgt_shadow, sizeof(tgt_shadow));
        std::memcpy(mood_.fog_color, tgt_fog, sizeof(tgt_fog));
        std::memcpy(mood_.scatter_color, tgt_scat, sizeof(tgt_scat));
        std::memcpy(mood_.bloom_rim, tgt_bloom, sizeof(tgt_bloom));
        std::memcpy(mood_.ground_bounce, tgt_ground, sizeof(tgt_ground));
        std::memcpy(mood_.spec_contact, tgt_spec, sizeof(tgt_spec));
        mood_.ssgi_str = m.ssgi_str;
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
        mood_.ssgi_str += (m.ssgi_str - mood_.ssgi_str) * kAlpha;
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
}

void LightingPipeline::dispatch(VkCommandBuffer cmd, const Context &ctx, BindlessDescriptors &desc,
                                VkImage scene_color, VkImageView scene_color_view, std::uint32_t scene_color_slot,
                                std::uint32_t scene_depth_slot, std::uint32_t scene_w, std::uint32_t scene_h,
                                const LightingFrameInput &in, void *params_cpu_ptr, VkDeviceAddress params_bda) {
    if (!ready() || !params_cpu_ptr || params_bda == 0 || scene_depth_slot == 0 || scene_color_slot == 0) return;
    if (!ensure_targets(ctx, desc, scene_w, scene_h)) return;

    // Cache VkImageView -> VkDescriptorSet so steady-state subframes do zero vkUpdateDescriptorSets calls,
    // while cycling through kSceneSetCount slots whenever a new render target view appears.
    VkDescriptorSet scene_set = VK_NULL_HANDLE;
    for (std::size_t i = 0; i < kSceneSetCount; ++i) {
        if (scene_set_views_[i] == scene_color_view) {
            scene_set = scene_sets_[i];
            break;
        }
    }
    if (scene_set == VK_NULL_HANDLE) {
        const std::size_t slot = scene_set_cursor_;
        scene_set_cursor_ = (scene_set_cursor_ + 1u) % kSceneSetCount;
        scene_set = scene_sets_[slot];
        scene_set_views_[slot] = scene_color_view;
        VkDescriptorImageInfo sii{VK_NULL_HANDLE, scene_color_view, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet swds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, scene_set, 0, 0, 1,
                                  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &sii, nullptr, nullptr};
        vkUpdateDescriptorSets(ctx.device, 1, &swds, 0, nullptr);
    }

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
    slots[kPassGtao]        = base; slots[kPassGtao].aux_slots[2]        = gtao_tex_.w;       slots[kPassGtao].aux_slots[3]        = gtao_tex_.h;
    slots[kPassVol]         = base; slots[kPassVol].aux_slots[2]         = vol_tex_.w;        slots[kPassVol].aux_slots[3]         = vol_tex_.h;
    slots[kPassBloom]       = base; slots[kPassBloom].aux_slots[2]       = bloom_tex_.w;      slots[kPassBloom].aux_slots[3]       = bloom_tex_.h;
    slots[kPassSsgi]        = base; slots[kPassSsgi].aux_slots[2]        = ssgi_tex_.w;       slots[kPassSsgi].aux_slots[3]        = ssgi_tex_.h;
    slots[kPassGtaoDenoise] = base; slots[kPassGtaoDenoise].aux_slots[2] = gtao_blur_tex_.w;  slots[kPassGtaoDenoise].aux_slots[3] = gtao_blur_tex_.h;
    slots[kPassBloomBlur]   = base; slots[kPassBloomBlur].aux_slots[0]   = bloom_tex_.slot;   slots[kPassBloomBlur].aux_slots[2]   = bloom_blur_tex_.w; slots[kPassBloomBlur].aux_slots[3] = bloom_blur_tex_.h;
    slots[kPassSsgiBlur]    = base; slots[kPassSsgiBlur].aux_slots[0]    = ssgi_tex_.slot;    slots[kPassSsgiBlur].aux_slots[2]    = ssgi_blur_tex_.w;  slots[kPassSsgiBlur].aux_slots[3]  = ssgi_blur_tex_.h;
    slots[kPassFlash]       = base; slots[kPassFlash].aux_slots[0]       = bloom_tex_.slot;   slots[kPassFlash].aux_slots[2]       = 4u;                slots[kPassFlash].aux_slots[3]     = 2u;
    slots[kPassComposite]   = base; slots[kPassComposite].ext_slots[1]   = want_ssgi ? flash_tex_.slot : 0u; slots[kPassComposite].aux_slots[2] = scene_w; slots[kPassComposite].aux_slots[3] = scene_h;
    slots[kPassFxaa]        = base; slots[kPassFxaa].aux_slots[0]        = lit_tmp_tex_.slot; slots[kPassFxaa].aux_slots[2]        = scene_w;           slots[kPassFxaa].aux_slots[3]      = scene_h;

    // Batch image layout transitions into one vkCmdPipelineBarrier2 per stage boundary.
    std::array<VkImageMemoryBarrier2, 10> barriers{};
    std::uint32_t barrier_count = 0;
    auto push_barrier = [&](VkImage img, VkImageLayout old_l, VkImageLayout new_l,
                            VkAccessFlags2 src_access, VkAccessFlags2 dst_access) {
        VkImageMemoryBarrier2 &b = barriers[barrier_count++];
        b = {};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.srcAccessMask = src_access;
        b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.dstAccessMask = dst_access;
        b.oldLayout = old_l;
        b.newLayout = new_l;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = img;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    };
    auto queue_to_general = [&](VkImage img) {
        push_barrier(img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                     VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                     VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    };
    auto queue_to_read = [&](VkImage img) {
        push_barrier(img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    };
    auto flush_barriers = [&]() {
        if (barrier_count == 0) return;
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = barrier_count;
        dep.pImageMemoryBarriers = barriers.data();
        vkCmdPipelineBarrier2(cmd, &dep);
        barrier_count = 0;
    };

    if (need_gtao_pass) {
        queue_to_general(gtao_tex_.image);
        queue_to_general(gtao_blur_tex_.image);
    }
    if (want_vol)       queue_to_general(vol_tex_.image);
    if (need_bloom_src) queue_to_general(bloom_tex_.image);
    if (want_bloom)     queue_to_general(bloom_blur_tex_.image);
    if (want_ssgi) {
        queue_to_general(ssgi_tex_.image);
        queue_to_general(ssgi_blur_tex_.image);
        queue_to_general(flash_tex_.image);
    }
    if (want_fxaa) queue_to_general(lit_tmp_tex_.image);
    flush_barriers();

    // Bind global bindless Set 0 once; individual passes only rebind Set 1.
    const VkDescriptorSet bindless_set = desc.set();
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_, 0, 1, &bindless_set, 0, nullptr);

    // Stage 1: Enabled half/quarter-res passes read scene_depth/scene_color in parallel across GPU CUs.
    auto run_pass = [&](VkPipeline pipe, VkDescriptorSet out_set, std::uint32_t w, std::uint32_t h, std::size_t idx) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_, 1, 1, &out_set, 0, nullptr);
        const VkDeviceAddress addr = params_bda + idx * sizeof(LightingComputeParams);
        vkCmdPushConstants(cmd, pipe_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(addr), &addr);
        vkCmdDispatch(cmd, (w + 7u) / 8u, (h + 7u) / 8u, 1u);
    };

    if (need_gtao_pass) run_pass(gtao_pipe_, gtao_set_, gtao_tex_.w, gtao_tex_.h, kPassGtao);
    if (want_vol)       run_pass(vol_pipe_, vol_set_, vol_tex_.w, vol_tex_.h, kPassVol);
    if (need_bloom_src) run_pass(bloom_pipe_, bloom_set_, bloom_tex_.w, bloom_tex_.h, kPassBloom);
    if (want_ssgi)      run_pass(ssgi_pipe_, ssgi_set_, ssgi_tex_.w, ssgi_tex_.h, kPassSsgi);

    // Stage 2: Bilateral GTAO/ContactShadow denoise + Kawase Bloom/SSGI blur + 4x2 dynamic flash.
    if (need_gtao_pass) queue_to_read(gtao_tex_.image);
    if (need_bloom_src) queue_to_read(bloom_tex_.image);
    if (want_ssgi)      queue_to_read(ssgi_tex_.image);
    flush_barriers();
    if (need_gtao_pass) run_pass(gtao_denoise_pipe_, gtao_blur_set_, gtao_blur_tex_.w, gtao_blur_tex_.h, kPassGtaoDenoise);
    if (want_bloom)     run_pass(glow_blur_pipe_, bloom_blur_set_, bloom_blur_tex_.w, bloom_blur_tex_.h, kPassBloomBlur);
    if (want_ssgi) {
        run_pass(glow_blur_pipe_, ssgi_blur_set_, ssgi_blur_tex_.w, ssgi_blur_tex_.h, kPassSsgiBlur);
        run_pass(flash_pipe_, flash_set_, 4u, 2u, kPassFlash);
    }

    // Barrier batch: transition produced targets -> SHADER_READ_ONLY.
    if (need_gtao_pass) queue_to_read(gtao_blur_tex_.image);
    if (want_vol)       queue_to_read(vol_tex_.image);
    if (want_bloom)     queue_to_read(bloom_blur_tex_.image);
    if (want_ssgi) {
        queue_to_read(ssgi_blur_tex_.image);
        queue_to_read(flash_tex_.image);
    }

    if (want_fxaa) {
        flush_barriers();
        // Stage 3: Unified full-res composite reading scene_color -> writing lit_tmp_tex_.
        run_pass(comp_pipe_, lit_tmp_set_, scene_w, scene_h, kPassComposite);

        // Stage 4: Pre-HUD Edge Anti-Aliasing (FXAA 3.11) -> scene_color.
        queue_to_read(lit_tmp_tex_.image);
        push_barrier(scene_color, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                     VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        flush_barriers();
        run_pass(fxaa_pipe_, scene_set, scene_w, scene_h, kPassFxaa);
    } else {
        // FXAA off: composite in place directly on scene_color, skipping lit_tmp_tex_ and fxaa_prehud.comp!
        push_barrier(scene_color, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                     VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                     VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        flush_barriers();
        run_pass(comp_inplace_pipe_, scene_set, scene_w, scene_h, kPassComposite);
    }
}

} // namespace mhf::psp::vk
