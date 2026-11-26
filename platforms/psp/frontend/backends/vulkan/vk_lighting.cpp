#include "vk_lighting.hpp"
#include "vk_pipelines.hpp"
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
    // Swamp (Swamp & Old Swamp) — subtle distant mist without double-fogging over PSP's native Swamp fog.
    {{0.68f, 0.74f, 0.84f}, 0.66f, {0.42f, 0.50f, 0.58f}, 0.00020f, {0.78f, 0.86f, 0.94f}, 0.16f,
     0.80f, 0.22f, 0.10f, 0.12f, {0.28f, 0.36f, 0.44f}, 0.08f, 0.14f, 46.0f, 0.46f, 0.26f, 0.05f},
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
    kPassExposure = 10,
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

    constexpr std::uint32_t kTotalSets = 17u + static_cast<std::uint32_t>(kSceneSetCount);
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
    exposure_set_ = sets[9];
    taa_hist_sets_[0] = sets[10];
    taa_hist_sets_[1] = sets[11];
    hiz_sets_[0] = sets[12];
    hiz_sets_[1] = sets[13];
    hiz_sets_[2] = sets[14];
    hiz_sets_[3] = sets[15];
    water_ssr_set_ = sets[16];
    for (std::size_t i = 0; i < kSceneSetCount; ++i) {
        scene_sets_[i] = sets[17 + i];
    }
    scene_set_cursor_ = 0;

    std::array<VkDescriptorSetLayout, 2> pipe_sets = {bindless_layout, storage_layout_};
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkDeviceAddress)};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 2, pipe_sets.data(), 1, &pcr};
    if (vkCreatePipelineLayout(ctx.device, &plci, nullptr, &pipe_layout_) != VK_SUCCESS) return false;

    // Initialize True 3D Froxel Volumetric Fog (160x90x64 RGBA16F) and Set 3 (R8_UINT MaterialClass).
    if (!froxel_.init(ctx, bindless_layout)) return false;

    {
        VkDescriptorSetLayoutBinding mb{};
        mb.binding = 0;
        mb.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        mb.descriptorCount = 1;
        mb.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo mdsl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 1, &mb};
        if (vkCreateDescriptorSetLayout(ctx.device, &mdsl, nullptr, &mat_set_layout_) != VK_SUCCESS) return false;

        VkDescriptorPoolSize mps{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, static_cast<std::uint32_t>(kMatSetCount)};
        VkDescriptorPoolCreateInfo mdpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0,
                                         static_cast<std::uint32_t>(kMatSetCount), 1, &mps};
        if (vkCreateDescriptorPool(ctx.device, &mdpci, nullptr, &mat_pool_) != VK_SUCCESS) return false;

        std::array<VkDescriptorSetLayout, kMatSetCount> mdsls{};
        mdsls.fill(mat_set_layout_);
        VkDescriptorSetAllocateInfo mdsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, mat_pool_,
                                          static_cast<std::uint32_t>(kMatSetCount), mdsls.data()};
        if (vkAllocateDescriptorSets(ctx.device, &mdsai, mat_sets_.data()) != VK_SUCCESS) return false;
        mat_set_views_.fill(VK_NULL_HANDLE);
        mat_set_cursor_ = 0;

        // 1x1 R8_UINT fallback texture (cleared to MaterialClass::Unknown = 0) when HDR/MRT is inactive.
        VkImageCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        fci.imageType = VK_IMAGE_TYPE_2D;
        fci.format = VK_FORMAT_R8_UINT;
        fci.extent = {1u, 1u, 1u};
        fci.mipLevels = 1u;
        fci.arrayLayers = 1u;
        fci.samples = VK_SAMPLE_COUNT_1_BIT;
        fci.tiling = VK_IMAGE_TILING_OPTIMAL;
        fci.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        VmaAllocationCreateInfo fvaci{};
        fvaci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        if (vmaCreateImage(ctx.allocator, &fci, &fvaci, &fallback_mat_tex_.image, &fallback_mat_tex_.alloc, nullptr) != VK_SUCCESS) {
            return false;
        }
        VkImageViewCreateInfo fvci{};
        fvci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        fvci.image = fallback_mat_tex_.image;
        fvci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        fvci.format = VK_FORMAT_R8_UINT;
        fvci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(ctx.device, &fvci, nullptr, &fallback_mat_tex_.view) != VK_SUCCESS) return false;
        fallback_mat_tex_.w = 1u;
        fallback_mat_tex_.h = 1u;
        fallback_mat_cleared_ = false;
    }

    std::array<VkDescriptorSetLayout, 4> comp_sets = {bindless_layout, storage_layout_, froxel_.sample_set_layout(), mat_set_layout_};
    VkPipelineLayoutCreateInfo cplci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 4, comp_sets.data(), 1, &pcr};
    if (vkCreatePipelineLayout(ctx.device, &cplci, nullptr, &comp_pipe_layout_) != VK_SUCCESS) return false;

    auto make_pipe_layout = [&](const auto &spv, VkPipelineLayout layout, const VkSpecializationInfo *spec) -> VkPipeline {
        VkShaderModule mod = ctx.create_shader_module(spv);
        if (!mod) return VK_NULL_HANDLE;
        VkComputePipelineCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, mod, "main", spec};
        ci.layout = layout;
        VkPipeline pipe = VK_NULL_HANDLE;
        vkCreateComputePipelines(ctx.device, ctx.pipeline_cache, 1, &ci, nullptr, &pipe);
        vkDestroyShaderModule(ctx.device, mod, nullptr);
        return pipe;
    };
    auto make_pipe = [&](const auto &spv) -> VkPipeline { return make_pipe_layout(spv, pipe_layout_, nullptr); };

    gtao_pipe_ = make_pipe(shaders::kGtaoCompSpv);
    gtao_denoise_pipe_ = make_pipe(shaders::kGtaoDenoiseCompSpv);
    vol_pipe_ = make_pipe(shaders::kVolumetricFogCompSpv);
    bloom_pipe_ = make_pipe(shaders::kBloomExtractCompSpv);
    ssgi_pipe_ = make_pipe(shaders::kSsgiBounceCompSpv);
    glow_blur_pipe_ = make_pipe(shaders::kGlowBlurCompSpv);
    flash_pipe_ = make_pipe(shaders::kEmissiveFlashCompSpv);
    comp_pipe_ = make_pipe_layout(shaders::kLightingCompositeCompSpv, comp_pipe_layout_, nullptr);
    const std::uint32_t inplace_val = 1u;
    const VkSpecializationMapEntry inplace_entry{0, 0, sizeof(std::uint32_t)};
    const VkSpecializationInfo inplace_spec{1, &inplace_entry, sizeof(inplace_val), &inplace_val};
    comp_inplace_pipe_ = make_pipe_layout(shaders::kLightingCompositeCompSpv, comp_pipe_layout_, &inplace_spec);
    exposure_pipe_ = make_pipe(shaders::kExposureCompSpv);
    fxaa_pipe_ = make_pipe(shaders::kFxaaPrehudCompSpv);
    taa_pipe_ = make_pipe(shaders::kTaaResolveCompSpv);
    hiz_pipe_ = make_pipe(shaders::kHiZGenCompSpv);
    water_ssr_pipe_ = make_pipe_layout(shaders::kWaterSsrCompSpv, comp_pipe_layout_, nullptr);

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
            if (i < live->shadow_zone_mist_profiles.size()) {
                zones_[i].mist = live->shadow_zone_mist_profiles[i];
            }
        }
        tex_to_zone_.reserve(live->shadow_zone_textures.size());
        for (const auto &[h32, zidx] : live->shadow_zone_textures) {
            if (zidx >= 0 && static_cast<std::size_t>(zidx) < zones_.size()) {
                tex_to_zone_[h32] = zidx;
            }
        }
        tex_to_mat_.reserve(live->material_registry.size());
        for (const auto &e : live->material_registry) {
            if (e.mat == MaterialClass::Unknown) continue;
            tex_to_mat_[MatKey{e.data_hash, e.clut_hash}].push_back(MatEntry{e.zone_hint, e.mat});
        }
        const char *dump_env = std::getenv("MHF_VK_DUMP_MAT_KEYS");
        const bool want_dump = (dump_env != nullptr && dump_env[0] == '1');
        if (!tex_to_zone_.empty() || !tex_to_mat_.empty() || want_dump) {
            zone_keys_ = textures::Pack::keys_only();
        }
    }

    live_ = live;
    return ready();
}

void LightingPipeline::shutdown(const Context &ctx, BindlessDescriptors &desc) {
    gtao_tex_.destroy(ctx, desc);
    gtao_blur_tex_.destroy(ctx, desc);
    for (auto &ht : hiz_tex_) ht.destroy(ctx, desc);
    water_ssr_tex_.destroy(ctx, desc);
    vol_tex_.destroy(ctx, desc);
    bloom_tex_.destroy(ctx, desc);
    bloom_blur_tex_.destroy(ctx, desc);
    ssgi_tex_.destroy(ctx, desc);
    ssgi_blur_tex_.destroy(ctx, desc);
    flash_tex_.destroy(ctx, desc);
    lit_tmp_tex_.destroy(ctx, desc);
    exposure_tex_.destroy(ctx, desc);
    noise3d_tex_.destroy(ctx, desc);
    if (noise3d_stage_buf_ != VK_NULL_HANDLE) {
        vmaDestroyBuffer(ctx.allocator, noise3d_stage_buf_, noise3d_stage_alloc_);
        noise3d_stage_buf_ = VK_NULL_HANDLE;
        noise3d_stage_alloc_ = VK_NULL_HANDLE;
    }
    exposure_initialized_ = false;
    exposure_reset_ = true;
    taa_hist_[0].destroy(ctx, desc);
    taa_hist_[1].destroy(ctx, desc);
    taa_hist_valid_ = false;
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
    kill_pipe(exposure_pipe_);
    kill_pipe(fxaa_pipe_);
    kill_pipe(taa_pipe_);
    kill_pipe(hiz_pipe_);
    kill_pipe(water_ssr_pipe_);
    scene_set_views_.fill(VK_NULL_HANDLE);
    if (comp_pipe_layout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(ctx.device, comp_pipe_layout_, nullptr);
        comp_pipe_layout_ = VK_NULL_HANDLE;
    }
    froxel_.shutdown(ctx);
    if (pipe_layout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(ctx.device, pipe_layout_, nullptr);
        pipe_layout_ = VK_NULL_HANDLE;
    }
    if (storage_pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(ctx.device, storage_pool_, nullptr);
        storage_pool_ = VK_NULL_HANDLE;
    }
    fallback_mat_tex_.destroy(ctx, desc);
    fallback_mat_cleared_ = false;
    mat_set_views_.fill(VK_NULL_HANDLE);
    if (mat_pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(ctx.device, mat_pool_, nullptr);
        mat_pool_ = VK_NULL_HANDLE;
    }
    if (mat_set_layout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(ctx.device, mat_set_layout_, nullptr);
        mat_set_layout_ = VK_NULL_HANDLE;
    }
    if (storage_layout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(ctx.device, storage_layout_, nullptr);
        storage_layout_ = VK_NULL_HANDLE;
    }
}

void LightingPipeline::ensure_fallback_mat(VkCommandBuffer cmd) {
    if (fallback_mat_cleared_ || fallback_mat_tex_.image == VK_NULL_HANDLE) return;
    VkImageSubresourceRange rng{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    Context::cmd_transition_image(cmd, fallback_mat_tex_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                                  VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                                  VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkClearColorValue ccv{};
    ccv.uint32[0] = 0u;
    vkCmdClearColorImage(cmd, fallback_mat_tex_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &ccv, 1, &rng);
    Context::cmd_transition_image(cmd, fallback_mat_tex_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    fallback_mat_cleared_ = true;
}

VkDescriptorSet LightingPipeline::get_or_bind_mat_set(const Context &ctx, VkImageView mat_id_view) {
    VkImageView v = (mat_id_view != VK_NULL_HANDLE) ? mat_id_view : fallback_mat_tex_.view;
    for (std::size_t i = 0; i < kMatSetCount; ++i) {
        if (mat_set_views_[i] == v) return mat_sets_[i];
    }
    const std::size_t idx = mat_set_cursor_;
    mat_set_cursor_ = (mat_set_cursor_ + 1u) % kMatSetCount;
    mat_set_views_[idx] = v;
    VkDescriptorImageInfo ii{VK_NULL_HANDLE, v, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mat_sets_[idx], 0, 0, 1,
                           VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &ii, nullptr, nullptr};
    vkUpdateDescriptorSets(ctx.device, 1, &w, 0, nullptr);
    return mat_sets_[idx];
}

MaterialClass LightingPipeline::note_texture(std::uint64_t content_hash, const textures::TextureFacts &facts,
                                             const textures::MemoryReader &reader, bool is_blended,
                                             std::uint32_t *out_data_hash, std::uint32_t *out_clut_hash) {
    if (!zone_keys_) return MaterialClass::Unknown;
    CachedTexSem sem{};
    if (const auto cit = content_hash_sem_.find(content_hash); cit != content_hash_sem_.end()) {
        sem = cit->second;
    } else {
        textures::Key key{};
        std::uint32_t cw = facts.width, ch = facts.height;
        if (zone_keys_->key_of(facts, reader, key, cw, ch)) {
            sem.has_key = true;
            sem.data_hash = key.data;
            sem.clut_hash = static_cast<std::uint32_t>(key.cache & 0xFFFFFFFFull);
            if (const auto zit = tex_to_zone_.find(key.data); zit != tex_to_zone_.end()) {
                sem.zone = zit->second;
            }
        }
        if (content_hash_sem_.size() > 16384u) content_hash_sem_.clear();
        content_hash_sem_.emplace(content_hash, sem);
    }
    if (sem.has_key) {
        if (out_data_hash) *out_data_hash = sem.data_hash;
        if (out_clut_hash) *out_clut_hash = sem.clut_hash;
    }
    if (sem.zone >= 0 && static_cast<std::size_t>(sem.zone) < zone_votes_.size()) {
        ++zone_votes_[static_cast<std::size_t>(sem.zone)];
    }

    MaterialClass mat = MaterialClass::Unknown;
    if (sem.has_key && !tex_to_mat_.empty()) {
        auto resolve_bucket = [&](const std::vector<MatEntry> &entries) -> MaterialClass {
            MaterialClass fallback = MaterialClass::Unknown;
            for (const auto &e : entries) {
                if (e.zone_hint >= 0 && e.zone_hint == active_zone_) return e.mat;
                if (e.zone_hint < 0) fallback = e.mat;
            }
            return fallback;
        };
        if (sem.clut_hash != 0u) {
            if (const auto it = tex_to_mat_.find(MatKey{sem.data_hash, sem.clut_hash}); it != tex_to_mat_.end()) {
                mat = resolve_bucket(it->second);
            }
        }
        if (mat == MaterialClass::Unknown) {
            if (const auto it = tex_to_mat_.find(MatKey{sem.data_hash, 0u}); it != tex_to_mat_.end()) {
                mat = resolve_bucket(it->second);
            }
        }
    }

    static const bool s_dump_mat = [] {
        const char *v = std::getenv("MHF_VK_DUMP_MAT_KEYS");
        return v != nullptr && v[0] == '1';
    }();
    if (s_dump_mat && sem.has_key) {
        const std::uint64_t log_k = (static_cast<std::uint64_t>(sem.clut_hash) << 32u) | sem.data_hash;
        if (logged_mat_keys_.emplace(log_k, 1u).second) {
            const char *zname = (live_ && active_zone_ >= 0 && static_cast<std::size_t>(active_zone_) < live_->shadow_zone_names.size())
                                    ? live_->shadow_zone_names[active_zone_].c_str()
                                    : "unknown";
            std::fprintf(stderr,
                         "[vk][mat-key] zone=%d(%s) data=0x%08x clut=0x%08x %ux%u blended=%u mat=%u\n",
                         active_zone_, zname, sem.data_hash, sem.clut_hash,
                         static_cast<unsigned>(facts.width), static_cast<unsigned>(facts.height),
                         is_blended ? 1u : 0u, static_cast<unsigned>(mat));
        }
    }
    return mat;
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
        needs_recreate(lit_tmp_tex_, full_w, full_h) || needs_recreate(exposure_tex_, 1u, 1u);
    if (!any_recreate) return true;

    ctx.wait_idle();
    scene_set_views_.fill(VK_NULL_HANDLE);
    if (needs_recreate(exposure_tex_, 1u, 1u)) {
        exposure_initialized_ = false;
        exposure_reset_ = true;
    }

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
        ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
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

    const std::uint32_t ew = std::max(qw / 2u, 1u);
    const std::uint32_t eh = std::max(qh / 2u, 1u);
    const std::uint32_t sw = std::max(ew / 2u, 1u);
    const std::uint32_t sh = std::max(eh / 2u, 1u);

    return make_target(gtao_tex_, hw, hh, VK_FORMAT_R8G8B8A8_UNORM, gtao_set_) &&
           make_target(gtao_blur_tex_, hw, hh, VK_FORMAT_R8G8B8A8_UNORM, gtao_blur_set_) &&
           make_target(hiz_tex_[0], hw, hh, VK_FORMAT_R32_SFLOAT, hiz_sets_[0]) &&
           make_target(hiz_tex_[1], qw, qh, VK_FORMAT_R32_SFLOAT, hiz_sets_[1]) &&
           make_target(hiz_tex_[2], ew, eh, VK_FORMAT_R32_SFLOAT, hiz_sets_[2]) &&
           make_target(hiz_tex_[3], sw, sh, VK_FORMAT_R32_SFLOAT, hiz_sets_[3]) &&
           make_target(water_ssr_tex_, hw, hh, VK_FORMAT_R16G16B16A16_SFLOAT, water_ssr_set_) &&
           make_target(vol_tex_, qw, qh, VK_FORMAT_R16G16B16A16_SFLOAT, vol_set_) &&
           make_target(bloom_tex_, qw, qh, VK_FORMAT_R16G16B16A16_SFLOAT, bloom_set_) &&
           make_target(bloom_blur_tex_, qw, qh, VK_FORMAT_R16G16B16A16_SFLOAT, bloom_blur_set_) &&
           make_target(ssgi_tex_, qw, qh, VK_FORMAT_R16G16B16A16_SFLOAT, ssgi_set_) &&
           make_target(ssgi_blur_tex_, qw, qh, VK_FORMAT_R16G16B16A16_SFLOAT, ssgi_blur_set_) &&
           make_target(flash_tex_, 4u, 2u, VK_FORMAT_R16G16B16A16_SFLOAT, flash_set_) &&
           make_target(lit_tmp_tex_, full_w, full_h, VK_FORMAT_R8G8B8A8_UNORM, lit_tmp_set_) &&
           make_target(exposure_tex_, 1u, 1u, VK_FORMAT_R16_SFLOAT, exposure_set_);
}

bool LightingPipeline::ensure_taa_targets(const Context &ctx, BindlessDescriptors &desc,
                                          std::uint32_t full_w, std::uint32_t full_h) {
    if (taa_hist_[0].image != VK_NULL_HANDLE && taa_hist_[0].w == full_w && taa_hist_[0].h == full_h &&
        taa_hist_[1].image != VK_NULL_HANDLE && taa_hist_[1].w == full_w && taa_hist_[1].h == full_h) {
        return true;
    }
    ctx.wait_idle();
    taa_hist_valid_ = false;
    taa_hist_idx_ = 0;
    for (int i = 0; i < 2; ++i) {
        Target2D &t = taa_hist_[static_cast<std::size_t>(i)];
        t.destroy(ctx, desc);
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = VK_FORMAT_R8G8B8A8_UNORM;
        ici.extent = {full_w, full_h, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        VmaAllocationCreateInfo vaci{};
        vaci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        if (vmaCreateImage(ctx.allocator, &ici, &vaci, &t.image, &t.alloc, nullptr) != VK_SUCCESS) return false;

        VkImageViewCreateInfo ivci{};
        ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ivci.image = t.image;
        ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ivci.format = VK_FORMAT_R8G8B8A8_UNORM;
        ivci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(ctx.device, &ivci, nullptr, &t.view) != VK_SUCCESS) return false;

        t.w = full_w;
        t.h = full_h;
        t.slot = desc.allocate_texture(ctx, t.view);
        VkDescriptorImageInfo dii{VK_NULL_HANDLE, t.view, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet wds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, taa_hist_sets_[static_cast<std::size_t>(i)],
                                 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &dii, nullptr, nullptr};
        vkUpdateDescriptorSets(ctx.device, 1, &wds, 0, nullptr);
        if (t.slot == 0) return false;
    }
    return true;
}

VkDescriptorSet LightingPipeline::get_or_bind_scene_set(const Context &ctx, VkImageView scene_color_view) {
    for (std::size_t i = 0; i < kSceneSetCount; ++i) {
        if (scene_set_views_[i] == scene_color_view) {
            return scene_sets_[i];
        }
    }
    const std::size_t slot = scene_set_cursor_;
    scene_set_cursor_ = (scene_set_cursor_ + 1u) % kSceneSetCount;
    VkDescriptorSet scene_set = scene_sets_[slot];
    scene_set_views_[slot] = scene_color_view;
    VkDescriptorImageInfo sii{VK_NULL_HANDLE, scene_color_view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet swds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, scene_set, 0, 0, 1,
                              VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &sii, nullptr, nullptr};
    vkUpdateDescriptorSets(ctx.device, 1, &swds, 0, nullptr);
    return scene_set;
}

void LightingPipeline::fill_water_frame_params(WaterFrameParams &out, const LightingFrameInput &in,
                                               const float *view_mat, float vp_w, float vp_h, float elapsed_sec,
                                               std::uint32_t bg_color_slot, std::uint32_t bg_depth_slot) const noexcept {
    out = WaterFrameParams{};
    out.proj_info[0] = 1.0f / std::max(std::fabs(in.proj[0]), 1e-4f);
    out.proj_info[1] = 1.0f / std::max(std::fabs(in.proj[5]), 1e-4f);
    out.proj_info[2] = in.proj[10];
    out.proj_info[3] = in.proj[14];
    out.vz_and_size[0] = in.vz_scale;
    out.vz_and_size[1] = in.vz_center;
    out.vz_and_size[2] = std::max(vp_w, 1.0f);
    out.vz_and_size[3] = std::max(vp_h, 1.0f);

    // Prefer exact area-space camera from game host (world-locked across camera movement!), falling back to view_mat.
    const float *vm = in.fog_camera.valid ? in.fog_camera.world_to_eye.data() : view_mat;
    if (vm != nullptr) {
        out.inv_view[0]  = vm[0]; out.inv_view[1]  = vm[4]; out.inv_view[2]  = vm[8];  out.inv_view[3]  = 0.0f;
        out.inv_view[4]  = vm[1]; out.inv_view[5]  = vm[5]; out.inv_view[6]  = vm[9];  out.inv_view[7]  = 0.0f;
        out.inv_view[8]  = vm[2]; out.inv_view[9]  = vm[6]; out.inv_view[10] = vm[10]; out.inv_view[11] = 0.0f;
        const float tx = vm[12], ty = vm[13], tz = vm[14];
        out.inv_view[12] = -(vm[0] * tx + vm[1] * ty + vm[2] * tz);
        out.inv_view[13] = -(vm[4] * tx + vm[5] * ty + vm[6] * tz);
        out.inv_view[14] = -(vm[8] * tx + vm[9] * ty + vm[10] * tz);
        out.inv_view[15] = 1.0f;
        // World-up (0, 1, 0) in screen view space (+Y down).
        const float ux = vm[4], uy = -vm[5], uz = vm[6];
        const float ulen = std::sqrt(ux * ux + uy * uy + uz * uz);
        if (ulen > 1e-4f) {
            out.world_up_view[0] = ux / ulen;
            out.world_up_view[1] = uy / ulen;
            out.world_up_view[2] = uz / ulen;
        }
        out.world_up_view[3] = 1.0f;
    } else if (in.has_view_rot) {
        const float *r = in.view_rot;
        const float ux = r[1], uy = -r[4], uz = r[7];
        const float ulen = std::sqrt(ux * ux + uy * uy + uz * uz);
        if (ulen > 1e-4f) {
            out.world_up_view[0] = ux / ulen;
            out.world_up_view[1] = uy / ulen;
            out.world_up_view[2] = uz / ulen;
        }
        out.world_up_view[3] = 0.0f;
    }

    float sw[3]{0.34f, 0.78f, 0.52f};
    resolve_sun_dir_world(in, sw);
    if (vm != nullptr) {
        const float sx = vm[0] * sw[0] + vm[4] * sw[1] + vm[8] * sw[2];
        const float sy = -(vm[1] * sw[0] + vm[5] * sw[1] + vm[9] * sw[2]);
        const float sz = vm[2] * sw[0] + vm[6] * sw[1] + vm[10] * sw[2];
        const float sl = std::sqrt(sx * sx + sy * sy + sz * sz);
        if (sl > 1e-4f) {
            out.sun_dir_view[0] = sx / sl;
            out.sun_dir_view[1] = sy / sl;
            out.sun_dir_view[2] = sz / sl;
        }
    } else if (in.has_view_rot) {
        const float *r = in.view_rot;
        const float sx = r[0] * sw[0] + r[1] * sw[1] + r[2] * sw[2];
        const float sy = -(r[3] * sw[0] + r[4] * sw[1] + r[5] * sw[2]);
        const float sz = r[6] * sw[0] + r[7] * sw[1] + r[8] * sw[2];
        const float sl = std::sqrt(sx * sx + sy * sy + sz * sz);
        if (sl > 1e-4f) {
            out.sun_dir_view[0] = sx / sl;
            out.sun_dir_view[1] = sy / sl;
            out.sun_dir_view[2] = sz / sl;
        }
    }
    out.sun_dir_view[3] = active_is_cave_ ? 1.0f : 0.0f;

    // Area-aware / biome-calibrated linear SDR water optics (respecting MHFU's authentic per-map art style):
    //   sky_zenith.rgb  = subtle zenith sky reflection tint      | sky_zenith.w  = elapsed_sec
    //   sky_horizon.rgb = subtle horizon reflection tint         | sky_horizon.w = max_fresnel (restrained!)
    //   deep_col.rgb    = deep-water scatter accent (blended w/ orig) | deep_col.w = wave_chop (0.3..0.9)
    //   sigma_abs.rgb   = Beer-Lambert extinction (1/cm)         | sigma_abs.w   = caustic_foam_pack (caustic_str)
    auto set_biome = [&](float zr, float zg, float zb,
                         float hr, float hg, float hb, float max_fresnel,
                         float dr, float dg, float db, float wave_chop,
                         float sr, float sg, float sb, float caustic_str) {
        out.sky_zenith[0] = zr;  out.sky_zenith[1] = zg;  out.sky_zenith[2] = zb;
        out.sky_horizon[0] = hr; out.sky_horizon[1] = hg; out.sky_horizon[2] = hb;
        out.sky_horizon[3] = max_fresnel;
        out.deep_col[0] = dr;    out.deep_col[1] = dg;    out.deep_col[2] = db;
        out.deep_col[3] = wave_chop;
        out.sigma_abs[0] = sr;   out.sigma_abs[1] = sg;   out.sigma_abs[2] = sb;
        out.sigma_abs[3] = caustic_str;
    };
    switch (active_biome_) {
    case BiomeKind::Jungle:
        // Jungle: vibrant tropical emerald-turquoise ocean, lively coastal swell, crisp shallow sand caustics.
        set_biome(0.14f, 0.38f, 0.56f,  0.34f, 0.52f, 0.56f, 0.44f,
                  0.015f, 0.14f, 0.14f, 0.82f,
                  0.014f, 0.0045f, 0.0048f, 0.34f);
        break;
    case BiomeKind::Swamp:
        // Swamp: eerie dark murky marsh-olive water, calm still ripples, high turbidity, muted overcast reflection, zero caustics.
        set_biome(0.16f, 0.19f, 0.18f,  0.24f, 0.27f, 0.25f, 0.38f,
                  0.032f, 0.048f, 0.032f, 0.32f,
                  0.024f, 0.018f, 0.026f, 0.00f);
        break;
    case BiomeKind::SnowyPeak:
    case BiomeKind::PokkeVillage:
        // Pokke Farm / Pokke Village / Snowy Mtns: crisp alpine lake & mossy stream, glassy mountain/pier reflections.
        set_biome(0.18f, 0.34f, 0.52f,  0.40f, 0.48f, 0.56f, 0.48f,
                  0.025f, 0.095f, 0.125f, 0.52f,
                  0.014f, 0.0075f, 0.0060f, 0.20f);
        break;
    case BiomeKind::ForestHills:
    case BiomeKind::GreatForest:
        // Forest & Hills / Great Forest: clear woodland river, mossy-teal shadows, dappled canopy reflection.
        set_biome(0.16f, 0.32f, 0.44f,  0.32f, 0.44f, 0.46f, 0.44f,
                  0.020f, 0.095f, 0.090f, 0.62f,
                  0.014f, 0.0065f, 0.0070f, 0.24f);
        break;
    case BiomeKind::Desert:
    case BiomeKind::DesertSandstorm:
        // Desert Oasis: warm sunlit turquoise-amber shallows, bright sky reflection.
        set_biome(0.20f, 0.38f, 0.56f,  0.48f, 0.50f, 0.48f, 0.46f,
                  0.030f, 0.120f, 0.130f, 0.58f,
                  0.012f, 0.0060f, 0.0055f, 0.30f);
        break;
    case BiomeKind::DarkCave:
    case BiomeKind::IceCave:
    case BiomeKind::Volcano:
        // Subterranean cave pools: dark still glossy water, minimal reflection, zero sun glints.
        set_biome(0.04f, 0.06f, 0.08f,  0.07f, 0.09f, 0.11f, 0.28f,
                  0.015f, 0.032f, 0.042f, 0.26f,
                  0.020f, 0.014f, 0.012f, 0.00f);
        break;
    default:
        set_biome(0.18f, 0.34f, 0.52f,  0.38f, 0.48f, 0.54f, 0.44f,
                  0.025f, 0.100f, 0.125f, 0.60f,
                  0.013f, 0.0065f, 0.0055f, 0.22f);
        break;
    }
    out.sky_zenith[3] = elapsed_sec;

    out.slots[0] = bg_color_slot;
    out.slots[1] = bg_depth_slot;
    out.slots[2] = noise3d_tex_.slot;
    out.slots[3] = BindlessDescriptors::sampler_index(true, false, true, true, 1);

    out.ripple_count = static_cast<std::uint32_t>(kMaxRipples);
    for (std::size_t i = 0; i < kMaxRipples; ++i) {
        out.ripples[i][0] = ripples_[i].x;
        out.ripples[i][1] = ripples_[i].z;
        out.ripples[i][2] = elapsed_sec - ripples_[i].birth_sec;
        out.ripples[i][3] = ripples_[i].radius;
    }
}

void LightingPipeline::resolve_sun_dir_world(const LightingFrameInput &in, float out[3]) const noexcept {
    float wx = 0.34f, wy = 0.78f, wz = 0.52f;
    if (active_is_cave_) {
        wx = 0.14f; wy = 0.95f; wz = 0.28f;
    } else if (active_zone_ >= 0 && static_cast<std::size_t>(active_zone_) < zones_.size() &&
               zones_[static_cast<std::size_t>(active_zone_)].sun_height_deg > 0.0f) {
        // Prefer deterministic per-zone sun angle so shadows never swing when per-object GE light0 changes!
        const auto &z = zones_[static_cast<std::size_t>(active_zone_)];
        const float elev = std::clamp(z.sun_height_deg, 18.0f, 75.0f) * (3.14159265f / 180.0f);
        const float azim = z.sun_turn * (3.14159265f / 180.0f);
        const float ce = std::cos(elev);
        wx = ce * std::sin(azim);
        wy = std::sin(elev);
        wz = ce * std::cos(azim);
    } else if (in.has_light0) {
        wx = in.light0_dir[0];
        wy = in.light0_dir[1];
        wz = in.light0_dir[2];
    }
    const float wlen = std::sqrt(wx * wx + wy * wy + wz * wz);
    if (wlen > 1e-4f) { wx /= wlen; wy /= wlen; wz /= wlen; }
    out[0] = wx; out[1] = wy; out[2] = wz;
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

    // World-space light direction: deterministic per-zone sun angle (prevents shadow direction jumps!),
    // with overhead cave fill indoors and first-static-draw GE light0 fallback for unknown zones.
    float sun_w[3]{};
    resolve_sun_dir_world(in, sun_w);
    float wx = sun_w[0], wy = sun_w[1], wz = sun_w[2];
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

bool LightingPipeline::ensure_noise3d(VkCommandBuffer cmd, const Context &ctx, BindlessDescriptors &desc) {
    if (noise3d_tex_.image != VK_NULL_HANDLE && noise3d_tex_.slot != 0u) return true;
    // Generate a tileable 32x32x32 3D Perlin-Worley vapor noise volume packed into a 256x128 (8x4) R8_UNORM atlas (32 KB).
    constexpr std::uint32_t kN = 32u;
    constexpr std::uint32_t kCols = 8u;
    constexpr std::uint32_t kRows = 4u;
    constexpr std::uint32_t kAtlasW = kN * kCols;
    constexpr std::uint32_t kAtlasH = kN * kRows;
    constexpr VkDeviceSize kBytes = kAtlasW * kAtlasH;

    auto hash3 = [](int x, int y, int z, int period) noexcept -> float {
        const int px = ((x % period) + period) % period;
        const int py = ((y % period) + period) % period;
        const int pz = ((z % period) + period) % period;
        std::uint32_t h = static_cast<std::uint32_t>(px * 374761393 + py * 668265263 + pz * 2147483647);
        h = (h ^ (h >> 13u)) * 1274126177u;
        return static_cast<float>((h ^ (h >> 16u)) & 0xFFFFu) / 65535.0f;
    };
    auto smooth_vnoise = [&](float fx, float fy, float fz, int period) noexcept -> float {
        const float sx = fx * static_cast<float>(period);
        const float sy = fy * static_cast<float>(period);
        const float sz = fz * static_cast<float>(period);
        const int x0 = static_cast<int>(std::floor(sx));
        const int y0 = static_cast<int>(std::floor(sy));
        const int z0 = static_cast<int>(std::floor(sz));
        float tx = sx - static_cast<float>(x0); tx = tx * tx * (3.0f - 2.0f * tx);
        float ty = sy - static_cast<float>(y0); ty = ty * ty * (3.0f - 2.0f * ty);
        float tz = sz - static_cast<float>(z0); tz = tz * tz * (3.0f - 2.0f * tz);
        auto mixf = [](float a, float b, float t) noexcept { return a + (b - a) * t; };
        const float c000 = hash3(x0, y0, z0, period),     c100 = hash3(x0 + 1, y0, z0, period);
        const float c010 = hash3(x0, y0 + 1, z0, period), c110 = hash3(x0 + 1, y0 + 1, z0, period);
        const float c001 = hash3(x0, y0, z0 + 1, period), c101 = hash3(x0 + 1, y0, z0 + 1, period);
        const float c011 = hash3(x0, y0 + 1, z0 + 1, period), c111 = hash3(x0 + 1, y0 + 1, z0 + 1, period);
        const float c00 = mixf(c000, c100, tx), c10 = mixf(c010, c110, tx);
        const float c01 = mixf(c001, c101, tx), c11 = mixf(c011, c111, tx);
        return mixf(mixf(c00, c10, ty), mixf(c01, c11, ty), tz);
    };

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, kBytes,
                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
    VmaAllocationCreateInfo bvaci{};
    bvaci.usage = VMA_MEMORY_USAGE_AUTO;
    bvaci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo binfo{};
    if (vmaCreateBuffer(ctx.allocator, &bci, &bvaci, &noise3d_stage_buf_, &noise3d_stage_alloc_, &binfo) != VK_SUCCESS ||
        !binfo.pMappedData) {
        return false;
    }
    auto *dst = static_cast<std::uint8_t *>(binfo.pMappedData);
    for (std::uint32_t z = 0u; z < kN; ++z) {
        const std::uint32_t tx = (z % kCols) * kN;
        const std::uint32_t ty = (z / kCols) * kN;
        const float fz = (static_cast<float>(z) + 0.5f) / static_cast<float>(kN);
        for (std::uint32_t y = 0u; y < kN; ++y) {
            const float fy = (static_cast<float>(y) + 0.5f) / static_cast<float>(kN);
            for (std::uint32_t x = 0u; x < kN; ++x) {
                const float fx = (static_cast<float>(x) + 0.5f) / static_cast<float>(kN);
                // 3-octave tileable FBM cloud/vapor density.
                const float n = smooth_vnoise(fx, fy, fz, 4) * 0.55f +
                                smooth_vnoise(fx, fy, fz, 8) * 0.30f +
                                smooth_vnoise(fx, fy, fz, 16) * 0.15f;
                const float shaped = std::clamp((n - 0.18f) * 1.45f, 0.0f, 1.0f);
                dst[(ty + y) * kAtlasW + (tx + x)] = static_cast<std::uint8_t>(std::lround(shaped * 255.0f));
            }
        }
    }

    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, nullptr};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R8_UNORM;
    ici.extent = {kAtlasW, kAtlasH, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    VmaAllocationCreateInfo ivaci{};
    ivaci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (vmaCreateImage(ctx.allocator, &ici, &ivaci, &noise3d_tex_.image, &noise3d_tex_.alloc, nullptr) != VK_SUCCESS) {
        return false;
    }
    VkImageViewCreateInfo ivci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, nullptr};
    ivci.image = noise3d_tex_.image;
    ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ivci.format = VK_FORMAT_R8_UNORM;
    ivci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(ctx.device, &ivci, nullptr, &noise3d_tex_.view) != VK_SUCCESS) {
        noise3d_tex_.destroy(ctx, desc);
        return false;
    }
    noise3d_tex_.w = kAtlasW;
    noise3d_tex_.h = kAtlasH;
    noise3d_tex_.slot = desc.allocate_texture(ctx, noise3d_tex_.view);
    if (noise3d_tex_.slot == 0u) {
        noise3d_tex_.destroy(ctx, desc);
        return false;
    }

    Context::cmd_transition_image(cmd, noise3d_tex_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                                  VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                                  VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {kAtlasW, kAtlasH, 1};
    vkCmdCopyBufferToImage(cmd, noise3d_stage_buf_, noise3d_tex_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    Context::cmd_transition_image(cmd, noise3d_tex_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    return true;
}

bool LightingPipeline::dispatch(VkCommandBuffer cmd, const Context &ctx, BindlessDescriptors &desc,
                                VkImage scene_color, VkImageView scene_color_view, std::uint32_t scene_color_slot,
                                std::uint32_t scene_depth_slot, std::uint32_t scene_w, std::uint32_t scene_h,
                                const LightingFrameInput &in, void *params_cpu_ptr, VkDeviceAddress params_bda,
                                bool run_fxaa, std::uint32_t hdr_input_slot, std::uint32_t gbuf_normal_slot,
                                VkDeviceAddress lights_bda, std::uint32_t light_count,
                                std::uint32_t shadow_map_slot, const float *shadow_from_view,
                                std::uint32_t rt_shadow_slot, const float *view_mat, float elapsed_sec,
                                VkImageView mat_id_view, std::uint32_t soft_depth_slot, bool has_water_draws) {
    if (!ready() || !params_cpu_ptr || params_bda == 0 || scene_depth_slot == 0 || scene_color_slot == 0) return false;
    if (!ensure_targets(ctx, desc, scene_w, scene_h)) return false;
    ensure_noise3d(cmd, ctx, desc);
    ensure_fallback_mat(cmd);
    froxel_.ensure_fallback(cmd);
    if (exposure_reset_) froxel_.reset_history();

    VkDescriptorSet scene_set = get_or_bind_scene_set(ctx, scene_color_view);
    VkDescriptorSet mat_set = get_or_bind_mat_set(ctx, mat_id_view);

    const std::uint32_t mask = live_ ? live_->vulkan_effects_mask.load(std::memory_order_relaxed) : LiveSettings::kVkEffectAll;
    const bool want_ao      = (mask & LiveSettings::kVkEffectAo) != 0u;
    const bool want_contact = (mask & LiveSettings::kVkEffectContactShadows) != 0u;
    const bool want_vol     = (mask & LiveSettings::kVkEffectVolumetricFog) != 0u;
    const bool want_bloom   = (mask & LiveSettings::kVkEffectBloom) != 0u;
    const bool want_ssgi    = (mask & LiveSettings::kVkEffectSsgi) != 0u;
    const bool want_spec    = (mask & LiveSettings::kVkEffectSpecular) != 0u;
    const bool want_fxaa    = run_fxaa;
    const bool want_vig     = (mask & LiveSettings::kVkEffectVignette) != 0u;
    const bool want_froxel  = (mask & LiveSettings::kVkEffectFroxelFog) != 0u;
    const bool want_water   = (mask & LiveSettings::kVkEffectWaterReflections) != 0u;

    const bool need_gtao_pass  = want_ao || want_contact || want_spec || want_ssgi ||
                                 (mask & LiveSettings::kVkEffectSunShadows) != 0u ||
                                 (mask & LiveSettings::kVkEffectDynamicLights) != 0u ||
                                 (mask & LiveSettings::kVkEffectRayTracing) != 0u;
    const bool need_bloom_src  = want_bloom;
    const bool use_hdr_src     = (hdr_input_slot != 0u);
    // Forward water replacement pass (ge_draw_water.frag) now handles immutable SSR, refraction,
    // Beer-Lambert absorption, caustics, and contact foam directly during water geometry rasterization.
    const bool run_water_ssr   = false;
    (void)want_water;
    (void)has_water_draws;
    const std::uint32_t src_color_slot = use_hdr_src ? hdr_input_slot : scene_color_slot;

    // Update analytic interactive water ripple ring buffer from area-space wake actors (0 VRAM, continuous at 120/180 Hz).
    if (in.fog_camera.ground_area != ripple_area_) {
        ripple_area_ = in.fog_camera.ground_area;
        for (auto &r : ripples_) r.birth_sec = -100.0f;
        for (auto &pa : prev_ripple_actors_) pa.valid = false;
    }
    if (in.fog_camera.wake_actor_count > 0u) {
        const std::uint32_t n_act = std::min<std::uint32_t>(in.fog_camera.wake_actor_count,
                                                            static_cast<std::uint32_t>(SceneCamera::kMaxWakeActors));
        for (std::uint32_t i = 0u; i < n_act; ++i) {
            const auto &wa = in.fog_camera.wake_actors[i];
            auto &pa = prev_ripple_actors_[i];
            if (!pa.valid) {
                pa = {wa.x, wa.z, true};
            } else {
                const float dx = wa.x - pa.x;
                const float dz = wa.z - pa.z;
                if (dx * dx + dz * dz >= 26.0f * 26.0f) {
                    ripples_[ripple_cursor_] = RippleRing{wa.x, wa.z, elapsed_sec, wa.radius};
                    ripple_cursor_ = (ripple_cursor_ + 1u) % kMaxRipples;
                    pa = {wa.x, wa.z, true};
                }
            }
        }
    }

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
    static const bool s_debug_mat = [] {
        const char *v = std::getenv("MHF_VK_DEBUG_MAT");
        return v != nullptr && v[0] == '1';
    }();
    base.ext_slots[3] = (active_is_cave_ ? 1u : 0u) | (want_vig ? 2u : 0u) | (want_fxaa ? 4u : 0u) |
                        (want_froxel ? 8u : 0u) | (in.has_psp_fog ? 16u : 0u) |
                        (s_debug_mat ? 64u : 0u) | (want_water ? 128u : 0u);
    base.vz_and_size[2] = static_cast<float>(scene_w);
    base.vz_and_size[3] = static_cast<float>(scene_h);
    base.tex_slots[0] = src_color_slot;
    base.tex_slots[1] = scene_depth_slot;
    base.tex_slots[2] = need_gtao_pass ? gtao_blur_tex_.slot : 0u;
    base.tex_slots[3] = want_vol ? vol_tex_.slot : 0u;
    base.aux_slots[0] = want_bloom ? bloom_blur_tex_.slot : 0u;
    base.aux_slots[1] = BindlessDescriptors::sampler_index(true, false, true, true, 1);
    base.ext_slots[0] = want_ssgi ? ssgi_blur_tex_.slot : 0u;
    base.ext_slots[1] = need_gtao_pass ? gtao_tex_.slot : 0u;
    base.phase1_slots[0] = gbuf_normal_slot;
    base.phase1_slots[1] = exposure_tex_.slot;
    base.phase1_slots[2] = (use_hdr_src ? 1u : 0u) | ((exposure_reset_ || !exposure_initialized_) ? 2u : 0u);
    base.phase1_slots[3] = 0u;
    base.lights_bda = lights_bda;
    base.light_count = std::min<std::uint32_t>(light_count, static_cast<std::uint32_t>(LightingFrameInput::kMaxViewLights));
    base.lights_pad = 0u;
    if (shadow_map_slot != 0u && shadow_from_view != nullptr && !active_is_cave_) {
        std::memcpy(base.shadow_from_view, shadow_from_view, sizeof(base.shadow_from_view));
        base.shadow_info[0] = shadow_map_slot;
        base.shadow_info[1] = BindlessDescriptors::sampler_index(false, false, true, true, 1);
    } else {
        base.shadow_info[0] = 0u;
        base.shadow_info[1] = 0u;
    }
    base.shadow_info[2] = rt_shadow_slot;
    base.shadow_info[3] = noise3d_tex_.slot;
    const float *fog_view = in.fog_camera.valid ? in.fog_camera.world_to_eye.data() : view_mat;
    if (fog_view != nullptr) {
        // Rigid column-major inverse: [R | t; 0 1] -> [R^T | -R^T t; 0 1]
        base.inv_view[0] = fog_view[0]; base.inv_view[1] = fog_view[4]; base.inv_view[2]  = fog_view[8];  base.inv_view[3]  = 0.0f;
        base.inv_view[4] = fog_view[1]; base.inv_view[5] = fog_view[5]; base.inv_view[6]  = fog_view[9];  base.inv_view[7]  = 0.0f;
        base.inv_view[8] = fog_view[2]; base.inv_view[9] = fog_view[6]; base.inv_view[10] = fog_view[10]; base.inv_view[11] = 0.0f;
        const float tx = fog_view[12], ty = fog_view[13], tz = fog_view[14];
        base.inv_view[12] = -(fog_view[0] * tx + fog_view[1] * ty + fog_view[2] * tz);
        base.inv_view[13] = -(fog_view[4] * tx + fog_view[5] * ty + fog_view[6] * tz);
        base.inv_view[14] = -(fog_view[8] * tx + fog_view[9] * ty + fog_view[10] * tz);
        base.inv_view[15] = 1.0f;
        base.env_time[0] = elapsed_sec;
        base.env_time[1] = base.inv_view[13]; // camera world Y (cm)
        // Per-biome + live PSP fog intensity modulation:
        // Clear areas (e.g. Base Camp, Pokke, dry Desert) stay clean (vstr < 0.12 -> no ground mist);
        // Misty Swamp/Forest numbers get thin static ground mist across the whole basin;
        // when the area's live PSP fog surges (closer kFogEnd / whiter kFogColor), it smoothly intensifies!
        const float live_fog = in.has_psp_fog ? in.psp_fog_intensity : 0.0f;
        const float psp_luma = in.has_psp_fog
            ? (in.psp_fog_rgb[0] * 0.2126f + in.psp_fog_rgb[1] * 0.7152f + in.psp_fog_rgb[2] * 0.0722f)
            : 0.35f;
        // Detect white/bright fog surges (e.g. Swamp Area 6/8 mist rolling in).
        const float white_surge = std::clamp((psp_luma - 0.24f) * 1.85f, 0.0f, 1.0f);
        float vstr = 0.0f;
        if (active_biome_ == BiomeKind::Swamp) {
            // Swamp: thin wispy ground mist in marsh areas (~0.38), surging denser/whiter/taller (~1.30) during fog events!
            vstr = (live_fog > 0.08f)
                ? std::clamp(0.32f + live_fog * 0.72f + white_surge * 0.45f, 0.0f, 1.40f)
                : 0.0f;
        } else if (active_biome_ == BiomeKind::DesertSandstorm) {
            vstr = std::clamp(0.40f + live_fog * 0.75f, 0.0f, 1.25f);
        } else if (active_biome_ == BiomeKind::GreatForest || active_biome_ == BiomeKind::Jungle) {
            vstr = (live_fog > 0.14f)
                ? std::clamp(live_fog * 0.75f + white_surge * live_fog * 0.25f, 0.0f, 1.10f)
                : 0.0f;
        } else if (active_biome_ == BiomeKind::SnowyPeak || active_biome_ == BiomeKind::IceCave) {
            vstr = (live_fog > 0.18f)
                ? std::clamp(live_fog * 0.65f + white_surge * live_fog * 0.25f, 0.0f, 1.00f)
                : 0.0f;
        } else if (active_biome_ == BiomeKind::DarkCave) {
            vstr = (live_fog > 0.20f) ? std::clamp(live_fog * 0.48f, 0.0f, 0.65f) : 0.0f;
        } else {
            vstr = (live_fog > 0.38f) ? std::clamp((live_fog - 0.38f) * 0.65f, 0.0f, 0.55f) : 0.0f;
        }
        if (active_biome_ == BiomeKind::Swamp) {
            const float weather = std::max(white_surge * live_fog, in.translucent_fog_signal);
            const float response = std::clamp((weather - 0.15f) / 0.55f, 0.0f, 1.0f);
            vstr = std::max(vstr, 1.10f * response);
        }
        const float weather = in.fog_camera.ground_mist_weather;
        if (weather >= 0.0f) {
            base.phase1_slots[3] = 1u + static_cast<std::uint32_t>(std::round(weather * 1024.0f));
        }
        base.env_time[2] = vstr;
        base.env_time[3] = 1.0f;
    } else {
        base.env_time[0] = elapsed_sec;
        base.env_time[3] = 0.0f;
    }

    // Per-zone GroundMistProfile (populated from host shadow_zone_mist_profiles table; game-agnostic).
    GroundMistProfile active_mist{};
    if (active_zone_ >= 0 && static_cast<std::size_t>(active_zone_) < zones_.size()) {
        active_mist = zones_[active_zone_].mist;
    } else if (active_biome_ == BiomeKind::Swamp && base.env_time[2] > 0.14f) {
        active_mist.enabled = true;
        active_mist.ground_mist_only = true;
    }
    base.mist_shape[0] = active_mist.base_height_cm;
    base.mist_shape[1] = active_mist.surge_height_cm;
    base.mist_shape[2] = active_mist.base_density;
    base.mist_shape[3] = active_mist.surge_density;
    base.mist_tint[0]  = active_mist.tint[0];
    base.mist_tint[1]  = active_mist.tint[1];
    base.mist_tint[2]  = active_mist.tint[2];
    base.mist_tint[3]  = active_mist.enabled ? 1.0f : 0.0f;

    const bool use_3d_froxel = want_vol && want_froxel && froxel_.ready() && (fog_view != nullptr);
    const bool run_2d_vol    = want_vol && !use_3d_froxel;
    if (use_3d_froxel) {
        base.ext_slots[3] |= 32u;
        base.tex_slots[3] = 0u;
    }

    auto *slots = reinterpret_cast<LightingComputeParams *>(params_cpu_ptr);
    slots[kPassGtao]        = base; slots[kPassGtao].aux_slots[2]        = gtao_tex_.w;       slots[kPassGtao].aux_slots[3]        = gtao_tex_.h;
    slots[kPassVol]         = base; slots[kPassVol].aux_slots[2]         = vol_tex_.w;        slots[kPassVol].aux_slots[3]         = vol_tex_.h;
    slots[kPassBloom]       = base; slots[kPassBloom].aux_slots[2]       = bloom_tex_.w;      slots[kPassBloom].aux_slots[3]       = bloom_tex_.h;
    slots[kPassSsgi]        = base; slots[kPassSsgi].aux_slots[2]        = ssgi_tex_.w;       slots[kPassSsgi].aux_slots[3]        = ssgi_tex_.h;
    slots[kPassGtaoDenoise] = base; slots[kPassGtaoDenoise].aux_slots[2] = gtao_blur_tex_.w;  slots[kPassGtaoDenoise].aux_slots[3] = gtao_blur_tex_.h;
    slots[kPassBloomBlur]   = base; slots[kPassBloomBlur].aux_slots[0]   = bloom_tex_.slot;   slots[kPassBloomBlur].aux_slots[2]   = bloom_blur_tex_.w; slots[kPassBloomBlur].aux_slots[3] = bloom_blur_tex_.h;
    slots[kPassSsgiBlur]    = base; slots[kPassSsgiBlur].aux_slots[0]    = ssgi_tex_.slot;    slots[kPassSsgiBlur].aux_slots[2]    = ssgi_blur_tex_.w;  slots[kPassSsgiBlur].aux_slots[3]  = ssgi_blur_tex_.h;
    slots[kPassFlash]       = base; slots[kPassFlash].aux_slots[0]       = bloom_tex_.slot;   slots[kPassFlash].aux_slots[2]       = 4u;                slots[kPassFlash].aux_slots[3]     = 2u;
    slots[kPassComposite]   = base; slots[kPassComposite].ext_slots[1]   = 0u; slots[kPassComposite].aux_slots[2] = scene_w; slots[kPassComposite].aux_slots[3] = scene_h;
    slots[kPassFxaa]        = base; slots[kPassFxaa].aux_slots[0]        = lit_tmp_tex_.slot; slots[kPassFxaa].aux_slots[2]        = scene_w;           slots[kPassFxaa].aux_slots[3]      = scene_h;
    slots[kPassExposure]    = base; slots[kPassExposure].aux_slots[2]    = 1u;                slots[kPassExposure].aux_slots[3]    = 1u;

    // Batch image layout transitions into one vkCmdPipelineBarrier2 per stage boundary.
    std::array<VkImageMemoryBarrier2, 12> barriers{};
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
    if (run_2d_vol)     queue_to_general(vol_tex_.image);
    if (need_bloom_src) queue_to_general(bloom_tex_.image);
    if (want_bloom)     queue_to_general(bloom_blur_tex_.image);
    if (want_ssgi) {
        queue_to_general(ssgi_tex_.image);
        queue_to_general(ssgi_blur_tex_.image);
    }
    if (want_fxaa) queue_to_general(lit_tmp_tex_.image);
    push_barrier(exposure_tex_.image,
                 exposure_initialized_ ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
                 VK_IMAGE_LAYOUT_GENERAL,
                 exposure_initialized_ ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : 0,
                 VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    flush_barriers();

    // True 3D Froxel Volumetric Fog (160x90x64 RGBA16F): world-locked vapor + directional self-shadowing +
    // temporal reprojection + Frostbite analytical integration.
    const VkDescriptorSet bindless_set = desc.set();
    bool froxel_dispatched = false;
    if (use_3d_froxel) {
        auto srgb_to_lin = [](float v) { return std::pow(std::clamp(v, 0.0f, 1.0f), 2.2f); };
        const float fog_lin[4] = {
            srgb_to_lin(base.fog_color[0]), srgb_to_lin(base.fog_color[1]), srgb_to_lin(base.fog_color[2]),
            base.fog_color[3]
        };
        const float scat_lin[4] = {
            srgb_to_lin(base.scatter_color[0]), srgb_to_lin(base.scatter_color[1]), srgb_to_lin(base.scatter_color[2]),
            base.scatter_color[3]
        };
        constexpr std::size_t kSlotFroxelInject = 11u;
        constexpr std::size_t kSlotFroxelInteg  = 12u;
        constexpr std::size_t kSlotFroxelFluid  = 13u;
        float ground_info[4]{};
        float hf_info[4]{};
        std::array<FogActor, FroxelFog::kMaxActors> wake_buf{};
        std::uint32_t wake_count = 0u;
        if (in.fog_camera.wake_actor_count > 0u) {
            const std::uint32_t n = std::min<std::uint32_t>(in.fog_camera.wake_actor_count,
                                                            static_cast<std::uint32_t>(FroxelFog::kMaxActors));
            for (std::uint32_t i = 0u; i < n; ++i) {
                const auto &wa = in.fog_camera.wake_actors[i];
                wake_buf[wake_count++] = FogActor{wa.x, wa.foot_y, wa.z, wa.radius};
            }
        } else if (in.fog_camera.has_hunter_feet) {
            const auto &feet = in.fog_camera.hunter_feet;
            wake_buf[wake_count++] = FogActor{feet[0], feet[1], feet[2], 68.0f};
        }
        const bool use_host_actors = (wake_count > 0u);
        froxel_dispatched = froxel_.dispatch(
            cmd, bindless_set, fog_view, base.inv_view, in.proj, base.sun_dir_view,
            fog_lin, scat_lin, elapsed_sec, base.env_time[2], in.frame_index,
            active_mist.enabled && active_mist.ground_mist_only,
            in.fog_camera.valid ? in.fog_camera.ground_area : 0u,
            active_mist.wind_x, active_mist.wind_z,
            in.fog_camera.heightfield.valid ? &in.fog_camera.heightfield : nullptr,
            noise3d_tex_.slot, base.aux_slots[1], base.shadow_info[0], base.shadow_info[1],
            (base.shadow_info[0] != 0u ? shadow_from_view : nullptr), rt_shadow_slot,
            lights_bda, base.light_count,
            use_host_actors ? wake_buf.data() : in.fog_actors,
            use_host_actors ? wake_count : in.fog_actor_count,
            &slots[kSlotFroxelInject], params_bda + kSlotFroxelInject * sizeof(LightingComputeParams),
            &slots[kSlotFroxelInteg],  params_bda + kSlotFroxelInteg  * sizeof(LightingComputeParams),
            &slots[kSlotFroxelFluid],  params_bda + kSlotFroxelFluid  * sizeof(LightingComputeParams),
            ground_info, hf_info);
        if (froxel_dispatched) {
            std::memcpy(slots[kPassComposite].ground_fog_info, ground_info, sizeof(ground_info));
            std::memcpy(slots[kPassComposite].ground_hf_info, hf_info, sizeof(hf_info));
            static std::uint64_t s_last_fog_log = ~0ull;
            static const bool s_log_fog = [] {
                const char *v = std::getenv("MHF_VK_PROFILE");
                return v != nullptr && v[0] == '1';
            }();
            if (s_log_fog && (in.frame_index != s_last_fog_log) && ((in.frame_index % 60u) == 0u)) {
                s_last_fog_log = in.frame_index;
                std::fprintf(stderr,
                             "[vk][fog] zone=%d biome=%u mist=%u live_fog=%.2f vstr=%.2f weather=%.2f actors=%u hf=%u cam=(%.0f,%.0f,%.0f) floor_y=%.1f\n",
                             active_zone_, static_cast<unsigned>(active_biome_), active_mist.enabled ? 1u : 0u,
                             in.has_psp_fog ? in.psp_fog_intensity : -1.0f,
                             base.env_time[2], in.fog_camera.ground_mist_weather,
                             use_host_actors ? wake_count : in.fog_actor_count,
                             hf_info[3] > 1.0f ? 1u : 0u,
                             base.inv_view[12], base.inv_view[13], base.inv_view[14], ground_info[1]);
            }
        }
    }

    // Bind global bindless Set 0 once; individual passes only rebind Set 1.
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_, 0, 1, &bindless_set, 0, nullptr);

    // Stage 1: Enabled half/quarter-res passes + 1-WG adaptive exposure run in parallel across GPU CUs.
    auto run_pass = [&](VkPipeline pipe, VkDescriptorSet out_set, std::uint32_t w, std::uint32_t h, std::size_t idx) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_, 1, 1, &out_set, 0, nullptr);
        const VkDeviceAddress addr = params_bda + idx * sizeof(LightingComputeParams);
        vkCmdPushConstants(cmd, pipe_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(addr), &addr);
        vkCmdDispatch(cmd, (w + 7u) / 8u, (h + 7u) / 8u, 1u);
    };

    if (need_gtao_pass) run_pass(gtao_pipe_, gtao_set_, gtao_tex_.w, gtao_tex_.h, kPassGtao);
    if (run_2d_vol)     run_pass(vol_pipe_, vol_set_, vol_tex_.w, vol_tex_.h, kPassVol);
    if (need_bloom_src) run_pass(bloom_pipe_, bloom_set_, bloom_tex_.w, bloom_tex_.h, kPassBloom);
    if (want_ssgi)      run_pass(ssgi_pipe_, ssgi_set_, ssgi_tex_.w, ssgi_tex_.h, kPassSsgi);
    {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, exposure_pipe_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_, 1, 1, &exposure_set_, 0, nullptr);
        const VkDeviceAddress addr = params_bda + kPassExposure * sizeof(LightingComputeParams);
        vkCmdPushConstants(cmd, pipe_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(addr), &addr);
        vkCmdDispatch(cmd, 1u, 1u, 1u);
        exposure_initialized_ = true;
        exposure_reset_ = false;
    }

    // Stage 2: Bilateral GTAO/ContactShadow denoise + Kawase Bloom/SSGI blur + 4x2 dynamic flash.
    if (need_gtao_pass) queue_to_read(gtao_tex_.image);
    if (need_bloom_src) queue_to_read(bloom_tex_.image);
    if (want_ssgi)      queue_to_read(ssgi_tex_.image);
    queue_to_read(exposure_tex_.image);
    flush_barriers();
    if (need_gtao_pass) run_pass(gtao_denoise_pipe_, gtao_blur_set_, gtao_blur_tex_.w, gtao_blur_tex_.h, kPassGtaoDenoise);
    if (want_bloom)     run_pass(glow_blur_pipe_, bloom_blur_set_, bloom_blur_tex_.w, bloom_blur_tex_.h, kPassBloomBlur);
    if (want_ssgi) {
        run_pass(glow_blur_pipe_, ssgi_blur_set_, ssgi_blur_tex_.w, ssgi_blur_tex_.h, kPassSsgiBlur);
    }

    // Barrier batch: transition produced targets -> SHADER_READ_ONLY.
    if (need_gtao_pass) queue_to_read(gtao_blur_tex_.image);
    if (run_2d_vol)     queue_to_read(vol_tex_.image);
    if (want_bloom)     queue_to_read(bloom_blur_tex_.image);
    if (want_ssgi) {
        queue_to_read(ssgi_blur_tex_.image);
    }

    // Step 3: 4-mip Min-Z Hi-Z Pyramid + Half-Res Water SSR + Semantic Sky/Canopy Fallback
    //         (runs ONLY when water draws were rendered this frame; 0.00ms in dry areas!).
    if (run_water_ssr) {
        auto *hiz_cpu = reinterpret_cast<HiZComputeParams *>(&slots[16]);
        const VkDeviceAddress hiz_bda = params_bda + 16u * sizeof(LightingComputeParams);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hiz_pipe_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_, 0, 1, &bindless_set, 0, nullptr);
        std::uint32_t prev_slot = scene_depth_slot;
        for (int m = 0; m < 4; ++m) {
            queue_to_general(hiz_tex_[static_cast<std::size_t>(m)].image);
            flush_barriers();
            hiz_cpu[m].info[0] = prev_slot;
            hiz_cpu[m].info[1] = hiz_tex_[static_cast<std::size_t>(m)].w;
            hiz_cpu[m].info[2] = hiz_tex_[static_cast<std::size_t>(m)].h;
            hiz_cpu[m].info[3] = 0u;
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_, 1, 1,
                                    &hiz_sets_[static_cast<std::size_t>(m)], 0, nullptr);
            const VkDeviceAddress addr = hiz_bda + static_cast<std::size_t>(m) * sizeof(HiZComputeParams);
            vkCmdPushConstants(cmd, pipe_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(addr), &addr);
            vkCmdDispatch(cmd, (hiz_tex_[static_cast<std::size_t>(m)].w + 7u) / 8u,
                          (hiz_tex_[static_cast<std::size_t>(m)].h + 7u) / 8u, 1u);
            queue_to_read(hiz_tex_[static_cast<std::size_t>(m)].image);
            prev_slot = hiz_tex_[static_cast<std::size_t>(m)].slot;
        }
        queue_to_general(water_ssr_tex_.image);
        flush_barriers();

        WaterSsrParams wp{};
        std::memcpy(wp.proj_info, base.proj_info, sizeof(wp.proj_info));
        wp.vz_and_size[0] = base.vz_and_size[0];
        wp.vz_and_size[1] = base.vz_and_size[1];
        wp.vz_and_size[2] = static_cast<float>(water_ssr_tex_.w);
        wp.vz_and_size[3] = static_cast<float>(water_ssr_tex_.h);
        std::memcpy(wp.inv_view, base.inv_view, sizeof(wp.inv_view));
        std::memcpy(wp.sun_dir_view, base.sun_dir_view, sizeof(wp.sun_dir_view));
        wp.world_up_view[0] = base.world_up_view[0];
        wp.world_up_view[1] = base.world_up_view[1];
        wp.world_up_view[2] = base.world_up_view[2];
        wp.world_up_view[3] = elapsed_sec;
        auto srgb_to_lin = [](float v) { return std::pow(std::clamp(v, 0.0f, 1.0f), 2.2f); };
        wp.fog_color[0] = srgb_to_lin(base.fog_color[0]);
        wp.fog_color[1] = srgb_to_lin(base.fog_color[1]);
        wp.fog_color[2] = srgb_to_lin(base.fog_color[2]);
        wp.fog_color[3] = static_cast<float>(base.ext_slots[2]);
        wp.slots0[0] = src_color_slot;
        wp.slots0[1] = scene_depth_slot;
        wp.slots0[2] = noise3d_tex_.slot;
        wp.slots0[3] = base.aux_slots[1];
        wp.hiz_slots[0] = hiz_tex_[0].slot;
        wp.hiz_slots[1] = hiz_tex_[1].slot;
        wp.hiz_slots[2] = hiz_tex_[2].slot;
        wp.hiz_slots[3] = hiz_tex_[3].slot;
        wp.extra_slots[0] = rt_shadow_slot;
        wp.extra_slots[1] = need_gtao_pass ? gtao_blur_tex_.slot : 0u;
        wp.extra_slots[2] = static_cast<std::uint32_t>(kMaxRipples);
        wp.extra_slots[3] = use_hdr_src ? 1u : 0u;
        for (std::size_t i = 0; i < kMaxRipples; ++i) {
            wp.ripples[i][0] = ripples_[i].x;
            wp.ripples[i][1] = ripples_[i].z;
            wp.ripples[i][2] = elapsed_sec - ripples_[i].birth_sec;
            wp.ripples[i][3] = ripples_[i].radius;
        }
        std::memcpy(&slots[14], &wp, sizeof(wp));

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, water_ssr_pipe_);
        const std::array<VkDescriptorSet, 4> wsets = {
            bindless_set, water_ssr_set_, froxel_.fallback_sample_set(), mat_set
        };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, comp_pipe_layout_, 0, 4, wsets.data(), 0, nullptr);
        const VkDeviceAddress waddr = params_bda + 14u * sizeof(LightingComputeParams);
        vkCmdPushConstants(cmd, comp_pipe_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(waddr), &waddr);
        vkCmdDispatch(cmd, (water_ssr_tex_.w + 7u) / 8u, (water_ssr_tex_.h + 7u) / 8u, 1u);
        queue_to_read(water_ssr_tex_.image);

        slots[kPassComposite].ext_slots[1] = water_ssr_tex_.slot;
        slots[kPassComposite].phase1_slots[0] = soft_depth_slot;
    }

    auto run_comp = [&](VkPipeline pipe, VkDescriptorSet out_set) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        const VkDescriptorSet froxel_set = froxel_dispatched ? froxel_.sample_set() : froxel_.fallback_sample_set();
        const std::array<VkDescriptorSet, 4> sets = {bindless_set, out_set, froxel_set, mat_set};
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, comp_pipe_layout_, 0, 4, sets.data(), 0, nullptr);
        const VkDeviceAddress addr = params_bda + kPassComposite * sizeof(LightingComputeParams);
        vkCmdPushConstants(cmd, comp_pipe_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(addr), &addr);
        vkCmdDispatch(cmd, (scene_w + 7u) / 8u, (scene_h + 7u) / 8u, 1u);
    };

    if (want_fxaa) {
        flush_barriers();
        // Stage 3: Unified full-res composite reading src_color_slot + 3D froxel volume -> writing lit_tmp_tex_.
        run_comp(comp_pipe_, lit_tmp_set_);

        // Stage 4: Pre-HUD Edge Anti-Aliasing (FXAA 3.11) -> scene_color.
        queue_to_read(lit_tmp_tex_.image);
        push_barrier(scene_color,
                     use_hdr_src ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_IMAGE_LAYOUT_GENERAL,
                     use_hdr_src ? 0 : VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                     VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        flush_barriers();
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_, 0, 1, &bindless_set, 0, nullptr);
        run_pass(fxaa_pipe_, scene_set, scene_w, scene_h, kPassFxaa);
    } else if (use_hdr_src) {
        // HDR input -> composite & tonemap directly into RGBA8 scene_color (no in-place read needed!).
        push_barrier(scene_color, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                     0, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        flush_barriers();
        run_comp(comp_pipe_, scene_set);
    } else {
        // FXAA off, LDR input: composite in place directly on scene_color!
        push_barrier(scene_color, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                     VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                     VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        flush_barriers();
        run_comp(comp_inplace_pipe_, scene_set);
    }
    return true;
}

bool LightingPipeline::dispatch_fxaa_only(VkCommandBuffer cmd, const Context &ctx, BindlessDescriptors &desc,
                                          VkImage scene_color, VkImageView scene_color_view,
                                          std::uint32_t scene_w, std::uint32_t scene_h,
                                          void *params_cpu_ptr, VkDeviceAddress params_bda) {
    if (!ready() || !params_cpu_ptr || params_bda == 0) return false;
    if (!ensure_targets(ctx, desc, scene_w, scene_h)) return false;

    VkDescriptorSet scene_set = get_or_bind_scene_set(ctx, scene_color_view);

    // Copy active 3D region (scene_w x scene_h) from scene_color -> lit_tmp_tex_, then FXAA back to scene_color.
    Context::cmd_transition_image(cmd, lit_tmp_tex_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                                  VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                  VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                  VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkImageCopy cp{};
    cp.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    cp.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    cp.extent = {scene_w, scene_h, 1};
    vkCmdCopyImage(cmd, scene_color, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   lit_tmp_tex_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);

    Context::cmd_transition_image(cmd, lit_tmp_tex_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    Context::cmd_transition_image(cmd, scene_color, VK_IMAGE_ASPECT_COLOR_BIT,
                                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                                  VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    LightingComputeParams p{};
    p.aux_slots[0] = lit_tmp_tex_.slot;
    p.aux_slots[1] = BindlessDescriptors::sampler_index(true, false, true, true, 1);
    p.aux_slots[2] = scene_w;
    p.aux_slots[3] = scene_h;
    p.ext_slots[3] = 4u; // enable FXAA bit
    std::memcpy(params_cpu_ptr, &p, sizeof(p));

    const VkDescriptorSet bindless_set = desc.set();
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_, 0, 1, &bindless_set, 0, nullptr);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fxaa_pipe_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_, 1, 1, &scene_set, 0, nullptr);
    vkCmdPushConstants(cmd, pipe_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params_bda), &params_bda);
    vkCmdDispatch(cmd, (scene_w + 7u) / 8u, (scene_h + 7u) / 8u, 1u);
    return true;
}

bool LightingPipeline::dispatch_taa(VkCommandBuffer cmd, const Context &ctx, BindlessDescriptors &desc,
                                    VkImage scene_color, std::uint32_t scene_color_slot, std::uint32_t scene_depth_slot,
                                    std::uint32_t scene_w, std::uint32_t scene_h, const LightingFrameInput &in,
                                    const float prev_from_curr_3x4[12], bool valid_reproj, bool reset_history,
                                    const float curr_jitter_uv[2], const float prev_jitter_uv[2],
                                    void *params_cpu_ptr, VkDeviceAddress params_bda) {
    if (!ready() || !params_cpu_ptr || params_bda == 0 || scene_color_slot == 0 || scene_depth_slot == 0) return false;
    if (!ensure_taa_targets(ctx, desc, scene_w, scene_h)) return false;

    const int prev_idx = taa_hist_idx_;
    const int next_idx = 1 - prev_idx;
    const bool do_reset = reset_history || !taa_hist_valid_;

    TaaComputeParams tp{};
    const float p00 = std::fabs(in.proj[0]) > 1e-4f ? in.proj[0] : 1.0f;
    const float p11 = std::fabs(in.proj[5]) > 1e-4f ? in.proj[5] : 1.76f;
    tp.proj_info[0] = 1.0f / p00;
    tp.proj_info[1] = 1.0f / p11;
    tp.proj_info[2] = in.proj[10];
    tp.proj_info[3] = std::fabs(in.proj[14]) > 1e-4f ? in.proj[14] : -10.0f;
    tp.vz_and_size[0] = in.vz_scale;
    tp.vz_and_size[1] = in.vz_center;
    tp.vz_and_size[2] = static_cast<float>(scene_w);
    tp.vz_and_size[3] = static_cast<float>(scene_h);
    if (prev_from_curr_3x4 && valid_reproj) {
        std::memcpy(tp.prev_from_curr0, prev_from_curr_3x4 + 0, 16);
        std::memcpy(tp.prev_from_curr1, prev_from_curr_3x4 + 4, 16);
        std::memcpy(tp.prev_from_curr2, prev_from_curr_3x4 + 8, 16);
    } else {
        tp.prev_from_curr0[0] = 1.0f;
        tp.prev_from_curr1[1] = 1.0f;
        tp.prev_from_curr2[2] = 1.0f;
    }
    if (curr_jitter_uv) {
        tp.jitter_info[0] = curr_jitter_uv[0];
        tp.jitter_info[1] = curr_jitter_uv[1];
    }
    if (prev_jitter_uv) {
        tp.jitter_info[2] = prev_jitter_uv[0];
        tp.jitter_info[3] = prev_jitter_uv[1];
    }
    tp.slots[0] = scene_color_slot;
    tp.slots[1] = do_reset ? 0u : taa_hist_[static_cast<std::size_t>(prev_idx)].slot;
    tp.slots[2] = scene_depth_slot;
    tp.slots[3] = BindlessDescriptors::sampler_index(true, false, true, true, 1);
    tp.flags[0] = scene_w;
    tp.flags[1] = scene_h;
    tp.flags[2] = do_reset ? 1u : 0u;
    tp.flags[3] = (valid_reproj && !do_reset) ? 1u : 0u;
    std::memcpy(params_cpu_ptr, &tp, sizeof(tp));

    Target2D &dst_hist = taa_hist_[static_cast<std::size_t>(next_idx)];
    Context::cmd_transition_image(cmd, dst_hist.image, VK_IMAGE_ASPECT_COLOR_BIT,
                                  VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                  VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    const VkDescriptorSet bindless_set = desc.set();
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_, 0, 1, &bindless_set, 0, nullptr);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, taa_pipe_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_, 1, 1,
                            &taa_hist_sets_[static_cast<std::size_t>(next_idx)], 0, nullptr);
    vkCmdPushConstants(cmd, pipe_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params_bda), &params_bda);
    vkCmdDispatch(cmd, (scene_w + 7u) / 8u, (scene_h + 7u) / 8u, 1u);

    // Copy resolved TAA image back into scene_color and leave dst_hist in SHADER_READ_ONLY_OPTIMAL for next frame.
    Context::cmd_transition_image(cmd, dst_hist.image, VK_IMAGE_ASPECT_COLOR_BIT,
                                  VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                  VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    Context::cmd_transition_image(cmd, scene_color, VK_IMAGE_ASPECT_COLOR_BIT,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                                  VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

    VkImageCopy cp{};
    cp.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    cp.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    cp.extent = {scene_w, scene_h, 1};
    vkCmdCopyImage(cmd, dst_hist.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   scene_color, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);

    Context::cmd_transition_image(cmd, dst_hist.image, VK_IMAGE_ASPECT_COLOR_BIT,
                                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

    taa_hist_idx_ = next_idx;
    taa_hist_valid_ = true;
    return true;
}

} // namespace mhf::psp::vk
