#include "vk_froxel_fog.hpp"
#include "vk_shaders_spv.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace mhf::psp::vk {

void FroxelFog::VolTex::destroy(const Context &ctx) noexcept {
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

bool FroxelFog::create_vol3d(const Context &ctx, VolTex &v, std::uint32_t w, std::uint32_t h, std::uint32_t d,
                             VkImageUsageFlags usage) {
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_3D;
    ici.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    ici.extent = {w, h, d};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = usage;
    VmaAllocationCreateInfo vaci{};
    vaci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (vmaCreateImage(ctx.allocator, &ici, &vaci, &v.image, &v.alloc, nullptr) != VK_SUCCESS) return false;

    VkImageViewCreateInfo ivci{};
    ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivci.image = v.image;
    ivci.viewType = VK_IMAGE_VIEW_TYPE_3D;
    ivci.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    ivci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(ctx.device, &ivci, nullptr, &v.view) != VK_SUCCESS) {
        v.destroy(ctx);
        return false;
    }
    v.layout = VK_IMAGE_LAYOUT_UNDEFINED;
    return true;
}

bool FroxelFog::create_tex2d(const Context &ctx, VolTex &v, std::uint32_t w, std::uint32_t h,
                             VkImageUsageFlags usage) {
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    ici.extent = {w, h, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = usage;
    VmaAllocationCreateInfo vaci{};
    vaci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (vmaCreateImage(ctx.allocator, &ici, &vaci, &v.image, &v.alloc, nullptr) != VK_SUCCESS) return false;

    VkImageViewCreateInfo ivci{};
    ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivci.image = v.image;
    ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ivci.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    ivci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(ctx.device, &ivci, nullptr, &v.view) != VK_SUCCESS) {
        v.destroy(ctx);
        return false;
    }
    v.layout = VK_IMAGE_LAYOUT_UNDEFINED;
    return true;
}

bool FroxelFog::init(const Context &ctx, VkDescriptorSetLayout bindless_layout) {
    VkSamplerCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vkCreateSampler(ctx.device, &sci, nullptr, &sampler3d_) != VK_SUCCESS) return false;
    if (vkCreateSampler(ctx.device, &sci, nullptr, &sampler2d_) != VK_SUCCESS) {
        shutdown(ctx);
        return false;
    }

    const VkImageUsageFlags vol_usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    const VkImageUsageFlags fb_usage  = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    if (!create_vol3d(ctx, scatter_vol_[0], kWidth, kHeight, kDepth, vol_usage) ||
        !create_vol3d(ctx, scatter_vol_[1], kWidth, kHeight, kDepth, vol_usage) ||
        !create_vol3d(ctx, integ_vol_, kWidth, kHeight, kDepth, vol_usage) ||
        !create_vol3d(ctx, fallback_vol_, 1u, 1u, 1u, fb_usage) ||
        !create_tex2d(ctx, fluid_vol_[0], kFluidSize, kFluidSize, vol_usage) ||
        !create_tex2d(ctx, fluid_vol_[1], kFluidSize, kFluidSize, vol_usage) ||
        !create_tex2d(ctx, fallback_fluid_, 1u, 1u, fb_usage)) {
        shutdown(ctx);
        return false;
    }

    // Work Set (3D froxel inject/integrate & 2D fluid): binding 0 = writeonly image, binding 1 = combined sampler.
    VkDescriptorSetLayoutBinding wb[2]{};
    wb[0].binding = 0;
    wb[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    wb[0].descriptorCount = 1;
    wb[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    wb[1].binding = 1;
    wb[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wb[1].descriptorCount = 1;
    wb[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo wdsl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 2, wb};
    if (vkCreateDescriptorSetLayout(ctx.device, &wdsl, nullptr, &work_set_layout_) != VK_SUCCESS ||
        vkCreateDescriptorSetLayout(ctx.device, &wdsl, nullptr, &fluid_set_layout_) != VK_SUCCESS) {
        shutdown(ctx);
        return false;
    }

    // Sample Set 2 (for lighting_composite.comp):
    //   binding 0 = combined sampler3D u_froxel_vol
    //   binding 1 = combined sampler2D u_ground_fluid
    VkDescriptorSetLayoutBinding sb[2]{};
    sb[0].binding = 0;
    sb[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sb[0].descriptorCount = 1;
    sb[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    sb[1].binding = 1;
    sb[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sb[1].descriptorCount = 1;
    sb[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo sdsl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 2, sb};
    if (vkCreateDescriptorSetLayout(ctx.device, &sdsl, nullptr, &sample_set_layout_) != VK_SUCCESS) {
        shutdown(ctx);
        return false;
    }

    VkDescriptorPoolSize psizes[2]{
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 6u},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 12u},
    };
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 9u, 2u, psizes};
    if (vkCreateDescriptorPool(ctx.device, &dpci, nullptr, &pool_) != VK_SUCCESS) {
        shutdown(ctx);
        return false;
    }

    VkDescriptorSetLayout alloc_layouts[9] = {
        work_set_layout_, work_set_layout_, work_set_layout_, work_set_layout_,
        fluid_set_layout_, fluid_set_layout_,
        sample_set_layout_, sample_set_layout_, sample_set_layout_,
    };
    VkDescriptorSet alloc_sets[9]{};
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, pool_, 9u, alloc_layouts};
    if (vkAllocateDescriptorSets(ctx.device, &dsai, alloc_sets) != VK_SUCCESS) {
        shutdown(ctx);
        return false;
    }
    inject_sets_[0]      = alloc_sets[0];
    inject_sets_[1]      = alloc_sets[1];
    integ_sets_[0]       = alloc_sets[2];
    integ_sets_[1]       = alloc_sets[3];
    fluid_sets_[0]       = alloc_sets[4];
    fluid_sets_[1]       = alloc_sets[5];
    sample_sets_[0]      = alloc_sets[6];
    sample_sets_[1]      = alloc_sets[7];
    fallback_sample_set_ = alloc_sets[8];

    auto write_work = [&](VkDescriptorSet ds, VkImageView out_v, VkSampler in_s, VkImageView in_v) {
        VkDescriptorImageInfo out_i{VK_NULL_HANDLE, out_v, VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorImageInfo in_i{in_s, in_v, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet w[2]{};
        w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &out_i, nullptr, nullptr};
        w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &in_i, nullptr, nullptr};
        vkUpdateDescriptorSets(ctx.device, 2, w, 0, nullptr);
    };
    write_work(inject_sets_[0], scatter_vol_[0].view, sampler3d_, scatter_vol_[1].view);
    write_work(inject_sets_[1], scatter_vol_[1].view, sampler3d_, scatter_vol_[0].view);
    write_work(integ_sets_[0],  integ_vol_.view,      sampler3d_, scatter_vol_[0].view);
    write_work(integ_sets_[1],  integ_vol_.view,      sampler3d_, scatter_vol_[1].view);
    write_work(fluid_sets_[0],  fluid_vol_[0].view,   sampler2d_, fluid_vol_[1].view);
    write_work(fluid_sets_[1],  fluid_vol_[1].view,   sampler2d_, fluid_vol_[0].view);

    auto write_sample = [&](VkDescriptorSet ds, VkImageView vol3d_v, VkImageView fluid2d_v) {
        VkDescriptorImageInfo i3{sampler3d_, vol3d_v, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorImageInfo i2{sampler2d_, fluid2d_v, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet w[2]{};
        w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &i3, nullptr, nullptr};
        w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &i2, nullptr, nullptr};
        vkUpdateDescriptorSets(ctx.device, 2, w, 0, nullptr);
    };
    write_sample(sample_sets_[0],      integ_vol_.view,    fluid_vol_[0].view);
    write_sample(sample_sets_[1],      integ_vol_.view,    fluid_vol_[1].view);
    write_sample(fallback_sample_set_, fallback_vol_.view, fallback_fluid_.view);

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkDeviceAddress)};
    VkDescriptorSetLayout inj_layouts[2] = {bindless_layout, work_set_layout_};
    VkPipelineLayoutCreateInfo inj_plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 2, inj_layouts, 1, &pcr};
    VkPipelineLayoutCreateInfo fl_plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &fluid_set_layout_, 1, &pcr};
    if (vkCreatePipelineLayout(ctx.device, &inj_plci, nullptr, &inject_pipe_layout_) != VK_SUCCESS ||
        vkCreatePipelineLayout(ctx.device, &inj_plci, nullptr, &integ_pipe_layout_) != VK_SUCCESS ||
        vkCreatePipelineLayout(ctx.device, &fl_plci,  nullptr, &fluid_pipe_layout_) != VK_SUCCESS) {
        shutdown(ctx);
        return false;
    }

    auto make_pipe = [&](const auto &spv, VkPipelineLayout layout) -> VkPipeline {
        VkShaderModule mod = ctx.create_shader_module(spv);
        if (!mod) return VK_NULL_HANDLE;
        VkComputePipelineCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                      VK_SHADER_STAGE_COMPUTE_BIT, mod, "main", nullptr};
        cpci.layout = layout;
        VkPipeline p = VK_NULL_HANDLE;
        vkCreateComputePipelines(ctx.device, ctx.pipeline_cache, 1, &cpci, nullptr, &p);
        vkDestroyShaderModule(ctx.device, mod, nullptr);
        return p;
    };
    inject_pipe_    = make_pipe(shaders::kFroxelInjectCompSpv,    inject_pipe_layout_);
    integrate_pipe_ = make_pipe(shaders::kFroxelIntegrateCompSpv, integ_pipe_layout_);
    fluid_pipe_     = make_pipe(shaders::kFroxelFluidCompSpv,     fluid_pipe_layout_);
    return ready();
}

void FroxelFog::shutdown(const Context &ctx) {
    if (inject_pipe_ != VK_NULL_HANDLE)       { vkDestroyPipeline(ctx.device, inject_pipe_, nullptr); inject_pipe_ = VK_NULL_HANDLE; }
    if (integrate_pipe_ != VK_NULL_HANDLE)    { vkDestroyPipeline(ctx.device, integrate_pipe_, nullptr); integrate_pipe_ = VK_NULL_HANDLE; }
    if (fluid_pipe_ != VK_NULL_HANDLE)        { vkDestroyPipeline(ctx.device, fluid_pipe_, nullptr); fluid_pipe_ = VK_NULL_HANDLE; }
    if (inject_pipe_layout_ != VK_NULL_HANDLE){ vkDestroyPipelineLayout(ctx.device, inject_pipe_layout_, nullptr); inject_pipe_layout_ = VK_NULL_HANDLE; }
    if (integ_pipe_layout_ != VK_NULL_HANDLE) { vkDestroyPipelineLayout(ctx.device, integ_pipe_layout_, nullptr); integ_pipe_layout_ = VK_NULL_HANDLE; }
    if (fluid_pipe_layout_ != VK_NULL_HANDLE) { vkDestroyPipelineLayout(ctx.device, fluid_pipe_layout_, nullptr); fluid_pipe_layout_ = VK_NULL_HANDLE; }
    if (pool_ != VK_NULL_HANDLE)              { vkDestroyDescriptorPool(ctx.device, pool_, nullptr); pool_ = VK_NULL_HANDLE; }
    if (work_set_layout_ != VK_NULL_HANDLE)   { vkDestroyDescriptorSetLayout(ctx.device, work_set_layout_, nullptr); work_set_layout_ = VK_NULL_HANDLE; }
    if (fluid_set_layout_ != VK_NULL_HANDLE)  { vkDestroyDescriptorSetLayout(ctx.device, fluid_set_layout_, nullptr); fluid_set_layout_ = VK_NULL_HANDLE; }
    if (sample_set_layout_ != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(ctx.device, sample_set_layout_, nullptr); sample_set_layout_ = VK_NULL_HANDLE; }
    if (sampler3d_ != VK_NULL_HANDLE)         { vkDestroySampler(ctx.device, sampler3d_, nullptr); sampler3d_ = VK_NULL_HANDLE; }
    if (sampler2d_ != VK_NULL_HANDLE)         { vkDestroySampler(ctx.device, sampler2d_, nullptr); sampler2d_ = VK_NULL_HANDLE; }
    scatter_vol_[0].destroy(ctx);
    scatter_vol_[1].destroy(ctx);
    integ_vol_.destroy(ctx);
    fallback_vol_.destroy(ctx);
    fluid_vol_[0].destroy(ctx);
    fluid_vol_[1].destroy(ctx);
    fallback_fluid_.destroy(ctx);
    history_valid_ = false;
    fallback_cleared_ = false;
    zone_floor_valid_ = false;
    fluid_valid_ = false;
    prev_actor_count_ = 0u;
}

void FroxelFog::ensure_fallback(VkCommandBuffer cmd) {
    if (fallback_cleared_ || fallback_vol_.image == VK_NULL_HANDLE || fallback_fluid_.image == VK_NULL_HANDLE) return;
    VkImageSubresourceRange rng{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    // Clear 3D froxel fallback to (0, 0, 0, 1) (zero scatter, full transmittance).
    Context::cmd_transition_image(cmd, fallback_vol_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                                  VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                                  VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkClearColorValue ccv3{};
    ccv3.float32[0] = 0.0f; ccv3.float32[1] = 0.0f; ccv3.float32[2] = 0.0f; ccv3.float32[3] = 1.0f;
    vkCmdClearColorImage(cmd, fallback_vol_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &ccv3, 1, &rng);
    Context::cmd_transition_image(cmd, fallback_vol_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    fallback_vol_.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // Clear 2D fluid fallback to (0, 0, 1, 0) (zero velocity, full undisturbed mist, zero rim).
    Context::cmd_transition_image(cmd, fallback_fluid_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                                  VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                                  VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkClearColorValue ccv2{};
    ccv2.float32[0] = 0.0f; ccv2.float32[1] = 0.0f; ccv2.float32[2] = 1.0f; ccv2.float32[3] = 0.0f;
    vkCmdClearColorImage(cmd, fallback_fluid_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &ccv2, 1, &rng);
    Context::cmd_transition_image(cmd, fallback_fluid_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    fallback_fluid_.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    fallback_cleared_ = true;
}

bool FroxelFog::dispatch(VkCommandBuffer cmd, VkDescriptorSet bindless_set,
                         const float view_mat[16], const float inv_view[16],
                         const float proj[16], const float sun_dir_view[4],
                         const float fog_color_lin[4], const float scatter_color_lin[4],
                         float elapsed_sec, float vapor_strength, std::uint64_t frame_idx,
                         std::uint32_t noise3d_slot, std::uint32_t linear_sampler,
                         std::uint32_t shadow_map_slot, std::uint32_t nearest_sampler,
                         const float *shadow_from_view, std::uint32_t rt_shadow_slot,
                         VkDeviceAddress lights_bda, std::uint32_t light_count,
                         const FogActor *actors, std::uint32_t actor_count,
                         void *inject_cpu, VkDeviceAddress inject_bda,
                         void *integ_cpu, VkDeviceAddress integ_bda,
                         void *fluid_cpu, VkDeviceAddress fluid_bda,
                         float out_ground_info[4]) {
    if (!ready() || !view_mat || !inv_view || !inject_cpu || !integ_cpu || !fluid_cpu) return false;
    ensure_fallback(cmd);

    // 0. Track world-locked zone floor Y & closest actor (hunter) for exact ground height + fluid grid centering!
    const float cx = inv_view[12], cy = inv_view[13], cz = inv_view[14];
    bool camera_cut = !history_valid_;
    if (zone_floor_valid_) {
        const float dx = cx - last_cam_world_[0];
        const float dy = cy - last_cam_world_[1];
        const float dz = cz - last_cam_world_[2];
        const float d2 = dx * dx + dy * dy + dz * dz;
        if (d2 > 4500.0f * 4500.0f) {
            zone_floor_valid_ = false;
            fluid_valid_ = false;
            prev_actor_count_ = 0u;
            camera_cut = true;
        } else if (d2 > 400.0f * 400.0f) {
            camera_cut = true;
        }
    }
    last_cam_world_[0] = cx;
    last_cam_world_[1] = cy;
    last_cam_world_[2] = cz;

    // Find the primary actor (closest skinned actor to camera = player/hunter in 3rd-person view).
    const std::uint32_t n_act = std::min<std::uint32_t>(actor_count, static_cast<std::uint32_t>(kMaxActors));
    int primary_idx = -1;
    float best_d2 = 1e20f;
    for (std::uint32_t i = 0u; i < n_act; ++i) {
        const float dx = actors[i].x - cx;
        const float dz = actors[i].z - cz;
        const float d2 = dx * dx + dz * dz;
        if (d2 < best_d2 && d2 < 1400.0f * 1400.0f) {
            best_d2 = d2;
            primary_idx = static_cast<int>(i);
        }
    }

    // Lock zone_floor_y_ strictly to the lowest marsh/basin floor walked in this zone
    // (never climbs upward when walking up a hill or orbiting camera; resets on zone teleport!).
    const float center_x = (primary_idx >= 0) ? actors[primary_idx].x : cx;
    const float center_z = (primary_idx >= 0) ? actors[primary_idx].z : cz;
    const float target_floor_y = (primary_idx >= 0) ? actors[primary_idx].foot_y : (cy - 175.0f);
    if (!zone_floor_valid_) {
        zone_floor_y_ = target_floor_y;
        zone_floor_valid_ = true;
    } else if (primary_idx >= 0 && target_floor_y < zone_floor_y_) {
        // Only settle downward into the lowest ground level of the area so mist stays 100% static!
        zone_floor_y_ = zone_floor_y_ * 0.90f + target_floor_y * 0.10f;
    }

    // 1. Pass 0: World-Locked 128x128 Interactive Fluid Wake Simulation (froxel_fluid.comp).
    //    Step once per 30 Hz game frame so 120 FPS frame-gen subframes NEVER zero out actor velocity!
    constexpr float kTexelCm = kFluidWorldSize / static_cast<float>(kFluidSize);
    const bool step_fluid = !fluid_valid_ || (frame_idx != last_fluid_frame_idx_);
    float origin_x = static_cast<float>(fluid_cell_origin_[0]) * kTexelCm;
    float origin_z = static_cast<float>(fluid_cell_origin_[1]) * kTexelCm;

    if (step_fluid) {
        const int new_cell_x = static_cast<int>(std::floor((center_x - 0.5f * kFluidWorldSize) / kTexelCm));
        const int new_cell_z = static_cast<int>(std::floor((center_z - 0.5f * kFluidWorldSize) / kTexelCm));
        int shift_x = 0, shift_z = 0;
        bool reset_fluid = !fluid_valid_;
        if (fluid_valid_) {
            shift_x = new_cell_x - fluid_cell_origin_[0];
            shift_z = new_cell_z - fluid_cell_origin_[1];
            if (std::abs(shift_x) >= static_cast<int>(kFluidSize) / 2 ||
                std::abs(shift_z) >= static_cast<int>(kFluidSize) / 2) {
                reset_fluid = true;
                shift_x = 0;
                shift_z = 0;
            }
        }
        fluid_cell_origin_[0] = new_cell_x;
        fluid_cell_origin_[1] = new_cell_z;
        origin_x = static_cast<float>(new_cell_x) * kTexelCm;
        origin_z = static_cast<float>(new_cell_z) * kTexelCm;

        const float dt = 1.0f / 30.0f;
        last_elapsed_sec_ = elapsed_sec;
        last_fluid_frame_idx_ = frame_idx;

        const int fcur = fluid_ping_;
        const int fprev = 1 - fluid_ping_;
        if (fluid_vol_[fprev].layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            Context::cmd_transition_image(cmd, fluid_vol_[fprev].image, VK_IMAGE_ASPECT_COLOR_BIT,
                                          fluid_vol_[fprev].layout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                          VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            fluid_vol_[fprev].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
        Context::cmd_transition_image(cmd, fluid_vol_[fcur].image, VK_IMAGE_ASPECT_COLOR_BIT,
                                      fluid_vol_[fcur].layout, VK_IMAGE_LAYOUT_GENERAL,
                                      fluid_vol_[fcur].layout == VK_IMAGE_LAYOUT_UNDEFINED
                                          ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT
                                          : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                      fluid_vol_[fcur].layout == VK_IMAGE_LAYOUT_UNDEFINED
                                          ? 0
                                          : VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        fluid_vol_[fcur].layout = VK_IMAGE_LAYOUT_GENERAL;

        FroxelFluidParams fp{};
        fp.grid_info[0] = origin_x;
        fp.grid_info[1] = origin_z;
        fp.grid_info[2] = kFluidWorldSize;
        fp.grid_info[3] = dt;
        fp.shift_and_t[0] = static_cast<float>(shift_x);
        fp.shift_and_t[1] = static_cast<float>(shift_z);
        fp.shift_and_t[2] = elapsed_sec;
        fp.shift_and_t[3] = reset_fluid ? 1.0f : 0.0f;
        fp.counts[0] = n_act;
        for (std::uint32_t i = 0u; i < n_act; ++i) {
            const FogActor &a = actors[i];
            float vx = 0.0f, vz = 0.0f;
            float best_match_d2 = 240.0f * 240.0f;
            for (std::uint32_t j = 0u; j < prev_actor_count_; ++j) {
                const float dx = a.x - prev_actors_[j].x;
                const float dz = a.z - prev_actors_[j].z;
                const float d2 = dx * dx + dz * dz;
                if (d2 < best_match_d2) {
                    best_match_d2 = d2;
                    vx = dx / dt;
                    vz = dz / dt;
                }
            }
            const float spd = std::sqrt(vx * vx + vz * vz);
            fp.actors[i].pos_rad[0] = a.x;
            fp.actors[i].pos_rad[1] = a.foot_y;
            fp.actors[i].pos_rad[2] = a.z;
            fp.actors[i].pos_rad[3] = std::clamp(a.radius, 65.0f, 260.0f);
            fp.actors[i].vel_pad[0] = vx;
            fp.actors[i].vel_pad[1] = 0.0f;
            fp.actors[i].vel_pad[2] = vz;
            fp.actors[i].vel_pad[3] = spd;
            prev_actors_[i] = a;
        }
        prev_actor_count_ = n_act;
        std::memcpy(fluid_cpu, &fp, sizeof(fp));

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fluid_pipe_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fluid_pipe_layout_, 0, 1, &fluid_sets_[fcur], 0, nullptr);
        vkCmdPushConstants(cmd, fluid_pipe_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(fluid_bda), &fluid_bda);
        vkCmdDispatch(cmd, (kFluidSize + 7u) / 8u, (kFluidSize + 7u) / 8u, 1u);

        Context::cmd_transition_image(cmd, fluid_vol_[fcur].image, VK_IMAGE_ASPECT_COLOR_BIT,
                                      VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        fluid_vol_[fcur].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        last_fluid_idx_ = fcur;
        fluid_ping_ = 1 - fluid_ping_;
        fluid_valid_ = true;
    }

    if (out_ground_info) {
        out_ground_info[0] = origin_x;
        out_ground_info[1] = zone_floor_y_;
        out_ground_info[2] = origin_z;
        out_ground_info[3] = kFluidWorldSize;
    }

    // 2. Pass 1 & 2: 3D Froxel Volume (160x90x64) for valley atmosphere, sun shafts & point lights.
    const int cur = ping_;
    const int prev = 1 - ping_;

    if (scatter_vol_[prev].layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        Context::cmd_transition_image(cmd, scatter_vol_[prev].image, VK_IMAGE_ASPECT_COLOR_BIT,
                                      scatter_vol_[prev].layout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                      VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        scatter_vol_[prev].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
    Context::cmd_transition_image(cmd, scatter_vol_[cur].image, VK_IMAGE_ASPECT_COLOR_BIT,
                                  scatter_vol_[cur].layout, VK_IMAGE_LAYOUT_GENERAL,
                                  scatter_vol_[cur].layout == VK_IMAGE_LAYOUT_UNDEFINED
                                      ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT
                                      : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                  scatter_vol_[cur].layout == VK_IMAGE_LAYOUT_UNDEFINED
                                      ? 0
                                      : VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    scatter_vol_[cur].layout = VK_IMAGE_LAYOUT_GENERAL;

    const float p00 = std::fabs(proj[0]) > 1e-4f ? proj[0] : 1.0f;
    const float p11 = std::fabs(proj[5]) > 1e-4f ? proj[5] : 1.76f;

    FroxelInjectParams ip{};
    std::memcpy(ip.inv_view, inv_view, sizeof(ip.inv_view));
    std::memcpy(ip.prev_view, history_valid_ ? prev_view_ : view_mat, sizeof(ip.prev_view));
    if (shadow_from_view) {
        std::memcpy(ip.shadow_from_view, shadow_from_view, sizeof(ip.shadow_from_view));
    }
    ip.proj_info[0] = 1.0f / p00;
    ip.proj_info[1] = 1.0f / p11;
    ip.proj_info[2] = p00;
    ip.proj_info[3] = p11;
    ip.prev_proj_info[0] = history_valid_ ? prev_p00_ : p00;
    ip.prev_proj_info[1] = history_valid_ ? prev_p11_ : p11;
    ip.prev_proj_info[2] = kNearZ;
    ip.prev_proj_info[3] = kFarZ;
    std::memcpy(ip.sun_dir_view, sun_dir_view, sizeof(ip.sun_dir_view));
    std::memcpy(ip.fog_color, fog_color_lin, sizeof(ip.fog_color));
    std::memcpy(ip.scatter_color, scatter_color_lin, sizeof(ip.scatter_color));
    ip.world_env[0] = elapsed_sec;
    ip.world_env[1] = zone_floor_y_;
    ip.world_env[2] = vapor_strength;
    ip.world_env[3] = static_cast<float>(frame_idx & 0xFFFFu);
    ip.slots[0] = noise3d_slot;
    ip.slots[1] = linear_sampler;
    ip.slots[2] = shadow_map_slot;
    ip.slots[3] = nearest_sampler;
    ip.rt_and_flags[0] = rt_shadow_slot;
    ip.rt_and_flags[1] = camera_cut ? 1u : 0u;
    ip.rt_and_flags[2] = light_count;
    ip.rt_and_flags[3] = 0u;
    ip.lights_bda = lights_bda;
    std::memcpy(inject_cpu, &ip, sizeof(ip));

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, inject_pipe_);
    VkDescriptorSet inj_sets[2] = {bindless_set, inject_sets_[cur]};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, inject_pipe_layout_, 0, 2, inj_sets, 0, nullptr);
    vkCmdPushConstants(cmd, inject_pipe_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(inject_bda), &inject_bda);
    vkCmdDispatch(cmd, (kWidth + 7u) / 8u, (kHeight + 7u) / 8u, kDepth);

    Context::cmd_transition_image(cmd, scatter_vol_[cur].image, VK_IMAGE_ASPECT_COLOR_BIT,
                                  VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    scatter_vol_[cur].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    Context::cmd_transition_image(cmd, integ_vol_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                                  integ_vol_.layout, VK_IMAGE_LAYOUT_GENERAL,
                                  integ_vol_.layout == VK_IMAGE_LAYOUT_UNDEFINED
                                      ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT
                                      : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                  integ_vol_.layout == VK_IMAGE_LAYOUT_UNDEFINED
                                      ? 0
                                      : VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    integ_vol_.layout = VK_IMAGE_LAYOUT_GENERAL;

    FroxelIntegrateParams intp{};
    intp.slice_info[0] = kNearZ;
    intp.slice_info[1] = kFarZ;
    std::memcpy(integ_cpu, &intp, sizeof(intp));

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, integrate_pipe_);
    VkDescriptorSet int_sets[2] = {bindless_set, integ_sets_[cur]};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, integ_pipe_layout_, 0, 2, int_sets, 0, nullptr);
    vkCmdPushConstants(cmd, integ_pipe_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(integ_bda), &integ_bda);
    vkCmdDispatch(cmd, (kWidth + 7u) / 8u, (kHeight + 7u) / 8u, 1u);

    Context::cmd_transition_image(cmd, integ_vol_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                                  VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    integ_vol_.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    std::memcpy(prev_view_, view_mat, sizeof(prev_view_));
    prev_p00_ = p00;
    prev_p11_ = p11;
    history_valid_ = true;
    ping_ = 1 - ping_;
    return true;
}

} // namespace mhf::psp::vk
