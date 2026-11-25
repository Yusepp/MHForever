#include "vk_froxel_fog.hpp"
#include "vk_shaders_spv.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace mhf::psp::vk {

void FroxelFog::Vol3D::destroy(const Context &ctx) noexcept {
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

bool FroxelFog::create_vol3d(const Context &ctx, Vol3D &v, std::uint32_t w, std::uint32_t h, std::uint32_t d,
                             VkImageUsageFlags usage) {
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, nullptr};
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

    VkImageViewCreateInfo ivci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, nullptr};
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

bool FroxelFog::init(const Context &ctx, VkDescriptorSetLayout bindless_layout) {
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, nullptr};
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vkCreateSampler(ctx.device, &sci, nullptr, &sampler3d_) != VK_SUCCESS) return false;

    const VkImageUsageFlags vol_usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    if (!create_vol3d(ctx, scatter_vol_[0], kWidth, kHeight, kDepth, vol_usage) ||
        !create_vol3d(ctx, scatter_vol_[1], kWidth, kHeight, kDepth, vol_usage) ||
        !create_vol3d(ctx, integ_vol_, kWidth, kHeight, kDepth, vol_usage) ||
        !create_vol3d(ctx, fallback_vol_, 1u, 1u, 1u, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT)) {
        shutdown(ctx);
        return false;
    }

    // Work Set 1: binding 0 = writeonly image3D, binding 1 = combined sampler3D.
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
    if (vkCreateDescriptorSetLayout(ctx.device, &wdsl, nullptr, &work_set_layout_) != VK_SUCCESS) {
        shutdown(ctx);
        return false;
    }

    // Sample Set 2 (for lighting_composite.comp): binding 0 = combined sampler3D u_froxel_vol.
    VkDescriptorSetLayoutBinding sb{};
    sb.binding = 0;
    sb.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sb.descriptorCount = 1;
    sb.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo sdsl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 1, &sb};
    if (vkCreateDescriptorSetLayout(ctx.device, &sdsl, nullptr, &sample_set_layout_) != VK_SUCCESS) {
        shutdown(ctx);
        return false;
    }

    VkDescriptorPoolSize psizes[2]{
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4u},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 6u},
    };
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 6u, 2u, psizes};
    if (vkCreateDescriptorPool(ctx.device, &dpci, nullptr, &pool_) != VK_SUCCESS) {
        shutdown(ctx);
        return false;
    }

    VkDescriptorSetLayout alloc_layouts[6] = {
        work_set_layout_, work_set_layout_, work_set_layout_, work_set_layout_,
        sample_set_layout_, sample_set_layout_,
    };
    VkDescriptorSet alloc_sets[6]{};
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, pool_, 6u, alloc_layouts};
    if (vkAllocateDescriptorSets(ctx.device, &dsai, alloc_sets) != VK_SUCCESS) {
        shutdown(ctx);
        return false;
    }
    inject_sets_[0] = alloc_sets[0];
    inject_sets_[1] = alloc_sets[1];
    integ_sets_[0]  = alloc_sets[2];
    integ_sets_[1]  = alloc_sets[3];
    sample_set_     = alloc_sets[4];
    fallback_sample_set_ = alloc_sets[5];

    auto write_work = [&](VkDescriptorSet ds, VkImageView out_v, VkImageView in_v) {
        VkDescriptorImageInfo out_i{VK_NULL_HANDLE, out_v, VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorImageInfo in_i{sampler3d_, in_v, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet w[2]{};
        w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &out_i, nullptr, nullptr};
        w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &in_i, nullptr, nullptr};
        vkUpdateDescriptorSets(ctx.device, 2, w, 0, nullptr);
    };
    write_work(inject_sets_[0], scatter_vol_[0].view, scatter_vol_[1].view);
    write_work(inject_sets_[1], scatter_vol_[1].view, scatter_vol_[0].view);
    write_work(integ_sets_[0],  integ_vol_.view,      scatter_vol_[0].view);
    write_work(integ_sets_[1],  integ_vol_.view,      scatter_vol_[1].view);

    auto write_sample = [&](VkDescriptorSet ds, VkImageView v) {
        VkDescriptorImageInfo i{sampler3d_, v, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 0, 0, 1,
                               VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &i, nullptr, nullptr};
        vkUpdateDescriptorSets(ctx.device, 1, &w, 0, nullptr);
    };
    write_sample(sample_set_, integ_vol_.view);
    write_sample(fallback_sample_set_, fallback_vol_.view);

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkDeviceAddress)};
    VkDescriptorSetLayout inj_layouts[2] = {bindless_layout, work_set_layout_};
    VkPipelineLayoutCreateInfo inj_plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 2, inj_layouts, 1, &pcr};
    if (vkCreatePipelineLayout(ctx.device, &inj_plci, nullptr, &inject_pipe_layout_) != VK_SUCCESS) {
        shutdown(ctx);
        return false;
    }
    VkDescriptorSetLayout int_layouts[2] = {bindless_layout, work_set_layout_};
    VkPipelineLayoutCreateInfo int_plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 2, int_layouts, 1, &pcr};
    if (vkCreatePipelineLayout(ctx.device, &int_plci, nullptr, &integ_pipe_layout_) != VK_SUCCESS) {
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
    inject_pipe_ = make_pipe(shaders::kFroxelInjectCompSpv, inject_pipe_layout_);
    integrate_pipe_ = make_pipe(shaders::kFroxelIntegrateCompSpv, integ_pipe_layout_);
    return ready();
}

void FroxelFog::shutdown(const Context &ctx) {
    if (inject_pipe_ != VK_NULL_HANDLE) { vkDestroyPipeline(ctx.device, inject_pipe_, nullptr); inject_pipe_ = VK_NULL_HANDLE; }
    if (integrate_pipe_ != VK_NULL_HANDLE) { vkDestroyPipeline(ctx.device, integrate_pipe_, nullptr); integrate_pipe_ = VK_NULL_HANDLE; }
    if (inject_pipe_layout_ != VK_NULL_HANDLE) { vkDestroyPipelineLayout(ctx.device, inject_pipe_layout_, nullptr); inject_pipe_layout_ = VK_NULL_HANDLE; }
    if (integ_pipe_layout_ != VK_NULL_HANDLE) { vkDestroyPipelineLayout(ctx.device, integ_pipe_layout_, nullptr); integ_pipe_layout_ = VK_NULL_HANDLE; }
    if (pool_ != VK_NULL_HANDLE) { vkDestroyDescriptorPool(ctx.device, pool_, nullptr); pool_ = VK_NULL_HANDLE; }
    if (work_set_layout_ != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(ctx.device, work_set_layout_, nullptr); work_set_layout_ = VK_NULL_HANDLE; }
    if (sample_set_layout_ != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(ctx.device, sample_set_layout_, nullptr); sample_set_layout_ = VK_NULL_HANDLE; }
    if (sampler3d_ != VK_NULL_HANDLE) { vkDestroySampler(ctx.device, sampler3d_, nullptr); sampler3d_ = VK_NULL_HANDLE; }
    scatter_vol_[0].destroy(ctx);
    scatter_vol_[1].destroy(ctx);
    integ_vol_.destroy(ctx);
    fallback_vol_.destroy(ctx);
    history_valid_ = false;
    fallback_cleared_ = false;
    zone_floor_valid_ = false;
}

void FroxelFog::ensure_fallback(VkCommandBuffer cmd) {
    if (fallback_cleared_ || fallback_vol_.image == VK_NULL_HANDLE) return;
    Context::cmd_transition_image(cmd, fallback_vol_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                                  VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                                  VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkClearColorValue ccv{};
    ccv.float32[0] = 0.0f; ccv.float32[1] = 0.0f; ccv.float32[2] = 0.0f; ccv.float32[3] = 1.0f;
    VkImageSubresourceRange rng{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cmd, fallback_vol_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &ccv, 1, &rng);
    Context::cmd_transition_image(cmd, fallback_vol_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    fallback_vol_.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
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
                         void *inject_cpu, VkDeviceAddress inject_bda,
                         void *integ_cpu, VkDeviceAddress integ_bda) {
    if (!ready() || !view_mat || !inv_view || !inject_cpu || !integ_cpu) return false;
    ensure_fallback(cmd);

    // Track world-locked zone floor Y (resets on zone teleport > 4500 cm, never bobs with camera pitch!).
    const float cx = inv_view[12], cy = inv_view[13], cz = inv_view[14];
    bool camera_cut = !history_valid_;
    if (zone_floor_valid_) {
        const float dx = cx - last_cam_world_[0];
        const float dy = cy - last_cam_world_[1];
        const float dz = cz - last_cam_world_[2];
        const float d2 = dx * dx + dy * dy + dz * dz;
        if (d2 > 4500.0f * 4500.0f) {
            zone_floor_valid_ = false;
            camera_cut = true;
        } else if (d2 > 400.0f * 400.0f) {
            camera_cut = true;
        }
    }
    if (!zone_floor_valid_) {
        zone_floor_y_ = cy - 175.0f;
        zone_floor_valid_ = true;
    } else {
        // Slowly settle to the lowest ground level walked in this zone so fog stays glued to the Swamp floor!
        const float cand = cy - 165.0f;
        if (cand < zone_floor_y_) {
            zone_floor_y_ = zone_floor_y_ * 0.96f + cand * 0.04f;
        } else {
            zone_floor_y_ = zone_floor_y_ * 0.998f + cand * 0.002f;
        }
    }
    last_cam_world_[0] = cx;
    last_cam_world_[1] = cy;
    last_cam_world_[2] = cz;

    const int cur = ping_;
    const int prev = 1 - ping_;

    // Ensure previous history volume is in SHADER_READ_ONLY_OPTIMAL (even on first frame when reset=1).
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

    // Pass 1: 3D Froxel Density + Lighting + Temporal Reprojection (160x90x64).
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, inject_pipe_);
    VkDescriptorSet inj_sets[2] = {bindless_set, inject_sets_[cur]};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, inject_pipe_layout_, 0, 2, inj_sets, 0, nullptr);
    vkCmdPushConstants(cmd, inject_pipe_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(inject_bda), &inject_bda);
    vkCmdDispatch(cmd, (kWidth + 7u) / 8u, (kHeight + 7u) / 8u, kDepth);

    // Transition scatter_vol_[cur] -> SHADER_READ_ONLY and integ_vol_ -> GENERAL.
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

    // Pass 2: Frostbite Closed-Form Analytical Z-Integration (160x90 columns x 64 slices).
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
