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
                             VkFormat fmt, VkImageUsageFlags usage) {
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = fmt;
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
    ivci.format = fmt;
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
    if (vkCreateSampler(ctx.device, &sci, nullptr, &sampler3d_) != VK_SUCCESS ||
        vkCreateSampler(ctx.device, &sci, nullptr, &sampler2d_) != VK_SUCCESS) {
        shutdown(ctx);
        return false;
    }

    const VkImageUsageFlags vol_usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    const VkImageUsageFlags fb_usage  = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    if (!create_vol3d(ctx, scatter_vol_[0], kWidth, kHeight, kDepth, vol_usage) ||
        !create_vol3d(ctx, scatter_vol_[1], kWidth, kHeight, kDepth, vol_usage) ||
        !create_vol3d(ctx, integ_vol_, kWidth, kHeight, kDepth, vol_usage) ||
        !create_vol3d(ctx, fallback_vol_, 1u, 1u, 1u, fb_usage) ||
        !create_tex2d(ctx, fluid_state_[0], kFluidSize, kFluidSize, VK_FORMAT_R16G16B16A16_SFLOAT, vol_usage) ||
        !create_tex2d(ctx, fluid_state_[1], kFluidSize, kFluidSize, VK_FORMAT_R16G16B16A16_SFLOAT, vol_usage) ||
        !create_tex2d(ctx, fluid_press_[0], kFluidSize, kFluidSize, VK_FORMAT_R16G16_SFLOAT, vol_usage) ||
        !create_tex2d(ctx, fluid_press_[1], kFluidSize, kFluidSize, VK_FORMAT_R16G16_SFLOAT, vol_usage) ||
        !create_tex2d(ctx, fallback_fluid_, 1u, 1u, VK_FORMAT_R16G16B16A16_SFLOAT, fb_usage)) {
        shutdown(ctx);
        return false;
    }

    // Layout 1: 1 storage image + 1 combined sampler (froxel inject/integrate, fluid advect, fluid jacobi).
    VkDescriptorSetLayoutBinding wb[2]{};
    wb[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    wb[1] = {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo wdsl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 2, wb};

    // Layout 2: 2 storage images + 1 combined sampler (fluid_vort_div.comp).
    VkDescriptorSetLayoutBinding vb[3]{};
    vb[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    vb[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    vb[2] = {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo vdsl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 3, vb};

    // Layout 3: 1 storage image + 2 combined samplers (fluid_project.comp).
    VkDescriptorSetLayoutBinding pb[3]{};
    pb[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    pb[1] = {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    pb[2] = {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo pdsl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 3, pb};

    // Layout 4 (Set 2 for lighting_composite.comp): 2 combined samplers (3D froxel + 2D Navier-Stokes fluid).
    VkDescriptorSetLayoutBinding sb[2]{};
    sb[0] = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    sb[1] = {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo sdsl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 2, sb};

    if (vkCreateDescriptorSetLayout(ctx.device, &wdsl, nullptr, &work_set_layout_) != VK_SUCCESS ||
        vkCreateDescriptorSetLayout(ctx.device, &vdsl, nullptr, &vort_div_layout_) != VK_SUCCESS ||
        vkCreateDescriptorSetLayout(ctx.device, &pdsl, nullptr, &project_layout_) != VK_SUCCESS ||
        vkCreateDescriptorSetLayout(ctx.device, &sdsl, nullptr, &sample_set_layout_) != VK_SUCCESS) {
        shutdown(ctx);
        return false;
    }

    // 15 descriptor sets total: 4 froxel + 2 advect + 2 vort_div + 2 jacobi + 2 project + 3 sample.
    VkDescriptorPoolSize psizes[2]{
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 16u},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 24u},
    };
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 15u, 2u, psizes};
    if (vkCreateDescriptorPool(ctx.device, &dpci, nullptr, &pool_) != VK_SUCCESS) {
        shutdown(ctx);
        return false;
    }

    VkDescriptorSetLayout alloc_layouts[15] = {
        work_set_layout_, work_set_layout_, work_set_layout_, work_set_layout_,
        work_set_layout_, work_set_layout_,
        vort_div_layout_, vort_div_layout_,
        work_set_layout_, work_set_layout_,
        project_layout_,  project_layout_,
        sample_set_layout_, sample_set_layout_, sample_set_layout_,
    };
    VkDescriptorSet alloc_sets[15]{};
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, pool_, 15u, alloc_layouts};
    if (vkAllocateDescriptorSets(ctx.device, &dsai, alloc_sets) != VK_SUCCESS) {
        shutdown(ctx);
        return false;
    }
    inject_sets_[0]      = alloc_sets[0];
    inject_sets_[1]      = alloc_sets[1];
    integ_sets_[0]       = alloc_sets[2];
    integ_sets_[1]       = alloc_sets[3];
    advect_sets_[0]      = alloc_sets[4];
    advect_sets_[1]      = alloc_sets[5];
    vort_div_sets_[0]    = alloc_sets[6];
    vort_div_sets_[1]    = alloc_sets[7];
    jacobi_sets_[0]      = alloc_sets[8];
    jacobi_sets_[1]      = alloc_sets[9];
    project_sets_[0]     = alloc_sets[10];
    project_sets_[1]     = alloc_sets[11];
    sample_sets_[0]      = alloc_sets[12];
    sample_sets_[1]      = alloc_sets[13];
    fallback_sample_set_ = alloc_sets[14];

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
    write_work(advect_sets_[0], fluid_state_[0].view, sampler2d_, fluid_state_[1].view);
    write_work(advect_sets_[1], fluid_state_[1].view, sampler2d_, fluid_state_[0].view);
    write_work(jacobi_sets_[0], fluid_press_[1].view, sampler2d_, fluid_press_[0].view);
    write_work(jacobi_sets_[1], fluid_press_[0].view, sampler2d_, fluid_press_[1].view);

    auto write_vort_div = [&](VkDescriptorSet ds, VkImageView st_out, VkImageView div_out, VkImageView st_in) {
        VkDescriptorImageInfo o0{VK_NULL_HANDLE, st_out, VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorImageInfo o1{VK_NULL_HANDLE, div_out, VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorImageInfo i2{sampler2d_, st_in, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet w[3]{};
        w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &o0, nullptr, nullptr};
        w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &o1, nullptr, nullptr};
        w[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 2, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &i2, nullptr, nullptr};
        vkUpdateDescriptorSets(ctx.device, 3, w, 0, nullptr);
    };
    // When cur==0: advect wrote state[0] -> vort_div reads state[0], writes state[1] + press[0].
    write_vort_div(vort_div_sets_[0], fluid_state_[1].view, fluid_press_[0].view, fluid_state_[0].view);
    write_vort_div(vort_div_sets_[1], fluid_state_[0].view, fluid_press_[0].view, fluid_state_[1].view);

    auto write_project = [&](VkDescriptorSet ds, VkImageView st_out, VkImageView st_in, VkImageView press_in) {
        VkDescriptorImageInfo o0{VK_NULL_HANDLE, st_out, VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorImageInfo i1{sampler2d_, st_in, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorImageInfo i2{sampler2d_, press_in, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet w[3]{};
        w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &o0, nullptr, nullptr};
        w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &i1, nullptr, nullptr};
        w[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 2, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &i2, nullptr, nullptr};
        vkUpdateDescriptorSets(ctx.device, 3, w, 0, nullptr);
    };
    // After 12 (even) Jacobi steps, final pressure is back in press[0]; project reads state[prev]+press[0] -> writes state[cur]!
    write_project(project_sets_[0], fluid_state_[0].view, fluid_state_[1].view, fluid_press_[0].view);
    write_project(project_sets_[1], fluid_state_[1].view, fluid_state_[0].view, fluid_press_[0].view);

    auto write_sample = [&](VkDescriptorSet ds, VkImageView vol3d_v, VkImageView fluid2d_v) {
        VkDescriptorImageInfo i3{sampler3d_, vol3d_v, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorImageInfo i2{sampler2d_, fluid2d_v, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet w[2]{};
        w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &i3, nullptr, nullptr};
        w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &i2, nullptr, nullptr};
        vkUpdateDescriptorSets(ctx.device, 2, w, 0, nullptr);
    };
    write_sample(sample_sets_[0],      integ_vol_.view,    fluid_state_[0].view);
    write_sample(sample_sets_[1],      integ_vol_.view,    fluid_state_[1].view);
    write_sample(fallback_sample_set_, fallback_vol_.view, fallback_fluid_.view);

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkDeviceAddress)};
    VkDescriptorSetLayout inj_layouts[2] = {bindless_layout, work_set_layout_};
    VkPipelineLayoutCreateInfo inj_plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 2, inj_layouts, 1, &pcr};
    VkPipelineLayoutCreateInfo adv_plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &work_set_layout_, 1, &pcr};
    VkPipelineLayoutCreateInfo vd_plci {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &vort_div_layout_, 0, nullptr};
    VkPipelineLayoutCreateInfo jac_plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &work_set_layout_, 0, nullptr};
    VkPipelineLayoutCreateInfo prj_plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &project_layout_,  0, nullptr};
    if (vkCreatePipelineLayout(ctx.device, &inj_plci, nullptr, &inject_pipe_layout_)   != VK_SUCCESS ||
        vkCreatePipelineLayout(ctx.device, &inj_plci, nullptr, &integ_pipe_layout_)    != VK_SUCCESS ||
        vkCreatePipelineLayout(ctx.device, &adv_plci, nullptr, &advect_pipe_layout_)   != VK_SUCCESS ||
        vkCreatePipelineLayout(ctx.device, &vd_plci,  nullptr, &vort_div_pipe_layout_) != VK_SUCCESS ||
        vkCreatePipelineLayout(ctx.device, &jac_plci, nullptr, &jacobi_pipe_layout_)   != VK_SUCCESS ||
        vkCreatePipelineLayout(ctx.device, &prj_plci, nullptr, &project_pipe_layout_)  != VK_SUCCESS) {
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
    advect_pipe_    = make_pipe(shaders::kFluidAdvectCompSpv,     advect_pipe_layout_);
    vort_div_pipe_  = make_pipe(shaders::kFluidVortDivCompSpv,    vort_div_pipe_layout_);
    jacobi_pipe_    = make_pipe(shaders::kFluidJacobiCompSpv,     jacobi_pipe_layout_);
    project_pipe_   = make_pipe(shaders::kFluidProjectCompSpv,    project_pipe_layout_);
    return ready();
}

void FroxelFog::shutdown(const Context &ctx) {
    auto dp = [&](VkPipeline &p) { if (p) { vkDestroyPipeline(ctx.device, p, nullptr); p = VK_NULL_HANDLE; } };
    auto dl = [&](VkPipelineLayout &l) { if (l) { vkDestroyPipelineLayout(ctx.device, l, nullptr); l = VK_NULL_HANDLE; } };
    auto ds = [&](VkDescriptorSetLayout &l) { if (l) { vkDestroyDescriptorSetLayout(ctx.device, l, nullptr); l = VK_NULL_HANDLE; } };
    dp(inject_pipe_); dp(integrate_pipe_); dp(advect_pipe_); dp(vort_div_pipe_); dp(jacobi_pipe_); dp(project_pipe_);
    dl(inject_pipe_layout_); dl(integ_pipe_layout_); dl(advect_pipe_layout_);
    dl(vort_div_pipe_layout_); dl(jacobi_pipe_layout_); dl(project_pipe_layout_);
    if (pool_ != VK_NULL_HANDLE) { vkDestroyDescriptorPool(ctx.device, pool_, nullptr); pool_ = VK_NULL_HANDLE; }
    ds(work_set_layout_); ds(vort_div_layout_); ds(project_layout_); ds(sample_set_layout_);
    if (sampler3d_ != VK_NULL_HANDLE) { vkDestroySampler(ctx.device, sampler3d_, nullptr); sampler3d_ = VK_NULL_HANDLE; }
    if (sampler2d_ != VK_NULL_HANDLE) { vkDestroySampler(ctx.device, sampler2d_, nullptr); sampler2d_ = VK_NULL_HANDLE; }
    scatter_vol_[0].destroy(ctx); scatter_vol_[1].destroy(ctx);
    integ_vol_.destroy(ctx); fallback_vol_.destroy(ctx);
    fluid_state_[0].destroy(ctx); fluid_state_[1].destroy(ctx);
    fluid_press_[0].destroy(ctx); fluid_press_[1].destroy(ctx);
    fallback_fluid_.destroy(ctx);
    history_valid_ = false;
    fallback_cleared_ = false;
    zone_floor_valid_ = false;
    zone_floor_frames_ = 0u;
    fluid_valid_ = false;
    prev_actor_count_ = 0u;
}

void FroxelFog::ensure_fallback(VkCommandBuffer cmd) {
    if (fallback_cleared_ || fallback_vol_.image == VK_NULL_HANDLE || fallback_fluid_.image == VK_NULL_HANDLE) return;
    VkImageSubresourceRange rng{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
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

    Context::cmd_transition_image(cmd, fallback_fluid_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                                  VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                                  VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkClearColorValue ccv2{};
    ccv2.float32[0] = 0.0f; ccv2.float32[1] = 0.0f; ccv2.float32[2] = 0.0f; ccv2.float32[3] = 0.0f;
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
                         float elapsed_sec, float vapor_strength, std::uint64_t frame_idx, bool ground_mist_only, std::uint16_t ground_area,
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

    auto trans = [&](VolTex &t, VkImageLayout new_l, VkAccessFlags2 dst_acc) {
        if (t.layout == new_l && new_l == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) return;
        const bool undef = (t.layout == VK_IMAGE_LAYOUT_UNDEFINED);
        const VkAccessFlags2 src_acc = undef ? 0
            : (t.layout == VK_IMAGE_LAYOUT_GENERAL ? VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT : VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        Context::cmd_transition_image(cmd, t.image, VK_IMAGE_ASPECT_COLOR_BIT, t.layout, new_l,
                                      undef ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                      src_acc, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, dst_acc);
        t.layout = new_l;
    };

    // 0. Detect zone teleports & lock static ground-mist floor to the area's marsh floor.
    const float cx = inv_view[12], cy = inv_view[13], cz = inv_view[14];
    bool camera_cut = !history_valid_;
    if (ground_area != ground_area_) {
        reset_history();
        ground_area_ = ground_area;
        camera_cut = true;
    }
    if (zone_floor_valid_) {
        const float dx = cx - last_cam_world_[0];
        const float dy = cy - last_cam_world_[1];
        const float dz = cz - last_cam_world_[2];
        const float d2 = dx * dx + dy * dy + dz * dz;
        if (ground_area == 0 && d2 > 4500.0f * 4500.0f) {
            zone_floor_valid_ = false;
            zone_floor_frames_ = 0u;
            fluid_valid_ = false;
            prev_actor_count_ = 0u;
            camera_cut = true;
        } else if (d2 > 400.0f * 400.0f) {
            camera_cut = true;
        }
    }
    last_cam_world_[0] = cx; last_cam_world_[1] = cy; last_cam_world_[2] = cz;

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
    const float center_x = (primary_idx >= 0) ? actors[primary_idx].x : cx;
    const float center_z = (primary_idx >= 0) ? actors[primary_idx].z : cz;
    const float target_floor_y = ground_area != 0 && n_act != 0
                                     ? actors[0].foot_y
                                     : (primary_idx >= 0 ? actors[primary_idx].foot_y : cy - 175.0f);
    if (!zone_floor_valid_) {
        zone_floor_y_ = target_floor_y;
        zone_floor_valid_ = true;
        zone_floor_frames_ = 1u;
    } else if (ground_area == 0 && primary_idx >= 0) {
        if (zone_floor_frames_ < 10u) {
            zone_floor_y_ = zone_floor_y_ * 0.65f + target_floor_y * 0.35f;
            ++zone_floor_frames_;
        } else if (target_floor_y < zone_floor_y_) {
            zone_floor_y_ = zone_floor_y_ * 0.92f + target_floor_y * 0.08f;
        } else if (target_floor_y > zone_floor_y_ + 220.0f) {
            // Safety recovery if hunter moved to a much higher elevation tier without a loading screen.
            zone_floor_y_ = zone_floor_y_ * 0.92f + target_floor_y * 0.08f;
        }
    }

    // 1. True 256x256 GPU Navier-Stokes Solver (stepped once per 30 Hz game frame = ~0.037ms amortized at 120 FPS!):
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
        const std::uint32_t gx = (kFluidSize + 7u) / 8u;
        const std::uint32_t gy = (kFluidSize + 7u) / 8u;

        // Prepare actor velocities & fluid params.
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
            fp.actors[i].pos_rad[0] = a.x;
            fp.actors[i].pos_rad[1] = a.foot_y;
            fp.actors[i].pos_rad[2] = a.z;
            fp.actors[i].pos_rad[3] = std::clamp(a.radius, 65.0f, 260.0f);
            fp.actors[i].vel_pad[0] = vx;
            fp.actors[i].vel_pad[1] = 0.0f;
            fp.actors[i].vel_pad[2] = vz;
            fp.actors[i].vel_pad[3] = std::sqrt(vx * vx + vz * vz);
            prev_actors_[i] = a;
        }
        prev_actor_count_ = n_act;
        std::memcpy(fluid_cpu, &fp, sizeof(fp));

        // Stage 1: MacCormack Advection + Bridson Curl-Noise + Actor Momentum (state[fprev] -> state[fcur]).
        trans(fluid_state_[fprev], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        trans(fluid_state_[fcur],  VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, advect_pipe_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, advect_pipe_layout_, 0, 1, &advect_sets_[fcur], 0, nullptr);
        vkCmdPushConstants(cmd, advect_pipe_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(fluid_bda), &fluid_bda);
        vkCmdDispatch(cmd, gx, gy, 1u);

        // Stage 2: Fedkiw Vorticity Confinement + Divergence (state[fcur] -> state[fprev], press[0]).
        trans(fluid_state_[fcur],  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        trans(fluid_state_[fprev], VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        trans(fluid_press_[0],     VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vort_div_pipe_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vort_div_pipe_layout_, 0, 1, &vort_div_sets_[fcur], 0, nullptr);
        vkCmdDispatch(cmd, gx, gy, 1u);

        // Stage 3: Harris Jacobi Pressure Poisson Solve (12 ping-pong iterations: press[0] <-> press[1]).
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, jacobi_pipe_);
        for (int it = 0; it < kJacobiIters; ++it) {
            const int src_p = it & 1;
            const int dst_p = 1 - src_p;
            trans(fluid_press_[src_p], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            trans(fluid_press_[dst_p], VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, jacobi_pipe_layout_, 0, 1, &jacobi_sets_[src_p], 0, nullptr);
            vkCmdDispatch(cmd, gx, gy, 1u);
        }

        // Stage 4: Stam Helmholtz-Hodge Incompressible Projection (state[fprev] + press[0] -> state[fcur]).
        trans(fluid_state_[fprev], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        trans(fluid_press_[0],     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        trans(fluid_state_[fcur],  VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, project_pipe_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, project_pipe_layout_, 0, 1, &project_sets_[fcur], 0, nullptr);
        vkCmdDispatch(cmd, gx, gy, 1u);

        trans(fluid_state_[fcur], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
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

    // 2. 3D Froxel Volume (160x90x64) for upper-air valley atmosphere, sun shafts & point lights.
    const int cur = ping_;
    const int prev = 1 - ping_;
    trans(scatter_vol_[prev], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    trans(scatter_vol_[cur],  VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    const float p00 = std::fabs(proj[0]) > 1e-4f ? proj[0] : 1.0f;
    const float p11 = std::fabs(proj[5]) > 1e-4f ? proj[5] : 1.76f;

    FroxelInjectParams ip{};
    std::memcpy(ip.inv_view, inv_view, sizeof(ip.inv_view));
    std::memcpy(ip.prev_view, history_valid_ ? prev_view_ : view_mat, sizeof(ip.prev_view));
    if (shadow_from_view) {
        std::memcpy(ip.shadow_from_view, shadow_from_view, sizeof(ip.shadow_from_view));
    }
    ip.proj_info[0] = 1.0f / p00; ip.proj_info[1] = 1.0f / p11; ip.proj_info[2] = p00; ip.proj_info[3] = p11;
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
    ip.slots[0] = noise3d_slot; ip.slots[1] = linear_sampler; ip.slots[2] = shadow_map_slot; ip.slots[3] = nearest_sampler;
    ip.rt_and_flags[0] = rt_shadow_slot; ip.rt_and_flags[1] = camera_cut ? 1u : 0u; ip.rt_and_flags[2] = light_count;
    ip.rt_and_flags[3] = ground_mist_only ? 1u : 0u;
    ip.lights_bda = lights_bda;
    std::memcpy(inject_cpu, &ip, sizeof(ip));

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, inject_pipe_);
    VkDescriptorSet inj_sets[2] = {bindless_set, inject_sets_[cur]};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, inject_pipe_layout_, 0, 2, inj_sets, 0, nullptr);
    vkCmdPushConstants(cmd, inject_pipe_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(inject_bda), &inject_bda);
    vkCmdDispatch(cmd, (kWidth + 7u) / 8u, (kHeight + 7u) / 8u, kDepth);

    trans(scatter_vol_[cur], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    trans(integ_vol_,        VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    FroxelIntegrateParams intp{};
    intp.slice_info[0] = kNearZ;
    intp.slice_info[1] = kFarZ;
    std::memcpy(integ_cpu, &intp, sizeof(intp));

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, integrate_pipe_);
    VkDescriptorSet int_sets[2] = {bindless_set, integ_sets_[cur]};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, integ_pipe_layout_, 0, 2, int_sets, 0, nullptr);
    vkCmdPushConstants(cmd, integ_pipe_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(integ_bda), &integ_bda);
    vkCmdDispatch(cmd, (kWidth + 7u) / 8u, (kHeight + 7u) / 8u, 1u);

    trans(integ_vol_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

    std::memcpy(prev_view_, view_mat, sizeof(prev_view_));
    prev_p00_ = p00;
    prev_p11_ = p11;
    history_valid_ = true;
    ping_ = 1 - ping_;
    return true;
}

} // namespace mhf::psp::vk
