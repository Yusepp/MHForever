#include "vk_shadows.hpp"
#include "vk_shaders_spv.hpp"

#include <algorithm>
#include <array>
#include <cstdio>

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

} // namespace

bool ShadowCompute::init(const Context &ctx, VkDescriptorSetLayout bindless_layout) {
    VkDescriptorSetLayoutBinding b{};
    b.binding = 0;
    b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 1, &b};
    if (vkCreateDescriptorSetLayout(ctx.device, &dslci, nullptr, &storage_layout_) != VK_SUCCESS) return false;

    constexpr std::uint32_t kTotalSets = 1u + static_cast<std::uint32_t>(kSceneSetCount);
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kTotalSets};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, kTotalSets, 1, &ps};
    if (vkCreateDescriptorPool(ctx.device, &dpci, nullptr, &storage_pool_) != VK_SUCCESS) return false;

    std::array<VkDescriptorSetLayout, kTotalSets> dsls{};
    dsls.fill(storage_layout_);
    std::array<VkDescriptorSet, kTotalSets> sets{};
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, storage_pool_, kTotalSets, dsls.data()};
    if (vkAllocateDescriptorSets(ctx.device, &dsai, sets.data()) != VK_SUCCESS) return false;
    factor_storage_set_ = sets[0];
    for (std::size_t i = 0; i < kSceneSetCount; ++i) {
        scene_storage_sets_[i] = sets[1 + i];
    }
    scene_set_cursor_ = 0;

    std::array<VkDescriptorSetLayout, 2> pipe_sets = {bindless_layout, storage_layout_};
    VkPushConstantRange fpcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkDeviceAddress)};
    VkPipelineLayoutCreateInfo flci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 2, pipe_sets.data(), 1, &fpcr};
    if (vkCreatePipelineLayout(ctx.device, &flci, nullptr, &factor_layout_) != VK_SUCCESS) return false;

    VkPushConstantRange apcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ShadowApplyPush)};
    VkPipelineLayoutCreateInfo alci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 2, pipe_sets.data(), 1, &apcr};
    if (vkCreatePipelineLayout(ctx.device, &alci, nullptr, &apply_layout_) != VK_SUCCESS) return false;

    VkShaderModule fmod = make_mod(ctx, shaders::kShadowFactorCompSpv);
    VkShaderModule amod = make_mod(ctx, shaders::kShadowApplyCompSpv);
    if (!fmod || !amod) {
        if (fmod) vkDestroyShaderModule(ctx.device, fmod, nullptr);
        if (amod) vkDestroyShaderModule(ctx.device, amod, nullptr);
        return false;
    }

    VkComputePipelineCreateInfo fpci{};
    fpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    fpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, fmod, "main", nullptr};
    fpci.layout = factor_layout_;
    vkCreateComputePipelines(ctx.device, ctx.pipeline_cache, 1, &fpci, nullptr, &factor_pipe_);

    VkComputePipelineCreateInfo apci{};
    apci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    apci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, amod, "main", nullptr};
    apci.layout = apply_layout_;
    vkCreateComputePipelines(ctx.device, ctx.pipeline_cache, 1, &apci, nullptr, &apply_pipe_);

    vkDestroyShaderModule(ctx.device, fmod, nullptr);
    vkDestroyShaderModule(ctx.device, amod, nullptr);
    return ready();
}

void ShadowCompute::shutdown(const Context &ctx, BindlessDescriptors &desc) {
    if (factor_slot_ != 0) { desc.free_texture(factor_slot_, 0); factor_slot_ = 0; }
    if (factor_view_ != VK_NULL_HANDLE) { vkDestroyImageView(ctx.device, factor_view_, nullptr); factor_view_ = VK_NULL_HANDLE; }
    if (factor_image_ != VK_NULL_HANDLE) { vmaDestroyImage(ctx.allocator, factor_image_, factor_alloc_); factor_image_ = VK_NULL_HANDLE; }
    if (factor_pipe_ != VK_NULL_HANDLE) { vkDestroyPipeline(ctx.device, factor_pipe_, nullptr); factor_pipe_ = VK_NULL_HANDLE; }
    if (apply_pipe_ != VK_NULL_HANDLE) { vkDestroyPipeline(ctx.device, apply_pipe_, nullptr); apply_pipe_ = VK_NULL_HANDLE; }
    if (factor_layout_ != VK_NULL_HANDLE) { vkDestroyPipelineLayout(ctx.device, factor_layout_, nullptr); factor_layout_ = VK_NULL_HANDLE; }
    if (apply_layout_ != VK_NULL_HANDLE) { vkDestroyPipelineLayout(ctx.device, apply_layout_, nullptr); apply_layout_ = VK_NULL_HANDLE; }
    if (storage_pool_ != VK_NULL_HANDLE) { vkDestroyDescriptorPool(ctx.device, storage_pool_, nullptr); storage_pool_ = VK_NULL_HANDLE; }
    if (storage_layout_ != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(ctx.device, storage_layout_, nullptr); storage_layout_ = VK_NULL_HANDLE; }
}

bool ShadowCompute::ensure_factor_image(const Context &ctx, BindlessDescriptors &desc, std::uint32_t w, std::uint32_t h) {
    if (factor_image_ != VK_NULL_HANDLE && factor_w_ == w && factor_h_ == h) return true;
    ctx.wait_idle();
    if (factor_slot_ != 0) { desc.free_texture(factor_slot_, 0); factor_slot_ = 0; }
    if (factor_view_ != VK_NULL_HANDLE) { vkDestroyImageView(ctx.device, factor_view_, nullptr); factor_view_ = VK_NULL_HANDLE; }
    if (factor_image_ != VK_NULL_HANDLE) { vmaDestroyImage(ctx.allocator, factor_image_, factor_alloc_); factor_image_ = VK_NULL_HANDLE; }

    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R8G8B8A8_UNORM;
    ici.extent = {w, h, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    VmaAllocationCreateInfo vaci{};
    vaci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (vmaCreateImage(ctx.allocator, &ici, &vaci, &factor_image_, &factor_alloc_, nullptr) != VK_SUCCESS) return false;

    VkImageViewCreateInfo ivci{};
    ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivci.image = factor_image_;
    ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ivci.format = VK_FORMAT_R8G8B8A8_UNORM;
    ivci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(ctx.device, &ivci, nullptr, &factor_view_) != VK_SUCCESS) return false;

    factor_w_ = w;
    factor_h_ = h;
    factor_slot_ = desc.allocate_texture(ctx, factor_view_);

    VkDescriptorImageInfo dii{VK_NULL_HANDLE, factor_view_, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet wds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, factor_storage_set_, 0, 0, 1,
                             VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &dii, nullptr, nullptr};
    vkUpdateDescriptorSets(ctx.device, 1, &wds, 0, nullptr);
    return true;
}

void ShadowCompute::dispatch_factor_and_apply(VkCommandBuffer cmd, const Context &ctx, const BindlessDescriptors &desc,
                                              VkDeviceAddress params_bda, VkImage scene_color, VkImageView scene_color_view,
                                              std::uint32_t scene_depth_slot, std::uint32_t scene_w, std::uint32_t scene_h,
                                              int down) {
    if (!ready() || factor_image_ == VK_NULL_HANDLE) return;

    const VkDescriptorSet scene_set = scene_storage_sets_[scene_set_cursor_];
    scene_set_cursor_ = (scene_set_cursor_ + 1u) % kSceneSetCount;
    VkDescriptorImageInfo sii{VK_NULL_HANDLE, scene_color_view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet swds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, scene_set, 0, 0, 1,
                              VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &sii, nullptr, nullptr};
    vkUpdateDescriptorSets(ctx.device, 1, &swds, 0, nullptr);

    Context::cmd_transition_image(cmd, factor_image_, VK_IMAGE_ASPECT_COLOR_BIT,
                                  VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                  VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    std::array<VkDescriptorSet, 2> fsets = {desc.set(), factor_storage_set_};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, factor_pipe_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, factor_layout_, 0, 2, fsets.data(), 0, nullptr);
    vkCmdPushConstants(cmd, factor_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params_bda), &params_bda);
    vkCmdDispatch(cmd, (factor_w_ + 7u) / 8u, (factor_h_ + 7u) / 8u, 1u);

    Context::cmd_transition_image(cmd, factor_image_, VK_IMAGE_ASPECT_COLOR_BIT,
                                  VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    Context::cmd_transition_image(cmd, scene_color, VK_IMAGE_ASPECT_COLOR_BIT,
                                  VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                  VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    std::array<VkDescriptorSet, 2> asets = {desc.set(), scene_set};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, apply_pipe_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, apply_layout_, 0, 2, asets.data(), 0, nullptr);
    ShadowApplyPush ap{factor_slot_, scene_depth_slot, scene_w, scene_h, 1, std::max(down, 1)};
    vkCmdPushConstants(cmd, apply_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ap), &ap);
    vkCmdDispatch(cmd, (scene_w + 7u) / 8u, (scene_h + 7u) / 8u, 1u);

    Context::cmd_transition_image(cmd, scene_color, VK_IMAGE_ASPECT_COLOR_BIT,
                                  VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                  VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
}

} // namespace mhf::psp::vk
