#include "vk_descriptors.hpp"

#include <algorithm>
#include <array>
#include <cstdio>

namespace mhf::psp::vk {

std::uint32_t BindlessDescriptors::sampler_index(bool linear, bool mips, bool clamp_s, bool clamp_t, int anisotropy) noexcept {
    std::uint32_t aniso_tier = 0u;
    if (mips) {
        if (anisotropy >= 16) aniso_tier = 3u;
        else if (anisotropy >= 8) aniso_tier = 2u;
        else if (anisotropy >= 4) aniso_tier = 1u;
    }
    return (linear ? 1u : 0u) | (mips ? 2u : 0u) | (clamp_s ? 4u : 0u) | (clamp_t ? 8u : 0u) | (aniso_tier << 4u);
}

bool BindlessDescriptors::init(const Context &ctx) {
    // 1. Pre-create all 64 immutable samplers.
    samplers_.resize(kSamplerCount, VK_NULL_HANDLE);
    constexpr float kAnisoLevels[4] = {1.0f, 4.0f, 8.0f, 16.0f};
    for (std::uint32_t i = 0; i < kSamplerCount; ++i) {
        const bool linear = (i & 1u) != 0u;
        const bool mips = (i & 2u) != 0u;
        const bool clamp_s = (i & 4u) != 0u;
        const bool clamp_t = (i & 8u) != 0u;
        const float aniso = std::min(kAnisoLevels[(i >> 4u) & 3u], ctx.caps.max_anisotropy);

        VkSamplerCreateInfo sci{};
        sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter = linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        sci.minFilter = linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        sci.mipmapMode = linear ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = clamp_s ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE : VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeV = clamp_t ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE : VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.anisotropyEnable = (mips && aniso > 1.0f) ? VK_TRUE : VK_FALSE;
        sci.maxAnisotropy = aniso;
        sci.maxLod = mips ? VK_LOD_CLAMP_NONE : 0.0f;
        if (vkCreateSampler(ctx.device, &sci, nullptr, &samplers_[i]) != VK_SUCCESS) {
            std::fprintf(stderr, "[vk] failed to create sampler %u\n", i);
            shutdown(ctx);
            return false;
        }
    }

    // 2. Global Descriptor Set Layout (Set 0).
    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    bindings[0].descriptorCount = kMaxTextures;
    bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    bindings[1].descriptorCount = kSamplerCount;
    bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].pImmutableSamplers = samplers_.data();

    std::array<VkDescriptorBindingFlags, 2> bflags = {
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
        0u,
    };
    VkDescriptorSetLayoutBindingFlagsCreateInfo bfci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO, nullptr,
        static_cast<std::uint32_t>(bflags.size()), bflags.data()};

    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.pNext = &bfci;
    dslci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    dslci.bindingCount = static_cast<std::uint32_t>(bindings.size());
    dslci.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(ctx.device, &dslci, nullptr, &layout_) != VK_SUCCESS) {
        std::fprintf(stderr, "[vk] failed to create bindless descriptor set layout\n");
        shutdown(ctx);
        return false;
    }

    // 3. Descriptor Pool & Set.
    std::array<VkDescriptorPoolSize, 2> sizes = {{
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kMaxTextures},
        {VK_DESCRIPTOR_TYPE_SAMPLER, kSamplerCount},
    }};
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    dpci.maxSets = 1;
    dpci.poolSizeCount = static_cast<std::uint32_t>(sizes.size());
    dpci.pPoolSizes = sizes.data();
    if (vkCreateDescriptorPool(ctx.device, &dpci, nullptr, &pool_) != VK_SUCCESS) {
        shutdown(ctx);
        return false;
    }

    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, pool_, 1, &layout_};
    if (vkAllocateDescriptorSets(ctx.device, &dsai, &set_) != VK_SUCCESS) {
        shutdown(ctx);
        return false;
    }

    // 4. Create 1x1 fallback white image at slot 0.
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R8G8B8A8_UNORM;
    ici.extent = {1, 1, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VmaAllocationCreateInfo vaci{};
    vaci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (vmaCreateImage(ctx.allocator, &ici, &vaci, &fallback_image_, &fallback_alloc_, nullptr) != VK_SUCCESS) {
        shutdown(ctx);
        return false;
    }
    VkImageViewCreateInfo ivci{};
    ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivci.image = fallback_image_;
    ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ivci.format = VK_FORMAT_R8G8B8A8_UNORM;
    ivci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(ctx.device, &ivci, nullptr, &fallback_view_) != VK_SUCCESS) {
        shutdown(ctx);
        return false;
    }
    update_texture(ctx, 0u, fallback_view_);
    return true;
}

void BindlessDescriptors::shutdown(const Context &ctx) {
    if (fallback_view_ != VK_NULL_HANDLE) {
        vkDestroyImageView(ctx.device, fallback_view_, nullptr);
        fallback_view_ = VK_NULL_HANDLE;
    }
    if (fallback_image_ != VK_NULL_HANDLE) {
        vmaDestroyImage(ctx.allocator, fallback_image_, fallback_alloc_);
        fallback_image_ = VK_NULL_HANDLE;
        fallback_alloc_ = VK_NULL_HANDLE;
    }
    if (pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(ctx.device, pool_, nullptr);
        pool_ = VK_NULL_HANDLE;
        set_ = VK_NULL_HANDLE;
    }
    if (layout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(ctx.device, layout_, nullptr);
        layout_ = VK_NULL_HANDLE;
    }
    for (VkSampler s : samplers_) {
        if (s != VK_NULL_HANDLE) vkDestroySampler(ctx.device, s, nullptr);
    }
    samplers_.clear();
}

std::uint32_t BindlessDescriptors::allocate_texture(const Context &ctx, VkImageView view) {
    std::uint32_t slot = 0;
    if (!free_slots_.empty()) {
        slot = free_slots_.back();
        free_slots_.pop_back();
    } else if (next_slot_ < kMaxTextures) {
        slot = next_slot_++;
    } else {
        return 0u;
    }
    update_texture(ctx, slot, view);
    return slot;
}

void BindlessDescriptors::update_texture(const Context &ctx, std::uint32_t slot, VkImageView view) {
    VkDescriptorImageInfo info{};
    info.imageView = view;
    info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set_;
    write.dstBinding = 0;
    write.dstArrayElement = slot;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    write.pImageInfo = &info;
    vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);
}

void BindlessDescriptors::free_texture(std::uint32_t slot, std::uint64_t retire_value) {
    if (slot == 0u) return;
    pending_frees_.push_back({slot, retire_value});
}

void BindlessDescriptors::collect_retired(std::uint64_t completed_timeline) {
    while (!pending_frees_.empty() && pending_frees_.front().retire_value <= completed_timeline) {
        free_slots_.push_back(pending_frees_.front().slot);
        pending_frees_.pop_front();
    }
}

} // namespace mhf::psp::vk
