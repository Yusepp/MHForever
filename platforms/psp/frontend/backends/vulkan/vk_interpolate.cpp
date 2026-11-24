#include "vk_interpolate.hpp"
#include "vk_shaders_spv.hpp"
#include "../gpu_common.hpp"

namespace mhf::psp::vk {

bool GpuInterpolator::init(const Context &ctx) {
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PlacePush)};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 0, nullptr, 1, &pcr};
    if (vkCreatePipelineLayout(ctx.device, &plci, nullptr, &layout_) != VK_SUCCESS) return false;

    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
                                  shaders::kPlaceV2CompSpv.size() * sizeof(std::uint32_t),
                                  shaders::kPlaceV2CompSpv.data()};
    VkShaderModule mod = VK_NULL_HANDLE;
    if (vkCreateShaderModule(ctx.device, &smci, nullptr, &mod) != VK_SUCCESS) return false;

    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, mod, "main", nullptr};
    cpci.layout = layout_;
    const VkResult r = vkCreateComputePipelines(ctx.device, ctx.pipeline_cache, 1, &cpci, nullptr, &pipe_);
    vkDestroyShaderModule(ctx.device, mod, nullptr);
    if (r != VK_SUCCESS) return false;

    // Double-buffered 2x output vertex capacity (one per frame-in-flight) so consecutive 120/180 Hz
    // subframes never overwrite each other's vertex stream and can overlap asynchronously on the GPU.
    const std::size_t out_bytes = static_cast<std::size_t>(max_in_vertices_) * 2u * sizeof(gpu_common::GpuVertex);
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = out_bytes;
    bci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo vaci{};
    vaci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    for (int i = 0; i < kOutSlots; ++i) {
        if (vmaCreateBuffer(ctx.allocator, &bci, &vaci, &out_buf_[i], &out_alloc_[i], nullptr) != VK_SUCCESS) {
            shutdown(ctx);
            return false;
        }
        VkBufferDeviceAddressInfo bdai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, out_buf_[i]};
        out_addr_[i] = vkGetBufferDeviceAddress(ctx.device, &bdai);
        if (out_addr_[i] == 0) {
            shutdown(ctx);
            return false;
        }
    }
    return true;
}

void GpuInterpolator::shutdown(const Context &ctx) {
    for (int i = 0; i < kOutSlots; ++i) {
        if (out_buf_[i] != VK_NULL_HANDLE) {
            vmaDestroyBuffer(ctx.allocator, out_buf_[i], out_alloc_[i]);
            out_buf_[i] = VK_NULL_HANDLE;
            out_alloc_[i] = VK_NULL_HANDLE;
            out_addr_[i] = 0;
        }
    }
    if (pipe_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(ctx.device, pipe_, nullptr);
        pipe_ = VK_NULL_HANDLE;
    }
    if (layout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(ctx.device, layout_, nullptr);
        layout_ = VK_NULL_HANDLE;
    }
}

void GpuInterpolator::dispatch_place_v2(VkCommandBuffer cmd, int slot, VkDeviceAddress raw_verts_old_bda,
                                        VkDeviceAddress raw_verts_new_bda, VkDeviceAddress transforms_bda,
                                        std::uint32_t vertex_count, float phase) {
    if (!ready() || vertex_count == 0) return;
    const int s = slot & 1;
    const std::uint32_t count = vertex_count <= max_in_vertices_ ? vertex_count : max_in_vertices_;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_);
    PlacePush push{raw_verts_old_bda, raw_verts_new_bda ? raw_verts_new_bda : raw_verts_old_bda,
                   transforms_bda, out_addr_[s], count, phase};
    vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    const std::uint32_t tri_count = count / 3u;
    vkCmdDispatch(cmd, (tri_count + 63u) / 64u, 1u, 1u);

    VkBufferMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = out_buf_[s];
    barrier.offset = 0;
    barrier.size = static_cast<VkDeviceSize>(count) * 2u * sizeof(gpu_common::GpuVertex);

    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.bufferMemoryBarrierCount = 1;
    dep.pBufferMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &dep);
}

} // namespace mhf::psp::vk
