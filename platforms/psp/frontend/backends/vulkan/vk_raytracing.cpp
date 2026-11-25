#include "vk_raytracing.hpp"
#include "vk_shaders_spv.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace mhf::psp::vk {

namespace {

std::uint64_t hash_rt_batch(const RtVertex *verts, std::uint32_t count) noexcept {
    std::uint64_t h = static_cast<std::uint64_t>(count) * 0x9E3779B97F4A7C15ull;
    if (count == 0u) return h;
    // Sample up to 16 vertices quantized to ~1 cm so identical static map chunks always hash identically.
    const std::uint32_t step = std::max<std::uint32_t>(1u, count / 16u);
    for (std::uint32_t i = 0u; i < count; i += step) {
        const auto qx = static_cast<std::uint32_t>(static_cast<std::int32_t>(std::lround(verts[i].x)));
        const auto qy = static_cast<std::uint32_t>(static_cast<std::int32_t>(std::lround(verts[i].y)));
        const auto qz = static_cast<std::uint32_t>(static_cast<std::int32_t>(std::lround(verts[i].z)));
        h ^= (static_cast<std::uint64_t>(qx) << 21u) ^
             (static_cast<std::uint64_t>(qy) << 42u) ^
             static_cast<std::uint64_t>(qz);
        h *= 0xBF58476D1CE4E5B9ull;
    }
    return h;
}

void invert_rigid_col(const float v[16], float out[16]) noexcept {
    out[0] = v[0]; out[1] = v[4]; out[2]  = v[8];  out[3]  = 0.0f;
    out[4] = v[1]; out[5] = v[5]; out[6]  = v[9];  out[7]  = 0.0f;
    out[8] = v[2]; out[9] = v[6]; out[10] = v[10]; out[11] = 0.0f;
    const float tx = v[12], ty = v[13], tz = v[14];
    out[12] = -(v[0] * tx + v[1] * ty + v[2] * tz);
    out[13] = -(v[4] * tx + v[5] * ty + v[6] * tz);
    out[14] = -(v[8] * tx + v[9] * ty + v[10] * tz);
    out[15] = 1.0f;
}

} // namespace

void RayTracedShadows::GpuBuf::destroy(const Context &ctx) noexcept {
    if (buf != VK_NULL_HANDLE) {
        vmaDestroyBuffer(ctx.allocator, buf, alloc);
        buf = VK_NULL_HANDLE;
        alloc = VK_NULL_HANDLE;
        addr = 0;
        mapped = nullptr;
        size = 0;
    }
}

void RayTracedShadows::BlasSlot::destroy(const Context &ctx) noexcept {
    if (as != VK_NULL_HANDLE && vkDestroyAccelerationStructureKHR) {
        vkDestroyAccelerationStructureKHR(ctx.device, as, nullptr);
        as = VK_NULL_HANDLE;
    }
    as_buf.destroy(ctx);
    vtx_buf.destroy(ctx);
    device_addr = 0;
    cached_gen = 0;
    last_used_seq = 0;
    cached_tris = 0;
}

bool RayTracedShadows::init(const Context &ctx, VkDescriptorSetLayout bindless_layout) {
    if (!ctx.caps.ray_query || !vkCreateAccelerationStructureKHR || !vkCmdBuildAccelerationStructuresKHR ||
        !vkGetAccelerationStructureBuildSizesKHR || !vkGetAccelerationStructureDeviceAddressKHR) {
        return false;
    }

    // Set 1: binding 0 = TLAS, binding 1 = R8_UNORM storage image.
    VkDescriptorSetLayoutBinding bindings[2]{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 2, bindings};
    if (vkCreateDescriptorSetLayout(ctx.device, &dslci, nullptr, &rt_set_layout_) != VK_SUCCESS) return false;

    VkDescriptorPoolSize psizes[2]{
        {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, static_cast<std::uint32_t>(kRingSlots)},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, static_cast<std::uint32_t>(kRingSlots)},
    };
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0,
                                    static_cast<std::uint32_t>(kRingSlots), 2, psizes};
    if (vkCreateDescriptorPool(ctx.device, &dpci, nullptr, &rt_pool_) != VK_SUCCESS) return false;

    VkDescriptorSetLayout layouts[kRingSlots];
    for (int i = 0; i < kRingSlots; ++i) layouts[i] = rt_set_layout_;
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, rt_pool_,
                                     static_cast<std::uint32_t>(kRingSlots), layouts};
    if (vkAllocateDescriptorSets(ctx.device, &dsai, rt_sets_) != VK_SUCCESS) return false;

    VkDescriptorSetLayout set_layouts[2] = {bindless_layout, rt_set_layout_};
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkDeviceAddress)};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 2, set_layouts, 1, &pcr};
    if (vkCreatePipelineLayout(ctx.device, &plci, nullptr, &pipe_layout_) != VK_SUCCESS) return false;

    VkShaderModule mod = ctx.create_shader_module(shaders::kRtShadowsCompSpv);
    if (!mod) return false;

    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                  VK_SHADER_STAGE_COMPUTE_BIT, mod, "main", nullptr};
    cpci.layout = pipe_layout_;
    const VkResult res = vkCreateComputePipelines(ctx.device, ctx.pipeline_cache, 1, &cpci, nullptr, &pipe_);
    vkDestroyShaderModule(ctx.device, mod, nullptr);
    return res == VK_SUCCESS;
}

void RayTracedShadows::shutdown(const Context &ctx, BindlessDescriptors &desc) {
    if (pipe_ != VK_NULL_HANDLE) { vkDestroyPipeline(ctx.device, pipe_, nullptr); pipe_ = VK_NULL_HANDLE; }
    if (pipe_layout_ != VK_NULL_HANDLE) { vkDestroyPipelineLayout(ctx.device, pipe_layout_, nullptr); pipe_layout_ = VK_NULL_HANDLE; }
    if (rt_pool_ != VK_NULL_HANDLE) { vkDestroyDescriptorPool(ctx.device, rt_pool_, nullptr); rt_pool_ = VK_NULL_HANDLE; }
    for (int i = 0; i < kRingSlots; ++i) {
        rt_sets_[i] = VK_NULL_HANDLE;
        if (tlas_[i] != VK_NULL_HANDLE && vkDestroyAccelerationStructureKHR) {
            vkDestroyAccelerationStructureKHR(ctx.device, tlas_[i], nullptr);
            tlas_[i] = VK_NULL_HANDLE;
        }
        tlas_buf_[i].destroy(ctx);
        inst_buf_[i].destroy(ctx);
        scratch_buf_[i].destroy(ctx);
        static_blas_[i].destroy(ctx);
        dyn_blas_[i].destroy(ctx);
        tlas_valid_[i] = false;
        rt_desc_dirty_[i] = true;
    }
    if (rt_set_layout_ != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(ctx.device, rt_set_layout_, nullptr); rt_set_layout_ = VK_NULL_HANDLE; }
    active_static_slot_ = -1;
    reset_zone_cache();
    if (out_slot_ != 0u) { desc.free_texture(out_slot_, UINT64_MAX); out_slot_ = 0u; }
    if (out_view_ != VK_NULL_HANDLE) { vkDestroyImageView(ctx.device, out_view_, nullptr); out_view_ = VK_NULL_HANDLE; }
    if (out_img_ != VK_NULL_HANDLE) { vmaDestroyImage(ctx.allocator, out_img_, out_alloc_); out_img_ = VK_NULL_HANDLE; out_alloc_ = VK_NULL_HANDLE; }
    out_w_ = out_h_ = 0u;
    out_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
}

bool RayTracedShadows::ensure_buf(const Context &ctx, GpuBuf &b, VkDeviceSize needed,
                                  VkBufferUsageFlags usage, bool host_visible, VkDeviceSize min_alloc) {
    if (needed < min_alloc) needed = min_alloc;
    if (b.buf != VK_NULL_HANDLE && b.size >= needed) return true;
    ctx.wait_idle();
    b.destroy(ctx);
    const VkDeviceSize alloc_sz = std::max<VkDeviceSize>(needed, (needed * 3u) / 2u);
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, alloc_sz,
                           usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
    VmaAllocationCreateInfo vaci{};
    vaci.usage = host_visible ? VMA_MEMORY_USAGE_AUTO : VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (host_visible) {
        vaci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }
    VmaAllocationInfo info{};
    if (vmaCreateBuffer(ctx.allocator, &bci, &vaci, &b.buf, &b.alloc, &info) != VK_SUCCESS) return false;
    b.size = alloc_sz;
    b.mapped = info.pMappedData;
    VkBufferDeviceAddressInfo bdai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, b.buf};
    b.addr = vkGetBufferDeviceAddress(ctx.device, &bdai);
    return b.addr != 0;
}

bool RayTracedShadows::build_blas(VkCommandBuffer cmd, const Context &ctx, BlasSlot &slot, GpuBuf &scratch,
                                  const std::vector<RtVertex> &verts, VkBuildAccelerationStructureFlagsKHR flags,
                                  VkDeviceSize min_vtx_alloc, VkDeviceSize min_as_alloc) {
    const std::uint32_t tri_count = static_cast<std::uint32_t>(verts.size() / 3u);
    if (tri_count == 0u) return false;
    const VkDeviceSize vbytes = static_cast<VkDeviceSize>(tri_count) * 3u * sizeof(RtVertex);
    if (!ensure_buf(ctx, slot.vtx_buf, vbytes, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                    true, min_vtx_alloc)) {
        return false;
    }
    std::memcpy(slot.vtx_buf.mapped, verts.data(), vbytes);

    VkAccelerationStructureGeometryTrianglesDataKHR tris{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR, nullptr};
    tris.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    tris.vertexData.deviceAddress = slot.vtx_buf.addr;
    tris.vertexStride = sizeof(RtVertex);
    tris.maxVertex = tri_count * 3u - 1u;
    tris.indexType = VK_INDEX_TYPE_NONE_KHR;

    VkAccelerationStructureGeometryKHR geom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR, nullptr};
    geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geom.geometry.triangles = tris;

    VkAccelerationStructureBuildGeometryInfoKHR bgi{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR, nullptr};
    bgi.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    bgi.flags = flags;
    bgi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    bgi.geometryCount = 1;
    bgi.pGeometries = &geom;

    VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR, nullptr};
    vkGetAccelerationStructureBuildSizesKHR(ctx.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                            &bgi, &tri_count, &sizes);

    if (slot.as == VK_NULL_HANDLE || slot.as_buf.size < sizes.accelerationStructureSize) {
        ctx.wait_idle();
        if (slot.as != VK_NULL_HANDLE) {
            vkDestroyAccelerationStructureKHR(ctx.device, slot.as, nullptr);
            slot.as = VK_NULL_HANDLE;
        }
        const VkDeviceSize as_alloc = std::max(sizes.accelerationStructureSize, min_as_alloc);
        if (!ensure_buf(ctx, slot.as_buf, as_alloc,
                        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, false, as_alloc)) {
            return false;
        }
        VkAccelerationStructureCreateInfoKHR asci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR, nullptr};
        asci.buffer = slot.as_buf.buf;
        asci.size = slot.as_buf.size;
        asci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        if (vkCreateAccelerationStructureKHR(ctx.device, &asci, nullptr, &slot.as) != VK_SUCCESS) return false;
        VkAccelerationStructureDeviceAddressInfoKHR adai{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR, nullptr, slot.as};
        slot.device_addr = vkGetAccelerationStructureDeviceAddressKHR(ctx.device, &adai);
    }

    if (!ensure_buf(ctx, scratch, sizes.buildScratchSize + 256u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false,
                    4u * 1024u * 1024u)) {
        return false;
    }
    const VkDeviceAddress aligned_scratch = (scratch.addr + 255u) & ~VkDeviceAddress(255u);
    bgi.dstAccelerationStructure = slot.as;
    bgi.scratchData.deviceAddress = aligned_scratch;

    VkAccelerationStructureBuildRangeInfoKHR range{tri_count, 0u, 0u, 0u};
    const VkAccelerationStructureBuildRangeInfoKHR *prange = &range;
    vkCmdBuildAccelerationStructuresKHR(cmd, 1, &bgi, &prange);

    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, nullptr,
                        VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                        VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR};
    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO, nullptr, 0, 1, &mb, 0, nullptr, 0, nullptr};
    vkCmdPipelineBarrier2(cmd, &dep);
    slot.cached_tris = tri_count;
    return true;
}

bool RayTracedShadows::update_bvh(VkCommandBuffer cmd, const Context &ctx, int frame_slot,
                                  const std::vector<RtVertex> &static_verts,
                                  const std::vector<RtStaticBatch> &static_batches,
                                  const std::vector<RtVertex> &dyn_verts,
                                  const float view_mat[16]) {
    if (!ready()) return false;
    const int fs = ((frame_slot % kRingSlots) + kRingSlots) % kRingSlots;
    last_slot_ = fs;
    ++use_seq_;

    // 0. Detect zone transitions (loading fade or large camera teleport) and accumulate unseen static chunks
    //    persistently per zone so PSP CPU frustum culling NEVER pops off-screen building/tree shadows!
    if (view_mat != nullptr) {
        float inv_v[16]{};
        invert_rigid_col(view_mat, inv_v);
        const float cx = inv_v[12], cy = inv_v[13], cz = inv_v[14];
        if (has_last_cam_) {
            const float dx = cx - last_cam_world_[0];
            const float dy = cy - last_cam_world_[1];
            const float dz = cz - last_cam_world_[2];
            if (dx * dx + dy * dy + dz * dz > 6000.0f * 6000.0f) {
                reset_zone_cache();
            }
        }
        last_cam_world_[0] = cx;
        last_cam_world_[1] = cy;
        last_cam_world_[2] = cz;
        has_last_cam_ = true;
    }

    if (static_verts.size() < 3u) {
        if (++empty_static_frames_ >= 6u) {
            reset_zone_cache();
        }
    } else {
        empty_static_frames_ = 0u;
        constexpr std::size_t kMaxZoneStaticVerts = 300000u;
        bool added_any = false;
        if (!static_batches.empty()) {
            for (const RtStaticBatch &b : static_batches) {
                if (b.v_count < 3u || b.v_offset + b.v_count > static_verts.size()) continue;
                const std::uint32_t tri_verts = (b.v_count / 3u) * 3u;
                const std::uint64_t ch = hash_rt_batch(static_verts.data() + b.v_offset, tri_verts);
                if (zone_chunks_.find(ch) == zone_chunks_.end()) {
                    if (zone_static_verts_.size() + tri_verts <= kMaxZoneStaticVerts) {
                        zone_chunks_.insert(ch);
                        zone_static_verts_.insert(zone_static_verts_.end(),
                                                  static_verts.begin() + b.v_offset,
                                                  static_verts.begin() + b.v_offset + tri_verts);
                        added_any = true;
                    }
                }
            }
        } else {
            const std::uint32_t tri_verts = static_cast<std::uint32_t>((static_verts.size() / 3u) * 3u);
            const std::uint64_t ch = hash_rt_batch(static_verts.data(), tri_verts);
            if (zone_chunks_.find(ch) == zone_chunks_.end() &&
                zone_static_verts_.size() + tri_verts <= kMaxZoneStaticVerts) {
                zone_chunks_.insert(ch);
                zone_static_verts_.insert(zone_static_verts_.end(), static_verts.begin(),
                                          static_verts.begin() + tri_verts);
                added_any = true;
            }
        }
        if (added_any) ++zone_static_gen_;
    }

    // 1. Static zone BLAS: rebuild into a guaranteed GPU-idle slot ONLY when a brand-new chunk enters the zone!
    bool have_static = false;
    VkDeviceAddress static_addr = 0;
    if (zone_static_verts_.size() >= 3u) {
        if (active_static_slot_ >= 0 &&
            static_blas_[active_static_slot_].cached_gen == zone_static_gen_ &&
            static_blas_[active_static_slot_].as != VK_NULL_HANDLE) {
            static_blas_[active_static_slot_].last_used_seq = use_seq_;
            static_addr = static_blas_[active_static_slot_].device_addr;
            have_static = true;
        } else {
            // Find a slot guaranteed NOT referenced by any in-flight TLAS (use_seq_ - last_used_seq >= kRingSlots).
            int free_slot = -1;
            for (int s = 0; s < kRingSlots; ++s) {
                if (use_seq_ >= static_blas_[s].last_used_seq + static_cast<std::uint64_t>(kRingSlots)) {
                    free_slot = s;
                    break;
                }
            }
            if (free_slot >= 0) {
                if (build_blas(cmd, ctx, static_blas_[free_slot], scratch_buf_[fs], zone_static_verts_,
                               VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
                               4u * 1024u * 1024u, 8u * 1024u * 1024u)) {
                    static_blas_[free_slot].cached_gen = zone_static_gen_;
                    static_blas_[free_slot].last_used_seq = use_seq_;
                    active_static_slot_ = free_slot;
                    static_addr = static_blas_[free_slot].device_addr;
                    have_static = true;
                }
            } else if (active_static_slot_ >= 0 && static_blas_[active_static_slot_].as != VK_NULL_HANDLE) {
                // All slots in-flight this subframe: safely reuse previous static BLAS for 1 subframe.
                static_blas_[active_static_slot_].last_used_seq = use_seq_;
                static_addr = static_blas_[active_static_slot_].device_addr;
                have_static = true;
            }
        }
    }

    // 2. Dynamic character BLAS (hunter + monsters + small rigid weapons/props): per-frame-in-flight slot `fs`!
    bool have_dyn = false;
    if (dyn_verts.size() >= 3u) {
        have_dyn = build_blas(cmd, ctx, dyn_blas_[fs], scratch_buf_[fs], dyn_verts,
                              VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR,
                              1u * 1024u * 1024u, 2u * 1024u * 1024u);
    }

    if (!have_static && !have_dyn) {
        tlas_valid_[fs] = false;
        return false;
    }

    // 3. Per-frame-in-flight 2-Instance TLAS (Static zone + Dynamic characters).
    VkAccelerationStructureInstanceKHR insts[2]{};
    std::uint32_t icount = 0u;
    auto push_inst = [&](VkDeviceAddress blas_addr, std::uint32_t custom_idx) {
        VkAccelerationStructureInstanceKHR &in = insts[icount++];
        in.transform.matrix[0][0] = 1.0f;
        in.transform.matrix[1][1] = 1.0f;
        in.transform.matrix[2][2] = 1.0f;
        in.instanceCustomIndex = custom_idx;
        in.mask = 0xFFu;
        in.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        in.accelerationStructureReference = blas_addr;
    };
    if (have_static) push_inst(static_addr, 0u);
    if (have_dyn) push_inst(dyn_blas_[fs].device_addr, 1u);

    if (!ensure_buf(ctx, inst_buf_[fs], sizeof(insts),
                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, true)) {
        return false;
    }
    std::memcpy(inst_buf_[fs].mapped, insts, icount * sizeof(VkAccelerationStructureInstanceKHR));

    VkAccelerationStructureGeometryInstancesDataKHR idata{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR, nullptr};
    idata.arrayOfPointers = VK_FALSE;
    idata.data.deviceAddress = inst_buf_[fs].addr;

    VkAccelerationStructureGeometryKHR tgeom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR, nullptr};
    tgeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    tgeom.geometry.instances = idata;

    VkAccelerationStructureBuildGeometryInfoKHR tbgi{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR, nullptr};
    tbgi.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    tbgi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    tbgi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    tbgi.geometryCount = 1;
    tbgi.pGeometries = &tgeom;

    VkAccelerationStructureBuildSizesInfoKHR tsizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR, nullptr};
    vkGetAccelerationStructureBuildSizesKHR(ctx.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                            &tbgi, &icount, &tsizes);

    if (tlas_[fs] == VK_NULL_HANDLE || tlas_buf_[fs].size < tsizes.accelerationStructureSize) {
        ctx.wait_idle();
        if (tlas_[fs] != VK_NULL_HANDLE) {
            vkDestroyAccelerationStructureKHR(ctx.device, tlas_[fs], nullptr);
            tlas_[fs] = VK_NULL_HANDLE;
        }
        if (!ensure_buf(ctx, tlas_buf_[fs], tsizes.accelerationStructureSize,
                        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, false)) {
            return false;
        }
        VkAccelerationStructureCreateInfoKHR asci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR, nullptr};
        asci.buffer = tlas_buf_[fs].buf;
        asci.size = tsizes.accelerationStructureSize;
        asci.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        if (vkCreateAccelerationStructureKHR(ctx.device, &asci, nullptr, &tlas_[fs]) != VK_SUCCESS) return false;
        rt_desc_dirty_[fs] = true;
    }

    if (!ensure_buf(ctx, scratch_buf_[fs], tsizes.buildScratchSize + 256u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false)) {
        return false;
    }
    tbgi.dstAccelerationStructure = tlas_[fs];
    tbgi.scratchData.deviceAddress = (scratch_buf_[fs].addr + 255u) & ~VkDeviceAddress(255u);

    VkAccelerationStructureBuildRangeInfoKHR trange{icount, 0u, 0u, 0u};
    const VkAccelerationStructureBuildRangeInfoKHR *ptrange = &trange;
    vkCmdBuildAccelerationStructuresKHR(cmd, 1, &tbgi, &ptrange);

    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, nullptr,
                        VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR};
    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO, nullptr, 0, 1, &mb, 0, nullptr, 0, nullptr};
    vkCmdPipelineBarrier2(cmd, &dep);
    tlas_valid_[fs] = true;
    return true;
}

bool RayTracedShadows::ensure_output(const Context &ctx, BindlessDescriptors &desc, std::uint32_t hw, std::uint32_t hh) {
    if (out_img_ != VK_NULL_HANDLE && out_w_ == hw && out_h_ == hh) return true;
    ctx.wait_idle();
    if (out_slot_ != 0u) { desc.free_texture(out_slot_, UINT64_MAX); out_slot_ = 0u; }
    if (out_view_ != VK_NULL_HANDLE) { vkDestroyImageView(ctx.device, out_view_, nullptr); out_view_ = VK_NULL_HANDLE; }
    if (out_img_ != VK_NULL_HANDLE) { vmaDestroyImage(ctx.allocator, out_img_, out_alloc_); out_img_ = VK_NULL_HANDLE; out_alloc_ = VK_NULL_HANDLE; }

    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, nullptr};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R8G8B8A8_UNORM;
    ici.extent = {hw, hh, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    VmaAllocationCreateInfo vaci{};
    vaci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (vmaCreateImage(ctx.allocator, &ici, &vaci, &out_img_, &out_alloc_, nullptr) != VK_SUCCESS) return false;

    VkImageViewCreateInfo ivci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, nullptr};
    ivci.image = out_img_;
    ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ivci.format = VK_FORMAT_R8G8B8A8_UNORM;
    ivci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(ctx.device, &ivci, nullptr, &out_view_) != VK_SUCCESS) return false;

    out_slot_ = desc.allocate_texture(ctx, out_view_);
    out_w_ = hw;
    out_h_ = hh;
    out_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    for (int i = 0; i < kRingSlots; ++i) rt_desc_dirty_[i] = true;
    return out_slot_ != 0u;
}

bool RayTracedShadows::trace(VkCommandBuffer cmd, const Context &ctx, BindlessDescriptors &desc,
                             VkDescriptorSet bindless_set, int frame_slot, std::uint32_t full_w, std::uint32_t full_h,
                             std::uint32_t depth_slot, const float view_mat[16], const LightingFrameInput &in,
                             const float sun_dir_world[3], bool is_cave, bool want_godrays,
                             VkDeviceAddress lights_bda, std::uint32_t light_count,
                             void *params_cpu, VkDeviceAddress params_bda) {
    const int fs = ((frame_slot % kRingSlots) + kRingSlots) % kRingSlots;
    if (!ready() || !tlas_valid_[fs] || tlas_[fs] == VK_NULL_HANDLE || depth_slot == 0u || !params_cpu || params_bda == 0) {
        return false;
    }
    const std::uint32_t hw = std::max(full_w / 2u, 1u);
    const std::uint32_t hh = std::max(full_h / 2u, 1u);
    if (!ensure_output(ctx, desc, hw, hh)) return false;

    if (rt_desc_dirty_[fs]) {
        VkWriteDescriptorSetAccelerationStructureKHR asw{
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR, nullptr, 1, &tlas_[fs]};
        VkDescriptorImageInfo dii{VK_NULL_HANDLE, out_view_, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet writes[2]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].pNext = &asw;
        writes[0].dstSet = rt_sets_[fs];
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = rt_sets_[fs];
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[1].pImageInfo = &dii;
        vkUpdateDescriptorSets(ctx.device, 2, writes, 0, nullptr);
        rt_desc_dirty_[fs] = false;
    }

    Context::cmd_transition_image(cmd, out_img_, VK_IMAGE_ASPECT_COLOR_BIT, out_layout_, VK_IMAGE_LAYOUT_GENERAL,
                                  out_layout_ == VK_IMAGE_LAYOUT_UNDEFINED ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT
                                                                           : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                  out_layout_ == VK_IMAGE_LAYOUT_UNDEFINED ? 0 : VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    out_layout_ = VK_IMAGE_LAYOUT_GENERAL;

    RtShadowParams rp{};
    invert_rigid_col(view_mat, rp.inv_view);
    const float p00 = std::fabs(in.proj[0]) > 1e-4f ? in.proj[0] : 1.0f;
    const float p11 = std::fabs(in.proj[5]) > 1e-4f ? in.proj[5] : 1.76f;
    rp.proj_info[0] = 1.0f / p00;
    rp.proj_info[1] = 1.0f / p11;
    rp.proj_info[2] = in.proj[10];
    rp.proj_info[3] = std::fabs(in.proj[14]) > 1e-4f ? in.proj[14] : -10.0f;
    rp.vz_and_size[0] = in.vz_scale;
    rp.vz_and_size[1] = in.vz_center;
    rp.vz_and_size[2] = static_cast<float>(full_w);
    rp.vz_and_size[3] = static_cast<float>(full_h);
    rp.sun_dir_world[0] = sun_dir_world[0];
    rp.sun_dir_world[1] = std::max(sun_dir_world[1], 0.38f);
    rp.sun_dir_world[2] = sun_dir_world[2];
    rp.sun_dir_world[3] = 0.018f; // ~1 degree soft sun angular cone
    rp.out_slots[0] = hw;
    rp.out_slots[1] = hh;
    rp.out_slots[2] = depth_slot;
    rp.out_slots[3] = static_cast<std::uint32_t>(in.frame_index & 0xFFFFu);
    rp.lights_bda = lights_bda;
    rp.light_count = light_count;
    rp.flags = (is_cave ? 1u : 0u) | (want_godrays ? 2u : 0u);
    std::memcpy(params_cpu, &rp, sizeof(rp));

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_);
    VkDescriptorSet sets[2] = {bindless_set, rt_sets_[fs]};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_, 0, 2, sets, 0, nullptr);
    vkCmdPushConstants(cmd, pipe_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params_bda), &params_bda);
    vkCmdDispatch(cmd, (hw + 7u) / 8u, (hh + 7u) / 8u, 1u);

    Context::cmd_transition_image(cmd, out_img_, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    out_layout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    return true;
}

} // namespace mhf::psp::vk
