#include "vk_shadows.hpp"
#include "vk_shaders_spv.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace mhf::psp::vk {

namespace {

VkShaderModule make_module(VkDevice dev, const std::uint32_t *words, std::size_t count) {
    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = count * sizeof(std::uint32_t);
    ci.pCode = words;
    VkShaderModule mod = VK_NULL_HANDLE;
    if (vkCreateShaderModule(dev, &ci, nullptr, &mod) != VK_SUCCESS) return VK_NULL_HANDLE;
    return mod;
}

void mul_mat4_col(const float a[16], const float b[16], float out[16]) noexcept {
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            out[col * 4 + row] =
                a[0 * 4 + row] * b[col * 4 + 0] +
                a[1 * 4 + row] * b[col * 4 + 1] +
                a[2 * 4 + row] * b[col * 4 + 2] +
                a[3 * 4 + row] * b[col * 4 + 3];
        }
    }
}

void invert_rigid_col(const float v[16], float out[16]) noexcept {
    // Rigid column-major [R | t; 0 1] -> [R^T | -R^T t; 0 1]
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

bool SunShadowMap::init(const Context &ctx, VkDescriptorSetLayout bindless_layout, BindlessDescriptors &desc) {
    // 1. 2048x2048 D16_UNORM depth-only shadow map (8 MB VRAM, ultra-fast ROP/cache on iGPU).
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_D16_UNORM;
    ici.extent = {kMapSize, kMapSize, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    VmaAllocationCreateInfo vaci{};
    vaci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (vmaCreateImage(ctx.allocator, &ici, &vaci, &depth_img_, &depth_alloc_, nullptr) != VK_SUCCESS) {
        return false;
    }

    VkImageViewCreateInfo ivci{};
    ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivci.image = depth_img_;
    ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ivci.format = VK_FORMAT_D16_UNORM;
    ivci.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(ctx.device, &ivci, nullptr, &depth_view_) != VK_SUCCESS) {
        shutdown(ctx, desc);
        return false;
    }
    depth_slot_ = desc.allocate_texture(ctx, depth_view_);
    if (depth_slot_ == 0u) {
        shutdown(ctx, desc);
        return false;
    }
    depth_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;

    // 2. Depth-only graphics pipeline (push constant = 8-byte BDA to ShadowDrawParams).
    VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(VkDeviceAddress)};
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &bindless_layout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(ctx.device, &plci, nullptr, &layout_) != VK_SUCCESS) {
        shutdown(ctx, desc);
        return false;
    }

    VkShaderModule vs = make_module(ctx.device, shaders::kSunShadowVertSpv.data(), shaders::kSunShadowVertSpv.size());
    VkShaderModule fs = make_module(ctx.device, shaders::kSunShadowFragSpv.data(), shaders::kSunShadowFragSpv.size());
    if (!vs || !fs) {
        if (vs) vkDestroyShaderModule(ctx.device, vs, nullptr);
        if (fs) vkDestroyShaderModule(ctx.device, fs, nullptr);
        shutdown(ctx, desc);
        return false;
    }

    VkPipelineShaderStageCreateInfo stages[2]{
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vs, "main", nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fs, "main", nullptr},
    };
    VkVertexInputBindingDescription bind{0, sizeof(ShadowVertex), VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription attrs[2]{
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},
        {1, 0, VK_FORMAT_R32G32_SFLOAT, 12},
    };
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, nullptr, 0,
                                            1, &bind, 2, attrs};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0,
                                              VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_FALSE};
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, nullptr, 1, nullptr};
    // Hardware depth bias prevents shadow acne cleanly without peter-panning.
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, nullptr, 0,
                                              VK_FALSE, VK_FALSE, VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE,
                                              VK_FRONT_FACE_COUNTER_CLOCKWISE, VK_TRUE, 2.5f, 0.0f, 2.2f, 1.0f};
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, nullptr, 0,
                                            VK_SAMPLE_COUNT_1_BIT, VK_FALSE, 1.0f, nullptr, VK_FALSE, VK_FALSE};
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO, nullptr, 0,
                                             VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL, VK_FALSE, VK_FALSE, {}, {}, 0.0f, 1.0f};
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, nullptr, 0,
                                           VK_FALSE, VK_LOGIC_OP_COPY, 0, nullptr, {0, 0, 0, 0}};
    const VkDynamicState dyn_states[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, nullptr, 0, 2, dyn_states};

    VkPipelineRenderingCreateInfo rci{};
    rci.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rci.colorAttachmentCount = 0;
    rci.depthAttachmentFormat = VK_FORMAT_D16_UNORM;

    VkGraphicsPipelineCreateInfo gpci{};
    gpci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gpci.pNext = &rci;
    gpci.stageCount = 2;
    gpci.pStages = stages;
    gpci.pVertexInputState = &vi;
    gpci.pInputAssemblyState = &ia;
    gpci.pViewportState = &vp;
    gpci.pRasterizationState = &rs;
    gpci.pMultisampleState = &ms;
    gpci.pDepthStencilState = &ds;
    gpci.pColorBlendState = &cb;
    gpci.pDynamicState = &dyn;
    gpci.layout = layout_;

    const VkResult res = vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &gpci, nullptr, &pipe_);
    vkDestroyShaderModule(ctx.device, vs, nullptr);
    vkDestroyShaderModule(ctx.device, fs, nullptr);
    if (res != VK_SUCCESS) {
        shutdown(ctx, desc);
        return false;
    }
    return true;
}

void SunShadowMap::shutdown(const Context &ctx, BindlessDescriptors &desc) {
    if (pipe_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(ctx.device, pipe_, nullptr);
        pipe_ = VK_NULL_HANDLE;
    }
    if (layout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(ctx.device, layout_, nullptr);
        layout_ = VK_NULL_HANDLE;
    }
    if (depth_slot_ != 0u) {
        desc.free_texture(depth_slot_, UINT64_MAX);
        depth_slot_ = 0u;
    }
    if (depth_view_ != VK_NULL_HANDLE) {
        vkDestroyImageView(ctx.device, depth_view_, nullptr);
        depth_view_ = VK_NULL_HANDLE;
    }
    if (depth_img_ != VK_NULL_HANDLE) {
        vmaDestroyImage(ctx.allocator, depth_img_, depth_alloc_);
        depth_img_ = VK_NULL_HANDLE;
        depth_alloc_ = VK_NULL_HANDLE;
    }
    depth_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    valid_ = false;
}

bool SunShadowMap::render(VkCommandBuffer cmd, VkDescriptorSet bindless_set,
                          const float view_mat[16], const float sun_dir_world[3],
                          const std::vector<ShadowVertex> &verts, const std::vector<ShadowBatch> &batches,
                          VkBuffer vb_handle, VkDeviceSize vb_offset,
                          void *params_cpu, VkDeviceAddress params_bda, std::size_t max_batches) {
    if (!ready() || verts.empty() || batches.empty() || !params_cpu || params_bda == 0) {
        valid_ = false;
        return false;
    }

    // 1. Normalize sun direction (world vector pointing TOWARD the sun, clamped above horizon >= 22 deg).
    float lx = sun_dir_world[0], ly = sun_dir_world[1], lz = sun_dir_world[2];
    float llen = std::sqrt(lx * lx + ly * ly + lz * lz);
    if (llen < 1e-4f) {
        lx = 0.34f; ly = 0.78f; lz = 0.52f;
        llen = std::sqrt(lx * lx + ly * ly + lz * lz);
    }
    lx /= llen; ly /= llen; lz /= llen;
    if (ly < 0.38f) {
        ly = 0.38f;
        llen = std::sqrt(lx * lx + ly * ly + lz * lz);
        lx /= llen; ly /= llen; lz /= llen;
    }

    // 2. Build orthonormal light basis (right, up, dir_to_sun).
    float ref_x = 0.0f, ref_y = 1.0f, ref_z = 0.0f;
    if (std::fabs(ly) > 0.96f) { ref_x = 0.0f; ref_y = 0.0f; ref_z = 1.0f; }
    // right = normalize(cross(ref, L))
    float rx = ref_y * lz - ref_z * ly;
    float ry = ref_z * lx - ref_x * lz;
    float rz = ref_x * ly - ref_y * lx;
    const float rlen = std::max(std::sqrt(rx * rx + ry * ry + rz * rz), 1e-5f);
    rx /= rlen; ry /= rlen; rz /= rlen;
    // up = cross(L, right)
    const float ux = ly * rz - lz * ry;
    const float uy = lz * rx - lx * rz;
    const float uz = lx * ry - ly * rx;

    // 3. Focus point: 450 cm in front of the camera in world space, texel-snapped in light space!
    const float tx = view_mat[12], ty = view_mat[13], tz = view_mat[14];
    const float cam_x = -(view_mat[0] * tx + view_mat[1] * ty + view_mat[2] * tz);
    const float cam_y = -(view_mat[4] * tx + view_mat[5] * ty + view_mat[6] * tz);
    const float cam_z = -(view_mat[8] * tx + view_mat[9] * ty + view_mat[10] * tz);
    // Camera forward in world space = -row 2 of view rotation.
    const float fwd_x = -view_mat[2], fwd_y = -view_mat[6], fwd_z = -view_mat[10];
    float cx = cam_x + fwd_x * 480.0f;
    float cy = cam_y + fwd_y * 480.0f;
    float cz = cam_z + fwd_z * 480.0f;

    const float half_xy = 1450.0f; // 29m x 29m crisp shadow coverage around hunter & monster
    const float half_z  = 2400.0f; // 48m depth range along sun ray
    world_span_ = 2.0f * half_z;

    // Texel-snap center along right & up axes to prevent shadow crawling when moving!
    const float texel_world = (2.0f * half_xy) / static_cast<float>(kMapSize);
    const float c_r = std::floor((cx * rx + cy * ry + cz * rz) / texel_world) * texel_world;
    const float c_u = std::floor((cx * ux + cy * uy + cz * uz) / texel_world) * texel_world;
    const float c_l = cx * lx + cy * ly + cz * lz;
    cx = rx * c_r + ux * c_u + lx * c_l;
    cy = ry * c_r + uy * c_u + ly * c_l;
    cz = rz * c_r + uz * c_u + lz * c_l;

    // 4. Orthographic world -> shadow clip matrix (Vulkan NDC: X,Y in [-1,1], Z in [0,1] from near sun to far).
    // x_ndc = dot(P - C, right) / half_xy
    // y_ndc = dot(P - C, up)    / half_xy
    // z_ndc = (-dot(P - C, L) / half_z) * 0.5 + 0.5  (0 = closest to sun, 1 = farthest from sun)
    const float inv_xy = 1.0f / half_xy;
    const float inv_2z = 0.5f / half_z;
    shadow_vp_[0]  =  rx * inv_xy;
    shadow_vp_[1]  =  ux * inv_xy;
    shadow_vp_[2]  = -lx * inv_2z;
    shadow_vp_[3]  =  0.0f;

    shadow_vp_[4]  =  ry * inv_xy;
    shadow_vp_[5]  =  uy * inv_xy;
    shadow_vp_[6]  = -ly * inv_2z;
    shadow_vp_[7]  =  0.0f;

    shadow_vp_[8]  =  rz * inv_xy;
    shadow_vp_[9]  =  uz * inv_xy;
    shadow_vp_[10] = -lz * inv_2z;
    shadow_vp_[11] =  0.0f;

    shadow_vp_[12] = -(rx * cx + ry * cy + rz * cz) * inv_xy;
    shadow_vp_[13] = -(ux * cx + uy * cy + uz * cz) * inv_xy;
    shadow_vp_[14] =  (lx * cx + ly * cy + lz * cz) * inv_2z + 0.5f;
    shadow_vp_[15] =  1.0f;

    // 5. Transition shadow map -> DEPTH_ATTACHMENT_OPTIMAL and clear to 1.0.
    Context::cmd_transition_image(
        cmd, depth_img_, VK_IMAGE_ASPECT_DEPTH_BIT, depth_layout_, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        depth_layout_ == VK_IMAGE_LAYOUT_UNDEFINED ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT
                                                   : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        depth_layout_ == VK_IMAGE_LAYOUT_UNDEFINED ? 0 : VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
    depth_layout_ = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;

    VkRenderingAttachmentInfo depth_att{};
    depth_att.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depth_att.imageView = depth_view_;
    depth_att.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depth_att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depth_att.clearValue.depthStencil = {1.0f, 0u};

    VkRenderingInfo ri{};
    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea = {{0, 0}, {kMapSize, kMapSize}};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 0;
    ri.pDepthAttachment = &depth_att;

    vkCmdBeginRendering(cmd, &ri);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0, 1, &bindless_set, 0, nullptr);
    VkViewport vp{0.0f, 0.0f, static_cast<float>(kMapSize), static_cast<float>(kMapSize), 0.0f, 1.0f};
    VkRect2D sc{{0, 0}, {kMapSize, kMapSize}};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);
    vkCmdBindVertexBuffers(cmd, 0, 1, &vb_handle, &vb_offset);

    auto *dst_params = reinterpret_cast<ShadowDrawParams *>(params_cpu);
    const std::size_t n = std::min(batches.size(), max_batches);
    for (std::size_t i = 0; i < n; ++i) {
        const ShadowBatch &b = batches[i];
        if (b.v_count == 0u) continue;
        ShadowDrawParams &dp = dst_params[i];
        std::memcpy(dp.shadow_vp, shadow_vp_, sizeof(dp.shadow_vp));
        std::memcpy(dp.uv_xform, b.uv_xform, sizeof(dp.uv_xform));
        dp.tex_alpha[0] = b.tex_slot;
        dp.tex_alpha[1] = b.sampler_idx;
        dp.tex_alpha[2] = b.alpha_bits;
        dp.tex_alpha[3] = b.textured;
        const VkDeviceAddress paddr = params_bda + i * sizeof(ShadowDrawParams);
        vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(paddr), &paddr);
        vkCmdDraw(cmd, b.v_count, 1u, b.v_offset, 0u);
    }

    vkCmdEndRendering(cmd);

    // 6. Transition shadow map -> SHADER_READ_ONLY_OPTIMAL for sampling in lighting_composite.
    Context::cmd_transition_image(
        cmd, depth_img_, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    depth_layout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    valid_ = true;
    return true;
}

void SunShadowMap::compute_shadow_from_view(const float view_mat[16], float out_shadow_from_view[16]) const noexcept {
    float inv_view[16]{};
    invert_rigid_col(view_mat, inv_view);
    mul_mat4_col(shadow_vp_, inv_view, out_shadow_from_view);
}

} // namespace mhf::psp::vk
