#include "vk_particles.hpp"
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

struct BiomeParticlePreset {
    float wind[3];        // cm/s
    float density;        // [0..1] fraction of kAmbientCount
    float gravity;        // cm/s^2
    float turbulence;     // curl wind amplitude (cm/s)
    float size_cm;        // base world radius (cm)
    float col_a[3];       // linear HDR color A
    float col_b[3];       // linear HDR color B
    float kind;           // 0=glow, 1=snow, 2=mote/ash, 3=streak
    float additive;       // 0=alpha, 1=additive
    float alpha;          // global alpha scale
};

BiomeParticlePreset preset_for_biome(BiomeKind b, bool is_cave) noexcept {
    if (is_cave) {
        if (b == BiomeKind::IceCave) {
            return {{12.0f, -18.0f, 8.0f}, 0.22f, -12.0f, 22.0f, 2.2f,
                    {0.68f, 0.85f, 1.00f}, {0.82f, 0.94f, 1.00f}, 1.0f, 0.0f, 0.42f};
        }
        // DarkCave / CozyIndoor: very subtle drifting dust motes.
        return {{8.0f, -6.0f, 6.0f}, 0.16f, -4.0f, 18.0f, 2.0f,
                {0.72f, 0.68f, 0.62f}, {0.58f, 0.56f, 0.54f}, 2.0f, 0.0f, 0.28f};
    }
    switch (b) {
        case BiomeKind::PokkeVillage:
            return {{38.0f, -48.0f, 18.0f}, 0.56f, -35.0f, 42.0f, 3.2f,
                    {0.96f, 0.98f, 1.00f}, {0.82f, 0.91f, 1.00f}, 1.0f, 0.0f, 0.74f};
        case BiomeKind::SnowyPeak:
            return {{115.0f, -62.0f, 45.0f}, 0.76f, -45.0f, 75.0f, 3.4f,
                    {0.96f, 0.99f, 1.00f}, {0.84f, 0.92f, 1.00f}, 1.0f, 0.0f, 0.78f};
        case BiomeKind::Volcano:
        case BiomeKind::FortressSiege:
            return {{28.0f, 58.0f, 18.0f}, 0.52f, 38.0f, 65.0f, 3.0f,
                    {1.65f, 0.52f, 0.10f}, {1.35f, 0.85f, 0.22f}, 0.0f, 1.0f, 0.72f};
        case BiomeKind::Swamp:
            // Subtle warm-lime fireflies drifting over the reeds.
            return {{14.0f, 6.0f, -10.0f}, 0.28f, 2.0f, 38.0f, 2.8f,
                    {0.95f, 1.25f, 0.42f}, {0.55f, 1.05f, 0.68f}, 0.0f, 1.0f, 0.58f};
        case BiomeKind::Desert:
            return {{145.0f, -12.0f, 55.0f}, 0.34f, -10.0f, 55.0f, 2.6f,
                    {0.82f, 0.68f, 0.46f}, {0.72f, 0.58f, 0.38f}, 3.0f, 0.0f, 0.34f};
        case BiomeKind::DesertSandstorm:
            return {{260.0f, -22.0f, 95.0f}, 0.72f, -18.0f, 95.0f, 3.0f,
                    {0.86f, 0.72f, 0.48f}, {0.74f, 0.60f, 0.40f}, 3.0f, 0.0f, 0.46f};
        case BiomeKind::ForestHills:
        case BiomeKind::GreatForest:
        case BiomeKind::Jungle:
            return {{24.0f, -10.0f, 14.0f}, 0.36f, -6.0f, 36.0f, 2.5f,
                    {0.96f, 0.92f, 0.64f}, {0.72f, 0.88f, 0.56f}, 2.0f, 0.0f, 0.42f};
        case BiomeKind::Tower:
            return {{42.0f, 12.0f, 24.0f}, 0.32f, 8.0f, 44.0f, 2.6f,
                    {0.96f, 0.90f, 0.74f}, {0.82f, 0.86f, 0.94f}, 2.0f, 0.0f, 0.40f};
        case BiomeKind::CozyIndoor:
            return {{4.0f, 4.0f, 4.0f}, 0.12f, 2.0f, 12.0f, 1.8f,
                    {0.92f, 0.80f, 0.58f}, {0.78f, 0.68f, 0.52f}, 2.0f, 0.0f, 0.25f};
        case BiomeKind::Arena:
        case BiomeKind::DefaultOutdoor:
        default:
            return {{22.0f, -8.0f, 12.0f}, 0.22f, -6.0f, 28.0f, 2.2f,
                    {0.88f, 0.86f, 0.76f}, {0.76f, 0.78f, 0.74f}, 2.0f, 0.0f, 0.30f};
    }
}

} // namespace

bool ParticleSystem::init(const Context &ctx, VkDescriptorSetLayout bindless_layout) {
    // 1. Persistent device-local particle buffer (4096 * 64B = 256 KB).
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = static_cast<VkDeviceSize>(kMaxParticles) * sizeof(GpuParticle);
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    VmaAllocationCreateInfo vaci{};
    vaci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (vmaCreateBuffer(ctx.allocator, &bci, &vaci, &particle_buf_, &particle_alloc_, nullptr) != VK_SUCCESS) {
        return false;
    }
    VkBufferDeviceAddressInfo bdai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, particle_buf_};
    particle_bda_ = vkGetBufferDeviceAddress(ctx.device, &bdai);

    // 2. Compute pipeline layout & pipeline (Set 0 = bindless textures, push constant = 8-byte BDA).
    VkPushConstantRange sim_pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkDeviceAddress)};
    VkPipelineLayoutCreateInfo sim_plci{};
    sim_plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    sim_plci.setLayoutCount = 1;
    sim_plci.pSetLayouts = &bindless_layout;
    sim_plci.pushConstantRangeCount = 1;
    sim_plci.pPushConstantRanges = &sim_pcr;
    if (vkCreatePipelineLayout(ctx.device, &sim_plci, nullptr, &sim_layout_) != VK_SUCCESS) {
        shutdown(ctx);
        return false;
    }

    VkShaderModule sim_mod = make_module(ctx.device, shaders::kParticlesSimCompSpv.data(),
                                         shaders::kParticlesSimCompSpv.size());
    if (!sim_mod) {
        shutdown(ctx);
        return false;
    }
    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                  VK_SHADER_STAGE_COMPUTE_BIT, sim_mod, "main", nullptr};
    cpci.layout = sim_layout_;
    const VkResult sim_res = vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpci, nullptr, &sim_pipe_);
    vkDestroyShaderModule(ctx.device, sim_mod, nullptr);
    if (sim_res != VK_SUCCESS) {
        shutdown(ctx);
        return false;
    }

    // 3. Graphics pipeline layout & 4 variants (HDR/LDR x 1x/4x MSAA).
    VkPushConstantRange draw_pcr{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(VkDeviceAddress)};
    VkPipelineLayoutCreateInfo draw_plci{};
    draw_plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    draw_plci.setLayoutCount = 1;
    draw_plci.pSetLayouts = &bindless_layout;
    draw_plci.pushConstantRangeCount = 1;
    draw_plci.pPushConstantRanges = &draw_pcr;
    if (vkCreatePipelineLayout(ctx.device, &draw_plci, nullptr, &draw_layout_) != VK_SUCCESS) {
        shutdown(ctx);
        return false;
    }

    VkShaderModule vs_mod = make_module(ctx.device, shaders::kParticlesDrawVertSpv.data(),
                                        shaders::kParticlesDrawVertSpv.size());
    VkShaderModule fs_mod = make_module(ctx.device, shaders::kParticlesDrawFragSpv.data(),
                                        shaders::kParticlesDrawFragSpv.size());
    if (!vs_mod || !fs_mod) {
        if (vs_mod) vkDestroyShaderModule(ctx.device, vs_mod, nullptr);
        if (fs_mod) vkDestroyShaderModule(ctx.device, fs_mod, nullptr);
        shutdown(ctx);
        return false;
    }

    auto build_draw_pipe = [&](VkFormat color_fmt, VkSampleCountFlagBits samples) -> VkPipeline {
        VkPipelineShaderStageCreateInfo stages[2]{
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vs_mod, "main", nullptr},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fs_mod, "main", nullptr},
        };
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, nullptr, 0, 0, nullptr, 0, nullptr};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_FALSE};
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, nullptr, 1, nullptr};
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, nullptr, 0,
                                                  VK_FALSE, VK_FALSE, VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE,
                                                  VK_FRONT_FACE_COUNTER_CLOCKWISE, VK_FALSE, 0.0f, 0.0f, 0.0f, 1.0f};
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, nullptr, 0,
                                                samples, VK_FALSE, 1.0f, nullptr, VK_FALSE, VK_FALSE};
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO, nullptr, 0,
                                                 VK_TRUE, VK_FALSE, VK_COMPARE_OP_LESS_OR_EQUAL, VK_FALSE, VK_FALSE, {}, {}, 0.0f, 1.0f};
        // Premultiplied alpha: handles both alpha-blended snow/ash AND additive HDR embers/sparks in 1 pass!
        VkPipelineColorBlendAttachmentState cb_att{};
        cb_att.blendEnable = VK_TRUE;
        cb_att.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        cb_att.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        cb_att.colorBlendOp = VK_BLEND_OP_ADD;
        cb_att.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        cb_att.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        cb_att.alphaBlendOp = VK_BLEND_OP_ADD;
        cb_att.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, nullptr, 0,
                                               VK_FALSE, VK_LOGIC_OP_COPY, 1, &cb_att, {0, 0, 0, 0}};
        const VkDynamicState dyn_states[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, nullptr, 0, 2, dyn_states};

        VkPipelineRenderingCreateInfo rci{};
        rci.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        rci.colorAttachmentCount = 1;
        rci.pColorAttachmentFormats = &color_fmt;
        rci.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;

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
        gpci.layout = draw_layout_;

        VkPipeline pipe = VK_NULL_HANDLE;
        if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &gpci, nullptr, &pipe) != VK_SUCCESS) {
            return VK_NULL_HANDLE;
        }
        return pipe;
    };

    draw_pipe_hdr_ = build_draw_pipe(VK_FORMAT_B10G11R11_UFLOAT_PACK32, VK_SAMPLE_COUNT_1_BIT);
    draw_pipe_ldr_ = build_draw_pipe(VK_FORMAT_R8G8B8A8_UNORM, VK_SAMPLE_COUNT_1_BIT);
    if (ctx.caps.msaa_4x) {
        draw_pipe_hdr_msaa_ = build_draw_pipe(VK_FORMAT_B10G11R11_UFLOAT_PACK32, VK_SAMPLE_COUNT_4_BIT);
        draw_pipe_ldr_msaa_ = build_draw_pipe(VK_FORMAT_R8G8B8A8_UNORM, VK_SAMPLE_COUNT_4_BIT);
    }

    vkDestroyShaderModule(ctx.device, vs_mod, nullptr);
    vkDestroyShaderModule(ctx.device, fs_mod, nullptr);

    if (!draw_pipe_hdr_ || !draw_pipe_ldr_) {
        shutdown(ctx);
        return false;
    }
    need_reset_ = true;
    return true;
}

void ParticleSystem::shutdown(const Context &ctx) {
    auto kill_pipe = [&](VkPipeline &p) {
        if (p != VK_NULL_HANDLE) {
            vkDestroyPipeline(ctx.device, p, nullptr);
            p = VK_NULL_HANDLE;
        }
    };
    kill_pipe(sim_pipe_);
    kill_pipe(draw_pipe_hdr_);
    kill_pipe(draw_pipe_hdr_msaa_);
    kill_pipe(draw_pipe_ldr_);
    kill_pipe(draw_pipe_ldr_msaa_);
    if (sim_layout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(ctx.device, sim_layout_, nullptr);
        sim_layout_ = VK_NULL_HANDLE;
    }
    if (draw_layout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(ctx.device, draw_layout_, nullptr);
        draw_layout_ = VK_NULL_HANDLE;
    }
    if (particle_buf_ != VK_NULL_HANDLE) {
        vmaDestroyBuffer(ctx.allocator, particle_buf_, particle_alloc_);
        particle_buf_ = VK_NULL_HANDLE;
        particle_alloc_ = VK_NULL_HANDLE;
        particle_bda_ = 0;
    }
}

void ParticleSystem::simulate(VkCommandBuffer cmd, VkDescriptorSet bindless_set,
                              const ParticleFrameInput &in,
                              void *sim_params_cpu, VkDeviceAddress sim_params_bda) {
    if (!ready() || !sim_params_cpu || sim_params_bda == 0) return;

    if (in.biome != last_biome_ || in.is_cave != last_cave_) {
        need_reset_ = true;
        last_biome_ = in.biome;
        last_cave_ = in.is_cave;
    }

    const BiomeParticlePreset pre = preset_for_biome(in.biome, in.is_cave);

    // Recover camera world position from rigid column-major view matrix: cam = -R^T * t.
    const float *v = in.view;
    const float tx = v[12], ty = v[13], tz = v[14];
    const float cam_x = -(v[0] * tx + v[1] * ty + v[2] * tz);
    const float cam_y = -(v[4] * tx + v[5] * ty + v[6] * tz);
    const float cam_z = -(v[8] * tx + v[9] * ty + v[10] * tz);

    ParticleSimParams sp{};
    mul_mat4_col(in.proj, in.view, sp.view_proj);
    sp.cam_pos_dt[0] = cam_x;
    sp.cam_pos_dt[1] = cam_y;
    sp.cam_pos_dt[2] = cam_z;
    sp.cam_pos_dt[3] = std::clamp(in.dt, 0.002f, 0.05f);

    sp.wind_time[0] = pre.wind[0];
    sp.wind_time[1] = pre.wind[1];
    sp.wind_time[2] = pre.wind[2];
    sp.wind_time[3] = in.elapsed_time;

    sp.vz_and_size[0] = in.vz_scale;
    sp.vz_and_size[1] = in.vz_center;
    sp.vz_and_size[2] = static_cast<float>(std::max(in.depth_w, 1u));
    sp.vz_and_size[3] = static_cast<float>(std::max(in.depth_h, 1u));

    sp.biome_cfg[0] = pre.density;
    sp.biome_cfg[1] = pre.gravity;
    sp.biome_cfg[2] = pre.turbulence;
    sp.biome_cfg[3] = pre.size_cm;

    sp.ambient_col_a[0] = pre.col_a[0];
    sp.ambient_col_a[1] = pre.col_a[1];
    sp.ambient_col_a[2] = pre.col_a[2];
    sp.ambient_col_a[3] = pre.kind;

    sp.ambient_col_b[0] = pre.col_b[0];
    sp.ambient_col_b[1] = pre.col_b[1];
    sp.ambient_col_b[2] = pre.col_b[2];
    sp.ambient_col_b[3] = pre.additive;

    const std::uint32_t ecount = std::min<std::uint32_t>(in.emitter_count, 8u);
    sp.counts_slots[0] = kMaxParticles;
    sp.counts_slots[1] = kAmbientCount;
    sp.counts_slots[2] = ecount;
    sp.counts_slots[3] = in.depth_slot;

    sp.flags_seed[0] = need_reset_ ? 1u : 0u;
    sp.flags_seed[1] = in.frame_seed;
    sp.flags_seed[2] = static_cast<std::uint32_t>(in.biome);
    sp.flags_seed[3] = in.is_cave ? 1u : 0u;
    for (std::uint32_t i = 0; i < ecount; ++i) {
        sp.emitters[i] = in.emitters[i];
    }
    sp.particles_bda = particle_bda_;
    std::memcpy(sim_params_cpu, &sp, sizeof(sp));
    need_reset_ = false;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sim_pipe_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sim_layout_, 0, 1, &bindless_set, 0, nullptr);
    vkCmdPushConstants(cmd, sim_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(sim_params_bda), &sim_params_bda);
    vkCmdDispatch(cmd, (kMaxParticles + 63u) / 64u, 1u, 1u);

    // Ensure compute writes are visible to vertex/fragment shader reads in the upcoming 3D pass.
    VkMemoryBarrier2 mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cmd, &dep);
}

void ParticleSystem::draw(VkCommandBuffer cmd, VkDescriptorSet bindless_set,
                          const ParticleFrameInput &in, bool hdr_target, bool msaa_4x,
                          void *draw_params_cpu, VkDeviceAddress draw_params_bda) {
    if (!ready() || !draw_params_cpu || draw_params_bda == 0) return;

    VkPipeline pipe = hdr_target ? (msaa_4x && draw_pipe_hdr_msaa_ ? draw_pipe_hdr_msaa_ : draw_pipe_hdr_)
                                 : (msaa_4x && draw_pipe_ldr_msaa_ ? draw_pipe_ldr_msaa_ : draw_pipe_ldr_);
    if (!pipe) return;

    const BiomeParticlePreset pre = preset_for_biome(in.biome, in.is_cave);
    const float *v = in.view;

    ParticleDrawParams dp{};
    std::memcpy(dp.view, in.view, sizeof(dp.view));
    std::memcpy(dp.proj, in.proj, sizeof(dp.proj));
    // Camera world-space right (row 0 of view rotation) and up (row 1 of view rotation).
    dp.cam_right[0] = v[0]; dp.cam_right[1] = v[4]; dp.cam_right[2] = v[8];  dp.cam_right[3] = in.vz_scale;
    dp.cam_up[0]    = v[1]; dp.cam_up[1]    = v[5]; dp.cam_up[2]    = v[9];  dp.cam_up[3]    = in.vz_center;
    dp.screen_info[0] = 1.0f / static_cast<float>(std::max(in.depth_w, 1u));
    dp.screen_info[1] = 1.0f / static_cast<float>(std::max(in.depth_h, 1u));
    dp.screen_info[2] = in.elapsed_time;
    dp.screen_info[3] = pre.alpha;
    dp.slots[0] = in.soft_depth_slot;
    dp.particles_bda = particle_bda_;
    std::memcpy(draw_params_cpu, &dp, sizeof(dp));

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, draw_layout_, 0, 1, &bindless_set, 0, nullptr);
    vkCmdPushConstants(cmd, draw_layout_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(draw_params_bda), &draw_params_bda);
    vkCmdDraw(cmd, 6u, kMaxParticles, 0u, 0u);
}

} // namespace mhf::psp::vk
