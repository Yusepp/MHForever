#include "vk_pipelines.hpp"
#include "vk_shaders_spv.hpp"
#include "../gpu_common.hpp"

#include <array>
#include <cstddef>
#include <cstdio>

namespace mhf::psp::vk {

namespace {

VkPipeline create_fullscreen_pipe(const Context &ctx, VkPipelineLayout layout, VkShaderModule vs, VkShaderModule fs,
                                  VkFormat color_fmt, bool premul_blend) {
    std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vs, "main", nullptr};
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fs, "main", nullptr};

    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, nullptr, 0, 0, nullptr, 0, nullptr};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0,
                                              VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_FALSE};
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, nullptr, 1, nullptr};
    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState att{};
    att.blendEnable = premul_blend ? VK_TRUE : VK_FALSE;
    att.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    att.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    att.colorBlendOp = VK_BLEND_OP_ADD;
    att.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    att.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    att.alphaBlendOp = VK_BLEND_OP_ADD;
    att.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments = &att;

    std::array<VkDynamicState, 2> dyn = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, nullptr, 0,
                                        static_cast<std::uint32_t>(dyn.size()), dyn.data()};

    VkPipelineRenderingCreateInfo rci{};
    rci.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rci.colorAttachmentCount = 1;
    rci.pColorAttachmentFormats = &color_fmt;

    VkGraphicsPipelineCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pci.pNext = &rci;
    pci.stageCount = static_cast<std::uint32_t>(stages.size());
    pci.pStages = stages.data();
    pci.pVertexInputState = &vi;
    pci.pInputAssemblyState = &ia;
    pci.pViewportState = &vp;
    pci.pRasterizationState = &rs;
    pci.pMultisampleState = &ms;
    pci.pColorBlendState = &cb;
    pci.pDynamicState = &ds;
    pci.layout = layout;

    VkPipeline pipe = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(ctx.device, ctx.pipeline_cache, 1, &pci, nullptr, &pipe) != VK_SUCCESS) return VK_NULL_HANDLE;
    return pipe;
}

} // namespace

bool Pipelines::init(const Context &ctx, VkDescriptorSetLayout bindless_layout, VkFormat swap_fmt) {
    auto make_layout = [&](std::uint32_t push_size, VkShaderStageFlags stages) -> VkPipelineLayout {
        VkPushConstantRange pcr{stages, 0, push_size};
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0,
                                        1, &bindless_layout, 1, &pcr};
        VkPipelineLayout pl = VK_NULL_HANDLE;
        if (vkCreatePipelineLayout(ctx.device, &plci, nullptr, &pl) != VK_SUCCESS) return VK_NULL_HANDLE;
        return pl;
    };

    present_layout = make_layout(sizeof(PresentPush), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
    blur_layout = make_layout(sizeof(BlurPush), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
    ge_layout = make_layout(sizeof(DrawPush), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
    if (!present_layout || !blur_layout || !ge_layout) return false;

    ge_vert = ctx.create_shader_module(shaders::kGeDrawVertSpv);
    ge_frag = ctx.create_shader_module(shaders::kGeDrawFragSpv);
    VkShaderModule blur_vs = ctx.create_shader_module(shaders::kBlurVertSpv);
    VkShaderModule blur_fs = ctx.create_shader_module(shaders::kBlurFragSpv);
    if (!ge_vert || !ge_frag || !blur_vs || !blur_fs) {
        if (blur_vs) vkDestroyShaderModule(ctx.device, blur_vs, nullptr);
        if (blur_fs) vkDestroyShaderModule(ctx.device, blur_fs, nullptr);
        return false;
    }
    blur_pipe = create_fullscreen_pipe(ctx, blur_layout, blur_vs, blur_fs, color_format, false);
    vkDestroyShaderModule(ctx.device, blur_vs, nullptr);
    vkDestroyShaderModule(ctx.device, blur_fs, nullptr);
    if (!blur_pipe) return false;

    return rebuild_swapchain_pipelines(ctx, swap_fmt);
}

bool Pipelines::rebuild_swapchain_pipelines(const Context &ctx, VkFormat swap_fmt) {
    if (present_pipe) { vkDestroyPipeline(ctx.device, present_pipe, nullptr); present_pipe = VK_NULL_HANDLE; }
    if (overlay_pipe) { vkDestroyPipeline(ctx.device, overlay_pipe, nullptr); overlay_pipe = VK_NULL_HANDLE; }

    VkShaderModule p_vs = ctx.create_shader_module(shaders::kPresentVertSpv);
    VkShaderModule p_fs = ctx.create_shader_module(shaders::kPresentFragSpv);
    VkShaderModule o_fs = ctx.create_shader_module(shaders::kOverlayFragSpv);
    if (!p_vs || !p_fs || !o_fs) {
        if (p_vs) vkDestroyShaderModule(ctx.device, p_vs, nullptr);
        if (p_fs) vkDestroyShaderModule(ctx.device, p_fs, nullptr);
        if (o_fs) vkDestroyShaderModule(ctx.device, o_fs, nullptr);
        return false;
    }
    present_pipe = create_fullscreen_pipe(ctx, present_layout, p_vs, p_fs, swap_fmt, false);
    overlay_pipe = create_fullscreen_pipe(ctx, present_layout, p_vs, o_fs, swap_fmt, true);
    vkDestroyShaderModule(ctx.device, p_vs, nullptr);
    vkDestroyShaderModule(ctx.device, p_fs, nullptr);
    vkDestroyShaderModule(ctx.device, o_fs, nullptr);
    return present_pipe != VK_NULL_HANDLE && overlay_pipe != VK_NULL_HANDLE;
}

VkPipeline Pipelines::get_ge_pipeline(const Context &ctx, GePipelineKey key) {
    if (ctx.caps.extended_dynamic_state3) {
        key.blend_enable = 0;
        key.color_op = 0;
        key.src_factor = 1;
        key.dst_factor = 0;
        key.color_mask = 0xFu;
    }
    const std::uint64_t h = key.hash();
    if (const auto it = ge_pipes.find(h); it != ge_pipes.end()) return it->second;

    const std::uint32_t spec_data[3] = {key.spec_clear, key.spec_textured, key.spec_alpha_test};
    const std::array<VkSpecializationMapEntry, 3> spec_entries = {{
        {0, 0, sizeof(std::uint32_t)},
        {1, sizeof(std::uint32_t), sizeof(std::uint32_t)},
        {2, 2 * sizeof(std::uint32_t), sizeof(std::uint32_t)},
    }};
    VkSpecializationInfo spec{static_cast<std::uint32_t>(spec_entries.size()), spec_entries.data(),
                              sizeof(spec_data), spec_data};

    std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, ge_vert, "main", nullptr};
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, ge_frag, "main", &spec};

    using gpu_common::GpuVertex;
    VkVertexInputBindingDescription binding{0, sizeof(GpuVertex), VK_VERTEX_INPUT_RATE_VERTEX};
    std::array<VkVertexInputAttributeDescription, 5> attrs = {{
        {0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, static_cast<std::uint32_t>(offsetof(GpuVertex, x))},
        {1, 0, VK_FORMAT_R32G32_SFLOAT, static_cast<std::uint32_t>(offsetof(GpuVertex, u))},
        {2, 0, VK_FORMAT_R8G8B8A8_UNORM, static_cast<std::uint32_t>(offsetof(GpuVertex, color))},
        {3, 0, VK_FORMAT_R8G8B8A8_UNORM, static_cast<std::uint32_t>(offsetof(GpuVertex, spec))},
        {4, 0, VK_FORMAT_R32_SFLOAT, static_cast<std::uint32_t>(offsetof(GpuVertex, fog))},
    }};
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, nullptr, 0,
                                            1, &binding, static_cast<std::uint32_t>(attrs.size()), attrs.data()};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0,
                                              VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_FALSE};
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, nullptr, 1, nullptr};

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.depthClampEnable = VK_TRUE;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo dss{};
    dss.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;

    VkPipelineColorBlendAttachmentState att{};
    att.blendEnable = key.blend_enable ? VK_TRUE : VK_FALSE;
    att.srcColorBlendFactor = static_cast<VkBlendFactor>(key.src_factor);
    att.dstColorBlendFactor = static_cast<VkBlendFactor>(key.dst_factor);
    att.colorBlendOp = static_cast<VkBlendOp>(key.color_op);
    att.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    att.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    att.alphaBlendOp = VK_BLEND_OP_ADD;
    att.colorWriteMask = key.color_mask;

    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments = &att;

    std::vector<VkDynamicState> dyn = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE,
        VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE,
        VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
        VK_DYNAMIC_STATE_BLEND_CONSTANTS,
    };
    if (ctx.caps.extended_dynamic_state3) {
        dyn.push_back(VK_DYNAMIC_STATE_COLOR_BLEND_ENABLE_EXT);
        dyn.push_back(VK_DYNAMIC_STATE_COLOR_BLEND_EQUATION_EXT);
        dyn.push_back(VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT);
    }
    VkPipelineDynamicStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, nullptr, 0,
                                        static_cast<std::uint32_t>(dyn.size()), dyn.data()};

    VkPipelineRenderingCreateInfo rci{};
    rci.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rci.colorAttachmentCount = 1;
    rci.pColorAttachmentFormats = &color_format;
    rci.depthAttachmentFormat = depth_format;

    VkGraphicsPipelineCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pci.pNext = &rci;
    pci.stageCount = static_cast<std::uint32_t>(stages.size());
    pci.pStages = stages.data();
    pci.pVertexInputState = &vi;
    pci.pInputAssemblyState = &ia;
    pci.pViewportState = &vp;
    pci.pRasterizationState = &rs;
    pci.pMultisampleState = &ms;
    pci.pDepthStencilState = &dss;
    pci.pColorBlendState = &cb;
    pci.pDynamicState = &ds;
    pci.layout = ge_layout;

    VkPipeline pipe = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(ctx.device, ctx.pipeline_cache, 1, &pci, nullptr, &pipe) != VK_SUCCESS) {
        std::fprintf(stderr, "[vk] failed to create GE pipeline (key=0x%llx)\n", static_cast<unsigned long long>(h));
        return VK_NULL_HANDLE;
    }
    ge_pipes.emplace(h, pipe);
    return pipe;
}

void Pipelines::shutdown(const Context &ctx) {
    for (auto &[k, p] : ge_pipes) {
        if (p != VK_NULL_HANDLE) vkDestroyPipeline(ctx.device, p, nullptr);
    }
    ge_pipes.clear();
    if (present_pipe) { vkDestroyPipeline(ctx.device, present_pipe, nullptr); present_pipe = VK_NULL_HANDLE; }
    if (overlay_pipe) { vkDestroyPipeline(ctx.device, overlay_pipe, nullptr); overlay_pipe = VK_NULL_HANDLE; }
    if (blur_pipe) { vkDestroyPipeline(ctx.device, blur_pipe, nullptr); blur_pipe = VK_NULL_HANDLE; }
    if (ge_vert) { vkDestroyShaderModule(ctx.device, ge_vert, nullptr); ge_vert = VK_NULL_HANDLE; }
    if (ge_frag) { vkDestroyShaderModule(ctx.device, ge_frag, nullptr); ge_frag = VK_NULL_HANDLE; }
    if (present_layout) { vkDestroyPipelineLayout(ctx.device, present_layout, nullptr); present_layout = VK_NULL_HANDLE; }
    if (blur_layout) { vkDestroyPipelineLayout(ctx.device, blur_layout, nullptr); blur_layout = VK_NULL_HANDLE; }
    if (ge_layout) { vkDestroyPipelineLayout(ctx.device, ge_layout, nullptr); ge_layout = VK_NULL_HANDLE; }
}

} // namespace mhf::psp::vk
