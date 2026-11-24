#include "vk_transfer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace mhf::psp::vk {

bool PackStreamer::init(const Context &ctx, const std::string &pack_folder, const std::string &game_id,
                        const std::string &builtin_folder, bool generate_mips) {
    generate_mips_ = generate_mips;
    std::string err;
    if (!pack_folder.empty()) {
        const auto located = textures::Pack::locate(textures::from_utf8(pack_folder), game_id);
        if (!located.empty()) {
            pack_ = textures::Pack::open(located, game_id, err);
            if (pack_) {
                std::fprintf(stderr, "[vk][pack] loaded %zu entries from %s\n",
                             pack_->size(), textures::utf8(pack_->ini()).c_str());
            } else {
                std::fprintf(stderr, "[vk][pack] failed to open %s: %s\n", pack_folder.c_str(), err.c_str());
            }
        }
    }
    if (!builtin_folder.empty()) {
        const auto located = textures::Pack::locate(textures::from_utf8(builtin_folder), game_id);
        if (!located.empty()) {
            builtin_pack_ = textures::Pack::open(located, game_id, err);
        }
    }
    if (!pack_ && !builtin_pack_) return true;

    loader_ = std::make_unique<textures::Loader>(2, 512u << 20, ctx.caps.bc_compression);
    return true;
}

void PackStreamer::collect_staging(const Context &ctx, std::uint64_t completed_timeline) {
    while (!retired_staging_.empty() && retired_staging_.front().retire_value <= completed_timeline) {
        const auto &s = retired_staging_.front();
        if (s.buffer != VK_NULL_HANDLE) vmaDestroyBuffer(ctx.allocator, s.buffer, s.alloc);
        retired_staging_.pop_front();
    }
}

void PackStreamer::shutdown(const Context &ctx, BindlessDescriptors &desc) {
    loader_.reset();
    pack_.reset();
    builtin_pack_.reset();
    collect_staging(ctx, UINT64_MAX);
    for (auto &[id, img] : uploaded_) {
        if (img.bindless_slot != 0) desc.free_texture(img.bindless_slot, 0);
        if (img.view != VK_NULL_HANDLE) vkDestroyImageView(ctx.device, img.view, nullptr);
        if (img.image != VK_NULL_HANDLE) vmaDestroyImage(ctx.allocator, img.image, img.alloc);
    }
    uploaded_.clear();
    total_vram_bytes_ = 0;
}

bool PackStreamer::upload_image_async(const Context &ctx, BindlessDescriptors &desc, VkCommandBuffer cmd,
                                      std::uint64_t retire_timeline, const textures::Image &img, PackGpuImage &out) {
    if (cmd == VK_NULL_HANDLE || img.width == 0 || img.height == 0 || img.levels.empty()) return false;

    const bool is_bc7 = (img.format == textures::Image::Format::Bc7 && ctx.caps.bc_compression);
    const VkFormat vk_fmt = is_bc7 ? VK_FORMAT_BC7_UNORM_BLOCK : VK_FORMAT_R8G8B8A8_UNORM;
    const std::uint32_t file_mips = static_cast<std::uint32_t>(img.levels.size());
    const bool gen_mips = !is_bc7 && generate_mips_ && file_mips == 1u && img.width > 1u && img.height > 1u;
    const std::uint32_t mip_levels = gen_mips
        ? (static_cast<std::uint32_t>(std::floor(std::log2(static_cast<double>(std::max(img.width, img.height))))) + 1u)
        : file_mips;

    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = vk_fmt;
    ici.extent = {img.width, img.height, 1};
    ici.mipLevels = mip_levels;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                (gen_mips ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0u);

    VmaAllocationCreateInfo vaci{};
    vaci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (vmaCreateImage(ctx.allocator, &ici, &vaci, &out.image, &out.alloc, nullptr) != VK_SUCCESS) return false;

    std::size_t staging_bytes = 0;
    for (const auto &lvl : img.levels) staging_bytes += lvl.size();

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, staging_bytes,
                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
    VmaAllocationCreateInfo sb_aci{};
    sb_aci.usage = VMA_MEMORY_USAGE_AUTO;
    sb_aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VkBuffer staging = VK_NULL_HANDLE;
    VmaAllocation staging_alloc = VK_NULL_HANDLE;
    VmaAllocationInfo sinfo{};
    if (vmaCreateBuffer(ctx.allocator, &bci, &sb_aci, &staging, &staging_alloc, &sinfo) != VK_SUCCESS) {
        vmaDestroyImage(ctx.allocator, out.image, out.alloc);
        out.image = VK_NULL_HANDLE;
        return false;
    }

    auto *dst = static_cast<std::uint8_t *>(sinfo.pMappedData);
    std::vector<VkBufferImageCopy> copies;
    copies.reserve(file_mips);
    std::size_t off = 0;
    std::uint32_t mw = img.width, mh = img.height;
    for (std::uint32_t m = 0; m < file_mips; ++m) {
        std::memcpy(dst + off, img.levels[m].data(), img.levels[m].size());
        VkBufferImageCopy r{};
        r.bufferOffset = off;
        r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, m, 0, 1};
        r.imageExtent = {mw, mh, 1};
        copies.push_back(r);
        off += img.levels[m].size();
        mw = std::max(1u, mw >> 1u);
        mh = std::max(1u, mh >> 1u);
    }

    Context::cmd_transition_image(cmd, out.image, VK_IMAGE_ASPECT_COLOR_BIT,
                                  VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                                  VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                  0, mip_levels);
    vkCmdCopyBufferToImage(cmd, staging, out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           static_cast<std::uint32_t>(copies.size()), copies.data());

    if (gen_mips) {
        std::int32_t bw = static_cast<std::int32_t>(img.width);
        std::int32_t bh = static_cast<std::int32_t>(img.height);
        for (std::uint32_t m = 1; m < mip_levels; ++m) {
            Context::cmd_transition_image(cmd, out.image, VK_IMAGE_ASPECT_COLOR_BIT,
                                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                          VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                          VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                                          m - 1u, 1u);
            const std::int32_t nw = std::max(1, bw >> 1);
            const std::int32_t nh = std::max(1, bh >> 1);
            VkImageBlit blit{};
            blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, m - 1u, 0, 1};
            blit.srcOffsets[1] = {bw, bh, 1};
            blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, m, 0, 1};
            blit.dstOffsets[1] = {nw, nh, 1};
            vkCmdBlitImage(cmd, out.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
            Context::cmd_transition_image(cmd, out.image, VK_IMAGE_ASPECT_COLOR_BIT,
                                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                          VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                                          VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                                          m - 1u, 1u);
            bw = nw;
            bh = nh;
        }
        Context::cmd_transition_image(cmd, out.image, VK_IMAGE_ASPECT_COLOR_BIT,
                                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                      VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                                      mip_levels - 1u, 1u);
    } else {
        Context::cmd_transition_image(cmd, out.image, VK_IMAGE_ASPECT_COLOR_BIT,
                                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                      VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                                      0, mip_levels);
    }

    // Retire staging buffer asynchronously once this frame's GPU submission completes.
    retired_staging_.push_back({staging, staging_alloc, retire_timeline});

    VkImageViewCreateInfo ivci{};
    ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivci.image = out.image;
    ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ivci.format = vk_fmt;
    ivci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mip_levels, 0, 1};
    if (vkCreateImageView(ctx.device, &ivci, nullptr, &out.view) != VK_SUCCESS) {
        vmaDestroyImage(ctx.allocator, out.image, out.alloc);
        out.image = VK_NULL_HANDLE;
        return false;
    }
    out.mip_levels = mip_levels;
    out.vram_bytes = gen_mips ? (static_cast<std::size_t>(img.width) * img.height * 4u * 4u / 3u) : staging_bytes;
    out.bindless_slot = desc.allocate_texture(ctx, out.view);
    return out.bindless_slot != 0;
}

std::uint32_t PackStreamer::lookup_or_queue(const Context &ctx, BindlessDescriptors &desc,
                                            const textures::TextureFacts &facts,
                                            const textures::MemoryReader &reader,
                                            const std::function<VkCommandBuffer()> &begin_upload_cmd,
                                            std::uint64_t retire_timeline,
                                            std::uint64_t frame_num, bool &has_mips,
                                            textures::Filter &filter_out,
                                            bool &pending_out) {
    pending_out = false;
    if (!loader_) return 0;
    const textures::Pack *chosen = nullptr;
    textures::Match match{};
    for (const textures::Pack *p : {pack_.get(), builtin_pack_.get()}) {
        if (!p) continue;
        textures::Key key{};
        std::uint32_t cw = 0, ch = 0;
        if (!p->key_of(facts, reader, key, cw, ch)) continue;
        match = p->find(key);
        if (match.entry && !match.entry->keeps_original()) {
            chosen = p;
            break;
        }
    }
    if (!chosen || !match.entry) return 0;

    filter_out = match.filter;
    const std::string id = (chosen == builtin_pack_.get() ? "builtin:" : "pack:") + match.entry->image();
    if (auto it = uploaded_.find(id); it != uploaded_.end()) {
        it->second.last_used = frame_num;
        has_mips = it->second.mip_levels > 1u;
        return it->second.bindless_slot;
    }

    std::vector<std::filesystem::path> files;
    files.reserve(match.entry->levels.size());
    for (const auto &[lvl, name] : match.entry->levels) files.push_back(chosen->path_of(name));

    // Budget async GPU texture uploads per frame to keep staging memory bounded while streaming HD packs smoothly.
    if (frame_num != last_upload_frame_) {
        last_upload_frame_ = frame_num;
        uploads_this_frame_ = 0;
    }
    constexpr std::uint32_t kMaxUploadsPerFrame = 4u;

    std::shared_ptr<const textures::Image> decoded;
    const auto st = loader_->get(id, files, decoded);
    if (st == textures::Loader::State::Ready && decoded) {
        if (uploads_this_frame_ >= kMaxUploadsPerFrame) {
            pending_out = true;
            return 0;
        }
        ++uploads_this_frame_;
        const VkCommandBuffer cmd = begin_upload_cmd ? begin_upload_cmd() : VK_NULL_HANDLE;
        PackGpuImage gpu_img{};
        if (upload_image_async(ctx, desc, cmd, retire_timeline, *decoded, gpu_img)) {
            gpu_img.last_used = frame_num;
            total_vram_bytes_ += gpu_img.vram_bytes;
            has_mips = gpu_img.mip_levels > 1u;
            const std::uint32_t slot = gpu_img.bindless_slot;
            uploaded_.emplace(id, gpu_img);
            loader_->forget(id);
            return slot;
        }
        loader_->forget(id);
    } else if (st == textures::Loader::State::Loading) {
        pending_out = true;
    }
    return 0;
}

bool PackStreamer::trim_vram(const Context &ctx, BindlessDescriptors &desc, std::uint64_t retire_timeline) {
    if (uploaded_.empty()) return false;
    std::size_t budget = 1536ull << 20; // 1.5 GB fallback budget
    if (ctx.caps.memory_budget) {
        VmaBudget heaps[VK_MAX_MEMORY_HEAPS]{};
        vmaGetHeapBudgets(ctx.allocator, heaps);
        VkPhysicalDeviceMemoryProperties mp{};
        vkGetPhysicalDeviceMemoryProperties(ctx.physical_device, &mp);
        std::size_t device_budget = 0;
        for (std::uint32_t i = 0; i < mp.memoryHeapCount; ++i) {
            if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                device_budget += static_cast<std::size_t>(heaps[i].budget);
            }
        }
        if (device_budget > (256ull << 20)) budget = device_budget * 7u / 10u;
    }
    if (total_vram_bytes_ <= budget) return false;
    // Wait for all in-flight GPU work before destroying evicted pack images so no command buffer
    // or frame-gen snapshot can sample an unmapped GPUVM page!
    ctx.wait_idle();
    // Sort entries once by last_used (O(N log N)) instead of repeated O(N) full-map scans.
    std::vector<std::pair<std::uint64_t, std::string>> order;
    order.reserve(uploaded_.size());
    for (const auto &[id, img] : uploaded_) order.emplace_back(img.last_used, id);
    std::sort(order.begin(), order.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
    bool evicted = false;
    for (const auto &[used, id] : order) {
        if (total_vram_bytes_ <= budget) break;
        const auto it = uploaded_.find(id);
        if (it == uploaded_.end()) continue;
        if (it->second.bindless_slot != 0) desc.free_texture(it->second.bindless_slot, retire_timeline);
        if (it->second.view != VK_NULL_HANDLE) vkDestroyImageView(ctx.device, it->second.view, nullptr);
        if (it->second.image != VK_NULL_HANDLE) vmaDestroyImage(ctx.allocator, it->second.image, it->second.alloc);
        total_vram_bytes_ = total_vram_bytes_ > it->second.vram_bytes ? (total_vram_bytes_ - it->second.vram_bytes) : 0;
        uploaded_.erase(it);
        evicted = true;
    }
    return evicted;
}

} // namespace mhf::psp::vk
