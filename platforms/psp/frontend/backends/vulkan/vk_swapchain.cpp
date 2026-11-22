#include "vk_swapchain.hpp"

#include <algorithm>
#include <cstdio>

namespace mhf::psp::vk {

bool Swapchain::create(const Context &ctx, int width, int height) {
    if (width <= 0 || height <= 0) return false;

    VkSurfaceCapabilitiesKHR caps{};
    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(ctx.physical_device, ctx.surface, &caps) != VK_SUCCESS) return false;

    std::uint32_t fmt_count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(ctx.physical_device, ctx.surface, &fmt_count, nullptr);
    if (fmt_count == 0) return false;
    std::vector<VkSurfaceFormatKHR> formats(fmt_count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(ctx.physical_device, ctx.surface, &fmt_count, formats.data());

    VkSurfaceFormatKHR chosen = formats[0];
    for (const auto &f : formats) {
        if ((f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_UNORM) &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            chosen = f;
            break;
        }
    }

    std::uint32_t mode_count = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(ctx.physical_device, ctx.surface, &mode_count, nullptr);
    std::vector<VkPresentModeKHR> modes(mode_count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(ctx.physical_device, ctx.surface, &mode_count, modes.data());

    // Low-latency tear-free pacing: FIFO_LATEST_READY -> MAILBOX -> IMMEDIATE -> FIFO.
    present_mode = VK_PRESENT_MODE_FIFO_KHR;
    std::vector<VkPresentModeKHR> preferred;
#ifdef VK_EXT_PRESENT_MODE_FIFO_LATEST_READY_EXTENSION_NAME
    if (ctx.caps.present_fifo_latest_ready) {
        preferred.push_back(VK_PRESENT_MODE_FIFO_LATEST_READY_EXT);
    }
#endif
    preferred.push_back(VK_PRESENT_MODE_MAILBOX_KHR);
    preferred.push_back(VK_PRESENT_MODE_IMMEDIATE_KHR);
    for (VkPresentModeKHR want : preferred) {
        if (std::find(modes.begin(), modes.end(), want) != modes.end()) {
            present_mode = want;
            break;
        }
    }
    next_present_id = 1;

    if (caps.currentExtent.width != UINT32_MAX && caps.currentExtent.width > 0 && caps.currentExtent.height > 0) {
        extent = caps.currentExtent;
    } else {
        extent.width = std::clamp(static_cast<std::uint32_t>(width), caps.minImageExtent.width, caps.maxImageExtent.width);
        extent.height = std::clamp(static_cast<std::uint32_t>(height), caps.minImageExtent.height, caps.maxImageExtent.height);
    }

    std::uint32_t img_count = std::max(caps.minImageCount + 1u, 3u);
    if (caps.maxImageCount > 0 && img_count > caps.maxImageCount) img_count = caps.maxImageCount;

    VkSwapchainKHR old = handle;
    VkSwapchainCreateInfoKHR sci{};
    sci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    sci.surface = ctx.surface;
    sci.minImageCount = img_count;
    sci.imageFormat = chosen.format;
    sci.imageColorSpace = chosen.colorSpace;
    sci.imageExtent = extent;
    sci.imageArrayLayers = 1;
    sci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sci.preTransform = caps.currentTransform;
    sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sci.presentMode = present_mode;
    sci.clipped = VK_TRUE;
    sci.oldSwapchain = old;

    VkSwapchainKHR next = VK_NULL_HANDLE;
    if (const VkResult r = vkCreateSwapchainKHR(ctx.device, &sci, nullptr, &next); r != VK_SUCCESS) {
        std::fprintf(stderr, "[vk] vkCreateSwapchainKHR failed (%d)\n", static_cast<int>(r));
        return false;
    }

    if (old != VK_NULL_HANDLE) {
        destroy(ctx);
    }
    handle = next;
    format = chosen.format;

    std::uint32_t actual = 0;
    vkGetSwapchainImagesKHR(ctx.device, handle, &actual, nullptr);
    images.resize(actual);
    vkGetSwapchainImagesKHR(ctx.device, handle, &actual, images.data());

    views.resize(actual, VK_NULL_HANDLE);
    render_finished.resize(actual, VK_NULL_HANDLE);
    VkSemaphoreCreateInfo sem_ci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, nullptr, 0};
    for (std::uint32_t i = 0; i < actual; ++i) {
        VkImageViewCreateInfo ivci{};
        ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ivci.image = images[i];
        ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ivci.format = format;
        ivci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(ctx.device, &ivci, nullptr, &views[i]) != VK_SUCCESS ||
            vkCreateSemaphore(ctx.device, &sem_ci, nullptr, &render_finished[i]) != VK_SUCCESS) {
            destroy(ctx);
            return false;
        }
    }
    return true;
}

void Swapchain::destroy(const Context &ctx) {
    for (VkSemaphore s : render_finished) {
        if (s != VK_NULL_HANDLE) vkDestroySemaphore(ctx.device, s, nullptr);
    }
    render_finished.clear();
    for (VkImageView v : views) {
        if (v != VK_NULL_HANDLE) vkDestroyImageView(ctx.device, v, nullptr);
    }
    views.clear();
    images.clear();
    if (handle != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(ctx.device, handle, nullptr);
        handle = VK_NULL_HANDLE;
    }
}

bool Swapchain::recreate(const Context &ctx, int width, int height) {
    ctx.wait_idle();
    return create(ctx, width, height);
}

} // namespace mhf::psp::vk
