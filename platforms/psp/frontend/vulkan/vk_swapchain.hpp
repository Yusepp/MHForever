#pragma once

// Swapchain management for SDL3 Vulkan surfaces (MAILBOX -> IMMEDIATE -> FIFO, clean live resize/fullscreen).

#include "vk_context.hpp"

#include <cstdint>
#include <vector>

namespace mhf::psp::vk {

struct Swapchain {
    VkSwapchainKHR handle = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_B8G8R8A8_UNORM;
    VkExtent2D extent{0, 0};
    VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
    std::vector<VkImage> images;
    std::vector<VkImageView> views;
    std::vector<VkSemaphore> render_finished; // per swapchain image for safe vkQueuePresentKHR

    bool create(const Context &ctx, int width, int height);
    void destroy(const Context &ctx);
    bool recreate(const Context &ctx, int width, int height);
};

} // namespace mhf::psp::vk
