#pragma once

// Vulkan 1.3+ context: volk meta-loader (via SDL3), instance, surface, physical/logical device, queues,
// VMA allocator, timeline semaphores, persistent pipeline cache, and Synchronization2 barrier helpers.

#include <volk.h>
#include <vk_mem_alloc.h>

#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

struct SDL_Window;

namespace mhf::psp::vk {

struct QueueFamilyIndices {
    std::uint32_t graphics = UINT32_MAX;
    std::uint32_t transfer = UINT32_MAX;
    std::uint32_t compute = UINT32_MAX;
    bool has_dedicated_transfer() const noexcept { return transfer != UINT32_MAX && transfer != graphics; }
    bool has_dedicated_compute() const noexcept { return compute != UINT32_MAX && compute != graphics; }
};

struct Capabilities {
    std::uint32_t api_version = 0;
    float max_anisotropy = 1.0f;
    bool extended_dynamic_state3 = false;
    bool eds3_color_blend_enable = false;
    bool eds3_color_blend_equation = false;
    bool eds3_color_write_mask = false;
    bool memory_budget = false;
    bool dynamic_rendering_local_read = false;
    bool bc_compression = false;
    bool present_fifo_latest_ready = false;
    bool present_id = false;
    bool present_wait = false;
    std::string device_name;
    std::uint32_t vendor_id = 0;
    std::uint32_t device_id = 0;
};

struct Context {
    SDL_Window *window = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VmaAllocator allocator = VK_NULL_HANDLE;
    VkPipelineCache pipeline_cache = VK_NULL_HANDLE;
    std::filesystem::path pipeline_cache_path;

    QueueFamilyIndices families;
    VkQueue graphics_queue = VK_NULL_HANDLE;
    VkQueue transfer_queue = VK_NULL_HANDLE;
    VkQueue compute_queue = VK_NULL_HANDLE;
    std::mutex graphics_queue_mutex;

    VkSemaphore graphics_timeline = VK_NULL_HANDLE;
    std::uint64_t graphics_timeline_value = 0;
    VkSemaphore transfer_timeline = VK_NULL_HANDLE;
    std::uint64_t transfer_timeline_value = 0;

    Capabilities caps;

    bool init(SDL_Window *win);
    void shutdown();

    void wait_idle() const;
    void wait_timeline(VkSemaphore sem, std::uint64_t value) const;
    [[nodiscard]] std::uint64_t query_timeline(VkSemaphore sem) const;
    void save_pipeline_cache() const;

    // Synchronization2 helper for clean, explicit image layout transitions.
    static void cmd_transition_image(VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspect,
                                     VkImageLayout old_layout, VkImageLayout new_layout,
                                     VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                                     VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access,
                                     std::uint32_t base_mip = 0, std::uint32_t mip_count = 1);
};

} // namespace mhf::psp::vk
