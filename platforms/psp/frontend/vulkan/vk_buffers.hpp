#pragma once

// Persistently mapped per-frame linear ring buffers (vertex streaming, BDA DrawParams, and texture staging).

#include "vk_context.hpp"

#include <cstddef>
#include <cstdint>

namespace mhf::psp::vk {

struct MappedRingBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkDeviceAddress device_address = 0;
    std::uint8_t *mapped = nullptr;
    std::size_t capacity = 0;
    std::size_t offset = 0;

    bool create(const Context &ctx, std::size_t bytes, VkBufferUsageFlags usage);
    void destroy(const Context &ctx);
    void reset() noexcept { offset = 0; }

    struct Slice {
        std::uint8_t *ptr = nullptr;
        std::size_t offset = 0;
        VkDeviceAddress address = 0;
    };
    [[nodiscard]] Slice allocate(std::size_t bytes, std::size_t alignment = 16);
};

} // namespace mhf::psp::vk
