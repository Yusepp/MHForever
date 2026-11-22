#include "vk_buffers.hpp"

#include <cstdio>

namespace mhf::psp::vk {

bool MappedRingBuffer::create(const Context &ctx, std::size_t bytes, VkBufferUsageFlags usage) {
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = bytes;
    bci.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo vaci{};
    vaci.usage = VMA_MEMORY_USAGE_AUTO;
    vaci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo info{};
    if (const VkResult r = vmaCreateBuffer(ctx.allocator, &bci, &vaci, &buffer, &allocation, &info); r != VK_SUCCESS) {
        std::fprintf(stderr, "[vk] vmaCreateBuffer (%zu bytes) failed (%d)\n", bytes, static_cast<int>(r));
        return false;
    }
    mapped = static_cast<std::uint8_t *>(info.pMappedData);
    capacity = bytes;
    offset = 0;

    VkBufferDeviceAddressInfo bdai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, buffer};
    device_address = vkGetBufferDeviceAddress(ctx.device, &bdai);
    return mapped != nullptr && device_address != 0;
}

void MappedRingBuffer::destroy(const Context &ctx) {
    if (buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(ctx.allocator, buffer, allocation);
        buffer = VK_NULL_HANDLE;
        allocation = VK_NULL_HANDLE;
    }
    mapped = nullptr;
    device_address = 0;
    capacity = 0;
    offset = 0;
}

MappedRingBuffer::Slice MappedRingBuffer::allocate(std::size_t bytes, std::size_t alignment) {
    const std::size_t aligned = (offset + alignment - 1u) & ~(alignment - 1u);
    if (aligned + bytes > capacity) return {};
    offset = aligned + bytes;
    return {mapped + aligned, aligned, device_address + aligned};
}

} // namespace mhf::psp::vk
