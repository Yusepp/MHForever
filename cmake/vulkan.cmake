# Vulkan 1.3+ header-only & meta-loader dependencies (no link-time dependency on vulkan-1.lib / libvulkan.so):
# - Vulkan-Headers (Apache-2.0)
# - volk (MIT, meta-loader initialized via SDL_Vulkan_GetVkGetInstanceProcAddr)
# - VulkanMemoryAllocator / VMA (MIT, single-header allocator)
include(FetchContent)
include(${CMAKE_CURRENT_LIST_DIR}/download.cmake)

set(_vk_headers_url https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/v1.4.304.tar.gz)
set(_vk_headers_sha256 5e1a06b3f27e7581b55d1dea176fd97ee0a2f299406db2f437c1d2f297ceba5b)
set(_vk_headers_archive "${CMAKE_BINARY_DIR}/_deps/downloads/Vulkan-Headers-1.4.304.tar.gz")
if(NOT FETCHCONTENT_SOURCE_DIR_VULKAN_HEADERS AND NOT FETCHCONTENT_FULLY_DISCONNECTED)
    mhf_download("${_vk_headers_url}" "${_vk_headers_archive}" "${_vk_headers_sha256}" _vkh_problem)
    if(_vkh_problem)
        message(FATAL_ERROR "mhf: ${_vkh_problem}")
    endif()
endif()
FetchContent_Declare(vulkan_headers
    URL "${_vk_headers_archive}"
    URL_HASH SHA256=${_vk_headers_sha256}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_GetProperties(vulkan_headers)
if(NOT vulkan_headers_POPULATED)
    FetchContent_Populate(vulkan_headers)
endif()

set(_volk_url https://github.com/zeux/volk/archive/refs/tags/vulkan-sdk-1.4.304.0.tar.gz)
set(_volk_sha256 2a563c9dd926c804d91d7b84c59516678ccb32d039038049fc20eb22547c88fb)
set(_volk_archive "${CMAKE_BINARY_DIR}/_deps/downloads/volk-1.4.304.0.tar.gz")
if(NOT FETCHCONTENT_SOURCE_DIR_VOLK AND NOT FETCHCONTENT_FULLY_DISCONNECTED)
    mhf_download("${_volk_url}" "${_volk_archive}" "${_volk_sha256}" _volk_problem)
    if(_volk_problem)
        message(FATAL_ERROR "mhf: ${_volk_problem}")
    endif()
endif()
FetchContent_Declare(volk
    URL "${_volk_archive}"
    URL_HASH SHA256=${_volk_sha256}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_GetProperties(volk)
if(NOT volk_POPULATED)
    FetchContent_Populate(volk)
endif()

set(_vma_url https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/archive/refs/tags/v3.2.1.tar.gz)
set(_vma_sha256 5e7749504cb802427ffb7bec38a0b6a15db46ae253f00560acb3e624d9fe695c)
set(_vma_archive "${CMAKE_BINARY_DIR}/_deps/downloads/VMA-3.2.1.tar.gz")
if(NOT FETCHCONTENT_SOURCE_DIR_VMA AND NOT FETCHCONTENT_FULLY_DISCONNECTED)
    mhf_download("${_vma_url}" "${_vma_archive}" "${_vma_sha256}" _vma_problem)
    if(_vma_problem)
        message(FATAL_ERROR "mhf: ${_vma_problem}")
    endif()
endif()
FetchContent_Declare(vma
    URL "${_vma_archive}"
    URL_HASH SHA256=${_vma_sha256}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_GetProperties(vma)
if(NOT vma_POPULATED)
    FetchContent_Populate(vma)
endif()

add_library(mhf_vulkan_deps STATIC "${volk_SOURCE_DIR}/volk.c")
target_include_directories(mhf_vulkan_deps SYSTEM PUBLIC
    "${vulkan_headers_SOURCE_DIR}/include"
    "${volk_SOURCE_DIR}"
    "${vma_SOURCE_DIR}/include")
target_compile_definitions(mhf_vulkan_deps PUBLIC VK_NO_PROTOTYPES)
if(MSVC)
    target_compile_options(mhf_vulkan_deps PRIVATE /w)
else()
    target_compile_options(mhf_vulkan_deps PRIVATE -w)
endif()
