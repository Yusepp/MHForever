#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#pragma clang diagnostic ignored "-Wshadow"
#endif
#include "vk_context.hpp"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>

namespace mhf::psp::vk {

namespace {

VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                              VkDebugUtilsMessageTypeFlagsEXT /*type*/,
                                              const VkDebugUtilsMessengerCallbackDataEXT *data, void * /*user*/) {
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT && data && data->pMessage) {
        std::fprintf(stderr, "[vk][val] %s\n", data->pMessage);
    }
    return VK_FALSE;
}

bool has_extension(const std::vector<VkExtensionProperties> &exts, const char *name) {
    for (const auto &e : exts) {
        if (std::strcmp(e.extensionName, name) == 0) return true;
    }
    return false;
}

std::filesystem::path default_cache_path(std::uint32_t vendor, std::uint32_t device) {
    const char *base = std::getenv("MHF_DATA_DIR");
    std::filesystem::path dir;
    if (base && base[0] != '\0') {
        dir = std::filesystem::path(base) / "cache";
    } else if (const char *xdg = std::getenv("XDG_CACHE_HOME"); xdg && xdg[0] != '\0') {
        dir = std::filesystem::path(xdg) / "mhforever";
    } else if (const char *home = std::getenv("HOME"); home && home[0] != '\0') {
        dir = std::filesystem::path(home) / ".cache" / "mhforever";
    } else if (const char *local = std::getenv("LOCALAPPDATA"); local && local[0] != '\0') {
        dir = std::filesystem::path(local) / "MHForever" / "cache";
    } else {
        return {};
    }
    char name[64];
    std::snprintf(name, sizeof name, "vk_pipeline_cache_%04x_%04x.bin", vendor, device);
    return dir / name;
}

} // namespace

bool Context::init(SDL_Window *win) {
    window = win;
    if (!SDL_Vulkan_LoadLibrary(nullptr)) {
        std::fprintf(stderr, "[vk] SDL_Vulkan_LoadLibrary failed: %s\n", SDL_GetError());
        return false;
    }
    auto get_proc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(SDL_Vulkan_GetVkGetInstanceProcAddr());
    if (!get_proc) {
        std::fprintf(stderr, "[vk] SDL_Vulkan_GetVkGetInstanceProcAddr returned null: %s\n", SDL_GetError());
        return false;
    }
    volkInitializeCustom(get_proc);

    std::uint32_t inst_ver = VK_API_VERSION_1_0;
    if (vkEnumerateInstanceVersion) vkEnumerateInstanceVersion(&inst_ver);
    if (inst_ver < VK_API_VERSION_1_3) {
        std::fprintf(stderr, "[vk] Vulkan 1.3 required, loader reports %u.%u.%u\n",
                     VK_API_VERSION_MAJOR(inst_ver), VK_API_VERSION_MINOR(inst_ver), VK_API_VERSION_PATCH(inst_ver));
        return false;
    }

    Uint32 sdl_ext_count = 0;
    const char *const *sdl_exts = SDL_Vulkan_GetInstanceExtensions(&sdl_ext_count);
    if (!sdl_exts || sdl_ext_count == 0) {
        std::fprintf(stderr, "[vk] SDL_Vulkan_GetInstanceExtensions failed: %s\n", SDL_GetError());
        return false;
    }

    std::uint32_t avail_ext_count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &avail_ext_count, nullptr);
    std::vector<VkExtensionProperties> avail_exts(avail_ext_count);
    vkEnumerateInstanceExtensionProperties(nullptr, &avail_ext_count, avail_exts.data());

    std::vector<const char *> extensions(sdl_exts, sdl_exts + sdl_ext_count);
    VkInstanceCreateFlags inst_flags = 0;
    if (has_extension(avail_exts, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        inst_flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }

    const bool want_validation = [] {
        const char *v = std::getenv("MHF_VK_VALIDATION");
        return v != nullptr && v[0] != '0';
    }();
    std::vector<const char *> layers;
    bool use_debug_utils = false;
    if (want_validation) {
        std::uint32_t layer_count = 0;
        vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
        std::vector<VkLayerProperties> avail_layers(layer_count);
        vkEnumerateInstanceLayerProperties(&layer_count, avail_layers.data());
        for (const auto &l : avail_layers) {
            if (std::strcmp(l.layerName, "VK_LAYER_KHRONOS_validation") == 0) {
                layers.push_back("VK_LAYER_KHRONOS_validation");
                if (has_extension(avail_exts, VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
                    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
                    use_debug_utils = true;
                }
                break;
            }
        }
    }

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "MHForever";
    app.applicationVersion = VK_MAKE_API_VERSION(0, 0, 12, 0);
    app.pEngineName = "MHForever-VK";
    app.engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
    app.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.flags = inst_flags;
    ici.pApplicationInfo = &app;
    ici.enabledLayerCount = static_cast<std::uint32_t>(layers.size());
    ici.ppEnabledLayerNames = layers.data();
    ici.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
    ici.ppEnabledExtensionNames = extensions.data();

    if (const VkResult r = vkCreateInstance(&ici, nullptr, &instance); r != VK_SUCCESS) {
        std::fprintf(stderr, "[vk] vkCreateInstance failed (%d)\n", static_cast<int>(r));
        return false;
    }
    volkLoadInstance(instance);

    if (use_debug_utils && vkCreateDebugUtilsMessengerEXT) {
        VkDebugUtilsMessengerCreateInfoEXT dci{};
        dci.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        dci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        dci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                          VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                          VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        dci.pfnUserCallback = debug_callback;
        vkCreateDebugUtilsMessengerEXT(instance, &dci, nullptr, &debug_messenger);
    }

    if (!SDL_Vulkan_CreateSurface(window, instance, nullptr, &surface)) {
        std::fprintf(stderr, "[vk] SDL_Vulkan_CreateSurface failed: %s\n", SDL_GetError());
        shutdown();
        return false;
    }

    // Select best physical device supporting Vulkan 1.3 + mandatory features.
    std::uint32_t dev_count = 0;
    vkEnumeratePhysicalDevices(instance, &dev_count, nullptr);
    if (dev_count == 0) {
        std::fprintf(stderr, "[vk] no Vulkan physical devices found\n");
        shutdown();
        return false;
    }
    std::vector<VkPhysicalDevice> devices(dev_count);
    vkEnumeratePhysicalDevices(instance, &dev_count, devices.data());

    int best_score = -1;
    for (VkPhysicalDevice pd : devices) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(pd, &props);
        if (props.apiVersion < VK_API_VERSION_1_3) continue;

        std::uint32_t dext_count = 0;
        vkEnumerateDeviceExtensionProperties(pd, nullptr, &dext_count, nullptr);
        std::vector<VkExtensionProperties> dexts(dext_count);
        vkEnumerateDeviceExtensionProperties(pd, nullptr, &dext_count, dexts.data());
        if (!has_extension(dexts, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) continue;

        VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, nullptr};
        VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
        VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f12};
        vkGetPhysicalDeviceFeatures2(pd, &f2);

        if (!f13.dynamicRendering || !f13.synchronization2) continue;
        if (!f12.timelineSemaphore || !f12.bufferDeviceAddress || !f12.scalarBlockLayout ||
            !f12.descriptorBindingPartiallyBound || !f12.descriptorBindingSampledImageUpdateAfterBind ||
            !f12.runtimeDescriptorArray || !f12.shaderSampledImageArrayNonUniformIndexing) {
            continue;
        }

        std::uint32_t qcount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qcount, nullptr);
        std::vector<VkQueueFamilyProperties> qprops(qcount);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qcount, qprops.data());
        QueueFamilyIndices qf;
        for (std::uint32_t i = 0; i < qcount; ++i) {
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(pd, i, surface, &present);
            if ((qprops[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present && qf.graphics == UINT32_MAX) {
                qf.graphics = i;
            }
            if ((qprops[i].queueFlags & VK_QUEUE_TRANSFER_BIT) && !(qprops[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
                qf.transfer == UINT32_MAX) {
                qf.transfer = i;
            }
            if ((qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(qprops[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
                qf.compute == UINT32_MAX) {
                qf.compute = i;
            }
        }
        if (qf.graphics == UINT32_MAX) continue;
        if (qf.transfer == UINT32_MAX) qf.transfer = qf.graphics;
        if (qf.compute == UINT32_MAX) qf.compute = qf.graphics;

        int score = 0;
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score += 1000;
        else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) score += 500;
        if (has_extension(dexts, VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME)) score += 100;
        if (score > best_score) {
            best_score = score;
            physical_device = pd;
            families = qf;
            caps.api_version = props.apiVersion;
            caps.max_anisotropy = f2.features.samplerAnisotropy ? std::clamp(props.limits.maxSamplerAnisotropy, 1.0f, 16.0f) : 1.0f;
            caps.bc_compression = f2.features.textureCompressionBC == VK_TRUE;
            caps.device_name = props.deviceName;
            caps.vendor_id = props.vendorID;
            caps.device_id = props.deviceID;
        }
    }

    if (physical_device == VK_NULL_HANDLE) {
        std::fprintf(stderr, "[vk] no GPU meets Vulkan 1.3 requirements (dynamicRendering, sync2, BDA, bindless)\n");
        shutdown();
        return false;
    }

    // Query optional extensions on selected GPU.
    std::uint32_t dext_count = 0;
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &dext_count, nullptr);
    std::vector<VkExtensionProperties> dexts(dext_count);
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &dext_count, dexts.data());

    std::vector<const char *> dev_exts = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    if (has_extension(dexts, "VK_KHR_portability_subset")) dev_exts.push_back("VK_KHR_portability_subset");
    if (has_extension(dexts, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) {
        dev_exts.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
        caps.memory_budget = true;
    }

    VkPhysicalDeviceExtendedDynamicState3FeaturesEXT eds3_query{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT, nullptr};
    VkPhysicalDeviceExtendedDynamicState3FeaturesEXT eds3_enable{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT, nullptr};
    if (has_extension(dexts, VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME)) {
        VkPhysicalDeviceFeatures2 qf2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &eds3_query};
        vkGetPhysicalDeviceFeatures2(physical_device, &qf2);
        if (eds3_query.extendedDynamicState3ColorBlendEnable &&
            eds3_query.extendedDynamicState3ColorBlendEquation &&
            eds3_query.extendedDynamicState3ColorWriteMask) {
            dev_exts.push_back(VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME);
            caps.extended_dynamic_state3 = true;
            caps.eds3_color_blend_enable = true;
            caps.eds3_color_blend_equation = true;
            caps.eds3_color_write_mask = true;
            eds3_enable.extendedDynamicState3ColorBlendEnable = VK_TRUE;
            eds3_enable.extendedDynamicState3ColorBlendEquation = VK_TRUE;
            eds3_enable.extendedDynamicState3ColorWriteMask = VK_TRUE;
        }
    }

    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, nullptr};
    void **pnext_tail = &f13.pNext;
    if (caps.extended_dynamic_state3) {
        *pnext_tail = &eds3_enable;
        pnext_tail = &eds3_enable.pNext;
    }

#ifdef VK_EXT_PRESENT_MODE_FIFO_LATEST_READY_EXTENSION_NAME
    VkPhysicalDevicePresentModeFifoLatestReadyFeaturesEXT fifo_lr_query{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_MODE_FIFO_LATEST_READY_FEATURES_EXT, nullptr};
    VkPhysicalDevicePresentModeFifoLatestReadyFeaturesEXT fifo_lr_enable{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_MODE_FIFO_LATEST_READY_FEATURES_EXT, nullptr};
    if (has_extension(dexts, VK_EXT_PRESENT_MODE_FIFO_LATEST_READY_EXTENSION_NAME)) {
        VkPhysicalDeviceFeatures2 qf2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &fifo_lr_query};
        vkGetPhysicalDeviceFeatures2(physical_device, &qf2);
        if (fifo_lr_query.presentModeFifoLatestReady) {
            dev_exts.push_back(VK_EXT_PRESENT_MODE_FIFO_LATEST_READY_EXTENSION_NAME);
            caps.present_fifo_latest_ready = true;
            fifo_lr_enable.presentModeFifoLatestReady = VK_TRUE;
            *pnext_tail = &fifo_lr_enable;
            pnext_tail = &fifo_lr_enable.pNext;
        }
    }
#endif

#if defined(VK_KHR_PRESENT_ID_EXTENSION_NAME) && defined(VK_KHR_PRESENT_WAIT_EXTENSION_NAME)
    VkPhysicalDevicePresentIdFeaturesKHR present_id_query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR, nullptr};
    VkPhysicalDevicePresentWaitFeaturesKHR present_wait_query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES_KHR,
                                                              &present_id_query};
    VkPhysicalDevicePresentIdFeaturesKHR present_id_enable{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR, nullptr};
    VkPhysicalDevicePresentWaitFeaturesKHR present_wait_enable{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES_KHR, nullptr};
    if (has_extension(dexts, VK_KHR_PRESENT_ID_EXTENSION_NAME) &&
        has_extension(dexts, VK_KHR_PRESENT_WAIT_EXTENSION_NAME)) {
        VkPhysicalDeviceFeatures2 qf2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &present_wait_query};
        vkGetPhysicalDeviceFeatures2(physical_device, &qf2);
        if (present_id_query.presentId && present_wait_query.presentWait) {
            dev_exts.push_back(VK_KHR_PRESENT_ID_EXTENSION_NAME);
            dev_exts.push_back(VK_KHR_PRESENT_WAIT_EXTENSION_NAME);
            caps.present_id = true;
            caps.present_wait = true;
            present_id_enable.presentId = VK_TRUE;
            present_wait_enable.presentWait = VK_TRUE;
            *pnext_tail = &present_id_enable;
            pnext_tail = &present_id_enable.pNext;
            *pnext_tail = &present_wait_enable;
            pnext_tail = &present_wait_enable.pNext;
        }
    }
#endif

    f13.dynamicRendering = VK_TRUE;
    f13.synchronization2 = VK_TRUE;

    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
    f12.timelineSemaphore = VK_TRUE;
    f12.bufferDeviceAddress = VK_TRUE;
    f12.scalarBlockLayout = VK_TRUE;
    f12.descriptorIndexing = VK_TRUE;
    f12.descriptorBindingPartiallyBound = VK_TRUE;
    f12.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
    f12.runtimeDescriptorArray = VK_TRUE;
    f12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;

    VkPhysicalDeviceFeatures2 avail_f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, nullptr};
    vkGetPhysicalDeviceFeatures2(physical_device, &avail_f2);

    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f12};
    f2.features.depthClamp = avail_f2.features.depthClamp;
    f2.features.samplerAnisotropy = avail_f2.features.samplerAnisotropy;
    f2.features.textureCompressionBC = avail_f2.features.textureCompressionBC;
    f2.features.independentBlend = avail_f2.features.independentBlend;

    std::set<std::uint32_t> unique_families = {families.graphics, families.transfer, families.compute};
    std::vector<VkDeviceQueueCreateInfo> qcis;
    const float prio = 1.0f;
    for (std::uint32_t fam : unique_families) {
        VkDeviceQueueCreateInfo qci{};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = fam;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        qcis.push_back(qci);
    }

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext = &f2;
    dci.queueCreateInfoCount = static_cast<std::uint32_t>(qcis.size());
    dci.pQueueCreateInfos = qcis.data();
    dci.enabledExtensionCount = static_cast<std::uint32_t>(dev_exts.size());
    dci.ppEnabledExtensionNames = dev_exts.data();

    if (const VkResult r = vkCreateDevice(physical_device, &dci, nullptr, &device); r != VK_SUCCESS) {
        std::fprintf(stderr, "[vk] vkCreateDevice failed (%d)\n", static_cast<int>(r));
        shutdown();
        return false;
    }
    volkLoadDevice(device);

    vkGetDeviceQueue(device, families.graphics, 0, &graphics_queue);
    vkGetDeviceQueue(device, families.transfer, 0, &transfer_queue);
    vkGetDeviceQueue(device, families.compute, 0, &compute_queue);

    // Initialize VMA with volk dynamic pointers + BDA (+ optional memory budget).
    VmaVulkanFunctions vma_funcs{};
    vma_funcs.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vma_funcs.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo aci{};
    aci.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    if (caps.memory_budget) aci.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
    aci.physicalDevice = physical_device;
    aci.device = device;
    aci.instance = instance;
    aci.vulkanApiVersion = VK_API_VERSION_1_3;
    aci.pVulkanFunctions = &vma_funcs;
    if (const VkResult r = vmaCreateAllocator(&aci, &allocator); r != VK_SUCCESS) {
        std::fprintf(stderr, "[vk] vmaCreateAllocator failed (%d)\n", static_cast<int>(r));
        shutdown();
        return false;
    }

    // Timeline semaphores for graphics and transfer queues.
    VkSemaphoreTypeCreateInfo stci{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, nullptr, VK_SEMAPHORE_TYPE_TIMELINE, 0};
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &stci, 0};
    if (vkCreateSemaphore(device, &sci, nullptr, &graphics_timeline) != VK_SUCCESS ||
        vkCreateSemaphore(device, &sci, nullptr, &transfer_timeline) != VK_SUCCESS) {
        std::fprintf(stderr, "[vk] failed to create timeline semaphores\n");
        shutdown();
        return false;
    }

    // Load persistent pipeline cache if present.
    pipeline_cache_path = default_cache_path(caps.vendor_id, caps.device_id);
    std::vector<std::uint8_t> cache_bytes;
    if (!pipeline_cache_path.empty()) {
        std::ifstream in(pipeline_cache_path, std::ios::binary | std::ios::ate);
        if (in) {
            const auto sz = in.tellg();
            if (sz > 0 && sz < 64 * 1024 * 1024) {
                cache_bytes.resize(static_cast<std::size_t>(sz));
                in.seekg(0);
                in.read(reinterpret_cast<char *>(cache_bytes.data()), sz);
            }
        }
    }
    VkPipelineCacheCreateInfo pcci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO, nullptr, 0,
                                   cache_bytes.size(), cache_bytes.empty() ? nullptr : cache_bytes.data()};
    vkCreatePipelineCache(device, &pcci, nullptr, &pipeline_cache);

    std::fprintf(stderr,
                 "[vk] %s (Vulkan %u.%u.%u, EDS3=%s, BC=%s, MemBudget=%s, DedicatedTransfer=%s, AsyncCompute=%s, "
                 "FifoLatestReady=%s, PresentWait=%s)\n",
                 caps.device_name.c_str(),
                 VK_API_VERSION_MAJOR(caps.api_version), VK_API_VERSION_MINOR(caps.api_version),
                 VK_API_VERSION_PATCH(caps.api_version),
                 caps.extended_dynamic_state3 ? "yes" : "no",
                 caps.bc_compression ? "yes" : "no",
                 caps.memory_budget ? "yes" : "no",
                 families.has_dedicated_transfer() ? "yes" : "no",
                 families.has_dedicated_compute() ? "yes" : "no",
                 caps.present_fifo_latest_ready ? "yes" : "no",
                 caps.present_wait ? "yes" : "no");
    return true;
}

void Context::save_pipeline_cache() const {
    if (device == VK_NULL_HANDLE || pipeline_cache == VK_NULL_HANDLE || pipeline_cache_path.empty()) return;
    std::size_t size = 0;
    if (vkGetPipelineCacheData(device, pipeline_cache, &size, nullptr) != VK_SUCCESS || size == 0) return;
    std::vector<std::uint8_t> data(size);
    if (vkGetPipelineCacheData(device, pipeline_cache, &size, data.data()) != VK_SUCCESS) return;
    std::error_code ec;
    std::filesystem::create_directories(pipeline_cache_path.parent_path(), ec);
    std::ofstream out(pipeline_cache_path, std::ios::binary | std::ios::trunc);
    if (out) out.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(size));
}

void Context::wait_idle() const {
    if (device != VK_NULL_HANDLE) vkDeviceWaitIdle(device);
}

void Context::wait_timeline(VkSemaphore sem, std::uint64_t value) const {
    if (value == 0 || sem == VK_NULL_HANDLE) return;
    VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, nullptr, 0, 1, &sem, &value};
    vkWaitSemaphores(device, &wi, UINT64_MAX);
}

std::uint64_t Context::query_timeline(VkSemaphore sem) const {
    std::uint64_t v = 0;
    if (sem != VK_NULL_HANDLE) vkGetSemaphoreCounterValue(device, sem, &v);
    return v;
}

void Context::shutdown() {
    if (device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device);
        save_pipeline_cache();
        if (pipeline_cache != VK_NULL_HANDLE) {
            vkDestroyPipelineCache(device, pipeline_cache, nullptr);
            pipeline_cache = VK_NULL_HANDLE;
        }
        if (graphics_timeline != VK_NULL_HANDLE) {
            vkDestroySemaphore(device, graphics_timeline, nullptr);
            graphics_timeline = VK_NULL_HANDLE;
        }
        if (transfer_timeline != VK_NULL_HANDLE) {
            vkDestroySemaphore(device, transfer_timeline, nullptr);
            transfer_timeline = VK_NULL_HANDLE;
        }
        if (allocator != VK_NULL_HANDLE) {
            vmaDestroyAllocator(allocator);
            allocator = VK_NULL_HANDLE;
        }
        vkDestroyDevice(device, nullptr);
        device = VK_NULL_HANDLE;
    }
    if (surface != VK_NULL_HANDLE && instance != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(instance, surface, nullptr);
        surface = VK_NULL_HANDLE;
    }
    if (debug_messenger != VK_NULL_HANDLE && instance != VK_NULL_HANDLE && vkDestroyDebugUtilsMessengerEXT) {
        vkDestroyDebugUtilsMessengerEXT(instance, debug_messenger, nullptr);
        debug_messenger = VK_NULL_HANDLE;
    }
    if (instance != VK_NULL_HANDLE) {
        vkDestroyInstance(instance, nullptr);
        instance = VK_NULL_HANDLE;
    }
}

void Context::cmd_transition_image(VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspect,
                                   VkImageLayout old_layout, VkImageLayout new_layout,
                                   VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                                   VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access,
                                   std::uint32_t base_mip, std::uint32_t mip_count) {
    VkImageMemoryBarrier2 b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    b.srcStageMask = src_stage;
    b.srcAccessMask = src_access;
    b.dstStageMask = dst_stage;
    b.dstAccessMask = dst_access;
    b.oldLayout = old_layout;
    b.newLayout = new_layout;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {aspect, base_mip, mip_count, 0, 1};

    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd, &dep);
}

} // namespace mhf::psp::vk
