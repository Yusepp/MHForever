# MHForever Vulkan Backend Architecture & Implementation Plan (`feat/vulkan_backedn`)

> **Status:** Experimental design & implementation plan on branch `feat/vulkan_backedn` (branched from `dev` at `93088e8`).
> **Goal:** Add a **modern Vulkan 1.3+ backend** alongside the existing OpenGL 3.3 and CPU software backends — not as a 1:1 OpenGL port, but engineered around Vulkan-native paradigms (dynamic rendering, bindless textures, Buffer Device Address, VMA, async transfers, GPU compute interpolation, and decoupled presentation) for lower CPU overhead, lower VRAM usage, zero shader hitching, and steady 120–180 FPS.

---

## 1. Audit of the Current OpenGL 3.3 Backend

The current hardware renderer lives in [`platforms/psp/frontend/gl_backend.cpp`](platforms/psp/frontend/gl_backend.cpp) (6,561 lines) behind the [`mhf::psp::gpu::Backend`](platforms/psp/gpu/backend.hpp) interface and is wired through [`platforms/psp/frontend/psp_frontend.cpp`](platforms/psp/frontend/psp_frontend.cpp) and [`core/host/sdl_host.cpp`](core/host/sdl_host.cpp).

### 1.1 What `GlBackend` Currently Does
- **Context & Threading:** Owns a single-threaded OpenGL 3.3 Core context on the GE thread (`SDL_WINDOW_OPENGL`, `SDL_GL_CreateContext`, `SDL_GL_SetSwapInterval(0)`).
- **GE Pipeline Boundary:** [`gpu::Ge`](platforms/psp/gpu/ge.hpp) executes PSP display lists, decodes vertices, runs transform/lighting/clipping, and emits screen-space primitives (`triangle`, `triangle_unclipped`, `sprite`, `block_transfer`, `flush`, `present`) to `gpu::Backend`.
- **Render Passes & Shaders (7 GLSL 330 programs):**
  1. `kDrawVertex` / `kDrawFragment` — Main PSP draw uber-shader (screen-space `GpuVertex`, 5 texture env modes, specular, fog, alpha test with bitmask, z-range clip, clear mode).
  2. `kShadowMapVertex` / `kShadowMapFragment` — Sun-space shadow map + scenery-kind mask (up to 8192×8192) with alpha-tested foliage cutouts and slope depth push.
  3. `kShadowFactorFragment` — Full/half-res shadow evaluation (eye-space depth reconstruction, 8-tap blocker search, 16-tap variable-penumbra PCF, 12-tap player shadow map).
  4. `kShadowFacingVertex` / `kShadowFacingFragment` — Sun-facing shading & air-clear geometry pass against scene depth.
  5. `kShadowApplyFragment` — 3×3 scenery blur + depth-aware bilateral upsample.
  6. `kPresentVertex` / `kPresentFragment` & `kOverlayFragment` — Letterboxed/widescreen final presentation + premultiplied RGBA system dialogs, in-game menu, hunting horn songs, sharpness bar, and gathering hints.
  7. `kBlurVertex` / `kBlurFragment` — Two-pass separable 5-tap Gaussian blur for the paused in-game menu backdrop.
- **Non-GL Logic Currently Embedded Inside `gl_backend.cpp` (~65% of the file):**
  - Texture pack lookup, hash matching, KTX2/PNG background decoding, and HD text glyph sheet polling.
  - HUD scaling (`size_hud`), widescreen 2D/weather heuristics (`overlay_stretch`, `figure_draws_`).
  - Projected shadow zone detection, sun-angle tables, caster classification, and player piece pairing.
  - Frame interpolation tape recording, `v2` draw matching ([`frame_matching.hpp`](platforms/psp/frontend/frame_matching.hpp)), and `place_v2` vertex re-projection.

### 1.2 Structural Bottlenecks in OpenGL 3.3
1. **Per-batch CPU & driver state thrashing:** Every batch in `emit_state()` (`gl_backend.cpp:4150`) issues `glUseProgram`, `glBindVertexArray`, `glBindTexture`, `glTexParameteri` (mutating texture-object sampler state!), `glBindFramebuffer`, ~15 individual `glUniform*` calls, `glBufferData(..., GL_STREAM_DRAW)` (driver-managed buffer orphaning), and `glDrawArrays`.
2. **Single-threaded GE + Frame Interpolation contention:** Because OpenGL contexts are bound to one thread, `render_between()` and `place_v2()` execute on the same GE thread that processes the next 30 Hz game display list. As noted in [`docs/10-roadmap.md`](docs/10-roadmap.md), heavy scenes spend ~10 ms/frame in GE + GL submission, capping 180 fps at ~150 fps and causing frame-pacing jitter.
3. **CPU-bound `place_v2` & shadow uploads:** Every intermediate frame (`t = 1/N .. (N-1)/N`) re-projects all interpolated vertices on the CPU and re-uploads them to the GPU; shadow caster geometry is likewise re-uploaded from CPU vectors every flip.
4. **Implicit synchronization & driver stalls:** Render-to-texture self-sampling, `glBlitFramebuffer` block transfers, and `glReadPixels` (`copy_shown_to_vram`, `MHF_GL_READBACK=1`) rely on opaque driver hazard tracking and synchronous CPU stalls.
5. **No persistent shader/pipeline cache:** GLSL text is compiled on every startup, and state-dependent driver recompilations happen lazily mid-frame.
6. **Opaque VRAM management & uncompressed texture packs:** Texture packs (2–4 GB of KTX2/PNG) are decoded to uncompressed RGBA8 on CPU threads and uploaded synchronously on the GE thread via `glTexImage2D`, consuming 4×–6× more VRAM than necessary and causing zone-entry micro-stutters.

---

## 2. Target Vulkan Version, Extensions & Dependencies

### 2.1 Target API Baseline: **Vulkan 1.3** (with Optional **Vulkan 1.4** Fast Paths)
Vulkan 1.3 eliminates almost all legacy Vulkan 1.0 boilerplate (`VkRenderPass`, `VkFramebuffer`, legacy barriers) and is supported on every target platform: Linux (Mesa RADV/ANV/NVK, Bazzite/Steam Deck, NVIDIA 515+) and Windows 10/11 (AMD, NVIDIA, Intel).

#### Mandatory Vulkan 1.3 Core Features
| Feature | Purpose in MHForever |
| :--- | :--- |
| **Dynamic Rendering** (`VK_KHR_dynamic_rendering`) | Zero `VkRenderPass` / `VkFramebuffer` object explosion when switching PSP render targets, shadow maps, blur targets, or changing internal resolution live (1×–10×). |
| **Synchronization2** (`VK_KHR_synchronization2`) | Clean `VkImageMemoryBarrier2` / `VkDependencyInfo` for render-to-texture self-sampling, shadow passes, and async compute/transfer barriers. |
| **Timeline Semaphores** (`VK_KHR_timeline_semaphore`) | Single monotonic timeline per queue (graphics, transfer, compute/present) replacing binary fence juggling. |
| **Descriptor Indexing** (`VK_EXT_descriptor_indexing`) | Global bindless `texture2D[]` array (`PARTIALLY_BOUND`, `UPDATE_AFTER_BIND`, `RUNTIME_DESCRIPTOR_ARRAY`, `SAMPLED_IMAGE_ARRAY_NON_UNIFORM_INDEXING`). |
| **Buffer Device Address** (`VK_KHR_buffer_device_address`) | 64-bit GPU pointers (`VkDeviceAddress`) for per-batch draw parameters, vertex streams, and compute-shader interpolation buffers — zero uniform descriptor sets. |
| **Extended Dynamic State 1 & 2** (core in 1.3) | Dynamic viewport, scissor, depth test/write/compare op, cull mode, primitive topology — slashes graphics pipeline permutations. |
| **`depthClamp` & `samplerAnisotropy`** | Exact parity with `GL_DEPTH_CLAMP` and up to 16× anisotropic filtering on texture packs. |

#### Optional Extensions & Vulkan 1.4 Fast Paths (Detected at Startup, Graceful Fallback)
| Extension / Feature | Benefit When Present | Fallback |
| :--- | :--- | :--- |
| **`VK_EXT_extended_dynamic_state3`** | Dynamic blend enable, blend equations, blend factors, and color write mask — collapses PSP blend/mask states to **1 pipeline per shader specialization**. | Small cached pipeline map keyed by `(blend_state, color_mask)`. |
| **`VK_KHR_dynamic_rendering_local_read`** (Vulkan 1.4 core) | Tile-local attachment reads for shadow factor + apply on TBDR GPUs (Steam Deck / mobile / integrated GPUs). | Standard `SHADER_READ_ONLY_OPTIMAL` transition. |
| **`VK_EXT_memory_budget`** | Live VRAM usage & budget telemetry through VMA for smart texture-pack eviction. | Fixed configurable byte budget (like OpenGL's `trim_pack_memory`). |
| **`VK_KHR_present_id` / `VK_KHR_present_wait`** & **`VK_EXT_present_mode_fifo_latest_ready`** | Accurate, low-latency frame pacing for 60/120/180 fps without busy-wait spinning. | `MAILBOX` / `FIFO` / `IMMEDIATE` + `steady_clock` pacing. |
| **`VK_EXT_graphics_pipeline_library`** | Fast asynchronous pipeline linking if fallback blend variants are ever compiled on the fly. | Persistent on-disk `VkPipelineCache`. |
| **`VK_KHR_portability_enumeration` / `subset`** | Future-proof macOS / MoltenVK compatibility. | Standard enumeration on Linux/Windows. |

### 2.2 Third-Party Libraries & Cross-Compilation Strategy
All additions follow [`CONTRIBUTING.md`](CONTRIBUTING.md), [`THIRD_PARTY.md`](THIRD_PARTY.md), and our Linux Clang + Windows Clang-CL cross-compilation setup:

1. **`volk` (MIT License):** Meta-loader with `VK_NO_PROTOTYPES`. Initialized via SDL3:
   ```cpp
   SDL_Vulkan_LoadLibrary(nullptr);
   auto get_proc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(SDL_Vulkan_GetVkGetInstanceProcAddr());
   volkInitializeCustom(get_proc);
   ```
   - **Zero link-time dependency on `vulkan-1.lib` or `libvulkan.so`** — works seamlessly in our Linux→Windows Clang-CL cross-compile container and lets the binary start cleanly (falling back to OpenGL/Software) on machines without a Vulkan driver.
2. **`Vulkan-Headers` (Apache-2.0):** Header-only pinned version (Vulkan 1.4 headers with `VK_API_VERSION_1_3` requested at runtime, querying 1.4 features if available).
3. **`VulkanMemoryAllocator` / VMA (MIT License):** Single header (`vk_mem_alloc.h`) for all `VkBuffer` and `VkImage` allocations, sub-allocation, defragmentation, and `VK_EXT_memory_budget` tracking.
4. **Precompiled SPIR-V Shaders:** Author shaders in GLSL 460 (`#version 460`, `GL_EXT_buffer_reference`, `GL_EXT_nonuniform_qualifier`) under `platforms/psp/frontend/vulkan/shaders/`, compiled offline/at build time via `glslc` / `glslangValidator` into embedded `constexpr uint32_t[]` headers (`vk_shaders_spv.hpp`). Zero runtime compiler dependency, zero startup shader-parse latency, and identical bytecode across Linux & Windows.

---

## 3. Architectural Refactor: Decouple Shared Frontend Logic First

Before implementing `VkBackend`, we should **not** duplicate 4,500+ lines of game-specific heuristics from `gl_backend.cpp`. Instead, extract the backend-agnostic logic into shared modules under `platforms/psp/frontend/`:

```text
platforms/psp/frontend/
├── gpu_common.hpp / .cpp         # GpuVertex, DrawKey, DrawSetup (backend-agnostic TextureHandle),
│                                 # Camera, Origin, TapeEntry, pixel packing/unpacking
├── hud_layout.hpp / .cpp         # size_hud(), window_row(), widescreen overlay_stretch & figure_draws
├── shadow_planner.hpp / .cpp     # zone detection, sun tables, caster classification, pair_player_pieces
├── texture_manager.hpp / .cpp    # pack lookup, hash tracking, background KTX2/PNG decode queue, HD text
├── gl_backend.hpp / .cpp         # Thin OpenGL 3.3 backend (~1,800 lines after extraction)
├── vk_backend.hpp / .cpp         # Modern Vulkan 1.3+ backend (entry point & gpu::Backend impl)
└── vulkan/
    ├── vk_context.hpp / .cpp     # volk, VkInstance, physical/logical device, queues, VMA, pipeline cache
    ├── vk_swapchain.hpp / .cpp   # SDL_Vulkan_CreateSurface, swapchain lifecycle, present mode & resize
    ├── vk_descriptors.hpp / .cpp # Global bindless texture array + immutable sampler cache
    ├── vk_buffers.hpp / .cpp     # Per-frame mapped ring buffers (vertices, BDA DrawParams, staging)
    ├── vk_targets.hpp / .cpp     # RenderTarget pool, layout tracker, RTT self-sampling, block transfers
    ├── vk_pipelines.hpp / .cpp   # Specialized SPIR-V pipelines, dynamic state 1/2/3, disk cache
    ├── vk_shadows.hpp / .cpp     # Device-local caster buffers + compute shadow factor/apply
    ├── vk_interpolate.hpp / .cpp # Decoupled presenter thread + GPU compute place_v2
    └── shaders/                  # GLSL 460 sources + generated SPIR-V headers
```

### Window & Backend Selection Wiring
- **`core/host/sdl_host.hpp` / `.cpp`:** Replace `bool opengl` with `enum class GraphicsApi { Software, OpenGL, Vulkan };`:
  - `OpenGL` → `SDL_WINDOW_OPENGL`
  - `Vulkan` → `SDL_WINDOW_VULKAN`
  - `Software` → SDL software renderer
- **`platforms/psp/frontend/psp_frontend.hpp` / `.cpp`:** Add backend selection to `Frontend::open(...)`. If `Vulkan` is requested and `make_vk_backend(...)` returns `nullptr` (missing driver or < Vulkan 1.3), log the exact reason and cleanly fall back to OpenGL (recreating the SDL window with `SDL_WINDOW_OPENGL`) or Software.
- **CLI & Launcher:**
  - CLI: `--backend opengl|vulkan|software` (keeping `--software` as an alias for `--backend software`; default remains `opengl` while experimental).
  - Launcher: `graphicsBackend` setting (`"opengl"` | `"vulkan"` | `"software"`, default `"opengl"`) in `launcher.cpp`, `cli.cpp`, `bridge.ts`, and `SettingsPage.tsx`.

---

## 4. Vulkan-Native Paradigms (Why This Is Not a 1:1 OpenGL Port)

### 4.1 Bindless Texturing & Immutable Sampler Cache
- **Descriptor Set 0 (Global, bound once per command buffer):**
  - `binding = 0`: `VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE` array of **16,384 slots** (`PARTIALLY_BOUND | UPDATE_AFTER_BIND`). Every decoded PSP texture, texture-pack image, HD text atlas, render-target color/depth view, and shadow map is assigned a slot index (`uint32_t`). Slot 0 is a 1×1 opaque white fallback texture.
  - `binding = 1`: `VK_DESCRIPTOR_TYPE_SAMPLER` array of **immutable `VkSampler`s** pre-created at startup for all combinations of `(filter: nearest|linear, mipmap: none|linear, wrapS: repeat|clamp, wrapT: repeat|clamp, anisotropy: 1|2|4|8|16)`.
- **Result:** Zero `glTexParameteri` mutations, zero per-draw `vkCmdBindDescriptorSets`, and zero descriptor pool exhaustion. Slots are recycled via a lock-free/timeline-gated free list (a freed slot is only reused after the GPU timeline semaphore passes its last-used frame).

### 4.2 Buffer Device Address (BDA) + Ring-Buffered Streaming
- **Frames in Flight (`kFramesInFlight = 2`):** Each frame slice owns:
  1. **Command pool + primary command buffer** (reset with one `vkResetCommandPool` call per frame — zero per-command allocation).
  2. **Vertex ring buffer** (`VMA_MEMORY_USAGE_AUTO`, `HOST_ACCESS_SEQUENTIAL_WRITE_BIT | MAPPED_BIT`, e.g., 16 MB): `GpuVertex` data is `memcpy`'d directly into persistently mapped memory (ReBAR/host-visible VRAM on discrete GPUs, zero-copy unified memory on Steam Deck/APUs!).
  3. **Draw-Params ring buffer** (`DrawParams` SSBO, 64-byte aligned, accessed via `uint64_t draw_params_bda` in push constants): holds `uv_xform`, `tex_index`, `sampler_index`, `env_color`, `fog_color`, `alpha_ref_mask`, `zrange`, `target_size`, and packed flags.
- **Push Constants (≤ 128 bytes, guaranteed everywhere):**
  ```glsl
  layout(push_constant) uniform Push {
      uint64_t draw_params_addr; // BDA to DrawParams[]
      uint32_t draw_index;       // index into DrawParams[]
      uint32_t flags;
  } pc;
  ```
- **Batch Merging / Multi-Draw:** Because textures, samplers, and draw parameters are indexed by `draw_index`, consecutive PSP batches that share the same pipeline/scissor/target no longer need to flush or rebind anything — they can either issue back-to-back `vkCmdDraw` with only a 4-byte push-constant bump, or even merge into a single `vkCmdDrawIndirect` / `vkCmdDrawIndexedIndirect` (using `gl_DrawID` to index `DrawParams[]`)!

### 4.3 Specialized SPIR-V Pipelines + Dynamic State 3 + Persistent Cache
- **Problem in OpenGL:** `kDrawFragment` branches on `u_clear`, `u_tex_enable`, `u_tex_func` (5 modes), `u_tex_rgba`, `u_tex_double`, `u_fog_enable`, `u_alpha_func` (8 modes), and `u_zrange_enable` on every single pixel.
- **Vulkan Solution:**
  - Use **SPIR-V Specialization Constants** for the small set of hot fragment permutations (e.g., `CLEAR`, `TEX_ENABLE`, `TEX_FUNC`, `ALPHA_TEST`, `FOG`, `ZRANGE`) so the GPU compiles tight, branch-free fragment shaders.
  - Use **Dynamic State 1/2/3** (`vkCmdSetViewport`, `vkCmdSetScissor`, `vkCmdSetDepthTestEnable`, `vkCmdSetDepthWriteEnable`, `vkCmdSetDepthCompareOp`, `vkCmdSetColorBlendEnableEXT`, `vkCmdSetColorBlendEquationEXT`, `vkCmdSetColorWriteMaskEXT`) so PSP blend/depth/mask register changes **never** trigger pipeline switches or pipeline compilation!
  - Persist `VkPipelineCache` to `<data_dir>/cache/vk_pipeline_cache_<vendor>_<device>.bin` so every subsequent launch has **zero pipeline compilation stutter**.

### 4.4 Async Texture-Pack Streaming & Native BCn GPU Compression
- **Async Transfer Queue:** If the physical device exposes a dedicated transfer queue family (or secondary queue), decoded texture-pack images are staged and `vkCmdCopyBufferToImage` + mip-upload are submitted on the **transfer queue** with a timeline semaphore. The bindless descriptor slot is updated only once the transfer completes (showing the original PSP texture for 1–2 frames until ready — zero hitching when entering new zones!).
- **Native BC7 / BC3 / BC1 from KTX2:** [`texture_images.cpp`](platforms/psp/frontend/texture_images.cpp) already uses Basis Universal (`mhf_basisu`). Instead of transcoding KTX2 to uncompressed RGBA8 on the CPU, transcode directly to **`VK_FORMAT_BC7_UNORM_BLOCK`** (or `BC3`/`BC1` for alpha/opaque) when `textureCompressionBC` is supported (100% of desktop Linux/Windows GPUs).
  - **Impact:** **4×–6× less VRAM** for HD texture packs (e.g., a 2.6 GB KHD pack fits comfortably in ~500 MB VRAM), **4×–6× faster upload**, and faster GPU texture-cache hit rates!
- **VMA Budget-Aware Eviction:** Query `vmaGetHeapBudgets()` (`VK_EXT_memory_budget`) to evict LRU pack textures smoothly before hitting system-RAM spillover.

### 4.5 Explicit Render-Target Layout Tracker & Self-Sampling
- Every PSP framebuffer VRAM offset maps to a `VkRenderTarget` (upscaled `R8G8B8A8_UNORM` color + `D32_SFLOAT` depth, both with `SAMPLED_BIT | TRANSFER_SRC_BIT | TRANSFER_DST_BIT`).
- A lightweight **Layout Tracker** records each image's current `VkImageLayout`, last-writing stage/access, and emits minimal `vkCmdPipelineBarrier2` transitions only when an image changes role:
  - Drawing → `COLOR_ATTACHMENT_OPTIMAL` / `DEPTH_STENCIL_ATTACHMENT_OPTIMAL`
  - Render-to-texture read → `SHADER_READ_ONLY_OPTIMAL`
  - Self-sampling (PSP reading from the active render target) → `vkCmdCopyImage` to a transient copy image + barrier, cleanly isolated from the main pass.
  - Block transfers (`block_transfer`) → `vkCmdBlitImage` / `vkCmdCopyImage` with exact `TRANSFER_SRC` / `TRANSFER_DST` barriers.
  - VRAM readback (`copy_shown_to_vram` / `MHF_VK_READBACK=1`) → asynchronous `vkCmdCopyImageToBuffer` into a ring of host-cached staging buffers (no synchronous `glReadPixels` stall during normal play).

### 4.6 GPU Compute-Accelerated Projected Shadows
- **Device-Local Caster Buffers:** Static scenery shadow casters stay in device-local VRAM across flips (re-uploaded only when the zone/scenery changes), eliminating per-flip CPU→GPU geometry uploads.
- **Shadow Map Pass:** Dynamic rendering depth-only (+ `R8_UNORM` scenery-kind attachment) at 2048² / 4096² / 8192², with `vkCmdSetDepthBias` + alpha-tested cutout pipeline.
- **Compute Shadow Evaluation & Bilateral Upsample:**
  - Replace fullscreen fragment quads (`kShadowFactorFragment` and `kShadowApplyFragment`) with **Compute Shaders** (`8×8` workgroups, `vkCmdDispatch`):
    - Reads scene depth + shadow map + player shadow map directly from bindless descriptors.
    - Uses workgroup `shared` memory (LDS) for the 3×3 scenery blur and depth-aware bilateral upsample, cutting global memory texture fetches by ~5×!
    - Writes the final darkened scene directly via storage image (`imageLoad`/`imageStore`) or blends cleanly in a single pass.
    - On GPUs with an **async compute queue**, shadow map evaluation can overlap with 2D/UI preparation.

### 4.7 Decoupled Presentation Thread + GPU Compute `place_v2` Frame Interpolation
This directly solves the #1 performance ceiling in MHForever's 120/180 fps mode:
- **Two-Thread Vulkan Pipeline:**
  1. **GE Thread:** Runs `Ge::run()`, records the 30 Hz game frame's raw draws + `DrawSummary` + `TapeEntry` into an immutable `FrameSnapshot`, runs `Matcher::match()`, uploads the snapshot's raw unclipped triangles and matched matrix pairs **once per 30 Hz game flip**, and hands the snapshot + timeline value to the Presenter thread. The GE thread **immediately** moves on to the next game frame — never waiting for intermediate 120/180 Hz subframes or vsync!
  2. **Presenter / Interpolation Thread:** Owns its own Vulkan `VkCommandPool` and wakes at the exact target display interval (60 / 120 / 180 Hz). For each intermediate phase `t ∈ (0, 1)`:
     - **Compute Pass (`vkCmdDispatch`):** Runs `place_v2.comp`, which reads the older/newer matched transforms (uploaded via BDA), computes rigid-arc / affine eye-space interpolation per draw, re-projects all raw triangles to screen space, and writes `GpuVertex` directly into a device-local vertex buffer.
     - **Graphics Pass (`vkCmdBeginRendering`):** Renders the interpolated 3D scene + shadows + 2D HUD + overlay + menu blur directly from that device-local vertex buffer and presents to the swapchain (`vkQueuePresentKHR`).
- **Impact:** Zero CPU `place_v2` vertex loops at 120/180 fps, zero CPU→GPU vertex re-uploads on intermediate frames, and zero GE-thread stalls from presentation!

---

## 5. Phased Implementation Roadmap & Verification Gates

| Phase | Deliverables | Verification Gate |
| :--- | :--- | :--- |
| **Phase 0: Shared Refactor & Wiring** | Extract `gpu_common`, `hud_layout`, `shadow_planner`, `texture_manager` from `gl_backend.cpp`. Add `GraphicsApi` enum to `sdl_host` & `psp_frontend`. Add `--backend` CLI flag and `graphicsBackend` launcher setting (default `opengl`). | All existing `ctest` tests pass; OpenGL output & FPS are 100% unchanged; clean `check_repo_content.py`. |
| **Phase 1: Vulkan 1.3 Bootstrap & Present** | Vendor/fetch `volk`, `Vulkan-Headers`, `VMA`. Add SPIR-V shader build pipeline. Implement `vk_context`, `vk_swapchain`, `vk_descriptors`, `vk_buffers`. Render CPU framebuffers (boot screens, PSMF movies) + dialog/menu overlays + clean fallback to OpenGL if Vulkan 1.3 is unavailable. | Game boots to title screen & plays intro video on Linux (Bazzite) & Windows cross-build with zero Vulkan Validation Layer errors. |
| **Phase 2: Core GE 3D/2D Renderer** | Dynamic rendering to upscaled `VkRenderTarget`s (1×–10×), BDA `DrawParams`, bindless texture array + immutable samplers, specialized SPIR-V GE pipelines + dynamic state 1/2/3 + disk `VkPipelineCache`, RTT self-sampling barriers, block transfers, `MHF_VK_READBACK=1`. | Full MHFU gameplay (village + hunting quests) renders identically to OpenGL; `homebrew/getest` pixel-verified; zero validation errors. |
| **Phase 3: Texture Packs, HUD, Widescreen & Blur** | Dedicated async transfer queue for texture-pack streaming, native BC7/BC3 KTX2 uploads via BasisU, VMA `VK_EXT_memory_budget` eviction, HD text, HUD scaling, widescreen, 2-pass menu blur. | KHD 1.3 texture pack loads with zero zone-entry stutter and ~4× lower VRAM usage than OpenGL; live scale/widescreen/HUD toggles work cleanly. |
| **Phase 4: Compute Projected Shadows** | Device-local shadow caster buffers, shadow map pass (2048/4096/8192), compute `shadow_factor.comp` + LDS-cached `shadow_apply.comp`, sun-facing pass, cave/zone transitions. | Visual parity with OpenGL shadows across Pokke Village, Snowy Mountains, Desert, and caves; measurable GPU frame-time reduction at 4× + High shadows. |
| **Phase 5: Decoupled Presenter & GPU Compute Interpolation** | Separate Presenter thread with its own command pool & timeline sync; `place_v2.comp` compute shader for GPU-side vertex re-projection; `MAILBOX` / `FIFO_LATEST_READY` pacing; frame dump support. | Steady 120 / 180 FPS in Pokke Village and quests at 4× resolution with shadows enabled; GE thread decoupled from presentation. |

---

## 6. Key Risks & Mitigations

1. **Driver quirks with `VK_EXT_extended_dynamic_state3`:** Some older drivers don't expose all EDS3 blend features. **Mitigation:** Query `dynamicRendering` (mandatory 1.3) and treat EDS3 as an accelerator with a clean hashed-pipeline fallback backed by `VkPipelineCache`.
2. **Thread safety on `vkQueueSubmit` / `LiveSettings`:** The GE thread and Presenter thread may both submit to the graphics queue on single-queue GPUs. **Mitigation:** Wrap `vkQueueSubmit2` / `vkQueuePresentKHR` behind a lightweight per-queue mutex (sub-microsecond lock) while keeping all command buffer recording lock-free on separate per-thread `VkCommandPool`s.
3. **Cross-compiling Windows releases without a Windows Vulkan SDK:** **Mitigation:** `volk` + `Vulkan-Headers` + `VMA` are 100% header/C source and require **zero** `vulkan-1.lib` import library; `SDL3` dynamically loads `vulkan-1.dll` on Windows at runtime.
4. **Validation & Regression Parity:** **Mitigation:** Support `MHF_VK_VALIDATION=1` in dev builds to enable `VK_LAYER_KHRONOS_validation` + GPU-assisted validation, and `MHF_VK_READBACK=1` for automated pixel comparisons against the software renderer and OpenGL backend.
