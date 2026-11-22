# MHForever Next-Gen Frame Generation & Pacing Architecture (Vulkan 1.3+)

This document specifies MHForever's geometry-level frame generation and hardware-grade frame pacing architecture on the Vulkan 1.3+ backend (`feat/vulkan_backedn`).

Unlike post-process neural frame generators (FSR 3/4 FG, DLSS 3/4 MFG, Lossless Scaling) that infer 2D screen-space motion via optical flow and suffer from HUD ghosting, disocclusion halos, and foliage smearing, MHForever owns the **exact 3D scene graph** (`ge.cpp` + `vk_backend.cpp`): vertices, skinned meshes, camera/world matrices, UVs, vertex colors, fog, depth, and a cleanly separated 2D HUD pass.

---

## 5-Pillar Upgrade Architecture

### Pillar 1 — True 120/180 FPS Skinned Character & Monster Interpolation
- **Problem:** Rigid-only matrix interpolation (`proj * blended_eye * inv(older_world)`) moves the camera and static scenery at 120/180 FPS, but skeletal deformations (hunter swings, monster wings/tails/claws, Palicos, particles) remain stepped at 30 Hz.
- **Solution:**
  - Record per-draw raw vertex ranges (`draw_raw_start`, `draw_raw_count`) in both `blend_older_` and `blend_newer_` snapshots and keep both raw vertex buffers persistently mapped in GPU-accessible memory.
  - For every matched draw (`partner >= 0`) with identical raw vertex count, pass both `mv_old = eye_blend * inv(older_world)` and `mv_new = eye_blend * inv(newer_world)` (as compact 3×4 affine matrices) plus `proj` and `(older_raw_start, newer_raw_start, flags, max_deform_dist)` to `place_v2.comp`.
  - Inside `place_v2.comp`, map both `rv_old.world_pos` and `rv_new.world_pos` into the **same** blended eye space (`eye0`, `eye1`) so rigid camera/body arc motion is factored out and `eye1 - eye0` is purely local skeletal/morph deformation.
  - Blend `mix(eye0, eye1, phase)` with a triangle-level deformation distance guard (`max_deform_dist`) that automatically falls back to rigid `eye0` if topology/order shifts or an object teleports.
  - Blend **UVs** (when `abs(du), abs(dv) < 0.1`, matching `interpolation::kMaxScrollStep` so water/lava/glows scroll smoothly while flipbooks don't smear), **vertex color RGBA**, **specular RGBA**, and **fog** at 120/180 FPS.

### Pillar 2 — Hardware-Grade Frame Pacing (DLSS 4 Flip Metering Equivalent on Vulkan)
- **Problem:** Raw last-frame delta timing jitters on OS wakeup variance, and unbounded `MAILBOX`/`IMMEDIATE` submission can queue extra frames or drop work inconsistently.
- **Solution:**
  - Enable modern Vulkan presentation extensions when supported by the host driver:
    - `VK_EXT_present_mode_fifo_latest_ready` (preferred over `MAILBOX`/`IMMEDIATE`/`FIFO`): tear-free vblank-evaluated latest image.
    - `VK_KHR_present_id` + `VK_KHR_present_wait`: attach monotonically increasing `VkPresentIdKHR` to every `vkQueuePresentKHR` and use `vkWaitForPresentKHR` for hardware scanout backpressure (frame queue depth locked to 1).
  - Replace raw `flip_period_ = dt` with an **EMA-smoothed, drift-clamped phase-locked scheduler** so 30 Hz game ticks subdivide into uniform 8.33 ms (120 FPS) / 5.55 ms (180 FPS) intervals without beat jitter.
  - Future step (FG-2B): Dedicated Presenter thread with lock-free triple-buffered `(older, newer, matching)` snapshots and `VK_EXT_present_timing` absolute scanout deadlines.

### Pillar 3 — Higher-Order Motion Quality (Hermite Splines + GPU Near-Plane Clipping)
- **Velocity-Matched Cubic Hermite Motion (Zero Extra Latency):**
  - Track `prev_cam_motion_` (`F[n-2] -> F[n-1]`) alongside `last_cam_motion_` (`F[n-1] -> F[n]`).
  - When consecutive frames are continuous (no cut), evaluate a velocity-matched Hermite curve for camera rotation/translation phase so curved camera pans and monster arcs are $C^1$-continuous across 30 Hz boundaries without adding any extra frame of latency.
- **True GPU Sutherland-Hodgman Near-Plane Clipping:**
  - Replace triangle culling (`if w <= 1e-4 return`) in `place_v2.comp` with exact clip-space Sutherland-Hodgman clipping against `w = 1e-4`.
  - Deterministically emit 2 output triangles (6 vertices) per input triangle (`out_base = tri * 6`):
    - 0 in front → 2 degenerate triangles
    - 3 in front → 1 triangle + 1 degenerate
    - 1 in front → 1 clipped triangle + 1 degenerate
    - 2 in front → 2 clipped triangles (clipped quad)
  - Zero atomics, zero indirect sync hazards, and **zero foreground ground/fence popping**.

### Pillar 4 — GPU-Driven Subframe Efficiency
- Persistent host-mapped snapshot vertex buffers (`snap_raw_old_`, `snap_raw_new_`) — no per-frame `vmaMapMemory`/`vmaUnmapMemory` churn.
- Redundant Vulkan state elimination in `render_interpolated_subframe()` (bind descriptor set once, cache active `VkPipeline`, vertex buffer, scissor, depth state, and blend state across batches) — cuts subframe CPU recording time by 60–80%.
- Future step (FG-4B): Async Compute queue (`families.compute`) + timeline semaphore overlap and `vkCmdDrawIndirect` bucketed batching.

### Pillar 5 — Input Responsiveness (Optional Low-Latency Mode)
- Keep 1-frame buffered interpolation as the default for zero overshoot artifacts on skinned meshes and monster attacks.
- Future experimental option: Zero-Latency Camera Extrapolation (VR Async Timewarp style micro-rotation from live right-stick/mouse delta on subframes).

---

## Implementation Status

| Phase | Scope | Status |
| :--- | :--- | :--- |
| **FG-1** | Dual-stream per-vertex skinned/deforming + UV/color/fog interpolation in `place_v2.comp` with safety clamp | Implemented |
| **FG-2** | `VK_EXT_present_mode_fifo_latest_ready` + `VK_KHR_present_id`/`VK_KHR_present_wait` + EMA phase-locked pacing | Implemented |
| **FG-3** | Velocity-matched Hermite camera/motion smoothing + GPU Sutherland-Hodgman near-plane clipping | Implemented |
| **FG-4** | Persistent mapped dual snapshot buffers + state-cached subframe recording | Implemented |
| **FG-5** | Optional Async Compute overlap, `VK_EXT_present_timing`, and Bone-Matrix SLERP/DLB | Planned |
