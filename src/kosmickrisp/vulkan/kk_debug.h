/*
 * Copyright 2025 LunarG, Inc.
 * SPDX-License-Identifier: MIT
 */
#ifndef KK_DEBUG_H
#define KK_DEBUG_H 1

enum kk_debug {
   /* Print out the NIR from the compiler */
   KK_DEBUG_NIR = 1ull << 0,
   /* Print out the generated MSL source code from the compiler */
   KK_DEBUG_MSL = 1ull << 1,
   /* Forces robustness for all shaders */
   KK_DEBUG_FORCE_ROBUSTNESS = 1ull << 2,
   /* Log sampler min/max reduction emulation events (reduction samplers,
    * emulating programs compiled, re-recorded command buffers) */
   KK_DEBUG_MINMAX = 1ull << 3,
   /* Log command buffers that started late, ran long on the GPU or completed
    * late, and device creations/destructions with their application */
   KK_DEBUG_COMMITS = 1ull << 4,
   /* Log every command buffer's record (encoders, draws, dispatches,
    * operations, kernels and program MSL hashes) when its feedback comes */
   KK_DEBUG_CBS = 1ull << 5,
   /* Log every Metal compile (library, compute or render pipeline) with its
    * MSL hashes, thread, MTL4Compiler and the compiles in flight with it */
   KK_DEBUG_COMPILES = 1ull << 6,
   /* Do not advertise subgroups in the vertex stage (the lane id is recovered
    * there by emit_vertex_lane_id_helper's kk_vertex_lane_id): A/B for the
    * recovery */
   KK_DEBUG_NO_VERTEX_SUBGROUPS = 1ull << 8,
};

extern enum kk_debug kk_mesa_debug_flags;

#define KK_DEBUG(flag) unlikely(kk_mesa_debug_flags &KK_DEBUG_##flag)

enum kk_experimental {
   /* Advertise custom border features */
   KK_EXPERIMENTAL_CUSTOM_BORDER = 1ull << 0,
   /* Using allowGPUOptimizedContents on M5 for this format causes
    * vkCmdCopyImage() tests in CTS to fail, but it is otherwise usable */
   KK_EXPERIMENTAL_M5_OPTIMIZE_E5B9G9R9 = 1ull << 2,
};

extern enum kk_experimental kk_mesa_experimental_flags;

#define KK_EXPERIMENTAL(flag)                                                  \
   unlikely(kk_mesa_experimental_flags &KK_EXPERIMENTAL_##flag)

extern void kk_process_debug_variable(void);

#endif /* KK_DEBUG_H */
