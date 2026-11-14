/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef KK_SHADER_H
#define KK_SHADER_H 1

#include "kk_cb_record.h"
#include "kk_device_memory.h"
#include "kk_private.h"

#include "kosmickrisp/bridge/mtl_format.h"

#include "poly/nir/poly_nir.h"

#include "vk_pipeline_cache.h"

#include "vk_shader.h"

struct kk_cmd_buffer;
struct kk_descriptor_state;

struct kk_tess_info {
   enum tess_primitive_mode mode : 8;
   enum gl_tess_spacing spacing  : 8;
   bool points;
   bool ccw;
};
static_assert(sizeof(struct kk_tess_info) == 4, "packed");

static struct kk_tess_info
kk_tess_info_merge(struct kk_tess_info a, struct kk_tess_info b)
{
   static_assert(TESS_PRIMITIVE_UNSPECIFIED == 0, "zero state");
   static_assert(TESS_SPACING_UNSPECIFIED == 0, "zero state");

   /* Just merge by OR'ing the raw bits */
   uint32_t x, y;
   memcpy(&x, &a, sizeof(x));
   memcpy(&y, &b, sizeof(y));

   x |= y;

   struct kk_tess_info out;
   memcpy(&out, &x, sizeof(out));
   return out;
}

struct kk_shader_info {
   mesa_shader_stage stage;
   bool uses_per_draw_data;

   /* Required for fragment shader cull distance discards. */
   uint8_t num_cull_distances;

   /* VS or CS with a minmax program: descriptor sets whose reduction samplers
    * need it (kk_compile_programs) */
   uint32_t minmax_sets;

   union {
      /* Vertex shader is the pipeline, store all relevant data here. */
      struct {
         /* Required to know which extra states does the pipeline contain for
          * serialization. */
         uint32_t additional_stages_bits;

         uint32_t tess_local_thread_size;

         /* Data needed to start render pass and bind pipeline. */
         uint32_t attribs_read;
         uint32_t sample_count;
         uint64_t outputs_written;

         /* Data needed for serialization. */
         enum mtl_primitive_topology_class topology;
         enum mtl_pixel_format rt_formats[MAX_DRAW_BUFFERS];
         enum mtl_pixel_format d_format;
         enum mtl_pixel_format s_format;
         uint32_t view_mask;
         VkCompareOp depth_compare_op;
         struct {
            uint8_t depth_fail;
            uint8_t fail;
            uint8_t pass;
            uint8_t compare;
            uint8_t compare_mask;
            uint8_t write_mask;
         } stencil_front, stencil_back;
         uint8_t color_attachment_count;
         bool has_ms;
         bool has_ds;
         bool has_depth_write;
         bool has_stencil_test;

         /* Application or generated GS info, valid when additional_stages_bits
          * includes a geometry shader */
         struct poly_gs_info gs;
      } vs;

      struct {
         uint64_t tcs_per_vertex_outputs;
         uint32_t tcs_output_stride;
         uint8_t tcs_output_patch_size;
         uint8_t tcs_nr_patch_outputs;

         struct kk_tess_info info;

         /* Outputs a TES stores for a GS */
         uint64_t tes_outputs;
      } tess;

      struct poly_gs_info gs;

      struct {
         bool uses_flat_varyings;
      } fs;

      struct {
         struct mtl_size local_size;
      } cs;
   };
};

/* Threadgroup size the VS, TES and GS compute passes are compiled for and
 * dispatched with */
#define KK_POLY_WG_SIZE 64u

/* Metal handles for binding. */
struct kk_pipeline_handles {
   union {
      struct {
         mtl_render_pipeline_state *render;
         /* VS, TCS, TES and the GS compute pass at most run before the render
          * pipeline. */
         mtl_compute_pipeline_state *pre_render[4];
         /* XFB pre-GS pipeline and count pipeline, if needed */
         mtl_compute_pipeline_state *gs_count;
         mtl_compute_pipeline_state *pre_gs;
         mtl_depth_stencil_state *ds_handle;
         uint32_t pre_render_count;
      } gfx;
      mtl_compute_pipeline_state *cs;
   };
   /* MSL hashes and features command buffer records name it by */
   struct kk_cb_program record;
};

struct kk_msl_source;

struct msl_compile_data {
   char *code;
   char *entrypoint_name;
   struct kk_msl_source *source;
};

struct kk_shader {
   struct vk_shader vk;

   struct kk_pipeline_handles pipeline;
   struct kk_shader_info info;
   struct msl_compile_data msl_data[MESA_SHADER_STAGES];
   /* Compute pass of a GS. Its rasterization VS is in
    * msl_data[MESA_SHADER_GEOMETRY]. */
   struct msl_compile_data gs_main;
   /* XFB pre-GS and count shader; code is NULL when the program isn't needed. */
   struct msl_compile_data gs_count;
   struct msl_compile_data pre_gs;

   /* VS or CS: the program emulating min/max reduction for samplers from
    * descriptor sets, or NULL (kk_compile_programs) */
   struct kk_shader *minmax;
   /* Metal pipelines created, when minmax isn't NULL (one of the two
    * programs is compiled on first use) */
   bool pipeline_ready;

   /* Background compilation (kk_device::compile_*): the program is queued at
    * creation and the pool compiles it, so the ring thread does not spend a
    * title's whole shader compilation one pipeline at a time. First use waits
    * for compile_done; destruction does not (compile_orphaned). */
   struct kk_shader *compile_next;
   bool compile_queued;
   /* A worker (or a thread that needed it first) is compiling it */
   bool compile_running;
   bool compile_done;
   VkResult compile_result;
   /* Destroyed while queued or compiling: the worker that compiles it frees
    * it afterwards */
   bool compile_orphaned;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(kk_shader, vk.base, VkShaderEXT,
                               VK_OBJECT_TYPE_SHADER_EXT);

extern const struct vk_device_shader_ops kk_device_shader_ops;

/* Sets up and tears down the pool that compiles queued programs. */
void kk_compile_pool_init(struct kk_device *dev);
void kk_compile_pool_finish(struct kk_device *dev);

/* Queues a program's compilation (idempotent) and waits for it. Waiting on a
 * program that was never queued compiles it right there. */
void kk_compile_enqueue(struct kk_device *dev, struct kk_shader *s);
VkResult kk_compile_wait(struct kk_device *dev, struct kk_shader *s);

static inline nir_address_format
kk_buffer_addr_format(VkPipelineRobustnessBufferBehaviorEXT robustness)
{
   switch (robustness) {
   case VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED:
      return nir_address_format_64bit_global_32bit_offset;
   case VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS:
   case VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2:
      return nir_address_format_64bit_bounded_global;
   default:
      UNREACHABLE("Invalid robust buffer access behavior");
   }
}

bool kk_nir_lower_missing_descriptors(
   nir_shader *nir, uint32_t set_layout_count,
   struct vk_descriptor_set_layout *const *set_layouts, bool *reported);

bool
kk_nir_lower_descriptors(nir_shader *nir,
                         const struct vk_pipeline_robustness_state *rs,
                         uint32_t set_layout_count,
                         struct vk_descriptor_set_layout *const *set_layouts);

/* KK_WORKAROUND_18 */
bool kk_nir_add_device_barrier_workaround(nir_shader *nir);

bool kk_nir_lower_poly(struct nir_shader *nir);
bool kk_nir_lower_null_images(nir_shader *nir);

bool kk_nir_lower_textures(nir_shader *nir);
bool kk_nir_lower_sampler_minmax(nir_shader *nir, bool dynamic,
                                 uint32_t set_layout_count,
                                 struct vk_descriptor_set_layout *const *set_layouts,
                                 uint32_t *dynamic_sets_out);

bool kk_nir_lower_vs_multiview(nir_shader *nir, uint32_t view_mask);
bool kk_nir_lower_fs_multiview(nir_shader *nir, uint32_t view_mask);

/* Indicates that the sampler should be overridden to clamp to 0 instead of 1 */
#define KK_TEXTURE_FLAG_CLAMP_TO_0 (1 << 0)

bool kk_nir_lower_custom_border(nir_shader *nir);
bool kk_nir_lower_image_view_min_lod(nir_shader *nir);

VkResult kk_compile_nir_shader(struct kk_device *dev, nir_shader *nir,
                               const VkAllocationCallbacks *alloc,
                               struct kk_shader **shader_out);

/* The device's shared Metal pipeline states (kk_shader.c) */
/* The device's tables of compiled Metal objects (kk_device::msl_libraries and
 * the pipeline state tables); init returns false out of memory. */
bool kk_metal_caches_init(struct kk_device *dev);
void kk_metal_caches_finish(struct kk_device *dev);

void kk_cmd_bind_compute_shader(struct kk_cmd_buffer *cmd,
                                struct kk_shader *shader);

const struct kk_pipeline_handles *
kk_cmd_shader_pipeline(struct kk_cmd_buffer *cmd, struct kk_shader *shader,
                       const struct kk_descriptor_state *desc);

#endif /* KK_SHADER_H */
