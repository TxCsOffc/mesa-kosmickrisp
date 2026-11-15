/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef KK_CMD_BUFFER_H
#define KK_CMD_BUFFER_H 1

#include "kk_private.h"

#include "kk_descriptor_set.h"
#include "kk_image.h"
#include "kk_nir_lower_vbo.h"
#include "kk_shader.h"

#include "kosmickrisp/bridge/mtl_types.h"

#include "kosmickrisp/libkk/kk_query.h"

#include "util/u_dynarray.h"

#include "vk_command_buffer.h"

#include <stdio.h>

struct kk_query_pool;

struct kk_root_descriptor_table {
   uint64_t addr;

   union {
      struct {
         /* Vertex input state */
         uint32_t buffer_strides[KK_MAX_VBUFS];
         uint64_t attrib_base[KK_MAX_ATTRIBS];
         uint32_t attrib_clamps[KK_MAX_ATTRIBS];

         float blend_constant[4];
         float clip_z_coeff;

         float viewport_z_range[KK_MAX_VIEWPORTS * 2];
         bool emulate_depth_clamp;
         bool emulate_viewport_z;
      } draw;
      struct {
         uint32_t base_group[3];
      } cs;
   };

   /* Client push constants */
   uint8_t push[KK_MAX_PUSH_SIZE];

   /* Descriptor set base addresses */
   uint64_t sets[KK_MAX_SETS];

   /* Dynamic buffer bindings */
   struct kk_buffer_address dynamic_buffers[KK_MAX_DYNAMIC_BUFFERS];

   /* Start index in dynamic_buffers where each set starts */
   uint8_t set_dynamic_buffer_start[KK_MAX_SETS];
};

struct kk_descriptor_state {
   bool root_dirty;
   struct kk_root_descriptor_table root;

   uint32_t set_sizes[KK_MAX_SETS];
   struct kk_descriptor_set *sets[KK_MAX_SETS];

   uint32_t push_dirty;
   struct kk_push_descriptor_set *push[KK_MAX_SETS];
};

struct kk_per_draw_data {
   uint32_t draw_id;
   uint32_t index_size;
   /* Mask of outputs flowing VS->TCS, VS->GS, or TES->GS . */
   uint64_t vertex_outputs;

   /* Address of vertex param buffer if geom/tess is used, else 0 */
   uint64_t vertex_params;

   /* Address of tessellation param buffer if tessellation used, else 0 */
   uint64_t tess_params;

   uint64_t base_vertex_addr;
   uint64_t base_instance_addr;

   /* Address of geometry param buffer if a GS is used, else 0 */
   uint64_t geometry_params;

   /* Whether the last vertex of a primitive is its provoking vertex */
   uint32_t provoking_last;

   /* GS stream selected for rasterization */
   uint32_t rasterization_stream;
};

struct kk_attachment {
   VkRenderingAttachmentFlagsKHR flags;
   VkFormat vk_format;
   struct kk_image_view *iview;

   VkResolveModeFlagBits resolve_mode;
   struct kk_image_view *resolve_iview;

   VkAttachmentLoadOp load_op;
   VkAttachmentStoreOp store_op;
   VkClearValue clear_value;
};

struct kk_rendering_state {
   VkRenderingFlagBits flags;

   VkRect2D area;
   uint32_t layer_count;
   uint32_t view_mask;
   uint32_t samples;

   uint32_t color_att_count;
   struct kk_attachment color_att[KK_MAX_RTS];
   uint8_t color_map[KK_MAX_RTS];
   struct kk_attachment depth_att;
   struct kk_attachment stencil_att;
   struct kk_attachment fsr_att;

   bool ms_bresenham_lines;
   /* Barrier tracking to understand if we need to split render passes. */
   bool write_available;
   bool ds_write_available;
   bool storage_write_available;
   bool sample_locations_enable;
   uint32_t sample_locations_count;
   VkSampleLocationEXT sample_locations[KK_MAX_SAMPLES];
   bool force_attachment_store;
   bool rendering_to_whole_framebuffer;
   /* The rendering has attachments and its first render encoder is yet to
    * start (cs_start_render): that one takes the attachments' load actions
    * and sample count. It starts at the first use when it would follow other
    * encoders of its Metal command buffer (kk_CmdBeginRendering). */
   bool first_start;
   /* A draw of the rendering recorded pre_gfx work (cs_get_pre_gfx): its
    * render encoders that would follow other encoders start a Metal command
    * buffer (cs_start_render). */
   bool pre_gfx;
};

/* Dirty tracking bits for state not tracked by vk_dynamic_graphics_state or
 * shaders_dirty.
 */
enum kk_dirty {
   KK_DIRTY_VB = BITFIELD_BIT(0),
   KK_DIRTY_OCCLUSION = BITFIELD_BIT(1),
};

struct kk_graphics_state {
   struct kk_rendering_state render;
   struct kk_descriptor_state descriptors;
   struct kk_per_draw_data per_draw_data;

   mtl_depth_stencil_state *depth_stencil_state;
   mtl_render_pass_descriptor *render_pass_descriptor;
   bool is_depth_stencil_dynamic;
   bool is_cull_front_and_back;
   bool need_to_start_render_pass;

   enum kk_dirty dirty;
   uint32_t pipeline_sample_count;

   /* Program of the bound VS draws use (kk_cmd_shader_pipeline) */
   const struct kk_pipeline_handles *pipe;

   struct {
      enum mtl_visibility_result_mode mode;

      /* If enabled, index of the current occlusion query in its chunk of the
       * occlusion heap (kk_oq_heap). There can only be one active at a time
       * (hardware constraint).
       */
      uint32_t index;
      bool used;

      /* Chunk of the current occlusion query */
      uint32_t chunk;
      /* Chunk the render pass descriptor writes visibility results to */
      uint32_t pass_chunk;
   } occlusion;

   /* Index buffer */
   struct {
      struct kk_addr_range gpu;
      uint8_t bytes_per_index;
      uint64_t null_addr;
   } index;

   /* Vertex buffers */
   struct {
      struct kk_addr_range addr_range[KK_MAX_VBUFS];
   } vb;

   /* Tessellation state */
   struct {
      /* Grid buffer for when the draw is indirect */
      struct kk_ptr indirect_ptr;
      uint64_t out_draws_addr;
      struct kk_tess_info info;
      enum mesa_prim prim;
   } tess;

   /* Rasterization draw of the GS output */
   struct {
      struct kk_addr_range index;
      uint32_t index_count;
      uint32_t instance_count;
   } gs;

   /* Transform feedback */
   struct {
      struct kk_addr_range buffers[POLY_MAX_SO_BUFFERS];
      /* Per-buffer byte offsets in lazily allocated command buffer memory */
      uint64_t offsets;
      /* Active queries for each stream: primitives written, then needed */
      uint64_t queries[POLY_MAX_VERTEX_STREAMS];
      /* Zero offset when XFB is disabled, then scratch for unused counters */
      uint64_t scratch;
      bool enabled;
   } xfb;

   /* Needed by vk_command_buffer::dynamic_graphics_state */
   struct vk_vertex_input_state _dynamic_vi;
   struct vk_sample_locations_state _dynamic_sl;
};

struct kk_compute_state {
   struct kk_descriptor_state descriptors;
};

struct kk_conditional_rendering_state {
   uint64_t address;
   bool inverted;
   bool enabled;
};

struct kk_encoder;

struct kk_uploader {
   /** List of kk_cmd_bo */
   struct list_head bos;

   /* Current addresses */
   struct kk_bo *bo;
   uint32_t offset;
};

/* A pending resolve of a range of count timestamp counter-heap entries into a
 * query pool BO. Recorded by vkCmdWriteTimestamp2 and flushed at cs_end. */
struct kk_ts_resolve {
   mtl_counter_heap *heap;
   uint32_t index;
   uint32_t count;
   uint64_t dst_addr;
};

struct kk_ts_stage_entry {
   mtl_counter_heap *heap;
   enum mtl_render_stages stage;
   uint32_t index;
};

/* Highball: why a Metal render encoder started (HB_KK_SPLITSTATS=1 prints a summary). */
enum kk_hb_split_reason {
   KK_HB_SPLIT_BEGIN = 0,    /* first encoder of a vkCmdBeginRendering */
   KK_HB_SPLIT_RESUME,       /* first encoder of a RESUMING rendering */
   KK_HB_SPLIT_BARRIER,      /* vkCmdPipelineBarrier2 inside a render pass */
   KK_HB_SPLIT_COMPUTE,      /* compute work (driver or app) broke the pass */
   KK_HB_SPLIT_STATE,        /* sample locations / attachment map changed */
   KK_HB_SPLIT_OTHER,
   KK_HB_SPLIT_COUNT,
};
void kk_hb_split_note(enum kk_hb_split_reason reason);

struct kk_cmd_buffer {
   struct vk_command_buffer vk;
   enum kk_hb_split_reason hb_restart_reason;

   struct {
      mtl_command_allocator *allocator;
      /* The Metal command buffer being encoded, the last of cmd_bufs */
      mtl_command_buffer *cmd_buf;
      /* Metal command buffers of the recording in commit order. Render
       * encoders of renderings with pre_gfx work start one (cs_start_render),
       * so that their pre_gfx pass can be committed right before them. */
      struct util_dynarray cmd_bufs;
      /* cmd_buf has an encoder */
      bool cmd_buf_used;
      /* Timestamp writes outside render encoders in the current cmd_buf */
      uint32_t ts_writes;
      /* The open render encoder is the first encoder of cmd_buf */
      bool render_first;
      mtl_render_encoder *render;
      mtl_compute_encoder *compute;
      /* Compute pass of pre_gfx dispatches recorded while the render encoder
       * is open (cs_get_pre_gfx), at the end of the Metal command buffer
       * committed right before the render encoder's: the one the render
       * encoder started after, or one of its own. Two Metal command buffers
       * are encoded at once then, each with an allocator of its own:
       * pre_gfx.cmd_buf with pre_gfx.allocator, cmd_buf with allocator. */
      struct {
         mtl_command_allocator *allocator;
         mtl_command_buffer *cmd_buf;
         mtl_compute_encoder *compute;
      } pre_gfx;
   } metal;
   /* Pending timestamp resolves (struct kk_ts_resolve), flushed at cs_end. */
   struct util_dynarray ts_resolves;
   /* Timestamps already sampled by the current render encoder. (struct
    * kk_ts_stage_entry). */
   struct util_dynarray ts_stage_map;
   /* Chunks of the process' timestamp counter heaps this command buffer
    * samples into (uint16_t, kk_query_pool.c), held until reset or free;
    * ts_chunk_used entries of the last one are taken. */
   struct util_dynarray ts_chunks;
   uint32_t ts_chunk_used;
   /* Deferred writes due to being mid render encoder */
   struct util_dynarray post_render_writes;

   void *drawable;
   mtl_argument_table *argument_table;

   struct {
      struct kk_graphics_state gfx;
      struct kk_compute_state cs;
      struct kk_conditional_rendering_state cond_render;
      struct kk_shader *shaders[MESA_SHADER_STAGES];
      /* Address of the binding 0 for when compute dispatches modify it.
       * We are trying to be nice to ourselves. */
      uint64_t root_addr;
      /* Only tracks graphics shaders since compute is always bound for now. */
      uint32_t dirty_shaders;
   } state;

   struct kk_uploader uploader;

   /* Owned large BOs */
   struct util_dynarray large_bos;

   /* The geometry heap was reset for the open or next render encoder: its
    * allocations stay until that render encoder ends (cs_end). */
   bool uses_heap;
   /* Set at vkBeginCommandBuffer. One-time-submit buffers skip command
    * enqueueing in the trampolines since they can never be replayed. */
   bool one_time_submit;
   /* Metal command buffers are single-shot: a resubmission must re-record
    * by replaying the enqueued commands (see rerecord_cmd_buffer). */
   bool submitted;
   /* Recorded by kk_queue_submit's replay of another command buffer */
   bool replay;
   /* Update-after-bind descriptor sets without reduction samplers that draws
    * and dispatches used a pipeline's program without sampler min/max
    * emulation for (struct kk_descriptor_set *, kk_cmd_shader_pipeline): if
    * one of them holds a reduction sampler at submission, the command buffer
    * is re-recorded. */
   struct util_dynarray minmax_uab_sets;

   /* What the command buffer encoded, for command queue errors (kk_queue) */
   struct kk_cb_record record;
};

VK_DEFINE_HANDLE_CASTS(kk_cmd_buffer, vk.base, VkCommandBuffer,
                       VK_OBJECT_TYPE_COMMAND_BUFFER)

extern const struct vk_command_buffer_ops kk_cmd_buffer_ops;

static inline struct kk_device *
kk_cmd_buffer_device(struct kk_cmd_buffer *cmd)
{
   return (struct kk_device *)cmd->vk.base.device;
}

static inline struct kk_cmd_pool *
kk_cmd_buffer_pool(struct kk_cmd_buffer *cmd)
{
   return (struct kk_cmd_pool *)cmd->vk.pool;
}

static inline struct kk_descriptor_state *
kk_get_descriptors_state(struct kk_cmd_buffer *cmd,
                         VkPipelineBindPoint bind_point)
{
   switch (bind_point) {
   case VK_PIPELINE_BIND_POINT_GRAPHICS:
      return &cmd->state.gfx.descriptors;
   case VK_PIPELINE_BIND_POINT_COMPUTE:
      return &cmd->state.cs.descriptors;
   default:
      UNREACHABLE("Unhandled bind point");
   }
};

void cs_start_render(struct kk_cmd_buffer *cmd);
mtl_render_encoder *cs_get_render(struct kk_cmd_buffer *cmd);
mtl_compute_encoder *cs_get_compute(struct kk_cmd_buffer *cmd);
mtl_compute_encoder *cs_get_pre_gfx(struct kk_cmd_buffer *cmd);
void kk_cmd_count_ts_writes(struct kk_cmd_buffer *cmd, uint32_t writes);
void cs_end(struct kk_cmd_buffer *cmd);
void kk_cmd_bind_root_to_argument_table(struct kk_cmd_buffer *cmd,
                                        uint64_t addr);

static void
kk_cmd_buffer_dirty_all_gfx(struct kk_cmd_buffer *cmd)
{
   /* Ensure we flush all graphics state */
   vk_dynamic_graphics_state_dirty_all(&cmd->vk.dynamic_graphics_state);
   cmd->state.dirty_shaders = ~0u;
   cmd->state.gfx.dirty = ~0u;
   cmd->state.gfx.descriptors.root_dirty = true;
}

void kk_cmd_release_dynamic_ds_state(struct kk_cmd_buffer *cmd);

mtl_depth_stencil_state *
kk_compile_depth_stencil_state(struct kk_device *device,
                               const struct vk_depth_stencil_state *ds,
                               bool has_depth, bool has_stencil);

void kk_meta_resolve_rendering(struct kk_cmd_buffer *cmd,
                               const VkRenderingInfo *pRenderingInfo);

struct kk_ptr kk_pool_alloc(struct kk_cmd_buffer *cmd, uint32_t size,
                            uint32_t alignment);

struct kk_ptr kk_pool_upload(struct kk_cmd_buffer *cmd, const void *data,
                             uint32_t size, uint32_t alignment);

uint64_t kk_upload_descriptor_root(struct kk_cmd_buffer *cmd,
                                   VkPipelineBindPoint bind_point);

void kk_cmd_buffer_flush_push_descriptors(struct kk_cmd_buffer *cmd,
                                          struct kk_descriptor_state *desc);

void kk_apply_attachment_store_ops(struct kk_cmd_buffer *cmd, bool force_store);

/* Attachments of the render pass descriptor load (restarted render encoders
 * of the rendering) */
void kk_render_pass_descriptor_load_attachments(struct kk_cmd_buffer *cmd);

bool kk_attachment_do_renderpass_resolve(const struct kk_attachment *attachment,
                                         bool rendering_to_whole_framebuffer,
                                         VkImageAspectFlags aspect);

enum kk_grid_mode {
   KK_GRID_DIRECT = 0u,
   KK_GRID_INDIRECT,
   /* Thread counts, then threadgroup size, like poly's grids */
   KK_GRID_INDIRECT_LOCAL,
};
struct kk_grid {
   enum kk_grid_mode mode;
   union {
      struct mtl_size size;
      uint64_t addr;
   };
};

static struct kk_grid
kk_grid_3d(uint32_t x, uint32_t y, uint32_t z)
{
   return (struct kk_grid){
      .mode = KK_GRID_DIRECT,
      .size = {x, y, z},
   };
}

static struct kk_grid
kk_grid_2d(uint32_t x, uint32_t y)
{
   return kk_grid_3d(x, y, 1u);
}

static struct kk_grid
kk_grid_1d(uint32_t x)
{
   return kk_grid_3d(x, 1u, 1u);
}

static struct kk_grid
kk_grid_indirect(uint64_t addr)
{
   return (struct kk_grid){
      .mode = KK_GRID_INDIRECT,
      .addr = addr,
   };
}

static struct kk_grid
kk_grid_indirect_local(uint64_t addr)
{
   return (struct kk_grid){
      .mode = KK_GRID_INDIRECT_LOCAL,
      .addr = addr,
   };
}

static bool
kk_grid_is_indirect(struct kk_grid grid)
{
   return grid.mode != KK_GRID_DIRECT;
}

/* pre_gfx: while a render encoder is open, the dispatch goes into its pre_gfx
 * pass, which runs before it (cs_get_pre_gfx) instead of ending it. Only for
 * work that reads nothing the render pass's draws may write (index, indirect
 * and predicate buffers: transfers are not allowed in a render pass, and a
 * self-dependency cannot order shader writes before index or indirect reads)
 * and whose results only the render pass's draws read. */
void kk_dispatch_precomp(struct kk_cmd_buffer *cmd, struct kk_grid grid,
                         bool pre_gfx, enum libkk_program idx, void *data,
                         size_t data_size);

#define MESA_DISPATCH_PRECOMP kk_dispatch_precomp

void kk_cmd_write(struct kk_cmd_buffer *cmd, struct libkk_imm_write write);

void kk_cmd_buffer_set_label(struct kk_cmd_buffer *cb, const char *label);

#endif
