/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef KK_DEVICE_H
#define KK_DEVICE_H 1

#include "kk_private.h"

#include "kk_query_table.h"
#include "kk_queue.h"

#include "kosmickrisp/bridge/mtl_types.h"

#include "kosmickrisp/clc/kk_precompiled_shader.h"
#include "libkk_shaders.h"

#include "util/u_dynarray.h"

#include "vk_device.h"
#include "vk_meta.h"
#include "vk_queue.h"

#include "c11/threads.h"

struct kk_bo;
struct kk_physical_device;
struct vk_pipeline_cache;

struct kk_residency_set {
   simple_mtx_t mutex;
   mtl_residency_set *handle;
};

struct mtl_sampler_packed {
   enum mtl_sampler_address_mode mode_u;
   enum mtl_sampler_address_mode mode_v;
   enum mtl_sampler_address_mode mode_w;
   enum mtl_sampler_border_color border_color;

   enum mtl_sampler_min_mag_filter min_filter;
   enum mtl_sampler_min_mag_filter mag_filter;
   enum mtl_sampler_mip_filter mip_filter;
   enum mtl_sampler_reduction_mode reduction_mode;

   enum mtl_compare_function compare_func;
   float min_lod;
   float max_lod;
   uint32_t max_anisotropy;
   bool normalized_coordinates;
};

struct kk_rc_sampler {
   struct mtl_sampler_packed key;

   mtl_sampler *handle;

   /* Reference count for this hardware sampler, protected by the heap mutex */
   uint16_t refcount;

   /* Index of this hardware sampler in the hardware sampler heap */
   uint16_t index;
};

struct kk_sampler_heap {
   simple_mtx_t lock;

   struct kk_query_table table;

   /* Map of mtl_sampler_packed to kk_rc_sampler */
   struct hash_table *ht;
};

struct kk_precompiled_cache {
   struct kk_precompiled_shader shaders[LIBKK_NUM_PROGRAMS];
};

/* A bounded table of compiled Metal objects (MSL libraries, pipeline states)
 * shared by a device's pipelines, kept in kk_shader.c. Lookups and publishes
 * take its lock, which is never held across a compile (that would serialize
 * the compile pool); an entry without an object is a compile in flight that
 * other callers wait for on cnd. */
struct kk_metal_cache {
   struct hash_table *table;
   /* Entries by use, least recently used first; their count and the bound */
   struct list_head lru;
   uint32_t count;
   uint32_t max;
   void (*free_key)(void *key);
   mtx_t lock;
   cnd_t cnd;
};

struct kk_device {
   struct vk_device vk;

   mtl_device *mtl_handle;
   mtl_compiler *mtl_compiler_handle;

   /* Compiled Metal objects shared between a device's pipelines (kk_shader.c):
    * MSL libraries by the blake3 digest of their source, and Metal pipeline
    * states by what they are built from. A title asks for the same program once
    * per pipeline (Dragon's Dogma 2: ~100 times per program, 26,452 program
    * compiles for 4,594 distinct programs in one run), and every request
    * otherwise costs a Metal compile call plus the guest round trip to reach it
    * (FEX -> vkd3d -> Venus -> virglrenderer -> here); the compiler spends its
    * time in the pipeline states. Each table is bounded (kk_metal_cache::max):
    * past it the least recently used entries are dropped, so a title that
    * creates and destroys pipelines for hours does not grow the process without
    * bound. Set up before any pipeline is created and torn down after the
    * compile pool (kk_metal_caches_init). */
   struct kk_metal_cache msl_libraries;
   struct kk_metal_cache metal_render_pipelines;
   struct kk_metal_cache metal_compute_pipelines;

   /* Dispatch table exposed to the user. Required since we need to record all
    * commands due to Metal limitations */
   struct vk_device_dispatch_table exposed_dispatch_table;

   struct kk_sampler_heap samplers;
   struct kk_oq_heap occlusion_queries;

   /* Track all heaps the user allocated so we can set them all as resident when
    * recording as required by Metal. */
   struct kk_residency_set residency_set;

   struct kk_precompiled_cache precompiled_cache;

   bool has_queue;
   struct kk_queue queue;

   struct vk_meta_device meta;

   /* Geomtry heap */
   struct kk_bo *heap;
   util_once_flag heap_init_once;

   /* Sampler min/max reduction emulation before Apple10 (kk_shader.c): a
    * descriptor set got a reduction sampler, so pipelines read from a
    * pipeline cache (no set layouts to check) compile their min/max
    * emulating program first */
   bool reduction_sampler_sets;
   /* Compiling a program's Metal pipelines is the expensive part of creating a
    * pipeline, and virglrenderer drains a guest's Venus context on one ring
    * thread: a title's first run (Dragon's Dogma 2 asks for ~130,000 pipelines
    * of ~1,100 programs) would otherwise compile them strictly one after
    * another there — 62 minutes measured. Programs are queued here and compiled
    * by a pool of threads instead (kk_shader.c), and the first use of a program
    * waits for its compile. */
   mtx_t compile_mtx;
   cnd_t compile_cnd;
   /* Programs whose pipelines are not compiled yet, linked by
    * kk_shader::compile_next, oldest first */
   struct kk_shader *compile_head;
   struct kk_shader *compile_tail;
   /* Queued or being compiled; the pool is only torn down when it reaches 0 */
   uint32_t compile_pending;
   bool compile_shutdown;
   uint32_t compile_thread_count;
   thrd_t *compile_threads;

   /* os_time_get_nano() at creation (command queue error diagnostics) */
   int64_t created_ns;
   /* In the process' list of devices alive (kk_device_foreach) */
   struct list_head lifecycle_link;
};

VK_DEFINE_HANDLE_CASTS(kk_device, vk.base, VkDevice, VK_OBJECT_TYPE_DEVICE)

static inline struct kk_physical_device *
kk_device_physical(const struct kk_device *dev)
{
   return (struct kk_physical_device *)dev->vk.physical;
}

VkResult kk_device_init_meta(struct kk_device *dev);
void kk_device_finish_meta(struct kk_device *dev);
VkResult kk_device_init_lib(struct kk_device *dev);
void kk_device_finish_lib(struct kk_device *dev);
void kk_device_add_heap_to_residency_set(struct kk_device *dev, mtl_heap *heap);
void kk_device_remove_heap_from_residency_set(struct kk_device *dev,
                                              mtl_heap *heap);
void kk_device_add_buffer_to_residency_set(struct kk_device *dev,
                                           mtl_buffer *buffer);
void kk_device_remove_buffer_from_residency_set(struct kk_device *dev,
                                                mtl_buffer *buffer);
/* Placement sparse textures and buffers */
void kk_device_add_allocation_to_residency_set(struct kk_device *dev,
                                               mtl_allocation *allocation);
void kk_device_remove_allocation_from_residency_set(
   struct kk_device *dev, mtl_allocation *allocation);
/* Textures made from a heap or a buffer (and the base of views made from such
 * a texture): Metal 4 takes an attachment as resident only when its base
 * texture object is in a residency set, its heap or buffer does not count.
 * Every texture the driver makes this way is added once by its owner and
 * removed before its release. */
void kk_device_add_texture_to_residency_set(struct kk_device *dev,
                                            mtl_texture *texture);
void kk_device_remove_texture_from_residency_set(struct kk_device *dev,
                                                 mtl_texture *texture);
void kk_device_make_resources_resident(struct kk_device *dev);

/* The guest application of a device ("gamescope", "steamwebhelper (mesa
 * zink)"): Venus passes VkApplicationInfo on. */
void kk_device_app_name(const struct kk_device *dev, char *name, size_t size);
/* Logs the devices alive in the process (every guest device under
 * virglrenderer) and the last ones created and destroyed. */
void kk_device_log_lifecycle(void);
/* Calls cb for every device alive in the process (they stay alive until it
 * returns): cb must not wait for anything a device's creation or destruction
 * could hold. */
void kk_device_foreach(void (*cb)(struct kk_device *dev, void *data),
                       void *data);

/* Required to create a sampler */
mtl_sampler *kk_sampler_create(struct kk_device *dev,
                               const struct mtl_sampler_packed *packed);
VkResult kk_sampler_heap_add(struct kk_device *dev,
                             struct mtl_sampler_packed desc,
                             struct kk_rc_sampler **out);
void kk_sampler_heap_remove(struct kk_device *dev, struct kk_rc_sampler *rc);

#endif // KK_DEVICE_H
