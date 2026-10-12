/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef KK_QUERY_POOL_H
#define KK_QUERY_POOL_H 1

#include "kk_private.h"

#include "vulkan/runtime/vk_query_pool.h"

/* Entries of one Metal counter heap: Metal refuses larger timestamp heaps
 * ("Requested Heap size is too large (requested 8192 max is 4096)").
 */
#define KK_TS_HEAP_MAX_ENTRIES 4096u

struct kk_cmd_buffer;

struct kk_oq_state {
   unsigned queries;
};

struct kk_query_pool {
   struct vk_query_pool vk;

   struct kk_bo *bo;

   uint32_t index_start;
   uint32_t query_start;
   uint32_t query_stride;

   struct kk_oq_state oq;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(kk_query_pool, vk.base, VkQueryPool,
                               VK_OBJECT_TYPE_QUERY_POOL)

uint32_t *kk_pool_index_ptr(const struct kk_query_pool *pool);

/* Gives the timestamp counter heap entries a command buffer recorded samples
 * into back to the process (reset/free). */
void kk_cmd_release_timestamps(struct kk_cmd_buffer *cmd);

#endif /* KK_QUERY_POOL_H */
