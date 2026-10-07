/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef KK_QUEUE_H
#define KK_QUEUE_H 1

#include "kk_private.h"

#include "kk_cmd_pool.h"

#include "kosmickrisp/bridge/mtl_types.h"

#include "vk_queue.h"

#include "c11/threads.h"
#include "util/simple_mtx.h"

#define KK_QUEUE_PENDING_WAITS 8u

struct kk_queue {
   struct vk_queue vk;
   struct mtl_command_queue *mtl_handle;

   /* Resources for command buffer re-recording. These are required for
    * SIMULTANEOUS_USE command buffers where the same command buffer can be
    * submitted multiple times at the same time. Metal's command buffers are a
    * one time use, so the queue will own the re-recorded ones. */
   struct kk_cmd_pool *cmd_pool;
   struct util_dynarray free_cmd_buffers;
   uint32_t commits_in_flight;
   mtx_t mutex;
   cnd_t cond;

   /* vkQueueBindSparse (kk_sparse.c): a one page placement sparse buffer and
    * the heap page it maps after unmaps of buffer pages, and the event that
    * separates texture mapping updates from earlier buffer mapping updates.
    * lock orders the queue's mapping updates. */
   /* The last KK_QUEUE_PENDING_WAITS semaphore waits that were not signalled
    * yet when submitted (wait before signal), retained, under `mutex`: a
    * command queue error names the ones still unsignalled. Metal ends a queue
    * wait that is never signalled after a few seconds, on some GPUs with
    * MTL4CommandQueueErrorTimeout. */
   struct {
      mtl_shared_event *event;
      uint64_t value;
      int64_t submitted_ns;
   } pending_waits[KK_QUEUE_PENDING_WAITS];
   uint32_t pending_wait_next;

   struct {
      mtl_buffer *flush_buffer;
      struct kk_bo *flush_bo;
      mtl_event *separator;
      uint64_t separator_value;
      simple_mtx_t lock;
      bool texture_mapped;
      bool buffer_after_texture;
   } sparse;
};

static inline struct kk_device *
kk_queue_device(struct kk_queue *queue)
{
   return (struct kk_device *)queue->vk.base.device;
}

VkResult kk_queue_init(struct kk_device *dev, struct kk_queue *queue,
                       const VkDeviceQueueCreateInfo *pCreateInfo,
                       uint32_t index_in_family);

void kk_queue_finish(struct kk_device *dev, struct kk_queue *queue);

#endif
