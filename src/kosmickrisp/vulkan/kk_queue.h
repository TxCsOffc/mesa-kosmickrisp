/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef KK_QUEUE_H
#define KK_QUEUE_H 1

#include "kk_private.h"

#include "kk_cb_record.h"
#include "kk_cmd_pool.h"

#include "kosmickrisp/bridge/mtl_types.h"

#include "vk_queue.h"

#include "c11/threads.h"
#include "util/simple_mtx.h"

#define KK_QUEUE_PENDING_WAITS 8u
#define KK_QUEUE_RECORDS       256u

/* A commit's command buffer record and, once its feedback handler ran, its
 * outcome */
struct kk_queue_record {
   uint64_t seq; /* 0: unused */
   int64_t commit_ns;
   bool done;
   /* The IOGPU reason of its command queue error ("Hang", "InnocentVictim",
    * ...), "ok" when it completed */
   char outcome[32];
   double gpu_start, gpu_end, commit_time, feedback_time;
   struct kk_cb_record cb;
};

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

   /* Under `mutex`: commits whose feedback handler has not run yet
    * (kk_queue_finish waits for them, the handlers use the queue), the
    * number of the last command buffer committed and the GPU end time
    * (seconds) of the last one completed, for command queue error
    * diagnostics. */
   uint32_t feedbacks_pending;
   uint64_t commit_seq;
   double last_gpu_end;

   /* Under `mutex`: the records of the last KK_QUEUE_RECORDS commits, at
    * seq % KK_QUEUE_RECORDS. A command queue error prints the failed command
    * buffer's and those of every command buffer of the process in flight
    * (kk_queue_log_failure). */
   struct kk_queue_record records[KK_QUEUE_RECORDS];

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

   /* vkQueueBindSparse (kk_sparse.c): a one page placement sparse buffer and
    * the heap page it maps after unmaps of buffer pages, and the event that
    * separates texture mapping updates from earlier buffer mapping updates.
    * lock orders the queue's mapping updates. */
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

void kk_queue_log_sync_wait(struct kk_queue *queue);

#endif
