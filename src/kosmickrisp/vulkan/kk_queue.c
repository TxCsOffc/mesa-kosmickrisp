/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "kk_queue.h"
#include "kk_buffer.h"
#include "kk_cmd_buffer.h"
#include "kk_debug.h"
#include "kk_descriptor_set.h"
#include "kk_device.h"
#include "kk_entrypoints.h"
#include "kk_gpu_time.h"
#include "kk_physical_device.h"
#include "kk_sparse.h"
#include "kk_sync.h"

#include "kosmickrisp/bridge/mtl_bridge.h"
#include "kosmickrisp/bridge/vk_to_mtl_map.h"

#include "vk_cmd_queue.h"

#include <ctype.h>
#include <string.h>

struct kk_commit_data {
   struct kk_queue *queue;
   struct kk_cmd_buffer *cmd;
};

/* Feedback handlers use the queue and the device: wait for all of them. */
static void
kk_queue_wait_for_feedbacks(struct kk_queue *queue)
{
   mtx_lock(&queue->mutex);
   while (queue->feedbacks_pending > 0u)
      cnd_wait(&queue->cond, &queue->mutex);
   mtx_unlock(&queue->mutex);
}

static void
kk_queue_release_cmd_buffer_locked(struct kk_queue *queue,
                                   struct kk_cmd_buffer *cmd)
{
   kk_cmd_buffer_ops.reset(&cmd->vk, 0u);
   util_dynarray_append(&queue->free_cmd_buffers, &cmd->vk);
}

/* The IOGPU reason of a command queue error, from its userInfo
 * ("... (00000003:kIOGPUCommandBufferCallbackErrorHang) ..." gives "Hang"),
 * else the Metal error; "ok" without an error. */
static void
kk_queue_feedback_outcome(const struct mtl_feedback_data *data, char *out,
                          size_t size)
{
   static const char reason[] = "kIOGPUCommandBufferCallbackError";

   if (data->error == MTL_COMMAND_QUEUE_ERROR_NONE) {
      snprintf(out, size, "ok");
      return;
   }
   const char *at = data->error_info ? strstr(data->error_info, reason) : NULL;
   size_t n = 0u;
   if (at) {
      at += sizeof(reason) - 1u;
      for (; n + 1u < size && isalnum((unsigned char)at[n]); n++)
         out[n] = at[n];
   }
   if (n)
      out[n] = '\0';
   else
      snprintf(out, size, "%s", mtl_command_queue_error_to_string(data->error));
}

/* A commit's GPU time relative to the failed command buffer's start */
static void
kk_queue_format_gpu_time(const struct kk_queue_record *r, double failed_start,
                         char *out, size_t size)
{
   if (r->gpu_start <= 0.0)
      snprintf(out, size, "never started");
   else if (failed_start <= 0.0)
      snprintf(out, size, "ran %.1f ms", (r->gpu_end - r->gpu_start) * 1000.0);
   else if (r->gpu_end < r->gpu_start || r->gpu_end - r->gpu_start > 1000.0)
      snprintf(out, size, "started %+.1f ms, no end",
               (r->gpu_start - failed_start) * 1000.0);
   else
      snprintf(out, size, "%+.1f to %+.1f ms",
               (r->gpu_start - failed_start) * 1000.0,
               (r->gpu_end - failed_start) * 1000.0);
}

#define KK_QUEUE_IN_FLIGHT_LINES 24u
#define KK_QUEUE_IN_FLIGHT_WAIT_NS (300ll * 1000 * 1000)

struct kk_queue_in_flight {
   const struct kk_device *failed_dev;
   uint64_t failed_seq;
   /* GPU start of the failed command buffer, its feedback (host seconds) */
   double failed_start, failed_feedback;
   int64_t now;
   unsigned pending, lines, skipped, hangs;
   unsigned pass;
   struct kk_cb_record hang[2];
};

/* Commits of the process other than the failed one that were on the GPU
 * with it: still awaiting feedback, failed, or ran while it did. */
static bool
kk_queue_record_in_flight(const struct kk_queue_record *r,
                          const struct kk_device *dev,
                          const struct kk_queue_in_flight *s)
{
   if (r->seq == 0u || r->commit_ns > s->now ||
       (dev == s->failed_dev && r->seq == s->failed_seq))
      return false;
   if (!r->done || strcmp(r->outcome, "ok"))
      return true;
   return r->gpu_end >= s->failed_start && r->gpu_start <= s->failed_feedback;
}

static void
kk_queue_count_in_flight(struct kk_device *dev, void *data)
{
   struct kk_queue_in_flight *s = data;
   if (!dev->has_queue)
      return;
   mtx_lock(&dev->queue.mutex);
   for (uint32_t i = 0u; i < KK_QUEUE_RECORDS; i++) {
      const struct kk_queue_record *r = &dev->queue.records[i];
      s->pending += !r->done && kk_queue_record_in_flight(r, dev, s);
   }
   mtx_unlock(&dev->queue.mutex);
}

static void
kk_queue_log_in_flight(struct kk_device *dev, void *data)
{
   struct kk_queue_in_flight *s = data;
   if (!dev->has_queue)
      return;
   char app[64];
   kk_device_app_name(dev, app, sizeof(app));

   struct kk_queue *queue = &dev->queue;
   mtx_lock(&queue->mutex);
   for (uint32_t k = 1u; k <= KK_QUEUE_RECORDS; k++) {
      const struct kk_queue_record *r =
         &queue->records[(queue->commit_seq + k) % KK_QUEUE_RECORDS];
      if (!kk_queue_record_in_flight(r, dev, s))
         continue;
      /* Faults must precede innocent victims and successful compositor work:
       * otherwise the 24-line cap can hide the culprit and its shader excerpts. */
      const bool culprit = r->done && strcmp(r->outcome, "ok") &&
                           strcmp(r->outcome, "InnocentVictim");
      const unsigned pass = culprit ? 0u : !r->done || strcmp(r->outcome, "ok") ? 1u : 2u;
      if (pass != s->pass)
         continue;
      if (s->lines == KK_QUEUE_IN_FLIGHT_LINES) {
         s->skipped++;
         continue;
      }
      s->lines++;
      char text[448], gpu[48];
      kk_cb_record_format(&r->cb, text, sizeof(text));
      if (r->done) {
         kk_queue_format_gpu_time(r, s->failed_start, gpu, sizeof(gpu));
         mesa_loge("kk: with it: command buffer %" PRIu64 " of %s, %s, GPU "
                   "%s: %s",
                   r->seq, app, r->outcome, gpu, text);
      } else {
         mesa_loge("kk: with it: command buffer %" PRIu64 " of %s, no "
                   "feedback, committed %.1f ms ago: %s",
                   r->seq, app, (s->now - r->commit_ns) / 1e6, text);
      }
      if (culprit)
         kk_cb_record_log_dispatches(&r->cb);
      if (!strcmp(r->outcome, "Hang") && s->hangs < ARRAY_SIZE(s->hang))
         s->hang[s->hangs++] = r->cb;
   }
   mtx_unlock(&queue->mutex);
}

/* STEAMAC-G: a command queue error has to tell GPU work that ran too long
 * from a command buffer that never started (a semaphore wait, the GPU busy
 * elsewhere), name what the command buffer contained and what came right
 * before: the queue's last completed work and the last guest device created
 * or destroyed in the process. Never signalled semaphore waits submitted
 * before their signal are listed. The first error in the process also names
 * every command buffer of the process that was on the GPU with it and their
 * outcome: GPU recovery fails all of them, the one that hung with "Hang", the
 * others as innocent victims. */
static void
kk_queue_log_failure(struct kk_queue *queue,
                     const struct mtl_feedback_data *data)
{
   struct kk_device *dev = kk_queue_device(queue);
   const int64_t now = os_time_get_nano();
   char app[64];
   kk_device_app_name(dev, app, sizeof(app));

   mtx_lock(&queue->mutex);
   const uint64_t committed = queue->commit_seq;
   const uint32_t awaiting = queue->feedbacks_pending;
   const double last_gpu_end = queue->last_gpu_end;
   const struct kk_queue_record *record =
      &queue->records[data->tag % KK_QUEUE_RECORDS];
   const bool recorded = record->seq == data->tag;
   const struct kk_cb_record cb = record->cb;
   char outcome[sizeof(record->outcome)];
   memcpy(outcome, record->outcome, sizeof(outcome));
   mtx_unlock(&queue->mutex);

   mesa_loge("kk: command buffer %" PRIu64 " of %s failed %.1f ms after its "
             "commit (device created %.1f s ago, %" PRIu64 " command buffers "
             "committed, %u awaiting feedback)",
             data->tag, app,
             (data->feedback_time - data->commit_time) * 1000.0,
             (now - dev->created_ns) / 1e9, committed, awaiting);
   char gpu_time[160];
   kk_gpu_time_format(data->gpu_start, data->gpu_end, data->commit_time,
                      data->feedback_time, gpu_time, sizeof(gpu_time));
   mesa_loge("kk: it %s (GPU start %.6f s, end %.6f s, commit %.6f s, feedback %.6f s)",
             gpu_time, data->gpu_start, data->gpu_end, data->commit_time,
             data->feedback_time);
   if (last_gpu_end > 0.0) {
      mesa_loge("kk: the queue's last completed command buffer ended %.1f ms "
                "after this commit",
                (last_gpu_end - data->commit_time) * 1000.0);
   } else {
      mesa_loge("kk: no command buffer of the queue completed before");
   }
   if (recorded) {
      char text[448];
      kk_cb_record_format(&cb, text, sizeof(text));
      mesa_loge("kk: it failed with %s and had %s", outcome, text);
      kk_cb_record_log_dispatches(&cb);
   }
   if (data->error_info && !(recorded && strcmp(outcome, "ok") &&
                             strstr(data->error_info, outcome)))
      mesa_loge("kk: Metal error info: %s", data->error_info);

   mtx_lock(&queue->mutex);
   for (uint32_t i = 0; i < KK_QUEUE_PENDING_WAITS; i++) {
      if (queue->pending_waits[i].event == NULL)
         continue;
      const uint64_t signalled =
         mtl_shared_event_get_signaled_value(queue->pending_waits[i].event);
      if (signalled < queue->pending_waits[i].value) {
         mesa_loge("kk: semaphore wait for %" PRIu64 " submitted %.1f ms ago "
                   "is still unsignalled (at %" PRIu64 ")",
                   queue->pending_waits[i].value,
                   (now - queue->pending_waits[i].submitted_ns) / 1e6,
                   signalled);
      }
   }
   mtx_unlock(&queue->mutex);

   kk_device_log_lifecycle();

   /* Once per process: the first error names the cause, later ones are
    * mostly its victims. The other command buffers' feedback comes on other
    * threads: wait a little for their outcome. */
   static bool first_logged;
   if (p_atomic_xchg(&first_logged, true))
      return;
   struct kk_queue_in_flight s = {
      .failed_dev = dev,
      .failed_seq = data->tag,
      .failed_start = data->gpu_start,
      .failed_feedback = data->feedback_time,
      .now = now,
   };
   const int64_t deadline = now + KK_QUEUE_IN_FLIGHT_WAIT_NS;
   do {
      s.pending = 0u;
      kk_device_foreach(kk_queue_count_in_flight, &s);
      if (s.pending == 0u)
         break;
      os_time_sleep(5000);
   } while (os_time_get_nano() < deadline);

   for (s.pass = 0u; s.pass < 3u; s.pass++)
      kk_device_foreach(kk_queue_log_in_flight, &s);
   if (s.skipped)
      mesa_loge("kk: with it: %u more command buffers", s.skipped);
   else if (s.lines == 0u)
      mesa_loge("kk: no other command buffer of the process was on the GPU "
                "with it");

   /* The MSL of programs that could spin or wait, of the command buffers
    * that hung or else of the failed one */
   unsigned excerpts = 0u;
   for (unsigned i = 0u; i < s.hangs; i++)
      excerpts += kk_cb_record_log_excerpts(&s.hang[i], 3u - excerpts);
   if (recorded && excerpts < 3u)
      kk_cb_record_log_excerpts(&cb, 3u - excerpts);
}

/* MESA_KK_DEBUG=commits */
static void
kk_queue_log_slow_commit(struct kk_queue *queue,
                         const struct mtl_feedback_data *data)
{
   const double delay_ms = (data->gpu_start - data->commit_time) * 1000.0;
   const double gpu_ms = (data->gpu_end - data->gpu_start) * 1000.0;
   const double feedback_ms = (data->feedback_time - data->commit_time) * 1000.0;
   if (delay_ms < 20.0 && gpu_ms < 50.0 && feedback_ms < 100.0)
      return;

   char app[64];
   kk_device_app_name(kk_queue_device(queue), app, sizeof(app));
   mesa_logi("kk: command buffer %" PRIu64 " of %s started %.1f ms after its "
             "commit, ran %.1f ms on the GPU, feedback after %.1f ms",
             data->tag, app, delay_ms, gpu_ms, feedback_ms);
}

/* The commit's outcome, for kk_queue_log_failure and MESA_KK_DEBUG=cbs */
static void
kk_queue_note_feedback(struct kk_queue *queue,
                       const struct mtl_feedback_data *data)
{
   struct kk_queue_record *record =
      &queue->records[data->tag % KK_QUEUE_RECORDS];
   char outcome[sizeof(record->outcome)];
   kk_queue_feedback_outcome(data, outcome, sizeof(outcome));

   mtx_lock(&queue->mutex);
   const bool recorded = record->seq == data->tag;
   if (recorded) {
      memcpy(record->outcome, outcome, sizeof(outcome));
      record->gpu_start = data->gpu_start;
      record->gpu_end = data->gpu_end;
      record->done = true;
   }
   const struct kk_cb_record cb = record->cb;
   mtx_unlock(&queue->mutex);

   if (KK_DEBUG(CBS) && recorded) {
      char app[64], text[448];
      kk_device_app_name(kk_queue_device(queue), app, sizeof(app));
      kk_cb_record_format(&cb, text, sizeof(text));
      mesa_logi("kk: command buffer %" PRIu64 " of %s: %s, started %.2f ms "
                "after its commit, %.2f ms on the GPU: %s",
                data->tag, app, outcome,
                (data->gpu_start - data->commit_time) * 1000.0,
                (data->gpu_end - data->gpu_start) * 1000.0, text);
   }
}

static void
kk_queue_check_feedback(struct kk_queue *queue, struct mtl_feedback_data *data)
{
   struct kk_device *dev = kk_queue_device(queue);

   kk_queue_note_feedback(queue, data);
   if (data->error != MTL_COMMAND_QUEUE_ERROR_NONE) {
      kk_queue_log_failure(queue, data);
      vk_device_set_lost(
         &dev->vk, "Command queue error %d (%s), with message \"%s\"",
         (int)data->error, mtl_command_queue_error_to_string(data->error),
         data->error_message ? data->error_message : "");
   } else if (KK_DEBUG(COMMITS)) {
      kk_queue_log_slow_commit(queue, data);
   }
}

/* The handler's last use of the queue: kk_queue_finish may free it after. */
static void
kk_queue_feedback_done(struct kk_queue *queue,
                       const struct mtl_feedback_data *data)
{
   mtx_lock(&queue->mutex);
   if (data->error == MTL_COMMAND_QUEUE_ERROR_NONE &&
       data->gpu_end > queue->last_gpu_end)
      queue->last_gpu_end = data->gpu_end;
   assert(queue->feedbacks_pending > 0u);
   if (--queue->feedbacks_pending == 0u)
      cnd_broadcast(&queue->cond);
   mtx_unlock(&queue->mutex);
}

static void
commit_callback(struct mtl_feedback_data *data)
{
   struct kk_queue *queue = &((struct kk_device *)data->user_data)->queue;

   kk_queue_check_feedback(queue, data);
   kk_queue_feedback_done(queue, data);
}

static void
rerecord_commit_callback(struct mtl_feedback_data *data)
{
   struct kk_commit_data *commit = (struct kk_commit_data *)data->user_data;
   struct kk_queue *queue = commit->queue;
   struct kk_device *dev = kk_queue_device(queue);

   kk_queue_check_feedback(queue, data);

   /* Completion callbacks are called from multiple threads, so we need to
    * ensure the access to queue resources is safe. */
   mtx_lock(&queue->mutex);

   kk_queue_release_cmd_buffer_locked(queue, commit->cmd);

   assert(queue->commits_in_flight > 0u);
   if (--queue->commits_in_flight == 0u) {
      kk_ResetCommandPool(kk_device_to_handle(dev),
                          kk_cmd_pool_to_handle(queue->cmd_pool), 0u);
   }

   mtx_unlock(&queue->mutex);

   vk_free(&dev->vk.alloc, commit);

   kk_queue_feedback_done(queue, data);
}

static bool
kk_cmd_buffer_has_work(struct kk_cmd_buffer *cmd)
{
   return cmd->metal.cmd_buf;
}

/* Draws or dispatches used a pipeline's program without sampler min/max
 * emulation for an update-after-bind set that got a reduction sampler after
 * recording (kk_cmd_shader_pipeline). */
static bool
kk_cmd_buffer_needs_minmax_replay(struct kk_cmd_buffer *cmd)
{
   util_dynarray_foreach(&cmd->minmax_uab_sets, struct kk_descriptor_set *,
                         set) {
      if (p_atomic_read(&(*set)->reduction_samplers)) {
         if (KK_DEBUG(MINMAX))
            mesa_logi("kk: command buffer re-recorded: an update-after-bind "
                      "set got a sampler with a min/max reduction");
         return true;
      }
   }
   return false;
}

static void
kk_queue_commit(struct kk_queue *queue, struct kk_cmd_buffer *cmd,
                mtl_feedback_handler_callback callback, void *user_data)
{
   assert(kk_cmd_buffer_has_work(cmd));

   mtx_lock(&queue->mutex);
   const uint64_t seq = ++queue->commit_seq;
   queue->feedbacks_pending++;
   struct kk_queue_record *record = &queue->records[seq % KK_QUEUE_RECORDS];
   record->seq = seq;
   record->commit_ns = os_time_get_nano();
   record->done = false;
   record->outcome[0] = '\0';
   record->gpu_start = record->gpu_end = 0.0;
   record->cb = cmd->record;
   mtx_unlock(&queue->mutex);

   mtl_commit_options *options = mtl_new_commit_options();
   mtl_commit_options_add_feedback_handler(options, callback, user_data, seq);

   /* The recording's Metal command buffers in order, a render encoder's
    * pre_gfx pass in one before the render encoder's (cs_get_pre_gfx) */
   mtl_command_queue_commit(
      queue->mtl_handle, util_dynarray_begin(&cmd->metal.cmd_bufs),
      util_dynarray_num_elements(&cmd->metal.cmd_bufs, mtl_command_buffer *),
      options);
   mtl_release(options);
}

static VkResult
rerecord_and_commit_cmd_buffer(struct kk_queue *queue,
                               struct kk_cmd_buffer *cmd)
{
   struct kk_device *dev = kk_queue_device(queue);
   struct vk_command_buffer *vk_cmd = NULL;
   struct kk_cmd_buffer *rerecord;
   VkResult result = VK_SUCCESS;

   /* Completion callbacks are called from multiple threads, so we need to
    * ensure the access to queue resources is safe. */
   mtx_lock(&queue->mutex);

   if (util_dynarray_num_elements(&queue->free_cmd_buffers,
                                  struct vk_command_buffer *) > 0u) {
      vk_cmd = util_dynarray_pop(&queue->free_cmd_buffers,
                                 struct vk_command_buffer *);
   } else {
      result = kk_cmd_buffer_ops.create(
         &queue->cmd_pool->vk, VK_COMMAND_BUFFER_LEVEL_PRIMARY, &vk_cmd);
      if (result != VK_SUCCESS)
         goto unlock;
   }

   rerecord = container_of(vk_cmd, struct kk_cmd_buffer, vk);
   VkCommandBuffer rerecord_handle = kk_cmd_buffer_to_handle(rerecord);
   const VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   result = kk_BeginCommandBuffer(rerecord_handle, &begin_info);
   if (result != VK_SUCCESS)
      goto release;
   rerecord->replay = true;

   vk_cmd_queue_execute(&cmd->vk.cmd_queue, rerecord_handle,
                        &dev->vk.dispatch_table);

   result = kk_EndCommandBuffer(rerecord_handle);
   if (result != VK_SUCCESS)
      goto release;

   if (!kk_cmd_buffer_has_work(rerecord))
      goto release;

   struct kk_commit_data *commit = vk_alloc(&dev->vk.alloc, sizeof(*commit), 8,
                                            VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (commit == NULL) {
      result = vk_error(queue, VK_ERROR_OUT_OF_HOST_MEMORY);
      goto release;
   }
   commit->queue = queue;
   commit->cmd = rerecord;

   /* Need to ensure the new buffers allocated at record are resident. */
   kk_device_make_resources_resident(dev);

   queue->commits_in_flight++;
   mtx_unlock(&queue->mutex);

   kk_queue_commit(queue, rerecord, rerecord_commit_callback, commit);
   return VK_SUCCESS;

release:
   kk_queue_release_cmd_buffer_locked(queue, rerecord);
unlock:
   mtx_unlock(&queue->mutex);
   return result;
}

static VkResult
kk_queue_submit(struct vk_queue *vk_queue, struct vk_queue_submit *submit)
{
   struct kk_queue *queue = container_of(vk_queue, struct kk_queue, vk);
   struct kk_device *dev = kk_queue_device(queue);

   if (vk_queue_is_lost(&queue->vk))
      return VK_ERROR_DEVICE_LOST;

   for (struct vk_sync_wait *wait = submit->waits,
                            *end = submit->waits + submit->wait_count;
        wait != end; ++wait) {
      struct kk_sync_timeline *sync =
         container_of(wait->sync, struct kk_sync_timeline, base);
      if (mtl_shared_event_get_signaled_value(sync->mtl_handle) <
          wait->wait_value) {
         mtx_lock(&queue->mutex);
         uint32_t i = queue->pending_wait_next++ % KK_QUEUE_PENDING_WAITS;
         if (queue->pending_waits[i].event)
            mtl_release(queue->pending_waits[i].event);
         queue->pending_waits[i].event = mtl_retain(sync->mtl_handle);
         queue->pending_waits[i].value = wait->wait_value;
         queue->pending_waits[i].submitted_ns = os_time_get_nano();
         mtx_unlock(&queue->mutex);
      }
      mtl_wait_for_event(queue->mtl_handle, sync->mtl_handle, wait->wait_value);
   }

   /* Ensure any changes to residency are propagated before we submit any
    * work. All resources should have been allocated before submission.
    * Otherwise, users are playing with fire. */
   kk_device_make_resources_resident(dev);

   /* Mapping updates go on the Metal queue after the waits; signals below
    * follow them, command buffers barrier against them
    * (kk_encoder_wait_for_sparse_binds). */
   if (vk_queue_submit_has_bind(submit))
      kk_queue_bind_sparse(queue, submit);

   for (uint32_t i = 0; i < submit->command_buffer_count; ++i) {
      struct kk_cmd_buffer *cmd_buffer =
         container_of(submit->command_buffers[i], struct kk_cmd_buffer, vk);

      if (cmd_buffer->drawable) {
         mtl_command_queue_wait_for_drawable(queue->mtl_handle,
                                             cmd_buffer->drawable);
      }

      /* Metal's command buffers are one time use, re-record multiple
       * submissions, and command buffers whose sampler min/max emulation
       * changed after recording. */
      if (cmd_buffer->submitted ||
          kk_cmd_buffer_needs_minmax_replay(cmd_buffer)) {
         VkResult result = rerecord_and_commit_cmd_buffer(queue, cmd_buffer);
         if (result != VK_SUCCESS)
            return result;
      } else if (kk_cmd_buffer_has_work(cmd_buffer)) {
         kk_queue_commit(queue, cmd_buffer, commit_callback, dev);
      }

      cmd_buffer->submitted = true;

      if (cmd_buffer->drawable) {
         mtl_command_queue_signal_drawable(queue->mtl_handle,
                                           cmd_buffer->drawable);
         mtl_release(cmd_buffer->drawable);
         cmd_buffer->drawable = NULL;
      }
   }

   for (uint32_t i = 0u; i < submit->signal_count; ++i) {
      struct vk_sync_signal *signal = &submit->signals[i];
      struct kk_sync_timeline *sync =
         container_of(signal->sync, struct kk_sync_timeline, base);
      mtl_signal_event(queue->mtl_handle, sync->mtl_handle,
                       signal->signal_value);
   }

   return VK_SUCCESS;
}

VkResult
kk_queue_init(struct kk_device *dev, struct kk_queue *queue,
              const VkDeviceQueueCreateInfo *pCreateInfo,
              uint32_t index_in_family)
{
   const VkDeviceQueueGlobalPriorityCreateInfo *priority_info =
      vk_find_struct_const(pCreateInfo->pNext,
                           DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO);
   const VkQueueGlobalPriority global_priority =
      priority_info ? priority_info->globalPriority
                    : VK_QUEUE_GLOBAL_PRIORITY_MEDIUM;

   /* From the Vulkan 1.3.295 spec:
    *
    *    "If the globalPriorityQuery feature is enabled and the requested
    *    global priority is not reported via
    *    VkQueueFamilyGlobalPriorityPropertiesKHR, the driver implementation
    *    must fail the queue creation. In this scenario,
    *    VK_ERROR_INITIALIZATION_FAILED is returned."
    */
   if (dev->vk.enabled_features.globalPriorityQuery &&
       global_priority != VK_QUEUE_GLOBAL_PRIORITY_MEDIUM)
      return VK_ERROR_INITIALIZATION_FAILED;

   if (global_priority > VK_QUEUE_GLOBAL_PRIORITY_MEDIUM)
      return VK_ERROR_NOT_PERMITTED;

   VkResult result;

   result = vk_queue_init(&queue->vk, &dev->vk, pCreateInfo, index_in_family);
   if (result != VK_SUCCESS)
      return result;

   const VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = pCreateInfo->queueFamilyIndex,
   };
   VkCommandPool pool_handle;
   result = kk_CreateCommandPool(kk_device_to_handle(dev), &pool_info, NULL,
                                 &pool_handle);
   if (result != VK_SUCCESS) {
      vk_queue_finish(&queue->vk);
      return result;
   }
   queue->cmd_pool = kk_cmd_pool_from_handle(pool_handle);

   queue->free_cmd_buffers = UTIL_DYNARRAY_INIT;
   memset(queue->pending_waits, 0, sizeof(queue->pending_waits));
   memset(queue->records, 0, sizeof(queue->records));
   queue->pending_wait_next = 0;
   queue->commits_in_flight = 0u;
   queue->feedbacks_pending = 0u;
   queue->commit_seq = 0u;
   queue->last_gpu_end = 0.0;
   mtx_init(&queue->mutex, mtx_plain);
   cnd_init(&queue->cond);

   queue->mtl_handle = mtl_new_command_queue(dev->mtl_handle);
   mtl_command_queue_add_residency_set(queue->mtl_handle,
                                       dev->residency_set.handle);

   queue->vk.driver_submit = kk_queue_submit;

   if (dev->vk.enabled_features.sparseBinding) {
      result = kk_queue_sparse_init(dev, queue);
      if (result != VK_SUCCESS) {
         kk_queue_finish(dev, queue);
         return result;
      }
   }

   return VK_SUCCESS;
}

void
kk_queue_finish(struct kk_device *dev, struct kk_queue *queue)
{
   /* There may be in-flight command buffers, and feedback handlers of
    * completed ones may still run (after vkDeviceWaitIdle): wait for them. */
   kk_queue_wait_for_feedbacks(queue);

   kk_queue_sparse_finish(dev, queue);

   for (uint32_t i = 0; i < KK_QUEUE_PENDING_WAITS; i++) {
      if (queue->pending_waits[i].event)
         mtl_release(queue->pending_waits[i].event);
   }

   mtl_command_queue_remove_residency_set(queue->mtl_handle,
                                          dev->residency_set.handle);
   mtl_release(queue->mtl_handle);

   kk_DestroyCommandPool(kk_device_to_handle(dev),
                         kk_cmd_pool_to_handle(queue->cmd_pool), NULL);
   util_dynarray_fini(&queue->free_cmd_buffers);
   cnd_destroy(&queue->cond);
   mtx_destroy(&queue->mutex);

   vk_queue_finish(&queue->vk);
}
