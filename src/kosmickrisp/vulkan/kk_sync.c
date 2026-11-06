/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "kk_sync.h"

#include "kk_device.h"
#include "kk_debug.h"

#include "kosmickrisp/bridge/mtl_bridge.h"
#include "util/log.h"

static VkResult
kk_timeline_init(struct vk_device *device, struct vk_sync *sync,
                 uint64_t initial_value)
{
   struct kk_sync_timeline *timeline =
      container_of(sync, struct kk_sync_timeline, base);

   struct kk_device *dev = container_of(device, struct kk_device, vk);
   timeline->mtl_handle = mtl_new_shared_event(dev->mtl_handle);
   mtl_shared_event_set_signaled_value(timeline->mtl_handle, initial_value);
   if (KK_DEBUG(SYNCS))
      mesa_logi("kk: sync create event=%p initial=%" PRIu64,
                (void *)timeline->mtl_handle, initial_value);

   return VK_SUCCESS;
}

static void
kk_timeline_finish(struct vk_device *device, struct vk_sync *sync)
{
   struct kk_sync_timeline *timeline =
      container_of(sync, struct kk_sync_timeline, base);
   if (KK_DEBUG(SYNCS))
      mesa_logi("kk: sync destroy event=%p", (void *)timeline->mtl_handle);
   mtl_release(timeline->mtl_handle);
}

static VkResult
kk_timeline_signal(struct vk_device *device, struct vk_sync *sync,
                   uint64_t value)
{
   struct kk_sync_timeline *timeline =
      container_of(sync, struct kk_sync_timeline, base);
   mtl_shared_event_set_signaled_value(timeline->mtl_handle, value);
   if (KK_DEBUG(SYNCS))
      mesa_logi("kk: sync CPU signal event=%p value=%" PRIu64,
                (void *)timeline->mtl_handle, value);
   return VK_SUCCESS;
}

static VkResult
kk_timeline_get_value(struct vk_device *device, struct vk_sync *sync,
                      uint64_t *value)
{
   struct kk_sync_timeline *timeline =
      container_of(sync, struct kk_sync_timeline, base);
   *value = mtl_shared_event_get_signaled_value(timeline->mtl_handle);
   return VK_SUCCESS;
}

static VkResult
kk_timeline_wait(struct vk_device *device, struct vk_sync *sync,
                 uint64_t wait_value, enum vk_sync_wait_flags wait_flags,
                 uint64_t abs_timeout_ns)
{
   struct kk_sync_timeline *timeline =
      container_of(sync, struct kk_sync_timeline, base);

   /* abs_timeout_ns is the point in time when we should stop waiting, not the
    * absolute time in ns. Therefore, we need to compute the delta from now to
    * when we should stop waiting and convert to ms for Metal to be happy
    * (Similar to what dzn does).
    */
   uint64_t timeout_ms = 0u;
   if (abs_timeout_ns == OS_TIMEOUT_INFINITE) {
      timeout_ms = OS_TIMEOUT_INFINITE;
   } else {
      uint64_t cur_time = os_time_get_nano();
      uint64_t rel_timeout_ns =
         abs_timeout_ns > cur_time ? abs_timeout_ns - cur_time : 0;

      timeout_ms =
         (rel_timeout_ns / 1000000) + (rel_timeout_ns % 1000000 ? 1 : 0);
   }
   /* Opt-in only: report an outstanding CPU wait before it can block forever.
    * Keep the original absolute deadline, including finite waits. */
   const bool trace = KK_DEBUG(SYNCS) && timeout_ms > 1000;
   const uint64_t started = trace ? os_time_get_nano() : 0;
   if (trace) {
      if (mtl_shared_event_wait_until_signaled_value(
             timeline->mtl_handle, wait_value, 1000))
         return VK_SUCCESS;
      char app[64];
      struct kk_device *dev = container_of(device, struct kk_device, vk);
      kk_device_app_name(dev, app, sizeof(app));
      mesa_logi("kk: sync CPU wait of %s event=%p target=%" PRIu64
                " current=%" PRIu64 " flags=0x%x deadline=%" PRIu64,
                app, (void *)timeline->mtl_handle, wait_value,
                mtl_shared_event_get_signaled_value(timeline->mtl_handle),
                wait_flags, abs_timeout_ns);
      if (dev->has_queue)
         kk_queue_log_sync_wait(&dev->queue);
      if (abs_timeout_ns != OS_TIMEOUT_INFINITE) {
         const uint64_t now = os_time_get_nano();
         const uint64_t remaining = abs_timeout_ns > now ? abs_timeout_ns - now : 0;
         timeout_ms = remaining / 1000000 + (remaining % 1000000 != 0);
      }
   }
   int completed = mtl_shared_event_wait_until_signaled_value(
      timeline->mtl_handle, wait_value, timeout_ms);
   if (trace)
      mesa_logi("kk: sync CPU wait ended event=%p target=%" PRIu64
                " current=%" PRIu64 " completed=%d elapsed-ms=%.1f",
                (void *)timeline->mtl_handle, wait_value,
                mtl_shared_event_get_signaled_value(timeline->mtl_handle),
                completed, (os_time_get_nano() - started) / 1e6);

   return completed != 0 ? VK_SUCCESS : VK_TIMEOUT;
}

const struct vk_sync_type kk_sync_type = {
   .size = sizeof(struct kk_sync_timeline),
   .features = VK_SYNC_FEATURE_TIMELINE | VK_SYNC_FEATURE_GPU_WAIT |
               VK_SYNC_FEATURE_CPU_WAIT | VK_SYNC_FEATURE_CPU_SIGNAL |
               VK_SYNC_FEATURE_WAIT_PENDING |
               VK_SYNC_FEATURE_WAIT_BEFORE_SIGNAL,
   .init = kk_timeline_init,
   .finish = kk_timeline_finish,
   .signal = kk_timeline_signal,
   .get_value = kk_timeline_get_value,
   .reset = NULL,
   .move = NULL,
   .wait = kk_timeline_wait,
   .wait_many = NULL,
   .import_opaque_fd = NULL,
   .export_opaque_fd = NULL,
   .import_sync_file = NULL,
   .export_sync_file = NULL,
   .import_win32_handle = NULL,
   .export_win32_handle = NULL,
   .set_win32_export_params = NULL,
};
