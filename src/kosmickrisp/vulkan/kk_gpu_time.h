/* Copyright 2026 steamac
 * SPDX-License-Identifier: MIT
 */
#ifndef KK_GPU_TIME_H
#define KK_GPU_TIME_H

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

static inline bool
kk_gpu_time_valid_start(double start, double commit, double feedback)
{
   return isfinite(start) && isfinite(commit) && isfinite(feedback) &&
          start > 0.0 && start >= commit - 0.005 && start <= feedback + 0.005;
}

static inline bool
kk_gpu_time_valid_end(double start, double end, double feedback)
{
   return isfinite(end) && end >= start && end <= feedback + 0.005;
}

/* A stale start must not turn hours of completed work into "in flight".
 * Commit time is a conservative lower bound when GPU timestamps are absent. */
static inline double
kk_gpu_time_window_start(double start, double commit, double feedback)
{
   return kk_gpu_time_valid_start(start, commit, feedback) ? start : commit;
}

/* Error feedback can contain stale or sentinel timestamps (STEAMAC-G).
 * Keep the raw values in the caller, but never turn those into GPU durations.
 * Permit small CPU/GPU clock calibration differences, not arbitrary runtimes. */
static inline void
kk_gpu_time_format(double start, double end, double commit, double feedback,
                   char *out, size_t size)
{
   const bool valid_start = kk_gpu_time_valid_start(start, commit, feedback);
   if (start == 0.0) {
      snprintf(out, size, "has no GPU start timestamp");
   } else if (!valid_start) {
      snprintf(out, size, "has an inconsistent GPU start timestamp");
   } else if (!kk_gpu_time_valid_end(start, end, feedback)) {
      snprintf(out, size, "started %.1f ms after its commit; GPU end unavailable or inconsistent",
               (start - commit) * 1000.0);
   } else {
      snprintf(out, size, "started %.1f ms after its commit and ran %.1f ms on the GPU",
               (start - commit) * 1000.0, (end - start) * 1000.0);
   }
}

#endif
