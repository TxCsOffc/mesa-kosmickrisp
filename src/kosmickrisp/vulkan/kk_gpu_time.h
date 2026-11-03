/* Copyright 2026 steamac
 * SPDX-License-Identifier: MIT
 */
#ifndef KK_GPU_TIME_H
#define KK_GPU_TIME_H

#include <math.h>
#include <stddef.h>
#include <stdio.h>

/* Error feedback can contain stale or sentinel timestamps (STEAMAC-G).
 * Keep the raw values in the caller, but never turn those into GPU durations.
 * Permit small CPU/GPU clock calibration differences, not arbitrary runtimes. */
static inline void
kk_gpu_time_format(double start, double end, double commit, double feedback,
                   char *out, size_t size)
{
   const double tolerance = 0.005;
   if (start == 0.0) {
      snprintf(out, size, "has no GPU start timestamp");
   } else if (!isfinite(start) || !isfinite(commit) || !isfinite(feedback) ||
              start < commit - tolerance || start > feedback + tolerance) {
      snprintf(out, size, "has an inconsistent GPU start timestamp");
   } else if (!isfinite(end) || end < start || end > feedback + tolerance) {
      snprintf(out, size, "started %.1f ms after its commit; GPU end unavailable or inconsistent",
               (start - commit) * 1000.0);
   } else {
      snprintf(out, size, "started %.1f ms after its commit and ran %.1f ms on the GPU",
               (start - commit) * 1000.0, (end - start) * 1000.0);
   }
}

#endif
