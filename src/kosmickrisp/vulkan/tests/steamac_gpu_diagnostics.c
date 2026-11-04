/* Copyright 2026 steamac
 * SPDX-License-Identifier: MIT
 * Standalone: cc -std=c11 -Wall -Wextra -Werror steamac_gpu_diagnostics.c -o /tmp/kk-diag && /tmp/kk-diag
 */
#include "../kk_cb_record.h"
#include "../kk_gpu_time.h"
#include <assert.h>
#include <string.h>

int main(void)
{
   char text[192];
   kk_gpu_time_format(100.001, 100.026, 100.0, 100.030, text, sizeof(text));
   assert(strstr(text, "ran 25.0 ms"));
   kk_gpu_time_format(100.001, 6148414588.946290, 100.0, 100.046, text, sizeof(text));
   assert(strstr(text, "GPU end unavailable or inconsistent"));
   assert(!strstr(text, "ran"));
   kk_gpu_time_format(61521.381583, 195425.188578, 195424.685835, 195425.196, text, sizeof(text));
   assert(strstr(text, "inconsistent GPU start"));
   kk_gpu_time_format(0.0, 0.0, 100.0, 100.020, text, sizeof(text));
   assert(strstr(text, "no GPU start timestamp"));
   kk_gpu_time_format(NAN, INFINITY, 100.0, 100.020, text, sizeof(text));
   assert(strstr(text, "inconsistent GPU start"));
   kk_gpu_time_format(99.9985, 100.010, 100.0, 100.020, text, sizeof(text));
   assert(strstr(text, "ran 11.5 ms"));

   /* 1.9.9 event 048177a5: start predates commit by 43.7 hours.
    * Previously unrelated completed records appeared to overlap it. */
   const double commit = 1578359.967056, feedback = 1578360.064688;
   const double stale_start = 1421121.994090;
   assert(kk_gpu_time_window_start(stale_start, commit, feedback) == commit);
   assert(kk_gpu_time_window_start(0.0, commit, feedback) == commit);
   assert(kk_gpu_time_window_start(NAN, commit, feedback) == commit);
   assert(kk_gpu_time_window_start(commit + 0.001, commit, feedback) == commit + 0.001);
   assert(!kk_gpu_time_valid_start(stale_start, commit, feedback));
   assert(!kk_gpu_time_valid_end(commit, 6148414588.946290, feedback));
   assert(!kk_gpu_time_valid_end(commit, NAN, feedback));
   assert(kk_gpu_time_valid_end(commit, feedback + 0.004, feedback));

   struct kk_cb_record record = {0};
   struct kk_dispatch_record shape = {.hash = 0x1234, .groups = {64, 32, 1}, .local = {8, 8, 1}};
   kk_cb_record_dispatch(&record, &shape);
   assert(record.largest_direct_groups == 2048);
   assert(record.largest_direct.hash == 0x1234);
   shape.hash = 0x5678;
   shape.groups[0] = 1;
   kk_cb_record_dispatch(&record, &shape);
   assert(record.largest_direct.hash == 0x1234);
   shape.indirect = true;
   shape.indirect_addr = 0xabcd;
   kk_cb_record_dispatch(&record, &shape);
   assert(record.last_indirect.hash == 0x5678);
   assert(record.last_indirect.indirect_addr == 0xabcd);
   assert(record.largest_direct_groups == 2048);
   shape.indirect = false;
   shape.groups[0] = shape.groups[1] = shape.groups[2] = UINT32_MAX;
   kk_cb_record_dispatch(&record, &shape);
   assert(record.largest_direct_groups == UINT64_MAX);
   return 0;
}
