/* Copyright 2026 steamac
 * SPDX-License-Identifier: MIT
 * Standalone: cc -std=c11 -Wall -Wextra -Werror steamac_gpu_diagnostics.c -o /tmp/kk-diag && /tmp/kk-diag
 */
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

   return 0;
}
