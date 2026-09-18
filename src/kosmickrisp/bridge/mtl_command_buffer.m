/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "mtl_command_buffer.h"

#include <Metal/MTL4BufferRange.h>
#include <Metal/MTL4CommandAllocator.h>
#include <Metal/MTL4CommandBuffer.h>
#include <Metal/MTL4Counters.h>

#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>

static bool log_every_resolve;

void
mtl_command_buffer_set_log(bool every_call)
{
   log_every_resolve = every_call;
}

void
mtl_command_allocator_reset(mtl_command_allocator *allocator)
{
   @autoreleasepool {
      id<MTL4CommandAllocator> alloc = (id<MTL4CommandAllocator>)allocator;
      [alloc reset];
   }
}

void
mtl_begin_command_buffer(mtl_command_buffer *command_buffer,
                         mtl_command_allocator *allocator)
{
   @autoreleasepool {
      id<MTL4CommandBuffer> cmd_buf = (id<MTL4CommandBuffer>)command_buffer;
      id<MTL4CommandAllocator> alloc = (id<MTL4CommandAllocator>)allocator;
      [cmd_buf beginCommandBufferWithAllocator:alloc];
   }
}

void
mtl_end_command_buffer(mtl_command_buffer *command_buffer)
{
   @autoreleasepool {
      id<MTL4CommandBuffer> cmd_buf = (id<MTL4CommandBuffer>)command_buffer;
      [cmd_buf endCommandBuffer];
   }
}

void
mtl_command_resolve_counter_heap(mtl_command_buffer *command_buffer,
                                 mtl_counter_heap *heap, uint32_t first_index,
                                 uint32_t count, uint64_t dst_addr)
{
   @autoreleasepool {
      id<MTL4CommandBuffer> cmd_buf = (id<MTL4CommandBuffer>)command_buffer;
      id<MTL4CounterHeap> h = (id<MTL4CounterHeap>)heap;
      MTL4BufferRange range = {
         .bufferAddress = dst_addr,
         .length = count * sizeof(MTL4TimestampHeapEntry),
      };
      /* STEAMAC-7R: AGX dereferences NULL while resolving on macOS 26.0 beta.
       * Neither the heap/range nor the destination reached that crash report.
       * Keep a bounded sample in release builds, full call/return pairs with
       * cbs, and always report invalid operands without hiding the error. */
      static _Thread_local unsigned logged;
      const uint64_t capacity = h.count;
      const bool invalid = !cmd_buf || !h || !dst_addr || !count ||
         first_index > capacity || count > capacity - first_index;
      const bool log = log_every_resolve || invalid || logged < 64;
      uint64_t thread = 0;
      if (log) {
         pthread_threadid_np(NULL, &thread);
         fprintf(stderr, "kk: counter resolve thread=%" PRIu64
                 " cmd=%p heap=%p capacity=%" PRIu64 " type=%ld"
                 " first=%u count=%u dst=0x%" PRIx64 " bytes=%" PRIu64
                 " invalid=%u\n", thread, (void *)cmd_buf, (void *)h,
                 capacity, (long)h.type, first_index, count, dst_addr,
                 (uint64_t)range.length, invalid);
      }
      [cmd_buf resolveCounterHeap:h
                        withRange:NSMakeRange(first_index, count)
                       intoBuffer:range
                        waitFence:nil
                      updateFence:nil];
      if (log) {
         fprintf(stderr, "kk: counter resolve ready thread=%" PRIu64 "\n",
                 thread);
         if (!log_every_resolve && !invalid && ++logged == 64)
            fprintf(stderr, "kk: counter resolve: 64 logged on thread %"
                    PRIu64 ", use MESA_KK_DEBUG=cbs for every call\n", thread);
      }
   }
}

void
mtl_command_buffer_set_label(mtl_command_buffer *command_buffer,
                             const char *label)
{
   @autoreleasepool {
      id<MTL4CommandBuffer> cb = (id<MTL4CommandBuffer>)command_buffer;
      cb.label = @(label);
   }
}

void
mtl_command_buffer_push_debug_group(mtl_command_buffer *command_buffer,
                                    const char *label)
{
    @autoreleasepool {
      id<MTL4CommandBuffer> cb = (id<MTL4CommandBuffer>)command_buffer;
      [cb pushDebugGroup:@(label)];
    }
}

void mtl_command_buffer_pop_debug_group(mtl_command_buffer *command_buffer)
{
   @autoreleasepool {
      id<MTL4CommandBuffer> cb = (id<MTL4CommandBuffer>)command_buffer;
      [cb popDebugGroup];
   }
}
