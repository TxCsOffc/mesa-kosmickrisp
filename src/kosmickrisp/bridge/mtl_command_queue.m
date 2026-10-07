/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "mtl_command_queue.h"

#include <Metal/MTLDevice.h>
#include <Metal/MTLCommandQueue.h>

/* MTL4CommitOptions */
mtl_commit_options *
mtl_new_commit_options(void)
{
   @autoreleasepool {
      return [MTL4CommitOptions new];
   }
}

void
mtl_commit_options_add_feedback_handler(mtl_commit_options *options,
                                        mtl_feedback_handler_callback callback,
                                        void *data)
{
   @autoreleasepool {
      MTL4CommitOptions* opt = (MTL4CommitOptions *)options;
      [opt addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
         @autoreleasepool {
            NSError *error = feedback.error;
            struct mtl_feedback_data feedback_data = (struct mtl_feedback_data){
               .user_data = data,
               .error_message =
                  error ? error.localizedDescription.UTF8String : NULL,
               .gpu_start = feedback.GPUStartTime,
               .gpu_end = feedback.GPUEndTime,
               .error = error
                           ? (enum mtl_command_queue_error)error.code
                           : MTL_COMMAND_QUEUE_ERROR_NONE,
            };
            callback(&feedback_data);
         }
      }];
   }
}

/* MTL4CommandQueue */
mtl_command_queue *
mtl_new_command_queue(mtl_device *device)
{
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;
      return [dev newMTL4CommandQueue];
   }
}

void
mtl_command_queue_add_residency_set(mtl_command_queue *cmd_queue,
                                    mtl_residency_set *set)
{
   @autoreleasepool {
      id<MTL4CommandQueue> queue = (id<MTL4CommandQueue>)cmd_queue;
      id<MTLResidencySet> s = (id<MTLResidencySet>)set;
      return [queue addResidencySet:s];
   }
}

void
mtl_command_queue_remove_residency_set(mtl_command_queue *cmd_queue,
                                       mtl_residency_set *set)
{
   @autoreleasepool {
      id<MTL4CommandQueue> queue = (id<MTL4CommandQueue>)cmd_queue;
      id<MTLResidencySet> s = (id<MTLResidencySet>)set;
      return [queue removeResidencySet:s];
   }
}

void
mtl_signal_event(mtl_command_queue *queue, mtl_event *event, uint64_t value)
{
   @autoreleasepool {
      id<MTL4CommandQueue> q = (id<MTL4CommandQueue>)queue;
      id<MTLEvent> e = (id<MTLEvent>)event;
      [q signalEvent:e value:value];
   }
}

void
mtl_wait_for_event(mtl_command_queue *queue, mtl_event *event, uint64_t value)
{
   @autoreleasepool {
      id<MTL4CommandQueue> q = (id<MTL4CommandQueue>)queue;
      id<MTLEvent> e = (id<MTLEvent>)event;
      [q waitForEvent:e value:value];
   }
}

#define MTL_SPARSE_MAPPING_BATCH 64u

void
mtl_command_queue_update_texture_mappings(
   mtl_command_queue *queue, mtl_texture *texture, mtl_heap *heap,
   const struct mtl_sparse_texture_mapping *mappings, uint32_t count)
{
   @autoreleasepool {
      id<MTL4CommandQueue> q = (id<MTL4CommandQueue>)queue;
      MTL4UpdateSparseTextureMappingOperation ops[MTL_SPARSE_MAPPING_BATCH];
      for (uint32_t base = 0; base < count; base += MTL_SPARSE_MAPPING_BATCH) {
         uint32_t n = count - base < MTL_SPARSE_MAPPING_BATCH ? count - base : MTL_SPARSE_MAPPING_BATCH;
         for (uint32_t i = 0; i < n; i++) {
            const struct mtl_sparse_texture_mapping *m = &mappings[base + i];
            ops[i] = (MTL4UpdateSparseTextureMappingOperation){
               .mode = m->unmap ? MTLSparseTextureMappingModeUnmap
                                : MTLSparseTextureMappingModeMap,
               .textureRegion = MTLRegionMake3D(m->origin.x, m->origin.y, m->origin.z,
                                                m->size.x, m->size.y, m->size.z),
               .textureLevel = m->level,
               .textureSlice = m->slice,
               .heapOffset = m->heap_offset,
            };
         }
         [q updateTextureMappings:(id<MTLTexture>)texture
                             heap:(id<MTLHeap>)heap
                       operations:ops
                            count:n];
      }
   }
}

void
mtl_command_queue_update_buffer_mappings(
   mtl_command_queue *queue, mtl_buffer *buffer, mtl_heap *heap,
   const struct mtl_sparse_buffer_mapping *mappings, uint32_t count)
{
   @autoreleasepool {
      id<MTL4CommandQueue> q = (id<MTL4CommandQueue>)queue;
      MTL4UpdateSparseBufferMappingOperation ops[MTL_SPARSE_MAPPING_BATCH];
      for (uint32_t base = 0; base < count; base += MTL_SPARSE_MAPPING_BATCH) {
         uint32_t n = count - base < MTL_SPARSE_MAPPING_BATCH ? count - base : MTL_SPARSE_MAPPING_BATCH;
         for (uint32_t i = 0; i < n; i++) {
            const struct mtl_sparse_buffer_mapping *m = &mappings[base + i];
            ops[i] = (MTL4UpdateSparseBufferMappingOperation){
               .mode = m->unmap ? MTLSparseTextureMappingModeUnmap
                                : MTLSparseTextureMappingModeMap,
               .bufferRange = NSMakeRange(m->offset, m->length),
               .heapOffset = m->heap_offset,
            };
         }
         [q updateBufferMappings:(id<MTLBuffer>)buffer
                            heap:(id<MTLHeap>)heap
                      operations:ops
                           count:n];
      }
   }
}

void
mtl_command_queue_commit(mtl_command_queue *queue,
                         mtl_command_buffer **command_buffers, uint32_t count,
                         mtl_commit_options *options)
{
   @autoreleasepool {
      id<MTL4CommandQueue> q = (id<MTL4CommandQueue>)queue;
      id<MTL4CommandBuffer> *cmds = (id<MTL4CommandBuffer> *)command_buffers;
      MTL4CommitOptions *opt = (MTL4CommitOptions *)options;
      [q commit:cmds count:count options:opt];
   }
}

void
mtl_command_queue_wait_for_drawable(mtl_command_queue *queue, void *drawable)
{
   @autoreleasepool {
      id<MTL4CommandQueue> q = (id<MTL4CommandQueue>)queue;
      id<MTLDrawable> d = (id<MTLDrawable>)drawable;
      [q waitForDrawable:d];
   }
}

void
mtl_command_queue_signal_drawable(mtl_command_queue *queue, void *drawable)
{
   @autoreleasepool {
      id<MTL4CommandQueue> q = (id<MTL4CommandQueue>)queue;
      id<MTLDrawable> d = (id<MTLDrawable>)drawable;
      [q signalDrawable:d];
   }
}
