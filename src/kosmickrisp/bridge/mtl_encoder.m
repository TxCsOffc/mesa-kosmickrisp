/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "mtl_encoder.h"
#include "util/hash_table.h"

#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>
#include <Metal/MTL4RenderPass.h>
#include <Metal/MTLDevice.h>

#include <Metal/MTL4CommandBuffer.h>
#include <Metal/MTL4ComputeCommandEncoder.h>
#include <Metal/MTL4Counters.h>
#include <Metal/MTL4RenderCommandEncoder.h>
#include <Metal/MTLRenderPass.h>

/* The M4 reports abort in Apple's background/load program compiler without
 * its reason reaching stderr. Keep the inputs before entering AGX, not only
 * after it returns. A small per-thread shape cache bounds the bookkeeping
 * and avoids logging the same alternating framebuffers every frame. cbs
 * bypasses that cache for an exact call/return trace on any GPU.
 */
static bool encoder_log_all;

void
mtl_encoder_set_log(bool enabled)
{
   encoder_log_all = enabled;
}

static bool
encoder_diagnostics_enabled(id<MTLDevice> device)
{
   static _Thread_local id<MTLDevice> last_device;
   static _Thread_local bool enabled;
   if (encoder_log_all)
      return true;
   if (device != last_device) {
      last_device = device;
      enabled = [device supportsFamily:MTLGPUFamilyApple9];
   }
   return enabled;
}

static bool
encoder_log_shape(uint32_t hash)
{
   static _Thread_local uint32_t shapes[64];
   static _Thread_local bool occupied[64];
   const unsigned slot = hash % 64;
   if (!encoder_log_all && occupied[slot] && shapes[slot] == hash)
      return false;
   shapes[slot] = hash;
   occupied[slot] = true;
   return true;
}

struct encoder_texture_shape {
   uint64_t format, type, width, height, depth, levels, layers, samples;
   uint64_t usage, storage, buffer_backed;
};

static struct encoder_texture_shape
encoder_texture_shape(id<MTLTexture> texture)
{
   return (struct encoder_texture_shape) {
      texture.pixelFormat, texture.textureType,
      texture.width, texture.height, texture.depth,
      texture.mipmapLevelCount, texture.arrayLength, texture.sampleCount,
      texture.usage, texture.storageMode, texture.buffer != nil,
   };
}

static void
encoder_print_texture(const char *name, id<MTLTexture> texture,
                      const struct encoder_texture_shape *s)
{
   fprintf(stderr, " %s=%p{fmt=%" PRIu64 " type=%" PRIu64
           " size=%" PRIu64 "x%" PRIu64 "x%" PRIu64
           " levels=%" PRIu64 " layers=%" PRIu64 " samples=%" PRIu64
           " usage=0x%" PRIx64 " storage=%" PRIu64 " buffer_backed=%" PRIu64 "}",
           name, (void *)texture, s->format, s->type, s->width, s->height,
           s->depth, s->levels, s->layers, s->samples, s->usage,
           s->storage, s->buffer_backed);
}

static uint64_t
encoder_thread_id(void)
{
   uint64_t thread;
   pthread_threadid_np(NULL, &thread);
   return thread;
}

static bool
encoder_log_buffer_texture(struct mtl_buffer_image_copy *data,
                           bool to_texture, uint32_t *hash)
{
   id<MTLBuffer> buffer = (id<MTLBuffer>)data->buffer;
   id<MTLTexture> texture = (id<MTLTexture>)data->image;
   if (!encoder_diagnostics_enabled(texture.device ?: buffer.device) && buffer && texture)
      return false;

   /* Never hash pointers: new resources with the same layout are one shape. */
   struct {
      struct encoder_texture_shape texture;
      uint64_t length, offset, row, image, x, y, z, w, h, d, level, slice;
      uint64_t options, to_texture;
   } shape = {
      encoder_texture_shape(texture), buffer.length,
      data->buffer_offset_B, data->buffer_stride_B,
      data->buffer_2d_image_size_B,
      data->image_origin.x, data->image_origin.y, data->image_origin.z,
      data->image_size.x, data->image_size.y, data->image_size.z,
      data->image_level, data->image_slice, data->options, to_texture,
   };
   *hash = _mesa_hash_data(&shape, sizeof(shape));
   if (!encoder_log_shape(*hash))
      return false;
   flockfile(stderr);
   fprintf(stderr, "kk: encoder copy shape=%08x thread=%" PRIu64
           " %s buffer=%p length=%" PRIu64 " offset=%" PRIu64
           " row=%" PRIu64 " image=%" PRIu64,
           *hash, encoder_thread_id(), to_texture ? "buffer->texture" : "texture->buffer",
           (void *)buffer, shape.length, shape.offset, shape.row, shape.image);
   encoder_print_texture("texture", texture, &shape.texture);
   fprintf(stderr, " origin=%" PRIu64 ",%" PRIu64 ",%" PRIu64
           " extent=%" PRIu64 ",%" PRIu64 ",%" PRIu64
           " level=%" PRIu64 " slice=%" PRIu64 " options=0x%" PRIx64 "\n",
           shape.x, shape.y, shape.z, shape.w, shape.h, shape.d,
           shape.level, shape.slice, shape.options);
   funlockfile(stderr);
   return true;
}

static bool
encoder_log_render(id<MTL4CommandBuffer> command, MTL4RenderPassDescriptor *desc,
                   uint32_t *hash)
{
   if (!encoder_diagnostics_enabled(command.device))
      return false;
   MTLRenderPassAttachmentDescriptor *attachments[10];
   struct attachment_shape {
      struct encoder_texture_shape texture, resolve;
      uint64_t level, slice, plane, load, store, resolve_level, resolve_slice;
      uint64_t resolve_plane;
   };
   struct {
      uint64_t width, height, layers, samples;
      struct attachment_shape attachments[10];
   } shape = {0};
   shape.width = desc.renderTargetWidth;
   shape.height = desc.renderTargetHeight;
   shape.layers = desc.renderTargetArrayLength;
   shape.samples = desc.defaultRasterSampleCount;
   for (unsigned i = 0; i < 10; ++i) {
      MTLRenderPassAttachmentDescriptor *a =
         i < 8 ? desc.colorAttachments[i] :
         i == 8 ? desc.depthAttachment : desc.stencilAttachment;
      attachments[i] = a;
      shape.attachments[i] = (struct attachment_shape) {
         encoder_texture_shape(a.texture), encoder_texture_shape(a.resolveTexture),
         a.level, a.slice, a.depthPlane, a.loadAction, a.storeAction,
         a.resolveLevel, a.resolveSlice, a.resolveDepthPlane,
      };
   }
   *hash = _mesa_hash_data(&shape, sizeof(shape));
   if (!encoder_log_shape(*hash))
      return false;
   flockfile(stderr);
   fprintf(stderr, "kk: encoder render shape=%08x thread=%" PRIu64
           " cmd=%p target=%" PRIu64 "x%" PRIu64 " layers=%" PRIu64
           " default_samples=%" PRIu64 " clear_depth=%g clear_stencil=%" PRIu64 "\n",
           *hash, encoder_thread_id(), (void *)command, shape.width, shape.height,
           shape.layers, shape.samples, desc.depthAttachment.clearDepth,
           (uint64_t)desc.stencilAttachment.clearStencil);
   for (unsigned i = 0; i < 10; ++i) {
      MTLRenderPassAttachmentDescriptor *a = attachments[i];
      if (!a.texture && !a.resolveTexture)
         continue;
      const struct attachment_shape *s = &shape.attachments[i];
      fprintf(stderr, "kk: encoder render shape=%08x attachment=%u", *hash, i);
      encoder_print_texture("texture", a.texture, &s->texture);
      encoder_print_texture("resolve", a.resolveTexture, &s->resolve);
      fprintf(stderr, " level=%" PRIu64 " slice=%" PRIu64 " plane=%" PRIu64
              " load=%" PRIu64 " store=%" PRIu64
              " resolve_level=%" PRIu64 " resolve_slice=%" PRIu64
              " resolve_plane=%" PRIu64,
              s->level, s->slice, s->plane, s->load, s->store,
              s->resolve_level, s->resolve_slice, s->resolve_plane);
      if (i < 8) {
         const MTLClearColor color = desc.colorAttachments[i].clearColor;
         fprintf(stderr, " clear=%g,%g,%g,%g", color.red, color.green,
                 color.blue, color.alpha);
      }
      fputc('\n', stderr);
   }
   funlockfile(stderr);
   return true;
}

/* Common encoder utils */
void
mtl_end_encoding(void *encoder)
{
   @autoreleasepool {
      id<MTL4CommandEncoder> enc = (id<MTL4CommandEncoder>)encoder;
      [enc endEncoding];
   }
}

void
mtl_barrier_after_stages(void *encoder, enum mtl_stages after_stages,
                         enum mtl_stages before_queue_stages)
{
   @autoreleasepool {
      id<MTL4CommandEncoder> enc = (id<MTL4CommandEncoder>)encoder;
      /* TODO_KOSMICKRISP Expose visibility option since resource alias should only be required
       * if occlusion queries are used */
      [enc barrierAfterStages:(MTLStages)after_stages beforeQueueStages:(MTLStages)before_queue_stages
           visibilityOptions:MTL4VisibilityOptionResourceAlias];
   }
}

void
mtl_barrier_after_encoder_stages(void *encoder, enum mtl_stages after_stages,
                                 enum mtl_stages before_queue_stages)
{
   @autoreleasepool {
      id<MTL4CommandEncoder> enc = (id<MTL4CommandEncoder>)encoder;
      [enc barrierAfterEncoderStages:(MTLStages)after_stages beforeEncoderStages:(MTLStages)before_queue_stages
           visibilityOptions:MTL4VisibilityOptionResourceAlias];
   }
}

void
mtl_barrier_after_queue_stages(void *encoder,
                               enum mtl_stages after_queue_stages,
                               enum mtl_stages before_stages)
{
   @autoreleasepool {
      id<MTL4CommandEncoder> enc = (id<MTL4CommandEncoder>)encoder;
      [enc barrierAfterQueueStages:(MTLStages)after_queue_stages
                      beforeStages:(MTLStages)before_stages
                 visibilityOptions:MTL4VisibilityOptionResourceAlias];
   }
}

void
mtl_update_fence(void *encoder, mtl_fence *fence, enum mtl_stages after_stages)
{
   @autoreleasepool {
      id<MTL4CommandEncoder> enc = (id<MTL4CommandEncoder>)encoder;
      id<MTLFence> f = (id<MTLFence>)fence;
      [enc updateFence:f afterEncoderStages:(MTLStages)after_stages];
   }
}

void
mtl_wait_for_fence(void *encoder, mtl_fence *fence,
                   enum mtl_stages before_stages)
{
   @autoreleasepool {
      id<MTL4CommandEncoder> enc = (id<MTL4CommandEncoder>)encoder;
      id<MTLFence> f = (id<MTLFence>)fence;
      [enc waitForFence:f beforeEncoderStages:(MTLStages)before_stages];
   }
}

void
mtl_encoder_set_label(void *encoder, const char *label)
{
   @autoreleasepool {
      id<MTL4CommandEncoder> enc = (id<MTL4CommandEncoder>)encoder;
      enc.label = @(label);
   }
}

void
mtl_encoder_insert_debug_signpost(void *encoder, const char *label)
{
   @autoreleasepool {
      id<MTL4CommandEncoder> enc = (id<MTL4CommandEncoder>)encoder;
      [enc insertDebugSignpost:@(label)];
   }
}

void
mtl_encoder_push_debug_group(void *encoder, const char *label)
{
   @autoreleasepool {
      id<MTL4CommandEncoder> enc = (id<MTL4CommandEncoder>)encoder;
      [enc pushDebugGroup:@(label)];
   }
}

void
mtl_encoder_pop_debug_group(void *encoder)
{
   @autoreleasepool {
      id<MTL4CommandEncoder> enc = (id<MTL4CommandEncoder>)encoder;
      [enc popDebugGroup];
   }
}

/* MTLComputeEncoder */
mtl_compute_encoder *
mtl_new_compute_command_encoder(mtl_command_buffer *cmd_buffer)
{
   @autoreleasepool {
      id<MTL4CommandBuffer> cmd_buf = (id<MTL4CommandBuffer>)cmd_buffer;
      return [[cmd_buf computeCommandEncoder] retain];
   }
}

void
mtl_copy_from_buffer_to_buffer(mtl_compute_encoder *encoder,
                               mtl_buffer *src_buf, size_t src_offset,
                               mtl_buffer *dst_buf, size_t dst_offset,
                               size_t size)
{
   @autoreleasepool {
      id<MTL4ComputeCommandEncoder> enc = (id<MTL4ComputeCommandEncoder>)encoder;
      id<MTLBuffer> mtl_src_buffer = (id<MTLBuffer>)src_buf;
      id<MTLBuffer> mtl_dst_buffer = (id<MTLBuffer>)dst_buf;
      [enc copyFromBuffer:mtl_src_buffer sourceOffset:src_offset toBuffer:mtl_dst_buffer destinationOffset:dst_offset size:size];
   }
}

void
mtl_copy_from_buffer_to_texture(mtl_compute_encoder *encoder,
                                struct mtl_buffer_image_copy *data)
{
   @autoreleasepool {
      const MTLSize size = MTLSizeMake(data->image_size.x, data->image_size.y, data->image_size.z);
      const MTLOrigin origin = MTLOriginMake(data->image_origin.x, data->image_origin.y, data->image_origin.z);
      id<MTL4ComputeCommandEncoder> enc = (id<MTL4ComputeCommandEncoder>)encoder;
      id<MTLBuffer> buffer = (id<MTLBuffer>)data->buffer;
      id<MTLTexture> image = (id<MTLTexture>)data->image;
      uint32_t shape = 0;
      const bool logged = encoder_log_buffer_texture(data, true, &shape);
      [enc copyFromBuffer:buffer
             sourceOffset:data->buffer_offset_B
        sourceBytesPerRow:data->buffer_stride_B
      sourceBytesPerImage:data->buffer_2d_image_size_B
               sourceSize:size
                toTexture:image
         destinationSlice:data->image_slice
         destinationLevel:data->image_level
        destinationOrigin:origin
                  options:(MTLBlitOption)data->options];
      if (logged)
         fprintf(stderr, "kk: encoder copy ready shape=%08x thread=%" PRIu64 "\n",
                 shape, encoder_thread_id());
   }
}

void
mtl_copy_from_texture_to_buffer(mtl_compute_encoder *encoder,
                                struct mtl_buffer_image_copy *data)
{
   @autoreleasepool {
      const MTLSize size = MTLSizeMake(data->image_size.x, data->image_size.y, data->image_size.z);
      const MTLOrigin origin = MTLOriginMake(data->image_origin.x, data->image_origin.y, data->image_origin.z);
      id<MTL4ComputeCommandEncoder> enc = (id<MTL4ComputeCommandEncoder>)encoder;
      id<MTLBuffer> buffer = (id<MTLBuffer>)data->buffer;
      id<MTLTexture> image = (id<MTLTexture>)data->image;
      uint32_t shape = 0;
      const bool logged = encoder_log_buffer_texture(data, false, &shape);
      [enc copyFromTexture:image
               sourceSlice:data->image_slice
               sourceLevel:data->image_level
              sourceOrigin:origin
                sourceSize:size
                  toBuffer:buffer
         destinationOffset:data->buffer_offset_B
    destinationBytesPerRow:data->buffer_stride_B
  destinationBytesPerImage:data->buffer_2d_image_size_B
                   options:(MTLBlitOption)data->options];
      if (logged)
         fprintf(stderr, "kk: encoder copy ready shape=%08x thread=%" PRIu64 "\n",
                 shape, encoder_thread_id());
   }
}

void
mtl_copy_from_texture_to_texture(mtl_compute_encoder *encoder,
                                 mtl_texture *src_tex_handle, size_t src_slice,
                                 size_t src_level, struct mtl_origin src_origin,
                                 struct mtl_size src_size,
                                 mtl_texture *dst_tex_handle, size_t dst_slice,
                                 size_t dst_level, struct mtl_origin dst_origin)
{
   @autoreleasepool {
      MTLOrigin mtl_src_origin = MTLOriginMake(src_origin.x, src_origin.y, src_origin.z);
      MTLSize mtl_src_size = MTLSizeMake(src_size.x, src_size.y, src_size.z);
      MTLOrigin mtl_dst_origin = MTLOriginMake(dst_origin.x, dst_origin.y, dst_origin.z);
      id<MTLTexture> mtl_dst_tex = (id<MTLTexture>)dst_tex_handle;
      id<MTL4ComputeCommandEncoder> enc = (id<MTL4ComputeCommandEncoder>)encoder;
      id<MTLTexture> mtl_src_tex = (id<MTLTexture>)src_tex_handle;
      [enc copyFromTexture:mtl_src_tex
               sourceSlice:src_slice
               sourceLevel:src_level
              sourceOrigin:mtl_src_origin
                sourceSize:mtl_src_size
                 toTexture:mtl_dst_tex
          destinationSlice:dst_slice
          destinationLevel:dst_level
         destinationOrigin:mtl_dst_origin];
   }
}

void
mtl_compute_set_pipeline_state(mtl_compute_encoder *encoder,
                               mtl_compute_pipeline_state *state_handle)
{
   @autoreleasepool {
      id<MTL4ComputeCommandEncoder> enc = (id<MTL4ComputeCommandEncoder>)encoder;
      id<MTLComputePipelineState> state = (id<MTLComputePipelineState>)state_handle;
      [enc setComputePipelineState:state];
   }
}

void
mtl_compute_set_argument_table(mtl_compute_encoder *encoder,
                               mtl_argument_table *table)
{
   @autoreleasepool {
      id<MTL4ComputeCommandEncoder> enc = (id<MTL4ComputeCommandEncoder>)encoder;
      id<MTL4ArgumentTable> t = (id<MTL4ArgumentTable>)table;
      [enc setArgumentTable:t];
   }
}

void
mtl_dispatch_threads(mtl_compute_encoder *encoder,
                     struct mtl_size grid_size, struct mtl_size local_size)
{
   @autoreleasepool {
      id<MTL4ComputeCommandEncoder> enc = (id<MTL4ComputeCommandEncoder>)encoder;
      MTLSize thread_count = MTLSizeMake(grid_size.x, grid_size.y, grid_size.z);
      MTLSize threads_per_threadgroup = MTLSizeMake(local_size.x, local_size.y,
                                                    local_size.z);

      [enc dispatchThreads:thread_count threadsPerThreadgroup:threads_per_threadgroup];
   }
}

void
mtl_dispatch_threadgroups_with_indirect_buffer(mtl_compute_encoder *encoder,
                                               uint64_t addr,
                                               struct mtl_size local_size)
{
   @autoreleasepool {
      id<MTL4ComputeCommandEncoder> enc = (id<MTL4ComputeCommandEncoder>)encoder;
      MTLSize threads_per_threadgroup = MTLSizeMake(local_size.x,
                                                    local_size.y,
                                                    local_size.z);

      [enc dispatchThreadgroupsWithIndirectBuffer:addr threadsPerThreadgroup:threads_per_threadgroup];
   }
}

void
mtl_dispatch_threads_with_indirect_buffer(mtl_compute_encoder *encoder,
                                          uint64_t addr)
{
   @autoreleasepool {
      id<MTL4ComputeCommandEncoder> enc = (id<MTL4ComputeCommandEncoder>)encoder;
      [enc dispatchThreadsWithIndirectBuffer:addr];
   }
}

/* MTLRenderEncoder */

/* Encoder commands */
mtl_render_encoder *
mtl_new_render_command_encoder_with_descriptor(
   mtl_command_buffer *command_buffer, mtl_render_pass_descriptor *descriptor)
{
   @autoreleasepool {
      id<MTL4CommandBuffer> cmd = (id<MTL4CommandBuffer>)command_buffer;
      MTL4RenderPassDescriptor *desc = (MTL4RenderPassDescriptor *)descriptor;
      uint32_t shape = 0;
      const bool logged = encoder_log_render(cmd, desc, &shape);
      id<MTL4RenderCommandEncoder> encoder = [cmd renderCommandEncoderWithDescriptor:desc];
      if (logged)
         fprintf(stderr, "kk: encoder render ready shape=%08x thread=%" PRIu64
                 " encoder=%p\n", shape, encoder_thread_id(), (void *)encoder);
      return [encoder retain];
   }
}

void
mtl_set_viewports(mtl_render_encoder *encoder, struct mtl_viewport *viewports,
                  uint32_t count)
{
   @autoreleasepool {
      id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
      MTLViewport *vps = (MTLViewport *)viewports;
      [enc setViewports:vps count:count];
   }
}

void
mtl_set_scissor_rects(mtl_render_encoder *encoder,
                      struct mtl_scissor_rect *scissor_rects, uint32_t count)
{
   @autoreleasepool {
      id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
      MTLScissorRect *rects = (MTLScissorRect *)scissor_rects;
      [enc setScissorRects:rects count:count];
   }
}

void
mtl_render_set_pipeline_state(mtl_render_encoder *encoder,
                              mtl_render_pipeline_state *pipeline)
{
   @autoreleasepool {
      id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
      id<MTLRenderPipelineState> pipe = (id<MTLRenderPipelineState>)pipeline;
      [enc setRenderPipelineState:pipe];
   }
}

void
mtl_set_depth_stencil_state(mtl_render_encoder *encoder,
                            mtl_depth_stencil_state *state)
{
   @autoreleasepool {
      id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
      id<MTLDepthStencilState> s = (id<MTLDepthStencilState>)state;
      [enc setDepthStencilState:s];
   }
}

void
mtl_set_stencil_references(mtl_render_encoder *encoder, uint32_t front,
                           uint32_t back)
{
   @autoreleasepool {
      id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
      [enc setStencilFrontReferenceValue:front backReferenceValue:back];
   }
}

void
mtl_set_front_face_winding(mtl_render_encoder *encoder,
                           enum mtl_winding winding)
{
   @autoreleasepool {
      id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
      [enc setFrontFacingWinding:(MTLWinding)winding];
   }
}

void
mtl_set_cull_mode(mtl_render_encoder *encoder, enum mtl_cull_mode mode)
{
   @autoreleasepool {
      id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
      [enc setCullMode:(MTLCullMode)mode];
   }
}

void
mtl_set_triangle_fill_mode(mtl_render_encoder *encoder,
                           enum mtl_triangle_fill_mode mode)
{
   @autoreleasepool {
      id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
      [enc setTriangleFillMode:(MTLTriangleFillMode)mode];
   }
}

void
mtl_set_visibility_result_mode(mtl_render_encoder *encoder,
                               enum mtl_visibility_result_mode mode,
                               size_t offset)
{
   @autoreleasepool {
      id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
      [enc setVisibilityResultMode:(MTLVisibilityResultMode)mode offset:offset];
   }
}

void
mtl_set_depth_bias(mtl_render_encoder *encoder, float depth_bias,
                   float slope_scale, float clamp)
{
   @autoreleasepool {
      id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
      [enc setDepthBias:depth_bias slopeScale:slope_scale clamp:clamp];
   }
}

void
mtl_set_depth_clip_mode(mtl_render_encoder *encoder,
                        enum mtl_depth_clip_mode mode)
{
   @autoreleasepool {
      id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
      [enc setDepthClipMode:(MTLDepthClipMode)mode];
   }
}

void
mtl_set_depth_test_bounds(mtl_render_encoder *encoder, float min, float max)
{
   @autoreleasepool {
      id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
      [enc setDepthTestMinBound:min maxBound:max];
   }
}

void
mtl_set_vertex_amplification_count(mtl_render_encoder *encoder,
                                   uint32_t *layer_ids, uint32_t id_count)
{
   @autoreleasepool {
      id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
      MTLVertexAmplificationViewMapping mappings[32];
      for (uint32_t i = 0u; i < id_count; ++i) {
         mappings[i].renderTargetArrayIndexOffset = layer_ids[i];
         mappings[i].viewportArrayIndexOffset = 0u;
      }
      [enc setVertexAmplificationCount:id_count viewMappings:mappings];
   }
}

void
mtl_render_set_argument_table(mtl_render_encoder *encoder,
                              mtl_argument_table *table,
                              enum mtl_render_stages stages)
{
   @autoreleasepool {
      id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
      id<MTL4ArgumentTable> t = (id<MTL4ArgumentTable>)table;
      [enc setArgumentTable:t atStages:(MTLRenderStages)stages];
   }
}

void
mtl_draw_primitives(mtl_render_encoder *encoder,
                    enum mtl_primitive_type primitve_type, uint32_t vertexStart,
                    uint32_t vertexCount, uint32_t instanceCount,
                    uint32_t baseInstance)
{
   @autoreleasepool {
      id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
      MTLPrimitiveType type = (MTLPrimitiveType)primitve_type;
      [enc drawPrimitives:type vertexStart:vertexStart vertexCount:vertexCount instanceCount:instanceCount baseInstance:baseInstance];
   }
}

void
mtl_draw_indexed_primitives(mtl_render_encoder *encoder,
                            enum mtl_primitive_type primitve_type,
                            uint32_t index_count,
                            enum mtl_index_type index_type, uint64_t index_addr,
                            uint64_t index_buffer_length,
                            uint32_t instance_count, int32_t base_vertex,
                            uint32_t base_instance)
{
   @autoreleasepool {
      id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
      MTLIndexType ndx_type = (MTLIndexType)index_type;
      MTLPrimitiveType primitive = (MTLPrimitiveType)primitve_type;
      [enc drawIndexedPrimitives:primitive
                      indexCount:index_count
                       indexType:ndx_type
                     indexBuffer:index_addr
               indexBufferLength:index_buffer_length
                   instanceCount:instance_count
                      baseVertex:base_vertex
                    baseInstance:base_instance];
   }
}

void
mtl_draw_primitives_indirect(mtl_render_encoder *encoder,
                             enum mtl_primitive_type primitve_type,
                             uint64_t addr)
{
   @autoreleasepool {
      id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
      [enc drawPrimitives:(MTLPrimitiveType)primitve_type indirectBuffer:addr];
   }
}

void
mtl_draw_indexed_primitives_indirect(mtl_render_encoder *encoder,
                                     enum mtl_primitive_type primitve_type,
                                     enum mtl_index_type index_type,
                                     uint64_t index_addr,
                                     uint64_t index_buffer_length,
                                     uint64_t addr)
{
   @autoreleasepool {
      id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
      MTLPrimitiveType type = (MTLPrimitiveType)primitve_type;
      MTLIndexType ndx_type = (MTLIndexType)index_type;
      [enc drawIndexedPrimitives:type
                       indexType:ndx_type
                     indexBuffer:index_addr
               indexBufferLength:index_buffer_length
                  indirectBuffer:addr];
   }
}

void
mtl_compute_write_timestamp(mtl_compute_encoder *encoder,
                            mtl_counter_heap *heap, uint32_t index)
{
   @autoreleasepool {
      id<MTL4ComputeCommandEncoder> enc =
         (id<MTL4ComputeCommandEncoder>)encoder;
      id<MTL4CounterHeap> h = (id<MTL4CounterHeap>)heap;
      [enc writeTimestampWithGranularity:MTL4TimestampGranularityPrecise
                                intoHeap:h
                                 atIndex:index];
   }
}

void
mtl_render_write_timestamp(mtl_render_encoder *encoder,
                           enum mtl_render_stages stage, mtl_counter_heap *heap,
                           uint32_t index)
{
   @autoreleasepool {
      id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
      id<MTL4CounterHeap> h = (id<MTL4CounterHeap>)heap;
      [enc writeTimestampWithGranularity:MTL4TimestampGranularityRelaxed
                              afterStage:(MTLRenderStages)stage
                                intoHeap:h
                                 atIndex:index];
   }
}


void
mtl_render_set_color_store_action(mtl_render_encoder *encoder,
                                  enum mtl_store_action action,
                                  uint32_t index)
{
    @autoreleasepool {
        id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
        [enc setColorStoreAction:(MTLStoreAction)action
                         atIndex:index];
    }
}

void mtl_render_set_depth_store_action(mtl_render_encoder *encoder,
                                       enum mtl_store_action action)
{
    @autoreleasepool {
        id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
        [enc setDepthStoreAction:(MTLStoreAction)action];
    }
}

void mtl_render_set_stencil_store_action(mtl_render_encoder *encoder,
                                       enum mtl_store_action action)
{
    @autoreleasepool {
        id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
        [enc setStencilStoreAction:(MTLStoreAction)action];
    }
}
