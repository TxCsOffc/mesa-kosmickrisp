/*
 * Copyright 2026 Highball
 * SPDX-License-Identifier: MIT
 */

#include "mtl_metalfx.h"

#include <Metal/Metal.h>
#include <MetalFX/MetalFX.h>
#include <os/lock.h>
#include <stdio.h>
#include <string.h>

#define MTL_METALFX_RETIRED_MAX 8

@interface HbMetalFXEntry : NSObject {
@public
   NSData *key;
   id scaler;
}
@end

@implementation HbMetalFXEntry
- (void)dealloc
{
   [key release];
   [scaler release];
   [super dealloc];
}
@end

struct mtl_metalfx_context {
   os_unfair_lock lock;
   NSMutableDictionary *scalers;
   NSMutableArray *retired;
};

void *
mtl_metalfx_context_create(void)
{
   struct mtl_metalfx_context *ctx = calloc(1, sizeof(*ctx));
   if (!ctx)
      return NULL;
   ctx->lock = OS_UNFAIR_LOCK_INIT;
   ctx->scalers = [[NSMutableDictionary alloc] init];
   ctx->retired = [[NSMutableArray alloc] init];
   return ctx;
}

void
mtl_metalfx_context_destroy(void *context)
{
   struct mtl_metalfx_context *ctx = context;
   if (!ctx)
      return;
   @autoreleasepool {
      [ctx->scalers release];
      [ctx->retired release];
   }
   free(ctx);
}

static void
retire_locked(struct mtl_metalfx_context *ctx, HbMetalFXEntry *entry)
{
   [ctx->retired addObject:entry];
   while (ctx->retired.count > MTL_METALFX_RETIRED_MAX)
      [ctx->retired removeObjectAtIndex:0];
}

void
mtl_metalfx_release(void *context, const uint64_t *ids, uint32_t count)
{
   struct mtl_metalfx_context *ctx = context;
   if (!ctx)
      return;
   @autoreleasepool {
      os_unfair_lock_lock(&ctx->lock);
      for (uint32_t i = 0; i < count; i++) {
         NSNumber *id_key = [NSNumber numberWithUnsignedLongLong:ids[i]];
         HbMetalFXEntry *entry = ctx->scalers[id_key];
         if (entry) {
            retire_locked(ctx, entry);
            [ctx->scalers removeObjectForKey:id_key];
         }
      }
      os_unfair_lock_unlock(&ctx->lock);
   }
}

static id
make_scaler(const struct mtl_metalfx_scale *s, id<MTLDevice> dev, id<MTL4Compiler> compiler,
            id<MTLTexture> *tex, const MTLPixelFormat *fmt, bool output_motion,
            bool jittered, id<MTLTexture> reactive, NSUInteger in_w, NSUInteger in_h)
{
   id<MTLTexture> color = tex[MTL_METALFX_COLOR], output = tex[MTL_METALFX_OUTPUT];
   id made = nil;

   if (s->type == MTL_METALFX_TEMPORAL) {
      MTLFXTemporalScalerDescriptor *desc = [[MTLFXTemporalScalerDescriptor new] autorelease];
      desc.colorTextureFormat = fmt[MTL_METALFX_COLOR];
      desc.depthTextureFormat = fmt[MTL_METALFX_DEPTH];
      desc.motionTextureFormat = fmt[MTL_METALFX_MOTION];
      desc.outputTextureFormat = fmt[MTL_METALFX_OUTPUT];
      desc.inputWidth = in_w;
      desc.inputHeight = in_h;
      desc.outputWidth = output.width;
      desc.outputHeight = output.height;
      desc.autoExposureEnabled = s->auto_exposure;
      desc.inputContentPropertiesEnabled = YES;
      float min_scale = MAX(1.0f, [MTLFXTemporalScalerDescriptor supportedInputContentMinScaleForDevice:dev]);
      float max_scale = MIN(3.0f, [MTLFXTemporalScalerDescriptor supportedInputContentMaxScaleForDevice:dev]);
      if (max_scale < min_scale)
         max_scale = min_scale;
      desc.inputContentMinScale = min_scale;
      desc.inputContentMaxScale = max_scale;
      if (@available(macOS 27.0, *)) {
         desc.outputResolutionMotionVectorsEnabled = output_motion;
         desc.jitteredMotionVectorsEnabled = jittered;
      }
      if (reactive) {
         desc.reactiveMaskTextureEnabled = YES;
         desc.reactiveMaskTextureFormat = fmt[MTL_METALFX_REACTIVE];
      }
      desc.requiresSynchronousInitialization = YES;
      made = [desc newTemporalScalerWithDevice:dev compiler:compiler];
   } else {
      MTLFXTemporalDenoisedScalerDescriptor *desc =
         [[MTLFXTemporalDenoisedScalerDescriptor new] autorelease];
      desc.colorTextureFormat = fmt[MTL_METALFX_COLOR];
      desc.depthTextureFormat = fmt[MTL_METALFX_DEPTH];
      desc.motionTextureFormat = fmt[MTL_METALFX_MOTION];
      desc.diffuseAlbedoTextureFormat = fmt[MTL_METALFX_DIFFUSE_ALBEDO];
      desc.specularAlbedoTextureFormat = fmt[MTL_METALFX_SPECULAR_ALBEDO];
      desc.normalTextureFormat = fmt[MTL_METALFX_NORMALS];
      desc.roughnessTextureFormat = fmt[MTL_METALFX_ROUGHNESS];
      desc.outputTextureFormat = fmt[MTL_METALFX_OUTPUT];
      if (fmt[MTL_METALFX_SPECULAR_HIT_DISTANCE] != MTLPixelFormatInvalid) {
         desc.specularHitDistanceTextureEnabled = YES;
         desc.specularHitDistanceTextureFormat = fmt[MTL_METALFX_SPECULAR_HIT_DISTANCE];
      }
      desc.inputWidth = in_w;
      desc.inputHeight = in_h;
      desc.outputWidth = output.width;
      desc.outputHeight = output.height;
      desc.autoExposureEnabled = s->auto_exposure;
      desc.requiresSynchronousInitialization = YES;
      made = [desc newTemporalDenoisedScalerWithDevice:dev compiler:compiler];
   }

   fprintf(stderr, "kk: MetalFX %s %lux%lu -> %lux%lu (color %lu depth %lu motion %lu output %lu)%s%s%s: %s\n",
           s->type == MTL_METALFX_TEMPORAL ? "temporal" : "denoised",
           (unsigned long)in_w, (unsigned long)in_h,
           (unsigned long)output.width, (unsigned long)output.height,
           (unsigned long)fmt[MTL_METALFX_COLOR], (unsigned long)fmt[MTL_METALFX_DEPTH],
           (unsigned long)fmt[MTL_METALFX_MOTION], (unsigned long)fmt[MTL_METALFX_OUTPUT],
           output_motion ? ", output-resolution motion" : "", jittered ? ", jittered motion" : "",
           reactive ? ", reactive mask" : "", made ? "ok" : "FAILED");
   return made;
}

bool
mtl_metalfx_encode(void *context, mtl_device *device, mtl_compiler *compiler,
                   mtl_command_buffer *command_buffer, const struct mtl_metalfx_scale *s)
{
   struct mtl_metalfx_context *ctx = context;
   if (!ctx || (s->type != MTL_METALFX_TEMPORAL && s->type != MTL_METALFX_DENOISED))
      return false;

   if (@available(macOS 26.0, *)) {
   } else {
      return false;
   }

   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;
      id<MTLTexture> tex[MTL_METALFX_IMAGE_COUNT];
      MTLPixelFormat fmt[MTL_METALFX_IMAGE_COUNT];
      for (uint32_t i = 0; i < MTL_METALFX_IMAGE_COUNT; i++) {
         id<MTLTexture> t = (id<MTLTexture>)s->textures[i];
         fmt[i] = t ? t.pixelFormat : MTLPixelFormatInvalid;
         if (t && s->formats[i] && s->formats[i] != (uint32_t)t.pixelFormat) {
            id<MTLTexture> view = [t newTextureViewWithPixelFormat:(MTLPixelFormat)s->formats[i]];
            if (view) {
               t = [view autorelease];
               fmt[i] = t.pixelFormat;
            }
         }
         tex[i] = t;
      }

      id<MTLTexture> color = tex[MTL_METALFX_COLOR], output = tex[MTL_METALFX_OUTPUT];
      id<MTLTexture> depth = tex[MTL_METALFX_DEPTH], motion = tex[MTL_METALFX_MOTION];
      if (!color || !output || !depth || !motion)
         return false;
      const bool denoised = s->type == MTL_METALFX_DENOISED;
      if (denoised && (!tex[MTL_METALFX_DIFFUSE_ALBEDO] || !tex[MTL_METALFX_SPECULAR_ALBEDO] ||
                       !tex[MTL_METALFX_NORMALS] || !tex[MTL_METALFX_ROUGHNESS]))
         return false;

      bool output_motion = false, jittered = false;
      if (@available(macOS 27.0, *)) {
         output_motion = !denoised && s->motion_display_res &&
                         (motion.width != color.width || motion.height != color.height);
         jittered = !denoised && s->jittered_motion;
      }
      id<MTLTexture> reactive = denoised ? nil : tex[MTL_METALFX_REACTIVE];

      NSUInteger in_w = s->input_content_width && s->input_content_width < color.width ? s->input_content_width : color.width;
      NSUInteger in_h = s->input_content_height && s->input_content_height < color.height ? s->input_content_height : color.height;
      uint32_t key_words[8 + MTL_METALFX_IMAGE_COUNT] = {
         s->type, (uint32_t)in_w, (uint32_t)in_h, (uint32_t)output.width,
         (uint32_t)output.height, s->auto_exposure | (output_motion << 1) | (jittered << 2),
         reactive != nil, 0,
      };
      for (uint32_t i = 0; i < MTL_METALFX_IMAGE_COUNT; i++)
         key_words[8 + i] = i == MTL_METALFX_EXPOSURE ? 0 : (uint32_t)fmt[i];
      NSData *key = [NSData dataWithBytes:key_words length:sizeof(key_words)];
      NSNumber *id_key = [NSNumber numberWithUnsignedLongLong:s->scaler_id];

      id scaler = nil;
      os_unfair_lock_lock(&ctx->lock);
      HbMetalFXEntry *entry = ctx->scalers[id_key];
      if (entry && [entry->key isEqualToData:key])
         scaler = [[entry->scaler retain] autorelease];
      os_unfair_lock_unlock(&ctx->lock);

      if (!scaler) {
         id made = make_scaler(s, dev, (id<MTL4Compiler>)compiler, tex, fmt, output_motion,
                               jittered, reactive, in_w, in_h);
         if (!made)
            return false;
         HbMetalFXEntry *fresh = [[HbMetalFXEntry new] autorelease];
         fresh->key = [key retain];
         fresh->scaler = made;
         os_unfair_lock_lock(&ctx->lock);
         HbMetalFXEntry *old = ctx->scalers[id_key];
         if (old)
            retire_locked(ctx, old);
         ctx->scalers[id_key] = fresh;
         os_unfair_lock_unlock(&ctx->lock);
         scaler = [[made retain] autorelease];
      }

      id<MTL4CommandBuffer> cb = (id<MTL4CommandBuffer>)command_buffer;
      const float pre_exposure = s->pre_exposure > 0.0f ? s->pre_exposure : 1.0f;
      if (!denoised) {
         id<MTL4FXTemporalScaler> t = (id<MTL4FXTemporalScaler>)scaler;
         t.colorTexture = color;
         t.depthTexture = depth;
         t.motionTexture = motion;
         t.outputTexture = output;
         t.exposureTexture = tex[MTL_METALFX_EXPOSURE];
         if (reactive)
            t.reactiveMaskTexture = reactive;
         t.inputContentWidth = in_w;
         t.inputContentHeight = in_h;
         t.reset = s->reset;
         t.depthReversed = s->depth_reversed;
         t.motionVectorScaleX = s->motion_scale_x;
         t.motionVectorScaleY = s->motion_scale_y;
         t.jitterOffsetX = s->jitter_x;
         t.jitterOffsetY = s->jitter_y;
         t.preExposure = pre_exposure;
         if (@available(macOS 27.0, *)) {
            if (s->has_offsets) {
               t.colorContentOffsetX = s->offset_x[MTL_METALFX_COLOR];
               t.colorContentOffsetY = s->offset_y[MTL_METALFX_COLOR];
               t.depthContentOffsetX = s->offset_x[MTL_METALFX_DEPTH];
               t.depthContentOffsetY = s->offset_y[MTL_METALFX_DEPTH];
               t.motionContentOffsetX = s->offset_x[MTL_METALFX_MOTION];
               t.motionContentOffsetY = s->offset_y[MTL_METALFX_MOTION];
               if (reactive) {
                  t.reactiveMaskContentOffsetX = s->offset_x[MTL_METALFX_REACTIVE];
                  t.reactiveMaskContentOffsetY = s->offset_y[MTL_METALFX_REACTIVE];
               }
            }
         }
         [t encodeToCommandBuffer:cb];
      } else {
         id<MTL4FXTemporalDenoisedScaler> d = (id<MTL4FXTemporalDenoisedScaler>)scaler;
         simd_float4x4 world_to_view, view_to_clip;
         memcpy(&world_to_view, s->world_to_view, sizeof(world_to_view));
         memcpy(&view_to_clip, s->view_to_clip, sizeof(view_to_clip));
         d.colorTexture = color;
         d.depthTexture = depth;
         d.motionTexture = motion;
         d.diffuseAlbedoTexture = tex[MTL_METALFX_DIFFUSE_ALBEDO];
         d.specularAlbedoTexture = tex[MTL_METALFX_SPECULAR_ALBEDO];
         d.normalTexture = tex[MTL_METALFX_NORMALS];
         d.roughnessTexture = tex[MTL_METALFX_ROUGHNESS];
         d.specularHitDistanceTexture = tex[MTL_METALFX_SPECULAR_HIT_DISTANCE];
         d.exposureTexture = tex[MTL_METALFX_EXPOSURE];
         d.outputTexture = output;
         d.preExposure = pre_exposure;
         d.jitterOffsetX = s->jitter_x;
         d.jitterOffsetY = s->jitter_y;
         d.motionVectorScaleX = s->motion_scale_x;
         d.motionVectorScaleY = s->motion_scale_y;
         d.shouldResetHistory = s->reset;
         d.depthReversed = s->depth_reversed;
         d.worldToViewMatrix = world_to_view;
         d.viewToClipMatrix = view_to_clip;
         [d encodeToCommandBuffer:cb];
      }
      return true;
   }
}
