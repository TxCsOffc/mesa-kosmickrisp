/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "kk_buffer_view.h"

#include <inttypes.h>

#include "kk_buffer.h"
#include "kk_descriptor_types.h"
#include "kk_device.h"
#include "kk_entrypoints.h"
#include "kk_format.h"
#include "kk_image_layout.h"
#include "kk_nir_lower_vbo.h"
#include "kk_physical_device.h"
#include "kk_sparse.h"

#include "kosmickrisp/bridge/mtl_bridge.h"
#include "kosmickrisp/bridge/mtl_format.h"

#include "util/log.h"
#include "vk_enum_to_str.h"
#include "vk_format.h"

#include <stdatomic.h>

/* Metal has no 3-component pixel formats: uniform texel buffers of
 * R32G32B32 formats are R32 texture buffers of three texels per element
 * (struct kk_texel_buffer_descriptor, KK_TEXEL_BUFFER_RGB32). vkd3d-proton
 * views structured buffers of 12-byte strides as R32G32B32_UINT texel
 * buffers. Returns the R32 format, PIPE_FORMAT_NONE for other formats. */
static enum pipe_format
kk_texel_buffer_rgb32_component(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_R32G32B32_UINT:
      return PIPE_FORMAT_R32_UINT;
   case PIPE_FORMAT_R32G32B32_SINT:
      return PIPE_FORMAT_R32_SINT;
   case PIPE_FORMAT_R32G32B32_FLOAT:
      return PIPE_FORMAT_R32_FLOAT;
   default:
      return PIPE_FORMAT_NONE;
   }
}

VkFormatFeatureFlags2
kk_get_buffer_format_features(struct kk_physical_device *pdev,
                              VkFormat vk_format)
{
   VkFormatFeatureFlags2 features = 0;
   enum pipe_format p_format = vk_format_to_pipe_format(vk_format);

   if (p_format == PIPE_FORMAT_NONE)
      return 0;

   const struct kk_va_format *format = kk_get_va_format(p_format);
   if (format) {
      if (format->texel_buffer.read)
         features |= VK_FORMAT_FEATURE_2_UNIFORM_TEXEL_BUFFER_BIT;

      if (format->texel_buffer.write) {
         features |= VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_BIT;
         features |= VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT;
         features |= VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT;
      }

      /* Only these formats allow atomics for texel buffers */
      if (vk_format == VK_FORMAT_R32_UINT || vk_format == VK_FORMAT_R32_SINT)
         features |= VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_ATOMIC_BIT;
   }

   if (kk_texel_buffer_rgb32_component(p_format) != PIPE_FORMAT_NONE)
      features |= VK_FORMAT_FEATURE_2_UNIFORM_TEXEL_BUFFER_BIT;

   if (kk_vbo_supports_format(p_format))
      features |= VK_FORMAT_FEATURE_2_VERTEX_BUFFER_BIT;

   return features;
}

/* A view no Metal texture buffer can back: vkd3d-proton creates typed views
 * without checking the format's features (and Venus forwards them), so one
 * reached the driver and dereferenced the missing format (STEAMAC-2Z). The view
 * gets no texture; its descriptors read as null descriptors (zero). */
static void
kk_buffer_view_warn_unsupported(VkFormat format, bool sparse)
{
   static atomic_uint warned[8];
   const unsigned bit = 1u << (format & 31);
   if ((uint32_t)format < 256 &&
       (atomic_fetch_or(&warned[format >> 5], bit) & bit))
      return;
   mesa_logw("kk: texel buffer views of %s%s are not supported; they read "
             "as null descriptors",
             vk_Format_to_str(format), sparse ? " in sparse buffers" : "");
}

/* Once per process each: the guest's own usage error, not a driver fault */
static void
kk_buffer_view_warn_zero_elements(VkFormat format)
{
   static atomic_bool warned;
   if (atomic_exchange(&warned, true))
      return;
   mesa_logw("kk: a texel buffer view of %s has zero elements; it reads as a "
             "null descriptor",
             vk_Format_to_str(format));
}

static void
kk_buffer_view_warn_oversized(VkFormat format, uint64_t elements, uint64_t limit)
{
   static atomic_bool warned;
   if (atomic_exchange(&warned, true))
      return;
   mesa_logw("kk: a texel buffer view of %s has %" PRIu64 " elements, past "
             "Metal's limit for this view (%" PRIu64 "); it is narrowed",
             vk_Format_to_str(format), elements, limit);
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_CreateBufferView(VkDevice _device, const VkBufferViewCreateInfo *pCreateInfo,
                    const VkAllocationCallbacks *pAllocator,
                    VkBufferView *pBufferView)
{
   VK_FROM_HANDLE(kk_device, dev, _device);
   struct kk_buffer_view *view =
      vk_buffer_view_create(&dev->vk, pCreateInfo, pAllocator, sizeof(*view));
   if (!view)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   struct kk_buffer *buffer =
      container_of(view->vk.buffer, struct kk_buffer, vk);
   enum pipe_format p_format = vk_format_to_pipe_format(view->vk.format);
   const enum pipe_format rgb32_component =
      kk_texel_buffer_rgb32_component(p_format);
   const bool rgb32 = rgb32_component != PIPE_FORMAT_NONE;
   if (rgb32)
      p_format = rgb32_component;
   const struct kk_va_format *supported_format = kk_get_va_format(p_format);

   if (!supported_format || !supported_format->texel_buffer.read ||
       (rgb32 && buffer->sparse)) {
      kk_buffer_view_warn_unsupported(view->vk.format, buffer->sparse);
      *pBufferView = kk_buffer_view_to_handle(view);
      return VK_SUCCESS;
   }

   /* VK_WHOLE_SIZE rounds down to complete texels; it can produce zero
    * elements. The guest may also exceed maxTexelBufferElements (including
    * with VK_WHOLE_SIZE, VUID 04059; vkd3d-proton makes typed views over a
    * buffer's whole range without clamping). An unrepresentable width must
    * not reach either Metal constructor, which aborts, and must not fail
    * here: under Venus vkCreateBufferView is asynchronous
    * (vn_async_vkCreateBufferView), so the guest never learns of an error and
    * the first descriptor update that names the view is a command-stream
    * error that kills the guest process (Dragon's Dogma 2 at the end of
    * loading a save). A view of zero elements is a null descriptor, as an
    * unsupported format is; a view past Metal's capacity for its format and
    * alignment prefix is narrowed, and texels past it read out of bounds. */
   if (view->vk.elements == 0) {
      kk_buffer_view_warn_zero_elements(view->vk.format);
      *pBufferView = kk_buffer_view_to_handle(view);
      return VK_SUCCESS;
   }

   enum mtl_texture_usage usage = MTL_TEXTURE_USAGE_SHADER_READ;
   if (supported_format->texel_buffer.write && !rgb32)
      usage |= MTL_TEXTURE_USAGE_SHADER_WRITE;

   /* Only these formats allow atomics for texel buffers */
   if (view->vk.format == VK_FORMAT_R32_UINT ||
       view->vk.format == VK_FORMAT_R32_SINT)
      usage |= MTL_TEXTURE_USAGE_SHADER_ATOMIC;

   /* Metal needs texture buffers at KK_MIN_TEXEL_BUFFER_ALIGNMENT; views may
    * start at any texel. Start the texture at the aligned offset below the
    * view, texel_offset texels earlier (shaders add it to coordinates).
    * R32G32B32 views count R32 texels.
    */
   const uint64_t mtl_offset = kk_buffer_mtl_offset(buffer, view->vk.offset);
   uint64_t texture_offset =
      mtl_offset & ~(uint64_t)(KK_MIN_TEXEL_BUFFER_ALIGNMENT - 1);
   const uint32_t texel_B = util_format_get_blocksize(p_format);
   /* Views of sparse buffers start at the page below the view (kk_sparse.c) */
   if (buffer->sparse)
      texture_offset = mtl_offset & ~(uint64_t)(KK_SPARSE_PAGE_SIZE_B - 1);
   if ((mtl_offset - texture_offset) % texel_B != 0) {
      if (buffer->sparse) {
         vk_buffer_view_destroy(&dev->vk, pAllocator, &view->vk);
         return vk_errorf(dev, VK_ERROR_FORMAT_NOT_SUPPORTED,
                          "%u-byte texels across sparse pages", texel_B);
      }
      texture_offset = mtl_offset;
   }
   view->texel_offset = (mtl_offset - texture_offset) / texel_B;
   const uint64_t per_element = rgb32 ? 3u : 1u;
   const uint64_t max_elements =
      (KK_MTL_MAX_TEXEL_BUFFER_ELEMENTS - view->texel_offset) / per_element;
   const uint64_t elements = MIN2(view->vk.elements, max_elements);
   if (elements < view->vk.elements)
      kk_buffer_view_warn_oversized(view->vk.format, view->vk.elements,
                                    max_elements);
   const uint64_t width_px = elements * per_element + view->texel_offset;
   assert(width_px <= KK_MTL_MAX_TEXEL_BUFFER_ELEMENTS);
   if (rgb32)
      view->texel_offset |= KK_TEXEL_BUFFER_RGB32;

   struct kk_image_layout layout = {
      .width_px = width_px,
      .height_px = 1u,
      .depth_px = 1u,
      .layers = 1u,
      .type = MTL_TEXTURE_TYPE_TEXTURE_BUFFER,
      .sample_count_sa = 1u,
      .levels = 1u,
      .linear = true,
      .optimized_layout = false,
      .usage = usage,
      .format = {.pipe = p_format, .mtl = supported_format->mtl_pixel_format},
      .swizzle =
         {
            .red = supported_format->unswizzle.red,
            .green = supported_format->unswizzle.green,
            .blue = supported_format->unswizzle.blue,
            .alpha = supported_format->unswizzle.alpha,
         },
      .linear_stride_B = width_px * texel_B,
   };
   if (buffer->sparse) {
      VkResult result = kk_buffer_view_sparse_init(
         dev, view, buffer, texture_offset / KK_SPARSE_PAGE_SIZE_B, &layout);
      if (result != VK_SUCCESS) {
         if (view->mtl_texel_buffer_handle)
            mtl_release(view->mtl_texel_buffer_handle);
         vk_buffer_view_destroy(&dev->vk, pAllocator, &view->vk);
         return result;
      }
   } else {
      view->mtl_texel_buffer_handle = mtl_new_texture_with_descriptor_linear(
         buffer->metal.handle, &layout, texture_offset);
      if (!view->mtl_texel_buffer_handle) {
         vk_buffer_view_destroy(&dev->vk, pAllocator, &view->vk);
         return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      }
      kk_device_add_texture_to_residency_set(dev, view->mtl_texel_buffer_handle);
   }
   view->texel_buffer_gpu_id =
      mtl_texture_get_gpu_resource_id(view->mtl_texel_buffer_handle);

   *pBufferView = kk_buffer_view_to_handle(view);

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
kk_DestroyBufferView(VkDevice _device, VkBufferView bufferView,
                     const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(kk_device, dev, _device);
   VK_FROM_HANDLE(kk_buffer_view, view, bufferView);

   if (!view)
      return;

   if (view->sparse)
      kk_buffer_view_sparse_finish(dev, view);
   else if (view->mtl_texel_buffer_handle)
      kk_device_remove_texture_from_residency_set(
         dev, view->mtl_texel_buffer_handle);
   if (view->mtl_texel_buffer_handle)
      mtl_release(view->mtl_texel_buffer_handle);
   vk_buffer_view_destroy(&dev->vk, pAllocator, &view->vk);
}

void
kk_buffer_view_set_label(struct kk_buffer_view *bview, const char *label)
{
   if (bview->mtl_texel_buffer_handle)
      mtl_resource_set_label(bview->mtl_texel_buffer_handle, label);
}
