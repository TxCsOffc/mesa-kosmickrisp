/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "kk_image.h"

#include "kk_device.h"
#include "kk_device_memory.h"
#include "kk_entrypoints.h"
#include "kk_format.h"
#include "kk_physical_device.h"
#include "kk_sparse.h"

#include "kosmickrisp/bridge/mtl_bridge.h"
#include "kosmickrisp/bridge/vk_to_mtl_map.h"

#include "util/u_atomic.h"

#include "vk_enum_defines.h"
#include "vk_enum_to_str.h"
#include "vk_format.h"
#include "wsi_common_private.h"

static VkFormatFeatureFlags2
kk_get_image_plane_format_features(struct kk_physical_device *pdev,
                                   VkFormat vk_format, VkImageTiling tiling,
                                   uint64_t drm_format_mod)
{
   VkFormatFeatureFlags2 features = 0;

   /* Metal does not support linear tiling for compressed formats */
   if (tiling == VK_IMAGE_TILING_LINEAR && vk_format_is_compressed(vk_format))
      return 0;

   enum pipe_format p_format = vk_format_to_pipe_format(vk_format);
   if (p_format == PIPE_FORMAT_NONE)
      return 0;

   /* You can't tile a non-power-of-two */
   if (!util_is_power_of_two_nonzero(util_format_get_blocksize(p_format)))
      return 0;

   const struct kk_va_format *va_format = kk_get_va_format(p_format);
   if (va_format == NULL)
      return 0;

   /* R64 formats are R32G32_UINT to Metal (kk_format.c) and exist only for
    * VK_EXT_shader_image_atomic_int64's 64-bit storage-image atomics: expose
    * storage (atomic) and transfer, but not sampling, filtering, color
    * attachment or blending. Linear tiling stays unsupported, and so is the
    * format as a whole where the atomics are not advertised (before Apple8,
    * or with MESA_KK_DEBUG=noimageint64atomics), so an application that checks
    * format support never creates an R64 image there. */
   if (p_format == PIPE_FORMAT_R64_UINT || p_format == PIPE_FORMAT_R64_SINT) {
      if (tiling != VK_IMAGE_TILING_OPTIMAL ||
          !pdev->vk.supported_extensions.EXT_shader_image_atomic_int64)
         return 0;

      return VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT |
             VK_FORMAT_FEATURE_2_STORAGE_IMAGE_ATOMIC_BIT |
             VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT |
             VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT;
   }

   // Textures can at least be sampled
   features |= VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT;
   features |= VK_FORMAT_FEATURE_2_BLIT_SRC_BIT;

   if (va_format->filter) {
      features |= VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
   }

   /* Software reduction only needs point sampling. In particular R32_SFLOAT
    * and depth formats need not support linear interpolation. Integer and
    * stencil-only views have no floating-point filtering operation. */
   if (pdev->vk.supported_extensions.EXT_sampler_filter_minmax &&
       !util_format_is_pure_integer(p_format) &&
       (vk_format_has_depth(vk_format) || !vk_format_has_stencil(vk_format)))
      features |= VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_MINMAX_BIT;

   /* TODO: VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT */
   if (vk_format_has_depth(vk_format)) {
      features |= VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT;
   }

   /* We disable A8 format due to lower blend pass issues. Linear (buffer
    * backed) textures can be render targets in Metal too: dma-buf sharing in
    * virtual machines (virglrenderer on macOS) needs renderable linear images.
    */
   if (va_format->color && vk_format != VK_FORMAT_A8_UNORM) {
      features |= VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT;
      features |= VK_FORMAT_FEATURE_2_BLIT_DST_BIT;
      // TODO_KOSMICKRISP Support snorm formats once the following spec issue is
      // resolved: https://gitlab.khronos.org/vulkan/vulkan/-/issues/4293
      if (!vk_format_is_snorm(vk_format))
         features |= VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BLEND_BIT;
   }

   if (vk_format_is_depth_or_stencil(vk_format)) {
      if (tiling == VK_IMAGE_TILING_LINEAR)
         return 0;

      features |= VK_FORMAT_FEATURE_2_DEPTH_STENCIL_ATTACHMENT_BIT;
   }

   if (va_format->write) {
      features |= VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT;
      features |= VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT;
      features |= VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT;
   }

   if (va_format->atomic)
      features |= VK_FORMAT_FEATURE_2_STORAGE_IMAGE_ATOMIC_BIT;

   if (features != 0) {
      features |= VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT;
      features |= VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT;

      /* Metal does not allow CPU access to combined depth-stencil */
      if (!util_format_is_depth_and_stencil(p_format))
         features |= VK_FORMAT_FEATURE_2_HOST_IMAGE_TRANSFER_BIT;
   }

   return features;
}

VkFormatFeatureFlags2
kk_get_image_format_features(struct kk_physical_device *pdev,
                             VkFormat vk_format, VkImageTiling tiling,
                             uint64_t drm_format_mod)
{
   const struct vk_format_ycbcr_info *ycbcr_info =
      vk_format_get_ycbcr_info(vk_format);
   if (ycbcr_info == NULL) {
      return kk_get_image_plane_format_features(pdev, vk_format, tiling,
                                                drm_format_mod);
   }

   /* For multi-plane, we get the feature flags of each plane separately,
    * then take their intersection as the overall format feature flags
    */
   VkFormatFeatureFlags2 features = ~0ull;
   bool cosited_chroma = false;
   for (uint8_t plane = 0; plane < ycbcr_info->n_planes; plane++) {
      const struct vk_format_ycbcr_plane *plane_info =
         &ycbcr_info->planes[plane];
      features &= kk_get_image_plane_format_features(pdev, plane_info->format,
                                                     tiling, drm_format_mod);
      if (plane_info->denominator_scales[0] > 1 ||
          plane_info->denominator_scales[1] > 1)
         cosited_chroma = true;
   }
   if (features == 0)
      return 0;

   /* Uh... We really should be able to sample from YCbCr */
   assert(features & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT);
   assert(features & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_LINEAR_BIT);

   /* These aren't allowed for YCbCr formats */
   features &=
      ~(VK_FORMAT_FEATURE_2_BLIT_SRC_BIT | VK_FORMAT_FEATURE_2_BLIT_DST_BIT |
        VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT |
        VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BLEND_BIT |
        VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT);

   /* This is supported on all YCbCr formats */
   features |=
      VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER_BIT;

   if (ycbcr_info->n_planes > 1) {
      /* DISJOINT_BIT implies that each plane has its own separate binding,
       * while SEPARATE_RECONSTRUCTION_FILTER_BIT implies that luma and chroma
       * each have their own, separate filters, so these two bits make sense
       * for multi-planar formats only.
       *
       * For MIDPOINT_CHROMA_SAMPLES_BIT, NVIDIA HW on single-plane interleaved
       * YCbCr defaults to COSITED_EVEN, which is inaccurate and fails tests.
       * This can be fixed with a NIR tweak but for now, we only enable this bit
       * for multi-plane formats. See Issue #9525 on the mesa/main tracker.
       */
      features |=
         VK_FORMAT_FEATURE_DISJOINT_BIT |
         VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER_BIT |
         VK_FORMAT_FEATURE_2_MIDPOINT_CHROMA_SAMPLES_BIT;
   }

   if (cosited_chroma)
      features |= VK_FORMAT_FEATURE_COSITED_CHROMA_SAMPLES_BIT;

   return features;
}

static VkFormatFeatureFlags2
vk_image_usage_to_format_features(VkImageUsageFlagBits usage_flag)
{
   assert(util_bitcount(usage_flag) == 1);
   switch (usage_flag) {
   case VK_IMAGE_USAGE_TRANSFER_SRC_BIT:
      return VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT |
             VK_FORMAT_FEATURE_BLIT_SRC_BIT;
   case VK_IMAGE_USAGE_TRANSFER_DST_BIT:
      return VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT |
             VK_FORMAT_FEATURE_BLIT_DST_BIT;
   case VK_IMAGE_USAGE_SAMPLED_BIT:
      return VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT;
   case VK_IMAGE_USAGE_STORAGE_BIT:
      return VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT;
   case VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT:
      return VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT;
   case VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT:
      return VK_FORMAT_FEATURE_2_DEPTH_STENCIL_ATTACHMENT_BIT;
   case VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT:
      return VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT |
             VK_FORMAT_FEATURE_2_DEPTH_STENCIL_ATTACHMENT_BIT;
   case VK_IMAGE_USAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR:
      return VK_FORMAT_FEATURE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR;
   default:
      return 0;
   }
}

uint32_t
kk_image_max_dimension(const struct kk_physical_device *pdev,
                       VkImageType image_type)
{
   /* Base values taken from Apple7
    * https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf */
   switch (image_type) {
   case VK_IMAGE_TYPE_1D:
   case VK_IMAGE_TYPE_2D:
      /* Apple10 and later raise the 1D/2D/cube limit to 32768. */
      return pdev->info.gpu_apple_family >= 10 ? 32768 : 16384;
   case VK_IMAGE_TYPE_3D:
      return 2048;
   default:
      UNREACHABLE("Invalid image type");
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_GetPhysicalDeviceImageFormatProperties2(
   VkPhysicalDevice physicalDevice,
   const VkPhysicalDeviceImageFormatInfo2 *pImageFormatInfo,
   VkImageFormatProperties2 *pImageFormatProperties)
{
   VK_FROM_HANDLE(kk_physical_device, pdev, physicalDevice);

   const VkPhysicalDeviceExternalImageFormatInfo *external_info =
      vk_find_struct_const(pImageFormatInfo->pNext,
                           PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO);
   /* Host imports must keep their contents in the imported pages. Metal
    * cannot place a tiled texture there; a private heap would silently break
    * sharing between devices. Use the same bounded linear layout at query and
    * creation time, only for explicitly external host-memory images.
    * Temporary steamac workaround for mesa!44929 / issue #58. */
   const VkExternalMemoryHandleTypeFlags host_handles =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT |
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_MAPPED_FOREIGN_MEMORY_BIT_EXT;
   VkPhysicalDeviceImageFormatInfo2 linear_info;
   if (external_info && (external_info->handleType & host_handles) &&
       pImageFormatInfo->tiling == VK_IMAGE_TILING_OPTIMAL) {
      linear_info = *pImageFormatInfo;
      linear_info.tiling = VK_IMAGE_TILING_LINEAR;
      pImageFormatInfo = &linear_info;
   }

   /* Initialize to zero in case we return VK_ERROR_FORMAT_NOT_SUPPORTED */
   memset(&pImageFormatProperties->imageFormatProperties, 0,
          sizeof(pImageFormatProperties->imageFormatProperties));

   /* Metal does not support depth/stencil textures that are not 2D (we make 1D
    * textures 2D) */
   if (vk_format_is_depth_or_stencil(pImageFormatInfo->format) &&
       pImageFormatInfo->type == VK_IMAGE_TYPE_3D)
      return VK_ERROR_FORMAT_NOT_SUPPORTED;

   const enum pipe_format p_format =
      vk_format_to_pipe_format(pImageFormatInfo->format);

   /* Metal does not support EAC/ETC formats for 3D textures. */
   if (util_format_is_etc(p_format) &&
       pImageFormatInfo->type == VK_IMAGE_TYPE_3D)
      return VK_ERROR_FORMAT_NOT_SUPPORTED;

   const struct vk_format_ycbcr_info *ycbcr_info =
      vk_format_get_ycbcr_info(pImageFormatInfo->format);

   /* For the purposes of these checks, we don't care about all the extra
    * YCbCr features and we just want the accumulation of features available
    * to all planes of the given format.
    */
   VkFormatFeatureFlags2 features;
   if (ycbcr_info == NULL) {
      features = kk_get_image_plane_format_features(
         pdev, pImageFormatInfo->format, pImageFormatInfo->tiling, 0u);
   } else {
      features = ~0ull;
      assert(ycbcr_info->n_planes > 0);
      for (uint8_t plane = 0; plane < ycbcr_info->n_planes; plane++) {
         const VkFormat plane_format = ycbcr_info->planes[plane].format;
         features &= kk_get_image_plane_format_features(
            pdev, plane_format, pImageFormatInfo->tiling, 0u);
      }
   }

   if (features == 0)
      return VK_ERROR_FORMAT_NOT_SUPPORTED;

   if (pImageFormatInfo->tiling == VK_IMAGE_TILING_LINEAR &&
       pImageFormatInfo->type == VK_IMAGE_TYPE_3D)
      return VK_ERROR_FORMAT_NOT_SUPPORTED;

   /* Linear (buffer-backed) textures are 2D only, input attachments are 2D
    * arrays: a linear plane gets a 2D texture and a 2D array view of it (see
    * kk_image_plane_create_texture), so linear images can be input attachments.
    * Depth/stencil formats have no linear support in Metal.
    */
   if (pImageFormatInfo->tiling == VK_IMAGE_TILING_LINEAR &&
       (pImageFormatInfo->usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;

   if (ycbcr_info && pImageFormatInfo->type != VK_IMAGE_TYPE_2D)
      return VK_ERROR_FORMAT_NOT_SUPPORTED;

   /* Sparse images are Metal placement sparse textures with standard block
    * shapes (kk_sparse.c) */
   const bool sparse =
      pImageFormatInfo->flags & (VK_IMAGE_CREATE_SPARSE_BINDING_BIT |
                                 VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT |
                                 VK_IMAGE_CREATE_SPARSE_ALIASED_BIT);
   if (sparse && !kk_sparse_image_supported(
                    pdev, pImageFormatInfo->format, pImageFormatInfo->type,
                    VK_SAMPLE_COUNT_1_BIT, pImageFormatInfo->tiling,
                    pImageFormatInfo->flags, NULL))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;

   /* From the Vulkan 1.3.279 spec:
    *
    *    VUID-VkImageCreateInfo-tiling-04121
    *
    *    "If tiling is VK_IMAGE_TILING_LINEAR, flags must not contain
    *    VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT"
    *
    *    VUID-VkImageCreateInfo-imageType-00970
    *
    *    "If imageType is VK_IMAGE_TYPE_1D, flags must not contain
    *    VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT"
    */
   if ((pImageFormatInfo->flags & VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT) &&
       (pImageFormatInfo->type == VK_IMAGE_TYPE_1D ||
        pImageFormatInfo->tiling == VK_IMAGE_TILING_LINEAR))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;

   /* From the Vulkan 1.3.279 spec:
    *
    *    VUID-VkImageCreateInfo-flags-09403
    *
    *    "If flags contains VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT, flags
    *    must not include VK_IMAGE_CREATE_SPARSE_ALIASED_BIT,
    *    VK_IMAGE_CREATE_SPARSE_BINDING_BIT, or
    *    VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT"
    */
   if ((pImageFormatInfo->flags & VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT) &&
       (pImageFormatInfo->flags & (VK_IMAGE_CREATE_SPARSE_ALIASED_BIT |
                                   VK_IMAGE_CREATE_SPARSE_BINDING_BIT |
                                   VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT)))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;

   if (pImageFormatInfo->tiling == VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT &&
       pImageFormatInfo->type != VK_IMAGE_TYPE_2D)
      return VK_ERROR_FORMAT_NOT_SUPPORTED;

   const uint32_t max_dim =
      kk_image_max_dimension(pdev, pImageFormatInfo->type);
   assert(util_is_power_of_two_nonzero(max_dim));
   uint32_t maxMipLevels = util_logbase2(max_dim) + 1;
   VkExtent3D maxExtent;
   uint32_t maxArraySize;
   switch (pImageFormatInfo->type) {
   case VK_IMAGE_TYPE_1D:
      maxExtent = (VkExtent3D){max_dim, 1, 1};
      maxArraySize = 2048u;
      break;
   case VK_IMAGE_TYPE_2D:
      maxExtent = (VkExtent3D){max_dim, max_dim, 1};
      maxArraySize = 2048u;
      break;
   case VK_IMAGE_TYPE_3D:
      maxExtent = (VkExtent3D){max_dim, max_dim, max_dim};
      maxArraySize = 1u;
      break;
   default:
      UNREACHABLE("Invalid image type");
   }
   /* Buffer-backed Metal textures cannot have mipmaps or array layers. These
    * limits also apply to the LINEAR layout used for DRM modifier images. */
   if (pImageFormatInfo->tiling == VK_IMAGE_TILING_LINEAR)
      maxArraySize = 1;

   if (ycbcr_info != NULL || pImageFormatInfo->tiling == VK_IMAGE_TILING_LINEAR)
      maxMipLevels = 1;

   if (pImageFormatInfo->tiling == VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT) {
      maxArraySize = 1;
      maxMipLevels = 1;
   }

   VkSampleCountFlags sampleCounts = VK_SAMPLE_COUNT_1_BIT;
   if (pImageFormatInfo->tiling == VK_IMAGE_TILING_OPTIMAL &&
       pImageFormatInfo->type == VK_IMAGE_TYPE_2D && ycbcr_info == NULL &&
       (features & (VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT |
                    VK_FORMAT_FEATURE_2_DEPTH_STENCIL_ATTACHMENT_BIT)) &&
       !(pImageFormatInfo->flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) &&
       !(pImageFormatInfo->usage & VK_IMAGE_USAGE_STORAGE_BIT) && !sparse) {
      sampleCounts = pdev->info.supported_sample_counts;
   }

   /* From the Vulkan 1.2.199 spec:
    *
    *    "VK_IMAGE_CREATE_EXTENDED_USAGE_BIT specifies that the image can be
    *    created with usage flags that are not supported for the format the
    *    image is created with but are supported for at least one format a
    *    VkImageView created from the image can have."
    *
    * If VK_IMAGE_CREATE_EXTENDED_USAGE_BIT is set, views can be created with
    * different usage than the image so we can't always filter on usage.
    * There is one exception to this below for storage.
    */
   const VkImageUsageFlags image_usage = pImageFormatInfo->usage;
   VkImageUsageFlags view_usage = image_usage;
   if (pImageFormatInfo->flags & VK_IMAGE_CREATE_EXTENDED_USAGE_BIT)
      view_usage = 0;

   u_foreach_bit(b, view_usage) {
      VkFormatFeatureFlags2 usage_features =
         vk_image_usage_to_format_features(1 << b);
      if (usage_features && !(features & usage_features))
         return VK_ERROR_FORMAT_NOT_SUPPORTED;
   }

   const VkExternalMemoryProperties *ext_mem_props = NULL;
   if (external_info != NULL && external_info->handleType != 0) {
      /* We only support heaps since that's the backing for all our memory and
       * simplifies implementation */
      switch (external_info->handleType) {
      case VK_EXTERNAL_MEMORY_HANDLE_TYPE_MTLHEAP_BIT_EXT:
         ext_mem_props = &kk_mtlheap_mem_props;
         break;
      case VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT:
      case VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_MAPPED_FOREIGN_MEMORY_BIT_EXT:
         /* Explicit host imports use the imported pages, never a private heap. */
         ext_mem_props = &kk_host_mem_props;
         break;
      default:
         /* From the Vulkan 1.3.256 spec:
          *
          *    "If handleType is not compatible with the [parameters] in
          *    VkPhysicalDeviceImageFormatInfo2, then
          *    vkGetPhysicalDeviceImageFormatProperties2 returns
          *    VK_ERROR_FORMAT_NOT_SUPPORTED."
          */
         return vk_errorf(pdev, VK_ERROR_FORMAT_NOT_SUPPORTED,
                          "unsupported VkExternalMemoryHandleTypeFlagBits: %s ",
                          vk_ExternalMemoryHandleTypeFlagBits_to_str(
                             external_info->handleType));
      }
   }

   const unsigned plane_count =
      vk_format_get_plane_count(pImageFormatInfo->format);

   /* From the Vulkan 1.3.259 spec, VkImageCreateInfo:
    *
    *    VUID-VkImageCreateInfo-imageCreateFormatFeatures-02260
    *
    *    "If format is a multi-planar format, and if imageCreateFormatFeatures
    *    (as defined in Image Creation Limits) does not contain
    *    VK_FORMAT_FEATURE_DISJOINT_BIT, then flags must not contain
    *    VK_IMAGE_CREATE_DISJOINT_BIT"
    *
    * This is satisfied trivially because we support DISJOINT on all
    * multi-plane formats.  Also,
    *
    *    VUID-VkImageCreateInfo-format-01577
    *
    *    "If format is not a multi-planar format, and flags does not include
    *    VK_IMAGE_CREATE_ALIAS_BIT, flags must not contain
    *    VK_IMAGE_CREATE_DISJOINT_BIT"
    */
   if (plane_count == 1 &&
       !(pImageFormatInfo->flags & VK_IMAGE_CREATE_ALIAS_BIT) &&
       (pImageFormatInfo->flags & VK_IMAGE_CREATE_DISJOINT_BIT))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;

   if (ycbcr_info &&
       ((pImageFormatInfo->flags & VK_IMAGE_CREATE_SPARSE_BINDING_BIT) ||
        (pImageFormatInfo->flags & VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT)))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;

   if ((pImageFormatInfo->flags & VK_IMAGE_CREATE_SPARSE_BINDING_BIT) &&
       (pImageFormatInfo->usage & VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;

   pImageFormatProperties->imageFormatProperties = (VkImageFormatProperties){
      .maxExtent = maxExtent,
      .maxMipLevels = maxMipLevels,
      .maxArrayLayers = maxArraySize,
      .sampleCounts = sampleCounts,
      .maxResourceSize = UINT32_MAX, /* TODO */
   };

   vk_foreach_struct(sType, s, pImageFormatProperties->pNext) {
      switch (sType) {
      case VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES: {
         VkExternalImageFormatProperties *p = s;
         /* From the Vulkan 1.3.256 spec:
          *
          *    "If handleType is 0, vkGetPhysicalDeviceImageFormatProperties2
          *    will behave as if VkPhysicalDeviceExternalImageFormatInfo was
          *    not present, and VkExternalImageFormatProperties will be
          *    ignored."
          *
          * This is true if and only if ext_mem_props == NULL
          */
         if (ext_mem_props != NULL)
            p->externalMemoryProperties = *ext_mem_props;
         break;
      }
      case VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_IMAGE_FORMAT_PROPERTIES: {
         VkSamplerYcbcrConversionImageFormatProperties *ycbcr_props = s;
         ycbcr_props->combinedImageSamplerDescriptorCount = plane_count;
         break;
      }
      case VK_STRUCTURE_TYPE_HOST_IMAGE_COPY_DEVICE_PERFORMANCE_QUERY_EXT: {
         VkHostImageCopyDevicePerformanceQueryEXT *host_props = s;
         /* Optimal device access and identical memory layout if optimization
          * is the same both with and without host transfer usage */
         bool with_host_transfer = kk_image_layout_can_optimize(
            pdev, pImageFormatInfo->usage, pImageFormatInfo->tiling,
            pImageFormatInfo->flags, p_format);
         bool without_host_transfer = kk_image_layout_can_optimize(
            pdev, pImageFormatInfo->usage & ~VK_IMAGE_USAGE_HOST_TRANSFER_BIT,
            pImageFormatInfo->tiling, pImageFormatInfo->flags, p_format);
         host_props->optimalDeviceAccess =
            with_host_transfer == without_host_transfer;
         host_props->identicalMemoryLayout = host_props->optimalDeviceAccess;
         break;
      }
      default:
         vk_debug_ignored_stype(sType);
         break;
      }
   }

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
kk_GetPhysicalDeviceSparseImageFormatProperties2(
   VkPhysicalDevice physicalDevice,
   const VkPhysicalDeviceSparseImageFormatInfo2 *pFormatInfo,
   uint32_t *pPropertyCount, VkSparseImageFormatProperties2 *pProperties)
{
   VK_FROM_HANDLE(kk_physical_device, pdev, physicalDevice);
   VK_OUTARRAY_MAKE_TYPED(VkSparseImageFormatProperties2, out, pProperties,
                          pPropertyCount);

   /* Only formats, types and usages a sparse residency image can have */
   const VkPhysicalDeviceImageFormatInfo2 format_info = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
      .format = pFormatInfo->format,
      .type = pFormatInfo->type,
      .tiling = pFormatInfo->tiling,
      .usage = pFormatInfo->usage,
      .flags = VK_IMAGE_CREATE_SPARSE_BINDING_BIT |
               VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT,
   };
   VkImageFormatProperties2 format_props = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
   };
   VkExtent3D tile;
   if (pFormatInfo->samples != VK_SAMPLE_COUNT_1_BIT ||
       kk_GetPhysicalDeviceImageFormatProperties2(
          physicalDevice, &format_info, &format_props) != VK_SUCCESS ||
       !kk_sparse_image_supported(pdev, pFormatInfo->format, pFormatInfo->type,
                                  pFormatInfo->samples, pFormatInfo->tiling, 0u,
                                  &tile))
      return;

   vk_outarray_append_typed(VkSparseImageFormatProperties2, &out, props)
   {
      props->properties = (VkSparseImageFormatProperties){
         .aspectMask = vk_format_aspects(pFormatInfo->format),
         .imageGranularity = tile,
         /* A mip tail per layer, standard block shapes, tail levels smaller
          * than a tile */
         .flags = 0u,
      };
   }
}

static VkResult
kk_image_init(struct kk_device *dev, struct kk_image *image,
              const VkImageCreateInfo *pCreateInfo)
{
   const VkExternalMemoryImageCreateInfo *external_info =
      vk_find_struct_const(pCreateInfo->pNext, EXTERNAL_MEMORY_IMAGE_CREATE_INFO);
   const VkExternalMemoryHandleTypeFlags host_handles =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT |
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_MAPPED_FOREIGN_MEMORY_BIT_EXT;
   VkImageCreateInfo linear_info;
   if (external_info && (external_info->handleTypes & host_handles) &&
       pCreateInfo->tiling == VK_IMAGE_TILING_OPTIMAL) {
      linear_info = *pCreateInfo;
      linear_info.tiling = VK_IMAGE_TILING_LINEAR;
      pCreateInfo = &linear_info;
   }

   vk_image_init(&dev->vk, &image->vk, pCreateInfo);

   /* Match the non-optimal limits advertised by the image format query before
    * calculating a linear layout (which only allocates level 0). Metal aborts,
    * rather than returning nil, on a buffer-backed mipmapped/array/3D/MSAA
    * descriptor. Optimal host imports still use tiled textures and a private
    * heap; importing memory must not impose these restrictions on them. */
   if (image->vk.tiling != VK_IMAGE_TILING_OPTIMAL &&
       (image->vk.mip_levels != 1 || image->vk.array_layers != 1 ||
        image->vk.image_type == VK_IMAGE_TYPE_3D ||
        image->vk.samples != VK_SAMPLE_COUNT_1_BIT)) {
      return vk_errorf(dev, VK_ERROR_FORMAT_NOT_SUPPORTED,
                       "unsupported buffer-backed image: tiling %u, type %u, "
                       "mips %u, layers %u, samples %u",
                       image->vk.tiling, image->vk.image_type,
                       image->vk.mip_levels, image->vk.array_layers,
                       image->vk.samples);
   }

   if ((image->vk.usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                           VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) &&
       image->vk.samples > 1) {
      image->vk.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
      image->vk.stencil_usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
   }

   if (image->vk.usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
      image->vk.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
   if (image->vk.usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) {
      if (util_format_is_depth_or_stencil(
             vk_format_to_pipe_format(image->vk.format))) {
         image->vk.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
         image->vk.stencil_usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
      } else {
         image->vk.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
      }
   }

   image->plane_count = vk_format_get_plane_count(pCreateInfo->format);
   image->disjoint = image->plane_count > 1 &&
                     (pCreateInfo->flags & VK_IMAGE_CREATE_DISJOINT_BIT);

   const struct vk_format_ycbcr_info *ycbcr_info =
      vk_format_get_ycbcr_info(pCreateInfo->format);
   for (uint8_t plane = 0; plane < image->plane_count; plane++) {
      VkFormat format =
         ycbcr_info ? ycbcr_info->planes[plane].format : pCreateInfo->format;
      const uint8_t width_scale =
         ycbcr_info ? ycbcr_info->planes[plane].denominator_scales[0] : 1;
      const uint8_t height_scale =
         ycbcr_info ? ycbcr_info->planes[plane].denominator_scales[1] : 1;
      kk_image_layout_init(dev, &image->vk, vk_format_to_pipe_format(format),
                           width_scale, height_scale,
                           &image->planes[plane].layout);
   }

   /* virglrenderer (Venus on macOS) emulates dma-bufs with LINEAR host images
    * and keeps the importer's VkImageDrmFormatModifierExplicitCreateInfoEXT
    * chained: the image must use the exporter's row pitch, or the shared
    * memory is read with the wrong stride.
    */
   const VkImageDrmFormatModifierExplicitCreateInfoEXT *explicit_info =
      vk_find_struct_const(pCreateInfo->pNext,
                           IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT);
   if (explicit_info && pCreateInfo->tiling == VK_IMAGE_TILING_LINEAR) {
      struct kk_image_layout *layout = &image->planes[0].layout;
      const VkSubresourceLayout *plane_layout = &explicit_info->pPlaneLayouts[0];
      const uint64_t row_B =
         util_format_get_stride(layout->format.pipe, layout->width_px);
      if (image->plane_count != 1 ||
          explicit_info->drmFormatModifierPlaneCount != 1 ||
          plane_layout->offset != 0 || plane_layout->rowPitch < row_B ||
          plane_layout->rowPitch > UINT32_MAX ||
          plane_layout->rowPitch % layout->align_B)
         return VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT;
      layout->linear_stride_B = plane_layout->rowPitch;
      layout->layer_stride_B = plane_layout->rowPitch * layout->height_px;
      layout->size_B = layout->layer_stride_B;
      layout->level_offsets_B[1] = layout->layer_stride_B;
   }

   if (pCreateInfo->flags & VK_IMAGE_CREATE_SPARSE_BINDING_BIT)
      return kk_image_sparse_init(dev, image);

   return VK_SUCCESS;
}

static void
kk_image_plane_size_align_B(struct kk_device *dev, const struct kk_image *image,
                            const struct kk_image_plane *plane,
                            uint64_t *size_B_out, uint64_t *align_B_out)
{
   *size_B_out = plane->layout.size_B;
   *align_B_out = plane->layout.align_B;
}

static void
kk_image_plane_add_to_residency_set(struct kk_device *dev,
                                    struct kk_image_plane *plane)
{
   if (plane->mtl_handle)
      kk_device_add_texture_to_residency_set(dev, plane->mtl_handle);
   if (plane->mtl_handle_array)
      kk_device_add_texture_to_residency_set(dev, plane->mtl_handle_array);
   plane->resident = true;
}

void
kk_image_plane_remove_from_residency_set(struct kk_device *dev,
                                         struct kk_image_plane *plane)
{
   if (plane->mtl_handle)
      kk_device_remove_texture_from_residency_set(dev, plane->mtl_handle);
   if (plane->mtl_handle_array)
      kk_device_remove_texture_from_residency_set(dev, plane->mtl_handle_array);
   plane->resident = false;
}

static void
kk_image_plane_finish(struct kk_device *dev, struct kk_image_plane *plane,
                      VkImageCreateFlags create_flags,
                      const VkAllocationCallbacks *pAllocator)
{
   if (plane->resident)
      kk_image_plane_remove_from_residency_set(dev, plane);
   if (plane->mtl_handle != NULL)
      mtl_release(plane->mtl_handle);
   if (plane->mtl_handle_array != NULL)
      mtl_release(plane->mtl_handle_array);
}

static void
kk_image_finish(struct kk_device *dev, struct kk_image *image,
                const VkAllocationCallbacks *pAllocator)
{
   for (uint8_t plane = 0; plane < image->plane_count; plane++) {
      kk_image_plane_finish(dev, &image->planes[plane], image->vk.create_flags,
                            pAllocator);
   }

   vk_image_finish(&image->vk);
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_CreateImage(VkDevice _device, const VkImageCreateInfo *pCreateInfo,
               const VkAllocationCallbacks *pAllocator, VkImage *pImage)
{
   VK_FROM_HANDLE(kk_device, dev, _device);
   struct kk_physical_device *pdev = kk_device_physical(dev);
   struct kk_image *image;
   VkResult result;

   if (wsi_common_is_swapchain_image(pCreateInfo))
      return wsi_common_create_swapchain_image(&pdev->wsi_device, pCreateInfo,
                                               pImage);

   image = vk_zalloc2(&dev->vk.alloc, pAllocator, sizeof(*image), 8,
                      VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!image)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   result = kk_image_init(dev, image, pCreateInfo);
   if (result != VK_SUCCESS) {
      kk_image_finish(dev, image, pAllocator);
      vk_free2(&dev->vk.alloc, pAllocator, image);
      return result;
   }

   /* Sparse textures map heap pages, the texture itself must be resident */
   if (image->vk.create_flags & VK_IMAGE_CREATE_SPARSE_BINDING_BIT)
      kk_image_plane_add_to_residency_set(dev, &image->planes[0]);

   *pImage = kk_image_to_handle(image);

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
kk_DestroyImage(VkDevice device, VkImage _image,
                const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(kk_device, dev, device);
   VK_FROM_HANDLE(kk_image, image, _image);

   if (!image)
      return;

   kk_image_finish(dev, image, pAllocator);
   vk_free2(&dev->vk.alloc, pAllocator, image);
}

static void
kk_image_plane_add_req(struct kk_device *dev, const struct kk_image *image,
                       const struct kk_image_plane *plane, uint64_t *size_B,
                       uint32_t *align_B)
{
   assert(util_is_power_of_two_or_zero64(*align_B));
   uint64_t plane_size_B, plane_align_B;
   kk_image_plane_size_align_B(dev, image, plane, &plane_size_B,
                               &plane_align_B);

   *align_B = MAX2(*align_B, plane_align_B);
   *size_B = align64(*size_B, plane_align_B);
   *size_B += plane_size_B;
}

static void
kk_get_image_memory_requirements(struct kk_device *dev, struct kk_image *image,
                                 VkImageAspectFlags aspects,
                                 VkMemoryRequirements2 *pMemoryRequirements)
{
   struct kk_physical_device *pdev = kk_device_physical(dev);
   uint32_t memory_types = (1 << pdev->mem_type_count) - 1;

   /* Remove non host visible heaps from the types for host image copy in case
    * of potential issues. This should be removed when we get ReBAR.
    */
   if (image->vk.usage & VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT) {
      for (uint32_t i = 0; i < pdev->mem_type_count; i++) {
         if (!(pdev->mem_types[i].propertyFlags &
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
            memory_types &= ~BITFIELD_BIT(i);
      }
   }

   // TODO hope for the best?

   uint64_t size_B = 0;
   uint32_t align_B = 0;
   if (image->disjoint) {
      uint8_t plane = kk_image_memory_aspects_to_plane(image, aspects);
      kk_image_plane_add_req(dev, image, &image->planes[plane], &size_B,
                             &align_B);
   } else {
      for (unsigned plane = 0; plane < image->plane_count; plane++) {
         kk_image_plane_add_req(dev, image, &image->planes[plane], &size_B,
                                &align_B);
      }
   }

   /* Pages of the opaque layout, mapped from heaps that take 64 KiB pages */
   if (image->vk.create_flags & VK_IMAGE_CREATE_SPARSE_BINDING_BIT) {
      memory_types = pdev->sparse_mem_types;
      align_B = KK_SPARSE_PAGE_SIZE_B;
      size_B = kk_image_sparse_size_B(image);
   }

   pMemoryRequirements->memoryRequirements.memoryTypeBits = memory_types;
   pMemoryRequirements->memoryRequirements.alignment = align_B;
   pMemoryRequirements->memoryRequirements.size = size_B;

   vk_foreach_struct(sType, ext, pMemoryRequirements->pNext) {
      switch (sType) {
      case VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS: {
         VkMemoryDedicatedRequirements *dedicated = ext;
         dedicated->prefersDedicatedAllocation =
            image->vk.tiling == VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
         dedicated->requiresDedicatedAllocation =
            image->vk.tiling == VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
         break;
      }
      default:
         vk_debug_ignored_stype(sType);
         break;
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
kk_GetImageMemoryRequirements2(VkDevice device,
                               const VkImageMemoryRequirementsInfo2 *pInfo,
                               VkMemoryRequirements2 *pMemoryRequirements)
{
   VK_FROM_HANDLE(kk_device, dev, device);
   VK_FROM_HANDLE(kk_image, image, pInfo->image);

   const VkImagePlaneMemoryRequirementsInfo *plane_info =
      vk_find_struct_const(pInfo->pNext, IMAGE_PLANE_MEMORY_REQUIREMENTS_INFO);
   const VkImageAspectFlags aspects =
      image->disjoint ? plane_info->planeAspect : image->vk.aspects;

   kk_get_image_memory_requirements(dev, image, aspects, pMemoryRequirements);
}

VKAPI_ATTR void VKAPI_CALL
kk_GetDeviceImageMemoryRequirements(VkDevice device,
                                    const VkDeviceImageMemoryRequirements *pInfo,
                                    VkMemoryRequirements2 *pMemoryRequirements)
{
   VK_FROM_HANDLE(kk_device, dev, device);
   ASSERTED VkResult result;
   struct kk_image image = {0};

   result = kk_image_init(dev, &image, pInfo->pCreateInfo);
   assert(result == VK_SUCCESS);

   const VkImageAspectFlags aspects =
      image.disjoint ? pInfo->planeAspect : image.vk.aspects;

   kk_get_image_memory_requirements(dev, &image, aspects, pMemoryRequirements);

   kk_image_finish(dev, &image, NULL);
}

VKAPI_ATTR void VKAPI_CALL
kk_GetImageSparseMemoryRequirements2(
   VkDevice device, const VkImageSparseMemoryRequirementsInfo2 *pInfo,
   uint32_t *pSparseMemoryRequirementCount,
   VkSparseImageMemoryRequirements2 *pSparseMemoryRequirements)
{
   VK_FROM_HANDLE(kk_image, image, pInfo->image);

   kk_image_get_sparse_memory_requirements(
      image, pSparseMemoryRequirementCount, pSparseMemoryRequirements);
}

VKAPI_ATTR void VKAPI_CALL
kk_GetDeviceImageSparseMemoryRequirements(
   VkDevice device, const VkDeviceImageMemoryRequirements *pInfo,
   uint32_t *pSparseMemoryRequirementCount,
   VkSparseImageMemoryRequirements2 *pSparseMemoryRequirements)
{
   VK_FROM_HANDLE(kk_device, dev, device);
   ASSERTED VkResult result;
   struct kk_image image = {0};

   result = kk_image_init(dev, &image, pInfo->pCreateInfo);
   assert(result == VK_SUCCESS);

   kk_image_get_sparse_memory_requirements(
      &image, pSparseMemoryRequirementCount, pSparseMemoryRequirements);

   kk_image_finish(dev, &image, NULL);
}

static void
kk_get_image_subresource_layout(struct kk_device *dev, struct kk_image *image,
                                const VkImageSubresource2KHR *pSubresource,
                                VkSubresourceLayout2KHR *pLayout)
{
   const VkImageSubresource *isr = &pSubresource->imageSubresource;

   const uint8_t p = kk_image_memory_aspects_to_plane(image, isr->aspectMask);
   const struct kk_image_plane *plane = &image->planes[p];

   uint64_t offset_B = 0;
   if (!image->disjoint) {
      uint32_t align_B = 0;
      for (unsigned i = 0; i < p; i++) {
         kk_image_plane_add_req(dev, image, &image->planes[i], &offset_B,
                                &align_B);
      }
   }

   pLayout->subresourceLayout = (VkSubresourceLayout){
      .offset = offset_B,
      .size = plane->layout.size_B,
      .rowPitch = plane->layout.linear_stride_B,
      .arrayPitch = plane->layout.layer_stride_B,
      .depthPitch = 1u,
   };

   VkSubresourceHostMemcpySize *memcpy_size =
      vk_find_struct(pLayout, SUBRESOURCE_HOST_MEMCPY_SIZE_EXT);
   if (memcpy_size) {
      memcpy_size->size = pLayout->subresourceLayout.size;
   }
}

VKAPI_ATTR void VKAPI_CALL
kk_GetImageSubresourceLayout2KHR(VkDevice device, VkImage _image,
                                 const VkImageSubresource2KHR *pSubresource,
                                 VkSubresourceLayout2KHR *pLayout)
{
   VK_FROM_HANDLE(kk_device, dev, device);
   VK_FROM_HANDLE(kk_image, image, _image);

   kk_get_image_subresource_layout(dev, image, pSubresource, pLayout);
}

VKAPI_ATTR void VKAPI_CALL
kk_GetDeviceImageSubresourceLayoutKHR(
   VkDevice device, const VkDeviceImageSubresourceInfoKHR *pInfo,
   VkSubresourceLayout2KHR *pLayout)
{
   VK_FROM_HANDLE(kk_device, dev, device);
   ASSERTED VkResult result;
   struct kk_image image = {0};

   result = kk_image_init(dev, &image, pInfo->pCreateInfo);
   assert(result == VK_SUCCESS);

   kk_get_image_subresource_layout(dev, &image, pInfo->pSubresource, pLayout);

   kk_image_finish(dev, &image, NULL);
}

mtl_texture *
kk_image_plane_create_texture(struct kk_image_plane *plane,
                              struct kk_image_layout *layout, uint64_t offset_B)
{
   uint64_t mem_offset_B = plane->mem_offset_B + offset_B;
   if (plane->layout.linear) {
      /* Metal requires buffer-backed textures to be plain 2D, so a linear
       * array layout gets a 2D texture and an array view of it. */
      if (layout->type != MTL_TEXTURE_TYPE_2D_ARRAY || layout->layers != 1)
         return mtl_new_texture_with_descriptor_linear(plane->mem->bo->map,
                                                       layout, mem_offset_B);

      struct kk_image_layout layout_2d = *layout;
      layout_2d.type = MTL_TEXTURE_TYPE_2D;
      mtl_texture *texture = mtl_new_texture_with_descriptor_linear(
         plane->mem->bo->map, &layout_2d, mem_offset_B);
      if (!texture)
         return NULL;

      const struct kk_view_layout view_layout = {
         .view_type = VK_IMAGE_VIEW_TYPE_2D_ARRAY,
         .sample_count_sa = 1,
         .format.pipe = layout->format.pipe,
         .format.mtl = layout->format.mtl,
         .num_levels = 1,
         .array_len = 1,
      };
      mtl_texture *view = mtl_new_texture_view_with_no_swizzle(texture, &view_layout);
      mtl_release(texture);
      return view;
   }

   return mtl_new_texture_with_descriptor(plane->mem->bo->mtl_handle, layout,
                                          mem_offset_B);
}

static VkResult
kk_image_plane_bind(struct kk_device *dev, struct kk_image *image,
                    struct kk_image_plane *plane, struct kk_device_memory *mem,
                    uint64_t *offset_B)
{
   uint64_t plane_size_B, plane_align_B;
   kk_image_plane_size_align_B(dev, image, plane, &plane_size_B,
                               &plane_align_B);
   *offset_B = align64(*offset_B, plane_align_B);

   /* Memory imported from a host pointer has no heap, which tiled textures
    * need. Give it a private heap of its size on the first tiled bind: tiled
    * contents are opaque to the host, so they needn't live in the imported
    * pages. Binds may race, so only one heap is kept. */
   if (!plane->layout.linear && !mem->bo->mtl_handle) {
      mtl_heap *heap =
         mtl_new_heap(dev->mtl_handle, mem->bo->size_B, KK_MTL_RESOURCE_OPTIONS,
                      false);
      if (!heap)
         return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      if (p_atomic_cmpxchg_ptr(&mem->bo->mtl_handle, NULL, heap) == NULL)
         kk_device_add_heap_to_residency_set(dev, heap);
      else
         mtl_release(heap);
   }

   /* Linear textures in Metal need to be allocated through a buffer... */
   plane->mem = mem;
   plane->mem_offset_B = *offset_B;
   plane->mtl_handle = kk_image_plane_create_texture(plane, &plane->layout, 0u);
   if (image->vk.base.object_name)
      mtl_resource_set_label(plane->mtl_handle, image->vk.base.object_name);

   plane->addr = mem->bo->gpu + *offset_B;

   /* Create auxiliary 2D array texture for 3D images so we can use 2D views of
    * it */
   if (plane->layout.type == MTL_TEXTURE_TYPE_3D &&
       (image->vk.create_flags &
        (VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT |
         VK_IMAGE_CREATE_2D_VIEW_COMPATIBLE_BIT_EXT))) {
      struct kk_image_layout array_layout = plane->layout;
      array_layout.type = MTL_TEXTURE_TYPE_2D_ARRAY;
      array_layout.layers = array_layout.layers * array_layout.depth_px;
      array_layout.depth_px = 1u;
      plane->mtl_handle_array = mtl_new_texture_with_descriptor(
         mem->bo->mtl_handle, &array_layout, *offset_B);
      if (image->vk.base.object_name)
         mtl_resource_set_label(plane->mtl_handle_array,
                                image->vk.base.object_name);
   }

   kk_image_plane_add_to_residency_set(dev, plane);

   *offset_B += plane_size_B;

   return VK_SUCCESS;
}

static VkResult
kk_bind_image_memory(struct kk_device *dev, const VkBindImageMemoryInfo *info)
{
   VK_FROM_HANDLE(kk_device_memory, mem, info->memory);
   VK_FROM_HANDLE(kk_image, image, info->image);
   VkResult result;

   /* Ignore this struct on Android, we cannot access swapchain structures
    * there. */
#ifdef KK_USE_WSI_PLATFORM
   const VkBindImageMemorySwapchainInfoKHR *swapchain_info =
      vk_find_struct_const(info->pNext, BIND_IMAGE_MEMORY_SWAPCHAIN_INFO_KHR);

   if (swapchain_info && swapchain_info->swapchain != VK_NULL_HANDLE) {
      VK_FROM_HANDLE(wsi_swapchain, swapchain, swapchain_info->swapchain);
      VkImage _wsi_image =
         swapchain->get_wsi_image(swapchain, swapchain_info->imageIndex)->image;
      VK_FROM_HANDLE(kk_image, wsi_img, _wsi_image);

      assert(image->plane_count == 1);
      assert(wsi_img->plane_count == 1);

      struct kk_image_plane *plane = &image->planes[0];
      struct kk_image_plane *swapchain_plane = &wsi_img->planes[0];

      /* Copy swapchain plane data retaining relevant resources. */
      plane->layout = swapchain_plane->layout;
      plane->mtl_handle = mtl_retain(swapchain_plane->mtl_handle);
      plane->mtl_handle_array =
         swapchain_plane->mtl_handle_array
            ? mtl_retain(swapchain_plane->mtl_handle_array)
            : NULL;
      plane->addr = swapchain_plane->addr;

      return VK_SUCCESS;
   }
#endif

   uint64_t offset_B = info->memoryOffset;
   if (image->disjoint) {
      const VkBindImagePlaneMemoryInfo *plane_info =
         vk_find_struct_const(info->pNext, BIND_IMAGE_PLANE_MEMORY_INFO);
      const uint8_t plane =
         kk_image_memory_aspects_to_plane(image, plane_info->planeAspect);
      result =
         kk_image_plane_bind(dev, image, &image->planes[plane], mem, &offset_B);
      if (result != VK_SUCCESS)
         return result;
   } else {
      for (unsigned plane = 0; plane < image->plane_count; plane++) {
         result = kk_image_plane_bind(dev, image, &image->planes[plane], mem,
                                      &offset_B);
         if (result != VK_SUCCESS)
            return result;
      }
   }

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_BindImageMemory2(VkDevice device, uint32_t bindInfoCount,
                    const VkBindImageMemoryInfo *pBindInfos)
{
   VK_FROM_HANDLE(kk_device, dev, device);
   VkResult first_error_or_success = VK_SUCCESS;

   for (uint32_t i = 0; i < bindInfoCount; ++i) {
      VkResult result = kk_bind_image_memory(dev, &pBindInfos[i]);

      const VkBindMemoryStatusKHR *status =
         vk_find_struct_const(pBindInfos[i].pNext, BIND_MEMORY_STATUS_KHR);
      if (status != NULL && status->pResult != NULL)
         *status->pResult = VK_SUCCESS;

      if (first_error_or_success == VK_SUCCESS)
         first_error_or_success = result;
   }

   return first_error_or_success;
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_GetImageOpaqueCaptureDescriptorDataEXT(
   VkDevice _device, const VkImageCaptureDescriptorDataInfoEXT *pInfo,
   void *pData)
{
   return VK_SUCCESS;
}

struct kk_host_copy_info {
   struct mtl_texture_memory_copy mtl_data;
   size_t buffer_slice_size_B;
};

static struct kk_host_copy_info
vk_image_to_memory_copy_to_mtl_texture_memory_copy(
   const VkImageToMemoryCopy *region, const struct kk_image_plane *plane)
{
   struct kk_host_copy_info copy;
   enum pipe_format p_format = plane->layout.format.pipe;
   if (region->imageSubresource.aspectMask == VK_IMAGE_ASPECT_DEPTH_BIT)
      p_format = util_format_get_depth_only(p_format);
   else if (region->imageSubresource.aspectMask == VK_IMAGE_ASPECT_STENCIL_BIT)
      p_format = PIPE_FORMAT_S8_UINT;

   const uint32_t buffer_width = region->memoryRowLength
                                    ? region->memoryRowLength
                                    : region->imageExtent.width;
   const uint32_t buffer_height = region->memoryImageHeight
                                     ? region->memoryImageHeight
                                     : region->imageExtent.height;

   const uint32_t buffer_stride_B =
      util_format_get_stride(p_format, buffer_width);
   const uint32_t buffer_size_2d_B =
      util_format_get_2d_size(p_format, buffer_stride_B, buffer_height);

   /* Metal requires this value to be 0 for 2D images, otherwise the number of
    * bytes between each 2D image of a 3D texture */
   copy.mtl_data.buffer_2d_image_size_B =
      plane->layout.depth_px == 1u ? 0u : buffer_size_2d_B;
   copy.mtl_data.buffer_stride_B = buffer_stride_B;
   copy.mtl_data.image_size = vk_extent_3d_to_mtl_size(&region->imageExtent);
   copy.mtl_data.image_origin =
      vk_offset_3d_to_mtl_origin(&region->imageOffset);
   copy.mtl_data.image_level = region->imageSubresource.mipLevel;
   copy.buffer_slice_size_B = buffer_size_2d_B * region->imageExtent.depth;

   return copy;
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_CopyImageToMemory(UNUSED VkDevice device,
                     const VkCopyImageToMemoryInfo *pCopyImageToMemoryInfo)
{
   VK_FROM_HANDLE(kk_image, image, pCopyImageToMemoryInfo->srcImage);

   for (unsigned r = 0; r < pCopyImageToMemoryInfo->regionCount; r++) {
      const VkImageToMemoryCopy *region = &pCopyImageToMemoryInfo->pRegions[r];

      const uint8_t plane_index = kk_image_memory_aspects_to_plane(
         image, region->imageSubresource.aspectMask);
      struct kk_image_plane *plane = &image->planes[plane_index];

      struct kk_host_copy_info info =
         vk_image_to_memory_copy_to_mtl_texture_memory_copy(region, plane);

      uint8_t *host_ptr = region->pHostPointer;
      kk_foreach_slice(slice, image, imageSubresource)
      {
         info.mtl_data.image_slice = slice;
         mtl_texture_get_bytes(plane->mtl_handle, host_ptr, &info.mtl_data);
         host_ptr += info.buffer_slice_size_B;
      }
   }

   return VK_SUCCESS;
}

static struct kk_host_copy_info
vk_memory_to_image_copy_to_mtl_texture_memory_copy(
   const VkMemoryToImageCopy *region, const struct kk_image_plane *plane)
{
   /* Prevent code duplication by mapping between structures */
   const VkImageToMemoryCopy mapped = {
      .memoryRowLength = region->memoryRowLength,
      .memoryImageHeight = region->memoryImageHeight,
      .imageSubresource = region->imageSubresource,
      .imageOffset = region->imageOffset,
      .imageExtent = region->imageExtent,
   };
   return vk_image_to_memory_copy_to_mtl_texture_memory_copy(&mapped, plane);
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_CopyMemoryToImage(UNUSED VkDevice device,
                     const VkCopyMemoryToImageInfo *pCopyMemoryToImageInfo)
{
   VK_FROM_HANDLE(kk_image, image, pCopyMemoryToImageInfo->dstImage);

   for (int r = 0; r < pCopyMemoryToImageInfo->regionCount; r++) {
      const VkMemoryToImageCopy *region = &pCopyMemoryToImageInfo->pRegions[r];

      const uint8_t plane_index = kk_image_memory_aspects_to_plane(
         image, region->imageSubresource.aspectMask);
      struct kk_image_plane *plane = &image->planes[plane_index];

      struct kk_host_copy_info info =
         vk_memory_to_image_copy_to_mtl_texture_memory_copy(region, plane);

      const uint8_t *host_ptr = region->pHostPointer;
      kk_foreach_slice(slice, image, imageSubresource)
      {
         info.mtl_data.image_slice = slice;
         mtl_texture_replace_region(plane->mtl_handle, host_ptr,
                                    &info.mtl_data);
         host_ptr += info.buffer_slice_size_B;
      }
   }

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_CopyImageToImage(VkDevice device,
                    const VkCopyImageToImageInfo *pCopyImageToImageInfo)
{
   VK_FROM_HANDLE(kk_image, src, pCopyImageToImageInfo->srcImage);

   /* Determine the buffer size required to satisfy all copies */
   uint64_t buffer_size = 0;
   for (uint32_t i = 0u; i < pCopyImageToImageInfo->regionCount; i++) {
      const VkImageCopy2 *region = &pCopyImageToImageInfo->pRegions[i];

      uint8_t src_index =
         kk_image_aspects_to_plane(src, region->srcSubresource.aspectMask);
      struct kk_image_plane *src_plane = &src->planes[src_index];

      buffer_size = MAX2(buffer_size, src_plane->layout.size_B);
   }

   /* Metal does not provide a direct image-to-image host copy, so we implement
    * host image-to-image copy using CopyImageToMemory and CopyMemoryToImage */
   uint8_t *temp = ralloc_size(NULL, buffer_size);
   if (!temp)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   VkResult result = VK_SUCCESS;

   for (uint32_t i = 0u; i < pCopyImageToImageInfo->regionCount; ++i) {
      const VkImageCopy2 *region = &pCopyImageToImageInfo->pRegions[i];

      VkImageToMemoryCopy src_copy = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_TO_MEMORY_COPY,
         .pNext = NULL,
         .pHostPointer = temp,
         .memoryRowLength = 0,
         .memoryImageHeight = 0,
         .imageSubresource = region->srcSubresource,
         .imageOffset = region->srcOffset,
         .imageExtent = region->extent,
      };
      VkCopyImageToMemoryInfo src_copy_info = {
         .sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_MEMORY_INFO,
         .pNext = NULL,
         .flags = pCopyImageToImageInfo->flags,
         .srcImage = pCopyImageToImageInfo->srcImage,
         .srcImageLayout = pCopyImageToImageInfo->srcImageLayout,
         .regionCount = 1,
         .pRegions = &src_copy,
      };
      result = kk_CopyImageToMemory(device, &src_copy_info);
      if (result != VK_SUCCESS)
         break;

      VkMemoryToImageCopy dst_copy = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_TO_IMAGE_COPY,
         .pNext = NULL,
         .pHostPointer = temp,
         .memoryRowLength = 0,
         .memoryImageHeight = 0,
         .imageSubresource = region->dstSubresource,
         .imageOffset = region->dstOffset,
         .imageExtent = region->extent,
      };
      VkCopyMemoryToImageInfo dst_copy_info = {
         .sType = VK_STRUCTURE_TYPE_COPY_MEMORY_TO_IMAGE_INFO,
         .pNext = NULL,
         .flags = pCopyImageToImageInfo->flags,
         .dstImage = pCopyImageToImageInfo->dstImage,
         .dstImageLayout = pCopyImageToImageInfo->dstImageLayout,
         .regionCount = 1,
         .pRegions = &dst_copy,
      };
      result = kk_CopyMemoryToImage(device, &dst_copy_info);
      if (result != VK_SUCCESS)
         break;
   }

   ralloc_free(temp);
   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_TransitionImageLayoutEXT(
   UNUSED VkDevice device, UNUSED uint32_t transitionCount,
   UNUSED const VkHostImageLayoutTransitionInfoEXT *transitions)
{
   /* We don't do anything with layouts so this should be a no-op */
   return VK_SUCCESS;
}

void
kk_image_set_label(struct kk_image *image, const char *label)
{
   for (uint8_t i = 0; i < image->plane_count; i++) {
      if (image->planes[i].mtl_handle)
         mtl_resource_set_label(image->planes[i].mtl_handle, label);
      if (image->planes[i].mtl_handle_array)
         mtl_resource_set_label(image->planes[i].mtl_handle_array, label);
   }
}
