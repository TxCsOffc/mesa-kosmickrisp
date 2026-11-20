/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef KK_IMAGE_H
#define KK_IMAGE_H 1

#include "kk_private.h"

#include "kk_device_memory.h"
#include "kk_image_layout.h"

#include "kosmickrisp/bridge/mtl_types.h"

#include "vk_image.h"

struct kk_device;
struct kk_device_memory;
struct kk_physical_device;
struct kk_queue;

VkFormatFeatureFlags2
kk_get_image_format_features(struct kk_physical_device *pdevice,
                             VkFormat format, VkImageTiling tiling,
                             uint64_t drm_format_mod);

uint32_t kk_image_max_dimension(const struct kk_physical_device *pdev,
                                VkImageType image_type);

struct kk_image_plane {
   struct kk_image_layout layout;
   struct kk_device_memory *mem;
   uint64_t mem_offset_B;
   // TODO_KOSMICKRISP Only have one handle since we will only create 2D arrays
   // anyway
   /* Metal handle with original handle type */
   mtl_texture *mtl_handle;
   /* Metal handle with 2D array type for 3D images */
   mtl_texture *mtl_handle_array;
   uint64_t addr;
   /* The base textures of mtl_handle and mtl_handle_array were added to the
    * device's residency set by this plane (bound to memory, or sparse) and
    * are removed when it is finished. Not set for planes that alias a
    * swapchain image's or a drawable's textures. */
   bool resident;
};

/* Sparse images (VK_IMAGE_CREATE_SPARSE_BINDING_BIT) are Metal placement
 * sparse textures created with the image; memory is mapped per 64 KiB page
 * (tile) by vkQueueBindSparse (kk_queue_bind_sparse).
 *
 * Opaque binds address the image as layers of layer_pages pages each: the
 * tiles of every level below Metal's tail (x, then y, then z), then Metal's
 * tail. The Vulkan mip tail starts at vk_tail_level, the first level smaller
 * than a tile in some dimension, which may be below Metal's tail (Metal maps
 * levels of at least half a tile individually): those levels are part of the
 * Vulkan mip tail region and only bound opaquely. */
struct kk_image_sparse {
   /* Tile (sparse image block) size in texels */
   VkExtent3D tile;
   uint32_t vk_tail_level;
   uint32_t mtl_tail_level;
   /* Page offset of each level within a layer, up to and including
    * mtl_tail_level (where Metal's tail starts) */
   uint64_t level_page[KK_MAX_MIP_LEVELS + 1];
   /* Tiles of each level below mtl_tail_level */
   VkExtent3D level_tiles[KK_MAX_MIP_LEVELS];
   uint64_t layer_pages;
   /* The Metal texture is an ordinary one with its own storage: binding it
    * does nothing and unbinding it zeroes it (see kk_image_sparse_init) */
   bool backed;
};

struct kk_image {
   struct vk_image vk;

   /** True if the planes are bound separately
    * * This is set based on VK_IMAGE_CREATE_DISJOINT_BIT
    */
   bool disjoint;

   /* Forces alpha value to be one when creating views. Needed for WSI */
   bool wsi_opaque_alpha;

   uint8_t plane_count;
   struct kk_image_plane planes[3];

   /* VK_IMAGE_CREATE_SPARSE_BINDING_BIT (single plane) */
   struct kk_image_sparse sparse;
};

#define kk_foreach_slice(ndx, image, subresource_member)                       \
   for (uint32_t ndx = region->subresource_member.baseArrayLayer;              \
        ndx < (region->subresource_member.baseArrayLayer +                     \
               vk_image_subresource_layer_count(&image->vk,                    \
                                                &region->subresource_member)); \
        ++ndx)

static inline mtl_resource *
kk_image_to_mtl_resource(const struct kk_image *image, int plane)
{
   if (image != NULL) {
      assert(plane < ARRAY_SIZE(image->planes));
      return (mtl_resource *)image->planes[plane].mtl_handle;
   }
   return NULL;
}

VK_DEFINE_NONDISP_HANDLE_CASTS(kk_image, vk.base, VkImage, VK_OBJECT_TYPE_IMAGE)

static inline uint64_t
kk_image_plane_base_address(const struct kk_image_plane *plane)
{
   return plane->addr;
}

static inline uint64_t
kk_image_base_address(const struct kk_image *image, uint8_t plane)
{
   return kk_image_plane_base_address(&image->planes[plane]);
}

static inline uint8_t
kk_image_aspects_to_plane(ASSERTED const struct kk_image *image,
                          VkImageAspectFlags aspectMask)
{
   /* Memory planes are only allowed for memory operations */
   assert(!(aspectMask & (VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT |
                          VK_IMAGE_ASPECT_MEMORY_PLANE_1_BIT_EXT |
                          VK_IMAGE_ASPECT_MEMORY_PLANE_2_BIT_EXT |
                          VK_IMAGE_ASPECT_MEMORY_PLANE_3_BIT_EXT)));

   /* Verify that the aspects are actually in the image */
   assert(!(aspectMask & ~image->vk.aspects));

   /* Must only be one aspect unless it's depth/stencil */
   assert(aspectMask ==
             (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT) ||
          util_bitcount(aspectMask) == 1);

   switch (aspectMask) {
   case VK_IMAGE_ASPECT_PLANE_1_BIT:
      return 1;
   case VK_IMAGE_ASPECT_PLANE_2_BIT:
      return 2;
   default:
      return 0;
   }
}

static inline uint8_t
kk_image_memory_aspects_to_plane(ASSERTED const struct kk_image *image,
                                 VkImageAspectFlags aspectMask)
{
   if (aspectMask & (VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT |
                     VK_IMAGE_ASPECT_MEMORY_PLANE_1_BIT_EXT |
                     VK_IMAGE_ASPECT_MEMORY_PLANE_2_BIT_EXT |
                     VK_IMAGE_ASPECT_MEMORY_PLANE_3_BIT_EXT)) {
      /* We don't support DRM format modifiers on anything but single-plane
       * color at the moment.
       */
      assert(aspectMask == VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT);
      return 0;
   } else {
      return kk_image_aspects_to_plane(image, aspectMask);
   }
}

/* A new texture of the plane's memory (not a view of mtl_handle): the caller
 * owns it and its residency (kk_device_add_texture_to_residency_set). */
mtl_texture *kk_image_plane_create_texture(struct kk_image_plane *plane,
                                           struct kk_image_layout *layout,
                                           uint64_t offset_B);

void kk_image_plane_remove_from_residency_set(struct kk_device *dev,
                                              struct kk_image_plane *plane);

void kk_image_set_label(struct kk_image *image, const char *label);

#endif
