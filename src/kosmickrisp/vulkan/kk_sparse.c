/*
 * Copyright 2026 steamac
 * SPDX-License-Identifier: MIT
 */

#include "kk_sparse.h"

#include "kk_bo.h"
#include "kk_buffer.h"
#include "kk_device.h"
#include "kk_device_memory.h"
#include "kk_format.h"
#include "kk_image.h"
#include "kk_physical_device.h"
#include "kk_queue.h"

#include "kosmickrisp/bridge/mtl_bridge.h"

#include "util/format/u_format.h"
#include "util/u_dynarray.h"
#include "util/u_math.h"
#include "vk_format.h"
#include "vk_queue.h"

/* Standard sparse image block shapes of single-sample 2D images, in texels
 * (Vulkan spec, "Standard Sparse Image Block Shapes") */
static bool
kk_standard_2d_block_shape(enum pipe_format format, VkExtent3D *shape)
{
   const struct util_format_description *desc = util_format_description(format);
   uint32_t width, height;
   switch (desc->block.bits) {
   case 8:
      width = 256, height = 256;
      break;
   case 16:
      width = 256, height = 128;
      break;
   case 32:
      width = 128, height = 128;
      break;
   case 64:
      width = 128, height = 64;
      break;
   case 128:
      width = 64, height = 64;
      break;
   default:
      return false;
   }
   *shape = (VkExtent3D){width * desc->block.width,
                         height * desc->block.height, 1};
   return true;
}

bool
kk_sparse_image_supported(const struct kk_physical_device *pdev,
                          VkFormat vk_format, VkImageType type,
                          VkSampleCountFlagBits samples, VkImageTiling tiling,
                          VkImageCreateFlags flags, VkExtent3D *tile_out)
{
   if (!pdev->info.placement_sparse)
      return false;

   /* Metal's sparse 3D tiles are 2D-shaped (one slice), not Vulkan's standard
    * 3D blocks; no sparse multisampling. */
   if (type != VK_IMAGE_TYPE_2D || samples != VK_SAMPLE_COUNT_1_BIT ||
       tiling != VK_IMAGE_TILING_OPTIMAL)
      return false;

   /* Block texel views alias subresources at heap offsets (kk_image_view.c),
    * which a sparse texture doesn't have. */
   if (flags & (VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT |
                VK_IMAGE_CREATE_DISJOINT_BIT |
                VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT))
      return false;

   if (vk_format_get_plane_count(vk_format) != 1 ||
       vk_format_get_ycbcr_info(vk_format) != NULL)
      return false;

   enum pipe_format format = vk_format_to_pipe_format(vk_format);
   if (format == PIPE_FORMAT_NONE)
      return false;

   /* Metal crashes creating a placement sparse Depth32Float_Stencil8 texture
    * and has no sparse stencil. */
   const struct util_format_description *desc = util_format_description(format);
   if (util_format_has_stencil(desc))
      return false;

   const struct kk_va_format *va = kk_get_va_format(format);
   if (va == NULL || va->mtl_pixel_format == 0)
      return false;

   /* Unmapped tiles read 0, with alpha 1 for Metal formats without alpha: a
    * zero-filled image, unless the view swizzle moves a color channel into
    * alpha. */
   if (va->unswizzle.alpha != PIPE_SWIZZLE_W &&
       va->unswizzle.alpha != PIPE_SWIZZLE_1)
      return false;

   VkExtent3D standard;
   if (!kk_standard_2d_block_shape(format, &standard))
      return false;

   struct kk_image_layout layout = {
      .type = MTL_TEXTURE_TYPE_2D,
      .sample_count_sa = 1,
      .format.pipe = format,
      .format.mtl = va->mtl_pixel_format,
   };
   struct mtl_size tile =
      mtl_placement_sparse_tile_size(pdev->mtl_dev_handle, &layout);
   if (tile.x != standard.width || tile.y != standard.height || tile.z != 1)
      return false;

   if (tile_out)
      *tile_out = standard;
   return true;
}

VkResult
kk_image_sparse_init(struct kk_device *dev, struct kk_image *image)
{
   struct kk_image_plane *plane = &image->planes[0];
   struct kk_image_layout *layout = &plane->layout;
   struct kk_image_sparse *sparse = &image->sparse;

   assert(image->plane_count == 1);

   plane->mtl_handle = mtl_new_placement_sparse_texture(dev->mtl_handle, layout);
   if (plane->mtl_handle == NULL)
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);

   struct mtl_size tile = mtl_placement_sparse_tile_size(dev->mtl_handle, layout);
   sparse->tile = (VkExtent3D){tile.x, tile.y, tile.z};

   const uint32_t levels = layout->levels;
   sparse->mtl_tail_level =
      MIN2(mtl_texture_get_first_mipmap_in_tail(plane->mtl_handle), levels);
   const uint64_t tail_pages =
      sparse->mtl_tail_level < levels
         ? DIV_ROUND_UP(mtl_texture_get_tail_size_in_bytes(plane->mtl_handle),
                        KK_SPARSE_PAGE_SIZE_B)
         : 0u;

   /* The Vulkan mip tail starts at the first level smaller than a tile in any
    * dimension (no ALIGNED_MIP_SIZE). Metal's tail starts at the first level
    * below half a tile: the levels in between are mapped per tile, but only
    * through opaque binds of the Vulkan mip tail region. */
   sparse->vk_tail_level = sparse->mtl_tail_level;
   uint64_t page = 0u;
   for (uint32_t level = 0u; level < sparse->mtl_tail_level; ++level) {
      const VkExtent3D extent = {
         u_minify(layout->width_px, level),
         u_minify(layout->height_px, level),
         u_minify(layout->depth_px, level),
      };
      if (sparse->vk_tail_level == sparse->mtl_tail_level &&
          (extent.width < tile.x || extent.height < tile.y ||
           extent.depth < tile.z))
         sparse->vk_tail_level = level;

      sparse->level_tiles[level] = (VkExtent3D){
         DIV_ROUND_UP(extent.width, tile.x),
         DIV_ROUND_UP(extent.height, tile.y),
         DIV_ROUND_UP(extent.depth, tile.z),
      };
      sparse->level_page[level] = page;
      page += (uint64_t)sparse->level_tiles[level].width *
              sparse->level_tiles[level].height *
              sparse->level_tiles[level].depth;
   }
   sparse->level_page[sparse->mtl_tail_level] = page;
   sparse->layer_pages = page + tail_pages;

   return VK_SUCCESS;
}

uint64_t
kk_image_sparse_size_B(const struct kk_image *image)
{
   return image->sparse.layer_pages * image->planes[0].layout.layers *
          KK_SPARSE_PAGE_SIZE_B;
}

void
kk_image_get_sparse_memory_requirements(
   const struct kk_image *image, uint32_t *count,
   VkSparseImageMemoryRequirements2 *requirements)
{
   VK_OUTARRAY_MAKE_TYPED(VkSparseImageMemoryRequirements2, out, requirements,
                          count);

   if (!(image->vk.create_flags & VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT))
      return;

   const struct kk_image_sparse *sparse = &image->sparse;
   const uint64_t tail_page = sparse->level_page[sparse->vk_tail_level];
   vk_outarray_append_typed(VkSparseImageMemoryRequirements2, &out, req)
   {
      req->memoryRequirements = (VkSparseImageMemoryRequirements){
         .formatProperties =
            {
               .aspectMask = image->vk.aspects,
               .imageGranularity = sparse->tile,
               .flags = 0,
            },
         .imageMipTailFirstLod = sparse->vk_tail_level,
         .imageMipTailSize =
            (sparse->layer_pages - tail_page) * KK_SPARSE_PAGE_SIZE_B,
         .imageMipTailOffset = tail_page * KK_SPARSE_PAGE_SIZE_B,
         .imageMipTailStride = sparse->layer_pages * KK_SPARSE_PAGE_SIZE_B,
      };
   }
}

static mtl_heap *
kk_sparse_memory_heap(const struct kk_device_memory *mem)
{
   /* memoryTypeBits of sparse resources only allow kk_alloc_sparse_bo memory */
   assert(mem == NULL || mem->bo->mtl_handle != NULL);
   return mem ? mem->bo->mtl_handle : NULL;
}

/* Returns whether a page was unmapped */
static bool
kk_sparse_bind_buffer(struct kk_queue *queue,
                      const VkSparseBufferMemoryBindInfo *info)
{
   VK_FROM_HANDLE(kk_buffer, buffer, info->buffer);
   bool unmapped = false;

   for (uint32_t i = 0; i < info->bindCount; ++i) {
      const VkSparseMemoryBind *bind = &info->pBinds[i];
      VK_FROM_HANDLE(kk_device_memory, mem, bind->memory);

      const struct mtl_sparse_buffer_mapping mapping = {
         .unmap = mem == NULL,
         .offset = bind->resourceOffset / KK_SPARSE_PAGE_SIZE_B,
         .length = DIV_ROUND_UP(bind->size, KK_SPARSE_PAGE_SIZE_B),
         .heap_offset = mem ? bind->memoryOffset / KK_SPARSE_PAGE_SIZE_B : 0u,
      };
      mtl_command_queue_update_buffer_mappings(queue->mtl_handle,
                                               buffer->metal.handle,
                                               kk_sparse_memory_heap(mem),
                                               &mapping, 1u);
      unmapped |= mem == NULL;
   }

   return unmapped;
}

/* Metal (macOS 27.0.1, M1 Max) holds back unmaps of placement sparse buffer
 * pages until the queue maps a page of a resident resource: unmapped pages
 * kept reading and writing their old memory. Mapping the queue's resident one
 * page buffer applies them. */
VkResult
kk_queue_sparse_init(struct kk_device *dev, struct kk_queue *queue)
{
   VkResult result = kk_alloc_sparse_bo(dev, &queue->vk.base,
                                        KK_SPARSE_PAGE_SIZE_B,
                                        &queue->sparse_flush.bo);
   if (result != VK_SUCCESS)
      return result;

   queue->sparse_flush.buffer =
      mtl_new_placement_sparse_buffer(dev->mtl_handle, KK_SPARSE_PAGE_SIZE_B);
   if (queue->sparse_flush.buffer == NULL)
      return vk_error(queue, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   kk_device_add_allocation_to_residency_set(dev, queue->sparse_flush.buffer);

   return VK_SUCCESS;
}

void
kk_queue_sparse_finish(struct kk_device *dev, struct kk_queue *queue)
{
   if (queue->sparse_flush.buffer) {
      kk_device_remove_allocation_from_residency_set(
         dev, queue->sparse_flush.buffer);
      mtl_release(queue->sparse_flush.buffer);
   }
   if (queue->sparse_flush.bo)
      kk_destroy_bo(dev, queue->sparse_flush.bo);
}

static void
kk_sparse_apply_unmaps(struct kk_queue *queue)
{
   const struct mtl_sparse_buffer_mapping mapping = {
      .length = 1u,
   };
   mtl_command_queue_update_buffer_mappings(
      queue->mtl_handle, queue->sparse_flush.buffer,
      queue->sparse_flush.bo->mtl_handle, &mapping, 1u);
}

/* Page `page` of the opaque layout (see struct kk_image_sparse) as the start
 * of a Metal mapping, at most `max_pages` long and within one tile row or
 * Metal's tail. */
static struct mtl_sparse_texture_mapping
kk_sparse_opaque_page_mapping(const struct kk_image_sparse *sparse,
                              uint64_t page, uint64_t max_pages)
{
   const uint32_t layer = page / sparse->layer_pages;
   const uint64_t layer_page = page % sparse->layer_pages;
   const uint32_t tail_level = sparse->mtl_tail_level;

   if (layer_page >= sparse->level_page[tail_level]) {
      const uint64_t tail_tile = layer_page - sparse->level_page[tail_level];
      const uint64_t n = MIN2(max_pages, sparse->layer_pages - layer_page);
      return (struct mtl_sparse_texture_mapping){
         .origin = {tail_tile, 0, 0},
         .size = {n, 1, 1},
         .level = tail_level,
         .slice = layer,
      };
   }

   uint32_t level = 0u;
   while (layer_page >= sparse->level_page[level + 1])
      level++;

   const VkExtent3D tiles = sparse->level_tiles[level];
   const uint64_t tile = layer_page - sparse->level_page[level];
   const uint64_t x = tile % tiles.width;
   const uint64_t y = (tile / tiles.width) % tiles.height;
   const uint64_t z = tile / ((uint64_t)tiles.width * tiles.height);
   return (struct mtl_sparse_texture_mapping){
      .origin = {x, y, z},
      .size = {MIN2(max_pages, tiles.width - x), 1, 1},
      .level = level,
      .slice = layer,
   };
}

static void
kk_sparse_bind_image_opaque(struct kk_queue *queue,
                            const VkSparseImageOpaqueMemoryBindInfo *info,
                            struct util_dynarray *mappings)
{
   VK_FROM_HANDLE(kk_image, image, info->image);
   const struct kk_image_sparse *sparse = &image->sparse;

   for (uint32_t i = 0; i < info->bindCount; ++i) {
      const VkSparseMemoryBind *bind = &info->pBinds[i];
      VK_FROM_HANDLE(kk_device_memory, mem, bind->memory);

      const uint64_t first = bind->resourceOffset / KK_SPARSE_PAGE_SIZE_B;
      const uint64_t count = DIV_ROUND_UP(bind->size, KK_SPARSE_PAGE_SIZE_B);
      const uint64_t heap_first =
         mem ? bind->memoryOffset / KK_SPARSE_PAGE_SIZE_B : 0u;

      util_dynarray_clear(mappings);
      for (uint64_t p = 0; p < count;) {
         struct mtl_sparse_texture_mapping mapping =
            kk_sparse_opaque_page_mapping(sparse, first + p, count - p);
         mapping.unmap = mem == NULL;
         mapping.heap_offset = mem ? heap_first + p : 0u;
         util_dynarray_append(mappings, mapping);
         p += mapping.size.x;
      }

      mtl_command_queue_update_texture_mappings(
         queue->mtl_handle, image->planes[0].mtl_handle,
         kk_sparse_memory_heap(mem), util_dynarray_begin(mappings),
         util_dynarray_num_elements(mappings,
                                    struct mtl_sparse_texture_mapping));
   }
}

static void
kk_sparse_bind_image(struct kk_queue *queue,
                     const VkSparseImageMemoryBindInfo *info)
{
   VK_FROM_HANDLE(kk_image, image, info->image);
   const VkExtent3D tile = image->sparse.tile;

   for (uint32_t i = 0; i < info->bindCount; ++i) {
      const VkSparseImageMemoryBind *bind = &info->pBinds[i];
      VK_FROM_HANDLE(kk_device_memory, mem, bind->memory);

      /* Levels below the Vulkan mip tail; Metal walks the region's tiles in
       * x, y, z order through consecutive heap pages. */
      assert(bind->subresource.mipLevel < image->sparse.vk_tail_level);
      const struct mtl_sparse_texture_mapping mapping = {
         .unmap = mem == NULL,
         .origin = {bind->offset.x / tile.width, bind->offset.y / tile.height,
                    bind->offset.z / tile.depth},
         .size = {DIV_ROUND_UP(bind->extent.width, tile.width),
                  DIV_ROUND_UP(bind->extent.height, tile.height),
                  DIV_ROUND_UP(bind->extent.depth, tile.depth)},
         .level = bind->subresource.mipLevel,
         .slice = bind->subresource.arrayLayer,
         .heap_offset = mem ? bind->memoryOffset / KK_SPARSE_PAGE_SIZE_B : 0u,
      };
      mtl_command_queue_update_texture_mappings(
         queue->mtl_handle, image->planes[0].mtl_handle,
         kk_sparse_memory_heap(mem), &mapping, 1u);
   }
}

void
kk_queue_bind_sparse(struct kk_queue *queue,
                     const struct vk_queue_submit *submit)
{
   bool unmapped = false;
   for (uint32_t i = 0; i < submit->buffer_bind_count; ++i)
      unmapped |= kk_sparse_bind_buffer(queue, &submit->buffer_binds[i]);

   struct util_dynarray mappings;
   util_dynarray_init(&mappings, NULL);
   for (uint32_t i = 0; i < submit->image_opaque_bind_count; ++i)
      kk_sparse_bind_image_opaque(queue, &submit->image_opaque_binds[i],
                                  &mappings);
   util_dynarray_fini(&mappings);

   for (uint32_t i = 0; i < submit->image_bind_count; ++i)
      kk_sparse_bind_image(queue, &submit->image_binds[i]);

   if (unmapped)
      kk_sparse_apply_unmaps(queue);
}
