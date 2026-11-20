/*
 * Copyright 2026 steamac
 * SPDX-License-Identifier: MIT
 */

#include "kk_sparse.h"

#include "kk_bo.h"
#include "kk_buffer.h"
#include "kk_buffer_view.h"
#include "kk_descriptor_types.h"
#include "kk_device.h"
#include "kk_device_memory.h"
#include "kk_format.h"
#include "kk_image.h"
#include "kk_physical_device.h"
#include "kk_queue.h"

#include "kosmickrisp/bridge/mtl_bridge.h"

#include "util/format/u_format.h"
#include "util/list.h"
#include "util/simple_mtx.h"
#include "util/u_atomic.h"
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

   /* Metal (macOS 27.3, M5) gives the first three levels of a placement
    * sparse block-compressed texture overlapping storage when the base level
    * is one block wide: whichever of them is written last is what all of them
    * hold, and the others read back wrong or zero. Such a texture is tiny, so
    * it gets an ordinary texture with its own storage. It still presents one
    * page of mip tail to bind: binding it does nothing, and unbinding it
    * zeroes the texture, as sparseResidencyNonResidentStrict promises. What
    * it cannot do is share that page's contents with another image bound to
    * the same memory (sparseResidencyAliased).
    * HB_KK_SPARSE_SMALL_BC=1 turns this off. */
   const uint32_t block_w = util_format_get_blockwidth(layout->format.pipe);
   const char *keep_sparse = getenv("HB_KK_SPARSE_SMALL_BC");
   const char *all_backed = getenv("HB_KK_SPARSE_ALL_BACKED");
   sparse->backed = (all_backed && all_backed[0] == '1') ||
                    (block_w > 1u && layout->width_px <= block_w &&
                    layout->depth_px == 1u &&
                    util_format_get_2d_size(layout->format.pipe, 1u,
                                            layout->height_px) <=
                       KK_SPARSE_PAGE_SIZE_B &&
                    !(keep_sparse && keep_sparse[0] == '1'));
   if (all_backed && all_backed[0] == '1')
      fprintf(stderr, "kk sparse: backed %ux%ux%u levels %u layers %u format %s\n",
              layout->width_px, layout->height_px, layout->depth_px,
              layout->levels, layout->layers,
              util_format_short_name(layout->format.pipe));

   plane->mtl_handle =
      sparse->backed
         ? mtl_new_private_texture(dev->mtl_handle, layout)
         : mtl_new_placement_sparse_texture(dev->mtl_handle, layout);
   if (plane->mtl_handle == NULL)
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);

   struct mtl_size tile = mtl_placement_sparse_tile_size(dev->mtl_handle, layout);
   sparse->tile = (VkExtent3D){tile.x, tile.y, tile.z};

   const uint32_t levels = layout->levels;
   sparse->mtl_tail_level =
      sparse->backed
         ? 0u
         : MIN2(mtl_texture_get_first_mipmap_in_tail(plane->mtl_handle),
                levels);
   const uint64_t tail_pages =
      sparse->backed
         ? 1u
         : sparse->mtl_tail_level < levels
              ? DIV_ROUND_UP(
                   mtl_texture_get_tail_size_in_bytes(plane->mtl_handle),
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

/* Mapping updates on the Metal queue. Metal (macOS 27.0.1, M1 Max) crashes in
 * MTL4CommandQueue updateTextureMappings (a NULL dereference in the AGX
 * compute context) when a mapping update of a texture follows one of a buffer
 * or texture buffer that followed one of a texture, with no other operation
 * on the queue in between. An event signal in between avoids the crash, but
 * alone it let later buffer mapping updates overtake earlier ones (vkd3d-
 * proton's remap stress lost unmaps); the queue also waits for it. */
static void
kk_sparse_update_texture(struct kk_queue *queue, mtl_texture *texture,
                         mtl_heap *heap,
                         const struct mtl_sparse_texture_mapping *mappings,
                         uint32_t count, bool writable)
{
   simple_mtx_lock(&queue->sparse.lock);
   if (!writable && queue->sparse.buffer_after_texture) {
      mtl_signal_event(queue->mtl_handle, queue->sparse.separator,
                       ++queue->sparse.separator_value);
      mtl_wait_for_event(queue->mtl_handle, queue->sparse.separator,
                         queue->sparse.separator_value);
      queue->sparse.buffer_after_texture = false;
   }
   if (writable)
      queue->sparse.buffer_after_texture |= queue->sparse.texture_mapped;
   else
      queue->sparse.texture_mapped = true;
   mtl_command_queue_update_texture_mappings(queue->mtl_handle, texture, heap,
                                             mappings, count);
   simple_mtx_unlock(&queue->sparse.lock);
}

static void
kk_sparse_update_texture_buffer(
   struct kk_queue *queue, mtl_texture *texture, mtl_heap *heap,
   const struct mtl_sparse_texture_mapping *mappings, uint32_t count)
{
   simple_mtx_lock(&queue->sparse.lock);
   queue->sparse.buffer_after_texture |= queue->sparse.texture_mapped;
   mtl_command_queue_update_texture_mappings(queue->mtl_handle, texture, heap,
                                             mappings, count);
   simple_mtx_unlock(&queue->sparse.lock);
}

static void
kk_sparse_update_buffer(struct kk_queue *queue, mtl_buffer *buffer,
                        mtl_heap *heap,
                        const struct mtl_sparse_buffer_mapping *mappings,
                        uint32_t count)
{
   simple_mtx_lock(&queue->sparse.lock);
   queue->sparse.buffer_after_texture |= queue->sparse.texture_mapped;
   mtl_command_queue_update_buffer_mappings(queue->mtl_handle, buffer, heap,
                                            mappings, count);
   simple_mtx_unlock(&queue->sparse.lock);
}

/* What vkQueueBindSparse mapped to each page of a sparse buffer, so its texel
 * buffer views (placement sparse texture buffers: Metal makes no texture from
 * a placement sparse buffer, newTextureWithDescriptor returns nil) can be
 * mapped alike when they are created and on every later bind. Their 1D tiles
 * are the buffer's 64 KiB pages, linearly. Mapped heaps are retained: memory
 * may be freed while pages still map it (vkd3d-proton does), and a later view
 * maps those pages too. Shared by the buffer and its views, any of which may
 * be destroyed last.
 *
 * Sparse residency buffers also have a residency texture for sparse loads
 * through their views (Metal has no sparse read of texture buffers): an
 * R8Uint placement sparse 2D array whose 64 KiB tiles are the buffer's pages
 * (KK_SPARSE_RESIDENCY_*), mapped to the same heap pages by the same queue
 * mapping updates, so a sparse_read of a page's tile has the page's
 * residency, ordered with the page's data. Its texels alias the page's bytes
 * and are never used. */
struct kk_sparse_buffer_page {
   mtl_heap *heap; /* NULL: unmapped */
   uint64_t heap_page;
};

struct kk_sparse_buffer {
   uint32_t refcount;
   /* Orders page updates and the views' Metal mapping updates */
   simple_mtx_t lock;
   struct list_head views;
   mtl_texture *residency;
   uint64_t residency_gpu_id;
   uint64_t page_count;
   struct kk_sparse_buffer_page pages[];
};

#define KK_SPARSE_RESIDENCY_ROW_TILES   (1u << KK_SPARSE_RESIDENCY_ROW_TILES_LOG2)
#define KK_SPARSE_RESIDENCY_LAYER_TILES (1u << KK_SPARSE_RESIDENCY_LAYER_TILES_LOG2)

static VkResult
kk_sparse_buffer_residency_init(struct kk_device *dev,
                                struct kk_sparse_buffer *sparse)
{
   const uint64_t pages = sparse->page_count;
   const uint32_t tile_px = 1u << KK_SPARSE_RESIDENCY_TILE_PX_LOG2;
   const uint64_t rows = DIV_ROUND_UP(MIN2(pages, KK_SPARSE_RESIDENCY_LAYER_TILES),
                                      KK_SPARSE_RESIDENCY_ROW_TILES);
   const uint64_t layers = DIV_ROUND_UP(pages, KK_SPARSE_RESIDENCY_LAYER_TILES);
   if (layers > 2048u)
      return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                       "sparse residency buffer of %" PRIu64 " pages",
                       pages);

   const struct kk_va_format *va = kk_get_va_format(PIPE_FORMAT_R8_UINT);
   struct kk_image_layout layout = {
      .width_px = MIN2(pages, KK_SPARSE_RESIDENCY_ROW_TILES) * tile_px,
      .height_px = rows * tile_px,
      .depth_px = 1u,
      .layers = layers,
      .type = MTL_TEXTURE_TYPE_2D_ARRAY,
      .sample_count_sa = 1u,
      .levels = 1u,
      .usage = MTL_TEXTURE_USAGE_SHADER_READ,
      .format = {.pipe = PIPE_FORMAT_R8_UINT, .mtl = va->mtl_pixel_format},
   };
   struct mtl_size tile = mtl_placement_sparse_tile_size(dev->mtl_handle, &layout);
   if (tile.x != tile_px || tile.y != tile_px || tile.z != 1u)
      return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                       "R8Uint sparse tiles are %ux%ux%u, not 256x256x1",
                       (unsigned)tile.x, (unsigned)tile.y, (unsigned)tile.z);

   sparse->residency = mtl_new_placement_sparse_texture(dev->mtl_handle, &layout);
   if (sparse->residency == NULL)
      return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                       "Metal failed to create the residency texture of a "
                       "sparse buffer of %" PRIu64 " pages",
                       pages);
   kk_device_add_allocation_to_residency_set(dev, sparse->residency);
   sparse->residency_gpu_id = mtl_texture_get_gpu_resource_id(sparse->residency);
   return VK_SUCCESS;
}

/* Maps (heap) or unmaps (NULL) the residency texture tiles of count buffer
 * pages from page first, to heap pages from heap_first */
static void
kk_sparse_buffer_map_residency(struct kk_queue *queue,
                               struct kk_sparse_buffer *sparse, mtl_heap *heap,
                               uint64_t first, uint64_t count,
                               uint64_t heap_first)
{
   if (sparse->residency == NULL)
      return;

   struct mtl_sparse_texture_mapping mappings[16];
   uint32_t n = 0u;
   for (uint64_t p = first; p < first + count;) {
      const uint64_t x = p % KK_SPARSE_RESIDENCY_ROW_TILES;
      const uint64_t len =
         MIN2(KK_SPARSE_RESIDENCY_ROW_TILES - x, first + count - p);
      mappings[n++] = (struct mtl_sparse_texture_mapping){
         .unmap = heap == NULL,
         .origin = {x, (p / KK_SPARSE_RESIDENCY_ROW_TILES) %
                          (KK_SPARSE_RESIDENCY_LAYER_TILES /
                           KK_SPARSE_RESIDENCY_ROW_TILES),
                    0},
         .size = {len, 1, 1},
         .slice = p / KK_SPARSE_RESIDENCY_LAYER_TILES,
         .heap_offset = heap ? heap_first + (p - first) : 0u,
      };
      p += len;
      if (n == ARRAY_SIZE(mappings) || p == first + count) {
         kk_sparse_update_texture(queue, sparse->residency, heap, mappings, n,
                                   false);
         n = 0u;
      }
   }
}

VkResult
kk_buffer_sparse_init(struct kk_device *dev, struct kk_buffer *buffer)
{
   const uint64_t page_count =
      DIV_ROUND_UP(buffer->vk.size, KK_SPARSE_PAGE_SIZE_B);
   struct kk_sparse_buffer *sparse = vk_zalloc(
      &dev->vk.alloc,
      sizeof(*sparse) + page_count * sizeof(struct kk_sparse_buffer_page), 8,
      VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (sparse == NULL)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   sparse->refcount = 1u;
   simple_mtx_init(&sparse->lock, mtx_plain);
   list_inithead(&sparse->views);
   sparse->page_count = page_count;
   if (buffer->vk.create_flags & VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT) {
      VkResult result = kk_sparse_buffer_residency_init(dev, sparse);
      if (result != VK_SUCCESS) {
         simple_mtx_destroy(&sparse->lock);
         vk_free(&dev->vk.alloc, sparse);
         return result;
      }
   }
   buffer->sparse = sparse;
   return VK_SUCCESS;
}

static void
kk_sparse_buffer_unref(struct kk_device *dev, struct kk_sparse_buffer *sparse)
{
   if (!p_atomic_dec_zero(&sparse->refcount))
      return;

   assert(list_is_empty(&sparse->views));
   for (uint64_t p = 0; p < sparse->page_count; ++p) {
      if (sparse->pages[p].heap)
         mtl_release(sparse->pages[p].heap);
   }
   if (sparse->residency) {
      kk_device_remove_allocation_from_residency_set(dev, sparse->residency);
      mtl_release(sparse->residency);
   }
   simple_mtx_destroy(&sparse->lock);
   vk_free(&dev->vk.alloc, sparse);
}

void
kk_buffer_sparse_finish(struct kk_device *dev, struct kk_buffer *buffer)
{
   if (buffer->sparse)
      kk_sparse_buffer_unref(dev, buffer->sparse);
}

VkResult
kk_buffer_view_sparse_init(struct kk_device *dev, struct kk_buffer_view *view,
                           struct kk_buffer *buffer, uint64_t first_page,
                           struct kk_image_layout *layout)
{
   struct kk_sparse_buffer *sparse = buffer->sparse;
   const uint64_t texel_B = util_format_get_blocksize(layout->format.pipe);

   view->mtl_texel_buffer_handle =
      mtl_new_placement_sparse_texture(dev->mtl_handle, layout);
   if (view->mtl_texel_buffer_handle == NULL)
      return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                       "Metal failed to create a placement sparse texture "
                       "buffer of %u texels",
                       layout->width_px);
   kk_device_add_allocation_to_residency_set(dev,
                                             view->mtl_texel_buffer_handle);

   view->sparse_first_page = first_page;
   view->sparse_page_count =
      MIN2(DIV_ROUND_UP(layout->width_px * texel_B, KK_SPARSE_PAGE_SIZE_B),
           sparse->page_count - first_page);
   if (sparse->residency) {
      /* Views of sparse buffers keep texels within pages (kk_buffer_view.c):
       * Metal's texture buffer formats are 1 to 16 bytes, powers of two. */
      assert(util_is_power_of_two_nonzero64(texel_B));
      view->residency_gpu_id = sparse->residency_gpu_id;
      view->residency_page_shift =
         util_logbase2(KK_SPARSE_PAGE_SIZE_B) - util_logbase2_64(texel_B);
      view->residency_last_page = sparse->page_count - 1u;
   }

   simple_mtx_lock(&sparse->lock);
   /* Map what the buffer maps, after its earlier mapping updates (the same
    * queue); later binds update the view too. */
   if (dev->has_queue) {
      const struct kk_sparse_buffer_page *pages = &sparse->pages[first_page];
      for (uint64_t p = 0; p < view->sparse_page_count;) {
         uint64_t n = 1u;
         if (pages[p].heap == NULL) {
            p++;
            continue;
         }
         while (p + n < view->sparse_page_count &&
                pages[p + n].heap == pages[p].heap &&
                pages[p + n].heap_page == pages[p].heap_page + n)
            n++;
         const struct mtl_sparse_texture_mapping mapping = {
            .origin = {p, 0, 0},
            .size = {n, 1, 1},
            .heap_offset = pages[p].heap_page,
         };
         kk_sparse_update_texture_buffer(&dev->queue,
                                         view->mtl_texel_buffer_handle,
                                         pages[p].heap, &mapping, 1u);
         p += n;
      }
   }
   list_addtail(&view->sparse_link, &sparse->views);
   p_atomic_inc(&sparse->refcount);
   view->sparse = sparse;
   simple_mtx_unlock(&sparse->lock);

   return VK_SUCCESS;
}

void
kk_buffer_view_sparse_finish(struct kk_device *dev, struct kk_buffer_view *view)
{
   struct kk_sparse_buffer *sparse = view->sparse;

   simple_mtx_lock(&sparse->lock);
   list_del(&view->sparse_link);
   simple_mtx_unlock(&sparse->lock);

   kk_device_remove_allocation_from_residency_set(
      dev, view->mtl_texel_buffer_handle);
   kk_sparse_buffer_unref(dev, sparse);
}

/* Buffer binds of a submit go in passes, each over every bind of the submit:
 * the buffer's pages (recorded), its texel buffer views, then its residency
 * texture, so the texture mapping updates of a submit need at most one
 * separator (kk_sparse_update_texture). Heaps the pages no longer map are
 * appended to `released` and released after the submit's last mapping update:
 * the residency texture and the views map them until their passes (memory
 * may already be freed). Returns whether a page was unmapped. */
static bool
kk_sparse_bind_buffer_pages(struct kk_queue *queue,
                            const VkSparseBufferMemoryBindInfo *info,
                            struct util_dynarray *released)
{
   VK_FROM_HANDLE(kk_buffer, buffer, info->buffer);
   struct kk_sparse_buffer *sparse = buffer->sparse;
   bool unmapped = false;

   for (uint32_t i = 0; i < info->bindCount; ++i) {
      const VkSparseMemoryBind *bind = &info->pBinds[i];
      VK_FROM_HANDLE(kk_device_memory, mem, bind->memory);
      mtl_heap *heap = kk_sparse_memory_heap(mem);

      const uint64_t first = bind->resourceOffset / KK_SPARSE_PAGE_SIZE_B;
      const uint64_t count = DIV_ROUND_UP(bind->size, KK_SPARSE_PAGE_SIZE_B);
      const uint64_t heap_first =
         mem ? bind->memoryOffset / KK_SPARSE_PAGE_SIZE_B : 0u;
      assert(first + count <= sparse->page_count);

      simple_mtx_lock(&sparse->lock);
      for (uint64_t p = 0; p < count; ++p) {
         struct kk_sparse_buffer_page *page = &sparse->pages[first + p];
         if (heap)
            mtl_retain(heap);
         if (page->heap)
            util_dynarray_append(released, page->heap);
         *page = (struct kk_sparse_buffer_page){
            .heap = heap,
            .heap_page = heap ? heap_first + p : 0u,
         };
      }

      const struct mtl_sparse_buffer_mapping mapping = {
         .unmap = heap == NULL,
         .offset = first,
         .length = count,
         .heap_offset = heap_first,
      };
      kk_sparse_update_buffer(queue, buffer->metal.handle, heap, &mapping, 1u);
      simple_mtx_unlock(&sparse->lock);

      unmapped |= heap == NULL;
   }

   return unmapped;
}

static void
kk_sparse_bind_buffer_textures(struct kk_queue *queue,
                               const VkSparseBufferMemoryBindInfo *info,
                               bool residency)
{
   VK_FROM_HANDLE(kk_buffer, buffer, info->buffer);
   struct kk_sparse_buffer *sparse = buffer->sparse;

   for (uint32_t i = 0; i < info->bindCount; ++i) {
      const VkSparseMemoryBind *bind = &info->pBinds[i];
      VK_FROM_HANDLE(kk_device_memory, mem, bind->memory);
      mtl_heap *heap = kk_sparse_memory_heap(mem);

      const uint64_t first = bind->resourceOffset / KK_SPARSE_PAGE_SIZE_B;
      const uint64_t count = DIV_ROUND_UP(bind->size, KK_SPARSE_PAGE_SIZE_B);
      const uint64_t heap_first =
         mem ? bind->memoryOffset / KK_SPARSE_PAGE_SIZE_B : 0u;

      simple_mtx_lock(&sparse->lock);
      if (residency) {
         kk_sparse_buffer_map_residency(queue, sparse, heap, first, count,
                                        heap_first);
         simple_mtx_unlock(&sparse->lock);
         continue;
      }

      list_for_each_entry(struct kk_buffer_view, view, &sparse->views,
                          sparse_link) {
         const uint64_t start = MAX2(first, view->sparse_first_page);
         const uint64_t end =
            MIN2(first + count,
                 view->sparse_first_page + view->sparse_page_count);
         if (start >= end)
            continue;

         const struct mtl_sparse_texture_mapping view_mapping = {
            .unmap = heap == NULL,
            .origin = {start - view->sparse_first_page, 0, 0},
            .size = {end - start, 1, 1},
            .heap_offset = heap ? heap_first + (start - first) : 0u,
         };
         kk_sparse_update_texture_buffer(
            queue, view->mtl_texel_buffer_handle, heap, &view_mapping, 1u);
      }
      simple_mtx_unlock(&sparse->lock);
   }
}

/* Metal (macOS 27.0.1, M1 Max) holds back unmaps of placement sparse buffer
 * pages until the queue maps a page of a resident resource: unmapped pages
 * kept reading and writing their old memory. Mapping the queue's resident one
 * page buffer applies them. */
VkResult
kk_queue_sparse_init(struct kk_device *dev, struct kk_queue *queue)
{
   simple_mtx_init(&queue->sparse.lock, mtx_plain);

   queue->sparse.separator = mtl_new_event(dev->mtl_handle);
   if (queue->sparse.separator == NULL)
      return vk_error(queue, VK_ERROR_OUT_OF_DEVICE_MEMORY);

   VkResult result = kk_alloc_sparse_bo(dev, &queue->vk.base,
                                        KK_SPARSE_PAGE_SIZE_B,
                                        &queue->sparse.flush_bo);
   if (result != VK_SUCCESS)
      return result;

   queue->sparse.flush_buffer =
      mtl_new_placement_sparse_buffer(dev->mtl_handle, KK_SPARSE_PAGE_SIZE_B);
   if (queue->sparse.flush_buffer == NULL)
      return vk_error(queue, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   kk_device_add_allocation_to_residency_set(dev, queue->sparse.flush_buffer);

   return VK_SUCCESS;
}

void
kk_queue_sparse_finish(struct kk_device *dev, struct kk_queue *queue)
{
   if (queue->sparse.zero_buffer) {
      kk_device_remove_allocation_from_residency_set(
         dev, queue->sparse.zero_buffer);
      mtl_release(queue->sparse.zero_buffer);
   }
   if (queue->sparse.zero_allocator)
      mtl_release(queue->sparse.zero_allocator);
   if (queue->sparse.flush_buffer) {
      kk_device_remove_allocation_from_residency_set(
         dev, queue->sparse.flush_buffer);
      mtl_release(queue->sparse.flush_buffer);
   }
   if (queue->sparse.flush_bo)
      kk_destroy_bo(dev, queue->sparse.flush_bo);
   if (queue->sparse.separator)
      mtl_release(queue->sparse.separator);
   if (dev->vk.enabled_features.sparseBinding)
      simple_mtx_destroy(&queue->sparse.lock);
}

/* Metal (macOS 27.0.1, M1 Max) holds back unmaps of placement sparse buffer
 * pages until the queue maps a page of a resident resource: unmapped pages
 * kept reading and writing their old memory. Mapping the queue's resident one
 * page buffer applies them. */
static void
kk_sparse_apply_unmaps(struct kk_queue *queue)
{
   const struct mtl_sparse_buffer_mapping mapping = {
      .length = 1u,
   };
   kk_sparse_update_buffer(queue, queue->sparse.flush_buffer,
                           queue->sparse.flush_bo->mtl_handle, &mapping, 1u);
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

/* Zeroes every level of a backed sparse image (kk_image_sparse_init) with
 * copies from a zeroed buffer, in order with the queue's other work. */
static void
kk_sparse_zero_backed_image(struct kk_queue *queue, struct kk_image *image)
{
   struct kk_device *dev = kk_queue_device(queue);
   const struct kk_image_layout *layout = &image->planes[0].layout;
   const enum pipe_format format = layout->format.pipe;

   simple_mtx_lock(&queue->sparse.lock);
   if (queue->sparse.zero_buffer == NULL) {
      queue->sparse.zero_buffer =
         mtl_new_private_buffer(dev->mtl_handle, KK_SPARSE_PAGE_SIZE_B);
      queue->sparse.zero_allocator = mtl_new_command_allocator(dev->mtl_handle);
      if (queue->sparse.zero_buffer)
         kk_device_add_allocation_to_residency_set(dev,
                                                   queue->sparse.zero_buffer);
      kk_device_make_resources_resident(dev);
   }
   if (queue->sparse.zero_buffer == NULL ||
       queue->sparse.zero_allocator == NULL) {
      simple_mtx_unlock(&queue->sparse.lock);
      return;
   }

   mtl_command_buffer *cmd_buf = mtl_new_command_buffer(dev->mtl_handle);
   mtl_begin_command_buffer(cmd_buf, queue->sparse.zero_allocator);
   mtl_compute_encoder *encoder = mtl_new_compute_command_encoder(cmd_buf);
   for (uint32_t layer = 0u; layer < layout->layers; ++layer) {
      for (uint32_t level = 0u; level < layout->levels; ++level) {
         const uint32_t width = u_minify(layout->width_px, level);
         const uint32_t height = u_minify(layout->height_px, level);
         const size_t stride_B = util_format_get_stride(format, width);
         struct mtl_buffer_image_copy copy = {
            .image_size = {width, height, 1},
            .buffer = queue->sparse.zero_buffer,
            .image = image->planes[0].mtl_handle,
            .buffer_stride_B = stride_B,
            .buffer_2d_image_size_B =
               stride_B * util_format_get_nblocksy(format, height),
            .image_slice = layer,
            .image_level = level,
         };
         mtl_copy_from_buffer_to_texture(encoder, &copy);
      }
   }
   mtl_barrier_after_stages(encoder, MTL_STAGE_ALL, MTL_STAGE_ALL);
   mtl_end_encoding(encoder);
   mtl_release(encoder);
   mtl_end_command_buffer(cmd_buf);

   mtl_commit_options *options = mtl_new_commit_options();
   mtl_command_queue_commit(queue->mtl_handle, &cmd_buf, 1u, options);
   mtl_release(options);
   mtl_release(cmd_buf);
   simple_mtx_unlock(&queue->sparse.lock);
}

static void
kk_sparse_bind_image_opaque(struct kk_queue *queue,
                            const VkSparseImageOpaqueMemoryBindInfo *info,
                            struct util_dynarray *mappings)
{
   VK_FROM_HANDLE(kk_image, image, info->image);
   const struct kk_image_sparse *sparse = &image->sparse;

   if (sparse->backed) {
      const char *all_backed = getenv("HB_KK_SPARSE_ALL_BACKED");
      if (all_backed && all_backed[0] == '1')
         return;
      for (uint32_t i = 0; i < info->bindCount; ++i) {
         if (info->pBinds[i].memory == VK_NULL_HANDLE) {
            kk_sparse_zero_backed_image(queue, image);
            break;
         }
      }
      return;
   }

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

      kk_sparse_update_texture(
         queue, image->planes[0].mtl_handle, kk_sparse_memory_heap(mem),
         util_dynarray_begin(mappings),
         util_dynarray_num_elements(mappings,
                                    struct mtl_sparse_texture_mapping),
         image->planes[0].layout.usage & MTL_TEXTURE_USAGE_SHADER_WRITE);
   }
}

static void
kk_sparse_bind_image(struct kk_queue *queue,
                     const VkSparseImageMemoryBindInfo *info)
{
   VK_FROM_HANDLE(kk_image, image, info->image);
   const VkExtent3D tile = image->sparse.tile;

   if (image->sparse.backed)
      return;

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
      kk_sparse_update_texture(queue, image->planes[0].mtl_handle,
                               kk_sparse_memory_heap(mem), &mapping, 1u,
                               image->planes[0].layout.usage &
                                  MTL_TEXTURE_USAGE_SHADER_WRITE);
   }
}

void
kk_queue_bind_sparse(struct kk_queue *queue,
                     const struct vk_queue_submit *submit)
{
   struct util_dynarray released;
   util_dynarray_init(&released, NULL);
   bool unmapped = false;
   for (uint32_t i = 0; i < submit->buffer_bind_count; ++i) {
      unmapped |= kk_sparse_bind_buffer_pages(queue, &submit->buffer_binds[i],
                                              &released);
   }
   for (uint32_t i = 0; i < submit->buffer_bind_count; ++i)
      kk_sparse_bind_buffer_textures(queue, &submit->buffer_binds[i], false);
   for (uint32_t i = 0; i < submit->buffer_bind_count; ++i)
      kk_sparse_bind_buffer_textures(queue, &submit->buffer_binds[i], true);

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

   util_dynarray_foreach(&released, mtl_heap *, heap)
      mtl_release(*heap);
   util_dynarray_fini(&released);
}
