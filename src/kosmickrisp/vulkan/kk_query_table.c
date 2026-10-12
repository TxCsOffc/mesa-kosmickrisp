/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "kk_query_table.h"

#include "kk_device.h"
#include "kk_physical_device.h"

static uint32_t query_size = sizeof(uint64_t);

static VkResult
kk_query_table_grow_locked(struct kk_device *dev, struct kk_query_table *table,
                           uint32_t new_alloc)
{
   struct kk_bo *bo;
   BITSET_WORD *new_in_use;
   uint32_t *new_free_table;
   VkResult result;

   assert(new_alloc <= table->max_alloc);

   const uint32_t new_mem_size = new_alloc * query_size;
   result = kk_alloc_bo(dev, &dev->vk.base, new_mem_size, 256, &bo);
   if (result != VK_SUCCESS)
      return result;

   /* We don't allow resize */
   assert(table->bo == NULL);
   table->bo = bo;

   assert((new_alloc % BITSET_WORDBITS) == 0);
   const size_t new_in_use_size = BITSET_BYTES(new_alloc);
   new_in_use =
      vk_realloc(&dev->vk.alloc, table->in_use, new_in_use_size,
                 sizeof(BITSET_WORD), VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (new_in_use == NULL) {
      return vk_errorf(dev, VK_ERROR_OUT_OF_HOST_MEMORY,
                       "Failed to allocate query in-use set");
   }
   memset((char *)new_in_use, 0, new_in_use_size);
   table->in_use = new_in_use;

   const size_t new_free_table_size = new_alloc * sizeof(uint32_t);
   new_free_table =
      vk_realloc(&dev->vk.alloc, table->free_table, new_free_table_size, 4,
                 VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (new_free_table == NULL) {
      return vk_errorf(dev, VK_ERROR_OUT_OF_HOST_MEMORY,
                       "Failed to allocate query free table");
   }
   table->free_table = new_free_table;

   return VK_SUCCESS;
}

VkResult
kk_query_table_init(struct kk_device *dev, struct kk_query_table *table,
                    uint32_t query_count)
{
   memset(table, 0, sizeof(*table));
   VkResult result;

   simple_mtx_init(&table->mutex, mtx_plain);

   assert(util_is_power_of_two_nonzero(query_count));

   table->max_alloc = query_count;
   table->next_query = 0;
   table->free_count = 0;

   result = kk_query_table_grow_locked(dev, table, query_count);
   if (result != VK_SUCCESS) {
      kk_query_table_finish(dev, table);
      return result;
   }

   return VK_SUCCESS;
}

void
kk_query_table_finish(struct kk_device *dev, struct kk_query_table *table)
{
   if (table->bo != NULL)
      kk_destroy_bo(dev, table->bo);
   vk_free(&dev->vk.alloc, table->in_use);
   vk_free(&dev->vk.alloc, table->free_table);
   simple_mtx_destroy(&table->mutex);
}

static VkResult
kk_query_table_alloc_locked(struct kk_device *dev, struct kk_query_table *table,
                            uint32_t *index_out)
{
   while (1) {
      uint32_t index;
      if (table->free_count > 0) {
         index = table->free_table[--table->free_count];
      } else if (table->next_query < table->max_alloc) {
         index = table->next_query++;
      } else {
         return vk_errorf(dev, VK_ERROR_OUT_OF_HOST_MEMORY,
                          "Query table not large enough");
      }

      if (!BITSET_TEST(table->in_use, index)) {
         BITSET_SET(table->in_use, index);
         *index_out = index;
         return VK_SUCCESS;
      }
   }
}

static VkResult
kk_query_table_take_locked(struct kk_device *dev, struct kk_query_table *table,
                           uint32_t index)
{
   if (index >= table->max_alloc) {
      return vk_errorf(dev, VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS,
                       "Query %u does not exist", index);
   }

   if (BITSET_TEST(table->in_use, index)) {
      return vk_errorf(dev, VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS,
                       "Query %u is already in use", index);
   } else {
      BITSET_SET(table->in_use, index);
      return VK_SUCCESS;
   }
}

static VkResult
kk_query_table_add_locked(struct kk_device *dev, struct kk_query_table *table,
                          uint64_t value, uint32_t *index_out)
{
   VkResult result = kk_query_table_alloc_locked(dev, table, index_out);
   if (result != VK_SUCCESS)
      return result;

   uint64_t *map = (uint64_t *)table->bo->cpu + *index_out;
   *map = value;

   return VK_SUCCESS;
}

VkResult
kk_query_table_add(struct kk_device *dev, struct kk_query_table *table,
                   uint64_t value, uint32_t *index_out)
{
   simple_mtx_lock(&table->mutex);
   VkResult result = kk_query_table_add_locked(dev, table, value, index_out);
   simple_mtx_unlock(&table->mutex);

   return result;
}

static VkResult
kk_query_table_insert_locked(struct kk_device *dev,
                             struct kk_query_table *table, uint32_t index,
                             uint64_t value)
{
   VkResult result = kk_query_table_take_locked(dev, table, index);
   if (result != VK_SUCCESS)
      return result;

   uint64_t *map = (uint64_t *)table->bo->cpu + index;
   *map = value;

   return result;
}

VkResult
kk_query_table_insert(struct kk_device *dev, struct kk_query_table *table,
                      uint32_t index, uint64_t value)
{
   simple_mtx_lock(&table->mutex);
   VkResult result = kk_query_table_insert_locked(dev, table, index, value);
   simple_mtx_unlock(&table->mutex);

   return result;
}

static int
compar_u32(const void *_a, const void *_b)
{
   const uint32_t *a = _a, *b = _b;
   return *a - *b;
}

static void
kk_query_table_compact_free_table(struct kk_query_table *table)
{
   if (table->free_count <= 1)
      return;

   qsort(table->free_table, table->free_count, sizeof(*table->free_table),
         compar_u32);

   uint32_t j = 1;
   for (uint32_t i = 1; i < table->free_count; i++) {
      if (table->free_table[i] == table->free_table[j - 1])
         continue;

      assert(table->free_table[i] > table->free_table[j - 1]);
      table->free_table[j++] = table->free_table[i];
   }

   table->free_count = j;
}

void
kk_query_table_remove(struct kk_device *dev, struct kk_query_table *table,
                      uint32_t index)
{
   simple_mtx_lock(&table->mutex);

   uint64_t *map = (uint64_t *)table->bo->cpu + index;
   *map = 0u;

   assert(BITSET_TEST(table->in_use, index));

   /* There may be duplicate entries in the free table.  For most operations,
    * this is fine as we always consult kk_query_table::in_use when
    * allocating.  However, it does mean that there's nothing preventing our
    * free table from growing larger than the memory we allocated for it.  In
    * the unlikely event that we end up with more entries than we can fit in
    * the allocated space, compact the table to ensure that the new entry
    * we're about to add fits.
    */
   if (table->free_count >= table->max_alloc)
      kk_query_table_compact_free_table(table);
   assert(table->free_count < table->max_alloc);

   BITSET_CLEAR(table->in_use, index);
   table->free_table[table->free_count++] = index;

   simple_mtx_unlock(&table->mutex);
}

static VkResult kk_oq_heap_add_chunk_locked(struct kk_device *dev,
                                            struct kk_oq_heap *heap,
                                            const char **why);

VkResult
kk_oq_heap_init(struct kk_device *dev, struct kk_oq_heap *heap)
{
   memset(heap, 0, sizeof(*heap));
   simple_mtx_init(&heap->mutex, mtx_plain);

   VkResult result =
      kk_alloc_bo(dev, &dev->vk.base, KK_OQ_MAX_CHUNKS * sizeof(uint64_t), 8,
                  &heap->chunk_table);
   if (result != VK_SUCCESS) {
      simple_mtx_destroy(&heap->mutex);
      return result;
   }
   memset(heap->chunk_table->cpu, 0, KK_OQ_MAX_CHUNKS * sizeof(uint64_t));

   /* Render passes always have chunk 0 as visibility buffer to start with */
   const char *why;
   result = kk_oq_heap_add_chunk_locked(dev, heap, &why);
   if (result != VK_SUCCESS) {
      kk_oq_heap_finish(dev, heap);
      return result;
   }
   return VK_SUCCESS;
}

void
kk_oq_heap_finish(struct kk_device *dev, struct kk_oq_heap *heap)
{
   for (uint32_t c = 0; c < heap->chunk_count; c++) {
      kk_destroy_bo(dev, heap->chunks[c]->bo);
      vk_free(&dev->vk.alloc, heap->chunks[c]);
   }
   if (heap->chunk_table)
      kk_destroy_bo(dev, heap->chunk_table);
   simple_mtx_destroy(&heap->mutex);
}

static uint32_t
kk_oq_chunk_available(const struct kk_oq_chunk *chunk)
{
   return chunk->free_count + (KK_OQ_CHUNK_QUERIES - chunk->next);
}

static VkResult
kk_oq_heap_add_chunk_locked(struct kk_device *dev, struct kk_oq_heap *heap,
                            const char **why)
{
   STATIC_ASSERT(KK_OQ_MAX_CHUNKS == 4096 && KK_OQ_CHUNK_QUERIES == 32768);
   if (heap->chunk_count == KK_OQ_MAX_CHUNKS) {
      *why = "out of occlusion query chunks (4096 of 32768 queries)";
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   }

   struct kk_oq_chunk *chunk =
      vk_alloc(&dev->vk.alloc,
               sizeof(*chunk) + KK_OQ_CHUNK_QUERIES * sizeof(chunk->free[0]),
               8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (chunk == NULL) {
      *why = "out of host memory for an occlusion query chunk";
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }

   VkResult result =
      kk_alloc_bo(dev, &dev->vk.base, KK_OQ_CHUNK_QUERIES * sizeof(uint64_t),
                  256, &chunk->bo);
   if (result != VK_SUCCESS) {
      vk_free(&dev->vk.alloc, chunk);
      *why = "visibility result buffer allocation failed";
      return result;
   }
   chunk->next = 0;
   chunk->free_count = 0;

   ((uint64_t *)heap->chunk_table->cpu)[heap->chunk_count] = chunk->bo->gpu;
   heap->chunks[heap->chunk_count++] = chunk;
   return VK_SUCCESS;
}

static uint32_t
kk_oq_chunk_take(struct kk_oq_heap *heap, uint32_t c)
{
   struct kk_oq_chunk *chunk = heap->chunks[c];
   uint32_t local =
      chunk->free_count ? chunk->free[--chunk->free_count] : chunk->next++;
   ((uint64_t *)chunk->bo->cpu)[local] = 0;
   return c * KK_OQ_CHUNK_QUERIES + local;
}

VkResult
kk_oq_heap_alloc(struct kk_device *dev, struct kk_oq_heap *heap,
                 uint32_t count, uint32_t *indices, const char **why)
{
   VkResult result = VK_SUCCESS;

   simple_mtx_lock(&heap->mutex);

   if (count <= KK_OQ_CHUNK_QUERIES) {
      /* One chunk, so draws of one pool never restart the render pass */
      uint32_t c = 0;
      while (c < heap->chunk_count &&
             kk_oq_chunk_available(heap->chunks[c]) < count)
         c++;
      if (c == heap->chunk_count) {
         result = kk_oq_heap_add_chunk_locked(dev, heap, why);
         if (result != VK_SUCCESS)
            goto unlock;
      }
      for (uint32_t i = 0; i < count; i++)
         indices[i] = kk_oq_chunk_take(heap, c);
   } else {
      uint32_t available = 0;
      for (uint32_t c = 0; c < heap->chunk_count; c++)
         available += kk_oq_chunk_available(heap->chunks[c]);
      while (available < count) {
         result = kk_oq_heap_add_chunk_locked(dev, heap, why);
         if (result != VK_SUCCESS)
            goto unlock;
         available += KK_OQ_CHUNK_QUERIES;
      }
      uint32_t i = 0;
      for (uint32_t c = 0; i < count; c++) {
         while (i < count && kk_oq_chunk_available(heap->chunks[c]))
            indices[i++] = kk_oq_chunk_take(heap, c);
      }
   }

unlock:
   simple_mtx_unlock(&heap->mutex);
   return result;
}

void
kk_oq_heap_free(struct kk_oq_heap *heap, const uint32_t *indices,
                uint32_t count)
{
   simple_mtx_lock(&heap->mutex);
   for (uint32_t i = 0; i < count; i++) {
      struct kk_oq_chunk *chunk = heap->chunks[indices[i] / KK_OQ_CHUNK_QUERIES];
      chunk->free[chunk->free_count++] = indices[i] % KK_OQ_CHUNK_QUERIES;
   }
   simple_mtx_unlock(&heap->mutex);
}
