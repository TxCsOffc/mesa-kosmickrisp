/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef KK_QUERY_TABLE_H
#define KK_QUERY_TABLE_H 1

#include "kk_private.h"

#include "kk_bo.h"

#include "kosmickrisp/libkk/kk_query.h"

#include "util/bitset.h"
#include "util/simple_mtx.h"

struct kk_device;

struct kk_query_table {
   simple_mtx_t mutex;

   uint32_t max_alloc;  /**< Maximum possible number of queries */
   uint32_t next_query; /**< Next unallocated query */
   uint32_t free_count; /**< Size of free_table */

   struct kk_bo *bo; /**< Memoery where queries are stored */

   /* Bitset of all queries currently in use.  This is the single source
    * of truth for what is and isn't free.  The free_table and next_query are
    * simply hints to make finding a free descrptor fast.  Every free
    * query will either be above next_query or in free_table but not
    * everything which satisfies those two criteria is actually free.
    */
   BITSET_WORD *in_use;

   /* Stack for free query elements */
   uint32_t *free_table;
};

VkResult kk_query_table_init(struct kk_device *dev,
                             struct kk_query_table *table,
                             uint32_t query_count);

void kk_query_table_finish(struct kk_device *dev, struct kk_query_table *table);

VkResult kk_query_table_add(struct kk_device *dev, struct kk_query_table *table,
                            uint64_t value, uint32_t *index_out);

VkResult kk_query_table_insert(struct kk_device *dev,
                               struct kk_query_table *table, uint32_t index,
                               uint64_t value);

void kk_query_table_remove(struct kk_device *dev, struct kk_query_table *table,
                           uint32_t index);

/* Occlusion query results of a device: chunks of KK_OQ_CHUNK_QUERIES 64-bit
 * values, each a Metal visibility result buffer of its own (Metal's maximum
 * visibility query offset is 256 KB), created as pools need them. A render
 * pass writes the chunk of its active query; a query of another chunk
 * restarts the Metal render pass with that chunk (kk_flush_render_pass).
 * Pools of up to one chunk get their queries from a single chunk.
 *
 * KK_OQ_MAX_CHUNKS only sizes the chunk table libkk reads (32 KB): 4096
 * chunks are 2^27 queries and 32 GB of results, so device memory runs out
 * first. 16 chunks (524288 queries) let a device have only 128 of
 * vkd3d-proton's occlusion query pools of 4096 at a time.
 */
#define KK_OQ_MAX_CHUNKS 4096u

struct kk_oq_chunk {
   struct kk_bo *bo;
   uint32_t next;       /**< Queries below `next` were allocated before */
   uint32_t free_count; /**< Entries of `free` */
   uint32_t free[];     /**< Freed queries (chunk-local) */
};

struct kk_oq_heap {
   simple_mtx_t mutex;
   uint32_t chunk_count;
   /* Created chunks; an entry never changes once set, so recording reads it
    * without the mutex. */
   struct kk_oq_chunk *chunks[KK_OQ_MAX_CHUNKS];
   /* GPU addresses of the chunks (uint64_t each), for libkk's query kernels */
   struct kk_bo *chunk_table;
};

VkResult kk_oq_heap_init(struct kk_device *dev, struct kk_oq_heap *heap);
void kk_oq_heap_finish(struct kk_device *dev, struct kk_oq_heap *heap);
/* Allocates `count` zeroed queries into `indices` (all or none). On failure
 * `why` names the reason. */
VkResult kk_oq_heap_alloc(struct kk_device *dev, struct kk_oq_heap *heap,
                          uint32_t count, uint32_t *indices, const char **why);
void kk_oq_heap_free(struct kk_oq_heap *heap, const uint32_t *indices,
                     uint32_t count);

static inline struct kk_bo *
kk_oq_heap_chunk_bo(struct kk_oq_heap *heap, uint32_t index)
{
   return heap->chunks[index / KK_OQ_CHUNK_QUERIES]->bo;
}

#endif /* KK_QUERY_TABLE_H */
