/*
 * Copyright © 2024 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */
#ifndef KK_DESCRIPTOR_TYPES
#define KK_DESCRIPTOR_TYPES 1

#include "kk_private.h"

/* TODO_KOSMICKRISP Reduce size to 32 bytes by moving border to a heap. */
struct kk_sampled_image_descriptor {
   uint64_t image_gpu_resource_id;
   uint16_t sampler_index;

   /* Negative if there is no border color, else the clamp=0 sampler index used
    * for custom border color emulation.
    */
   int16_t clamp_0_sampler_index_or_negative;

   uint16_t sampler_lod_bias_fp16;
   uint16_t sampler_lod_min_fp16;
   uint16_t sampler_lod_max_fp16;

   uint16_t image_min_lod_fp16;
   uint16_t image_min_lod_uint16;

   uint16_t sampler_filter_flags;
   uint32_t border[4];

   /* Views with baseMipLevel > 0 of sparse residency images: Metal reports
    * the residency of such a view's level (or LOD) as that of the image's
    * level (LOD), without the base (the data is right). Sparse operations
    * take their residency from residency_gpu_resource_id, a view of the
    * same levels and all levels below them, at level (LOD) + base
    * (residency_base_level), sampling with sampler_residency_index, the
    * sampler without LOD clamps (sampler_index if its clamps can't matter;
    * the shader applies them), gathers as bilinear samples of level base
    * with sampler_gather_index (LINEAR, no clamps).
    * residency_gpu_resource_id is 0 for other views.
    */
   uint64_t residency_gpu_resource_id;
   uint16_t residency_base_level;
   uint16_t sampler_residency_index;
   uint16_t sampler_gather_index;
   uint16_t pad16;
   uint64_t pad64;
};

static_assert(sizeof(struct kk_sampled_image_descriptor) == 64,
              "kk_sampled_image_descriptor has no holes");

/* Zero for weighted-average and hardware reduction samplers. */
#define KK_SAMPLER_REDUCTION_MIN (1 << 0)
#define KK_SAMPLER_REDUCTION_MAX (1 << 1)
#define KK_SAMPLER_MIN_LINEAR (1 << 2)
#define KK_SAMPLER_MAG_LINEAR (1 << 3)
#define KK_SAMPLER_MIP_LINEAR (1 << 4)
#define KK_SAMPLER_UNNORMALIZED (1 << 5)

/* residency_gpu_resource_id: as in struct kk_sampled_image_descriptor, the
 * base level is the difference of the two textures' level counts. */
struct kk_storage_image_descriptor {
   uint64_t image_gpu_resource_id;
   uint64_t residency_gpu_resource_id;
};

static_assert(sizeof(struct kk_storage_image_descriptor) == 16,
              "kk_storage_image_descriptor has no holes");

/* Uniform and storage texel buffers. Metal needs buffer-backed textures at
 * KK_MIN_TEXEL_BUFFER_ALIGNMENT, Vulkan allows views at any texel
 * (storage/uniformTexelBufferOffsetSingleTexelAlignment, required by
 * vkd3d-proton): the Metal texture starts at the aligned offset below the
 * view, and shaders add texel_offset to coordinates (and take it off sizes).
 * image_gpu_resource_id is first, like in the other image descriptors.
 *
 * Views of sparse residency buffers: Metal has no sparse read of texture
 * buffers, so sparse loads read the residency of their page from the buffer's
 * residency texture (kk_sparse.c, KK_SPARSE_RESIDENCY_*): texel x of the Metal
 * texture is in buffer page
 * min(residency_first_page + (x >> residency_page_shift), residency_last_page).
 * residency_gpu_resource_id is 0 for other views (always resident).
 *
 * Views of R32G32B32 formats (no Metal pixel format, uniform texel buffers
 * only): the Metal texture is an R32 texture buffer of three texels per
 * element, texel_offset counts those texels and has KK_TEXEL_BUFFER_RGB32 set;
 * fetches read texels 3 * x + texel_offset + {0, 1, 2} (alpha 1), size queries
 * divide by 3. Never set for views of sparse buffers.
 */
struct kk_texel_buffer_descriptor {
   uint64_t image_gpu_resource_id;
   uint32_t texel_offset;
   uint32_t residency_first_page;
   uint64_t residency_gpu_resource_id;
   uint32_t residency_page_shift;
   uint32_t residency_last_page;
};

static_assert(sizeof(struct kk_texel_buffer_descriptor) == 32,
              "kk_texel_buffer_descriptor has no holes");

#define KK_TEXEL_BUFFER_RGB32 (1u << 31)

/* The residency texture of a sparse residency buffer: an R8Uint 2D array of
 * 256x256 texel (64 KiB) tiles, one per buffer page, 64 tiles per row and
 * 64 rows per layer: page p is the tile at texel
 * ((p % 64) * 256, (p / 64 % 64) * 256) of layer p / 4096. */
#define KK_SPARSE_RESIDENCY_TILE_PX_LOG2     8u
#define KK_SPARSE_RESIDENCY_ROW_TILES_LOG2   6u
#define KK_SPARSE_RESIDENCY_LAYER_TILES_LOG2 12u

/* This has to match nir_address_format_64bit_bounded_global */
struct kk_buffer_address {
   uint64_t base_addr;
   uint32_t size;
   uint32_t zero; /* Must be zero! */
};

static_assert(sizeof(struct kk_buffer_address) == 16,
              "kk_buffer_address has no holes");

#endif /* KK_DESCRIPTOR_TYPES */
