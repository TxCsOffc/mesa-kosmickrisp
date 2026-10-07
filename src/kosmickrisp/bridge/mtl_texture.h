/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef MTL_TEXTURE_H
#define MTL_TEXTURE_H 1

#include "mtl_types.h"

#include <inttypes.h>

/* TODO_KOSMICKRISP Move this to bridge. */
struct kk_view_layout;

/* Utils*/
uint64_t mtl_texture_get_gpu_resource_id(mtl_texture *texture);
/* The texture a view was made from (parentTexture, recursively), else the
 * texture itself. Not retained. Metal 4 validation accepts an attachment only
 * when this base texture is in a residency set; the heap it was placed in or
 * the buffer it was made from do not count. */
mtl_texture *mtl_texture_get_base(mtl_texture *texture);

/* Texture view creation */
mtl_texture *mtl_new_texture_view_with(mtl_texture *texture,
                                       const struct kk_view_layout *layout);
mtl_texture *
mtl_new_texture_view_with_no_swizzle(mtl_texture *texture,
                                     const struct kk_view_layout *layout);

void mtl_texture_get_bytes(mtl_texture *texture, void *host_ptr,
                           struct mtl_texture_memory_copy *data);

void mtl_texture_replace_region(mtl_texture *texture, const void *host_ptr,
                                struct mtl_texture_memory_copy *data);

/* Placement sparse textures */
uint32_t mtl_texture_get_first_mipmap_in_tail(mtl_texture *texture);
uint64_t mtl_texture_get_tail_size_in_bytes(mtl_texture *texture);

#endif /* MTL_TEXTURE_H */
