/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef KK_BUFFER_VIEW_H
#define KK_BUFFER_VIEW_H 1

#include "kk_private.h"

#include "kosmickrisp/bridge/mtl_types.h"

#include "vk_buffer_view.h"

#include "util/list.h"

struct kk_physical_device;

VkFormatFeatureFlags2
kk_get_buffer_format_features(struct kk_physical_device *pdev, VkFormat format);

struct kk_buffer_view {
   struct vk_buffer_view vk;
   mtl_texture *mtl_texel_buffer_handle;
   uint64_t texel_buffer_gpu_id;
   /* Texels from the start of the Metal texture (aligned down) to the view
    * (struct kk_texel_buffer_descriptor). */
   uint32_t texel_offset;

   /* Views of sparse buffers are placement sparse texture buffers whose pages
    * follow the buffer's sparse_page_count pages from sparse_first_page
    * (kk_sparse.c): Metal makes no texture from a placement sparse buffer. */
   struct kk_sparse_buffer *sparse;
   struct list_head sparse_link;
   uint64_t sparse_first_page;
   uint64_t sparse_page_count;
   /* Residency texture of a sparse residency buffer and the descriptor's
    * page lookup (struct kk_texel_buffer_descriptor); 0 otherwise. */
   uint64_t residency_gpu_id;
   uint32_t residency_page_shift;
   uint32_t residency_last_page;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(kk_buffer_view, vk.base, VkBufferView,
                               VK_OBJECT_TYPE_BUFFER_VIEW)

void kk_buffer_view_set_label(struct kk_buffer_view *bview, const char *label);

#endif
