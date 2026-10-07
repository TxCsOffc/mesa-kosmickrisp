/*
 * Copyright 2026 steamac
 * SPDX-License-Identifier: MIT
 */

#ifndef KK_SPARSE_H
#define KK_SPARSE_H 1

#include "kk_private.h"

#include "vulkan/vulkan_core.h"

#include <stdbool.h>

struct kk_buffer;
struct kk_buffer_view;
struct kk_device;
struct kk_image;
struct kk_image_layout;
struct kk_physical_device;
struct kk_queue;
struct vk_queue_submit;

/* Whether images of these parameters can be sparse (binding and residency):
 * Metal placement sparse textures whose 64 KiB tiles have the standard 2D
 * block shape (returned in tile_out, in texels). */
bool kk_sparse_image_supported(const struct kk_physical_device *pdev,
                               VkFormat format, VkImageType type,
                               VkSampleCountFlagBits samples,
                               VkImageTiling tiling, VkImageCreateFlags flags,
                               VkExtent3D *tile_out);

/* Creates the image's placement sparse texture and its page layout */
VkResult kk_image_sparse_init(struct kk_device *dev, struct kk_image *image);

uint64_t kk_image_sparse_size_B(const struct kk_image *image);

void kk_image_get_sparse_memory_requirements(
   const struct kk_image *image, uint32_t *count,
   VkSparseImageMemoryRequirements2 *requirements);

/* Page mappings of a sparse buffer, followed by its texel buffer views */
VkResult kk_buffer_sparse_init(struct kk_device *dev, struct kk_buffer *buffer);
void kk_buffer_sparse_finish(struct kk_device *dev, struct kk_buffer *buffer);

/* Creates the placement sparse texture buffer of a view of a sparse buffer,
 * starting at page first_page, and maps it like the buffer */
VkResult kk_buffer_view_sparse_init(struct kk_device *dev,
                                    struct kk_buffer_view *view,
                                    struct kk_buffer *buffer,
                                    uint64_t first_page,
                                    struct kk_image_layout *layout);
void kk_buffer_view_sparse_finish(struct kk_device *dev,
                                  struct kk_buffer_view *view);

/* Resources of vkQueueBindSparse on the queue (sparseBinding enabled) */
VkResult kk_queue_sparse_init(struct kk_device *dev, struct kk_queue *queue);
void kk_queue_sparse_finish(struct kk_device *dev, struct kk_queue *queue);

/* vkQueueBindSparse binds of a submit, as Metal 4 queue mapping updates */
void kk_queue_bind_sparse(struct kk_queue *queue,
                          const struct vk_queue_submit *submit);

#endif /* KK_SPARSE_H */
