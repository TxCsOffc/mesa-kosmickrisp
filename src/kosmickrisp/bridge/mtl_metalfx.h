/*
 * Copyright 2026 Highball
 * SPDX-License-Identifier: MIT
 */
#ifndef MTL_METALFX_H
#define MTL_METALFX_H 1

#include "mtl_types.h"

#include <stdbool.h>
#include <stdint.h>

#define MTL_METALFX_IMAGE_COUNT 10

enum mtl_metalfx_type {
   MTL_METALFX_TEMPORAL = 1,
   MTL_METALFX_DENOISED = 2,
};

enum mtl_metalfx_image {
   MTL_METALFX_COLOR = 0,
   MTL_METALFX_DEPTH = 1,
   MTL_METALFX_MOTION = 2,
   MTL_METALFX_OUTPUT = 3,
   MTL_METALFX_EXPOSURE = 4,
   MTL_METALFX_DIFFUSE_ALBEDO = 5,
   MTL_METALFX_REACTIVE = 5,
   MTL_METALFX_SPECULAR_ALBEDO = 6,
   MTL_METALFX_NORMALS = 7,
   MTL_METALFX_ROUGHNESS = 8,
   MTL_METALFX_SPECULAR_HIT_DISTANCE = 9,
};

struct mtl_metalfx_scale {
   uint64_t scaler_id;
   uint32_t type;
   bool auto_exposure;
   bool reset;
   bool depth_reversed;
   bool jittered_motion;
   bool motion_display_res;
   bool has_offsets;
   uint32_t offset_x[MTL_METALFX_IMAGE_COUNT], offset_y[MTL_METALFX_IMAGE_COUNT];
   mtl_texture *textures[MTL_METALFX_IMAGE_COUNT];
   uint32_t formats[MTL_METALFX_IMAGE_COUNT];
   uint32_t input_content_width, input_content_height;
   float motion_scale_x, motion_scale_y;
   float jitter_x, jitter_y;
   float pre_exposure;
   float world_to_view[16], view_to_clip[16];
};

void *mtl_metalfx_context_create(void);
void mtl_metalfx_context_destroy(void *context);
void mtl_metalfx_release(void *context, const uint64_t *ids, uint32_t count);
bool mtl_metalfx_encode(void *context, mtl_device *device, mtl_compiler *compiler,
                        mtl_command_buffer *command_buffer,
                        const struct mtl_metalfx_scale *scale);

#endif
