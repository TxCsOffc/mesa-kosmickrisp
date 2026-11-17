/*
 * Copyright 2026 Highball
 * SPDX-License-Identifier: MIT
 */

#include "mtl_metalfx.h"

void *
mtl_metalfx_context_create(void)
{
   return NULL;
}

void
mtl_metalfx_context_destroy(void *context)
{
}

void
mtl_metalfx_release(void *context, const uint64_t *ids, uint32_t count)
{
}

bool
mtl_metalfx_encode(void *context, mtl_device *device, mtl_compiler *compiler,
                   mtl_command_buffer *command_buffer,
                   const struct mtl_metalfx_scale *scale)
{
   return false;
}
