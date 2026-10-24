/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "mtl_residency_set.h"

#include <Metal/MTLDevice.h>
#include <Metal/MTLResidencySet.h>
#include <Metal/MTLTexture.h>

/* A non-sparse texture buffer is a view of an already resident MTLBuffer,
 * not another backing allocation. Registering every view separately fills
 * IOGPUMetalResidencySet's bounded hash table and addAllocation: aborts
 * (STEAMAC-3V/4H: 1,048,576 table slots on M1 Max).
 *
 * Do not apply this to ordinary linear textures: Metal 4 requires the base
 * texture itself to be registered for attachments. Sparse texture buffers
 * have independent mappings and no parent buffer, so remain registered too.
 */
static bool
mtl_allocation_inherits_buffer_residency(id<MTLAllocation> allocation)
{
   if (![allocation conformsToProtocol:@protocol(MTLTexture)])
      return false;

   id<MTLTexture> texture = (id<MTLTexture>)allocation;
   return texture.textureType == MTLTextureTypeTextureBuffer &&
          texture.buffer != nil;
}

mtl_residency_set *
mtl_new_residency_set(mtl_device *device)
{
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;
      MTLResidencySetDescriptor *setDescriptor = [[[MTLResidencySetDescriptor alloc] init] autorelease];
      setDescriptor.initialCapacity = 100;
      NSError *error;
      id<MTLResidencySet> set = [dev newResidencySetWithDescriptor:setDescriptor
                                                             error:&error];

      if (error != nil) {
         fprintf(stderr, "Failed to create MTLResidencySet: %s\n", [error.localizedDescription UTF8String]);
      }

      return set;
   }
}

void
mtl_residency_set_add_allocation(mtl_residency_set *residency_set,
                                 mtl_allocation *allocation)
{
   @autoreleasepool {
      id<MTLResidencySet> set = (id<MTLResidencySet>)residency_set;
      id<MTLAllocation> alloc = (id<MTLAllocation>)allocation;
      if (mtl_allocation_inherits_buffer_residency(alloc))
         return;
      [set addAllocation:alloc];
   }
}

void
mtl_residency_set_remove_allocation(mtl_residency_set *residency_set,
                                    mtl_allocation *allocation)
{
   @autoreleasepool {
      id<MTLResidencySet> set = (id<MTLResidencySet>)residency_set;
      id<MTLAllocation> alloc = (id<MTLAllocation>)allocation;
      if (mtl_allocation_inherits_buffer_residency(alloc))
         return;
      [set removeAllocation:alloc];
   }
}

void
mtl_residency_set_commit(mtl_residency_set *residency_set)
{
   @autoreleasepool {
      id<MTLResidencySet> set = (id<MTLResidencySet>)residency_set;
      [set commit];
   }
}

void
mtl_residency_set_request_residency(mtl_residency_set *residency_set)
{
   @autoreleasepool {
      id<MTLResidencySet> set = (id<MTLResidencySet>)residency_set;
      [set requestResidency];
   }
}

void
mtl_residency_set_end_residency(mtl_residency_set *residency_set)
{
   @autoreleasepool {
      id<MTLResidencySet> set = (id<MTLResidencySet>)residency_set;
      [set endResidency];
   }
}
