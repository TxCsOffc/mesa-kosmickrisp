#include "mtl_acceleration_structure.h"

#include <Metal/MTL4AccelerationStructure.h>
#include <Metal/MTL4ComputeCommandEncoder.h>
#include <Metal/MTLAccelerationStructure.h>
#include <Metal/MTLDevice.h>
#include <Metal/MTLHeap.h>

static MTLAttributeFormat
mtl_as_vertex_format_to_mtl(enum mtl_as_vertex_format fmt)
{
   switch (fmt) {
   case MTL_AS_VERTEX_FORMAT_FLOAT3:
      return MTLAttributeFormatFloat3;
   case MTL_AS_VERTEX_FORMAT_FLOAT2:
      return MTLAttributeFormatFloat2;
   case MTL_AS_VERTEX_FORMAT_HALF3:
      return MTLAttributeFormatHalf3;
   case MTL_AS_VERTEX_FORMAT_HALF2:
      return MTLAttributeFormatHalf2;
   case MTL_AS_VERTEX_FORMAT_SHORT2_NORMALIZED:
      return MTLAttributeFormatShort2Normalized;
   case MTL_AS_VERTEX_FORMAT_SHORT4_NORMALIZED:
      return MTLAttributeFormatShort4Normalized;
   case MTL_AS_VERTEX_FORMAT_USHORT2_NORMALIZED:
      return MTLAttributeFormatUShort2Normalized;
   case MTL_AS_VERTEX_FORMAT_USHORT4_NORMALIZED:
      return MTLAttributeFormatUShort4Normalized;
   default:
      return MTLAttributeFormatFloat3;
   }
}

static MTL4AccelerationStructureDescriptor *
mtl_build_as_descriptor(const struct mtl_as_build_desc *desc, bool for_build)
{
   MTL4AccelerationStructureDescriptor *out = nil;

   if (desc->is_top_level) {
      MTL4InstanceAccelerationStructureDescriptor *tlas =
         [[MTL4InstanceAccelerationStructureDescriptor new] autorelease];
      tlas.instanceDescriptorType =
         MTLAccelerationStructureInstanceDescriptorTypeIndirect;
      tlas.instanceDescriptorStride =
         sizeof(MTLIndirectAccelerationStructureInstanceDescriptor);
      tlas.instanceCount = desc->instance_count;
      tlas.instanceTransformationMatrixLayout = MTLMatrixLayoutColumnMajor;
      if (for_build) {
         uint64_t addr = desc->instance_descriptor_addr
                            ? desc->instance_descriptor_addr
                            : desc->dummy_addr;
         tlas.instanceDescriptorBuffer =
            MTL4BufferRangeMake(addr, (uint64_t)-1);
      }
      out = tlas;
   } else {
      MTL4PrimitiveAccelerationStructureDescriptor *blas =
         [[MTL4PrimitiveAccelerationStructureDescriptor new] autorelease];
      NSMutableArray<MTL4AccelerationStructureGeometryDescriptor *> *geoms =
         [NSMutableArray arrayWithCapacity:desc->geometry_count ? desc->geometry_count : 1];
      uint64_t total_prims = 0;

      for (uint32_t i = 0; i < desc->geometry_count; i++) {
         const struct mtl_as_geometry *g = &desc->geometries[i];
         if (g->type == MTL_AS_GEOMETRY_TRIANGLES) {
            MTL4AccelerationStructureTriangleGeometryDescriptor *tri =
               [[MTL4AccelerationStructureTriangleGeometryDescriptor new] autorelease];
            tri.vertexFormat =
               mtl_as_vertex_format_to_mtl(g->triangles.vertex_format);
            tri.vertexStride = g->triangles.vertex_stride;
            tri.triangleCount = g->triangles.triangle_count;
            tri.intersectionFunctionTableOffset = g->geometry_index;
            tri.opaque = g->opaque;
            tri.allowDuplicateIntersectionFunctionInvocation =
               g->allow_duplicate_any_hit;
            if (for_build) {
               uint64_t vaddr = g->triangles.vertex_addr
                                   ? g->triangles.vertex_addr
                                   : desc->dummy_addr;
               tri.vertexBuffer = MTL4BufferRangeMake(vaddr, (uint64_t)-1);
               if (g->triangles.has_index_buffer && g->triangles.index_addr) {
                  tri.indexBuffer =
                     MTL4BufferRangeMake(g->triangles.index_addr, (uint64_t)-1);
                  tri.indexType = (MTLIndexType)g->triangles.index_type;
               }
               if (g->triangles.transform_addr) {
                  tri.transformationMatrixBuffer =
                     MTL4BufferRangeMake(g->triangles.transform_addr,
                                         (uint64_t)-1);
                  tri.transformationMatrixLayout = MTLMatrixLayoutRowMajor;
               }
            } else if (g->triangles.has_index_buffer) {
               tri.indexType = (MTLIndexType)g->triangles.index_type;
            }
            total_prims += g->triangles.triangle_count;
            [geoms addObject:tri];
         } else {
            MTL4AccelerationStructureBoundingBoxGeometryDescriptor *aabb =
               [[MTL4AccelerationStructureBoundingBoxGeometryDescriptor new] autorelease];
            aabb.boundingBoxStride =
               g->aabbs.aabb_stride >= 24 ? g->aabbs.aabb_stride : 24;
            aabb.boundingBoxCount = g->aabbs.aabb_count;
            aabb.intersectionFunctionTableOffset = g->geometry_index;
            aabb.opaque = g->opaque;
            aabb.allowDuplicateIntersectionFunctionInvocation =
               g->allow_duplicate_any_hit;
            if (for_build) {
               uint64_t baddr = g->aabbs.aabb_addr
                                   ? g->aabbs.aabb_addr
                                   : desc->dummy_addr;
               aabb.boundingBoxBuffer = MTL4BufferRangeMake(baddr, (uint64_t)-1);
            }
            total_prims += g->aabbs.aabb_count;
            [geoms addObject:aabb];
         }
      }

      if (geoms.count == 0) {
         MTL4AccelerationStructureTriangleGeometryDescriptor *empty =
            [[MTL4AccelerationStructureTriangleGeometryDescriptor new] autorelease];
         empty.vertexFormat = MTLAttributeFormatFloat3;
         empty.vertexStride = 12;
         empty.triangleCount = 0;
         if (for_build)
            empty.vertexBuffer =
               MTL4BufferRangeMake(desc->dummy_addr, (uint64_t)-1);
         [geoms addObject:empty];
      }

      blas.geometryDescriptors = geoms;
      out = blas;
      if (total_prims > (1ULL << 28))
         out.usage |= MTLAccelerationStructureUsageExtendedLimits;
   }

   out.usage |= (MTLAccelerationStructureUsage)desc->usage;
   return out;
}

bool
mtl_device_supports_raytracing(mtl_device *device)
{
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;
      return dev.supportsRaytracing;
   }
}

void
mtl_heap_acceleration_structure_size_and_align(mtl_device *device,
                                               uint64_t size_B,
                                               uint64_t *size_out,
                                               uint64_t *align_out)
{
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;
      MTLSizeAndAlign sa =
         [dev heapAccelerationStructureSizeAndAlignWithSize:size_B];
      if (size_out)
         *size_out = sa.size;
      if (align_out)
         *align_out = sa.align;
   }
}

bool
mtl_heap_is_private(mtl_heap *heap)
{
   @autoreleasepool {
      id<MTLHeap> hp = (id<MTLHeap>)heap;
      return hp && hp.storageMode == MTLStorageModePrivate;
   }
}

mtl_acceleration_structure *
mtl_new_acceleration_structure_from_heap(mtl_heap *heap, uint64_t size_B,
                                         uint64_t offset_B)
{
   @autoreleasepool {
      id<MTLHeap> hp = (id<MTLHeap>)heap;
      return (mtl_acceleration_structure *)[hp
         newAccelerationStructureWithSize:size_B
                                   offset:offset_B];
   }
}

mtl_acceleration_structure *
mtl_new_acceleration_structure_from_device(mtl_device *device, uint64_t size_B)
{
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;
      return (mtl_acceleration_structure *)[dev
         newAccelerationStructureWithSize:size_B];
   }
}

uint64_t
mtl_acceleration_structure_get_gpu_resource_id(mtl_acceleration_structure *as)
{
   @autoreleasepool {
      id<MTLAccelerationStructure> a = (id<MTLAccelerationStructure>)as;
      return a.gpuResourceID._impl;
   }
}

uint64_t
mtl_acceleration_structure_get_size(mtl_acceleration_structure *as)
{
   @autoreleasepool {
      id<MTLAccelerationStructure> a = (id<MTLAccelerationStructure>)as;
      return a.size;
   }
}

struct mtl_as_sizes
mtl_device_get_as_sizes(mtl_device *device,
                        const struct mtl_as_build_desc *desc)
{
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;
      MTL4AccelerationStructureDescriptor *mtl_desc =
         mtl_build_as_descriptor(desc, false);
      MTLAccelerationStructureSizes s =
         [dev accelerationStructureSizesWithDescriptor:mtl_desc];
      return (struct mtl_as_sizes){
         .acceleration_structure_size = s.accelerationStructureSize,
         .build_scratch_size = s.buildScratchBufferSize,
         .update_scratch_size = s.refitScratchBufferSize,
      };
   }
}

void
mtl_compute_build_as(mtl_compute_encoder *encoder,
                     mtl_acceleration_structure *dst,
                     const struct mtl_as_build_desc *desc,
                     uint64_t scratch_addr, uint64_t scratch_len)
{
   @autoreleasepool {
      id<MTL4ComputeCommandEncoder> enc =
         (id<MTL4ComputeCommandEncoder>)encoder;
      id<MTLAccelerationStructure> as = (id<MTLAccelerationStructure>)dst;
      MTL4AccelerationStructureDescriptor *mtl_desc =
         mtl_build_as_descriptor(desc, true);
      [enc buildAccelerationStructure:as
                           descriptor:mtl_desc
                        scratchBuffer:MTL4BufferRangeMake(scratch_addr,
                                                          scratch_len)];
   }
}

void
mtl_compute_refit_as(mtl_compute_encoder *encoder,
                     mtl_acceleration_structure *src,
                     mtl_acceleration_structure *dst,
                     const struct mtl_as_build_desc *desc,
                     uint64_t scratch_addr, uint64_t scratch_len, bool is_blas)
{
   @autoreleasepool {
      id<MTL4ComputeCommandEncoder> enc =
         (id<MTL4ComputeCommandEncoder>)encoder;
      id<MTLAccelerationStructure> src_as = (id<MTLAccelerationStructure>)src;
      id<MTLAccelerationStructure> dst_as = (id<MTLAccelerationStructure>)dst;
      MTL4AccelerationStructureDescriptor *mtl_desc =
         mtl_build_as_descriptor(desc, true);
      MTL4BufferRange scratch =
         scratch_addr ? MTL4BufferRangeMake(scratch_addr, scratch_len)
                      : MTL4BufferRangeMake(0, 0);
      if (is_blas) {
         [enc refitAccelerationStructure:src_as
                              descriptor:mtl_desc
                             destination:dst_as
                           scratchBuffer:scratch
                                 options:MTLAccelerationStructureRefitOptionVertexData];
      } else {
         [enc refitAccelerationStructure:src_as
                              descriptor:mtl_desc
                             destination:dst_as
                           scratchBuffer:scratch];
      }
   }
}

void
mtl_compute_copy_as(mtl_compute_encoder *encoder,
                    mtl_acceleration_structure *src,
                    mtl_acceleration_structure *dst, bool compact)
{
   @autoreleasepool {
      id<MTL4ComputeCommandEncoder> enc =
         (id<MTL4ComputeCommandEncoder>)encoder;
      id<MTLAccelerationStructure> src_as = (id<MTLAccelerationStructure>)src;
      id<MTLAccelerationStructure> dst_as = (id<MTLAccelerationStructure>)dst;
      if (compact)
         [enc copyAndCompactAccelerationStructure:src_as
                          toAccelerationStructure:dst_as];
      else
         [enc copyAccelerationStructure:src_as toAccelerationStructure:dst_as];
   }
}

void
mtl_compute_write_compacted_as_size(mtl_compute_encoder *encoder,
                                    mtl_acceleration_structure *src,
                                    uint64_t dst_addr)
{
   @autoreleasepool {
      id<MTL4ComputeCommandEncoder> enc =
         (id<MTL4ComputeCommandEncoder>)encoder;
      id<MTLAccelerationStructure> src_as = (id<MTLAccelerationStructure>)src;
      [enc writeCompactedAccelerationStructureSize:src_as
                                          toBuffer:MTL4BufferRangeMake(
                                                      dst_addr,
                                                      sizeof(uint64_t))];
   }
}
