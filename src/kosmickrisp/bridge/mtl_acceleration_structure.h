#ifndef MTL_ACCELERATION_STRUCTURE_H
#define MTL_ACCELERATION_STRUCTURE_H 1

#include "mtl_types.h"

#include <stdbool.h>
#include <stdint.h>

enum mtl_as_usage {
   MTL_AS_USAGE_NONE = 0,
   MTL_AS_USAGE_REFIT = 1 << 0,
   MTL_AS_USAGE_PREFER_FAST_BUILD = 1 << 1,
   MTL_AS_USAGE_EXTENDED_LIMITS = 1 << 2,
   MTL_AS_USAGE_PREFER_FAST_INTERSECTION = 1 << 4,
   MTL_AS_USAGE_MINIMIZE_MEMORY = 1 << 5,
};

enum mtl_as_geometry_type {
   MTL_AS_GEOMETRY_TRIANGLES = 0,
   MTL_AS_GEOMETRY_AABBS = 1,
};

enum mtl_as_vertex_format {
   MTL_AS_VERTEX_FORMAT_FLOAT3 = 0,
   MTL_AS_VERTEX_FORMAT_FLOAT2 = 1,
   MTL_AS_VERTEX_FORMAT_HALF3 = 2,
   MTL_AS_VERTEX_FORMAT_HALF2 = 3,
   MTL_AS_VERTEX_FORMAT_SHORT2_NORMALIZED = 4,
   MTL_AS_VERTEX_FORMAT_SHORT4_NORMALIZED = 5,
   MTL_AS_VERTEX_FORMAT_USHORT2_NORMALIZED = 6,
   MTL_AS_VERTEX_FORMAT_USHORT4_NORMALIZED = 7,
};

struct mtl_as_triangle_geometry {
   uint64_t vertex_addr;
   uint64_t vertex_stride;
   enum mtl_as_vertex_format vertex_format;
   uint64_t index_addr;
   enum mtl_index_type index_type;
   bool has_index_buffer;
   uint64_t transform_addr;
   uint32_t triangle_count;
};

struct mtl_as_aabb_geometry {
   uint64_t aabb_addr;
   uint64_t aabb_stride;
   uint32_t aabb_count;
};

struct mtl_as_geometry {
   enum mtl_as_geometry_type type;
   uint32_t geometry_index;
   bool opaque;
   bool allow_duplicate_any_hit;
   union {
      struct mtl_as_triangle_geometry triangles;
      struct mtl_as_aabb_geometry aabbs;
   };
};

struct mtl_as_build_desc {
   bool is_top_level;
   uint32_t usage;
   uint32_t geometry_count;
   const struct mtl_as_geometry *geometries;
   uint64_t instance_descriptor_addr;
   uint32_t instance_count;
   uint64_t dummy_addr;
};

struct mtl_as_sizes {
   uint64_t acceleration_structure_size;
   uint64_t build_scratch_size;
   uint64_t update_scratch_size;
};

bool mtl_device_supports_raytracing(mtl_device *device);
void mtl_heap_acceleration_structure_size_and_align(mtl_device *device,
                                                    uint64_t size_B,
                                                    uint64_t *size_out,
                                                    uint64_t *align_out);
bool mtl_heap_is_private(mtl_heap *heap);

mtl_acceleration_structure *
mtl_new_acceleration_structure_from_heap(mtl_heap *heap, uint64_t size_B,
                                         uint64_t offset_B);
mtl_acceleration_structure *
mtl_new_acceleration_structure_from_device(mtl_device *device, uint64_t size_B);

uint64_t mtl_acceleration_structure_get_gpu_resource_id(
   mtl_acceleration_structure *as);
uint64_t mtl_acceleration_structure_get_size(mtl_acceleration_structure *as);

struct mtl_as_sizes
mtl_device_get_as_sizes(mtl_device *device,
                        const struct mtl_as_build_desc *desc);

void mtl_compute_build_as(mtl_compute_encoder *encoder,
                          mtl_acceleration_structure *dst,
                          const struct mtl_as_build_desc *desc,
                          uint64_t scratch_addr, uint64_t scratch_len);

void mtl_compute_refit_as(mtl_compute_encoder *encoder,
                          mtl_acceleration_structure *src,
                          mtl_acceleration_structure *dst,
                          const struct mtl_as_build_desc *desc,
                          uint64_t scratch_addr, uint64_t scratch_len,
                          bool is_blas);

void mtl_compute_copy_as(mtl_compute_encoder *encoder,
                         mtl_acceleration_structure *src,
                         mtl_acceleration_structure *dst, bool compact);

void mtl_compute_write_compacted_as_size(mtl_compute_encoder *encoder,
                                         mtl_acceleration_structure *src,
                                         uint64_t dst_addr);

#endif
