#include "kk_acceleration_structure.h"

#include "kk_bo.h"
#include "kk_buffer.h"
#include "kk_cmd_buffer.h"
#include "kk_device.h"
#include "kk_entrypoints.h"
#include "kk_physical_device.h"
#include "kk_query_pool.h"

#include "kosmickrisp/bridge/mtl_bridge.h"

#include "util/hash_table.h"
#include "util/u_dynarray.h"
#include "util/u_atomic.h"
#include "util/u_math.h"
#include "vk_alloc.h"

#include <string.h>

#define KK_RT_VA_TABLE_SLOTS 65536u
#define KK_RT_VA_TOMBSTONE   1ull

struct kk_rt_sidecar {
   struct kk_bo *bo;
   uint32_t cap;
};

struct kk_rt_va_entry {
   uint64_t key;
   uint64_t val;
};

struct kk_rt_convert_params {
   uint64_t src_addr;
   uint64_t dst_addr;
   uint64_t va_table_addr;
   uint32_t va_table_mask;
   uint32_t count;
   uint32_t array_of_pointers;
   uint32_t pad;
   uint64_t sidecar_addr;
};

static const char kk_rt_instance_convert_msl[] =
   "#include <metal_stdlib>\n"
   "using namespace metal;\n"
   "struct kk_rt_convert_params {\n"
   "   uint64_t src_addr;\n"
   "   uint64_t dst_addr;\n"
   "   uint64_t va_table_addr;\n"
   "   uint32_t va_table_mask;\n"
   "   uint32_t count;\n"
   "   uint32_t array_of_pointers;\n"
   "   uint32_t pad;\n"
   "   uint64_t sidecar_addr;\n"
   "};\n"
   "struct kk_vk_as_instance {\n"
   "   float m00, m01, m02, m03;\n"
   "   float m10, m11, m12, m13;\n"
   "   float m20, m21, m22, m23;\n"
   "   uint32_t custom_index_and_mask;\n"
   "   uint32_t sbt_offset_and_flags;\n"
   "   uint64_t blas_ref;\n"
   "};\n"
   "struct kk_mtl_indirect_instance {\n"
   "   float col0_x, col0_y, col0_z;\n"
   "   float col1_x, col1_y, col1_z;\n"
   "   float col2_x, col2_y, col2_z;\n"
   "   float col3_x, col3_y, col3_z;\n"
   "   uint32_t options;\n"
   "   uint32_t mask;\n"
   "   uint32_t intersection_function_table_offset;\n"
   "   uint32_t user_id;\n"
   "   uint64_t acceleration_structure_id;\n"
   "};\n"
   "static_assert(sizeof(kk_vk_as_instance) == 64, \"\");\n"
   "static_assert(sizeof(kk_mtl_indirect_instance) == 72, \"\");\n"
   "static inline uint64_t kk_hash_u64(uint64_t x) {\n"
   "   x ^= x >> 30;\n"
   "   x *= 0xbf58476d1ce4e5b9ul;\n"
   "   x ^= x >> 27;\n"
   "   x *= 0x94d049bb133111ebul;\n"
   "   return x ^ (x >> 31);\n"
   "}\n"
   "static inline uint64_t kk_resolve_blas(uint64_t ref, const device ulong2 *table, uint32_t mask) {\n"
   "   if (ref == 0ul)\n"
   "      return 0ul;\n"
   "   if (table != nullptr && mask != 0u) {\n"
   "      uint32_t slot = uint32_t(kk_hash_u64(ref)) & mask;\n"
   "      for (uint32_t i = 0u; i <= mask; i++) {\n"
   "         ulong2 entry = table[(slot + i) & mask];\n"
   "         if (entry.x == ref)\n"
   "            return entry.y;\n"
   "         if (entry.x == 0ul)\n"
   "            break;\n"
   "      }\n"
   "   }\n"
   "   return ref;\n"
   "}\n"
   "kernel void kk_convert_instances(\n"
   "   constant kk_rt_convert_params &params [[buffer(0)]],\n"
   "   uint gid [[thread_position_in_grid]])\n"
   "{\n"
   "   if (gid >= params.count)\n"
   "      return;\n"
   "   device kk_mtl_indirect_instance *dst =\n"
   "      reinterpret_cast<device kk_mtl_indirect_instance *>(params.dst_addr) + gid;\n"
   "   const device kk_vk_as_instance *src;\n"
   "   if (params.array_of_pointers != 0u) {\n"
   "      const device uint64_t *ptrs = reinterpret_cast<const device uint64_t *>(params.src_addr);\n"
   "      uint64_t ptr = ptrs[gid];\n"
   "      if (ptr == 0ul) {\n"
   "         dst->col0_x = 0.0f; dst->col0_y = 0.0f; dst->col0_z = 0.0f;\n"
   "         dst->col1_x = 0.0f; dst->col1_y = 0.0f; dst->col1_z = 0.0f;\n"
   "         dst->col2_x = 0.0f; dst->col2_y = 0.0f; dst->col2_z = 0.0f;\n"
   "         dst->col3_x = 0.0f; dst->col3_y = 0.0f; dst->col3_z = 0.0f;\n"
   "         dst->options = 0u;\n"
   "         dst->mask = 0u;\n"
   "         dst->intersection_function_table_offset = 0u;\n"
   "         dst->user_id = 0u;\n"
   "         dst->acceleration_structure_id = 0ul;\n"
   "         if (params.sidecar_addr != 0ul)\n"
   "            reinterpret_cast<device uint32_t *>(params.sidecar_addr)[gid] = 0u;\n"
   "         return;\n"
   "      }\n"
   "      src = reinterpret_cast<const device kk_vk_as_instance *>(ptr);\n"
   "   } else {\n"
   "      src = reinterpret_cast<const device kk_vk_as_instance *>(params.src_addr) + gid;\n"
   "   }\n"
   "   const device ulong2 *table =\n"
   "      reinterpret_cast<const device ulong2 *>(params.va_table_addr);\n"
   "   uint64_t blas_id = kk_resolve_blas(src->blas_ref, table, params.va_table_mask);\n"
   "   dst->col0_x = src->m00; dst->col0_y = src->m10; dst->col0_z = src->m20;\n"
   "   dst->col1_x = src->m01; dst->col1_y = src->m11; dst->col1_z = src->m21;\n"
   "   dst->col2_x = src->m02; dst->col2_y = src->m12; dst->col2_z = src->m22;\n"
   "   dst->col3_x = src->m03; dst->col3_y = src->m13; dst->col3_z = src->m23;\n"
   "   dst->options = (src->sbt_offset_and_flags >> 24) & 0xfu;\n"
   "   dst->mask = blas_id ? (src->custom_index_and_mask >> 24) : 0u;\n"
   "   dst->intersection_function_table_offset = src->sbt_offset_and_flags & 0xffffffu;\n"
   "   dst->user_id = src->custom_index_and_mask & 0xffffffu;\n"
   "   dst->acceleration_structure_id = blas_id;\n"
   "   if (params.sidecar_addr != 0ul)\n"
   "      reinterpret_cast<device uint32_t *>(params.sidecar_addr)[gid] =\n"
   "         src->sbt_offset_and_flags & 0xffffffu;\n"
   "}\n";

static inline uint64_t
kk_rt_hash_u64(uint64_t x)
{
   x ^= x >> 30;
   x *= 0xbf58476d1ce4e5b9ull;
   x ^= x >> 27;
   x *= 0x94d049bb133111ebull;
   return x ^ (x >> 31);
}

static uint32_t
kk_u64_key_hash(const void *key)
{
   uint64_t v = (uintptr_t)key;
   return (uint32_t)kk_rt_hash_u64(v);
}

static bool
kk_u64_key_equal(const void *a, const void *b)
{
   return a == b;
}

VkResult
kk_rt_init(struct kk_device *dev)
{
   simple_mtx_init(&dev->rt.lock, mtx_plain);
   dev->rt.instance_convert_pso = NULL;
   dev->rt.dummy_bo = NULL;
   dev->rt.va_table_bo = NULL;
   dev->rt.va_table_mask = 0;
   dev->rt.as_by_id = _mesa_hash_table_create(NULL, kk_u64_key_hash,
                                              kk_u64_key_equal);
   dev->rt.sidecars = _mesa_hash_table_create(NULL, kk_u64_key_hash,
                                              kk_u64_key_equal);
   util_dynarray_init(&dev->rt.sidecar_garbage, NULL);
   if (!dev->rt.as_by_id) {
      simple_mtx_destroy(&dev->rt.lock);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   return VK_SUCCESS;
}

void
kk_rt_finish(struct kk_device *dev)
{
   if (dev->rt.instance_convert_pso) {
      mtl_release(dev->rt.instance_convert_pso);
      dev->rt.instance_convert_pso = NULL;
   }
   if (dev->rt.dummy_bo) {
      kk_destroy_bo(dev, dev->rt.dummy_bo);
      dev->rt.dummy_bo = NULL;
   }
   if (dev->rt.va_table_bo) {
      kk_destroy_bo(dev, dev->rt.va_table_bo);
      dev->rt.va_table_bo = NULL;
   }
   if (dev->rt.as_by_id) {
      _mesa_hash_table_destroy(dev->rt.as_by_id, NULL);
      dev->rt.as_by_id = NULL;
   }
   if (dev->rt.sidecars) {
      hash_table_foreach(dev->rt.sidecars, e) {
         struct kk_rt_sidecar *sc = e->data;
         if (sc->bo)
            kk_destroy_bo(dev, sc->bo);
         free(sc);
      }
      _mesa_hash_table_destroy(dev->rt.sidecars, NULL);
      dev->rt.sidecars = NULL;
   }
   util_dynarray_foreach(&dev->rt.sidecar_garbage, struct kk_bo *, bo)
      kk_destroy_bo(dev, *bo);
   util_dynarray_fini(&dev->rt.sidecar_garbage);
   simple_mtx_destroy(&dev->rt.lock);
}

static VkResult
kk_rt_ensure_buffers_locked(struct kk_device *dev)
{
   if (!dev->rt.dummy_bo) {
      VkResult res = kk_alloc_bo(dev, &dev->vk.base, 256u, 256u,
                                 &dev->rt.dummy_bo);
      if (res != VK_SUCCESS)
         return res;
      memset(dev->rt.dummy_bo->cpu, 0, 256u);
   }
   if (!dev->rt.va_table_bo) {
      uint64_t bytes = KK_RT_VA_TABLE_SLOTS * sizeof(struct kk_rt_va_entry);
      VkResult res = kk_alloc_bo(dev, &dev->vk.base, bytes, 256u,
                                 &dev->rt.va_table_bo);
      if (res != VK_SUCCESS)
         return res;
      memset(dev->rt.va_table_bo->cpu, 0, bytes);
      dev->rt.va_table_mask = KK_RT_VA_TABLE_SLOTS - 1u;
   }
   return VK_SUCCESS;
}

void
kk_rt_get_va_table(struct kk_device *dev, uint64_t *addr, uint32_t *mask)
{
   struct kk_bo *bo = p_atomic_read(&dev->rt.va_table_bo);
   if (!bo) {
      simple_mtx_lock(&dev->rt.lock);
      kk_rt_ensure_buffers_locked(dev);
      bo = dev->rt.va_table_bo;
      simple_mtx_unlock(&dev->rt.lock);
   }
   *addr = bo ? bo->gpu : 0;
   *mask = bo ? KK_RT_VA_TABLE_SLOTS - 1u : 0;
}

static mtl_compute_pipeline_state *
kk_rt_get_convert_pso(struct kk_device *dev)
{
   simple_mtx_lock(&dev->rt.lock);
   if (dev->rt.instance_convert_pso) {
      mtl_compute_pipeline_state *pso = dev->rt.instance_convert_pso;
      simple_mtx_unlock(&dev->rt.lock);
      return pso;
   }

   struct kk_physical_device *pdev = kk_device_physical(dev);
   mtl_library *lib = mtl_new_library(
      dev->mtl_compiler_handle, kk_rt_instance_convert_msl,
      pdev->info.msl_version, MTL_MATH_MODE_FAST,
      MTL_MATH_FLOATING_POINT_FUNCTIONS_FAST);
   mtl_compute_pipeline_state *pso = NULL;
   if (lib) {
      mtl_function_descriptor *fn =
         mtl_new_library_function_descriptor(lib, "kk_convert_instances");
      if (fn) {
         pso = mtl_new_compute_pipeline_state(dev->mtl_compiler_handle, fn, 64);
         mtl_release(fn);
      }
      mtl_release(lib);
   }
   dev->rt.instance_convert_pso = pso;
   simple_mtx_unlock(&dev->rt.lock);
   return pso;
}

static void
kk_rt_va_table_insert_locked(struct kk_device *dev, uint64_t va, uint64_t id)
{
   if (!dev->rt.va_table_bo || va == 0 || va == KK_RT_VA_TOMBSTONE)
      return;

   struct kk_rt_va_entry *table = dev->rt.va_table_bo->cpu;
   uint32_t mask = dev->rt.va_table_mask;
   uint32_t slot = (uint32_t)kk_rt_hash_u64(va) & mask;
   uint32_t first_tombstone = UINT32_MAX;

   for (uint32_t i = 0; i <= mask; i++) {
      uint32_t idx = (slot + i) & mask;
      uint64_t k = table[idx].key;
      if (k == va) {
         table[idx].val = id;
         return;
      }
      if (k == KK_RT_VA_TOMBSTONE && first_tombstone == UINT32_MAX)
         first_tombstone = idx;
      if (k == 0) {
         uint32_t dst = first_tombstone != UINT32_MAX ? first_tombstone : idx;
         table[dst].val = id;
         table[dst].key = va;
         return;
      }
   }
   if (first_tombstone != UINT32_MAX) {
      table[first_tombstone].val = id;
      table[first_tombstone].key = va;
   }
}

static void
kk_rt_va_table_remove_locked(struct kk_device *dev, uint64_t va, uint64_t id)
{
   if (!dev->rt.va_table_bo || va == 0 || va == KK_RT_VA_TOMBSTONE)
      return;

   struct kk_rt_va_entry *table = dev->rt.va_table_bo->cpu;
   uint32_t mask = dev->rt.va_table_mask;
   uint32_t slot = (uint32_t)kk_rt_hash_u64(va) & mask;

   for (uint32_t i = 0; i <= mask; i++) {
      uint32_t idx = (slot + i) & mask;
      uint64_t k = table[idx].key;
      if (k == 0)
         return;
      if (k == va) {
         if (table[idx].val == id) {
            table[idx].key = KK_RT_VA_TOMBSTONE;
            table[idx].val = 0;
         }
         return;
      }
   }
}

static enum mtl_as_vertex_format
vk_format_to_mtl_as_vertex_format(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_R32G32B32_SFLOAT:
      return MTL_AS_VERTEX_FORMAT_FLOAT3;
   case VK_FORMAT_R32G32_SFLOAT:
      return MTL_AS_VERTEX_FORMAT_FLOAT2;
   case VK_FORMAT_R16G16B16A16_SFLOAT:
      return MTL_AS_VERTEX_FORMAT_HALF3;
   case VK_FORMAT_R16G16_SFLOAT:
      return MTL_AS_VERTEX_FORMAT_HALF2;
   case VK_FORMAT_R16G16_SNORM:
      return MTL_AS_VERTEX_FORMAT_SHORT2_NORMALIZED;
   case VK_FORMAT_R16G16B16A16_SNORM:
      return MTL_AS_VERTEX_FORMAT_SHORT4_NORMALIZED;
   case VK_FORMAT_R16G16_UNORM:
      return MTL_AS_VERTEX_FORMAT_USHORT2_NORMALIZED;
   case VK_FORMAT_R16G16B16A16_UNORM:
      return MTL_AS_VERTEX_FORMAT_USHORT4_NORMALIZED;
   default:
      return MTL_AS_VERTEX_FORMAT_FLOAT3;
   }
}

static uint32_t
vk_build_flags_to_mtl_as_usage(VkBuildAccelerationStructureFlagsKHR flags)
{
   uint32_t usage = MTL_AS_USAGE_NONE;
   if (flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR)
      usage |= MTL_AS_USAGE_REFIT;
   if (flags & VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR)
      usage |= MTL_AS_USAGE_PREFER_FAST_BUILD;
   if (flags & VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR)
      usage |= MTL_AS_USAGE_PREFER_FAST_INTERSECTION;
   if (flags & VK_BUILD_ACCELERATION_STRUCTURE_LOW_MEMORY_BIT_KHR)
      usage |= MTL_AS_USAGE_MINIMIZE_MEMORY;
   return usage;
}

static void
kk_fill_mtl_as_geometries(const VkAccelerationStructureBuildGeometryInfoKHR *info,
                          const VkAccelerationStructureBuildRangeInfoKHR *ranges,
                          const uint32_t *max_primitive_counts,
                          struct mtl_as_geometry *out_geoms)
{
   for (uint32_t i = 0; i < info->geometryCount; i++) {
      const VkAccelerationStructureGeometryKHR *g =
         info->pGeometries ? &info->pGeometries[i] : info->ppGeometries[i];
      struct mtl_as_geometry *dst = &out_geoms[i];
      memset(dst, 0, sizeof(*dst));
      dst->geometry_index = i;
      dst->opaque = (g->flags & VK_GEOMETRY_OPAQUE_BIT_KHR) != 0;
      dst->allow_duplicate_any_hit =
         (g->flags & VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR) == 0;

      if (g->geometryType == VK_GEOMETRY_TYPE_TRIANGLES_KHR) {
         const VkAccelerationStructureGeometryTrianglesDataKHR *tri =
            &g->geometry.triangles;
         dst->type = MTL_AS_GEOMETRY_TRIANGLES;
         dst->triangles.vertex_format =
            vk_format_to_mtl_as_vertex_format(tri->vertexFormat);
         dst->triangles.vertex_stride = tri->vertexStride;
         dst->triangles.has_index_buffer =
            tri->indexType != VK_INDEX_TYPE_NONE_KHR;
         if (tri->indexType == VK_INDEX_TYPE_UINT16)
            dst->triangles.index_type = MTL_INDEX_TYPE_UINT16;
         else
            dst->triangles.index_type = MTL_INDEX_TYPE_UINT32;

         if (ranges) {
            dst->triangles.triangle_count = ranges[i].primitiveCount;
            uint64_t vaddr = tri->vertexData.deviceAddress;
            if (vaddr) {
               vaddr += (uint64_t)ranges[i].firstVertex * tri->vertexStride;
               if (!dst->triangles.has_index_buffer)
                  vaddr += ranges[i].primitiveOffset;
            }
            dst->triangles.vertex_addr = vaddr;
            if (dst->triangles.has_index_buffer && tri->indexData.deviceAddress) {
               dst->triangles.index_addr =
                  tri->indexData.deviceAddress + ranges[i].primitiveOffset;
            }
            if (tri->transformData.deviceAddress) {
               dst->triangles.transform_addr =
                  tri->transformData.deviceAddress + ranges[i].transformOffset;
            }
         } else {
            dst->triangles.triangle_count = max_primitive_counts[i];
         }
      } else {
         const VkAccelerationStructureGeometryAabbsDataKHR *aabb =
            &g->geometry.aabbs;
         dst->type = MTL_AS_GEOMETRY_AABBS;
         dst->aabbs.aabb_stride = aabb->stride >= 24 ? aabb->stride : 24;
         if (ranges) {
            dst->aabbs.aabb_count = ranges[i].primitiveCount;
            if (aabb->data.deviceAddress)
               dst->aabbs.aabb_addr =
                  aabb->data.deviceAddress + ranges[i].primitiveOffset;
         } else {
            dst->aabbs.aabb_count = max_primitive_counts[i];
         }
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
kk_GetAccelerationStructureBuildSizesKHR(
   VkDevice device, VkAccelerationStructureBuildTypeKHR buildType,
   const VkAccelerationStructureBuildGeometryInfoKHR *pBuildInfo,
   const uint32_t *pMaxPrimitiveCounts,
   VkAccelerationStructureBuildSizesInfoKHR *pSizeInfo)
{
   VK_FROM_HANDLE(kk_device, dev, device);

   struct mtl_as_build_desc desc = {
      .is_top_level =
         pBuildInfo->type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
      .usage = vk_build_flags_to_mtl_as_usage(pBuildInfo->flags),
   };

   struct mtl_as_geometry stack_geoms[8];
   struct mtl_as_geometry *geoms = stack_geoms;
   bool heap_alloc = false;

   if (desc.is_top_level) {
      desc.instance_count =
         (pBuildInfo->geometryCount > 0 && pMaxPrimitiveCounts)
            ? pMaxPrimitiveCounts[0]
            : 0;
   } else if (pBuildInfo->geometryCount > 0) {
      if (pBuildInfo->geometryCount > ARRAY_SIZE(stack_geoms)) {
         geoms = malloc(pBuildInfo->geometryCount * sizeof(*geoms));
         heap_alloc = true;
      }
      kk_fill_mtl_as_geometries(pBuildInfo, NULL, pMaxPrimitiveCounts, geoms);
      desc.geometry_count = pBuildInfo->geometryCount;
      desc.geometries = geoms;
   }

   struct mtl_as_sizes sizes = mtl_device_get_as_sizes(dev->mtl_handle, &desc);

   if (heap_alloc)
      free(geoms);

   uint64_t heap_size = sizes.acceleration_structure_size;
   uint64_t heap_align = 256;
   mtl_heap_acceleration_structure_size_and_align(
      dev->mtl_handle, MAX2(sizes.acceleration_structure_size, 256u),
      &heap_size, &heap_align);
   uint64_t pad = heap_align > 256u ? heap_align - 256u : 0u;

   pSizeInfo->accelerationStructureSize =
      MAX2(align64(heap_size + pad, 256u), 256u);
   pSizeInfo->buildScratchSize =
      MAX2(align64(sizes.build_scratch_size, 256u), 256u);
   pSizeInfo->updateScratchSize =
      MAX2(align64(sizes.update_scratch_size, 256u), 256u);
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_CreateAccelerationStructureKHR(
   VkDevice device, const VkAccelerationStructureCreateInfoKHR *pCreateInfo,
   const VkAllocationCallbacks *pAllocator,
   VkAccelerationStructureKHR *pAccelerationStructure)
{
   VK_FROM_HANDLE(kk_device, dev, device);
   VK_FROM_HANDLE(kk_buffer, buffer, pCreateInfo->buffer);

   struct kk_acceleration_structure *as = vk_object_zalloc(
      &dev->vk, pAllocator, sizeof(*as),
      VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR);
   if (!as)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   simple_mtx_lock(&dev->rt.lock);
   VkResult init_res = kk_rt_ensure_buffers_locked(dev);
   simple_mtx_unlock(&dev->rt.lock);
   if (init_res != VK_SUCCESS) {
      vk_object_free(&dev->vk, pAllocator, as);
      return init_res;
   }

   as->type = pCreateInfo->type;
   as->size = pCreateInfo->size;
   as->buffer_va = vk_buffer_address(&buffer->vk, pCreateInfo->offset);

   uint64_t req_size = MAX2(pCreateInfo->size, 256u);
   if (buffer->metal.heap && mtl_heap_is_private(buffer->metal.heap)) {
      uint64_t raw_offset = buffer->metal.heap_offset + pCreateInfo->offset;
      uint64_t heap_total = mtl_heap_get_size(buffer->metal.heap);
      uint64_t sa_size = req_size;
      uint64_t sa_align = 256;
      mtl_heap_acceleration_structure_size_and_align(dev->mtl_handle, req_size,
                                                     &sa_size, &sa_align);
      if (sa_align == 0)
         sa_align = 256;
      uint64_t aligned_offset = align64(raw_offset, sa_align);
      uint64_t shift = aligned_offset - raw_offset;
      if (shift < pCreateInfo->size && aligned_offset < heap_total) {
         uint64_t avail = MIN2(pCreateInfo->size - shift,
                               heap_total - aligned_offset);
         uint64_t cand = avail & ~(uint64_t)255u;
         while (cand >= 256u) {
            uint64_t cand_sa = cand;
            mtl_heap_acceleration_structure_size_and_align(
               dev->mtl_handle, cand, &cand_sa, NULL);
            if (cand_sa <= avail)
               break;
            cand -= 256u;
         }
         if (cand >= 256u) {
            as->mtl_handle = mtl_new_acceleration_structure_from_heap(
               buffer->metal.heap, cand, aligned_offset);
            if (as->mtl_handle)
               as->placed = true;
         }
      }
   }

   if (!as->mtl_handle) {
      as->mtl_handle =
         mtl_new_acceleration_structure_from_device(dev->mtl_handle, req_size);
      if (!as->mtl_handle) {
         vk_object_free(&dev->vk, pAllocator, as);
         return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      }
   }

   kk_device_add_allocation_to_residency_set(dev, as->mtl_handle);
   as->mtl_resource_id =
      mtl_acceleration_structure_get_gpu_resource_id(as->mtl_handle);

   simple_mtx_lock(&dev->rt.lock);
   kk_rt_va_table_insert_locked(dev, as->buffer_va, as->mtl_resource_id);
   _mesa_hash_table_insert(dev->rt.as_by_id,
                           (void *)(uintptr_t)as->mtl_resource_id, as);
   simple_mtx_unlock(&dev->rt.lock);

   *pAccelerationStructure = kk_acceleration_structure_to_handle(as);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
kk_DestroyAccelerationStructureKHR(
   VkDevice device, VkAccelerationStructureKHR accelerationStructure,
   const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(kk_device, dev, device);
   VK_FROM_HANDLE(kk_acceleration_structure, as, accelerationStructure);

   if (!as)
      return;

   simple_mtx_lock(&dev->rt.lock);
   kk_rt_va_table_remove_locked(dev, as->buffer_va, as->mtl_resource_id);
   _mesa_hash_table_remove_key(dev->rt.as_by_id,
                               (void *)(uintptr_t)as->mtl_resource_id);
   simple_mtx_unlock(&dev->rt.lock);

   if (as->mtl_handle) {
      kk_device_remove_allocation_from_residency_set(dev, as->mtl_handle);
      mtl_release(as->mtl_handle);
   }

   vk_object_free(&dev->vk, pAllocator, as);
}

VKAPI_ATTR VkDeviceAddress VKAPI_CALL
kk_GetAccelerationStructureDeviceAddressKHR(
   VkDevice device, const VkAccelerationStructureDeviceAddressInfoKHR *pInfo)
{
   VK_FROM_HANDLE(kk_acceleration_structure, as, pInfo->accelerationStructure);
   return as->mtl_resource_id;
}

VKAPI_ATTR void VKAPI_CALL
kk_GetDeviceAccelerationStructureCompatibilityKHR(
   VkDevice device, const VkAccelerationStructureVersionInfoKHR *pVersionInfo,
   VkAccelerationStructureCompatibilityKHR *pCompatibility)
{
   *pCompatibility = VK_ACCELERATION_STRUCTURE_COMPATIBILITY_INCOMPATIBLE_KHR;
}

static uint64_t
kk_rt_get_sidecar(struct kk_device *dev, struct kk_acceleration_structure *as,
                  uint32_t count)
{
   if (!as)
      return 0;

   simple_mtx_lock(&dev->rt.lock);
   void *key = (void *)(uintptr_t)as->buffer_va;
   struct hash_entry *e = _mesa_hash_table_search(dev->rt.sidecars, key);
   struct kk_rt_sidecar *sc = e ? e->data : NULL;
   if (!sc) {
      sc = calloc(1, sizeof(*sc));
      if (!sc) {
         simple_mtx_unlock(&dev->rt.lock);
         return 0;
      }
      _mesa_hash_table_insert(dev->rt.sidecars, key, sc);
   }
   if (sc->cap < count) {
      uint32_t cap = MAX2(util_next_power_of_two(count), 256u);
      struct kk_bo *bo = NULL;
      if (kk_alloc_bo(dev, &dev->vk.base, cap * 4ull, 256u, &bo) == VK_SUCCESS) {
         memset(bo->cpu, 0, cap * 4ull);
         if (sc->bo)
            util_dynarray_append(&dev->rt.sidecar_garbage, sc->bo);
         sc->bo = bo;
         sc->cap = cap;
      }
   }
   uint64_t addr = sc->bo ? sc->bo->gpu : 0;
   if (addr) {
      kk_rt_va_table_insert_locked(dev, as->buffer_va | (1ull << 63), addr);
      kk_rt_va_table_insert_locked(dev, as->mtl_resource_id | (1ull << 63), addr);
   }
   simple_mtx_unlock(&dev->rt.lock);
   return sc->cap >= count ? addr : 0;
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdBuildAccelerationStructuresKHR(
   VkCommandBuffer commandBuffer, uint32_t infoCount,
   const VkAccelerationStructureBuildGeometryInfoKHR *pInfos,
   const VkAccelerationStructureBuildRangeInfoKHR *const *ppBuildRangeInfos)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);
   struct kk_device *dev = kk_cmd_buffer_device(cmd);

   simple_mtx_lock(&dev->rt.lock);
   VkResult init_res = kk_rt_ensure_buffers_locked(dev);
   uint64_t dummy_addr = dev->rt.dummy_bo ? dev->rt.dummy_bo->gpu : 0;
   uint64_t va_table_addr = dev->rt.va_table_bo ? dev->rt.va_table_bo->gpu : 0;
   uint32_t va_table_mask = dev->rt.va_table_mask;
   simple_mtx_unlock(&dev->rt.lock);
   if (init_res != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, init_res);
      return;
   }

   mtl_compute_encoder *encoder = cs_get_compute(cmd);

   for (uint32_t i = 0; i < infoCount; i++) {
      const VkAccelerationStructureBuildGeometryInfoKHR *info = &pInfos[i];
      const VkAccelerationStructureBuildRangeInfoKHR *ranges =
         ppBuildRangeInfos[i];
      VK_FROM_HANDLE(kk_acceleration_structure, dst,
                     info->dstAccelerationStructure);
      VK_FROM_HANDLE(kk_acceleration_structure, src,
                     info->srcAccelerationStructure);

      struct mtl_as_build_desc desc = {
         .is_top_level =
            info->type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
         .usage = vk_build_flags_to_mtl_as_usage(info->flags),
         .dummy_addr = dummy_addr,
      };

      struct mtl_as_geometry stack_geoms[8];
      struct mtl_as_geometry *geoms = stack_geoms;
      bool heap_alloc = false;

      if (desc.is_top_level) {
         uint32_t count =
            (info->geometryCount > 0 && ranges) ? ranges[0].primitiveCount : 0;
         desc.instance_count = count;

         if (count > 0) {
            const VkAccelerationStructureGeometryKHR *g =
               info->pGeometries ? &info->pGeometries[0] : info->ppGeometries[0];
            uint64_t src_addr =
               g->geometry.instances.data.deviceAddress + ranges[0].primitiveOffset;
            uint32_t dst_bytes = count * 72u;
            struct kk_ptr dst_ptr = kk_pool_alloc(cmd, dst_bytes, 64u);
            if (!dst_ptr.gpu)
               return;

            mtl_compute_pipeline_state *pso = kk_rt_get_convert_pso(dev);
            if (!pso) {
               vk_command_buffer_set_error(&cmd->vk,
                                           VK_ERROR_INITIALIZATION_FAILED);
               return;
            }

            uint64_t sidecar_addr = kk_rt_get_sidecar(dev, dst, count);
            struct kk_rt_convert_params params = {
               .sidecar_addr = sidecar_addr,
               .src_addr = src_addr,
               .dst_addr = dst_ptr.gpu,
               .va_table_addr = va_table_addr,
               .va_table_mask = va_table_mask,
               .count = count,
               .array_of_pointers = g->geometry.instances.arrayOfPointers ? 1u : 0u,
            };
            struct kk_ptr params_ptr =
               kk_pool_upload(cmd, &params, sizeof(params), 8u);
            if (!params_ptr.gpu)
               return;

            mtl_set_address(cmd->argument_table, params_ptr.gpu, 0u);
            mtl_compute_set_pipeline_state(encoder, pso);
            mtl_dispatch_threads(encoder, (struct mtl_size){count, 1, 1},
                                 (struct mtl_size){MIN2(count, 64u), 1, 1});
            mtl_set_address(cmd->argument_table, cmd->state.root_addr, 0u);
            mtl_barrier_after_encoder_stages(encoder, MTL_STAGE_DISPATCH,
                                             MTL_STAGE_ACCELERATION_STRUCTURE);
            desc.instance_descriptor_addr = dst_ptr.gpu;
         } else {
            desc.instance_descriptor_addr = dummy_addr;
         }
      } else if (info->geometryCount > 0) {
         if (info->geometryCount > ARRAY_SIZE(stack_geoms)) {
            geoms = malloc(info->geometryCount * sizeof(*geoms));
            heap_alloc = true;
         }
         kk_fill_mtl_as_geometries(info, ranges, NULL, geoms);
         desc.geometry_count = info->geometryCount;
         desc.geometries = geoms;
      }

      uint64_t scratch_addr = info->scratchData.deviceAddress;
      if (info->mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR && src) {
         mtl_compute_refit_as(encoder, src->mtl_handle, dst->mtl_handle, &desc,
                              scratch_addr, (uint64_t)-1, !desc.is_top_level);
      } else {
         mtl_compute_build_as(encoder, dst->mtl_handle, &desc, scratch_addr,
                              (uint64_t)-1);
      }

      if (heap_alloc)
         free(geoms);
   }
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdCopyAccelerationStructureKHR(
   VkCommandBuffer commandBuffer,
   const VkCopyAccelerationStructureInfoKHR *pInfo)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(kk_acceleration_structure, src, pInfo->src);
   VK_FROM_HANDLE(kk_acceleration_structure, dst, pInfo->dst);

   mtl_compute_encoder *encoder = cs_get_compute(cmd);
   bool compact = pInfo->mode == VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;
   mtl_compute_copy_as(encoder, src->mtl_handle, dst->mtl_handle, compact);
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdWriteAccelerationStructuresPropertiesKHR(
   VkCommandBuffer commandBuffer, uint32_t accelerationStructureCount,
   const VkAccelerationStructureKHR *pAccelerationStructures,
   VkQueryType queryType, VkQueryPool queryPool, uint32_t firstQuery)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(kk_query_pool, pool, queryPool);

   mtl_compute_encoder *encoder = cs_get_compute(cmd);

   for (uint32_t i = 0; i < accelerationStructureCount; i++) {
      VK_FROM_HANDLE(kk_acceleration_structure, as, pAccelerationStructures[i]);
      uint32_t q = firstQuery + i;
      uint64_t report_addr =
         pool->bo->gpu + pool->query_start + (uint64_t)q * pool->query_stride;
      uint64_t avail_addr = pool->bo->gpu + (uint64_t)q * sizeof(uint32_t);

      if (queryType == VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR) {
         mtl_compute_write_compacted_as_size(encoder, as->mtl_handle,
                                             report_addr);
         mtl_barrier_after_encoder_stages(encoder,
                                          MTL_STAGE_ACCELERATION_STRUCTURE,
                                          MTL_STAGE_DISPATCH);
      } else {
         uint64_t val =
            queryType == VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SIZE_KHR
               ? mtl_acceleration_structure_get_size(as->mtl_handle)
               : 0;
         kk_cmd_write(cmd, (struct libkk_imm_write){
            .address = report_addr,
            .value = (uint32_t)(val & 0xffffffffu),
         });
         kk_cmd_write(cmd, (struct libkk_imm_write){
            .address = report_addr + 4,
            .value = (uint32_t)(val >> 32),
         });
      }

      kk_cmd_write(cmd, (struct libkk_imm_write){
         .address = avail_addr,
         .value = 1u,
      });
   }
}
