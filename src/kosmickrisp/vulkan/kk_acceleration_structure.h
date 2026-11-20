#ifndef KK_ACCELERATION_STRUCTURE_H
#define KK_ACCELERATION_STRUCTURE_H 1

#include "kk_private.h"

#include "kosmickrisp/bridge/mtl_types.h"

#include "util/simple_mtx.h"
#include "vk_object.h"

struct hash_table;
struct kk_bo;
struct kk_device;

struct kk_rt_state {
   simple_mtx_t lock;
   mtl_compute_pipeline_state *instance_convert_pso;
   struct kk_bo *dummy_bo;
   struct kk_bo *va_table_bo;
   uint32_t va_table_mask;
   struct hash_table *as_by_id;
};

struct kk_acceleration_structure {
   struct vk_object_base base;
   mtl_acceleration_structure *mtl_handle;
   uint64_t mtl_resource_id;
   uint64_t buffer_va;
   uint64_t size;
   VkAccelerationStructureTypeKHR type;
   bool placed;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(kk_acceleration_structure, base,
                               VkAccelerationStructureKHR,
                               VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR)

VkResult kk_rt_init(struct kk_device *dev);
void kk_rt_finish(struct kk_device *dev);

#endif
