#include "kosmickrisp/bridge/mtl_acceleration_structure.h"

#include "util/macros.h"

bool
mtl_device_supports_raytracing(mtl_device *device)
{
   return false;
}

void
mtl_heap_acceleration_structure_size_and_align(mtl_device *device,
                                               uint64_t size_B,
                                               uint64_t *size_out,
                                               uint64_t *align_out)
{
   if (size_out)
      *size_out = size_B;
   if (align_out)
      *align_out = 256;
}

bool
mtl_heap_is_private(mtl_heap *heap)
{
   return false;
}

mtl_acceleration_structure *
mtl_new_acceleration_structure_from_heap(mtl_heap *heap, uint64_t size_B,
                                         uint64_t offset_B)
{
   UNREACHABLE("Unsupported");
}

mtl_acceleration_structure *
mtl_new_acceleration_structure_from_device(mtl_device *device, uint64_t size_B)
{
   UNREACHABLE("Unsupported");
}

uint64_t
mtl_acceleration_structure_get_gpu_resource_id(mtl_acceleration_structure *as)
{
   UNREACHABLE("Unsupported");
}

uint64_t
mtl_acceleration_structure_get_size(mtl_acceleration_structure *as)
{
   UNREACHABLE("Unsupported");
}

struct mtl_as_sizes
mtl_device_get_as_sizes(mtl_device *device,
                        const struct mtl_as_build_desc *desc)
{
   UNREACHABLE("Unsupported");
}

void
mtl_compute_build_as(mtl_compute_encoder *encoder,
                     mtl_acceleration_structure *dst,
                     const struct mtl_as_build_desc *desc,
                     uint64_t scratch_addr, uint64_t scratch_len)
{
   UNREACHABLE("Unsupported");
}

void
mtl_compute_refit_as(mtl_compute_encoder *encoder,
                     mtl_acceleration_structure *src,
                     mtl_acceleration_structure *dst,
                     const struct mtl_as_build_desc *desc,
                     uint64_t scratch_addr, uint64_t scratch_len, bool is_blas)
{
   UNREACHABLE("Unsupported");
}

void
mtl_compute_copy_as(mtl_compute_encoder *encoder,
                    mtl_acceleration_structure *src,
                    mtl_acceleration_structure *dst, bool compact)
{
   UNREACHABLE("Unsupported");
}

void
mtl_compute_write_compacted_as_size(mtl_compute_encoder *encoder,
                                    mtl_acceleration_structure *src,
                                    uint64_t dst_addr)
{
   UNREACHABLE("Unsupported");
}
