/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */
#include "kk_cmd_buffer.h"
#include "kk_descriptor_set.h"
#include "kk_descriptor_set_layout.h"
#include "kk_shader.h"

#include "kosmickrisp/compiler/nir_to_msl.h"

#include "nir.h"
#include "nir_builder.h"
#include "nir_builder_opcodes.h"
#include "nir_intrinsics.h"
#include "nir_intrinsics_indices.h"
#include "shader_enums.h"
#include "vk_pipeline.h"

#include "vulkan/vulkan_core.h"

struct lower_descriptors_ctx {
   const struct kk_descriptor_set_layout *set_layouts[KK_MAX_SETS];

   bool clamp_desc_array_bounds;
   nir_address_format ubo_addr_format;
   nir_address_format ssbo_addr_format;
};

static const struct kk_descriptor_set_binding_layout *
get_binding_layout(uint32_t set, uint32_t binding,
                   const struct lower_descriptors_ctx *ctx)
{
   assert(set < KK_MAX_SETS);
   assert(ctx->set_layouts[set] != NULL);

   const struct kk_descriptor_set_layout *set_layout = ctx->set_layouts[set];

   assert(binding < set_layout->binding_count);
   return &set_layout->binding[binding];
}

static nir_def *
load_speculatable(nir_builder *b, unsigned num_components, unsigned bit_size,
                  nir_def *addr, unsigned align)
{
   return nir_build_load_global_constant(b, num_components, bit_size, addr,
                                         .align_mul = align,
                                         .access = ACCESS_CAN_SPECULATE);
}

static nir_def *
load_root(nir_builder *b, unsigned num_components, unsigned bit_size,
          nir_def *offset, unsigned align)
{
   nir_def *root = nir_load_buffer_ptr_kk(b, 1, 64, .binding = 0);

   /* We've bound the address of the root descriptor, index in. */
   nir_def *addr = nir_iadd(b, root, nir_u2u64(b, offset));

   return load_speculatable(b, num_components, bit_size, addr, align);
}

static nir_def *
load_per_draw(nir_builder *b, unsigned num_components, unsigned bit_size,
              nir_def *offset, unsigned align)
{
   nir_def *per_draw = nir_load_per_draw_ptr_kk(b, 1, 64);

   /* We've bound the address of the per-draw data, index in. */
   nir_def *addr = nir_iadd(b, per_draw, nir_u2u64(b, offset));

   return load_speculatable(b, num_components, bit_size, addr, align);
}

/* helper macro for computing root descriptor byte offsets */
#define kk_root_descriptor_offset(member)                                      \
   offsetof(struct kk_root_descriptor_table, member)

/* helper macro for computing per-draw data byte offsets */
#define kk_per_draw_offset(member) offsetof(struct kk_per_draw_data, member)

static nir_def *
load_descriptor_set_addr(nir_builder *b, uint32_t set,
                         UNUSED const struct lower_descriptors_ctx *ctx)
{
   uint32_t set_addr_offset =
      kk_root_descriptor_offset(sets) + set * sizeof(uint64_t);

   return load_root(b, 1, 64, nir_imm_int(b, set_addr_offset), 8);
}

static nir_def *
load_dynamic_buffer_start(nir_builder *b, uint32_t set,
                          const struct lower_descriptors_ctx *ctx)
{
   int dynamic_buffer_start_imm = 0;
   for (uint32_t s = 0; s < set; s++) {
      if (ctx->set_layouts[s] == NULL) {
         dynamic_buffer_start_imm = -1;
         break;
      }

      dynamic_buffer_start_imm +=
         ctx->set_layouts[s]->vk.dynamic_descriptor_count;
   }

   if (dynamic_buffer_start_imm >= 0) {
      return nir_imm_int(b, dynamic_buffer_start_imm);
   } else {
      uint32_t root_offset =
         kk_root_descriptor_offset(set_dynamic_buffer_start) + set;

      return nir_u2u32(b, load_root(b, 1, 8, nir_imm_int(b, root_offset), 1));
   }
}

static nir_def *
load_descriptor(nir_builder *b, unsigned num_components, unsigned bit_size,
                uint32_t set, uint32_t binding, nir_def *index,
                unsigned offset_B, const struct lower_descriptors_ctx *ctx)
{
   const struct kk_descriptor_set_binding_layout *binding_layout =
      get_binding_layout(set, binding, ctx);

   if (ctx->clamp_desc_array_bounds)
      index =
         nir_umin(b, index, nir_imm_int(b, binding_layout->array_size - 1));

   switch (binding_layout->type) {
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC: {
      /* Get the index in the root descriptor table dynamic_buffers array. */
      nir_def *dynamic_buffer_start = load_dynamic_buffer_start(b, set, ctx);

      index = nir_iadd(b, index,
                       nir_iadd_imm(b, dynamic_buffer_start,
                                    binding_layout->dynamic_buffer_index));

      nir_def *root_desc_offset = nir_iadd_imm(
         b, nir_imul_imm(b, index, sizeof(struct kk_buffer_address)),
         kk_root_descriptor_offset(dynamic_buffers));

      assert(num_components == 4 && bit_size == 32);
      nir_def *desc = load_root(b, 4, 32, root_desc_offset, 16);

      /* We know a priori that the the .w compnent (offset) is zero */
      return nir_vector_insert_imm(b, desc, nir_imm_int(b, 0), 3);
   }

   case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK: {
      nir_def *base_addr = nir_iadd_imm(
         b, load_descriptor_set_addr(b, set, ctx), binding_layout->offset);

      assert(binding_layout->stride == 1);
      const uint32_t binding_size = binding_layout->array_size;

      /* Convert it to nir_address_format_64bit_bounded_global */
      assert(num_components == 4 && bit_size == 32);
      return nir_vec4(b, nir_unpack_64_2x32_split_x(b, base_addr),
                      nir_unpack_64_2x32_split_y(b, base_addr),
                      nir_imm_int(b, binding_size), nir_imm_int(b, 0));
   }

   default: {
      assert(binding_layout->stride > 0);
      nir_def *desc_ubo_offset =
         nir_iadd_imm(b, nir_imul_imm(b, index, binding_layout->stride),
                      binding_layout->offset + offset_B);

      unsigned desc_align_mul = (1 << (ffs(binding_layout->stride) - 1));
      desc_align_mul = MIN2(desc_align_mul, 16);
      unsigned desc_align_offset = binding_layout->offset + offset_B;
      desc_align_offset %= desc_align_mul;

      nir_def *desc;
      nir_def *set_addr = load_descriptor_set_addr(b, set, ctx);
      desc = nir_load_global_constant_offset(
         b, num_components, bit_size, set_addr, desc_ubo_offset,
         .align_mul = desc_align_mul, .align_offset = desc_align_offset,
         .access = ACCESS_CAN_SPECULATE);

      if (binding_layout->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
          binding_layout->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
         /* We know a priori that the the .w compnent (offset) is zero */
         assert(num_components == 4 && bit_size == 32);
         desc = nir_vector_insert_imm(b, desc, nir_imm_int(b, 0), 3);
      }
      return desc;
   }
   }
}

static bool
is_idx_intrin(nir_intrinsic_instr *intrin)
{
   while (intrin->intrinsic == nir_intrinsic_vulkan_resource_reindex) {
      intrin = nir_src_as_intrinsic(intrin->src[0]);
      if (intrin == NULL)
         return false;
   }

   return intrin->intrinsic == nir_intrinsic_vulkan_resource_index;
}

static nir_def *
load_descriptor_for_idx_intrin(nir_builder *b, nir_intrinsic_instr *intrin,
                               const struct lower_descriptors_ctx *ctx)
{
   nir_def *index = nir_imm_int(b, 0);

   while (intrin->intrinsic == nir_intrinsic_vulkan_resource_reindex) {
      index = nir_iadd(b, index, intrin->src[1].ssa);
      intrin = nir_src_as_intrinsic(intrin->src[0]);
   }

   assert(intrin->intrinsic == nir_intrinsic_vulkan_resource_index);
   uint32_t set = nir_intrinsic_desc_set(intrin);
   uint32_t binding = nir_intrinsic_binding(intrin);
   index = nir_iadd(b, index, intrin->src[0].ssa);

   return load_descriptor(b, 4, 32, set, binding, index, 0, ctx);
}

static bool
try_lower_load_vulkan_descriptor(nir_builder *b, nir_intrinsic_instr *intrin,
                                 const struct lower_descriptors_ctx *ctx)
{
   ASSERTED const nir_descriptor_type desc_type =
      nir_intrinsic_desc_type(intrin);
   b->cursor = nir_before_instr(&intrin->instr);

   nir_intrinsic_instr *idx_intrin = nir_src_as_intrinsic(intrin->src[0]);
   if (idx_intrin == NULL || !is_idx_intrin(idx_intrin)) {
      assert(desc_type == nir_descriptor_type_storage_buffer);
      return false;
   }

   nir_def *desc = load_descriptor_for_idx_intrin(b, idx_intrin, ctx);

   nir_def_rewrite_uses(&intrin->def, desc);

   return true;
}

static bool
_lower_sysval_to_root_table(nir_builder *b, nir_intrinsic_instr *intrin,
                            uint32_t root_table_offset)
{
   b->cursor = nir_instr_remove(&intrin->instr);
   assert((root_table_offset & 3) == 0 && "aligned");

   nir_def *val = load_root(b, intrin->def.num_components, intrin->def.bit_size,
                            nir_imm_int(b, root_table_offset), 4);

   nir_def_rewrite_uses(&intrin->def, val);

   return true;
}

#define lower_sysval_to_root_table(b, intrin, member)                          \
   _lower_sysval_to_root_table(b, intrin, kk_root_descriptor_offset(member))

static bool
_lower_sysval_to_per_draw(nir_builder *b, nir_intrinsic_instr *intrin,
                          uint32_t per_draw_offset)
{
   b->cursor = nir_instr_remove(&intrin->instr);
   assert((per_draw_offset & 3) == 0 && "aligned");

   nir_def *val =
      load_per_draw(b, intrin->def.num_components, intrin->def.bit_size,
                    nir_imm_int(b, per_draw_offset), 4);

   nir_def_rewrite_uses(&intrin->def, val);

   return true;
}

#define lower_sysval_to_per_draw(b, intrin, member)                            \
   _lower_sysval_to_per_draw(b, intrin, kk_per_draw_offset(member))

static bool
lower_load_push_constant(nir_builder *b, nir_intrinsic_instr *load,
                         const struct lower_descriptors_ctx *ctx)
{
   const uint32_t push_region_offset = kk_root_descriptor_offset(push);
   const uint32_t base = nir_intrinsic_base(load);

   b->cursor = nir_before_instr(&load->instr);

   nir_def *offset =
      nir_iadd_imm(b, load->src[0].ssa, push_region_offset + base);

   nir_def *val = load_root(b, load->def.num_components, load->def.bit_size,
                            offset, load->def.bit_size / 8);

   nir_def_rewrite_uses(&load->def, val);

   return true;
}

static void
get_resource_deref_binding(nir_builder *b, nir_deref_instr *deref,
                           uint32_t *set, uint32_t *binding, nir_def **index)
{
   if (deref->deref_type == nir_deref_type_array) {
      *index = deref->arr.index.ssa;
      deref = nir_deref_instr_parent(deref);
   } else {
      *index = nir_imm_int(b, 0);
   }

   assert(deref->deref_type == nir_deref_type_var);
   nir_variable *var = deref->var;

   *set = var->data.descriptor_set;
   *binding = var->data.binding;
}

static nir_def *
load_resource_addr(nir_builder *b, unsigned num_components, unsigned bit_size,
                   nir_deref_instr *deref, unsigned offset_B,
                   const struct lower_descriptors_ctx *ctx)
{
   uint32_t set, binding;
   nir_def *index;
   get_resource_deref_binding(b, deref, &set, &binding, &index);

   const struct kk_descriptor_set_binding_layout *binding_layout =
      get_binding_layout(set, binding, ctx);

   if (ctx->clamp_desc_array_bounds)
      index =
         nir_umin(b, index, nir_imm_int(b, binding_layout->array_size - 1));

   assert(binding_layout->stride > 0);
   nir_def *desc_ubo_offset =
      nir_iadd_imm(b, nir_imul_imm(b, index, binding_layout->stride),
                   binding_layout->offset + offset_B);

   return nir_iadd(b, load_descriptor_set_addr(b, set, ctx),
                   nir_u2u64(b, desc_ubo_offset));
}

static nir_def *
load_resource_deref_desc(nir_builder *b, unsigned num_components,
                         unsigned bit_size, nir_deref_instr *deref,
                         unsigned offset_B,
                         const struct lower_descriptors_ctx *ctx)
{
   uint32_t set, binding;
   nir_def *index;
   get_resource_deref_binding(b, deref, &set, &binding, &index);
   return load_descriptor(b, num_components, bit_size, set, binding, index,
                          offset_B, ctx);
}

/* Residency of texel x (from the start of the Metal texture) of a texel
 * buffer: Metal has no sparse read of texture buffers, so views of sparse
 * residency buffers read their page's tile in the buffer's residency texture
 * (struct kk_texel_buffer_descriptor); other views are always resident. */
static nir_def *
load_texel_buffer_residency(nir_builder *b, nir_deref_instr *deref,
                            nir_def *x, const struct lower_descriptors_ctx *ctx)
{
   const unsigned offs =
      offsetof(struct kk_texel_buffer_descriptor, residency_gpu_resource_id);
   nir_def *desc = load_resource_deref_desc(b, 4, 32, deref, offs, ctx);
   nir_def *first_page = load_resource_deref_desc(
      b, 1, 32, deref,
      offsetof(struct kk_texel_buffer_descriptor, residency_first_page), ctx);

   nir_def *handle = nir_load_texture_handle_kk(
      b, 1, 64, load_resource_addr(b, 1, 64, deref, offs, ctx),
      .dest_type = nir_type_uint32, .image_dim = GLSL_SAMPLER_DIM_2D,
      .image_array = true, .access = ACCESS_NON_WRITEABLE,
      .flags = MSL_ACCESS_READ);
   nir_def *page = nir_umin(
      b,
      nir_iadd(b, first_page,
               nir_ushr(b, nir_u2u32(b, x), nir_channel(b, desc, 2))),
      nir_channel(b, desc, 3));
   const unsigned row_mask = (1u << KK_SPARSE_RESIDENCY_ROW_TILES_LOG2) - 1u;
   const unsigned rows_mask = (1u << (KK_SPARSE_RESIDENCY_LAYER_TILES_LOG2 -
                                      KK_SPARSE_RESIDENCY_ROW_TILES_LOG2)) -
                              1u;
   nir_def *coord = nir_vec4(
      b,
      nir_ishl_imm(b, nir_iand_imm(b, page, row_mask),
                   KK_SPARSE_RESIDENCY_TILE_PX_LOG2),
      nir_ishl_imm(b,
                   nir_iand_imm(b,
                                nir_ushr_imm(b, page,
                                             KK_SPARSE_RESIDENCY_ROW_TILES_LOG2),
                                rows_mask),
                   KK_SPARSE_RESIDENCY_TILE_PX_LOG2),
      nir_ushr_imm(b, page, KK_SPARSE_RESIDENCY_LAYER_TILES_LOG2),
      nir_imm_int(b, 0));
   nir_def *zero = nir_imm_int(b, 0), *resident = nir_imm_int(b, 1);
   nir_push_if(b, nir_ine_imm(b, nir_ior(b, nir_channel(b, desc, 0),
                                         nir_channel(b, desc, 1)),
                              0));
   nir_def *read = nir_bindless_image_sparse_load(
      b, 5, 32, handle, coord, zero, zero, .image_dim = GLSL_SAMPLER_DIM_2D,
      .image_array = true, .dest_type = nir_type_uint32,
      .access = ACCESS_NON_WRITEABLE);
   nir_def *code = nir_channel(b, read, 4);
   nir_pop_if(b, NULL);
   return nir_if_phi(b, code, resident);
}

/* ANDs a residency code into the last component of a sparse load's def */
static void
and_sparse_residency(nir_builder *b, nir_instr *instr, nir_def *def,
                     nir_def *residency)
{
   b->cursor = nir_after_instr(instr);
   const unsigned code_comp = def->num_components - 1;
   nir_def *code = nir_iand(b, nir_channel(b, def, code_comp), residency);
   nir_def_rewrite_uses_after(def, nir_vector_insert_imm(b, def, code, code_comp));
}

/* Replaces the residency code (last component, read as old_code right after
 * def) of a sparse load's def with new_code, at the cursor: all other uses of
 * def come after it (code emitted in between is in its own control flow). */
static void
replace_sparse_residency(nir_builder *b, nir_def *def, nir_def *old_code,
                         nir_def *new_code)
{
   nir_def *val =
      nir_vector_insert_imm(b, def, new_code, def->num_components - 1);
   nir_foreach_use_including_if_safe(use, def) {
      if (nir_src_is_if(use) ||
          (nir_src_use_instr(use) != nir_def_instr(val) &&
           nir_src_use_instr(use) != nir_def_instr(old_code)))
         nir_src_rewrite(use, val);
   }
}

static nir_def *
sparse_residency_code(nir_builder *b, nir_def *def)
{
   return nir_channel(b, def, def->num_components - 1);
}

/* Residency of a sparse imageLoad through a storage view with
 * baseMipLevel > 0 (struct kk_storage_image_descriptor): the same load from
 * the residency view, at the level plus the base. */
static void
lower_storage_view_residency(nir_builder *b, nir_intrinsic_instr *intr,
                             nir_deref_instr *deref, nir_alu_type type,
                             const struct lower_descriptors_ctx *ctx)
{
   const unsigned offs =
      offsetof(struct kk_storage_image_descriptor, residency_gpu_resource_id);
   const enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   const bool array = nir_intrinsic_image_array(intr);

   b->cursor = nir_after_instr(&intr->instr);
   nir_def *id = load_resource_deref_desc(b, 2, 32, deref, offs, ctx);
   nir_def *code = sparse_residency_code(b, &intr->def);
   nir_push_if(
      b, nir_ine_imm(
            b, nir_ior(b, nir_channel(b, id, 0), nir_channel(b, id, 1)), 0));
   nir_def *residency = nir_load_texture_handle_kk(
      b, 1, 64, load_resource_addr(b, 1, 64, deref, offs, ctx),
      .dest_type = type, .image_dim = dim, .image_array = array,
      .access = ACCESS_NON_WRITEABLE, .flags = MSL_ACCESS_READ);
   nir_def *base = nir_isub(
      b,
      nir_bindless_image_levels(b, 32, residency, .image_dim = dim,
                                .image_array = array),
      nir_bindless_image_levels(b, 32, intr->src[0].ssa, .image_dim = dim,
                                .image_array = array));
   nir_def *lod = intr->src[3].ssa;
   nir_def *read = nir_bindless_image_sparse_load(
      b, intr->def.num_components, intr->def.bit_size, residency,
      intr->src[1].ssa, intr->src[2].ssa,
      nir_iadd(b, lod, nir_u2uN(b, base, lod->bit_size)), .image_dim = dim,
      .image_array = array, .dest_type = type, .access = ACCESS_NON_WRITEABLE);
   nir_def *view_code = sparse_residency_code(b, read);
   nir_pop_if(b, NULL);
   replace_sparse_residency(b, &intr->def, code,
                            nir_if_phi(b, view_code, code));
}

static bool
lower_image_intrin(nir_builder *b, nir_intrinsic_instr *intr,
                   const struct lower_descriptors_ctx *ctx)
{
   b->cursor = nir_before_instr(&intr->instr);
   nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);

   /* Reads and queries use the texture descriptor; writes and atomics PBE. */
   unsigned offs =
      offsetof(struct kk_storage_image_descriptor, image_gpu_resource_id);

   nir_def *resource_addr = load_resource_addr(b, 1, 64, deref, offs, ctx);

   nir_alu_type type;
   if (nir_intrinsic_has_atomic_op(intr)) {
      type = nir_atomic_op_type(nir_intrinsic_atomic_op(intr));
      type |= intr->src[3].ssa->bit_size;
   } else if (nir_intrinsic_has_dest_type(intr)) {
      type = nir_intrinsic_dest_type(intr);
   } else if (nir_intrinsic_has_src_type(intr)) {
      type = nir_intrinsic_src_type(intr);
   } else {
      type = nir_type_uint32;
   }

   nir_variable *var = nir_deref_instr_get_variable(deref);
   nir_def *handle = nir_load_texture_handle_kk(
      b, 1, 64, resource_addr, .dest_type = type,
      .image_dim = nir_intrinsic_image_dim(intr),
      .image_array = nir_intrinsic_image_array(intr),
      .access = nir_intrinsic_access(intr) | var->data.access,
      .flags = msl_convert_access_flag(var->data.access));

   /* Texel buffers: the Metal texture starts texel_offset texels before the
    * view (struct kk_texel_buffer_descriptor).
    */
   nir_def *texel_offset = NULL, *residency = NULL;
   if (nir_intrinsic_image_dim(intr) == GLSL_SAMPLER_DIM_BUF) {
      texel_offset = load_resource_deref_desc(
         b, 1, 32, deref,
         offsetof(struct kk_texel_buffer_descriptor, texel_offset), ctx);
      if (intr->intrinsic != nir_intrinsic_image_deref_size) {
         nir_def *coord = intr->src[1].ssa;
         nir_def *x = nir_iadd(b, nir_channel(b, coord, 0),
                               nir_u2uN(b, texel_offset, coord->bit_size));
         nir_src_rewrite(&intr->src[1], nir_vector_insert_imm(b, coord, x, 0));
         if (intr->intrinsic == nir_intrinsic_image_deref_sparse_load)
            residency = load_texel_buffer_residency(b, deref, x, ctx);
      }
   }

   nir_rewrite_image_intrinsic(intr, handle, nir_image_intrinsic_type_bindless);

   if (residency)
      and_sparse_residency(b, &intr->instr, &intr->def, residency);

   if (intr->intrinsic == nir_intrinsic_bindless_image_sparse_load &&
       nir_intrinsic_image_dim(intr) != GLSL_SAMPLER_DIM_BUF &&
       nir_intrinsic_image_dim(intr) != GLSL_SAMPLER_DIM_MS)
      lower_storage_view_residency(b, intr, deref, type, ctx);

   if (texel_offset && intr->intrinsic == nir_intrinsic_bindless_image_size) {
      b->cursor = nir_after_instr(&intr->instr);
      nir_def *size = &intr->def;
      nir_def *x = nir_isub(b, nir_channel(b, size, 0),
                            nir_u2uN(b, texel_offset, size->bit_size));
      nir_def_rewrite_uses_after(size, nir_vector_insert_imm(b, size, x, 0));
   }

   return true;
}

static bool
try_lower_intrin(nir_builder *b, nir_intrinsic_instr *intrin,
                 const struct lower_descriptors_ctx *ctx)
{
   switch (intrin->intrinsic) {
   case nir_intrinsic_load_sampler_state_kk: {
      b->cursor = nir_before_instr(&intrin->instr);
      nir_def *state = load_resource_deref_desc(
         b, intrin->def.num_components, intrin->def.bit_size,
         nir_src_as_deref(intrin->src[0]), nir_intrinsic_base(intrin), ctx);
      nir_def_replace(&intrin->def, state);
      return true;
   }
   case nir_intrinsic_load_vulkan_descriptor:
      return try_lower_load_vulkan_descriptor(b, intrin, ctx);

   case nir_intrinsic_load_workgroup_size:
      UNREACHABLE("Should have been lowered by nir_lower_cs_intrinsics()");

   case nir_intrinsic_load_base_workgroup_id:
      return lower_sysval_to_root_table(b, intrin, cs.base_group);

   case nir_intrinsic_load_blend_const_color_rgba:
      return lower_sysval_to_root_table(b, intrin, draw.blend_constant);

   case nir_intrinsic_load_clip_z_coeff:
      return lower_sysval_to_root_table(b, intrin, draw.clip_z_coeff);

   case nir_intrinsic_load_is_depth_clamp_emulated_kk: {
      unsigned offset = kk_root_descriptor_offset(draw.emulate_depth_clamp);
      b->cursor = nir_instr_remove(&intrin->instr);
      nir_def *val = load_root(b, 1, 8, nir_imm_int(b, offset), 1);
      nir_def_rewrite_uses(&intrin->def, nir_ieq_imm(b, val, 1));
      return true;
   }

   case nir_intrinsic_load_is_viewport_z_transform_emulated_kk: {
      unsigned offset = kk_root_descriptor_offset(draw.emulate_viewport_z);
      b->cursor = nir_instr_remove(&intrin->instr);
      nir_def *val = load_root(b, 1, 8, nir_imm_int(b, offset), 1);
      nir_def_rewrite_uses(&intrin->def, nir_ieq_imm(b, val, 1));
      return true;
   }

   case nir_intrinsic_load_viewport_z_range_kk: {
      unsigned offset = kk_root_descriptor_offset(draw.viewport_z_range);
      assert((offset & 3) == 0 && "aligned");
      b->cursor = nir_instr_remove(&intrin->instr);
      nir_def *vp_offset =
         nir_imul_imm(b, intrin->src[0].ssa, sizeof(float) * 2u);
      nir_def *root_offset = nir_iadd_imm(b, vp_offset, offset);
      nir_def *val = load_root(b, 2, 32, root_offset, 4);
      nir_def_rewrite_uses(&intrin->def, val);
      return true;
   }

   case nir_intrinsic_load_push_constant:
      return lower_load_push_constant(b, intrin, ctx);

   case nir_intrinsic_load_draw_id:
      return lower_sysval_to_per_draw(b, intrin, draw_id);

   case nir_intrinsic_image_deref_load:
   case nir_intrinsic_image_deref_sparse_load:
   case nir_intrinsic_image_deref_store:
   case nir_intrinsic_image_deref_atomic:
   case nir_intrinsic_image_deref_atomic_swap:
   case nir_intrinsic_image_deref_size:
   case nir_intrinsic_image_deref_samples:
   case nir_intrinsic_image_deref_store_block_agx:
   case nir_intrinsic_image_deref_levels:
      return lower_image_intrin(b, intrin, ctx);

   default:
      return false;
   }
}

static nir_def *
tex_src_def(nir_tex_instr *tex, nir_tex_src_type type)
{
   const int i = nir_tex_instr_src_index(tex, type);
   return i < 0 ? NULL : tex->src[i].src.ssa;
}

static void
set_tex_src(nir_tex_instr *tex, nir_tex_src_type type, nir_def *def)
{
   nir_steal_tex_src(tex, type);
   if (def)
      nir_tex_instr_add_src(tex, type, def);
}

/* txs (level 0, as float) or query_levels of texture handle `handle` with
 * tex's type */
static nir_def *
tex_query(nir_builder *b, nir_tex_instr *tex, nir_texop op, nir_def *handle,
          unsigned num_components)
{
   const bool txs = op == nir_texop_txs;
   nir_tex_instr *query = nir_tex_instr_create(b->shader, txs ? 2 : 1);
   query->op = op;
   query->sampler_dim = tex->sampler_dim;
   query->is_array = tex->is_array;
   query->is_shadow = tex->is_shadow;
   query->dest_type = nir_type_int32;
   query->src[0] = nir_tex_src_for_ssa(nir_tex_src_texture_handle, handle);
   if (txs)
      query->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_int(b, 0));
   nir_def_init(&query->instr, &query->def, nir_tex_instr_dest_size(query), 32);
   nir_builder_instr_insert(b, &query->instr);
   return nir_i2f32(b, nir_trim_vector(b, &query->def, num_components));
}

/* A copy of sparse sample/fetch `tex` from texture `texture` (and sampler
 * `sampler` if not NULL) with op `op` and these LOD sources, at the cursor;
 * returns its residency code */
static nir_def *
residency_tex(nir_builder *b, nir_tex_instr *tex, nir_texop op,
              nir_def *texture, nir_def *sampler, nir_def *lod, nir_def *ddx,
              nir_def *ddy, nir_def *min_lod)
{
   nir_tex_instr *copy =
      nir_instr_as_tex(nir_instr_clone(b->shader, &tex->instr));
   nir_builder_instr_insert(b, &copy->instr);
   copy->op = op;
   set_tex_src(copy, nir_tex_src_texture_handle, texture);
   if (sampler)
      set_tex_src(copy, nir_tex_src_sampler_handle, sampler);
   set_tex_src(copy, nir_tex_src_bias, NULL);
   set_tex_src(copy, nir_tex_src_lod, lod);
   set_tex_src(copy, nir_tex_src_ddx, ddx);
   set_tex_src(copy, nir_tex_src_ddy, ddy);
   set_tex_src(copy, nir_tex_src_min_lod, min_lod);
   return sparse_residency_code(b, &copy->def);
}

/* Residency of sparse gather `tex` as that of a bilinear sample (struct
 * kk_sampled_image_descriptor::sampler_gather_index) of level `lod` of
 * `texture`: Metal reports a gather resident when texels of its footprint
 * across a tile edge are not (their values are zero). */
static nir_def *
gather_residency(nir_builder *b, nir_tex_instr *tex, nir_def *texture,
                 nir_deref_instr *sampler, nir_def *lod,
                 unsigned plane_offset_B,
                 const struct lower_descriptors_ctx *ctx)
{
   nir_def *gather = nir_load_sampler_handle_kk(
      b, load_resource_deref_desc(
            b, 1, 16, sampler,
            plane_offset_B + offsetof(struct kk_sampled_image_descriptor,
                                      sampler_gather_index),
            ctx));
   return residency_tex(b, tex, nir_texop_txl, texture, gather, lod, NULL, NULL,
                        NULL);
}

/* Residency of sparse samples and fetches through views with
 * baseMipLevel > 0 (struct kk_sampled_image_descriptor): Metal measures it
 * for the image's level or LOD without the base (gradient and implicit LODs
 * are computed from the image's size, so only their clamps miss the base).
 * The same operation on the residency view, which starts at level 0, at
 * level/LOD + base gives it:
 *  - fetches: level + base;
 *  - explicit LODs: max(LOD, the sampler's minLod and MinLod), then its
 *    maxLod (Metal's order; the residency sampler has no clamps), + base;
 *  - gradients and implicit LODs (as gradients of the coordinates, taken
 *    where the sample is): gradients scaled by the size ratio of the views'
 *    level 0 times 2^(base + bias), so that the LOD is the view's + base,
 *    the sampler's minLod and MinLod as min_lod_clamp + base; when the
 *    view's isotropic LOD is past the sampler's maxLod, an explicit LOD at
 *    maxLod + base (anisotropic LODs are lower: approximate there; not for
 *    cubes);
 *  - gathers: a bilinear sample (LINEAR, unclamped sampler) of level base,
 *    whose footprint is the gather's; also for other views, at level 0
 *    (gather_residency).
 *  - min/max reduction: its LOD limited to the view's levels, + base (the
 *    reduction samplers have no clamps).
 * Unnormalized coordinates are not handled (their samplers aren't
 * mipmapped: the residency view's level 0 only; D3D has none).
 */
static void
lower_sampled_view_residency(nir_builder *b, nir_tex_instr *tex,
                             nir_deref_instr *texture, nir_deref_instr *sampler,
                             unsigned plane_offset_B,
                             const struct lower_descriptors_ctx *ctx)
{
   const bool minmax = tex->backend_flags & MSL_TEX_FLAG_MINMAX;
   const bool cube = tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE;
   const unsigned dims = cube ? 3 : 2;
   nir_def *view = tex_src_def(tex, nir_tex_src_texture_handle);
   nir_def *lod = tex_src_def(tex, nir_tex_src_lod);
   nir_def *bias = tex_src_def(tex, nir_tex_src_bias);
   nir_def *ddx = tex_src_def(tex, nir_tex_src_ddx);
   nir_def *ddy = tex_src_def(tex, nir_tex_src_ddy);
   nir_def *min_lod = tex_src_def(tex, nir_tex_src_min_lod);

   b->cursor = nir_before_instr(&tex->instr);
   if (!minmax && (tex->op == nir_texop_tex || tex->op == nir_texop_txb)) {
      if (b->shader->info.stage == MESA_SHADER_FRAGMENT) {
         nir_def *coord =
            nir_trim_vector(b, tex_src_def(tex, nir_tex_src_coord), dims);
         ddx = nir_ddx(b, coord);
         ddy = nir_ddy(b, coord);
      } else {
         lod = bias ? bias : nir_imm_float(b, 0.0f);
         bias = NULL;
      }
   }

   b->cursor = nir_after_instr(&tex->instr);
   const unsigned offs =
      plane_offset_B +
      offsetof(struct kk_sampled_image_descriptor, residency_gpu_resource_id);
   nir_def *id = load_resource_deref_desc(b, 2, 32, texture, offs, ctx);
   nir_def *code = sparse_residency_code(b, &tex->def);
   nir_if *nif = nir_push_if(
      b, nir_ine_imm(
            b, nir_ior(b, nir_channel(b, id, 0), nir_channel(b, id, 1)), 0));
   nir_def *addr = load_resource_addr(b, 1, 64, texture, offs, ctx);
   nir_def *residency =
      tex->is_shadow
         ? nir_load_depth_texture_kk(b, 1, 64, addr,
                                     .image_dim = tex->sampler_dim,
                                     .image_array = tex->is_array)
         : nir_load_texture_handle_kk(
              b, 1, 64, addr, .dest_type = tex->dest_type,
              .image_dim = tex->sampler_dim, .image_array = tex->is_array);
   nir_def *base = nir_u2u32(
      b, load_resource_deref_desc(
            b, 1, 16, texture,
            plane_offset_B + offsetof(struct kk_sampled_image_descriptor,
                                      residency_base_level),
            ctx));
   nir_def *fbase = nir_u2f32(b, base);

   nir_def *view_code;
   if (tex->op == nir_texop_txf) {
      lod = lod ? lod : nir_imm_int(b, 0);
      lod = nir_iadd(b, lod, nir_u2uN(b, base, lod->bit_size));
      view_code = residency_tex(b, tex, nir_texop_txf, residency, NULL, lod,
                                NULL, NULL, NULL);
   } else if (tex->op == nir_texop_tg4) {
      view_code = gather_residency(b, tex, residency, sampler, fbase,
                                   plane_offset_B, ctx);
   } else if (minmax) {
      nir_def *last = nir_fadd_imm(
         b, tex_query(b, tex, nir_texop_query_levels, view, 1), -1.0f);
      lod = nir_fadd(b, nir_fmin(b, nir_fmax_imm(b, lod, 0.0f), last), fbase);
      view_code = residency_tex(b, tex, nir_texop_txl, residency, NULL, lod,
                                NULL, NULL, NULL);
   } else {
      assert(sampler);
      nir_def *clamp = nir_f2f32(
         b, load_resource_deref_desc(
               b, 2, 16, sampler,
               plane_offset_B + offsetof(struct kk_sampled_image_descriptor,
                                         sampler_lod_min_fp16),
               ctx));
      nir_def *smin = nir_channel(b, clamp, 0);
      nir_def *smax = nir_channel(b, clamp, 1);
      nir_def *unclamped = nir_load_sampler_handle_kk(
         b, load_resource_deref_desc(
               b, 1, 16, sampler,
               plane_offset_B + offsetof(struct kk_sampled_image_descriptor,
                                         sampler_residency_index),
               ctx));
      /* Metal's order (as the sample itself): max(minLod, MinLod), then
       * maxLod */
      nir_def *lo = smin;
      if (min_lod)
         lo = nir_fmax(b, lo, nir_f2f32(b, min_lod));

      /* Explicit LOD (or maxLod, below) */
      nir_def *explicit_lod = NULL;
      if (!ddx || !cube) {
         nir_def *l = ddx ? smax : nir_f2f32(b, lod);
         explicit_lod =
            nir_fadd(b, nir_fmin(b, nir_fmax(b, l, lo), smax), fbase);
      }

      if (!ddx) {
         view_code = residency_tex(b, tex, nir_texop_txl, residency, unclamped,
                                   explicit_lod, NULL, NULL, NULL);
      } else {
         const unsigned size_components = cube ? 2 : dims;
         nir_def *residency_size =
            tex_query(b, tex, nir_texop_txs, residency, size_components);
         nir_def *scale =
            nir_fdiv(b, tex_query(b, tex, nir_texop_txs, view, size_components),
                     residency_size);
         if (cube)
            scale = nir_channel(b, scale, 0);
         scale =
            nir_fmul(b, scale,
                     nir_fexp2(b, bias ? nir_fadd(b, fbase, nir_f2f32(b, bias))
                                       : fbase));
         nir_def *dx = nir_fmul(b, nir_f2f32(b, ddx), scale);
         nir_def *dy = nir_fmul(b, nir_f2f32(b, ddy), scale);
         nir_def *min_lod_clamp = nir_fadd(b, lo, fbase);
         if (cube) {
            view_code = residency_tex(b, tex, nir_texop_txd, residency,
                                      unclamped, NULL, dx, dy, min_lod_clamp);
         } else {
            /* Isotropic LOD of the view: of the residency view - base */
            nir_def *vx = nir_fmul(b, dx, residency_size);
            nir_def *vy = nir_fmul(b, dy, residency_size);
            nir_def *view_lod = nir_fsub(
               b,
               nir_fmul_imm(b,
                            nir_flog2(b, nir_fmax(b, nir_fdot(b, vx, vx),
                                                  nir_fdot(b, vy, vy))),
                            0.5f),
               fbase);
            nir_push_if(b, nir_flt(b, smax, nir_fmax(b, view_lod, lo)));
            nir_def *clamped =
               residency_tex(b, tex, nir_texop_txl, residency, unclamped,
                             explicit_lod, NULL, NULL, NULL);
            nir_push_else(b, NULL);
            nir_def *gradient =
               residency_tex(b, tex, nir_texop_txd, residency, unclamped, NULL,
                             dx, dy, min_lod_clamp);
            nir_pop_if(b, NULL);
            view_code = nir_if_phi(b, clamped, gradient);
         }
      }
   }
   nir_def *other_code = code;
   if (tex->op == nir_texop_tg4) {
      nir_push_else(b, nif);
      other_code = gather_residency(b, tex, view, sampler, nir_imm_float(b, 0),
                                    plane_offset_B, ctx);
   }
   nir_pop_if(b, nif);
   replace_sparse_residency(b, &tex->def, code,
                            nir_if_phi(b, view_code, other_code));
}

static bool
lower_tex(nir_builder *b, nir_tex_instr *tex,
          const struct lower_descriptors_ctx *ctx)
{
   b->cursor = nir_before_instr(&tex->instr);

   nir_def *texture = nir_steal_tex_src(tex, nir_tex_src_texture_deref);
   nir_def *sampler = nir_steal_tex_src(tex, nir_tex_src_sampler_deref);
   if (!texture) {
      assert(!sampler);
      return false;
   }

   nir_def *plane_ssa = nir_steal_tex_src(tex, nir_tex_src_plane);
   const uint32_t plane =
      plane_ssa ? nir_src_as_uint(nir_src_for_ssa(plane_ssa)) : 0;
   const uint64_t plane_offset_B =
      plane * sizeof(struct kk_sampled_image_descriptor);

   /* LOD bias is passed in the descriptor set, rather than embedded into
    * the sampler descriptor. There's no spot in the hardware descriptor,
    * plus this saves on precious sampler heap spots.
    */
   if (tex->op == nir_texop_lod_bias) {
      unsigned offs =
         offsetof(struct kk_sampled_image_descriptor, sampler_lod_bias_fp16);

      nir_def *bias = load_resource_deref_desc(
         b, 1, 16, nir_src_as_deref(nir_src_for_ssa(sampler)),
         plane_offset_B + offs, ctx);

      nir_def_replace(&tex->def, bias);
      return true;
   }

   if (tex->op == nir_texop_image_min_lod_agx) {
      assert(tex->dest_type == nir_type_float16 ||
             tex->dest_type == nir_type_uint16);

      unsigned offs =
         tex->dest_type == nir_type_float16
            ? offsetof(struct kk_sampled_image_descriptor, image_min_lod_fp16)
            : offsetof(struct kk_sampled_image_descriptor,
                       image_min_lod_uint16);

      nir_def *min = load_resource_deref_desc(
         b, 1, 16, nir_src_as_deref(nir_src_for_ssa(texture)),
         plane_offset_B + offs, ctx);

      nir_def_replace(&tex->def, min);
      return true;
   }

   if (tex->op == nir_texop_has_custom_border_color_agx) {
      unsigned offs = offsetof(struct kk_sampled_image_descriptor,
                               clamp_0_sampler_index_or_negative);

      nir_def *res = load_resource_deref_desc(
         b, 1, 16, nir_src_as_deref(nir_src_for_ssa(sampler)),
         plane_offset_B + offs, ctx);

      nir_def_replace(&tex->def, nir_ige_imm(b, res, 0));
      return true;
   }

   if (tex->op == nir_texop_custom_border_color_agx) {
      unsigned offs = offsetof(struct kk_sampled_image_descriptor, border);

      nir_def *border = load_resource_deref_desc(
         b, 4, 32, nir_src_as_deref(nir_src_for_ssa(sampler)),
         plane_offset_B + offs, ctx);

      nir_alu_type T = nir_alu_type_get_base_type(tex->dest_type);
      border = nir_convert_to_bit_size(b, border, T, tex->def.bit_size);

      nir_def_replace(&tex->def, border);
      return true;
   }

   {
      unsigned offs =
         offsetof(struct kk_sampled_image_descriptor, image_gpu_resource_id);

      nir_def *resource_addr = load_resource_addr(
         b, 1, 64, nir_src_as_deref(nir_src_for_ssa(texture)),
         plane_offset_B + offs, ctx);

      nir_def *handle = NULL;
      if (tex->is_shadow) {
         handle = nir_load_depth_texture_kk(b, 1, 64, resource_addr,
                                            .image_dim = tex->sampler_dim,
                                            .image_array = tex->is_array);
      } else {
         handle = nir_load_texture_handle_kk(
            b, 1, 64, resource_addr, .dest_type = tex->dest_type,
            .image_dim = tex->sampler_dim, .image_array = tex->is_array);
      }
      nir_tex_instr_add_src(tex, nir_tex_src_texture_handle, handle);

      /* Texel buffers: the Metal texture starts texel_offset texels before
       * the view (struct kk_texel_buffer_descriptor).
       */
      if (tex->sampler_dim == GLSL_SAMPLER_DIM_BUF) {
         nir_def *texel_offset = load_resource_deref_desc(
            b, 1, 32, nir_src_as_deref(nir_src_for_ssa(texture)),
            plane_offset_B +
               offsetof(struct kk_texel_buffer_descriptor, texel_offset),
            ctx);
         const int coord_idx = nir_tex_instr_src_index(tex, nir_tex_src_coord);
         if (tex->op == nir_texop_txs) {
            b->cursor = nir_after_instr(&tex->instr);
            nir_def *size = &tex->def;
            nir_def *x = nir_isub(b, nir_channel(b, size, 0),
                                  nir_u2uN(b, texel_offset, size->bit_size));
            nir_def_rewrite_uses_after(size,
                                       nir_vector_insert_imm(b, size, x, 0));
            b->cursor = nir_before_instr(&tex->instr);
         } else if (coord_idx >= 0) {
            nir_def *coord = tex->src[coord_idx].src.ssa;
            nir_def *x = nir_iadd(b, nir_channel(b, coord, 0),
                                  nir_u2uN(b, texel_offset, coord->bit_size));
            nir_src_rewrite(&tex->src[coord_idx].src,
                            nir_vector_insert_imm(b, coord, x, 0));
            if (tex->is_sparse) {
               nir_def *residency = load_texel_buffer_residency(
                  b, nir_src_as_deref(nir_src_for_ssa(texture)), x, ctx);
               and_sparse_residency(b, &tex->instr, &tex->def, residency);
               b->cursor = nir_before_instr(&tex->instr);
            }
         }
      }
   }

   if (sampler != NULL) {
      unsigned offs =
         offsetof(struct kk_sampled_image_descriptor, sampler_index);

      bool clamp_to_0 = tex->backend_flags & KK_TEXTURE_FLAG_CLAMP_TO_0;
      if (clamp_to_0)
         offs = offsetof(struct kk_sampled_image_descriptor,
                         clamp_0_sampler_index_or_negative);

      nir_def *index = load_resource_deref_desc(
         b, 1, 16, nir_src_as_deref(nir_src_for_ssa(sampler)),
         plane_offset_B + offs, ctx);

      nir_tex_instr_add_src(tex, nir_tex_src_sampler_handle,
                            nir_load_sampler_handle_kk(b, index));
   }

   if (tex->op == nir_texop_lod) {
      nir_def *lod_min = nir_f2f32(
         b, load_resource_deref_desc(
               b, 1, 16, nir_src_as_deref(nir_src_for_ssa(sampler)),
               plane_offset_B + offsetof(struct kk_sampled_image_descriptor,
                                         sampler_lod_min_fp16),
               ctx));
      nir_def *lod_max = nir_f2f32(
         b, load_resource_deref_desc(
               b, 1, 16, nir_src_as_deref(nir_src_for_ssa(sampler)),
               plane_offset_B + offsetof(struct kk_sampled_image_descriptor,
                                         sampler_lod_max_fp16),
               ctx));

      nir_tex_instr_add_src(tex, nir_tex_src_min_lod, lod_min);
      nir_tex_instr_add_src(tex, nir_tex_src_max_lod_kk, lod_max);
   }

   if (tex->is_sparse && tex->sampler_dim != GLSL_SAMPLER_DIM_BUF &&
       tex->sampler_dim != GLSL_SAMPLER_DIM_MS && tex->op != nir_texop_txf_ms)
      lower_sampled_view_residency(
         b, tex, nir_src_as_deref(nir_src_for_ssa(texture)),
         sampler ? nir_src_as_deref(nir_src_for_ssa(sampler)) : NULL,
         plane_offset_B, ctx);

   return true;
}

static bool
try_lower_descriptors_instr(nir_builder *b, nir_instr *instr, void *_data)
{
   const struct lower_descriptors_ctx *ctx = _data;

   switch (instr->type) {
   case nir_instr_type_tex:
      return lower_tex(b, nir_instr_as_tex(instr), ctx);
   case nir_instr_type_intrinsic:
      return try_lower_intrin(b, nir_instr_as_intrinsic(instr), ctx);
   default:
      return false;
   }
}

static bool
lower_ssbo_resource_index(nir_builder *b, nir_intrinsic_instr *intrin,
                          const struct lower_descriptors_ctx *ctx)
{
   const nir_descriptor_type desc_type = nir_intrinsic_desc_type(intrin);
   if (desc_type != nir_descriptor_type_storage_buffer)
      return false;

   b->cursor = nir_instr_remove(&intrin->instr);

   uint32_t set = nir_intrinsic_desc_set(intrin);
   uint32_t binding = nir_intrinsic_binding(intrin);
   nir_def *index = intrin->src[0].ssa;

   const struct kk_descriptor_set_binding_layout *binding_layout =
      get_binding_layout(set, binding, ctx);

   nir_def *binding_addr;
   uint8_t binding_stride;
   switch (binding_layout->type) {
   case VK_DESCRIPTOR_TYPE_MUTABLE_EXT:
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: {
      nir_def *set_addr = load_descriptor_set_addr(b, set, ctx);
      binding_addr = nir_iadd_imm(b, set_addr, binding_layout->offset);
      binding_stride = binding_layout->stride;
      break;
   }

   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC: {
      nir_def *root_desc_addr = nir_load_buffer_ptr_kk(b, 1, 64, .binding = 0);

      nir_def *dynamic_buffer_start =
         nir_iadd_imm(b, load_dynamic_buffer_start(b, set, ctx),
                      binding_layout->dynamic_buffer_index);

      nir_def *dynamic_binding_offset =
         nir_iadd_imm(b,
                      nir_imul_imm(b, dynamic_buffer_start,
                                   sizeof(struct kk_buffer_address)),
                      kk_root_descriptor_offset(dynamic_buffers));

      binding_addr =
         nir_iadd(b, root_desc_addr, nir_u2u64(b, dynamic_binding_offset));
      binding_stride = sizeof(struct kk_buffer_address);
      break;
   }

   default:
      UNREACHABLE("Not an SSBO descriptor");
   }

   /* Tuck the stride in the top 8 bits of the binding address */
   binding_addr = nir_ior_imm(b, binding_addr, (uint64_t)binding_stride << 56);

   const uint32_t binding_size = binding_layout->array_size * binding_stride;
   nir_def *offset_in_binding = nir_imul_imm(b, index, binding_stride);

   nir_def *addr = nir_vec4(b, nir_unpack_64_2x32_split_x(b, binding_addr),
                            nir_unpack_64_2x32_split_y(b, binding_addr),
                            nir_imm_int(b, binding_size), offset_in_binding);

   nir_def_rewrite_uses(&intrin->def, addr);

   return true;
}

static bool
lower_ssbo_resource_reindex(nir_builder *b, nir_intrinsic_instr *intrin,
                            const struct lower_descriptors_ctx *ctx)
{
   const nir_descriptor_type desc_type = nir_intrinsic_desc_type(intrin);
   if (desc_type != nir_descriptor_type_storage_buffer)
      return false;

   b->cursor = nir_instr_remove(&intrin->instr);

   nir_def *addr = intrin->src[0].ssa;
   nir_def *index = intrin->src[1].ssa;

   nir_def *addr_high32 = nir_channel(b, addr, 1);
   nir_def *stride = nir_ushr_imm(b, addr_high32, 24);
   nir_def *offset = nir_imul(b, index, stride);

   addr = nir_build_addr_iadd(b, addr, ctx->ssbo_addr_format, nir_var_mem_ssbo,
                              offset);
   nir_def_rewrite_uses(&intrin->def, addr);

   return true;
}

static bool
lower_load_ssbo_descriptor(nir_builder *b, nir_intrinsic_instr *intrin,
                           const struct lower_descriptors_ctx *ctx)
{
   const nir_descriptor_type desc_type = nir_intrinsic_desc_type(intrin);
   if (desc_type != nir_descriptor_type_storage_buffer)
      return false;

   b->cursor = nir_instr_remove(&intrin->instr);

   nir_def *addr = intrin->src[0].ssa;

   nir_def *desc;
   switch (ctx->ssbo_addr_format) {
   case nir_address_format_64bit_global_32bit_offset: {
      nir_def *base = nir_pack_64_2x32(b, nir_trim_vector(b, addr, 2));
      nir_def *offset = nir_channel(b, addr, 3);
      /* Mask off the binding stride */
      base = nir_iand_imm(b, base, BITFIELD64_MASK(56));
      desc = nir_load_global_constant_offset(b, 4, 32, base, offset,
                                             .align_mul = 16, .align_offset = 0,
                                             .access = ACCESS_CAN_SPECULATE);
      break;
   }

   case nir_address_format_64bit_bounded_global: {
      nir_def *base = nir_pack_64_2x32(b, nir_trim_vector(b, addr, 2));
      nir_def *size = nir_channel(b, addr, 2);
      nir_def *offset = nir_channel(b, addr, 3);
      /* Mask off the binding stride */
      base = nir_iand_imm(b, base, BITFIELD64_MASK(56));
      desc = nir_load_global_constant_bounded(
         b, 4, 32, base, offset, size, .align_mul = 16, .align_offset = 0,
         .access = ACCESS_CAN_SPECULATE);
      break;
   }

   default:
      UNREACHABLE("Unknown address mode");
   }

   nir_def_rewrite_uses(&intrin->def, desc);

   return true;
}

static bool
lower_ssbo_descriptor(nir_builder *b, nir_intrinsic_instr *intr, void *_data)
{
   const struct lower_descriptors_ctx *ctx = _data;

   switch (intr->intrinsic) {
   case nir_intrinsic_vulkan_resource_index:
      return lower_ssbo_resource_index(b, intr, ctx);
   case nir_intrinsic_vulkan_resource_reindex:
      return lower_ssbo_resource_reindex(b, intr, ctx);
   case nir_intrinsic_load_vulkan_descriptor:
      return lower_load_ssbo_descriptor(b, intr, ctx);
   default:
      return false;
   }
}

bool
kk_nir_lower_descriptors(nir_shader *nir,
                         const struct vk_pipeline_robustness_state *rs,
                         uint32_t set_layout_count,
                         struct vk_descriptor_set_layout *const *set_layouts)
{
   struct lower_descriptors_ctx ctx = {
      .clamp_desc_array_bounds =
         rs->storage_buffers !=
            VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED ||
         rs->uniform_buffers !=
            VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED ||
         rs->images != VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED,
      .ssbo_addr_format = kk_buffer_addr_format(rs->storage_buffers),
      .ubo_addr_format = kk_buffer_addr_format(rs->uniform_buffers),
   };

   assert(set_layout_count <= KK_MAX_SETS);
   for (uint32_t s = 0; s < set_layout_count; s++) {
      if (set_layouts[s] != NULL)
         ctx.set_layouts[s] = vk_to_kk_descriptor_set_layout(set_layouts[s]);
   }

   /* First lower everything but complex SSBOs, then lower complex SSBOs.
    *
    * TODO: See if we can unify this, not sure if the fast path matters on
    * Apple. This is inherited from NVK.
    */
   /* Sparse texel buffer loads add control flow (load_texel_buffer_residency) */
   bool pass_lower_descriptors = nir_shader_instructions_pass(
      nir, try_lower_descriptors_instr, nir_metadata_none, &ctx);

   bool pass_lower_ssbo = nir_shader_intrinsics_pass(
      nir, lower_ssbo_descriptor, nir_metadata_control_flow, &ctx);

   return pass_lower_descriptors || pass_lower_ssbo;
}

static bool
lower_poly(struct nir_builder *b, nir_intrinsic_instr *intrin, void *data)
{
   switch (intrin->intrinsic) {
   case nir_intrinsic_load_vs_outputs_poly:
      return lower_sysval_to_per_draw(b, intrin, vertex_outputs);
   case nir_intrinsic_load_vertex_param_buffer_poly:
      return lower_sysval_to_per_draw(b, intrin, vertex_params);
   case nir_intrinsic_load_tess_param_buffer_poly:
      return lower_sysval_to_per_draw(b, intrin, tess_params);
   case nir_intrinsic_load_geometry_param_buffer_poly:
      return lower_sysval_to_per_draw(b, intrin, geometry_params);
   case nir_intrinsic_load_provoking_last:
      return lower_sysval_to_per_draw(b, intrin, provoking_last);
   case nir_intrinsic_load_rasterization_stream:
      return lower_sysval_to_per_draw(b, intrin, rasterization_stream);
   case nir_intrinsic_load_index_size_poly:
      return lower_sysval_to_per_draw(b, intrin, index_size);
   case nir_intrinsic_load_first_vertex:
      /* Lower only compute shaders */
      if (*(bool *)data) {
         uint32_t root_table_offset = kk_per_draw_offset(base_vertex_addr);
         b->cursor = nir_instr_remove(&intrin->instr);
         assert((root_table_offset & 3) == 0 && "aligned");

         nir_def *addr = load_per_draw(b, intrin->def.num_components, 64u,
                                       nir_imm_int(b, root_table_offset), 4);

         nir_def *val = nir_load_global(b, 1u, intrin->def.bit_size, addr);

         nir_def_rewrite_uses(&intrin->def, val);
         return true;
      }
      return false;
   case nir_intrinsic_load_base_instance:
      /* Lower only compute shaders */
      if (*(bool *)data) {
         uint32_t root_table_offset = kk_per_draw_offset(base_instance_addr);
         b->cursor = nir_instr_remove(&intrin->instr);
         assert((root_table_offset & 3) == 0 && "aligned");

         nir_def *addr = load_per_draw(b, intrin->def.num_components, 64u,
                                       nir_imm_int(b, root_table_offset), 4);

         nir_def *val = nir_load_global(b, 1u, intrin->def.bit_size, addr);

         nir_def_rewrite_uses(&intrin->def, val);
         return true;
      }
      return false;
   case nir_intrinsic_ro_to_rw_poly:
      /* KK uses the same address for reads and writes. */
      nir_def_replace(&intrin->def, intrin->src[0].ssa);
      return true;
   case nir_intrinsic_load_stat_query_address_poly: {
      /* KK doesn't expose pipeline statistics. Let pre-GS write them to the
       * same scratch word as the unused overflow counters. */
      b->cursor = nir_instr_remove(&intrin->instr);
      nir_def *params = load_per_draw(
         b, 1, 64, nir_imm_int(b, kk_per_draw_offset(geometry_params)), 4);
      nir_def *addr = nir_load_global(
         b, 1, 64,
         nir_iadd_imm(b, params,
                      offsetof(struct poly_geometry_params, xfb_any_overflow)));
      nir_def_rewrite_uses(&intrin->def, addr);
      return true;
   }
   default:
      return false;
   }
}

bool
kk_nir_lower_poly(struct nir_shader *nir)
{
   bool is_compute = nir->info.stage == MESA_SHADER_COMPUTE;
   return nir_shader_intrinsics_pass(nir, lower_poly, nir_metadata_control_flow,
                                     &is_compute);
}

static bool
kk_barrier_is_device_global(mesa_scope scope, nir_variable_mode mode)
{
   if (scope != SCOPE_DEVICE && scope != SCOPE_QUEUE_FAMILY)
      return false;

   return mode & (nir_var_mem_global | nir_var_mem_ssbo);
}

/* Iterates over the shader to find memory barriers that target device or queue
 * families to add a "conditional" extra barrier with a load as conditional to
 * work around the issues in memory_model for M3+. The iteration is done in a
 * reverse order to iterate once since we break and insert new blocks. */
bool
kk_nir_add_device_barrier_workaround(nir_shader *nir)
{
   bool progress = false;

   nir_foreach_function_impl(impl, nir) {
      bool impl_progress = false;
      nir_builder b = nir_builder_create(impl);

      nir_foreach_block_reverse_safe(block, impl) {
         nir_foreach_instr_reverse_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

            if (intrin->intrinsic != nir_intrinsic_barrier)
               continue;

            mesa_scope scope = nir_intrinsic_memory_scope(intrin);
            nir_variable_mode mode = nir_intrinsic_memory_modes(intrin);
            if (!kk_barrier_is_device_global(scope, mode))
               continue;

            b.cursor = nir_after_instr(instr);

            /* The inserted load before the new barrier must not be constant */
            nir_def *root = nir_load_buffer_ptr_kk(&b, 1, 64, .binding = 0);
            nir_def *addr = nir_iadd_imm(
               &b, root, kk_root_descriptor_offset(dynamic_buffers[0].zero));
            nir_def *zero =
               nir_build_load_global(&b, 1, 32, addr, .align_mul = 4);

            nir_push_if(&b, nir_ine_imm(&b, zero, 0));
            nir_barrier(&b, .execution_scope = SCOPE_NONE,
                        .memory_scope = scope, .memory_modes = mode,
                        .memory_semantics = NIR_MEMORY_ACQ_REL);
            nir_pop_if(&b, NULL);

            impl_progress = true;
         }
      }

      progress |= impl_progress;
      nir_progress(impl_progress, impl, nir_metadata_none);
   }

   return progress;
}
