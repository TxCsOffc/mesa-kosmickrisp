#include "kk_cmd_buffer.h"
#include "kk_device.h"
#include "kk_entrypoints.h"
#include "kk_physical_device.h"
#include "kk_shader.h"

#include "nir.h"
#include "nir_builder.h"
#include "nir_deref.h"

#include "util/hash_table.h"
#include "util/u_debug.h"
#include "util/ralloc.h"
#include "util/u_atomic.h"

#include "vk_alloc.h"
#include "vk_pipeline.h"
#include "vk_pipeline_layout.h"
#include "vk_shader.h"

#define KK_RT_HANDLE_SIZE  32u
#define KK_RT_ATTRIB_SIZE  32u
#define KK_RT_PAYLOAD_BASE KK_RT_ATTRIB_SIZE

struct kk_rt_dispatch {
   uint64_t raygen_addr;
   uint64_t miss_addr;
   uint64_t hit_addr;
   uint64_t callable_addr;
   uint32_t miss_stride;
   uint32_t hit_stride;
   uint32_t callable_stride;
   uint32_t pad;
   uint32_t launch_size[3];
   uint32_t pad2;
};

struct kk_rt_stage {
   nir_shader *nir;
   uint32_t payload_size;
};

struct kk_rt_group {
   VkRayTracingShaderGroupTypeKHR type;
   uint32_t general;
   uint32_t closest_hit;
   uint32_t any_hit;
   uint32_t intersection;
   uint32_t handle;
};

struct kk_rt_pipeline {
   struct vk_pipeline base;
   struct vk_pipeline_layout *layout;
   void *mem_ctx;
   uint32_t stage_count;
   struct kk_rt_stage *stages;
   uint32_t group_count;
   struct kk_rt_group *groups;
   struct vk_shader *cs;
};

enum kk_rt_role {
   KK_RT_ROLE_RAYGEN,
   KK_RT_ROLE_ANY_HIT,
   KK_RT_ROLE_CLOSEST_HIT,
   KK_RT_ROLE_MISS,
};

struct kk_rt_vars {
   nir_variable *rq;
   nir_variable *flags;
   nir_variable *cull_mask;
   nir_variable *sbt_offset;
   nir_variable *sbt_stride;
   nir_variable *miss_index;
   nir_variable *origin;
   nir_variable *dir;
   nir_variable *tmin;
   nir_variable *tmax;
   nir_variable *record;
   nir_variable *payload_off;
   nir_variable *accept;
   nir_variable *terminate;
};

static uint32_t kk_rt_next_handle = 0;

static void
kk_rt_create_vars(nir_shader *s, struct kk_rt_vars *v)
{
   v->rq = nir_variable_create(s, nir_var_shader_temp, glsl_uint64_t_type(),
                               "kk_rt_rq");
   v->rq->data.ray_query = true;
   v->flags = nir_variable_create(s, nir_var_shader_temp, glsl_uint_type(),
                                  "kk_rt_flags");
   v->cull_mask = nir_variable_create(s, nir_var_shader_temp, glsl_uint_type(),
                                      "kk_rt_cull_mask");
   v->sbt_offset = nir_variable_create(s, nir_var_shader_temp,
                                       glsl_uint_type(), "kk_rt_sbt_offset");
   v->sbt_stride = nir_variable_create(s, nir_var_shader_temp,
                                       glsl_uint_type(), "kk_rt_sbt_stride");
   v->miss_index = nir_variable_create(s, nir_var_shader_temp,
                                       glsl_uint_type(), "kk_rt_miss_index");
   v->origin = nir_variable_create(s, nir_var_shader_temp, glsl_vec_type(3),
                                   "kk_rt_origin");
   v->dir = nir_variable_create(s, nir_var_shader_temp, glsl_vec_type(3),
                                "kk_rt_dir");
   v->tmin = nir_variable_create(s, nir_var_shader_temp, glsl_float_type(),
                                 "kk_rt_tmin");
   v->tmax = nir_variable_create(s, nir_var_shader_temp, glsl_float_type(),
                                 "kk_rt_tmax");
   v->record = nir_variable_create(s, nir_var_shader_temp,
                                   glsl_uint64_t_type(), "kk_rt_record");
   v->payload_off = nir_variable_create(s, nir_var_shader_temp,
                                        glsl_uint_type(), "kk_rt_payload_off");
   v->accept = nir_variable_create(s, nir_var_shader_temp, glsl_bool_type(),
                                   "kk_rt_accept");
   v->terminate = nir_variable_create(s, nir_var_shader_temp,
                                      glsl_bool_type(), "kk_rt_terminate");
}

static void
kk_rt_remap_vars(struct hash_table *remap, const struct kk_rt_vars *from,
                 const struct kk_rt_vars *to)
{
   const nir_variable *const *f = (const nir_variable *const *)from;
   nir_variable *const *t = (nir_variable *const *)to;
   for (unsigned i = 0; i < sizeof(*from) / sizeof(nir_variable *); i++)
      _mesa_hash_table_insert(remap, f[i], t[i]);
}

static nir_def *
kk_rt_load_dispatch(nir_builder *b, unsigned offset, unsigned comps,
                    unsigned bit_size)
{
   nir_def *params = nir_load_sbt_base_amd(b);
   return nir_load_global_constant(b, comps, bit_size,
                                   nir_iadd_imm(b, params, offset),
                                   .align_mul = bit_size / 8);
}

static nir_def *
kk_rt_scratch_load(nir_builder *b, unsigned comps, unsigned bit_size,
                   nir_def *offset)
{
   if (bit_size == 1)
      return nir_ine_imm(b, nir_load_scratch(b, comps, 32, offset), 0);
   return nir_load_scratch(b, comps, bit_size, offset,
                           .align_mul = bit_size / 8);
}

static void
kk_rt_scratch_store(nir_builder *b, nir_def *value, nir_def *offset,
                    unsigned write_mask)
{
   if (value->bit_size == 1)
      value = nir_b2i32(b, value);
   nir_store_scratch(b, value, offset, .align_mul = value->bit_size / 8,
                     .write_mask = write_mask);
}

static nir_def *
kk_rt_payload_offset(nir_builder *b, nir_deref_instr *deref)
{
   nir_variable *var = nir_deref_instr_get_variable(deref);
   nir_def *rel =
      nir_u2u32(b, nir_build_deref_offset(b, deref,
                                          glsl_get_natural_size_align_bytes));
   if (var->data.mode == nir_var_ray_hit_attrib)
      return rel;
   if (var->data.explicit_location)
      return nir_iadd_imm(b, rel, KK_RT_PAYLOAD_BASE + var->data.driver_location);
   return nir_iadd(b, nir_load_shader_call_data_offset_lvp(b), rel);
}

static void
kk_rt_lower_payload(nir_shader *nir, uint32_t *payload_size)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_create(impl);

   nir_foreach_block_safe(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         b.cursor = nir_before_instr(instr);

         switch (intr->intrinsic) {
         case nir_intrinsic_load_deref:
         case nir_intrinsic_store_deref: {
            nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
            if (!nir_deref_mode_is_one_of(deref, nir_var_shader_call_data |
                                                    nir_var_ray_hit_attrib))
               break;
            nir_def *off = kk_rt_payload_offset(&b, deref);
            if (intr->intrinsic == nir_intrinsic_load_deref) {
               nir_def *v = kk_rt_scratch_load(&b, intr->def.num_components,
                                               intr->def.bit_size, off);
               nir_def_replace(&intr->def, v);
            } else {
               kk_rt_scratch_store(&b, intr->src[1].ssa, off,
                                   nir_intrinsic_write_mask(intr));
               nir_instr_remove(instr);
            }
            break;
         }
         case nir_intrinsic_trace_ray: {
            nir_deref_instr *deref = nir_src_as_deref(intr->src[10]);
            if (deref)
               nir_src_rewrite(&intr->src[10], kk_rt_payload_offset(&b, deref));
            break;
         }
         case nir_intrinsic_execute_callable: {
            nir_deref_instr *deref = nir_src_as_deref(intr->src[1]);
            if (deref)
               nir_src_rewrite(&intr->src[1], nir_imm_int(&b, 0));
            break;
         }
         default:
            break;
         }
      }
   }
   nir_progress(true, impl, nir_metadata_none);

   uint32_t size = 0;
   nir_foreach_variable_with_modes(var, nir, nir_var_shader_call_data) {
      if (!var->data.explicit_location)
         continue;
      unsigned vsize = glsl_get_explicit_size(var->type, false);
      size = MAX2(size, var->data.driver_location + vsize);
   }
   *payload_size = size;
}

static bool
kk_rt_halt_to_return(nir_shader *nir)
{
   bool progress = false;
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_jump)
            continue;
         nir_jump_instr *jump = nir_instr_as_jump(instr);
         if (jump->type == nir_jump_halt) {
            jump->type = nir_jump_return;
            progress = true;
         }
      }
   }
   return nir_progress(progress, impl, nir_metadata_none);
}

static VkResult
kk_rt_process_stage(struct kk_device *dev, VkPipelineCreateFlags2KHR flags,
                    const void *pipeline_pnext,
                    const VkPipelineShaderStageCreateInfo *info, void *mem_ctx,
                    struct kk_rt_stage *out)
{
   const struct vk_device_shader_ops *ops = dev->vk.shader_ops;
   struct vk_pipeline_robustness_state rs;
   vk_pipeline_robustness_state_fill(&dev->vk.robustness_state, &rs,
                                     pipeline_pnext, info->pNext);

   mesa_shader_stage stage = vk_to_mesa_shader_stage(info->stage);
   const struct nir_shader_compiler_options *nir_options =
      ops->get_nir_options(dev->vk.physical, stage, &rs);
   struct spirv_to_nir_options spirv_options =
      ops->get_spirv_options(dev->vk.physical, stage, &rs);
   spirv_options.constant_addr_format = nir_address_format_64bit_global;

   nir_shader *nir;
   VkResult result = vk_pipeline_shader_stage_to_nir(
      &dev->vk, flags, info, &spirv_options, nir_options, mem_ctx, &nir);
   if (result != VK_SUCCESS)
      return result;

   NIR_PASS(_, nir, nir_lower_variable_initializers, nir_var_function_temp);
   NIR_PASS(_, nir, nir_lower_returns);
   NIR_PASS(_, nir, nir_inline_functions);
   nir_remove_non_entrypoints(nir);
   NIR_PASS(_, nir, kk_rt_halt_to_return);
   NIR_PASS(_, nir, nir_lower_returns);
   NIR_PASS(_, nir, nir_opt_deref);
   NIR_PASS(_, nir, nir_lower_system_values);
   NIR_PASS(_, nir, nir_split_var_copies);
   NIR_PASS(_, nir, nir_lower_var_copies);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_constant,
            nir_address_format_64bit_global);
   nir_foreach_variable_with_modes(var, nir, nir_var_shader_temp) {
      if (var->data.explicit_location)
         var->data.mode = nir_var_shader_call_data;
   }
   nir_fixup_deref_modes(nir);
   NIR_PASS(_, nir, nir_lower_vars_to_explicit_types,
            nir_var_shader_call_data | nir_var_ray_hit_attrib,
            glsl_get_natural_size_align_bytes);
   kk_rt_lower_payload(nir, &out->payload_size);
   NIR_PASS(_, nir, nir_opt_dce);
   NIR_PASS(_, nir, nir_remove_dead_variables,
            nir_var_shader_call_data | nir_var_ray_hit_attrib, NULL);

   out->nir = nir;
   return VK_SUCCESS;
}

struct kk_rt_role_ctx {
   const struct kk_rt_vars *v;
   enum kk_rt_role role;
};

static nir_def *
kk_rt_rq_load(nir_builder *b, const struct kk_rt_role_ctx *ctx,
              nir_ray_query_value value, unsigned comps, unsigned bit_size,
              unsigned column)
{
   nir_def *rq = &nir_build_deref_var(b, ctx->v->rq)->def;
   return nir_rq_load(b, comps, bit_size, rq, .ray_query_value = value,
                      .committed = ctx->role == KK_RT_ROLE_CLOSEST_HIT,
                      .column = column);
}

static bool
kk_rt_lower_role_instr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct kk_rt_role_ctx *ctx = data;
   const struct kk_rt_vars *v = ctx->v;
   bool hit = ctx->role == KK_RT_ROLE_ANY_HIT ||
              ctx->role == KK_RT_ROLE_CLOSEST_HIT;
   nir_def *def = NULL;

   b->cursor = nir_before_instr(&intr->instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_load_ray_launch_id:
      def = nir_load_global_invocation_id(b, 32);
      break;
   case nir_intrinsic_load_ray_launch_size:
      def = kk_rt_load_dispatch(b, offsetof(struct kk_rt_dispatch, launch_size),
                                3, 32);
      break;
   case nir_intrinsic_load_shader_record_ptr:
      def = nir_load_var(b, v->record);
      break;
   case nir_intrinsic_load_ray_flags:
      def = nir_load_var(b, v->flags);
      break;
   case nir_intrinsic_load_cull_mask:
      def = nir_iand_imm(b, nir_load_var(b, v->cull_mask), 0xff);
      break;
   case nir_intrinsic_load_ray_t_min:
      def = nir_load_var(b, v->tmin);
      break;
   case nir_intrinsic_load_ray_world_origin:
      def = nir_load_var(b, v->origin);
      break;
   case nir_intrinsic_load_ray_world_direction:
      def = nir_load_var(b, v->dir);
      break;
   case nir_intrinsic_load_shader_call_data_offset_lvp:
      def = nir_load_var(b, v->payload_off);
      break;
   case nir_intrinsic_load_ray_t_max:
      def = hit ? kk_rt_rq_load(b, ctx, nir_ray_query_value_intersection_t, 1,
                                32, 0)
                : nir_load_var(b, v->tmax);
      break;
   case nir_intrinsic_load_ray_instance_custom_index:
      def = hit ? kk_rt_rq_load(
                     b, ctx,
                     nir_ray_query_value_intersection_instance_custom_index, 1,
                     32, 0)
                : nir_imm_int(b, 0);
      break;
   case nir_intrinsic_load_instance_id:
      def = hit ? kk_rt_rq_load(b, ctx,
                                nir_ray_query_value_intersection_instance_id, 1,
                                32, 0)
                : nir_imm_int(b, 0);
      break;
   case nir_intrinsic_load_primitive_id:
      def = hit ? kk_rt_rq_load(
                     b, ctx, nir_ray_query_value_intersection_primitive_index,
                     1, 32, 0)
                : nir_imm_int(b, 0);
      break;
   case nir_intrinsic_load_ray_geometry_index:
      def = hit ? kk_rt_rq_load(
                     b, ctx, nir_ray_query_value_intersection_geometry_index, 1,
                     32, 0)
                : nir_imm_int(b, 0);
      break;
   case nir_intrinsic_load_ray_hit_kind:
      def = hit ? nir_bcsel(b,
                            kk_rt_rq_load(
                               b, ctx,
                               nir_ray_query_value_intersection_front_face, 1,
                               1, 0),
                            nir_imm_int(b, 0xfe), nir_imm_int(b, 0xff))
                : nir_imm_int(b, 0);
      break;
   case nir_intrinsic_load_ray_object_to_world:
      def = hit ? kk_rt_rq_load(
                     b, ctx, nir_ray_query_value_intersection_object_to_world,
                     3, 32, nir_intrinsic_column(intr))
                : nir_imm_zero(b, 3, 32);
      break;
   case nir_intrinsic_load_ray_world_to_object:
      def = hit ? kk_rt_rq_load(
                     b, ctx, nir_ray_query_value_intersection_world_to_object,
                     3, 32, nir_intrinsic_column(intr))
                : nir_imm_zero(b, 3, 32);
      break;
   case nir_intrinsic_load_ray_object_origin:
      def = hit ? kk_rt_rq_load(
                     b, ctx,
                     nir_ray_query_value_intersection_object_ray_origin, 3, 32,
                     0)
                : nir_load_var(b, v->origin);
      break;
   case nir_intrinsic_load_ray_object_direction:
      def = hit ? kk_rt_rq_load(
                     b, ctx,
                     nir_ray_query_value_intersection_object_ray_direction, 3,
                     32, 0)
                : nir_load_var(b, v->dir);
      break;
   case nir_intrinsic_load_ray_triangle_vertex_positions:
      def = nir_imm_zero(b, 3, 32);
      break;
   case nir_intrinsic_ignore_ray_intersection:
      nir_store_var(b, v->accept, nir_imm_false(b), 1);
      break;
   case nir_intrinsic_terminate_ray:
      nir_store_var(b, v->accept, nir_imm_true(b), 1);
      nir_store_var(b, v->terminate, nir_imm_true(b), 1);
      break;
   case nir_intrinsic_report_ray_intersection:
      def = nir_imm_false(b);
      break;
   case nir_intrinsic_execute_callable:
      break;
   default:
      return false;
   }

   if (def)
      nir_def_replace(&intr->def, def);
   else
      nir_instr_remove(&intr->instr);
   return true;
}

struct kk_rt_builder {
   struct kk_rt_pipeline *pipeline;
   struct kk_rt_vars vars;
   void *tmp_ctx;
};

static void
kk_rt_inline_stage(nir_builder *b, struct kk_rt_builder *rb,
                   const struct kk_rt_stage *stage, enum kk_rt_role role)
{
   nir_shader *clone = nir_shader_clone(rb->tmp_ctx, stage->nir);
   struct kk_rt_vars standin;
   kk_rt_create_vars(clone, &standin);

   struct kk_rt_role_ctx ctx = {.v = &standin, .role = role};
   nir_shader_intrinsics_pass(clone, kk_rt_lower_role_instr, nir_metadata_none,
                              &ctx);

   struct hash_table *remap = _mesa_pointer_hash_table_create(NULL);
   kk_rt_remap_vars(remap, &standin, &rb->vars);
   nir_inline_function_impl(b, nir_shader_get_entrypoint(clone), NULL, remap);
   _mesa_hash_table_destroy(remap, NULL);
}

static void
kk_rt_dispatch_groups(nir_builder *b, struct kk_rt_builder *rb,
                      nir_def *handle, enum kk_rt_role role)
{
   struct kk_rt_pipeline *p = rb->pipeline;
   for (uint32_t i = 0; i < p->group_count; i++) {
      const struct kk_rt_group *g = &p->groups[i];
      uint32_t index = VK_SHADER_UNUSED_KHR;
      mesa_shader_stage want = MESA_SHADER_NONE;
      switch (role) {
      case KK_RT_ROLE_RAYGEN:
         index = g->general;
         want = MESA_SHADER_RAYGEN;
         break;
      case KK_RT_ROLE_MISS:
         index = g->general;
         want = MESA_SHADER_MISS;
         break;
      case KK_RT_ROLE_ANY_HIT:
         if (g->type == VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR)
            index = g->any_hit;
         want = MESA_SHADER_ANY_HIT;
         break;
      case KK_RT_ROLE_CLOSEST_HIT:
         if (g->type == VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR)
            index = g->closest_hit;
         want = MESA_SHADER_CLOSEST_HIT;
         break;
      }
      if (index == VK_SHADER_UNUSED_KHR || index >= p->stage_count)
         continue;
      const struct kk_rt_stage *stage = &p->stages[index];
      if (stage->nir->info.stage != want)
         continue;

      nir_push_if(b, nir_ieq_imm(b, handle, g->handle));
      kk_rt_inline_stage(b, rb, stage, role);
      nir_pop_if(b, NULL);
   }
}

static nir_def *
kk_rt_load_record(nir_builder *b, struct kk_rt_builder *rb, nir_def *addr)
{
   nir_store_var(b, rb->vars.record,
                 nir_iadd_imm(b, addr, KK_RT_HANDLE_SIZE), 1);
   return nir_load_global(b, 1, 32, addr, .align_mul = 4);
}

static void
kk_rt_lower_trace(nir_builder *b, struct kk_rt_builder *rb,
                  nir_intrinsic_instr *intr)
{
   const struct kk_rt_vars *v = &rb->vars;
   b->cursor = nir_before_instr(&intr->instr);

   nir_def *as = intr->src[0].ssa;
   nir_def *flags = intr->src[1].ssa;
   nir_def *cull = intr->src[2].ssa;
   nir_def *sbt_offset = nir_iand_imm(b, intr->src[3].ssa, 0xf);
   nir_def *sbt_stride = nir_iand_imm(b, intr->src[4].ssa, 0xf);
   nir_def *miss_index = nir_iand_imm(b, intr->src[5].ssa, 0xffff);
   nir_def *origin = intr->src[6].ssa;
   nir_def *tmin = intr->src[7].ssa;
   nir_def *dir = intr->src[8].ssa;
   nir_def *tmax = intr->src[9].ssa;
   nir_def *payload = intr->src[10].ssa;

   nir_def *saved_record = nir_load_var(b, v->record);
   nir_store_var(b, v->flags, flags, 1);
   nir_store_var(b, v->cull_mask, cull, 1);
   nir_store_var(b, v->sbt_offset, sbt_offset, 1);
   nir_store_var(b, v->sbt_stride, sbt_stride, 1);
   nir_store_var(b, v->miss_index, miss_index, 1);
   nir_store_var(b, v->origin, origin, 0x7);
   nir_store_var(b, v->tmin, tmin, 1);
   nir_store_var(b, v->dir, dir, 0x7);
   nir_store_var(b, v->tmax, tmax, 1);
   nir_store_var(b, v->payload_off, payload, 1);

   nir_def *hit_addr =
      kk_rt_load_dispatch(b, offsetof(struct kk_rt_dispatch, hit_addr), 1, 64);
   nir_def *hit_stride = kk_rt_load_dispatch(
      b, offsetof(struct kk_rt_dispatch, hit_stride), 1, 32);
   nir_def *miss_addr =
      kk_rt_load_dispatch(b, offsetof(struct kk_rt_dispatch, miss_addr), 1, 64);
   nir_def *miss_stride = kk_rt_load_dispatch(
      b, offsetof(struct kk_rt_dispatch, miss_stride), 1, 32);

   nir_def *rq = &nir_build_deref_var(b, v->rq)->def;
   nir_rq_initialize(b, rq, as, flags, cull, origin, tmin, dir, tmax);

   nir_push_loop(b);
   {
      rq = &nir_build_deref_var(b, v->rq)->def;
      nir_break_if(b, nir_inot(b, nir_rq_proceed(b, 1, rq)));

      nir_def *ctype = nir_rq_load(
         b, 1, 32, rq, .ray_query_value = nir_ray_query_value_intersection_type,
         .committed = false);
      nir_push_if(b, nir_ieq_imm(b, ctype, 0));
      {
         nir_def *bary = nir_rq_load(
            b, 2, 32, rq,
            .ray_query_value = nir_ray_query_value_intersection_barycentrics,
            .committed = false);
         kk_rt_scratch_store(b, bary, nir_imm_int(b, 0), 0x3);
         nir_store_var(b, v->accept, nir_imm_true(b), 1);
         nir_store_var(b, v->terminate,
                       nir_test_mask(b, flags, 0x4), 1);

         nir_def *inst_sbt = nir_rq_load(
            b, 1, 32, rq,
            .ray_query_value = nir_ray_query_value_intersection_instance_sbt_index,
            .committed = false);
         nir_def *geom = nir_rq_load(
            b, 1, 32, rq,
            .ray_query_value = nir_ray_query_value_intersection_geometry_index,
            .committed = false);
         nir_def *index = nir_iadd(b, nir_iadd(b, inst_sbt, sbt_offset),
                                   nir_imul(b, geom, sbt_stride));
         nir_def *rec = nir_iadd(
            b, hit_addr, nir_u2u64(b, nir_imul(b, index, hit_stride)));
         nir_def *handle = kk_rt_load_record(b, rb, rec);
         kk_rt_dispatch_groups(b, rb, handle, KK_RT_ROLE_ANY_HIT);

         rq = &nir_build_deref_var(b, v->rq)->def;
         nir_push_if(b, nir_load_var(b, v->accept));
         {
            nir_rq_confirm_intersection(b, rq);
            nir_push_if(b, nir_load_var(b, v->terminate));
            nir_rq_terminate(b, rq);
            nir_pop_if(b, NULL);
         }
         nir_pop_if(b, NULL);
      }
      nir_pop_if(b, NULL);
   }
   nir_pop_loop(b, NULL);

   rq = &nir_build_deref_var(b, v->rq)->def;
   nir_def *committed = nir_rq_load(
      b, 1, 32, rq, .ray_query_value = nir_ray_query_value_intersection_type,
      .committed = true);
   nir_push_if(b, nir_ine_imm(b, committed, 0));
   {
      nir_push_if(b, nir_inot(b, nir_test_mask(b, flags, 0x8)));
      {
         nir_def *bary = nir_rq_load(
            b, 2, 32, rq,
            .ray_query_value = nir_ray_query_value_intersection_barycentrics,
            .committed = true);
         kk_rt_scratch_store(b, bary, nir_imm_int(b, 0), 0x3);
         nir_def *inst_sbt = nir_rq_load(
            b, 1, 32, rq,
            .ray_query_value = nir_ray_query_value_intersection_instance_sbt_index,
            .committed = true);
         nir_def *geom = nir_rq_load(
            b, 1, 32, rq,
            .ray_query_value = nir_ray_query_value_intersection_geometry_index,
            .committed = true);
         nir_def *index = nir_iadd(b, nir_iadd(b, inst_sbt, sbt_offset),
                                   nir_imul(b, geom, sbt_stride));
         nir_def *rec = nir_iadd(
            b, hit_addr, nir_u2u64(b, nir_imul(b, index, hit_stride)));
         nir_def *handle = kk_rt_load_record(b, rb, rec);
         kk_rt_dispatch_groups(b, rb, handle, KK_RT_ROLE_CLOSEST_HIT);
      }
      nir_pop_if(b, NULL);
   }
   nir_push_else(b, NULL);
   {
      nir_def *rec = nir_iadd(
         b, miss_addr, nir_u2u64(b, nir_imul(b, miss_index, miss_stride)));
      nir_def *handle = kk_rt_load_record(b, rb, rec);
      kk_rt_dispatch_groups(b, rb, handle, KK_RT_ROLE_MISS);
   }
   nir_pop_if(b, NULL);

   nir_store_var(b, v->record, saved_record, 1);
   nir_instr_remove(&intr->instr);
}

static nir_shader *
kk_rt_build_shader(struct kk_device *dev, struct kk_rt_pipeline *p,
                   const struct vk_pipeline_robustness_state *rs)
{
   const struct nir_shader_compiler_options *nir_options =
      dev->vk.shader_ops->get_nir_options(dev->vk.physical,
                                          MESA_SHADER_COMPUTE, rs);
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  nir_options, "kk_rt");
   nir_shader *s = b.shader;
   s->info.workgroup_size[0] = 8;
   s->info.workgroup_size[1] = 8;
   s->info.workgroup_size[2] = 1;

   struct kk_rt_builder rb = {
      .pipeline = p,
      .tmp_ctx = ralloc_context(NULL),
   };
   kk_rt_create_vars(s, &rb.vars);

   uint32_t payload = 0;
   for (uint32_t i = 0; i < p->stage_count; i++)
      payload = MAX2(payload, p->stages[i].payload_size);
   s->scratch_size = KK_RT_PAYLOAD_BASE + align(payload, 16);
   if (debug_get_bool_option("HB_KK_RT_DUMP", false))
      fprintf(stderr, "kk rt: payload %u scratch %u\n", payload, s->scratch_size);

   nir_def *size = kk_rt_load_dispatch(
      &b, offsetof(struct kk_rt_dispatch, launch_size), 3, 32);
   nir_def *id = nir_load_global_invocation_id(&b, 32);
   nir_def *oob = nir_bany(&b, nir_uge(&b, id, size));
   nir_push_if(&b, nir_inot(&b, oob));
   {
      nir_store_var(&b, rb.vars.accept, nir_imm_true(&b), 1);
      nir_store_var(&b, rb.vars.terminate, nir_imm_false(&b), 1);
      nir_def *rg_addr = kk_rt_load_dispatch(
         &b, offsetof(struct kk_rt_dispatch, raygen_addr), 1, 64);
      nir_def *handle = kk_rt_load_record(&b, &rb, rg_addr);
      kk_rt_dispatch_groups(&b, &rb, handle, KK_RT_ROLE_RAYGEN);
   }
   nir_pop_if(&b, NULL);

   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   struct util_dynarray traces;
   util_dynarray_init(&traces, NULL);
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(instr)->intrinsic ==
                nir_intrinsic_trace_ray)
            util_dynarray_append(&traces, nir_instr_as_intrinsic(instr));
      }
   }
   util_dynarray_foreach(&traces, nir_intrinsic_instr *, it)
      kk_rt_lower_trace(&b, &rb, *it);
   util_dynarray_fini(&traces);

   nir_foreach_block_safe(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(instr)->intrinsic ==
                nir_intrinsic_trace_ray)
            nir_instr_remove(instr);
      }
   }

   nir_progress(true, impl, nir_metadata_none);
   nir_fixup_deref_modes(s);
   ralloc_free(rb.tmp_ctx);
   nir_validate_shader(s, "kk_rt_build_shader");
   return s;
}

static void
kk_rt_pipeline_destroy(struct vk_device *vk_dev, struct vk_pipeline *pipeline,
                       const VkAllocationCallbacks *alloc)
{
   struct kk_rt_pipeline *p = container_of(pipeline, struct kk_rt_pipeline, base);
   if (p->cs)
      vk_shader_destroy(vk_dev, p->cs, alloc);
   if (p->layout)
      vk_pipeline_layout_unref(vk_dev, p->layout);
   ralloc_free(p->mem_ctx);
   vk_pipeline_free(vk_dev, alloc, pipeline);
}

static void
kk_rt_pipeline_cmd_bind(struct vk_command_buffer *vk_cmd,
                        struct vk_pipeline *pipeline)
{
   struct kk_cmd_buffer *cmd = container_of(vk_cmd, struct kk_cmd_buffer, vk);
   cmd->state.rt.pipeline =
      pipeline ? container_of(pipeline, struct kk_rt_pipeline, base) : NULL;
}

static const struct vk_pipeline_ops kk_rt_pipeline_ops = {
   .destroy = kk_rt_pipeline_destroy,
   .cmd_bind = kk_rt_pipeline_cmd_bind,
};

static VkResult
kk_rt_compile(struct kk_device *dev, struct kk_rt_pipeline *p,
              const VkRayTracingPipelineCreateInfoKHR *info,
              const VkAllocationCallbacks *alloc)
{
   struct vk_pipeline_robustness_state rs;
   vk_pipeline_robustness_state_fill(&dev->vk.robustness_state, &rs,
                                     info->pNext, NULL);

   nir_shader *nir = kk_rt_build_shader(dev, p, &rs);
   if (dev->vk.shader_ops->preprocess_nir)
      dev->vk.shader_ops->preprocess_nir(dev->vk.physical, nir, &rs);
   if (debug_get_bool_option("HB_KK_RT_DUMP", false))
      nir_print_shader(nir, stderr);

   VkPushConstantRange range = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
   if (p->layout) {
      for (uint32_t i = 0; i < p->layout->push_range_count; i++) {
         const VkPushConstantRange *r = &p->layout->push_ranges[i];
         range.size = MAX2(range.size, r->offset + r->size);
      }
   }

   struct vk_shader_compile_info compile_info = {
      .stage = MESA_SHADER_COMPUTE,
      .nir = nir,
      .robustness = &rs,
      .set_layout_count = p->layout ? p->layout->set_count : 0,
      .set_layouts = p->layout ? p->layout->set_layouts : NULL,
      .push_constant_range_count = range.size ? 1 : 0,
      .push_constant_ranges = range.size ? &range : NULL,
   };
   return vk_compile_shaders(&dev->vk, 1, &compile_info, NULL,
                             &dev->vk.enabled_features, alloc, &p->cs);
}

static VkResult
kk_rt_pipeline_create(struct kk_device *dev,
                      const VkRayTracingPipelineCreateInfoKHR *info,
                      const VkAllocationCallbacks *alloc, VkPipeline *out)
{
   VkPipelineCreateFlags2KHR flags = vk_rt_pipeline_create_flags(info);
   VK_FROM_HANDLE(vk_pipeline_layout, layout, info->layout);

   uint32_t stage_count = info->stageCount;
   uint32_t group_count = info->groupCount;
   const VkPipelineLibraryCreateInfoKHR *libs = info->pLibraryInfo;
   for (uint32_t l = 0; libs && l < libs->libraryCount; l++) {
      VK_FROM_HANDLE(vk_pipeline, lib_base, libs->pLibraries[l]);
      struct kk_rt_pipeline *lib =
         container_of(lib_base, struct kk_rt_pipeline, base);
      stage_count += lib->stage_count;
      group_count += lib->group_count;
   }

   struct kk_rt_pipeline *p = vk_pipeline_zalloc(
      &dev->vk, &kk_rt_pipeline_ops, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
      flags, alloc, sizeof(*p));
   if (!p)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   p->mem_ctx = ralloc_context(NULL);
   p->layout = layout ? vk_pipeline_layout_ref(layout) : NULL;
   p->stage_count = stage_count;
   p->group_count = group_count;
   p->stages = rzalloc_array(p->mem_ctx, struct kk_rt_stage, MAX2(stage_count, 1));
   p->groups = rzalloc_array(p->mem_ctx, struct kk_rt_group, MAX2(group_count, 1));

   VkResult result = VK_SUCCESS;
   for (uint32_t i = 0; i < info->stageCount; i++) {
      result = kk_rt_process_stage(dev, flags, info->pNext, &info->pStages[i],
                                   p->mem_ctx, &p->stages[i]);
      if (result != VK_SUCCESS)
         goto fail;
      p->base.stages |= info->pStages[i].stage;
   }

   for (uint32_t i = 0; i < info->groupCount; i++) {
      const VkRayTracingShaderGroupCreateInfoKHR *g = &info->pGroups[i];
      p->groups[i] = (struct kk_rt_group){
         .type = g->type,
         .general = g->generalShader,
         .closest_hit = g->closestHitShader,
         .any_hit = g->anyHitShader,
         .intersection = g->intersectionShader,
         .handle = p_atomic_inc_return(&kk_rt_next_handle),
      };
   }

   uint32_t stage_base = info->stageCount;
   uint32_t group_base = info->groupCount;
   for (uint32_t l = 0; libs && l < libs->libraryCount; l++) {
      VK_FROM_HANDLE(vk_pipeline, lib_base, libs->pLibraries[l]);
      struct kk_rt_pipeline *lib =
         container_of(lib_base, struct kk_rt_pipeline, base);
      for (uint32_t i = 0; i < lib->stage_count; i++) {
         p->stages[stage_base + i].nir =
            nir_shader_clone(p->mem_ctx, lib->stages[i].nir);
         p->stages[stage_base + i].payload_size = lib->stages[i].payload_size;
      }
      for (uint32_t i = 0; i < lib->group_count; i++) {
         struct kk_rt_group g = lib->groups[i];
         if (g.general != VK_SHADER_UNUSED_KHR)
            g.general += stage_base;
         if (g.closest_hit != VK_SHADER_UNUSED_KHR)
            g.closest_hit += stage_base;
         if (g.any_hit != VK_SHADER_UNUSED_KHR)
            g.any_hit += stage_base;
         if (g.intersection != VK_SHADER_UNUSED_KHR)
            g.intersection += stage_base;
         p->groups[group_base + i] = g;
      }
      p->base.stages |= lib->base.stages;
      stage_base += lib->stage_count;
      group_base += lib->group_count;
   }

   if (!(flags & VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR)) {
      result = kk_rt_compile(dev, p, info, alloc);
      if (result != VK_SUCCESS)
         goto fail;
   }

   *out = vk_pipeline_to_handle(&p->base);
   return VK_SUCCESS;

fail:
   kk_rt_pipeline_destroy(&dev->vk, &p->base, alloc);
   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_CreateRayTracingPipelinesKHR(
   VkDevice device, VkDeferredOperationKHR deferredOperation,
   VkPipelineCache pipelineCache, uint32_t createInfoCount,
   const VkRayTracingPipelineCreateInfoKHR *pCreateInfos,
   const VkAllocationCallbacks *pAllocator, VkPipeline *pPipelines)
{
   VK_FROM_HANDLE(kk_device, dev, device);
   VkResult result = VK_SUCCESS;
   uint32_t i = 0;
   for (; i < createInfoCount; i++) {
      VkResult r =
         kk_rt_pipeline_create(dev, &pCreateInfos[i], pAllocator, &pPipelines[i]);
      if (r != VK_SUCCESS) {
         pPipelines[i] = VK_NULL_HANDLE;
         result = r;
         if (vk_rt_pipeline_create_flags(&pCreateInfos[i]) &
             VK_PIPELINE_CREATE_2_EARLY_RETURN_ON_FAILURE_BIT_KHR) {
            i++;
            break;
         }
      }
   }
   for (; i < createInfoCount; i++)
      pPipelines[i] = VK_NULL_HANDLE;
   if (result == VK_SUCCESS && deferredOperation != VK_NULL_HANDLE)
      return VK_OPERATION_NOT_DEFERRED_KHR;
   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_GetRayTracingShaderGroupHandlesKHR(VkDevice device, VkPipeline pipeline,
                                      uint32_t firstGroup, uint32_t groupCount,
                                      size_t dataSize, void *pData)
{
   VK_FROM_HANDLE(vk_pipeline, base, pipeline);
   struct kk_rt_pipeline *p = container_of(base, struct kk_rt_pipeline, base);
   uint8_t *out = pData;
   for (uint32_t i = 0; i < groupCount; i++) {
      if ((i + 1) * KK_RT_HANDLE_SIZE > dataSize)
         break;
      memset(out + i * KK_RT_HANDLE_SIZE, 0, KK_RT_HANDLE_SIZE);
      uint32_t handle = p->groups[firstGroup + i].handle;
      memcpy(out + i * KK_RT_HANDLE_SIZE, &handle, sizeof(handle));
   }
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_GetRayTracingCaptureReplayShaderGroupHandlesKHR(
   VkDevice device, VkPipeline pipeline, uint32_t firstGroup,
   uint32_t groupCount, size_t dataSize, void *pData)
{
   return kk_GetRayTracingShaderGroupHandlesKHR(device, pipeline, firstGroup,
                                                groupCount, dataSize, pData);
}

VKAPI_ATTR VkDeviceSize VKAPI_CALL
kk_GetRayTracingShaderGroupStackSizeKHR(VkDevice device, VkPipeline pipeline,
                                        uint32_t group,
                                        VkShaderGroupShaderKHR groupShader)
{
   return 0;
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdSetRayTracingPipelineStackSizeKHR(VkCommandBuffer commandBuffer,
                                        uint32_t pipelineStackSize)
{
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdTraceRaysKHR(VkCommandBuffer commandBuffer,
                   const VkStridedDeviceAddressRegionKHR *pRaygenShaderBindingTable,
                   const VkStridedDeviceAddressRegionKHR *pMissShaderBindingTable,
                   const VkStridedDeviceAddressRegionKHR *pHitShaderBindingTable,
                   const VkStridedDeviceAddressRegionKHR *pCallableShaderBindingTable,
                   uint32_t width, uint32_t height, uint32_t depth)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);
   struct kk_rt_pipeline *p = cmd->state.rt.pipeline;
   if (!p || !p->cs || !width || !height || !depth)
      return;

   struct kk_rt_dispatch d = {
      .raygen_addr = pRaygenShaderBindingTable->deviceAddress,
      .miss_addr = pMissShaderBindingTable->deviceAddress,
      .hit_addr = pHitShaderBindingTable->deviceAddress,
      .callable_addr = pCallableShaderBindingTable->deviceAddress,
      .miss_stride = (uint32_t)pMissShaderBindingTable->stride,
      .hit_stride = (uint32_t)pHitShaderBindingTable->stride,
      .callable_stride = (uint32_t)pCallableShaderBindingTable->stride,
      .launch_size = {width, height, depth},
   };
   struct kk_ptr params = kk_pool_upload(cmd, &d, sizeof(d), 8u);
   if (!params.gpu)
      return;

   struct kk_descriptor_state *saved = malloc(sizeof(*saved));
   if (!saved) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return;
   }
   memcpy(saved, &cmd->state.cs.descriptors, sizeof(*saved));
   memcpy(&cmd->state.cs.descriptors, &cmd->state.rt.descriptors,
          sizeof(*saved));
   struct kk_shader *saved_cs = cmd->state.shaders[MESA_SHADER_COMPUTE];
   cmd->state.shaders[MESA_SHADER_COMPUTE] =
      container_of(p->cs, struct kk_shader, vk);

   struct kk_descriptor_state *desc = &cmd->state.cs.descriptors;
   desc->root.cs.rt_params = params.gpu;
   desc->root_dirty = true;

   kk_CmdDispatchBase(commandBuffer, 0, 0, 0, DIV_ROUND_UP(width, 8),
                      DIV_ROUND_UP(height, 8), depth);

   memcpy(&cmd->state.rt.descriptors, &cmd->state.cs.descriptors,
          sizeof(*saved));
   memcpy(&cmd->state.cs.descriptors, saved, sizeof(*saved));
   cmd->state.cs.descriptors.root_dirty = true;
   cmd->state.shaders[MESA_SHADER_COMPUTE] = saved_cs;
   free(saved);
}
