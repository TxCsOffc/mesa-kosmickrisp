#include "kk_acceleration_structure.h"
#include "kk_cmd_buffer.h"
#include "kk_device.h"
#include "kk_entrypoints.h"
#include "kk_physical_device.h"
#include "kk_rt_link.h"
#include "kk_shader.h"

#include "kosmickrisp/bridge/mtl_bridge.h"

#include "nir.h"
#include "nir_builder.h"
#include "nir_deref.h"
#include "nir_serialize.h"
#include "util/blob.h"
#include "util/mesa-blake3.h"

#include "util/hash_table.h"
#include "util/os_time.h"
#include "util/ralloc.h"
#include "util/u_atomic.h"
#include "util/u_debug.h"
#include "util/u_dynarray.h"

#include "vk_alloc.h"
#include "vk_descriptor_set_layout.h"
#include "vk_pipeline.h"
#include "vk_pipeline_layout.h"
#include "vk_shader.h"
#include "vk_shader_module.h"

#define KK_RT_HANDLE_SIZE  32u
#define KK_RT_STATE_BASE   32u
#define KK_RT_STATE_SIZE   288u
#define KK_RT_PAYLOAD_BASE (KK_RT_STATE_BASE + KK_RT_STATE_SIZE)

enum kk_rt_state_offset {
   KK_RT_S_LAUNCH_ID = 0,
   KK_RT_S_LAUNCH_SIZE = 16,
   KK_RT_S_FLAGS = 28,
   KK_RT_S_CULL = 32,
   KK_RT_S_TMIN = 36,
   KK_RT_S_TMAX = 40,
   KK_RT_S_PAYLOAD = 44,
   KK_RT_S_RECORD = 48,
   KK_RT_S_CUSTOM = 56,
   KK_RT_S_INSTANCE = 60,
   KK_RT_S_PRIMITIVE = 64,
   KK_RT_S_GEOMETRY = 68,
   KK_RT_S_HIT_KIND = 72,
   KK_RT_S_ACCEPT = 76,
   KK_RT_S_TERMINATE = 80,
   KK_RT_S_WORLD_ORIGIN = 96,
   KK_RT_S_WORLD_DIR = 112,
   KK_RT_S_OBJECT_ORIGIN = 128,
   KK_RT_S_OBJECT_DIR = 144,
   KK_RT_S_O2W = 160,
   KK_RT_S_W2O = 224,
};

enum kk_rt_needs {
   KK_RT_NEED_CUSTOM = 1u << 0,
   KK_RT_NEED_INSTANCE = 1u << 1,
   KK_RT_NEED_PRIMITIVE = 1u << 2,
   KK_RT_NEED_GEOMETRY = 1u << 3,
   KK_RT_NEED_HIT_KIND = 1u << 4,
   KK_RT_NEED_OBJECT_ORIGIN = 1u << 5,
   KK_RT_NEED_OBJECT_DIR = 1u << 6,
   KK_RT_NEED_O2W = 1u << 7,
   KK_RT_NEED_W2O = 1u << 8,
};

struct kk_rt_dispatch {
   uint64_t raygen_addr;
   uint64_t miss_addr;
   uint64_t hit_addr;
   uint64_t callable_addr;
   uint32_t miss_stride;
   uint32_t hit_stride;
   uint32_t callable_stride;
   uint32_t debug_flags;
   uint32_t launch_size[3];
   uint32_t pad2;
};

struct kk_rt_stage {
   nir_shader *nir;
   uint32_t payload_size;
   unsigned char hash[32];
};

enum kk_rt_role {
   KK_RT_ROLE_RAYGEN,
   KK_RT_ROLE_ANY_HIT,
   KK_RT_ROLE_CLOSEST_HIT,
   KK_RT_ROLE_MISS,
};

struct kk_rt_group {
   VkRayTracingShaderGroupTypeKHR type;
   uint32_t general;
   uint32_t closest_hit;
   uint32_t any_hit;
   uint32_t intersection;
   uint32_t handle;
   uint32_t fn[3];
   uint32_t needs[3];
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

struct kk_rt_function {
   unsigned char key[32];
   uint32_t index;
   uint32_t needs;
   mtl_binary_function *fn;
};

static uint32_t kk_rt_next_handle = 0;
static __thread struct kk_rt_link *kk_rt_pending_link;

void
kk_rt_set_pending_link(struct kk_rt_link *link)
{
   kk_rt_pending_link = link;
}

struct kk_rt_link *
kk_rt_get_pending_link(void)
{
   return kk_rt_pending_link;
}

static uint32_t
kk_rt_key_hash(const void *key)
{
   uint32_t h;
   memcpy(&h, key, sizeof(h));
   return h;
}

static bool
kk_rt_key_equal(const void *a, const void *b)
{
   return !memcmp(a, b, 32);
}

void
kk_rt_functions_init(struct kk_device *dev)
{
   simple_mtx_init(&dev->rt.fn_lock, mtx_plain);
   dev->rt.functions =
      _mesa_hash_table_create(NULL, kk_rt_key_hash, kk_rt_key_equal);
   util_dynarray_init(&dev->rt.function_list, NULL);
}

void
kk_rt_functions_finish(struct kk_device *dev)
{
   util_dynarray_foreach(&dev->rt.function_list, struct kk_rt_function *, f) {
      mtl_release((*f)->fn);
      free(*f);
   }
   util_dynarray_fini(&dev->rt.function_list);
   if (dev->rt.functions)
      _mesa_hash_table_destroy(dev->rt.functions, NULL);
   dev->rt.functions = NULL;
   simple_mtx_destroy(&dev->rt.fn_lock);
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
kk_rt_block_load(nir_builder *b, unsigned comps, unsigned bit_size,
                 nir_def *offset)
{
   if (bit_size == 1)
      return nir_ine_imm(b, nir_load_rt_kk(b, comps, 32, offset, .align_mul = 4), 0);
   return nir_load_rt_kk(b, comps, bit_size, offset, .align_mul = bit_size / 8);
}

static void
kk_rt_block_store(nir_builder *b, nir_def *value, nir_def *offset,
                  unsigned write_mask)
{
   if (value->bit_size == 1)
      value = nir_b2i32(b, value);
   nir_store_rt_kk(b, value, offset, .align_mul = value->bit_size / 8,
                   .write_mask = write_mask);
}

static nir_def *
kk_rt_state_load(nir_builder *b, unsigned field, unsigned comps,
                 unsigned bit_size)
{
   return kk_rt_block_load(b, comps, bit_size,
                           nir_imm_int(b, KK_RT_STATE_BASE + field));
}

static void
kk_rt_state_store(nir_builder *b, unsigned field, nir_def *value)
{
   kk_rt_block_store(b, value, nir_imm_int(b, KK_RT_STATE_BASE + field),
                     nir_component_mask(value->num_components));
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
               nir_def *v = kk_rt_block_load(&b, intr->def.num_components,
                                             intr->def.bit_size, off);
               nir_def_replace(&intr->def, v);
            } else {
               kk_rt_block_store(&b, intr->src[1].ssa, off,
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

   NIR_PASS(_, nir, nir_opt_dce);
   memset(nir->info.source_blake3, 0, sizeof(nir->info.source_blake3));
   struct blob blob;
   blob_init(&blob);
   nir_serialize(&blob, nir, true);
   _mesa_blake3_compute(blob.data, blob.size, out->hash);
   blob_finish(&blob);

   const char *dump_dir = getenv("HB_KK_RT_DUMP_DIR");
   if (dump_dir) {
      const uint32_t *code = NULL;
      size_t code_size = 0;
      VK_FROM_HANDLE(vk_shader_module, module, info->module);
      if (module) {
         code = (const uint32_t *)module->data;
         code_size = module->size;
      } else {
         const VkShaderModuleCreateInfo *mci =
            vk_find_struct_const(info->pNext, SHADER_MODULE_CREATE_INFO);
         if (mci) {
            code = mci->pCode;
            code_size = mci->codeSize;
         }
      }
      if (code) {
         char path[1024];
         snprintf(path, sizeof(path), "%s/%s-%02x%02x%02x%02x%02x%02x%02x%02x-%s.spv",
                  dump_dir, _mesa_shader_stage_to_abbrev(stage), out->hash[0],
                  out->hash[1], out->hash[2], out->hash[3], out->hash[4],
                  out->hash[5], out->hash[6], out->hash[7], info->pName);
         FILE *f = fopen(path, "wb");
         if (f) {
            fwrite(code, 1, code_size, f);
            fclose(f);
         }
      }
   }

   out->nir = nir;
   return VK_SUCCESS;
}

struct kk_rt_kernel_vars {
   nir_variable *rq;
   nir_variable *record;
};

struct kk_rt_role_ctx {
   enum kk_rt_role role;
   const struct kk_rt_kernel_vars *v;
   uint32_t needs;
   bool dropped_trace;
};

static nir_def *
kk_rt_role_state(nir_builder *b, struct kk_rt_role_ctx *ctx, unsigned field,
                 unsigned comps, unsigned bit_size, uint32_t need)
{
   ctx->needs |= need;
   return kk_rt_state_load(b, field, comps, bit_size);
}

static bool
kk_rt_lower_role_instr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct kk_rt_role_ctx *ctx = data;
   bool raygen = ctx->role == KK_RT_ROLE_RAYGEN;
   nir_def *def = NULL;

   b->cursor = nir_before_instr(&intr->instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_load_ray_launch_id:
      def = raygen ? nir_load_global_invocation_id(b, 32)
                   : kk_rt_state_load(b, KK_RT_S_LAUNCH_ID, 3, 32);
      break;
   case nir_intrinsic_load_ray_launch_size:
      def = raygen ? kk_rt_load_dispatch(
                        b, offsetof(struct kk_rt_dispatch, launch_size), 3, 32)
                   : kk_rt_state_load(b, KK_RT_S_LAUNCH_SIZE, 3, 32);
      break;
   case nir_intrinsic_load_shader_record_ptr:
      def = raygen ? nir_load_var(b, ctx->v->record)
                   : kk_rt_state_load(b, KK_RT_S_RECORD, 1, 64);
      break;
   case nir_intrinsic_load_ray_flags:
      def = kk_rt_state_load(b, KK_RT_S_FLAGS, 1, 32);
      break;
   case nir_intrinsic_load_cull_mask:
      def = nir_iand_imm(b, kk_rt_state_load(b, KK_RT_S_CULL, 1, 32), 0xff);
      break;
   case nir_intrinsic_load_ray_t_min:
      def = kk_rt_state_load(b, KK_RT_S_TMIN, 1, 32);
      break;
   case nir_intrinsic_load_ray_t_max:
      def = kk_rt_state_load(b, KK_RT_S_TMAX, 1, 32);
      break;
   case nir_intrinsic_load_ray_world_origin:
      def = kk_rt_state_load(b, KK_RT_S_WORLD_ORIGIN, 3, 32);
      break;
   case nir_intrinsic_load_ray_world_direction:
      def = kk_rt_state_load(b, KK_RT_S_WORLD_DIR, 3, 32);
      break;
   case nir_intrinsic_load_shader_call_data_offset_lvp:
      def = kk_rt_state_load(b, KK_RT_S_PAYLOAD, 1, 32);
      break;
   case nir_intrinsic_load_ray_instance_custom_index:
      def = kk_rt_role_state(b, ctx, KK_RT_S_CUSTOM, 1, 32, KK_RT_NEED_CUSTOM);
      break;
   case nir_intrinsic_load_instance_id:
      def = kk_rt_role_state(b, ctx, KK_RT_S_INSTANCE, 1, 32,
                             KK_RT_NEED_INSTANCE);
      break;
   case nir_intrinsic_load_primitive_id:
      def = kk_rt_role_state(b, ctx, KK_RT_S_PRIMITIVE, 1, 32,
                             KK_RT_NEED_PRIMITIVE);
      break;
   case nir_intrinsic_load_ray_geometry_index:
      def = kk_rt_role_state(b, ctx, KK_RT_S_GEOMETRY, 1, 32,
                             KK_RT_NEED_GEOMETRY);
      break;
   case nir_intrinsic_load_ray_hit_kind:
      def = kk_rt_role_state(b, ctx, KK_RT_S_HIT_KIND, 1, 32,
                             KK_RT_NEED_HIT_KIND);
      break;
   case nir_intrinsic_load_ray_object_origin:
      def = kk_rt_role_state(b, ctx, KK_RT_S_OBJECT_ORIGIN, 3, 32,
                             KK_RT_NEED_OBJECT_ORIGIN);
      break;
   case nir_intrinsic_load_ray_object_direction:
      def = kk_rt_role_state(b, ctx, KK_RT_S_OBJECT_DIR, 3, 32,
                             KK_RT_NEED_OBJECT_DIR);
      break;
   case nir_intrinsic_load_ray_object_to_world:
      def = kk_rt_role_state(b, ctx,
                             KK_RT_S_O2W + 16 * nir_intrinsic_column(intr), 3,
                             32, KK_RT_NEED_O2W);
      break;
   case nir_intrinsic_load_ray_world_to_object:
      def = kk_rt_role_state(b, ctx,
                             KK_RT_S_W2O + 16 * nir_intrinsic_column(intr), 3,
                             32, KK_RT_NEED_W2O);
      break;
   case nir_intrinsic_load_ray_triangle_vertex_positions:
      def = nir_imm_zero(b, 3, 32);
      break;
   case nir_intrinsic_ignore_ray_intersection:
      kk_rt_state_store(b, KK_RT_S_ACCEPT, nir_imm_int(b, 0));
      break;
   case nir_intrinsic_terminate_ray:
      kk_rt_state_store(b, KK_RT_S_ACCEPT, nir_imm_int(b, 1));
      kk_rt_state_store(b, KK_RT_S_TERMINATE, nir_imm_int(b, 1));
      break;
   case nir_intrinsic_report_ray_intersection:
      def = nir_imm_false(b);
      break;
   case nir_intrinsic_execute_callable:
      break;
   case nir_intrinsic_trace_ray:
      if (raygen)
         return false;
      ctx->dropped_trace = true;
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

static void
kk_rt_function_key(const struct kk_rt_pipeline *p,
                   const struct kk_rt_stage *stage, enum kk_rt_role role,
                   const struct vk_pipeline_robustness_state *rs,
                   unsigned char *key)
{
   struct mesa_blake3 ctx;
   _mesa_blake3_init(&ctx);
   _mesa_blake3_update(&ctx, stage->hash, sizeof(stage->hash));
   _mesa_blake3_update(&ctx, &role, sizeof(role));
   _mesa_blake3_update(&ctx, rs, sizeof(*rs));
   if (p->layout) {
      for (uint32_t i = 0; i < p->layout->set_count; i++) {
         if (p->layout->set_layouts[i])
            _mesa_blake3_update(&ctx, p->layout->set_layouts[i]->blake3,
                                sizeof(p->layout->set_layouts[i]->blake3));
         else
            _mesa_blake3_update(&ctx, &i, sizeof(i));
      }
   }
   blake3_hash out;
   _mesa_blake3_final(&ctx, out);
   memcpy(key, out, 32);
}

static VkResult
kk_rt_get_function(struct kk_device *dev, const struct kk_rt_pipeline *p,
                   const struct kk_rt_stage *stage, enum kk_rt_role role,
                   const struct vk_pipeline_robustness_state *rs,
                   uint32_t *fn_out, uint32_t *needs_out)
{
   unsigned char key[32];
   kk_rt_function_key(p, stage, role, rs, key);

   simple_mtx_lock(&dev->rt.fn_lock);
   struct hash_entry *e = _mesa_hash_table_search(dev->rt.functions, key);
   if (e) {
      struct kk_rt_function *f = e->data;
      *fn_out = f->index + 1;
      *needs_out = f->needs;
      simple_mtx_unlock(&dev->rt.fn_lock);
      return VK_SUCCESS;
   }

   simple_mtx_unlock(&dev->rt.fn_lock);

   char name[48];
   snprintf(name, sizeof(name),
            "kk_rt_fn_%02x%02x%02x%02x%02x%02x%02x%02x", key[0], key[1],
            key[2], key[3], key[4], key[5], key[6], key[7]);

   nir_shader *nir = nir_shader_clone(NULL, stage->nir);
   struct kk_rt_role_ctx ctx = {.role = role};
   nir_shader_intrinsics_pass(nir, kk_rt_lower_role_instr, nir_metadata_none,
                              &ctx);
   if (ctx.dropped_trace)
      fprintf(stderr, "kk rt: %s traces rays; recursion is not supported\n",
              name);

   char *msl = NULL;
   VkResult result = kk_compile_rt_function_msl(
      dev, nir, rs, p->layout ? p->layout->set_count : 0,
      p->layout ? p->layout->set_layouts : NULL, name, &msl);
   mtl_binary_function *fn = NULL;
   if (result == VK_SUCCESS) {
      struct kk_physical_device *pdev = kk_device_physical(dev);
      mtl_library *lib =
         mtl_new_library(dev->mtl_compiler_handle, msl, pdev->info.msl_version,
                         MTL_MATH_MODE_FAST,
                         MTL_MATH_FLOATING_POINT_FUNCTIONS_FAST);
      if (lib) {
         fn = mtl_new_binary_function(dev->mtl_compiler_handle, lib, name);
         mtl_release(lib);
      }
      if (!fn) {
         if (debug_get_bool_option("HB_KK_RT_DUMP", false))
            fprintf(stderr, "kk rt: %s failed:\n%s\n", name, msl);
         result = VK_ERROR_INVALID_SHADER_NV;
      }
   }
   free(msl);

   if (result != VK_SUCCESS)
      return result;

   simple_mtx_lock(&dev->rt.fn_lock);
   e = _mesa_hash_table_search(dev->rt.functions, key);
   if (e) {
      struct kk_rt_function *f = e->data;
      *fn_out = f->index + 1;
      *needs_out = f->needs;
      simple_mtx_unlock(&dev->rt.fn_lock);
      mtl_release(fn);
      return VK_SUCCESS;
   }
   struct kk_rt_function *f = calloc(1, sizeof(*f));
   memcpy(f->key, key, sizeof(key));
   f->index = util_dynarray_num_elements(&dev->rt.function_list,
                                         struct kk_rt_function *);
   f->needs = ctx.needs;
   f->fn = fn;
   util_dynarray_append(&dev->rt.function_list, f);
   _mesa_hash_table_insert(dev->rt.functions, f->key, f);
   simple_mtx_unlock(&dev->rt.fn_lock);

   *fn_out = f->index + 1;
   *needs_out = f->needs;
   return VK_SUCCESS;
}

static void
kk_rt_inline_raygen(nir_builder *b, const struct kk_rt_kernel_vars *v,
                    const struct kk_rt_stage *stage, void *tmp_ctx)
{
   nir_shader *clone = nir_shader_clone(tmp_ctx, stage->nir);
   nir_variable *record = nir_variable_create(clone, nir_var_shader_temp,
                                              glsl_uint64_t_type(), "record");
   struct kk_rt_kernel_vars standin = {.record = record};
   struct kk_rt_role_ctx ctx = {.role = KK_RT_ROLE_RAYGEN, .v = &standin};
   nir_shader_intrinsics_pass(clone, kk_rt_lower_role_instr, nir_metadata_none,
                              &ctx);

   struct hash_table *remap = _mesa_pointer_hash_table_create(NULL);
   _mesa_hash_table_insert(remap, record, v->record);
   nir_inline_function_impl(b, nir_shader_get_entrypoint(clone), NULL, remap);
   _mesa_hash_table_destroy(remap, NULL);
   nir_index_ssa_defs(b->impl);
}

static void
kk_rt_inline_callee(nir_builder *b, const struct kk_rt_stage *stage,
                    enum kk_rt_role role, void *tmp_ctx)
{
   nir_shader *clone = nir_shader_clone(tmp_ctx, stage->nir);
   struct kk_rt_role_ctx ctx = {.role = role};
   nir_shader_intrinsics_pass(clone, kk_rt_lower_role_instr, nir_metadata_none,
                              &ctx);
   struct hash_table *remap = _mesa_pointer_hash_table_create(NULL);
   nir_inline_function_impl(b, nir_shader_get_entrypoint(clone), NULL, remap);
   _mesa_hash_table_destroy(remap, NULL);
   nir_index_ssa_defs(b->impl);
}

static void
kk_rt_inline_misses(nir_builder *b, const struct kk_rt_pipeline *p,
                    nir_def *handle, void *tmp_ctx)
{
   bool *done = calloc(MAX2(p->group_count, 1), sizeof(bool));
   if (!done)
      return;
   for (uint32_t i = 0; i < p->group_count; i++) {
      const struct kk_rt_group *g = &p->groups[i];
      if (done[i] || g->type != VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR ||
          g->general >= p->stage_count ||
          p->stages[g->general].nir->info.stage != MESA_SHADER_MISS)
         continue;
      const struct kk_rt_stage *stage = &p->stages[g->general];
      nir_def *cond = nir_ieq_imm(b, handle, g->handle);
      for (uint32_t j = i + 1; j < p->group_count; j++) {
         const struct kk_rt_group *h = &p->groups[j];
         if (done[j] || h->type != VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR ||
             h->general >= p->stage_count ||
             memcmp(p->stages[h->general].hash, stage->hash, 32))
            continue;
         done[j] = true;
         cond = nir_ior(b, cond, nir_ieq_imm(b, handle, h->handle));
      }
      nir_push_if(b, cond);
      kk_rt_inline_callee(b, stage, KK_RT_ROLE_MISS, tmp_ctx);
      nir_pop_if(b, NULL);
   }
   free(done);
}

static uint32_t
kk_rt_unique_misses(const struct kk_rt_pipeline *p)
{
   uint32_t n = 0;
   for (uint32_t i = 0; i < p->group_count; i++) {
      const struct kk_rt_group *g = &p->groups[i];
      if (g->type != VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR ||
          g->general >= p->stage_count ||
          p->stages[g->general].nir->info.stage != MESA_SHADER_MISS)
         continue;
      bool seen = false;
      for (uint32_t j = 0; j < i && !seen; j++) {
         const struct kk_rt_group *h = &p->groups[j];
         seen = h->type == VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR &&
                h->general < p->stage_count &&
                p->stages[h->general].nir->info.stage == MESA_SHADER_MISS &&
                !memcmp(p->stages[h->general].hash, p->stages[g->general].hash, 32);
      }
      n += !seen;
   }
   return n;
}

static nir_def *
kk_rt_rq_load(nir_builder *b, nir_variable *rq, nir_ray_query_value value,
              bool committed, unsigned comps, unsigned bit_size,
              unsigned column)
{
   return nir_rq_load(b, comps, bit_size, &nir_build_deref_var(b, rq)->def,
                      .ray_query_value = value, .committed = committed,
                      .column = column);
}

static void
kk_rt_write_hit(nir_builder *b, nir_variable *rq, bool committed,
                uint32_t needs)
{
   nir_def *bary = kk_rt_rq_load(
      b, rq, nir_ray_query_value_intersection_barycentrics, committed, 2, 32, 0);
   kk_rt_block_store(b, bary, nir_imm_int(b, 0), 0x3);
   kk_rt_state_store(
      b, KK_RT_S_TMAX,
      kk_rt_rq_load(b, rq, nir_ray_query_value_intersection_t, committed, 1,
                    32, 0));
   if (needs & KK_RT_NEED_CUSTOM)
      kk_rt_state_store(
         b, KK_RT_S_CUSTOM,
         kk_rt_rq_load(b, rq,
                       nir_ray_query_value_intersection_instance_custom_index,
                       committed, 1, 32, 0));
   if (needs & KK_RT_NEED_INSTANCE)
      kk_rt_state_store(
         b, KK_RT_S_INSTANCE,
         kk_rt_rq_load(b, rq, nir_ray_query_value_intersection_instance_id,
                       committed, 1, 32, 0));
   if (needs & KK_RT_NEED_PRIMITIVE)
      kk_rt_state_store(
         b, KK_RT_S_PRIMITIVE,
         kk_rt_rq_load(b, rq, nir_ray_query_value_intersection_primitive_index,
                       committed, 1, 32, 0));
   if (needs & KK_RT_NEED_GEOMETRY)
      kk_rt_state_store(
         b, KK_RT_S_GEOMETRY,
         kk_rt_rq_load(b, rq, nir_ray_query_value_intersection_geometry_index,
                       committed, 1, 32, 0));
   if (needs & KK_RT_NEED_HIT_KIND)
      kk_rt_state_store(
         b, KK_RT_S_HIT_KIND,
         nir_bcsel(b,
                   kk_rt_rq_load(b, rq,
                                 nir_ray_query_value_intersection_front_face,
                                 committed, 1, 1, 0),
                   nir_imm_int(b, 0xfe), nir_imm_int(b, 0xff)));
   if (needs & KK_RT_NEED_OBJECT_ORIGIN)
      kk_rt_state_store(
         b, KK_RT_S_OBJECT_ORIGIN,
         kk_rt_rq_load(b, rq,
                       nir_ray_query_value_intersection_object_ray_origin,
                       committed, 3, 32, 0));
   if (needs & KK_RT_NEED_OBJECT_DIR)
      kk_rt_state_store(
         b, KK_RT_S_OBJECT_DIR,
         kk_rt_rq_load(b, rq,
                       nir_ray_query_value_intersection_object_ray_direction,
                       committed, 3, 32, 0));
   for (unsigned c = 0; c < 4; c++) {
      if (needs & KK_RT_NEED_O2W)
         kk_rt_state_store(
            b, KK_RT_S_O2W + 16 * c,
            kk_rt_rq_load(b, rq,
                          nir_ray_query_value_intersection_object_to_world,
                          committed, 3, 32, c));
      if (needs & KK_RT_NEED_W2O)
         kk_rt_state_store(
            b, KK_RT_S_W2O + 16 * c,
            kk_rt_rq_load(b, rq,
                          nir_ray_query_value_intersection_world_to_object,
                          committed, 3, 32, c));
   }
}

static nir_def *
kk_rt_set_record(nir_builder *b, nir_def *rec, unsigned slot)
{
   kk_rt_state_store(b, KK_RT_S_RECORD, nir_iadd_imm(b, rec, KK_RT_HANDLE_SIZE));
   return nir_load_global(b, 1, 32, nir_iadd_imm(b, rec, 4 * slot),
                          .align_mul = 4);
}

static void
kk_rt_call(nir_builder *b, nir_def *fn)
{
   nir_push_if(b, nir_ine_imm(b, fn, 0));
   nir_rt_call_kk(b, nir_iadd_imm(b, fn, -1));
   nir_pop_if(b, NULL);
}

static bool kk_rt_build_nocalls;

static void
kk_rt_lower_trace(nir_builder *b, const struct kk_rt_kernel_vars *v,
                  const struct kk_rt_pipeline *p, void *tmp_ctx,
                  uint32_t ahit_needs, uint32_t chit_needs,
                  nir_intrinsic_instr *intr)
{
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

   kk_rt_state_store(b, KK_RT_S_FLAGS, flags);
   kk_rt_state_store(b, KK_RT_S_CULL, cull);
   kk_rt_state_store(b, KK_RT_S_TMIN, tmin);
   kk_rt_state_store(b, KK_RT_S_WORLD_ORIGIN, origin);
   kk_rt_state_store(b, KK_RT_S_WORLD_DIR, dir);
   kk_rt_state_store(b, KK_RT_S_PAYLOAD, payload);

   nir_def *hit_addr =
      kk_rt_load_dispatch(b, offsetof(struct kk_rt_dispatch, hit_addr), 1, 64);
   nir_def *hit_stride = kk_rt_load_dispatch(
      b, offsetof(struct kk_rt_dispatch, hit_stride), 1, 32);
   nir_def *miss_addr =
      kk_rt_load_dispatch(b, offsetof(struct kk_rt_dispatch, miss_addr), 1, 64);
   nir_def *miss_stride = kk_rt_load_dispatch(
      b, offsetof(struct kk_rt_dispatch, miss_stride), 1, 32);
   nir_def *debug_flags = kk_rt_load_dispatch(
      b, offsetof(struct kk_rt_dispatch, debug_flags), 1, 32);

   nir_def *rq_flags = nir_bcsel(b, nir_test_mask(b, debug_flags, 16),
                                 nir_ior_imm(b, flags, 0x1), flags);
   nir_def *rq_tmax = nir_bcsel(b, nir_test_mask(b, debug_flags, 8),
                                nir_fneg(b, nir_imm_float(b, 1.0f)), tmax);
   nir_rq_initialize(b, &nir_build_deref_var(b, v->rq)->def, as, rq_flags,
                     cull, origin, tmin, dir, rq_tmax);

   nir_push_loop(b);
   {
      nir_def *rq = &nir_build_deref_var(b, v->rq)->def;
      nir_break_if(b, nir_inot(b, nir_rq_proceed(b, 1, rq)));

      nir_def *ctype = kk_rt_rq_load(
         b, v->rq, nir_ray_query_value_intersection_type, false, 1, 32, 0);
      nir_push_if(b, nir_ieq_imm(b, ctype, 0));
      {
         nir_def *inst_sbt = kk_rt_rq_load(
            b, v->rq, nir_ray_query_value_intersection_instance_sbt_index,
            false, 1, 32, 0);
         nir_def *geom = kk_rt_rq_load(
            b, v->rq, nir_ray_query_value_intersection_geometry_index, false,
            1, 32, 0);
         nir_def *index = nir_iadd(b, nir_iadd(b, inst_sbt, sbt_offset),
                                   nir_imul(b, geom, sbt_stride));
         nir_def *rec = nir_iadd(
            b, hit_addr, nir_u2u64(b, nir_imul(b, index, hit_stride)));
         nir_def *fn = nir_load_global(b, 1, 32, nir_iadd_imm(b, rec, 8),
                                       .align_mul = 4);
         fn = nir_bcsel(b, nir_test_mask(b, debug_flags, 2), nir_imm_int(b, 0), fn);
         if (kk_rt_build_nocalls)
            fn = nir_imm_int(b, 0);

         nir_variable *accept =
            nir_local_variable_create(b->impl, glsl_bool_type(), "accept");
         nir_variable *terminate =
            nir_local_variable_create(b->impl, glsl_bool_type(), "terminate");
         nir_store_var(b, accept, nir_imm_true(b), 1);
         nir_store_var(b, terminate, nir_test_mask(b, flags, 0x4), 1);

         nir_push_if(b, nir_ine_imm(b, fn, 0));
         {
            kk_rt_write_hit(b, v->rq, false, ahit_needs);
            kk_rt_state_store(b, KK_RT_S_ACCEPT, nir_imm_int(b, 1));
            kk_rt_state_store(b, KK_RT_S_TERMINATE,
                              nir_b2i32(b, nir_test_mask(b, flags, 0x4)));
            kk_rt_set_record(b, rec, 2);
            nir_rt_call_kk(b, nir_iadd_imm(b, fn, -1));
            nir_store_var(b, accept,
                          nir_ine_imm(b, kk_rt_state_load(b, KK_RT_S_ACCEPT, 1, 32), 0),
                          1);
            nir_store_var(
               b, terminate,
               nir_ine_imm(b, kk_rt_state_load(b, KK_RT_S_TERMINATE, 1, 32), 0),
               1);
         }
         nir_pop_if(b, NULL);

         rq = &nir_build_deref_var(b, v->rq)->def;
         nir_push_if(b, nir_load_var(b, accept));
         {
            nir_rq_confirm_intersection(b, rq);
            nir_push_if(b, nir_load_var(b, terminate));
            nir_rq_terminate(b, rq);
            nir_pop_if(b, NULL);
         }
         nir_pop_if(b, NULL);
      }
      nir_pop_if(b, NULL);
   }
   nir_pop_loop(b, NULL);

   nir_def *committed = kk_rt_rq_load(
      b, v->rq, nir_ray_query_value_intersection_type, true, 1, 32, 0);
   nir_push_if(b, nir_ine_imm(b, committed, 0));
   {
      nir_push_if(b, nir_inot(b, nir_test_mask(b, flags, 0x8)));
      {
         nir_def *inst_sbt = kk_rt_rq_load(
            b, v->rq, nir_ray_query_value_intersection_instance_sbt_index,
            true, 1, 32, 0);
         nir_def *geom = kk_rt_rq_load(
            b, v->rq, nir_ray_query_value_intersection_geometry_index, true, 1,
            32, 0);
         nir_def *index = nir_iadd(b, nir_iadd(b, inst_sbt, sbt_offset),
                                   nir_imul(b, geom, sbt_stride));
         nir_def *rec = nir_iadd(
            b, hit_addr, nir_u2u64(b, nir_imul(b, index, hit_stride)));
         nir_def *fn = nir_load_global(b, 1, 32, nir_iadd_imm(b, rec, 4),
                                       .align_mul = 4);
         fn = nir_bcsel(b, nir_test_mask(b, debug_flags, 4), nir_imm_int(b, 0), fn);
         if (kk_rt_build_nocalls)
            fn = nir_imm_int(b, 0);
         nir_push_if(b, nir_ine_imm(b, fn, 0));
         {
            kk_rt_write_hit(b, v->rq, true, chit_needs);
            kk_rt_set_record(b, rec, 1);
            nir_rt_call_kk(b, nir_iadd_imm(b, fn, -1));
         }
         nir_pop_if(b, NULL);
      }
      nir_pop_if(b, NULL);
   }
   nir_push_else(b, NULL);
   {
      nir_def *rec = nir_iadd(
         b, miss_addr, nir_u2u64(b, nir_imul(b, miss_index, miss_stride)));
      kk_rt_state_store(b, KK_RT_S_TMAX, tmax);
      nir_def *fn = kk_rt_set_record(b, rec, 3);
      if (kk_rt_unique_misses(p) <= 4)
         kk_rt_inline_misses(
            b, p, nir_load_global(b, 1, 32, rec, .align_mul = 4), tmp_ctx);
      else
         kk_rt_call(b, fn);
   }
   nir_pop_if(b, NULL);

   nir_instr_remove(&intr->instr);
}

static bool
kk_rt_set_block_range(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_rt_kk &&
       intr->intrinsic != nir_intrinsic_store_rt_kk)
      return false;
   nir_intrinsic_set_range(intr, *(uint32_t *)data);
   return true;
}

static nir_shader *
kk_rt_build_shader(struct kk_device *dev, struct kk_rt_pipeline *p,
                   const struct vk_pipeline_robustness_state *rs,
                   uint32_t ahit_needs, uint32_t chit_needs)
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

   void *tmp_ctx = ralloc_context(NULL);
   struct kk_rt_kernel_vars v = {
      .rq = nir_variable_create(s, nir_var_shader_temp, glsl_uint64_t_type(),
                                "kk_rt_rq"),
      .record = nir_variable_create(s, nir_var_shader_temp,
                                    glsl_uint64_t_type(), "kk_rt_record"),
   };
   v.rq->data.ray_query = true;

   uint32_t payload = 0;
   for (uint32_t i = 0; i < p->stage_count; i++)
      payload = MAX2(payload, p->stages[i].payload_size);
   uint32_t block = KK_RT_PAYLOAD_BASE + align(MAX2(payload, 4), 16);

   nir_def *size = kk_rt_load_dispatch(
      &b, offsetof(struct kk_rt_dispatch, launch_size), 3, 32);
   nir_def *id = nir_load_global_invocation_id(&b, 32);
   nir_push_if(&b, nir_inot(&b, nir_bany(&b, nir_uge(&b, id, size))));
   {
      kk_rt_state_store(&b, KK_RT_S_LAUNCH_ID, id);
      kk_rt_state_store(&b, KK_RT_S_LAUNCH_SIZE, size);
      nir_def *rg_addr = kk_rt_load_dispatch(
         &b, offsetof(struct kk_rt_dispatch, raygen_addr), 1, 64);
      nir_store_var(&b, v.record, nir_iadd_imm(&b, rg_addr, KK_RT_HANDLE_SIZE),
                    1);
      nir_def *handle = nir_load_global(&b, 1, 32, rg_addr, .align_mul = 4);
      for (uint32_t i = 0; i < p->group_count; i++) {
         const struct kk_rt_group *g = &p->groups[i];
         if (g->type != VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR ||
             g->general >= p->stage_count ||
             p->stages[g->general].nir->info.stage != MESA_SHADER_RAYGEN)
            continue;
         nir_push_if(&b, nir_ieq_imm(&b, handle, g->handle));
         kk_rt_inline_raygen(&b, &v, &p->stages[g->general], tmp_ctx);
         nir_pop_if(&b, NULL);
      }
   }
   nir_pop_if(&b, NULL);

   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   struct util_dynarray traces;
   util_dynarray_init(&traces, NULL);
   nir_foreach_block(block_it, impl) {
      nir_foreach_instr(instr, block_it) {
         if (instr->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(instr)->intrinsic ==
                nir_intrinsic_trace_ray)
            util_dynarray_append(&traces, nir_instr_as_intrinsic(instr));
      }
   }
   util_dynarray_foreach(&traces, nir_intrinsic_instr *, it)
      kk_rt_lower_trace(&b, &v, p, tmp_ctx, ahit_needs, chit_needs, *it);
   util_dynarray_fini(&traces);

   nir_progress(true, impl, nir_metadata_none);
   nir_index_ssa_defs(impl);
   nir_shader_intrinsics_pass(s, kk_rt_set_block_range, nir_metadata_all,
                              &block);
   nir_fixup_deref_modes(s);
   ralloc_free(tmp_ctx);
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
kk_rt_compile_group_functions(struct kk_device *dev, struct kk_rt_pipeline *p,
                              uint32_t first, uint32_t count,
                              const struct vk_pipeline_robustness_state *rs)
{
   for (uint32_t i = first; i < first + count; i++) {
      struct kk_rt_group *g = &p->groups[i];
      VkResult result = VK_SUCCESS;
      if (g->type == VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR) {
         if (g->general < p->stage_count &&
             p->stages[g->general].nir->info.stage == MESA_SHADER_MISS)
            result = kk_rt_get_function(dev, p, &p->stages[g->general],
                                        KK_RT_ROLE_MISS, rs, &g->fn[2],
                                        &g->needs[2]);
      } else if (g->type ==
                 VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR) {
         if (g->closest_hit < p->stage_count)
            result = kk_rt_get_function(dev, p, &p->stages[g->closest_hit],
                                        KK_RT_ROLE_CLOSEST_HIT, rs, &g->fn[0],
                                        &g->needs[0]);
         if (result == VK_SUCCESS && g->any_hit < p->stage_count)
            result = kk_rt_get_function(dev, p, &p->stages[g->any_hit],
                                        KK_RT_ROLE_ANY_HIT, rs, &g->fn[1],
                                        &g->needs[1]);
      }
      if (result != VK_SUCCESS)
         return result;
   }
   return VK_SUCCESS;
}

static struct kk_rt_link *
kk_rt_make_link(struct kk_device *dev, struct kk_rt_pipeline *p)
{
   struct kk_rt_link *link = calloc(1, sizeof(*link));
   if (!link)
      return NULL;
   link->ref = 1;
   uint32_t max_fns = p->group_count * 3;
   link->functions = calloc(MAX2(max_fns, 1), sizeof(*link->functions));
   link->indices = calloc(MAX2(max_fns, 1), sizeof(*link->indices));
   if (!link->functions || !link->indices) {
      kk_rt_link_unref(link);
      return NULL;
   }

   simple_mtx_lock(&dev->rt.fn_lock);
   struct kk_rt_function **list = util_dynarray_begin(&dev->rt.function_list);
   for (uint32_t i = 0; i < p->group_count; i++) {
      for (unsigned s = 0; s < 3; s++) {
         uint32_t fn = p->groups[i].fn[s];
         if (!fn)
            continue;
         bool seen = false;
         for (uint32_t j = 0; j < link->count && !seen; j++)
            seen = link->indices[j] == fn - 1;
         if (seen)
            continue;
         link->indices[link->count] = fn - 1;
         link->functions[link->count] = list[fn - 1]->fn;
         link->count++;
         link->table_size = MAX2(link->table_size, fn);
      }
   }
   simple_mtx_unlock(&dev->rt.fn_lock);
   return link;
}

static VkResult
kk_rt_compile(struct kk_device *dev, struct kk_rt_pipeline *p,
              const VkRayTracingPipelineCreateInfoKHR *info,
              const struct vk_pipeline_robustness_state *rs,
              const VkAllocationCallbacks *alloc)
{
   uint32_t ahit_needs = 0, chit_needs = 0;
   for (uint32_t i = 0; i < p->group_count; i++) {
      chit_needs |= p->groups[i].needs[0];
      ahit_needs |= p->groups[i].needs[1];
   }

   kk_rt_build_nocalls = (kk_hb_rt_live_flags() & 32u) != 0;
   struct kk_rt_link *link = kk_rt_make_link(dev, p);
   if (!link)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   nir_shader *nir = kk_rt_build_shader(dev, p, rs, ahit_needs, chit_needs);
   if (dev->vk.shader_ops->preprocess_nir)
      dev->vk.shader_ops->preprocess_nir(dev->vk.physical, nir, rs);
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
      .robustness = rs,
      .set_layout_count = p->layout ? p->layout->set_count : 0,
      .set_layouts = p->layout ? p->layout->set_layouts : NULL,
      .push_constant_range_count = range.size ? 1 : 0,
      .push_constant_ranges = range.size ? &range : NULL,
   };
   kk_rt_set_pending_link(link);
   VkResult result = vk_compile_shaders(&dev->vk, 1, &compile_info, NULL,
                                        &dev->vk.enabled_features, alloc,
                                        &p->cs);
   kk_rt_set_pending_link(NULL);
   fprintf(stderr,
           "kk rt: pipeline stages=%u groups=%u functions=%u table=%u "
           "needs ahit=%x chit=%x%s result=%d\n",
           p->stage_count, p->group_count, link->count, link->table_size,
           ahit_needs, chit_needs, kk_rt_build_nocalls ? " nocalls" : "",
           result);
   kk_rt_link_unref(link);
   return result;
}

static VkResult
kk_rt_pipeline_create(struct kk_device *dev,
                      const VkRayTracingPipelineCreateInfoKHR *info,
                      const VkAllocationCallbacks *alloc, VkPipeline *out)
{
   VkPipelineCreateFlags2KHR flags = vk_rt_pipeline_create_flags(info);
   VK_FROM_HANDLE(vk_pipeline_layout, layout, info->layout);
   int64_t t0 = os_time_get_nano();

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

   struct vk_pipeline_robustness_state rs;
   vk_pipeline_robustness_state_fill(&dev->vk.robustness_state, &rs,
                                     info->pNext, NULL);

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
         memcpy(p->stages[stage_base + i].hash, lib->stages[i].hash,
                sizeof(lib->stages[i].hash));
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

   result = kk_rt_compile_group_functions(dev, p, 0, info->groupCount, &rs);
   if (result != VK_SUCCESS)
      goto fail;

   if (!(flags & VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR)) {
      result = kk_rt_compile(dev, p, info, &rs, alloc);
      if (result != VK_SUCCESS)
         goto fail;
   }

   if (debug_get_bool_option("HB_KK_RT_DUMP", false) ||
       !(flags & VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR))
      fprintf(stderr, "kk rt: %s stages=%u groups=%u created in %.1f ms\n",
              (flags & VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR) ? "library"
                                                             : "pipeline",
              p->stage_count, p->group_count,
              (os_time_get_nano() - t0) / 1e6);

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
      const struct kk_rt_group *g = &p->groups[firstGroup + i];
      uint32_t words[KK_RT_HANDLE_SIZE / 4] = {g->handle, g->fn[0], g->fn[1],
                                               g->fn[2]};
      memcpy(out + i * KK_RT_HANDLE_SIZE, words, KK_RT_HANDLE_SIZE);
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
   uint32_t live = kk_hb_rt_live_flags();
   if (live & 1u)
      return;

   struct kk_rt_dispatch d = {
      .raygen_addr = pRaygenShaderBindingTable->deviceAddress,
      .miss_addr = pMissShaderBindingTable->deviceAddress,
      .hit_addr = pHitShaderBindingTable->deviceAddress,
      .callable_addr = pCallableShaderBindingTable->deviceAddress,
      .miss_stride = (uint32_t)pMissShaderBindingTable->stride,
      .hit_stride = (uint32_t)pHitShaderBindingTable->stride,
      .callable_stride = (uint32_t)pCallableShaderBindingTable->stride,
      .debug_flags = live,
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

   if (live & 64u) {
      cs_end(cmd);
      mtl_encoder_set_label(cs_get_compute(cmd), "kk rt trace");
   }
   kk_CmdDispatchBase(commandBuffer, 0, 0, 0, DIV_ROUND_UP(width, 8),
                      DIV_ROUND_UP(height, 8), depth);
   if (live & 64u)
      cs_end(cmd);

   memcpy(&cmd->state.rt.descriptors, &cmd->state.cs.descriptors,
          sizeof(*saved));
   memcpy(&cmd->state.cs.descriptors, saved, sizeof(*saved));
   cmd->state.cs.descriptors.root_dirty = true;
   cmd->state.shaders[MESA_SHADER_COMPUTE] = saved_cs;
   free(saved);
}
