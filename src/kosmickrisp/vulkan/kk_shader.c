/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "kk_shader.h"

#include "util/u_debug.h"


#include "kk_cmd_buffer.h"
#include "kk_debug.h"
#include "kk_descriptor_set.h"
#include "kk_descriptor_set_layout.h"
#include "kk_device.h"
#include "kk_format.h"
#include "kk_nir_lower_vbo.h"
#include "kk_physical_device.h"
#include "kk_sampler.h"

#include "kosmickrisp/bridge/mtl_bridge.h"
#include "kosmickrisp/bridge/ns_process_info.h"
#include "kosmickrisp/bridge/vk_to_mtl_map.h"
#include "kosmickrisp/compiler/nir_to_msl.h"

#include "c11/threads.h"

#include "nir_builder.h"
#include "nir_lower_blend.h"

#include "poly/nir/poly_nir.h"

#include "vk_blend.h"
#include "vk_format.h"
#include "vk_graphics_state.h"
#include "vk_nir_convert_ycbcr.h"
#include "vk_pipeline.h"
#include "util/hash_table.h"
#include "util/simple_mtx.h"

#include <stdlib.h>
#include <string.h>

struct kk_msl_source {
   char *code;
   uint64_t references;
   uint32_t hash;
};

static simple_mtx_t kk_msl_sources_lock = SIMPLE_MTX_INITIALIZER;
static struct hash_table *kk_msl_sources;

static void
kk_msl_source_intern(struct msl_compile_data *data)
{
   data->source = NULL;
   if (data->code == NULL)
      return;

   const uint32_t hash = _mesa_hash_string(data->code);
   simple_mtx_lock(&kk_msl_sources_lock);
   if (kk_msl_sources == NULL)
      kk_msl_sources = _mesa_hash_table_create(
         NULL, _mesa_hash_string, _mesa_key_string_equal);
   if (kk_msl_sources == NULL) {
      simple_mtx_unlock(&kk_msl_sources_lock);
      return;
   }

   struct hash_entry *entry = _mesa_hash_table_search_pre_hashed(
      kk_msl_sources, hash, data->code);
   if (entry != NULL) {
      struct kk_msl_source *source = entry->data;
      source->references++;
      char *old = data->code;
      data->code = source->code;
      data->source = source;
      simple_mtx_unlock(&kk_msl_sources_lock);
      if (old != source->code)
         ralloc_free(old);
      return;
   }

   struct kk_msl_source *source = malloc(sizeof(*source));
   if (source == NULL) {
      simple_mtx_unlock(&kk_msl_sources_lock);
      return;
   }

   char *compact = reralloc_size(NULL, data->code, strlen(data->code) + 1u);
   if (compact != NULL)
      data->code = compact;
   *source = (struct kk_msl_source){
      .code = data->code,
      .references = 1u,
      .hash = hash,
   };
   entry = _mesa_hash_table_insert_pre_hashed(
      kk_msl_sources, hash, source->code, source);
   if (entry == NULL) {
      free(source);
      simple_mtx_unlock(&kk_msl_sources_lock);
      return;
   }
   data->source = source;
   simple_mtx_unlock(&kk_msl_sources_lock);
}

static void
kk_msl_source_copy(struct msl_compile_data *dst,
                    const struct msl_compile_data *src)
{
   if (src->source != NULL) {
      simple_mtx_lock(&kk_msl_sources_lock);
      src->source->references++;
      dst->source = src->source;
      dst->code = src->code;
      simple_mtx_unlock(&kk_msl_sources_lock);
   } else {
      dst->code = ralloc_strdup(NULL, src->code);
      kk_msl_source_intern(dst);
   }
}

static void
kk_msl_source_finish(struct msl_compile_data *data)
{
   struct kk_msl_source *source = data->source;
   if (source == NULL) {
      ralloc_free(data->code);
   } else {
      simple_mtx_lock(&kk_msl_sources_lock);
      bool last = --source->references == 0u;
      if (last) {
         struct hash_entry *entry = _mesa_hash_table_search_pre_hashed(
            kk_msl_sources, source->hash, source->code);
         assert(entry != NULL && entry->data == source);
         _mesa_hash_table_remove(kk_msl_sources, entry);
         if (kk_msl_sources->entries == 0u) {
            _mesa_hash_table_destroy(kk_msl_sources, NULL);
            kk_msl_sources = NULL;
         }
      }
      simple_mtx_unlock(&kk_msl_sources_lock);
      if (last) {
         ralloc_free(source->code);
         free(source);
      }
   }
   data->source = NULL;
   data->code = NULL;
}

static const nir_shader_compiler_options *
kk_get_nir_options(struct vk_physical_device *vk_pdev, mesa_shader_stage stage,
                   UNUSED const struct vk_pipeline_robustness_state *rs)
{
   return &kk_nir_options;
}

static const struct vk_pipeline_robustness_state rs_all_supported = {
   .uniform_buffers =
      VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2,
   .storage_buffers =
      VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2,
   .images = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_ROBUST_IMAGE_ACCESS_2,
};

DEBUG_GET_ONCE_BOOL_OPTION(hb_no_ubo_robust, "HB_KK_NO_UBO_ROBUST", false)
/* Highball experiment: HB_KK_NO_SSBO_ROBUST=1 drops bounds checks on storage buffer access too. An out of bounds
 * read returns whatever memory follows instead of zero, an out of bounds write lands there. */
DEBUG_GET_ONCE_BOOL_OPTION(hb_no_ssbo_robust, "HB_KK_NO_SSBO_ROBUST", false)
/* HB_KK_NO_VERTEX_ROBUST=1 drops the bounds check on vertex attribute fetch: an attribute past its binding's range
 * reads the memory after it instead of zero. */
DEBUG_GET_ONCE_BOOL_OPTION(hb_no_vertex_robust, "HB_KK_NO_VERTEX_ROBUST", false)

static struct vk_pipeline_robustness_state
kk_hb_robustness(const struct vk_pipeline_robustness_state *rs)
{
   struct vk_pipeline_robustness_state out = *rs;
   if (debug_get_option_hb_no_ubo_robust())
      out.uniform_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED;
   if (debug_get_option_hb_no_ssbo_robust())
      out.storage_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED;
   if (debug_get_option_hb_no_vertex_robust())
      out.vertex_inputs = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED;
   return out;
}

static struct spirv_to_nir_options
kk_get_spirv_options(struct vk_physical_device *vk_pdev,
                     UNUSED mesa_shader_stage stage,
                     const struct vk_pipeline_robustness_state *in_rs)
{
   const struct vk_pipeline_robustness_state *rs = in_rs;
   if (KK_DEBUG(FORCE_ROBUSTNESS))
      rs = &rs_all_supported;
   const struct vk_pipeline_robustness_state hb_rs = kk_hb_robustness(rs);
   rs = &hb_rs;
   return (struct spirv_to_nir_options){
      .environment = NIR_SPIRV_VULKAN,
      .ssbo_addr_format = kk_buffer_addr_format(rs->storage_buffers),
      .phys_ssbo_addr_format = nir_address_format_64bit_global,
      .ubo_addr_format = kk_buffer_addr_format(rs->uniform_buffers),
      .shared_addr_format = nir_address_format_32bit_offset,
      .min_ssbo_alignment = KK_MIN_SSBO_ALIGNMENT,
      .min_ubo_alignment = KK_MIN_UBO_ALIGNMENT,
   };
}

static void
kk_preprocess_nir(UNUSED struct vk_physical_device *vk_pdev, nir_shader *nir,
                  UNUSED const struct vk_pipeline_robustness_state *rs)
{
   /* Gather info before preprocess_nir but after some general lowering, so
    * inputs_read and system_values_read are accurately set.
    */
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

   msl_preprocess_nir(nir);

   /* Cannot be part of msl_preprocess_nir since clc does not expose
    * has_base_workgroup_id */
   nir_lower_compute_system_values_options csv_options = {
      .has_base_global_invocation_id = 0,
      .has_base_workgroup_id = true,
   };
   NIR_PASS(_, nir, nir_lower_compute_system_values, &csv_options);
}

static bool
has_static_depth_stencil_state(const struct vk_graphics_pipeline_state *state)
{
   if (!state->ds)
      return false;

   return !(
      BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_DS_DEPTH_TEST_ENABLE) |
      BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_DS_DEPTH_WRITE_ENABLE) |
      BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_DS_DEPTH_COMPARE_OP) |
      BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_DS_STENCIL_TEST_ENABLE) |
      BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_DS_STENCIL_OP) |
      BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_DS_STENCIL_COMPARE_MASK) |
      BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_DS_STENCIL_WRITE_MASK));
}

struct kk_vs_key {
   bool is_points;
   /* Topology class, which picks the passthrough GS input */
   enum mesa_prim reduced_prim;
   struct vk_vertex_input_state vi;
};

static void
kk_populate_vs_key(struct kk_vs_key *key,
                   const struct vk_graphics_pipeline_state *state)
{
   memset(key, 0, sizeof(*key));
   key->is_points =
      (state->ia->primitive_topology == VK_PRIMITIVE_TOPOLOGY_POINT_LIST);
   key->reduced_prim =
      u_reduced_prim(vk_topology_to_mesa(state->ia->primitive_topology));
   key->vi = *state->vi;
}

static bool kk_is_possible_both_depth_clip_clamp(
   const struct vk_graphics_pipeline_state *state, bool enabled);
static inline uint8_t
kk_get_logical_color_att_index(const struct vk_graphics_pipeline_state *state,
                               uint8_t i);

struct kk_fs_key {
   VkFormat color_formats[MESA_VK_MAX_COLOR_ATTACHMENTS];
   struct vk_color_blend_state color_blend;
   struct vk_depth_stencil_state ds;
   uint32_t rasterization_samples;
   uint16_t static_sample_mask;
   bool sample_shading_enable;
   bool has_depth;
   bool has_stencil;
   bool dynamic_input_attachment_map;
   uint8_t depth_input_att;
   uint8_t stencil_input_att;
   /* The VS holding the pipeline is built from these too
    * (gather_graphics_pipeline_create_info), so equal keys must mean equal
    * attachments, not only "has depth" */
   VkFormat depth_format;
   VkFormat stencil_format;
   uint32_t color_attachment_count;
   uint8_t color_map[MESA_VK_MAX_COLOR_ATTACHMENTS];
   /* msl_nir_lower_multisample_alpha bakes both into the fragment shader */
   bool alpha_to_coverage_enable;
   bool alpha_to_one_enable;
};

static void
kk_populate_fs_key(struct kk_fs_key *key,
                   const struct vk_graphics_pipeline_state *state)
{
   memset(key, 0, sizeof(*key));

   /* Required since we [de]serialize blend, and render target swizzle for
    * non-native formats */
   memcpy(key->color_formats, state->rp->color_attachment_formats,
          sizeof(key->color_formats));

   if (has_static_depth_stencil_state(state))
      key->ds = *(state->ds);

   /* Blend state gets [de]serialized, so we need to hash it */
   if (state->cb)
      key->color_blend = *(state->cb);

   if (state->ms) {
      key->rasterization_samples = state->ms->rasterization_samples;
      key->static_sample_mask = state->ms->sample_mask;
      key->sample_shading_enable = state->ms->sample_shading_enable;
      key->alpha_to_coverage_enable = state->ms->alpha_to_coverage_enable;
      key->alpha_to_one_enable = state->ms->alpha_to_one_enable;
   }

   /* Depth writes are removed unless there's an actual attachment */
   key->has_depth = state->rp->depth_attachment_format != VK_FORMAT_UNDEFINED;
   key->has_stencil =
      state->rp->stencil_attachment_format != VK_FORMAT_UNDEFINED;

   /* Values that modify if a forced depth is present or not */
   key->depth_input_att =
      state->ial ? state->ial->depth_att : MESA_VK_ATTACHMENT_NO_INDEX;
   key->stencil_input_att =
      state->ial ? state->ial->stencil_att : MESA_VK_ATTACHMENT_NO_INDEX;
   key->dynamic_input_attachment_map =
      BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_INPUT_ATTACHMENT_MAP);

   key->depth_format = state->rp->depth_attachment_format;
   key->stencil_format = state->rp->stencil_attachment_format;
   key->color_attachment_count = state->rp->color_attachment_count;
   for (uint32_t i = 0; i < MESA_VK_MAX_COLOR_ATTACHMENTS; i++)
      key->color_map[i] = kk_get_logical_color_att_index(state, i);
}

enum kk_feature_key {
   KK_FEAT_CUSTOM_BORDER = BITFIELD_BIT(0),
   KK_FEAT_NULL_DESCRIPTOR = BITFIELD_BIT(1),
   KK_FEAT_IMAGE_VIEW_MIN_LOD = BITFIELD_BIT(2),
   KK_FEAT_SAMPLER_MINMAX = BITFIELD_BIT(3),
   /* Not part of the key: build the program that emulates min/max reduction
    * for samplers from descriptor sets too (kk_compile_shaders) */
   KK_FEAT_SAMPLER_MINMAX_DYNAMIC = BITFIELD_BIT(4),
};

static enum kk_feature_key
kk_make_feature_key(const struct vk_features *feats)
{
   enum kk_feature_key key = 0;
   if (feats->customBorderColors)
      key |= KK_FEAT_CUSTOM_BORDER;
   if (feats->nullDescriptor)
      key |= KK_FEAT_NULL_DESCRIPTOR;
   if (feats->minLod)
      key |= KK_FEAT_IMAGE_VIEW_MIN_LOD;
   if (feats->samplerFilterMinmax)
      key |= KK_FEAT_SAMPLER_MINMAX;
   return key;
}

static void
kk_hash_graphics_state(struct vk_physical_device *device,
                       const struct vk_graphics_pipeline_state *state,
                       const struct vk_features *enabled_features,
                       VkShaderStageFlags stages, blake3_hash blake3_out)
{
   struct mesa_blake3 blake3_ctx;
   _mesa_blake3_init(&blake3_ctx);

   if (stages & VK_SHADER_STAGE_VERTEX_BIT) {
      struct kk_vs_key key;
      kk_populate_vs_key(&key, state);
      _mesa_blake3_update(&blake3_ctx, &key, sizeof(key));
   }

   /* The fragment state is part of every graphics stage's key, not only the
    * fragment shader's: the VS that holds a KosmicKrisp pipeline (and the
    * empty FS built next to it when the pipeline has none) is built from the
    * attachment formats, the depth/stencil state and the sample count
    * (gather_graphics_pipeline_create_info), so two depth-only pipelines with
    * the same VS but D32 vs D16, GREATER vs LESS or 1 vs 4 samples must not
    * share a shader. */
   if (stages & VK_SHADER_STAGE_ALL_GRAPHICS) {
      struct kk_fs_key key;
      kk_populate_fs_key(&key, state);
      _mesa_blake3_update(&blake3_ctx, &key, sizeof(key));

      _mesa_blake3_update(&blake3_ctx, &state->mv->view_mask,
                          sizeof(state->mv->view_mask));
   }

   /* kk_lower_hw_vs and kk_lower_fs lower depth clipping and clamping by what
    * the rasterization state allows */
   if (stages & VK_SHADER_STAGE_ALL_GRAPHICS) {
      const bool clip_clamp[2] = {
         kk_is_possible_both_depth_clip_clamp(state, false),
         kk_is_possible_both_depth_clip_clamp(state, true),
      };
      _mesa_blake3_update(&blake3_ctx, clip_clamp, sizeof(clip_clamp));
   }

   enum kk_feature_key feature_key = kk_make_feature_key(enabled_features);
   _mesa_blake3_update(&blake3_ctx, &feature_key, sizeof(feature_key));
   const bool force_robustness = KK_DEBUG(FORCE_ROBUSTNESS);
   _mesa_blake3_update(&blake3_ctx, &force_robustness,
                       sizeof(force_robustness));
   /* Highball: the robustness switches change SPIR-V translation as well. */
   const uint8_t hb_robust[3] = {
      debug_get_bool_option("HB_KK_NO_UBO_ROBUST", false),
      debug_get_bool_option("HB_KK_NO_SSBO_ROBUST", false),
      debug_get_bool_option("HB_KK_NO_VERTEX_ROBUST", false),
   };
   _mesa_blake3_update(&blake3_ctx, hb_robust, sizeof(hb_robust));

   _mesa_blake3_final(&blake3_ctx, blake3_out);
}

static void
shared_var_info(const struct glsl_type *type, unsigned *size, unsigned *align)
{
   assert(glsl_type_is_vector_or_scalar(type));

   uint32_t comp_size =
      glsl_type_is_boolean(type) ? 4 : glsl_get_bit_size(type) / 8;
   unsigned length = glsl_get_vector_elements(type);
   *size = comp_size * length, *align = comp_size;
}

struct lower_ycbcr_state {
   uint32_t set_layout_count;
   struct vk_descriptor_set_layout *const *set_layouts;
};

static const struct vk_ycbcr_conversion_state *
lookup_ycbcr_conversion(const void *_state, uint32_t set, uint32_t binding,
                        uint32_t array_index)
{
   const struct lower_ycbcr_state *state = _state;
   assert(set < state->set_layout_count);
   assert(state->set_layouts[set] != NULL);
   const struct kk_descriptor_set_layout *set_layout =
      vk_to_kk_descriptor_set_layout(state->set_layouts[set]);
   assert(binding < set_layout->binding_count);

   const struct kk_descriptor_set_binding_layout *bind_layout =
      &set_layout->binding[binding];

   if (bind_layout->immutable_samplers == NULL)
      return NULL;

   array_index = MIN2(array_index, bind_layout->array_size - 1);

   const struct kk_sampler *sampler =
      bind_layout->immutable_samplers[array_index];

   return sampler && sampler->vk.ycbcr_conversion
             ? &sampler->vk.ycbcr_conversion->state
             : NULL;
}

static unsigned
type_size_vec4(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

static bool
kk_nir_swizzle_fragment_output(nir_builder *b, nir_intrinsic_instr *intrin,
                               void *data)
{
   if (intrin->intrinsic != nir_intrinsic_store_output &&
       intrin->intrinsic != nir_intrinsic_load_output)
      return false;

   unsigned slot = nir_intrinsic_io_semantics(intrin).location;
   if (slot < FRAG_RESULT_DATA0)
      return false;

   const struct vk_graphics_pipeline_state *state =
      (const struct vk_graphics_pipeline_state *)data;
   VkFormat vk_format =
      state->rp->color_attachment_formats[slot - FRAG_RESULT_DATA0];
   if (vk_format == VK_FORMAT_UNDEFINED)
      return false;

   enum pipe_format format = vk_format_to_pipe_format(vk_format);
   const struct kk_va_format *supported_format = kk_get_va_format(format);

   /* Check if we have to apply any swizzle */
   if (!supported_format->is_native) {
      unsigned channel_unswizzle[] = {
         supported_format->unswizzle.red, supported_format->unswizzle.green,
         supported_format->unswizzle.blue, supported_format->unswizzle.alpha};

      if (intrin->intrinsic == nir_intrinsic_store_output) {
         unsigned channel_swizzle[4] = {0u};
         for (uint32_t i = 0u; i < 4; ++i)
            channel_swizzle[channel_unswizzle[i]] = i;

         b->cursor = nir_before_instr(&intrin->instr);
         nir_def *to_replace = intrin->src[0].ssa;
         nir_def *swizzled = nir_swizzle(b, to_replace, channel_swizzle,
                                         to_replace->num_components);
         nir_src_rewrite(&intrin->src[0], swizzled);
      } else {
         b->cursor = nir_after_instr(&intrin->instr);
         nir_def *to_replace = &intrin->def;
         nir_def *swizzled = nir_swizzle(b, to_replace, channel_unswizzle,
                                         to_replace->num_components);
         nir_def_rewrite_uses_after(to_replace, swizzled);
      }
      return true;
   }

   return false;
}

static bool
kk_is_possible_both_depth_clip_clamp(
   const struct vk_graphics_pipeline_state *state, bool enabled)
{
   bool dyn_clamp =
      BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_RS_DEPTH_CLAMP_ENABLE);
   bool dyn_clip =
      BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_RS_DEPTH_CLIP_ENABLE);

   /* If rasterization state is null, clamp must be dynamic, and clip must be
    * inverted from clamp if not also dynamic. Thus, they cannot match unless
    * both are dynamic. */
   if (!state->rs)
      return dyn_clamp && dyn_clip;

   /* If clip is static and inverted from clamp, they can never be the same */
   if (!dyn_clip &&
       state->rs->depth_clip_enable == VK_MESA_DEPTH_CLIP_ENABLE_NOT_CLAMP)
      return false;

   /* Need to account for:
    * - Both are dynamic
    * - Both are static and match the desired value
    * - One is dynamic and the other is static and matches the desired value
    */
   bool static_clamp_match = state->rs->depth_clamp_enable == enabled;
   bool static_clip_match =
      vk_rasterization_state_depth_clip_enable(state->rs) == enabled;
   return (dyn_clamp || static_clamp_match) && (dyn_clip || static_clip_match);
}

static void
kk_lower_vs_vbo(nir_shader *nir, const struct vk_graphics_pipeline_state *state,
                const struct vk_pipeline_robustness_state *rs)
{
   assert(!(nir->info.inputs_read & BITFIELD64_MASK(VERT_ATTRIB_GENERIC0)) &&
          "Fixed-function attributes not used in Vulkan");
   NIR_PASS(_, nir, nir_recompute_io_bases, nir_var_shader_in);
   /* the shader_out portion of this is load-bearing even for tess eval */
   /* Fold constant offset srcs for IO. */
   NIR_PASS(_, nir, nir_opt_constant_folding);

   struct kk_attribute attributes[KK_MAX_ATTRIBS] = {};
   uint64_t attribs_read = nir->info.inputs_read >> VERT_ATTRIB_GENERIC0;
   u_foreach_bit(i, state->vi->attributes_valid) {
      const struct vk_vertex_attribute_state *attr = &state->vi->attributes[i];
      assert(state->vi->bindings_valid & BITFIELD_BIT(attr->binding));
      const struct vk_vertex_binding_state *binding =
         &state->vi->bindings[attr->binding];

      /* nir_assign_io_var_locations compacts vertex inputs, eliminating
       * unused inputs. We need to do the same here to match the locations.
       */
      unsigned slot = util_bitcount64(attribs_read & BITFIELD_MASK(i));
      attributes[slot].divisor = binding->divisor;
      attributes[slot].binding = attr->binding;
      attributes[slot].format = vk_format_to_pipe_format(attr->format);
      attributes[slot].buf = attr->binding;
      attributes[slot].instanced =
         binding->input_rate == VK_VERTEX_INPUT_RATE_INSTANCE;
   }
   const struct vk_pipeline_robustness_state hb_rs = kk_hb_robustness(rs);
   bool robustness2 =
      hb_rs.vertex_inputs ==
      VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2;
   NIR_PASS(_, nir, kk_nir_lower_vbo, attributes, robustness2);
}

/* Lowering for the stage which ends up as the vertex stage on hardware */
static void
kk_lower_hw_vs(nir_shader *nir, const struct vk_graphics_pipeline_state *state,
               bool is_point)
{
   if (is_point)
      NIR_PASS(_, nir, msl_ensure_vertex_point_size_output);
   else
      nir_shader_intrinsics_pass(nir, msl_nir_vs_remove_point_size_write,
                                 nir_metadata_control_flow, NULL);

   NIR_PASS(_, nir, msl_ensure_vertex_position_output);
   NIR_PASS(_, nir, nir_lower_clip_halfz_dynamic);

   /* In any situation where both clip and clamp can be disabled, we need to
    * add viewport Z transform emulation to the shader */
   if (kk_is_possible_both_depth_clip_clamp(state, false))
      NIR_PASS(_, nir, msl_nir_lower_vs_disabled_depth_clamp_clip);

   NIR_PASS(_, nir, msl_nir_vs_io_types);
}

static inline uint8_t
kk_get_logical_color_att_index(const struct vk_graphics_pipeline_state *state,
                               uint8_t i)
{
   return (state->cal) ? state->cal->color_map[i] : i;
}

static void
kk_lower_fs_blend(nir_shader *nir,
                  const struct vk_graphics_pipeline_state *state)
{
   nir_lower_blend_options opts = {
      .scalar_blend_const = false,
      .logicop_enable = state->cb->logic_op_enable,
      .logicop_func = vk_logic_op_to_pipe(state->cb->logic_op),
   };

   static_assert(ARRAY_SIZE(opts.rt) == 8, "max RTs out of sync");
   for (unsigned i = 0; i < ARRAY_SIZE(opts.rt); ++i) {
      opts.rt[i].format = PIPE_FORMAT_NONE;
   }

   for (unsigned i = 0; i < ARRAY_SIZE(opts.rt); ++i) {
      uint8_t logical_index = kk_get_logical_color_att_index(state, i);
      if (logical_index == MESA_VK_ATTACHMENT_UNUSED) {
         continue;
      }

      nir_lower_blend_rt *rt = &opts.rt[logical_index];

      const struct vk_color_blend_attachment_state *att =
         &state->cb->attachments[i];
      rt->format =
         vk_format_to_pipe_format(state->rp->color_attachment_formats[i]);
      rt->colormask = att->write_mask;

      if (!att->blend_enable) {
         rt->rgb.src_factor = rt->alpha.src_factor = PIPE_BLENDFACTOR_ONE;
         rt->rgb.dst_factor = rt->alpha.dst_factor = PIPE_BLENDFACTOR_ZERO;
         rt->rgb.func = rt->alpha.func = PIPE_BLEND_ADD;
      } else if (att->color_blend_op >= VK_BLEND_OP_ZERO_EXT) {
         rt->advanced_blend = true;
         rt->blend_mode = vk_advanced_blend_op_to_pipe(att->color_blend_op);
         rt->src_premultiplied = att->src_premultiplied;
         rt->dst_premultiplied = att->dst_premultiplied;
         rt->overlap = vk_blend_overlap_to_pipe(att->blend_overlap);

         /* Unused by runtime and not supported */
         assert(!att->clamp_results);
      } else {
         rt->rgb.src_factor =
            vk_blend_factor_to_pipe(att->src_color_blend_factor);
         rt->rgb.dst_factor =
            vk_blend_factor_to_pipe(att->dst_color_blend_factor);
         rt->rgb.func = vk_blend_op_to_pipe(att->color_blend_op);

         rt->alpha.src_factor =
            vk_blend_factor_to_pipe(att->src_alpha_blend_factor);
         rt->alpha.dst_factor =
            vk_blend_factor_to_pipe(att->dst_alpha_blend_factor);
         rt->alpha.func = vk_blend_op_to_pipe(att->alpha_blend_op);
      }
   }
   /* Fold constant offset srcs for IO. */
   NIR_PASS(_, nir, nir_opt_constant_folding);
   NIR_PASS(_, nir, nir_lower_blend, &opts);
}

static bool
lower_subpass_dim(nir_builder *b, nir_tex_instr *tex, UNUSED void *_data)
{
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_SUBPASS)
      tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   else if (tex->sampler_dim == GLSL_SAMPLER_DIM_SUBPASS_MS)
      tex->sampler_dim = GLSL_SAMPLER_DIM_MS;
   else
      return false;

   return true;
}

static void
kk_lower_fs(struct kk_device *dev, nir_shader *nir,
            const struct vk_graphics_pipeline_state *state)
{
   struct kk_physical_device *pdev = kk_device_physical(dev);

   nir->info.fs.uses_sample_shading |=
      state->ms && state->ms->sample_shading_enable;

   /* Must be lowered before blending */
   if (state->ms) {
      NIR_PASS(_, nir, msl_nir_lower_multisample_alpha,
               state->ms->alpha_to_coverage_enable,
               state->ms->alpha_to_one_enable);
   }

   /* msl_nir_lower_sample_shading needs to go before blending since
    * nir_lower_blend will always set uses_sample_shading to true if there's any
    * output read. I believe we do not need to lower it always, that is why it
    * goes here but need to double check. */
   if (nir->info.fs.uses_sample_shading)
      NIR_PASS(_, nir, msl_nir_lower_sample_shading);

   if (state->cb)
      kk_lower_fs_blend(nir, state);

   enum pipe_format rts[MAX_DRAW_BUFFERS] = {PIPE_FORMAT_NONE};
   const struct vk_render_pass_state *rp = state->rp;
   for (uint32_t i = 0u; i < rp->color_attachment_count; ++i) {
      uint8_t logical_index = kk_get_logical_color_att_index(state, i);
      if (logical_index != MESA_VK_ATTACHMENT_UNUSED) {
         rts[logical_index] =
            vk_format_to_pipe_format(rp->color_attachment_formats[i]);
      }
   }

   NIR_PASS(_, nir, msl_nir_fs_force_output_signedness, rts);

   if (state->rp->depth_attachment_format == VK_FORMAT_UNDEFINED ||
       nir->info.fs.early_fragment_tests) {
      NIR_PASS(_, nir, msl_nir_fs_remove_depth_write);
   }

   /* In any situation where both clip and clamp can be enabled, we need to
    * add clamp emulation to the shader */
   if (kk_is_possible_both_depth_clip_clamp(state, true))
      NIR_PASS(_, nir, msl_nir_lower_fs_combined_depth_clamp_clip);

   /* Input attachments are treated as 2D textures. Fixes sampler dimension */
   NIR_PASS(_, nir, nir_shader_tex_pass, lower_subpass_dim, nir_metadata_all,
            NULL);

   /* Swizzle non-native formats' outputs */
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, kk_nir_swizzle_fragment_output,
            nir_metadata_control_flow, (void *)state);

   NIR_PASS(_, nir, msl_nir_fs_io_types);

   if (state->ms && state->ms->rasterization_samples &&
       state->ms->sample_mask != UINT16_MAX) {

      /* KK_WORKAROUND_7 */
      if (!(pdev->settings.disabled_workarounds & BITFIELD64_BIT(7))) {
         if (!nir->info.fs.early_fragment_tests) {
            nir_function_impl *entrypoint = nir_shader_get_entrypoint(nir);
            nir_builder b = nir_builder_at(nir_after_impl(entrypoint));

            nir_def *sample_id = nir_load_sample_id(&b);
            nir_def *sample_bit = nir_ishl(&b, nir_imm_int(&b, 1), sample_id);
            nir_def *sample_mask_bit =
               nir_iand(&b, nir_load_sample_mask_in(&b), sample_bit);
            nir_discard_if(&b, nir_ieq_imm(&b, sample_mask_bit, 0u));
         }
      }
      NIR_PASS(_, nir, msl_lower_static_sample_mask, state->ms->sample_mask);
   }
   /* Check https://github.com/KhronosGroup/Vulkan-Portability/issues/54 for
    * explanation on why we need this. */
   else if (nir->info.fs.needs_full_quad_helper_invocations ||
            nir->info.fs.needs_coarse_quad_helper_invocations) {
      struct kk_physical_device *pdev = kk_device_physical(dev);

      /* Metal by default merges triangles which results in incorrect edges, see
       * mentioned issue above. The issue is that if we disable them through
       * writing to the sample mask, then derivatives are broken for multisample
       * rendering in M1 and M2 with macOS 26. This bug is fixed in macOS 27.
       *
       * This is why we are choosing the lesser evil which is incorrect edges
       * for multisampled rendering. Since CTS does not have tests for the edge
       * case with multisample, we get a clean run. This does not mean we are
       * Vulkan conformant with macOS26 for M1 and M2.
       */
      bool ms_bug_present = !ns_is_os_version_at_least(27, 0, 0) &&
                            (pdev->info.gpu_apple_family == 7 ||
                             pdev->info.gpu_apple_family == 8) &&
                            state->ms && state->ms->rasterization_samples > 1;
      if (!ms_bug_present)
         NIR_PASS(_, nir, msl_disable_triangle_merge);
   }

   /* KK_WORKAROUND_5 */
   if (!(pdev->settings.disabled_workarounds & BITFIELD64_BIT(5)))
      NIR_PASS(_, nir, msl_nir_fake_guard_for_discards);
   /* KK_WORKAROUND_4 */
   if (!(pdev->settings.disabled_workarounds & BITFIELD64_BIT(4))) {
      NIR_PASS(_, nir, nir_lower_helper_writes, true);
      NIR_PASS(_, nir, nir_lower_is_helper_invocation);
   }
}

static bool
kk_fs_reads_input_attachment(const nir_shader *nir, uint32_t depth_att,
                             uint32_t stencil_att, bool indices_known)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            const nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_image_deref_load &&
                intr->intrinsic != nir_intrinsic_image_deref_sparse_load)
               continue;

            nir_deref_instr *deref = nir_src_as_deref(intr->src[0u]);
            enum glsl_sampler_dim dim = glsl_get_sampler_dim(deref->type);
            if (dim != GLSL_SAMPLER_DIM_SUBPASS &&
                dim != GLSL_SAMPLER_DIM_SUBPASS_MS)
               continue;

            nir_variable *var = nir_deref_instr_get_variable(deref);
            if (!indices_known || var == NULL || var->data.index == depth_att ||
                var->data.index == stencil_att)
               return true;
         }
      }
   }

   return false;
}

static bool
kk_fs_needs_forced_depth_write(const nir_shader *nir,
                               const struct vk_graphics_pipeline_state *state)
{
   const struct vk_input_attachment_location_state *ial = state->ial;
   if (ial == NULL)
      return false;

   const bool has_depth_ia =
      state->rp->depth_attachment_format != VK_FORMAT_UNDEFINED &&
      ial->depth_att != MESA_VK_ATTACHMENT_NO_INDEX;
   const bool has_stencil_ia =
      state->rp->stencil_attachment_format != VK_FORMAT_UNDEFINED &&
      ial->stencil_att != MESA_VK_ATTACHMENT_NO_INDEX;

   if (!has_depth_ia && !has_stencil_ia)
      return false;

   if (has_static_depth_stencil_state(state)) {
      const struct vk_depth_stencil_state *ds = state->ds;

      const bool writes_depth =
         has_depth_ia && ds->depth.test_enable && ds->depth.write_enable;
      const bool writes_stencil =
         has_stencil_ia && ds->stencil.test_enable &&
         ds->stencil.write_enable &&
         (ds->stencil.front.write_mask | ds->stencil.back.write_mask);

      if (!writes_depth && !writes_stencil)
         return false;
   }

   const uint32_t depth_att =
      has_depth_ia ? ial->depth_att : NIR_VARIABLE_NO_INDEX;
   const uint32_t stencil_att =
      has_stencil_ia ? ial->stencil_att : NIR_VARIABLE_NO_INDEX;

   const bool indices_known =
      !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_INPUT_ATTACHMENT_MAP);

   return kk_fs_reads_input_attachment(nir, depth_att, stencil_att,
                                       indices_known);
}

static bool
kk_can_remove_io_var(nir_variable *var, void *data)
{
   return !(var->data.mode == nir_var_shader_out &&
            var->data.location == VARYING_SLOT_POS && var->data.invariant);
}

static void
kk_lower_nir(struct kk_device *dev, nir_shader *nir, bool emulated_stage,
             const struct vk_pipeline_robustness_state *rs,
             uint32_t set_layout_count,
             struct vk_descriptor_set_layout *const *set_layouts,
             const struct vk_graphics_pipeline_state *state,
             enum kk_feature_key features, uint32_t *minmax_sets)
{
   struct kk_physical_device *pdev = kk_device_physical(dev);

   if (nir->info.io_lowered)
      return;

   if (KK_DEBUG(FORCE_ROBUSTNESS))
      rs = &rs_all_supported;
   const struct vk_pipeline_robustness_state hb_rs = kk_hb_robustness(rs);
   rs = &hb_rs;

   const nir_opt_access_options access_options = {
      .is_vulkan = true,
   };
   NIR_PASS(_, nir, nir_opt_access, &access_options);

   /* Massage IO related variables to please Metal */
   if (nir->info.stage == MESA_SHADER_VERTEX) {
      NIR_PASS(_, nir, kk_nir_lower_vs_multiview, state->mv->view_mask);

      /* kk_nir_lower_vs_multiview may create a temporary array to assign the
       * correct view index. Since we don't handle derefs, we need to get rid of
       * them. */
      NIR_PASS(_, nir, nir_lower_vars_to_scratch, 0,
               glsl_get_natural_size_align_bytes,
               glsl_get_natural_size_align_bytes);
   } else if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      NIR_PASS(_, nir, kk_nir_lower_fs_multiview, state->mv->view_mask);

      /* Run before nir_lower_input_attachments() below, which rewrites the
       * subpass loads it looks for. */
      if (kk_fs_needs_forced_depth_write(nir, state))
         NIR_PASS(_, nir, msl_ensure_depth_write);

      msl_nir_lower_input_attachments(nir, state->ial, state->cal);
   }

   const struct lower_ycbcr_state ycbcr_state = {
      .set_layout_count = set_layout_count,
      .set_layouts = set_layouts,
   };
   NIR_PASS(_, nir, nir_vk_lower_ycbcr_tex, lookup_ycbcr_conversion,
            &ycbcr_state);

   /* Common msl texture lowering needs to happen after ycbcr lowering and
    * before descriptor lowering. */
   NIR_PASS(_, nir, msl_lower_textures, pdev->info.gpu_apple_family < 10);

   /* Lower push constants before lower_descriptors */
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_push_const,
            nir_address_format_32bit_offset);

   NIR_PASS(_, nir, nir_lower_memory_model);

   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_global,
            nir_address_format_64bit_global);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ssbo,
            kk_buffer_addr_format(rs->storage_buffers));
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ubo,
            kk_buffer_addr_format(rs->uniform_buffers));

   NIR_PASS(_, nir, nir_lower_io, nir_var_shader_in | nir_var_shader_out,
            type_size_vec4,
            nir_lower_io_lower_64bit_to_32 |
               nir_lower_io_use_interpolated_input_intrinsics);

   NIR_PASS(_, nir, nir_lower_vars_to_explicit_types, nir_var_mem_shared,
            shared_var_info);

   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_shared,
            nir_address_format_32bit_offset);

   if (nir->info.zero_initialize_shared_memory && nir->info.shared_size > 0) {
      /* QMD::SHARED_MEMORY_SIZE requires an alignment of 256B so it's safe to
       * align everything up to 16B so we can write whole vec4s.
       */
      nir->info.shared_size = align(nir->info.shared_size, 16);
      NIR_PASS(_, nir, nir_zero_initialize_shared_memory, nir->info.shared_size,
               16);

      /* We need to call lower_compute_system_values again because
       * nir_zero_initialize_shared_memory generates load_invocation_id which
       * has to be lowered to load_invocation_index.
       */
      NIR_PASS(_, nir, nir_lower_compute_system_values, NULL);
   }

   NIR_PASS(_, nir, nir_opt_dce);
   NIR_PASS(_, nir, nir_lower_variable_initializers, ~nir_var_function_temp);
   const nir_remove_dead_variables_options dead_io_options = {
      .can_remove_var = kk_can_remove_io_var,
   };
   NIR_PASS(_, nir, nir_remove_dead_variables,
            nir_var_shader_in | nir_var_shader_out | nir_var_system_value,
            &dead_io_options);
   nir->info.io_lowered = true;

   /* Required before kk_nir_lower_vbo so load_input intrinsics' parents are
    * load_const, otherwise the pass will complain */
   NIR_PASS(_, nir, nir_opt_constant_folding);

   /* These passes operate on lowered IO. */
   if ((nir->info.stage == MESA_SHADER_VERTEX ||
        nir->info.stage == MESA_SHADER_TESS_EVAL) &&
       !emulated_stage) {
      bool is_point =
         nir->info.stage == MESA_SHADER_TESS_EVAL
            ? nir->info.tess.point_mode
            : state->ia->primitive_topology == VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
      kk_lower_hw_vs(nir, state, is_point);
   } else if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      kk_lower_fs(dev, nir, state);
   }

   if (features & KK_FEAT_IMAGE_VIEW_MIN_LOD)
      NIR_PASS(_, nir, kk_nir_lower_image_view_min_lod);

   /* Apply view clamps to lambda, not to the individual integer-level fetches.
    * Border emulation then operates on those point samples. */
   if (pdev->info.gpu_apple_family < 10 &&
       (features & KK_FEAT_SAMPLER_MINMAX))
      NIR_PASS(_, nir, kk_nir_lower_sampler_minmax,
               features & KK_FEAT_SAMPLER_MINMAX_DYNAMIC, set_layout_count,
               set_layouts, minmax_sets);

   if (features & KK_FEAT_CUSTOM_BORDER)
      NIR_PASS(_, nir, kk_nir_lower_custom_border);

   /* Descriptor lowering needs to happen after lowering blend since we will
    * generate a nir_intrinsic_load_blend_const_color_rgba which gets lowered by
    * the lower descriptor pass
    */
   NIR_PASS(_, nir, kk_nir_lower_descriptors, rs, set_layout_count,
            set_layouts);

   /* KK_WORKAROUND_16 - must be called after kk_nir_lower_descriptors(), which
    * lowers all image intrinsics to be bindless */
   if (rs->images ==
          VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_ROBUST_IMAGE_ACCESS_2 &&
       !(pdev->settings.disabled_workarounds & BITFIELD64_BIT(16)))
      NIR_PASS(_, nir, msl_lower_robustness2_images);

   /* KK_WORKAROUND_18 */
   if (!(pdev->settings.disabled_workarounds & BITFIELD64_BIT(18)))
      NIR_PASS(_, nir, kk_nir_add_device_barrier_workaround);

   NIR_PASS(_, nir, kk_nir_lower_textures);

   if (features & KK_FEAT_NULL_DESCRIPTOR)
      NIR_PASS(_, nir, kk_nir_lower_null_images);

   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
}

static const struct vk_shader_ops kk_shader_ops;

static void kk_shader_free(struct kk_device *dev, struct kk_shader *shader,
                           const VkAllocationCallbacks *pAllocator);

static void
kk_shader_destroy(struct vk_device *vk_dev, struct kk_shader *shader,
                  const VkAllocationCallbacks *pAllocator)
{
   struct kk_device *dev = container_of(vk_dev, struct kk_device, vk);

   /* A pool job or a never-queued first-use compile may still be writing the
    * shader's pipeline handles (kk_compile_pool_init, kk_compile_wait). Waiting
    * for it here blocks the ring thread on Metal, and a title destroys
    * pipelines right after creating them (Dragon's Dogma 2's pipeline context
    * spent 73% of its time in vkDestroyPipeline waiting for compiles), so the
    * compiling thread frees it instead. The compile still finishes: an equal pipeline
    * created later gets its Metal pipeline state from the device's tables.
    * The runtime destroys pipeline shaders with the device's allocator; an
    * application allocator may not outlive this call, so with one the destroy
    * still waits. */
   const bool device_alloc = pAllocator == NULL || pAllocator == &dev->vk.alloc;
   mtx_lock(&dev->compile_mtx);
   const bool compiling =
      (shader->compile_queued || shader->compile_running) && !shader->compile_done;
   if (compiling && device_alloc) {
      shader->compile_orphaned = true;
      mtx_unlock(&dev->compile_mtx);
      return;
   }
   mtx_unlock(&dev->compile_mtx);
   if (compiling)
      (void) kk_compile_wait(dev, shader);

   kk_shader_free(dev, shader, pAllocator);
}

static void
kk_shader_free(struct kk_device *dev, struct kk_shader *shader,
               const VkAllocationCallbacks *pAllocator)
{
   struct vk_device *vk_dev = &dev->vk;
   struct kk_pipeline_handles *pipe = &shader->pipeline;

   if (shader->info.stage == MESA_SHADER_COMPUTE) {
      mtl_release(pipe->cs);
      pipe->cs = NULL;
      if (pipe->rt_table) {
         kk_device_remove_allocation_from_residency_set(dev, pipe->rt_table);
         mtl_release(pipe->rt_table);
         pipe->rt_table = NULL;
      }
      kk_rt_link_unref(shader->rt_link);
      shader->rt_link = NULL;
   } else if (shader->info.stage == MESA_SHADER_VERTEX) {
      /* Pre-render compute pipelines. */
      for (uint32_t i = 0u; i < pipe->gfx.pre_render_count; ++i) {
         mtl_release(pipe->gfx.pre_render[i]);
         pipe->gfx.pre_render[i] = NULL;
      }
      pipe->gfx.pre_render_count = 0u;

      if (pipe->gfx.gs_count)
         mtl_release(pipe->gfx.gs_count);
      if (pipe->gfx.pre_gs)
         mtl_release(pipe->gfx.pre_gs);

      /* Graphics pipeline. */
      mtl_release(pipe->gfx.render);
      if (pipe->gfx.ds_handle)
         mtl_release(pipe->gfx.ds_handle);
      pipe->gfx.render = NULL;
      pipe->gfx.ds_handle = NULL;

      for (uint32_t i = 0u; i < pipe->gfx.pre_render_count; ++i)
         mtl_release(pipe->gfx.pre_render[i]);

      /* Serialization data. */
      u_foreach_bit(i, shader->info.vs.additional_stages_bits) {
         struct msl_compile_data *data = &shader->msl_data[i];
         ralloc_free((void *)data->entrypoint_name);
         kk_msl_source_finish(data);
      }
   }

   struct msl_compile_data *data = &shader->msl_data[shader->info.stage];
   ralloc_free((void *)data->entrypoint_name);
   kk_msl_source_finish(data);

   ralloc_free((void *)shader->gs_main.entrypoint_name);
   kk_msl_source_finish(&shader->gs_main);
   ralloc_free((void *)shader->gs_count.entrypoint_name);
   kk_msl_source_finish(&shader->gs_count);
   ralloc_free((void *)shader->pre_gs.entrypoint_name);
   kk_msl_source_finish(&shader->pre_gs);

   if (shader->minmax)
      kk_shader_destroy(vk_dev, shader->minmax, pAllocator);

   vk_shader_free(&dev->vk, pAllocator, &shader->vk);
}

static bool
gather_vs_inputs(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_input)
      return false;

   struct nir_io_semantics io = nir_intrinsic_io_semantics(intr);
   BITSET_WORD *attribs_read = data;
   BITSET_SET(attribs_read, (io.location - VERT_ATTRIB_GENERIC0));
   return false;
}

static bool
fs_uses_flat_varying(nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block_safe(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type == nir_instr_type_intrinsic) {
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_load_input)
                  return true;
            }
         }
      }
   }
   return false;
}

static void
gather_shader_info(struct kk_shader *shader, nir_shader *nir,
                   const struct vk_graphics_pipeline_state *state)
{
   shader->info.stage = nir->info.stage;
   shader->info.uses_per_draw_data = msl_gather_uses_per_draw_data(nir);
   shader->info.writes_memory = false;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_intrinsic_writes_external_memory(nir_instr_as_intrinsic(instr)))
               shader->info.writes_memory = true;
         }
      }
   }
   shader->info.num_cull_distances = nir->info.cull_distance_array_size;
   if (nir->info.stage == MESA_SHADER_VERTEX) {
      nir_shader_intrinsics_pass(nir, gather_vs_inputs, nir_metadata_all,
                                 &shader->info.vs.attribs_read);
      shader->info.vs.outputs_written = nir->info.outputs_written;
   } else if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      /* Some meta shaders like vk-meta-resolve will have depth_layout as NONE
       * which is not a valid Metal layout */
      if (nir->info.fs.depth_layout == FRAG_DEPTH_LAYOUT_NONE)
         nir->info.fs.depth_layout = FRAG_DEPTH_LAYOUT_ANY;
      shader->info.fs.uses_flat_varyings = fs_uses_flat_varying(nir);
   } else if (nir->info.stage == MESA_SHADER_COMPUTE) {
      shader->info.cs.local_size.x = nir->info.workgroup_size[0];
      shader->info.cs.local_size.y = nir->info.workgroup_size[1];
      shader->info.cs.local_size.z = nir->info.workgroup_size[2];
   }
}

static void
modify_nir_info(nir_shader *nir)
{
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   if (nir->info.stage == MESA_SHADER_VERTEX) {
      /* Vertex attribute fetch is done in shader through argument buffers. */
      nir->info.inputs_read = 0u;
   } else if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      /* Some meta shaders like vk-meta-resolve will have depth_layout as NONE
       * which is not a valid Metal layout */
      if (nir->info.fs.depth_layout == FRAG_DEPTH_LAYOUT_NONE)
         nir->info.fs.depth_layout = FRAG_DEPTH_LAYOUT_ANY;

      /* These values are part of the declaration and go with IO. We only
       * require the instrunctions to understand interpolation mode. */
      BITSET_CLEAR(nir->info.system_values_read,
                   SYSTEM_VALUE_BARYCENTRIC_PERSP_PIXEL);
      BITSET_CLEAR(nir->info.system_values_read,
                   SYSTEM_VALUE_BARYCENTRIC_PERSP_SAMPLE);
      BITSET_CLEAR(nir->info.system_values_read,
                   SYSTEM_VALUE_BARYCENTRIC_PERSP_CENTROID);
      BITSET_CLEAR(nir->info.system_values_read,
                   SYSTEM_VALUE_BARYCENTRIC_LINEAR_PIXEL);
      BITSET_CLEAR(nir->info.system_values_read,
                   SYSTEM_VALUE_BARYCENTRIC_LINEAR_CENTROID);
      BITSET_CLEAR(nir->info.system_values_read,
                   SYSTEM_VALUE_BARYCENTRIC_LINEAR_SAMPLE);
   }
}

static bool
lower_vertex_id_zero_base(struct nir_builder *b, nir_intrinsic_instr *intrin,
                          void *data)
{
   if (intrin->intrinsic != nir_intrinsic_load_vertex_id_zero_base)
      return false;

   b->cursor = nir_instr_remove(&intrin->instr);

   nir_def *id = nir_channel(b, nir_load_global_invocation_id(b, 32), 0);
   nir_def_rewrite_uses(&intrin->def, id);

   return true;
}

static bool
kk_nir_lower_vertex_id_zero_base(struct nir_shader *nir)
{
   bool progress = nir_shader_intrinsics_pass(nir, lower_vertex_id_zero_base,
                                              nir_metadata_control_flow, NULL);

   if (progress) {
      BITSET_SET(nir->info.system_values_read,
                 SYSTEM_VALUE_GLOBAL_INVOCATION_ID);
      BITSET_CLEAR(nir->info.system_values_read,
                   SYSTEM_VALUE_VERTEX_ID_ZERO_BASE);
   }
   return progress;
}

VkResult
kk_compile_rt_function_msl(struct kk_device *dev, nir_shader *nir,
                           const struct vk_pipeline_robustness_state *rs,
                           uint32_t set_layout_count,
                           struct vk_descriptor_set_layout *const *set_layouts,
                           const char *name, char **msl_out)
{
   struct kk_physical_device *pdev = kk_device_physical(dev);

   nir->info.stage = MESA_SHADER_COMPUTE;
   memset(&nir->info.cs, 0, sizeof(nir->info.cs));
   nir->info.workgroup_size[0] = 1;
   nir->info.workgroup_size[1] = 1;
   nir->info.workgroup_size[2] = 1;

   kk_preprocess_nir(&pdev->vk, nir, rs);
   msl_preprocess_nir_workarounds(nir, pdev->settings.disabled_workarounds);
   uint32_t minmax_sets = 0;
   kk_lower_nir(dev, nir, false, rs, set_layout_count, set_layouts, NULL,
                kk_make_feature_key(&dev->vk.enabled_features), &minmax_sets);
   NIR_PASS(_, nir, kk_nir_lower_poly);
   msl_optimize_nir(nir);
   modify_nir_info(nir);

   struct nir_to_msl_options options = {
      .mem_ctx = NULL,
      .disabled_workarounds = pdev->settings.disabled_workarounds,
      .rt_function_name = name,
   };
   char *code = nir_to_msl(nir, &options);
   *msl_out = code ? strdup(code) : NULL;
   ralloc_free(code);
   ralloc_free(nir);
   return *msl_out ? VK_SUCCESS : VK_ERROR_OUT_OF_HOST_MEMORY;
}

/* Consumes nir */
static void
kk_compile_msl(struct kk_physical_device *pdev, nir_shader *nir,
               const struct vk_graphics_pipeline_state *state,
               struct msl_compile_data *data)
{
   NIR_PASS(_, nir, kk_nir_lower_poly);

   msl_optimize_nir(nir);
   modify_nir_info(nir);

   struct nir_to_msl_options translate_options = {
      .mem_ctx = NULL,
      .disabled_workarounds = pdev->settings.disabled_workarounds,
   };
   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      for (uint32_t i = 0u; i < MAX_DRAW_BUFFERS; ++i) {
         /* If the attachment is unused, default to 4 to avoid issues with alpha
          * to coverage with unused attachments. */
         translate_options.rts_component_count[i] = 4;
      }
      for (uint32_t i = 0u; i < state->rp->color_attachment_count; ++i) {
         uint8_t logical_index = kk_get_logical_color_att_index(state, i);
         if (logical_index == MESA_VK_ATTACHMENT_UNUSED) {
            continue;
         }
         enum pipe_format format =
            vk_format_to_pipe_format(state->rp->color_attachment_formats[i]);
         if (format != PIPE_FORMAT_NONE) {
            translate_options.rts_component_count[logical_index] =
               util_format_get_nr_components(format);
         }
      }
   }

   if (debug_get_bool_option("HB_KK_RT_DUMP_FINAL", false) &&
       nir->info.stage == MESA_SHADER_COMPUTE) {
      bool rt = false;
      nir_foreach_function_impl(impl, nir) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type == nir_instr_type_intrinsic &&
                   nir_instr_as_intrinsic(instr)->intrinsic ==
                      nir_intrinsic_rt_call_kk)
                  rt = true;
            }
         }
      }
      if (rt)
         nir_print_shader(nir, stderr);
   }
   data->code = nir_to_msl(nir, &translate_options);
   kk_msl_source_intern(data);
   const char *entrypoint_name = nir_shader_get_entrypoint(nir)->function->name;

   /* We need to steal so it doesn't get destroyed with the nir. Needs to happen
    * after nir_to_msl since that's where we rename the entrypoint.
    */
   ralloc_steal(NULL, (void *)entrypoint_name);
   data->entrypoint_name = (char *)entrypoint_name;

   if (KK_DEBUG(MSL)) {
      /* The hash command buffer records name it by (kk_cb_record) */
      uint32_t hash;
      uint8_t features;
      kk_msl_digest(data->code, &hash, &features);
      mesa_logi("kk: MSL %08x %s:\n%s\n", hash, entrypoint_name, data->code);
   }

   ralloc_free(nir);
}

static VkResult
kk_compile_shader(struct kk_device *dev, nir_shader *nir,
                  struct kk_shader *prev_stage,
                  const struct vk_pipeline_robustness_state *robustness,
                  const struct vk_graphics_pipeline_state *state,
                  const VkAllocationCallbacks *pAllocator,
                  struct kk_shader **shader_out)
{
   assert(nir->info.io_lowered && "nir must have lowered io");
   struct kk_physical_device *pdev = kk_device_physical(dev);

   /* A 64-bit atomic operation MSL cannot express (nir_to_msl.c) has no MSL, and Metal would report
    * it only in the background compile (kk_compile_pool_init) of a program whose pipeline creation
    * has already returned. Fail the shader here instead, so that creation still reports
    * VK_ERROR_INVALID_SHADER_NV and the application can skip the pipeline. */
   if (nir_uses_unsupported_64bit_atomics(nir)) {
      ralloc_free(nir);
      return vk_error(dev, VK_ERROR_INVALID_SHADER_NV);
   }

   struct kk_shader *shader;
   VkResult result = VK_SUCCESS;

   mesa_shader_stage stage = nir->info.stage;
   shader = vk_shader_zalloc(&dev->vk, &kk_shader_ops, stage, pAllocator,
                             sizeof(*shader));
   if (shader == NULL) {
      ralloc_free(nir);
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   gather_shader_info(shader, nir, state);
   if (stage == MESA_SHADER_COMPUTE)
      shader->rt_link = kk_rt_link_ref(kk_rt_get_pending_link());

   unsigned num_cull_distances =
      prev_stage ? prev_stage->info.num_cull_distances : 0;
   /* A GS lowers clip and cull distances on its rasterization VS instead; the
    * stage feeding it has none after linking */
   if (stage != MESA_SHADER_GEOMETRY)
      msl_nir_lower_clip_cull_distance(nir, num_cull_distances);

   /* When using poly to emulate tessellation or GS, the VS and TCS are turned
    * to compute shaders that will be dispatched before the draw call. The TES
    * becomes a VS, or a compute shader when a GS follows. A GS becomes a
    * compute pass plus a VS that rasterizes its output. */
   if (stage == MESA_SHADER_VERTEX) {
      /* VBO lowering needs to go here otherwise, the linking step removes all
       * inputs since we read vertex attributes from UBOs. */
      kk_lower_vs_vbo(nir, state, robustness);
      if (nir->info.next_stage == MESA_SHADER_TESS_CTRL ||
          nir->info.next_stage == MESA_SHADER_GEOMETRY) {
         NIR_PASS(_, nir, poly_nir_lower_vs_before_gs);
         NIR_PASS(_, nir, kk_nir_lower_vertex_id_zero_base);
         nir->info.stage = MESA_SHADER_COMPUTE;
         memset(&nir->info.cs, 0, sizeof(nir->info.cs));
         nir->xfb_info = NULL;
         NIR_PASS(_, nir, poly_nir_lower_sw_vs);
      } else
         /* Metal's instance_id contains base_instance. When the emulation path
          * is taken, since we launch compute, they correctly get translated.
          * For the non-emulated path we need to subtract base_instance... */
         NIR_PASS(_, nir, msl_nir_lower_instance_id);
   } else if (stage == MESA_SHADER_TESS_CTRL) {
      NIR_PASS(_, nir, poly_nir_lower_tcs, false);

      shader->info.tess.info.ccw = nir->info.tess.ccw;
      shader->info.tess.info.points = nir->info.tess.point_mode;
      shader->info.tess.info.spacing = nir->info.tess.spacing;
      shader->info.tess.info.mode = nir->info.tess._primitive_mode;

      shader->info.tess.tcs_output_patch_size = nir->info.tess.tcs_vertices_out;
      shader->info.tess.tcs_per_vertex_outputs =
         poly_tcs_per_vertex_outputs(nir);
      shader->info.tess.tcs_nr_patch_outputs =
         util_last_bit(nir->info.patch_outputs_written);
      shader->info.tess.tcs_output_stride = poly_tcs_output_stride(nir);
   } else if (stage == MESA_SHADER_TESS_EVAL) {
      shader->info.tess.info.ccw = nir->info.tess.ccw;
      shader->info.tess.info.points = nir->info.tess.point_mode;
      shader->info.tess.info.spacing = nir->info.tess.spacing;
      shader->info.tess.info.mode = nir->info.tess._primitive_mode;

      bool to_hw_vs = nir->info.next_stage != MESA_SHADER_GEOMETRY;
      /* This destroys info so it needs to happen after the gather */
      NIR_PASS(_, nir, poly_nir_lower_tes, to_hw_vs);
      if (!to_hw_vs) {
         shader->info.tess.tes_outputs = nir->info.outputs_written;
         NIR_PASS(_, nir, poly_nir_lower_vs_before_gs);
         nir->info.stage = MESA_SHADER_COMPUTE;
         memset(&nir->info.cs, 0, sizeof(nir->info.cs));
         nir->xfb_info = NULL;
      }
   } else if (stage == MESA_SHADER_GEOMETRY) {
      nir_shader *count = NULL, *rast = NULL, *pre_gs = NULL;
      /* Keeps work that only feeds side effects out of the rasterization VS,
       * where the GPU watchdog resets long vertex work */
      NIR_PASS(_, nir, poly_nir_lower_gs, &count, &rast, &pre_gs,
               &shader->info.gs, true);

      /* Keep these for XFB. KK doesn't expose pipeline statistics. */
      if (shader->info.gs.xfb) {
         if (count) {
            count->info.stage = MESA_SHADER_COMPUTE;
            memset(&count->info.cs, 0, sizeof(count->info.cs));
            count->xfb_info = NULL;
            kk_compile_msl(pdev, count, state, &shader->gs_count);
         }
         kk_compile_msl(pdev, pre_gs, state, &shader->pre_gs);
      } else {
         ralloc_free(count);
         ralloc_free(pre_gs);
      }

      /* poly selects the rasterized vertex through shader_temp variables,
       * localize them so msl_optimize_nir turns them into SSA */
      NIR_PASS(_, rast, nir_lower_global_vars_to_local);

      kk_lower_hw_vs(rast, state, shader->info.gs.mode == MESA_PRIM_POINTS);
      msl_nir_lower_clip_cull_distance(rast, 0);
      /* Descriptors were lowered with the GS, this lowers the system values
       * the hardware vertex lowering added */
      NIR_PASS(_, rast, kk_nir_lower_descriptors, robustness, 0, NULL);
      kk_compile_msl(pdev, rast, state,
                     &shader->msl_data[MESA_SHADER_GEOMETRY]);

      nir->info.stage = MESA_SHADER_COMPUTE;
      memset(&nir->info.cs, 0, sizeof(nir->info.cs));
      nir->xfb_info = NULL;
   }

   /* The geometry stage's own slot holds its rasterization VS */
   kk_compile_msl(pdev, nir, state,
                  stage == MESA_SHADER_GEOMETRY ? &shader->gs_main
                                                : &shader->msl_data[stage]);

   *shader_out = shader;

   return result;
}

static const struct vk_pipeline_robustness_state rs_none = {
   .uniform_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED,
   .storage_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED,
   .images = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_ROBUST_IMAGE_ACCESS_2,
};

VkResult
kk_compile_nir_shader(struct kk_device *dev, nir_shader *nir,
                      const VkAllocationCallbacks *alloc,
                      struct kk_shader **shader_out)
{
   const struct kk_physical_device *pdev = kk_device_physical(dev);

   assert(nir->info.stage == MESA_SHADER_COMPUTE);
   if (nir->options == NULL)
      nir->options = kk_get_nir_options((struct vk_physical_device *)&pdev->vk,
                                        nir->info.stage, &rs_none);

   struct kk_shader *shader = NULL;
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   VkResult result =
      kk_compile_shader(dev, nir, NULL, &rs_none, NULL, alloc, &shader);
   if (result != VK_SUCCESS)
      return result;

   *shader_out = shader;

   return VK_SUCCESS;
}

static bool
kk_nir_source_present(nir_src *src, void *data)
{
   return src->ssa != NULL;
}

static bool
kk_nir_check_varyings_input(nir_shader *nir, unsigned pass)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            const char *reason = NULL;
            if (instr->type > nir_instr_type_cmat_call)
               reason = "invalid instruction type";
            else if (instr->type == nir_instr_type_intrinsic &&
                     nir_instr_as_intrinsic(instr)->intrinsic >=
                        nir_num_intrinsics)
               reason = "invalid intrinsic opcode";
            else if (instr->type == nir_instr_type_alu &&
                     nir_instr_as_alu(instr)->op >= nir_num_opcodes)
               reason = "invalid ALU opcode";
            else if (!nir_foreach_src(instr, kk_nir_source_present, NULL))
               reason = "NULL SSA source";
            if (reason) {
               char hash[BLAKE3_HEX_LEN];
               _mesa_blake3_format(hash, nir->info.source_blake3);
               fprintf(stderr, "kk: invalid NIR before varyings pass %u: "
                               "stage=%s shader=%s instruction=%u: %s\n",
                       pass, _mesa_shader_stage_to_string(nir->info.stage),
                       hash, instr->type, reason);
               return false;
            }
         }
      }
   }
   return true;
}

static void
nir_opts(nir_shader *nir, void *data)
{
   bool progress;

   do {
      progress = false;

      NIR_PASS(progress, nir, nir_opt_loop);
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_remove_phis);
      NIR_PASS(progress, nir, nir_opt_dce);

      NIR_PASS(progress, nir, nir_opt_if, 0);
      NIR_PASS(progress, nir, nir_opt_dead_cf);
      NIR_PASS(progress, nir, nir_opt_cse);
      NIR_PASS(progress, nir, nir_opt_licm, NULL);

      NIR_PASS(progress, nir, nir_opt_peephole_select,
               &(nir_opt_peephole_select_options){
                  .limit = 8,
                  .expensive_alu_ok = true,
                  .discard_ok = true,
               });

      NIR_PASS(progress, nir, nir_opt_phi_precision);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_constant_folding);

      NIR_PASS(progress, nir, nir_opt_undef);
      NIR_PASS(progress, nir, nir_opt_loop_unroll);
   } while (progress);
}

static nir_shader *
get_empty_nir(struct kk_device *dev, mesa_shader_stage stage,
              const struct vk_graphics_pipeline_state *state,
              enum kk_feature_key features)
{
   nir_shader *nir = nir_shader_create(
      NULL, stage,
      kk_get_nir_options(&kk_device_physical(dev)->vk, stage, NULL));

   nir_function *function = nir_function_create(nir, "main_entrypoint");
   function->is_entrypoint = true;
   nir_function_impl_create(function);

   const struct vk_pipeline_robustness_state no_robustness = {
      .storage_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED,
      .uniform_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED,
      .vertex_inputs = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED,
      .images = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED,
      .null_uniform_buffer_descriptor = false,
      .null_storage_buffer_descriptor = false,
   };
   uint32_t minmax_sets = 0u;
   kk_lower_nir(dev, nir, false, &no_robustness, 0u, NULL, state, features,
                &minmax_sets);
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

   return nir;
}

/* --- shared Metal pipeline states -------------------------------------------
 * The compiled MSL library is shared per source (kk_get_msl_library), but the
 * Metal pipeline state built on top of it - the part Metal spends tens of
 * milliseconds on - was created once per Vulkan pipeline. A title that rebuilds
 * its pipeline set every launch compiles the same program tens of times: in one
 * Dragon's Dogma 2 run 26,452 program compiles covered 4,594 distinct programs,
 * and the host benchmark compiles one program 197 times in one process. Two
 * pipelines built from the same libraries, entry points and state are the same
 * Metal pipeline, so the later one finds the earlier one here.
 *
 * The keys hold the library handles, which kk_get_msl_library hands out as one
 * canonical instance per source: equal pipelines then compare equal without
 * hashing or holding the MSL again. The entry-point names are copied into the
 * key, because the shader owns its own and outlives nothing here.
 */
/* --- bounded tables of compiled Metal objects ------------------------------
 * The device shares its compiled MSL libraries and Metal pipeline states
 * between pipelines (kk_device::msl_libraries and the pipeline tables): a
 * title asks for the same program once per pipeline, and the compile pool
 * starts many pipelines with the same programs at once. An entry without an
 * object is a compile in flight: one caller compiles, the others wait for it
 * (one MTL4Compiler compiles them one at a time anyway). The tables are
 * bounded: past kk_metal_cache::max entries the least recently used ones are
 * dropped, releasing the table's reference (a pipeline still using the object
 * holds its own). */
struct kk_metal_cache_entry {
   struct list_head lru_link;
   void *key;
   /* A retained Metal object, or NULL while the caller that reserved the entry
    * compiles it */
   void *object;
};

/* Entries per table. Dragon's Dogma 2 compiles ~4,600 distinct programs in a
 * run (two libraries each at most); past this the oldest are compiled again
 * when asked for. MESA_KK_METAL_CACHE_MAX overrides it. */
#define KK_METAL_CACHE_MAX 4096u

static bool
kk_metal_cache_init(struct kk_metal_cache *c, uint32_t (*hash)(const void *),
                    bool (*equal)(const void *, const void *),
                    void (*free_key)(void *), uint32_t max)
{
   c->table = _mesa_hash_table_create(NULL, hash, equal);
   if (c->table == NULL)
      return false;
   list_inithead(&c->lru);
   c->count = 0;
   c->max = max;
   c->free_key = free_key;
   mtx_init(&c->lock, mtx_plain);
   cnd_init(&c->cnd);
   return true;
}

/* Under c->lock */
static void
kk_metal_cache_drop(struct kk_metal_cache *c, struct kk_metal_cache_entry *e)
{
   _mesa_hash_table_remove_key(c->table, e->key);
   list_del(&e->lru_link);
   c->count--;
   if (e->object != NULL)
      mtl_release(e->object);
   c->free_key(e->key);
   free(e);
}

static void
kk_metal_cache_finish(struct kk_metal_cache *c)
{
   if (c->table == NULL)
      return;
   list_for_each_entry_safe(struct kk_metal_cache_entry, e, &c->lru, lru_link)
      kk_metal_cache_drop(c, e);
   _mesa_hash_table_destroy(c->table, NULL);
   c->table = NULL;
   cnd_destroy(&c->cnd);
   mtx_destroy(&c->lock);
}

/* A reference to the object for key, or NULL when this caller has to compile
 * it. A caller that gets NULL owns the table's reservation for the key and
 * publishes with kk_metal_cache_publish(); a caller that gets an object has
 * had its key freed for it. */
static void *
kk_metal_cache_find(struct kk_metal_cache *c, void *key)
{
   mtx_lock(&c->lock);
   struct hash_entry *entry = _mesa_hash_table_search(c->table, key);
   /* An entry without an object is another caller's compile in flight */
   while (entry != NULL &&
          ((struct kk_metal_cache_entry *)entry->data)->object == NULL) {
      cnd_wait(&c->cnd, &c->lock);
      entry = _mesa_hash_table_search(c->table, key);
   }
   void *object = NULL;
   if (entry != NULL) {
      struct kk_metal_cache_entry *e = entry->data;
      object = mtl_retain(e->object);
      list_del(&e->lru_link);
      list_addtail(&e->lru_link, &c->lru);
   } else {
      /* Reserve the key: the placeholder keeps anyone else off it. Out of
       * memory nothing is reserved and the caller compiles on its own. */
      struct kk_metal_cache_entry *e = calloc(1, sizeof(*e));
      if (e != NULL) {
         e->key = key;
         list_addtail(&e->lru_link, &c->lru);
         c->count++;
         _mesa_hash_table_insert(c->table, key, e);
      }
   }
   mtx_unlock(&c->lock);
   if (object != NULL)
      c->free_key(key);
   return object;
}

/* Takes ownership of the reserved key and one reference to object, or of the
 * key alone when object is NULL (the compile failed; nobody waits forever on
 * it, and the next caller tries again). Drops the least recently used entries
 * past the bound. */
static void
kk_metal_cache_publish(struct kk_metal_cache *c, void *key, void *object)
{
   mtx_lock(&c->lock);
   struct hash_entry *entry = _mesa_hash_table_search(c->table, key);
   struct kk_metal_cache_entry *e = entry != NULL ? entry->data : NULL;
   if (e != NULL && e->object == NULL && e->key == key) {
      if (object != NULL) {
         e->object = mtl_retain(object);
         list_del(&e->lru_link);
         list_addtail(&e->lru_link, &c->lru);
      } else {
         kk_metal_cache_drop(c, e); /* frees the key */
      }
   } else {
      /* Nothing was reserved (kk_metal_cache_find out of memory) */
      c->free_key(key);
   }
   list_for_each_entry_safe(struct kk_metal_cache_entry, old, &c->lru,
                            lru_link) {
      if (c->count <= c->max)
         break;
      /* An entry being compiled is published later; its caller holds its key */
      if (old->object != NULL)
         kk_metal_cache_drop(c, old);
   }
   cnd_broadcast(&c->cnd);
   mtx_unlock(&c->lock);
}

/* MSL libraries by the digest of their source. The pipeline keys below name
 * their libraries by the same digests rather than by address: a library may
 * have been dropped from the table and another compiled at its address. */
struct kk_msl_library_key {
   blake3_hash source;
};

static uint32_t
kk_msl_library_key_hash(const void *key)
{
   return _mesa_hash_data(key, sizeof(struct kk_msl_library_key));
}

static bool
kk_msl_library_key_equal(const void *a, const void *b)
{
   return memcmp(a, b, sizeof(struct kk_msl_library_key)) == 0;
}

struct kk_metal_render_key {
   blake3_hash vs_source;
   blake3_hash fs_source;
   char *vs_entrypoint;
   char *fs_entrypoint;
   enum mtl_primitive_topology_class topology;
   enum mtl_pixel_format rt_formats[MAX_DRAW_BUFFERS];
   uint32_t raster_sample_count;
   uint32_t max_amplification;
};

struct kk_metal_compute_key {
   blake3_hash source;
   char *entrypoint;
   uint32_t max_total_threads;
};

static void
kk_metal_render_key_free(void *_key)
{
   struct kk_metal_render_key *key = _key;
   free(key->vs_entrypoint);
   free(key->fs_entrypoint);
   free(key);
}

static struct kk_metal_render_key *
kk_metal_render_key_create(struct kk_shader *vs,
                           const struct msl_compile_data *vs_data,
                           const struct msl_compile_data *fs_data,
                           const blake3_hash vs_source,
                           const blake3_hash fs_source)
{
   struct kk_metal_render_key *key = calloc(1, sizeof(*key));
   if (key == NULL)
      return NULL;

   memcpy(key->vs_source, vs_source, sizeof(key->vs_source));
   memcpy(key->fs_source, fs_source, sizeof(key->fs_source));
   key->vs_entrypoint = strdup(vs_data->entrypoint_name);
   key->fs_entrypoint = strdup(fs_data->entrypoint_name);
   /* What kk_compile_graphics_pipeline puts on the Metal descriptor */
   key->topology = vs->info.vs.topology;
   memcpy(key->rt_formats, vs->info.vs.rt_formats, sizeof(key->rt_formats));
   key->raster_sample_count =
      vs->info.vs.has_ms ? vs->info.vs.sample_count : 1u;
   key->max_amplification =
      vs->info.vs.view_mask ? util_bitcount(vs->info.vs.view_mask) : 0u;

   if (key->vs_entrypoint == NULL || key->fs_entrypoint == NULL) {
      kk_metal_render_key_free(key);
      return NULL;
   }
   return key;
}

static uint32_t
kk_metal_render_key_hash(const void *key)
{
   const struct kk_metal_render_key *k = key;
   uint32_t hash = _mesa_hash_data(k->vs_source, sizeof(k->vs_source));
   hash = _mesa_hash_data_with_seed(k->vs_entrypoint,
                                    strlen(k->vs_entrypoint), hash);
   hash = _mesa_hash_data_with_seed(k->fs_source, sizeof(k->fs_source), hash);
   hash = _mesa_hash_data_with_seed(k->fs_entrypoint,
                                    strlen(k->fs_entrypoint), hash);
   hash = _mesa_hash_data_with_seed(&k->topology, sizeof(k->topology), hash);
   hash = _mesa_hash_data_with_seed(k->rt_formats, sizeof(k->rt_formats),
                                    hash);
   hash = _mesa_hash_data_with_seed(&k->raster_sample_count,
                                    sizeof(k->raster_sample_count), hash);
   return _mesa_hash_data_with_seed(&k->max_amplification,
                                    sizeof(k->max_amplification), hash);
}

static bool
kk_metal_render_key_equal(const void *a, const void *b)
{
   const struct kk_metal_render_key *x = a;
   const struct kk_metal_render_key *y = b;
   return memcmp(x->vs_source, y->vs_source, sizeof(x->vs_source)) == 0 &&
          memcmp(x->fs_source, y->fs_source, sizeof(x->fs_source)) == 0 &&
          x->topology == y->topology &&
          x->raster_sample_count == y->raster_sample_count &&
          x->max_amplification == y->max_amplification &&
          memcmp(x->rt_formats, y->rt_formats, sizeof(x->rt_formats)) == 0 &&
          strcmp(x->vs_entrypoint, y->vs_entrypoint) == 0 &&
          strcmp(x->fs_entrypoint, y->fs_entrypoint) == 0;
}

static void
kk_metal_compute_key_free(void *_key)
{
   struct kk_metal_compute_key *key = _key;
   free(key->entrypoint);
   free(key);
}

static struct kk_metal_compute_key *
kk_metal_compute_key_create(const blake3_hash source, const char *entrypoint,
                            uint32_t max_total_threads)
{
   struct kk_metal_compute_key *key = calloc(1, sizeof(*key));
   if (key == NULL)
      return NULL;

   memcpy(key->source, source, sizeof(key->source));
   key->entrypoint = strdup(entrypoint);
   key->max_total_threads = max_total_threads;
   if (key->entrypoint == NULL) {
      kk_metal_compute_key_free(key);
      return NULL;
   }
   return key;
}

static uint32_t
kk_metal_compute_key_hash(const void *key)
{
   const struct kk_metal_compute_key *k = key;
   uint32_t hash = _mesa_hash_data(k->source, sizeof(k->source));
   hash = _mesa_hash_data_with_seed(k->entrypoint, strlen(k->entrypoint), hash);
   return _mesa_hash_data_with_seed(&k->max_total_threads,
                                    sizeof(k->max_total_threads), hash);
}

static bool
kk_metal_compute_key_equal(const void *a, const void *b)
{
   const struct kk_metal_compute_key *x = a;
   const struct kk_metal_compute_key *y = b;
   return memcmp(x->source, y->source, sizeof(x->source)) == 0 &&
          x->max_total_threads == y->max_total_threads &&
          strcmp(x->entrypoint, y->entrypoint) == 0;
}

/* The tables are created with the device: shaders are compiled into them from
 * the pool, which starts right after. */
bool
kk_metal_caches_init(struct kk_device *dev)
{
   uint32_t max = KK_METAL_CACHE_MAX;
   const char *env = getenv("MESA_KK_METAL_CACHE_MAX");
   if (env != NULL)
      max = MAX2(1, atoi(env));

   if (!kk_metal_cache_init(&dev->msl_libraries, kk_msl_library_key_hash,
                            kk_msl_library_key_equal, free, max))
      return false;
   if (!kk_metal_cache_init(&dev->metal_render_pipelines,
                            kk_metal_render_key_hash,
                            kk_metal_render_key_equal,
                            kk_metal_render_key_free, max) ||
       !kk_metal_cache_init(&dev->metal_compute_pipelines,
                            kk_metal_compute_key_hash,
                            kk_metal_compute_key_equal,
                            kk_metal_compute_key_free, max)) {
      kk_metal_caches_finish(dev);
      return false;
   }
   return true;
}

void
kk_metal_caches_finish(struct kk_device *dev)
{
   kk_metal_cache_finish(&dev->metal_render_pipelines);
   kk_metal_cache_finish(&dev->metal_compute_pipelines);
   kk_metal_cache_finish(&dev->msl_libraries);
}

/* HB_KK_MATH=safe|relaxed|fast (default fast) and HB_KK_MATH_FUNCTIONS=precise
 * pick the Metal math mode of graphics and compute libraries. */
static enum mtl_math_mode
kk_hb_math_mode(void)
{
   static int mode = -1;
   if (mode < 0) {
      const char *env = getenv("HB_KK_MATH");
      mode = env && !strcmp(env, "safe")      ? MTL_MATH_MODE_SAFE
             : env && !strcmp(env, "relaxed") ? MTL_MATH_MODE_RELAXED
                                              : MTL_MATH_MODE_FAST;
      if (env)
         mesa_logi("kk: math mode %s", env);
   }
   return mode;
}

static enum mtl_math_floating_point_functions
kk_hb_math_functions(void)
{
   const char *env = getenv("HB_KK_MATH_FUNCTIONS");
   return env && !strcmp(env, "precise")
             ? MTL_MATH_FLOATING_POINT_FUNCTIONS_PRECISE
             : MTL_MATH_FLOATING_POINT_FUNCTIONS_FAST;
}

/* HB_KK_MSL_OVERRIDE_DIR: a library whose MSL hashes (_mesa_hash_string, as
 * kk_msl_digest names programs) to H compiles <dir>/<H as %08x>.metal instead,
 * when that file exists. */
static char *
kk_hb_msl_override(const char *source)
{
   const char *dir = getenv("HB_KK_MSL_OVERRIDE_DIR");
   if (dir == NULL || dir[0] == '\0')
      return NULL;
   char path[1024];
   snprintf(path, sizeof(path), "%s/%08x.metal", dir, _mesa_hash_string(source));
   FILE *f = fopen(path, "rb");
   if (f == NULL)
      return NULL;
   fseek(f, 0, SEEK_END);
   long size = ftell(f);
   fseek(f, 0, SEEK_SET);
   char *text = size > 0 ? malloc(size + 1) : NULL;
   if (text != NULL && fread(text, 1, size, f) == (size_t)size) {
      text[size] = '\0';
      mesa_logi("kk: MSL override from %s", path);
   } else {
      free(text);
      text = NULL;
   }
   fclose(f);
   return text;
}

/* The compiled library of an MSL source, from the device's table or compiled
 * into it (kk_device::msl_libraries): a title asks for the same program once
 * per pipeline, and while Metal's own cache answers each request in fractions
 * of a millisecond, every request still pays the guest round trip to reach it.
 * Returns a reference the caller releases, like mtl_new_library does, and the
 * source's digest, which names the library in the pipeline keys. */
static mtl_library *
kk_get_msl_library(struct kk_device *device, const char *source,
                   blake3_hash source_digest)
{
   _mesa_blake3_compute(source, strlen(source), source_digest);

   struct kk_msl_library_key *key = malloc(sizeof(*key));
   if (key == NULL)
      return NULL;
   memcpy(key->source, source_digest, sizeof(key->source));

   mtl_library *library = kk_metal_cache_find(&device->msl_libraries, key);
   if (library != NULL)
      return library;

   struct kk_physical_device *pdev = kk_device_physical(device);
   char *override = kk_hb_msl_override(source);
   library = mtl_new_library(device->mtl_compiler_handle,
                             override ? override : source,
                             pdev->info.msl_version, kk_hb_math_mode(),
                             kk_hb_math_functions());
   free(override);
   /* Takes the key, which this caller no longer owns */
   kk_metal_cache_publish(&device->msl_libraries, key, library);
   return library;
}

static VkResult
kk_compile_compute_pipeline(struct kk_device *device, mtl_compiler *compiler,
                            const struct msl_compile_data *data,
                            uint32_t local_size_threads,
                            mtl_compute_pipeline_state **pipe)
{
   blake3_hash source;
   mtl_library *library = kk_get_msl_library(device, data->code, source);
   if (library == NULL)
      return VK_ERROR_INVALID_SHADER_NV;

   struct kk_metal_compute_key *key = kk_metal_compute_key_create(
      source, data->entrypoint_name, local_size_threads);
   if (key == NULL) {
      mtl_release(library);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }

   /* A compute pipeline with the same kernel and threadgroup size is the same
    * Metal pipeline (kk_metal_caches_init) */
   mtl_compute_pipeline_state *pipeline =
      kk_metal_cache_find(&device->metal_compute_pipelines, key);
   if (pipeline == NULL) {
      mtl_function_descriptor *function =
         mtl_new_library_function_descriptor(library, data->entrypoint_name);
      char hb_label[64];
      snprintf(hb_label, sizeof(hb_label), "kk cs %02x%02x%02x%02x%02x%02x%02x%02x",
               source[0], source[1], source[2], source[3], source[4], source[5], source[6], source[7]);
      mtl_set_next_pipeline_label(hb_label);
      pipeline = mtl_new_compute_pipeline_state(compiler, function,
                                                local_size_threads);
      mtl_release(function);
      /* Takes the key, which this caller no longer owns */
      kk_metal_cache_publish(&device->metal_compute_pipelines, key, pipeline);
      if (pipeline == NULL) {
         mtl_release(library);
         return VK_ERROR_INVALID_SHADER_NV;
      }
   }
   mtl_release(library);

   *pipe = pipeline;

   return VK_SUCCESS;
}

static VkResult
kk_compile_rt_kernel(struct kk_device *device, mtl_compiler *compiler,
                     struct kk_shader *s, uint32_t local_size_threads)
{
   const struct msl_compile_data *data = &s->msl_data[MESA_SHADER_COMPUTE];
   const struct kk_rt_link *link = s->rt_link;
   blake3_hash source;
   mtl_library *library = kk_get_msl_library(device, data->code, source);
   if (library == NULL)
      return VK_ERROR_INVALID_SHADER_NV;

   mtl_function_descriptor *function =
      mtl_new_library_function_descriptor(library, data->entrypoint_name);
   int64_t t0 = os_time_get_nano();
   mtl_compute_pipeline_state *pipeline = mtl_new_compute_pipeline_state_linked(
      compiler, function, local_size_threads, link->functions, link->count);
   uint32_t max_threads = 0, width = 0;
   if (pipeline)
      mtl_compute_pipeline_stats(pipeline, &max_threads, &width);
   if (pipeline && debug_get_bool_option("HB_KK_RT_PROBE_THREADS", false)) {
      mtl_compute_pipeline_state *probe = mtl_new_compute_pipeline_state_linked(
         compiler, function, 1024, link->functions, link->count);
      if (probe) {
         mtl_compute_pipeline_stats(probe, &max_threads, &width);
         mtl_release(probe);
      }
   }
   fprintf(stderr,
           "kk rt: linked kernel with %u functions in %.1f ms%s, max threads "
           "%u, simd %u, msl %zu bytes\n",
           link->count, (os_time_get_nano() - t0) / 1e6,
           pipeline ? "" : " (failed)", max_threads, width, strlen(data->code));
   mtl_release(function);
   mtl_release(library);
   if (pipeline == NULL)
      return VK_ERROR_INVALID_SHADER_NV;

   mtl_visible_function_table *table =
      mtl_new_visible_function_table(pipeline, link->table_size);
   if (table == NULL) {
      mtl_release(pipeline);
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   }
   for (uint32_t i = 0; i < link->count; i++) {
      if (!mtl_visible_function_table_set(table, pipeline, link->functions[i],
                                          link->indices[i])) {
         mtl_release(table);
         mtl_release(pipeline);
         return VK_ERROR_INVALID_SHADER_NV;
      }
   }
   kk_device_add_allocation_to_residency_set(device, table);
   s->pipeline.cs = pipeline;
   s->pipeline.rt_table = table;
   s->pipeline.rt_table_id = mtl_visible_function_table_resource_id(table);
   return VK_SUCCESS;
}

static mtl_depth_stencil_state *
kk_compile_ds_state(struct kk_device *device, struct kk_shader_info *info)
{
   mtl_stencil_descriptor *front = NULL;
   mtl_stencil_descriptor *back = NULL;
   mtl_depth_stencil_descriptor *descriptor =
      mtl_new_depth_stencil_descriptor();
   mtl_depth_stencil_descriptor_set_depth_write_enabled(
      descriptor, info->vs.has_depth_write);
   mtl_depth_stencil_descriptor_set_depth_compare_function(
      descriptor, info->vs.depth_compare_op);

   if (info->vs.has_stencil_test) {
      back = mtl_new_stencil_descriptor();
      mtl_stencil_descriptor_set_depth_failure_operation(
         back, info->vs.stencil_back.depth_fail);
      mtl_stencil_descriptor_set_stencil_failure_operation(
         back, info->vs.stencil_back.fail);
      mtl_stencil_descriptor_set_depth_stencil_pass_operation(
         back, info->vs.stencil_back.pass);
      mtl_stencil_descriptor_set_stencil_compare_function(
         back, info->vs.stencil_back.compare);
      mtl_stencil_descriptor_set_read_mask(back,
                                           info->vs.stencil_back.compare_mask);
      mtl_stencil_descriptor_set_write_mask(back,
                                            info->vs.stencil_back.write_mask);
      mtl_depth_stencil_descriptor_set_back_face_stencil(descriptor, back);

      front = mtl_new_stencil_descriptor();
      mtl_stencil_descriptor_set_depth_failure_operation(
         front, info->vs.stencil_front.depth_fail);
      mtl_stencil_descriptor_set_stencil_failure_operation(
         front, info->vs.stencil_front.fail);
      mtl_stencil_descriptor_set_depth_stencil_pass_operation(
         front, info->vs.stencil_front.pass);
      mtl_stencil_descriptor_set_stencil_compare_function(
         front, info->vs.stencil_front.compare);
      mtl_stencil_descriptor_set_read_mask(front,
                                           info->vs.stencil_front.compare_mask);
      mtl_stencil_descriptor_set_write_mask(front,
                                            info->vs.stencil_front.write_mask);
      mtl_depth_stencil_descriptor_set_front_face_stencil(descriptor, front);
   }

   mtl_depth_stencil_state *state =
      mtl_new_depth_stencil_state(device->mtl_handle, descriptor);

   if (front)
      mtl_release(front);
   if (back)
      mtl_release(back);
   mtl_release(descriptor);

   return state;
}

/* The depth/stencil state is independent of the selected program. Build it
 * once on the plain VS at creation (kk_compile_shaders, kk_deserialize_shader),
 * where binding reads it, even when only the minmax program is queued.
 * Binding must not wait for Metal compilation under virglrenderer's shared
 * submission lock, and pool threads must not write state that binding reads. */
static void
kk_create_ds_state(struct kk_device *dev, struct kk_shader *vs)
{
   if (vs->info.stage == MESA_SHADER_VERTEX && vs->info.vs.has_ds)
      vs->pipeline.gfx.ds_handle = kk_compile_ds_state(dev, &vs->info);
}

static void
kk_gather_ds_info(const struct vk_depth_stencil_state *ds, bool has_depth,
                  bool has_stencil, struct kk_shader_info *info)
{
   /* Depth */
   if (has_depth && ds->depth.test_enable) {
      info->vs.has_depth_write = ds->depth.write_enable;
      info->vs.depth_compare_op = ds->depth.compare_op;
   } else {
      info->vs.has_depth_write = false;
      info->vs.depth_compare_op = VK_COMPARE_OP_ALWAYS;
   }

   /* Stencil */
   info->vs.has_stencil_test = has_stencil && ds->stencil.test_enable;
   if (info->vs.has_stencil_test) {
      info->vs.stencil_back.depth_fail = ds->stencil.back.op.depth_fail;
      info->vs.stencil_back.fail = ds->stencil.back.op.fail;
      info->vs.stencil_back.pass = ds->stencil.back.op.pass;
      info->vs.stencil_back.compare = ds->stencil.back.op.compare;
      info->vs.stencil_back.compare_mask = ds->stencil.back.compare_mask;
      info->vs.stencil_back.write_mask = ds->stencil.back.write_mask;

      info->vs.stencil_front.depth_fail = ds->stencil.front.op.depth_fail;
      info->vs.stencil_front.fail = ds->stencil.front.op.fail;
      info->vs.stencil_front.pass = ds->stencil.front.op.pass;
      info->vs.stencil_front.compare = ds->stencil.front.op.compare;
      info->vs.stencil_front.compare_mask = ds->stencil.front.compare_mask;
      info->vs.stencil_front.write_mask = ds->stencil.front.write_mask;
   }
}

mtl_depth_stencil_state *
kk_compile_depth_stencil_state(struct kk_device *device,
                               const struct vk_depth_stencil_state *ds,
                               bool has_depth, bool has_stencil)
{
   struct kk_shader_info info = {};
   kk_gather_ds_info(ds, has_depth, has_stencil, &info);
   return kk_compile_ds_state(device, &info);
}

static void
kk_msl_copy(struct msl_compile_data *dst, const struct msl_compile_data *src)
{
   kk_msl_source_copy(dst, src);
   dst->entrypoint_name = ralloc_strdup(NULL, src->entrypoint_name);
}

/* Copies all msl shader data to the vertex and gathers information for pipeline
 * compilation. */
static void
gather_graphics_pipeline_create_info(
   const struct vk_graphics_pipeline_state *state, struct kk_shader **shaders,
   uint32_t shader_count)
{
   assert(state && shader_count >= 2 &&
          "state, vertex and fragment shaders are a must");
   struct kk_shader *vs = shaders[0];
   struct kk_shader_info *info = &vs->info;

   /* Render pass data */
   const struct vk_render_pass_state *rp = state->rp;
   bool has_depth = rp->depth_attachment_format != VK_FORMAT_UNDEFINED;
   bool has_stencil = rp->stencil_attachment_format != VK_FORMAT_UNDEFINED;
   {
      info->vs.color_attachment_count = rp->color_attachment_count;
      for (uint8_t i = 0u; i < MAX_DRAW_BUFFERS; ++i) {
         info->vs.rt_formats[i] = MTL_PIXEL_FORMAT_INVALID;
      }
      for (uint8_t i = 0u; i < state->rp->color_attachment_count; ++i) {
         uint8_t logical_index = kk_get_logical_color_att_index(state, i);
         if (logical_index == MESA_VK_ATTACHMENT_UNUSED) {
            continue;
         }
         VkFormat format = state->rp->color_attachment_formats[i];
         info->vs.rt_formats[logical_index] =
            format == VK_FORMAT_UNDEFINED
               ? MTL_PIXEL_FORMAT_INVALID
               : vk_format_to_mtl_pixel_format(format);
      }
      info->vs.d_format =
         has_depth ? vk_format_to_mtl_pixel_format(rp->depth_attachment_format)
                   : MTL_PIXEL_FORMAT_INVALID;
      info->vs.s_format =
         has_stencil
            ? vk_format_to_mtl_pixel_format(rp->stencil_attachment_format)
            : MTL_PIXEL_FORMAT_INVALID;

      info->vs.view_mask = state->mv->view_mask;
   }

   info->vs.has_ds = has_static_depth_stencil_state(state);
   if (info->vs.has_ds)
      kk_gather_ds_info(state->ds, has_depth, has_stencil, info);

   info->vs.has_ms = state->ms != NULL;
   info->vs.sample_count = 1u;
   if (info->vs.has_ms) {
      const struct vk_multisample_state *ms = state->ms;
      info->vs.sample_count = ms->rasterization_samples;
   }

   /* We need to store all other stage sources in the vertex too otherwise we
    * won't be able to create the whole pipeline correctly. */
   struct kk_shader *gs = NULL;
   for (uint32_t i = 1; i < shader_count; ++i) {
      struct kk_shader *s = shaders[i];
      mesa_shader_stage stage = s->info.stage;
      kk_msl_copy(&vs->msl_data[stage], &s->msl_data[stage]);

      info->vs.additional_stages_bits |= BITFIELD_BIT(stage);

      if (s->info.stage == MESA_SHADER_TESS_CTRL)
         info->vs.tess_local_thread_size = s->info.tess.tcs_output_patch_size;

      if (s->info.stage == MESA_SHADER_GEOMETRY) {
         kk_msl_copy(&vs->gs_main, &s->gs_main);
         kk_msl_copy(&vs->gs_count, &s->gs_count);
         kk_msl_copy(&vs->pre_gs, &s->pre_gs);
         info->vs.gs = s->info.gs;
         gs = s;
      }
   }

   uint8_t topology = state->ia->primitive_topology;
   if (info->vs.additional_stages_bits & BITFIELD_BIT(MESA_SHADER_TESS_CTRL)) {
      /* TODO_KOSMICKRISP Hackity hack */
      struct kk_shader *tesc = shaders[1];
      struct kk_shader *tese = shaders[2];

      struct kk_tess_info info =
         kk_tess_info_merge(tese->info.tess.info, tesc->info.tess.info);

      /* Determine primitive based on the merged state */
      if (info.points) {
         topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
      } else if (info.mode == TESS_PRIMITIVE_ISOLINES) {
         topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
      } else {
         topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
      }
   }

   if (gs) {
      /* Only the topology class matters here */
      enum mesa_prim prim = u_reduced_prim(gs->info.gs.mode);
      if (prim == MESA_PRIM_POINTS)
         topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
      else if (prim == MESA_PRIM_LINES)
         topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
      else
         topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
   }

   info->vs.topology =
      vk_primitive_topology_to_mtl_primitive_topology_class(topology);
}

static VkResult
kk_compile_graphics_pipeline(struct kk_device *device, mtl_compiler *compiler,
                             struct kk_shader *vs)
{
   struct kk_physical_device *pdev = kk_device_physical(device);

   VkResult result = VK_SUCCESS;
   struct kk_pipeline_handles *pipe = &vs->pipeline;
   mesa_shader_stage vs_stage = MESA_SHADER_VERTEX;

   /* When tessellation or geometry shaders are present, the pair that will be
    * used for the Metal render pipeline will be the fragment shader and the
    * previous stage. This means that any shader before that needs to be a
    * compute pipeline to be launched before the render pipeline. */
   uint32_t stages_bits = vs->info.vs.additional_stages_bits;
   pipe->gfx.pre_render_count = util_bitcount(stages_bits) - 1u;
   for (uint32_t i = 0u; i < pipe->gfx.pre_render_count; ++i) {
      uint32_t local_thread_size = vs_stage == MESA_SHADER_TESS_CTRL
                                      ? vs->info.vs.tess_local_thread_size
                                      : KK_POLY_WG_SIZE;
      result = kk_compile_compute_pipeline(device, compiler,
                                           &vs->msl_data[vs_stage],
                                           local_thread_size,
                                           &pipe->gfx.pre_render[i]);

      if (result != VK_SUCCESS)
         return VK_ERROR_INVALID_SHADER_NV;
      vs_stage = u_bit_scan(&stages_bits);
   }

   /* The GS compute pass is the last pre-render pipeline, its rasterization VS
    * pairs with the FS. */
   if (vs->info.vs.additional_stages_bits &
       BITFIELD_BIT(MESA_SHADER_GEOMETRY)) {
      result = kk_compile_compute_pipeline(
         device, compiler, &vs->gs_main, KK_POLY_WG_SIZE,
         &pipe->gfx.pre_render[pipe->gfx.pre_render_count++]);
      if (result != VK_SUCCESS)
         return VK_ERROR_INVALID_SHADER_NV;

      if (vs->gs_count.code) {
         result = kk_compile_compute_pipeline(
            device, compiler, &vs->gs_count, KK_POLY_WG_SIZE,
            &pipe->gfx.gs_count);
         if (result != VK_SUCCESS)
            return VK_ERROR_INVALID_SHADER_NV;
      }

      if (vs->pre_gs.code) {
         result = kk_compile_compute_pipeline(device, compiler, &vs->pre_gs,
                                              1u, &pipe->gfx.pre_gs);
         if (result != VK_SUCCESS)
            return VK_ERROR_INVALID_SHADER_NV;
      }
   }

   const struct msl_compile_data *vs_data = &vs->msl_data[vs_stage];
   blake3_hash vs_source, fs_source;
   mtl_library *vertex_library =
      kk_get_msl_library(device, vs_data->code, vs_source);
   if (vertex_library == NULL)
      return VK_ERROR_INVALID_SHADER_NV;

   const struct msl_compile_data *fs_data = &vs->msl_data[MESA_SHADER_FRAGMENT];
   mtl_library *fragment_library =
      kk_get_msl_library(device, fs_data->code, fs_source);
   if (fragment_library == NULL) {
      mtl_release(vertex_library);
      return VK_ERROR_INVALID_SHADER_NV;
   }

   /* A render pipeline built from the same two programs and state is the same
    * Metal pipeline (kk_metal_caches_init) */
   struct kk_metal_render_key *key = kk_metal_render_key_create(
      vs, vs_data, fs_data, vs_source, fs_source);
   if (key == NULL) {
      mtl_release(fragment_library);
      mtl_release(vertex_library);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }

   mtl_render_pipeline_state *render =
      kk_metal_cache_find(&device->metal_render_pipelines, key);
   if (render == NULL) {
      mtl_function_descriptor *vertex_function =
         mtl_new_library_function_descriptor(vertex_library,
                                             vs_data->entrypoint_name);
      mtl_function_descriptor *fragment_function =
         mtl_new_library_function_descriptor(fragment_library,
                                             fs_data->entrypoint_name);

      mtl_render_pipeline_descriptor *pipeline_descriptor =
         mtl_new_render_pipeline_descriptor();
      mtl_render_pipeline_descriptor_set_vertex_shader(pipeline_descriptor,
                                                       vertex_function);
      mtl_render_pipeline_descriptor_set_fragment_shader(pipeline_descriptor,
                                                         fragment_function);

      /* Layered rendering in Metal requires setting primitive topology class */
      mtl_render_pipeline_descriptor_set_input_primitive_topology(
         pipeline_descriptor, vs->info.vs.topology);

      /* If color attachments are reordered there might be unused / invalid gaps
       * in the array so need to check them all, not just first
       * color_attachment_count entries.
       */
      for (uint8_t i = 0; i < MAX_DRAW_BUFFERS; ++i) {
         if (vs->info.vs.rt_formats[i] != MTL_PIXEL_FORMAT_INVALID)
            mtl_render_pipeline_descriptor_set_color_attachment_format(
               pipeline_descriptor, i, vs->info.vs.rt_formats[i]);
      }

      if (vs->info.vs.view_mask) {
         uint32_t max_amplification = util_bitcount(vs->info.vs.view_mask);
         mtl_render_pipeline_descriptor_set_max_vertex_amplification_count(
            pipeline_descriptor, max_amplification);
      }

      if (vs->info.vs.has_ms) {
         mtl_render_pipeline_descriptor_set_raster_sample_count(
            pipeline_descriptor, vs->info.vs.sample_count);
      }

      char hb_label[80];
      snprintf(hb_label, sizeof(hb_label),
               "kk vs %02x%02x%02x%02x%02x%02x%02x%02x fs %02x%02x%02x%02x%02x%02x%02x%02x",
               vs_source[0], vs_source[1], vs_source[2], vs_source[3], vs_source[4], vs_source[5], vs_source[6], vs_source[7],
               fs_source[0], fs_source[1], fs_source[2], fs_source[3], fs_source[4], fs_source[5], fs_source[6], fs_source[7]);
      mtl_set_next_pipeline_label(hb_label);
      render = mtl_new_render_pipeline(compiler, pipeline_descriptor);

      mtl_release(pipeline_descriptor);
      mtl_release(fragment_function);
      mtl_release(vertex_function);

      /* Takes the key, which this caller no longer owns */
      kk_metal_cache_publish(&device->metal_render_pipelines, key, render);
      if (render == NULL) {
         mtl_release(fragment_library);
         mtl_release(vertex_library);
         return VK_ERROR_INVALID_SHADER_NV;
      }
   }

   mtl_release(fragment_library);
   mtl_release(vertex_library);
   pipe->gfx.render = render;

   return VK_SUCCESS;
}

static bool
move_primitive_id(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const unsigned *slot = data;

   if (intr->intrinsic != nir_intrinsic_load_input &&
       intr->intrinsic != nir_intrinsic_store_output)
      return false;

   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   if (sem.location != VARYING_SLOT_PRIMITIVE_ID)
      return false;

   sem.location = *slot;
   nir_intrinsic_set_io_semantics(intr, sem);

   /* Metal requires the vertex output and fragment input types to match */
   if (intr->intrinsic == nir_intrinsic_store_output)
      nir_intrinsic_set_src_type(intr, nir_type_uint32);
   else
      nir_intrinsic_set_dest_type(intr, nir_type_uint32);
   return true;
}

/* Metal's primitive_id attribute numbers the primitives the GS rasterization
 * pass draws, not the value the GS writes. Pass that value as a flat varying
 * in a slot neither shader uses instead. */
static void
kk_forward_gs_primitive_id(nir_shader *gs, nir_shader *fs)
{
   uint64_t used = gs->info.outputs_written | fs->info.inputs_read;
   unsigned slot = VARYING_SLOT_VAR31;
   while (slot >= VARYING_SLOT_VAR0 && (used & BITFIELD64_BIT(slot)))
      --slot;

   if (!(gs->info.outputs_written & VARYING_BIT_PRIMITIVE_ID) ||
       slot < VARYING_SLOT_VAR0)
      return;

   /* Makes progress only if the FS reads the primitive ID */
   const struct nir_lower_sysvals_to_varyings_options sysvals = {
      .primitive_id = true,
   };
   bool progress = false;
   NIR_PASS(progress, fs, nir_lower_sysvals_to_varyings, &sysvals);
   if (!progress)
      return;

   nir_shader_intrinsics_pass(gs, move_primitive_id, nir_metadata_all, &slot);
   nir_shader_intrinsics_pass(fs, move_primitive_id, nir_metadata_all, &slot);
   nir_shader_gather_info(gs, nir_shader_get_entrypoint(gs));
   nir_shader_gather_info(fs, nir_shader_get_entrypoint(fs));
}

static bool
load_patch_id(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_primitive_id)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *patch_id = nir_load_per_vertex_input(
      b, 1, 32, nir_imm_int(b, 0), nir_imm_int(b, 0),
      .dest_type = nir_type_uint32,
      .io_semantics.location = VARYING_SLOT_PRIMITIVE_ID,
      .io_semantics.num_slots = 1);
   nir_def_replace(&intr->def, patch_id);
   return true;
}

/* After tessellation, the GS primitive ID is the patch index the tessellation
 * shaders read, but poly numbers the tessellated primitives. Pass the TES
 * primitive ID to the GS as an input. */
static void
kk_forward_tes_primitive_id(nir_shader *tes, nir_shader *gs)
{
   if (!nir_shader_intrinsics_pass(gs, load_patch_id, nir_metadata_control_flow,
                                   NULL))
      return;

   nir_function_impl *impl = nir_shader_get_entrypoint(tes);
   nir_builder b = nir_builder_at(nir_after_impl(impl));
   nir_store_output(&b, nir_load_primitive_id(&b), nir_imm_int(&b, 0),
                    .src_type = nir_type_uint32,
                    .io_semantics.location = VARYING_SLOT_PRIMITIVE_ID,
                    .io_semantics.num_slots = 1);
   nir_progress(true, impl, nir_metadata_control_flow);

   nir_shader_gather_info(tes, impl);
   nir_shader_gather_info(gs, nir_shader_get_entrypoint(gs));
}

static bool
gather_output_type(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   nir_alu_type *types = data;
   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;

   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   for (unsigned i = 0; i < sem.num_slots; ++i)
      types[sem.location + i] = nir_intrinsic_src_type(intr);
   return false;
}

/* Add a passthrough GS so poly can capture the last stage's outputs.
 * prim is the point, line or triangle list it takes as input. */
static nir_shader *
kk_xfb_passthrough_gs(const nir_shader *last, enum mesa_prim prim)
{
   uint16_t xfb_outputs = last->xfb_info->output_count;
   struct poly_passthrough_gs_key *key =
      calloc(1, poly_passthrough_gs_key_size(xfb_outputs));

   key->prim = prim;
   key->outputs = last->info.outputs_written;
   key->clip_distance_array_size = last->info.clip_distance_array_size;
   key->cull_distance_array_size = last->info.cull_distance_array_size;

   for (uint32_t i = 0; i < NUM_TOTAL_VARYING_SLOTS; i++) {
      unsigned num_comps = 0;
      switch (i) {
      case VARYING_SLOT_TESS_LEVEL_OUTER:
      case VARYING_SLOT_TESS_LEVEL_INNER:
      case VARYING_SLOT_BOUNDING_BOX1:
         num_comps = 4;
         break;
      default:
         num_comps = nir_slot_num_components(i, MESA_SHADER_VERTEX);
         break;
      }
      key->output_components[i] = num_comps ?: 4;
      key->output_types[i] = nir_type_float32;
   }

   /* Keep the stored types for the Metal vertex/fragment interface. */
   nir_shader_intrinsics_pass((nir_shader *)last, gather_output_type,
                              nir_metadata_all, key->output_types);

   memcpy(key->xfb_stride, last->info.xfb_stride, sizeof(key->xfb_stride));
   memcpy(&key->xfb_info, last->xfb_info, nir_xfb_info_size(xfb_outputs));

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_GEOMETRY, last->options, "xfb passthrough");
   poly_nir_passthrough_gs(&b, key);
   free(key);

   nir_shader *gs = b.shader;
   gs->info.io_lowered = true;
   gs->info.next_stage = MESA_SHADER_FRAGMENT;
   nir_shader_gather_info(gs, b.impl);
   return gs;
}

/* Lowers, links and translates to MSL one program of a pipeline: shaders
 * (with room for shader_count + 2, generated[] marks the stages KK added) get
 * the stages, the VS or CS first. Takes ownership of nirs. minmax_sets gets
 * the descriptor sets of samplers whose min/max emulation the program leaves
 * to KK_FEAT_SAMPLER_MINMAX_DYNAMIC. */
static VkResult
kk_build_shaders(struct kk_device *dev, uint32_t shader_count,
                 const struct vk_shader_compile_info *infos, nir_shader **nirs,
                 const struct vk_graphics_pipeline_state *state,
                 enum kk_feature_key features,
                 const VkAllocationCallbacks *pAllocator,
                 struct kk_shader **shaders, bool *generated,
                 uint32_t *total_out, uint32_t *minmax_sets,
                 bool *reported_missing_descriptor)
{
   VkResult result = VK_SUCCESS;
   struct kk_physical_device *pdev = kk_device_physical(dev);

   /* Leave room for a generated fragment shader and an XFB passthrough GS. */
   nir_shader *nir_shaders[shader_count + 2u];

   /* Capture the last pre-rasterization stage, adding a passthrough GS if
    * the application doesn't have one. */
   uint32_t last_vtg = 0u;
   for (uint32_t i = 0u; i < shader_count; ++i) {
      if (infos[i].stage == MESA_SHADER_VERTEX ||
          infos[i].stage == MESA_SHADER_TESS_EVAL ||
          infos[i].stage == MESA_SHADER_GEOMETRY)
         last_vtg = i;
   }
   bool xfb_gs = state && nirs[last_vtg]->xfb_info &&
                 infos[last_vtg].stage != MESA_SHADER_GEOMETRY;

   /* Lower shaders, notably lowering IO. This is a prerequisite for intershader
    * optimization. */
   const struct vk_pipeline_robustness_state *vertex_robustness = &rs_none;
   uint32_t total_shaders = 0u;
   for (uint32_t i = 0u; i < shader_count; ++i) {
      const struct vk_shader_compile_info *info = &infos[i];
      nir_shader *nir = nirs[i];

      /* Stages before the tessellator or a GS run as compute */
      bool emulated_stage =
         (info->next_stage_mask & (VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
                                   VK_SHADER_STAGE_GEOMETRY_BIT)) ||
         (xfb_gs && i == last_vtg);

      msl_preprocess_nir_workarounds(nir, pdev->settings.disabled_workarounds);
      const struct vk_pipeline_robustness_state *robustness = info->robustness;
      if (kk_nir_lower_missing_descriptors(
             nir, info->set_layout_count, info->set_layouts,
             reported_missing_descriptor))
         robustness = &rs_all_supported;
      kk_lower_nir(dev, nir, emulated_stage, robustness,
                   info->set_layout_count, info->set_layouts, state, features,
                   minmax_sets);

      if (nir->info.stage == MESA_SHADER_VERTEX)
         vertex_robustness = info->robustness;

      /* Don't let linking remove outputs captured by XFB. */
      if (nir->info.stage == MESA_SHADER_GEOMETRY && nir->xfb_info) {
         NIR_PASS(_, nir, nir_opt_constant_folding);
         NIR_PASS(_, nir, nir_io_add_intrinsic_xfb_info);
      }

      generated[total_shaders] = false;
      nir_shaders[total_shaders++] = nir;

      if (xfb_gs && i == last_vtg) {
         enum mesa_prim prim;
         if (nir->info.stage == MESA_SHADER_TESS_EVAL) {
            /* The TCS may supply the primitive mode or point mode. */
            const nir_shader *tcs = nir_shaders[total_shaders - 2u];
            enum tess_primitive_mode mode =
               nir->info.tess._primitive_mode ?: tcs->info.tess._primitive_mode;
            if (nir->info.tess.point_mode || tcs->info.tess.point_mode)
               prim = MESA_PRIM_POINTS;
            else if (mode == TESS_PRIMITIVE_ISOLINES)
               prim = MESA_PRIM_LINES;
            else
               prim = MESA_PRIM_TRIANGLES;
         } else {
            /* Adjacency is unrolled first, so this works for any topology
             * in the same class. */
            prim = u_reduced_prim(
               vk_topology_to_mesa(state->ia->primitive_topology));
         }

         nir_shader *gs = kk_xfb_passthrough_gs(nir, prim);
         NIR_PASS(_, gs, nir_opt_constant_folding);
         NIR_PASS(_, gs, nir_io_add_intrinsic_xfb_info);
         nir->info.next_stage = MESA_SHADER_GEOMETRY;

         generated[total_shaders] = true;
         nir_shaders[total_shaders++] = gs;
      }
   }

   /* Since we don't support GPL nor shader objects and Metal render pipelines
    * require both vertex and fragment, we may need to provide a pass-through
    * fragment. */
   if (state && infos[shader_count - 1u].stage != MESA_SHADER_FRAGMENT) {
      generated[total_shaders] = true;
      nir_shaders[total_shaders++] =
         get_empty_nir(dev, MESA_SHADER_FRAGMENT, state, features);
   }

   /* nir_opt_varyings_bulk removes a GS primitive ID output that the FS does
    * not read as an input. The FS is last and a GS right before it. */
   if (total_shaders >= 2u &&
       nir_shaders[total_shaders - 2u]->info.stage == MESA_SHADER_GEOMETRY) {
      kk_forward_gs_primitive_id(nir_shaders[total_shaders - 2u],
                                 nir_shaders[total_shaders - 1u]);
   }

   for (unsigned pass = 0; pass < 2; pass++) {
      for (uint32_t i = 0; i < total_shaders; i++) {
         if (!kk_nir_check_varyings_input(nir_shaders[i], pass)) {
            for (uint32_t j = 0; j < total_shaders; j++)
               ralloc_free(nir_shaders[j]);
            return vk_error(dev, VK_ERROR_INVALID_SHADER_NV);
         }
      }
      /* A second pass matches vec3/vec4 interfaces after matrix lowering. */
      nir_opt_varyings_bulk(nir_shaders, total_shaders, true, UINT32_MAX,
                            UINT32_MAX, nir_opts, NULL);
   }

   /* After linking, so nir_opt_varyings leaves the new slot alone */
   if (total_shaders >= 3u &&
       nir_shaders[total_shaders - 3u]->info.stage == MESA_SHADER_TESS_EVAL &&
       nir_shaders[total_shaders - 2u]->info.stage == MESA_SHADER_GEOMETRY) {
      kk_forward_tes_primitive_id(nir_shaders[total_shaders - 3u],
                                  nir_shaders[total_shaders - 2u]);
   }

   for (uint32_t i = 0; i < total_shaders; i++) {
      struct kk_shader *prev_stage = i > 0 ? shaders[i - 1] : NULL;
      result =
         kk_compile_shader(dev, nir_shaders[i], prev_stage, vertex_robustness,
                           state, pAllocator, &shaders[i]);

      if (result != VK_SUCCESS) {
         /* Clean up all the shaders before this point */
         for (uint32_t j = 0; j < i; j++)
            kk_shader_destroy(&dev->vk, shaders[j], pAllocator);

         /* Clean up all the NIR after this point */
         for (uint32_t j = i + 1; j < total_shaders; j++)
            ralloc_free(nir_shaders[j]);

         return result;
      }
   }

   *total_out = total_shaders;
   return VK_SUCCESS;
}

static void
kk_digest_msl(struct kk_cb_program *record,
              const struct msl_compile_data *data, int slot)
{
   if (data->code == NULL)
      return;
   uint32_t hash;
   uint8_t features;
   kk_msl_digest(data->code, &hash, &features);
   kk_msl_note_excerpt(hash, features, data->code);
   if (slot >= 0)
      record->hash[slot] = hash;
   record->features |= features;
}

/* STEAMAC-G: the MSL hashes and features command buffer records name a
 * program by: its kernel, or its Metal vertex function (the stage before the
 * FS) and FS, with the features of the compute passes before it. */
static void
kk_digest_program(struct kk_shader *s)
{
   struct kk_cb_program *record = &s->pipeline.record;
   *record = (struct kk_cb_program){
      .compute = s->info.stage == MESA_SHADER_COMPUTE,
   };
   if (record->compute) {
      kk_digest_msl(record, &s->msl_data[MESA_SHADER_COMPUTE], 0);
      return;
   }

   const uint32_t pre = s->info.vs.additional_stages_bits &
                        ~BITFIELD_BIT(MESA_SHADER_FRAGMENT);
   const mesa_shader_stage vs_stage =
      pre ? (mesa_shader_stage)(util_last_bit(pre) - 1u) : MESA_SHADER_VERTEX;
   record->pre_render = pre != 0u;
   kk_digest_msl(record, &s->msl_data[vs_stage], 0);
   kk_digest_msl(record, &s->msl_data[MESA_SHADER_FRAGMENT], 1);
   u_foreach_bit(stage, pre | BITFIELD_BIT(MESA_SHADER_VERTEX)) {
      if (stage != vs_stage)
         kk_digest_msl(record, &s->msl_data[stage], -1);
   }
   kk_digest_msl(record, &s->gs_main, -1);
   kk_digest_msl(record, &s->gs_count, -1);
   kk_digest_msl(record, &s->pre_gs, -1);
}

/* Creates the Metal pipelines of a program (its VS or CS). */
static VkResult
kk_compile_program(struct kk_device *dev, struct kk_shader *s,
                   mtl_compiler *compiler)
{
   VkResult result;
   if (s->info.stage == MESA_SHADER_COMPUTE) {
      uint32_t local_size_threads = s->info.cs.local_size.x *
                                    s->info.cs.local_size.y *
                                    s->info.cs.local_size.z;
      if (s->rt_link)
         result = kk_compile_rt_kernel(dev, compiler, s, local_size_threads);
      else
         result =
            kk_compile_compute_pipeline(dev, compiler,
                                        &s->msl_data[MESA_SHADER_COMPUTE],
                                        local_size_threads, &s->pipeline.cs);
   } else {
      result = kk_compile_graphics_pipeline(dev, compiler, s);
   }
   if (result == VK_SUCCESS) {
      kk_digest_program(s);
      p_atomic_set(&s->pipeline_ready, true);
   }
   return result;
}

/* --- background compilation of programs ------------------------------------
 * Compiling the Metal pipelines of a program is the expensive part of creating
 * a pipeline, and virglrenderer drains a guest's Venus context on one ring
 * thread: leaving the compiles to that thread serializes a title's whole shader
 * compilation (Dragon's Dogma 2: ~1,100 programs and ~130,000 pipelines, which
 * measured 62 minutes of first-run compilation). Programs are queued at
 * creation and compiled here instead, on as many threads as the machine has
 * cores. The first use of a program compiles it if it is still queued, or waits
 * for the worker compiling it; its destruction leaves it to that worker.
 * The pool shares the Metal compiler (safe to use concurrently) and the
 * device's tables of compiled Metal objects (kk_metal_caches_init). */
static int
kk_compile_worker(void *data)
{
   struct kk_device *dev = data;

   mtx_lock(&dev->compile_mtx);
   for (;;) {
      while (dev->compile_head == NULL && !dev->compile_shutdown)
         cnd_wait(&dev->compile_cnd, &dev->compile_mtx);
      if (dev->compile_head == NULL)
         break;

      struct kk_shader *s = dev->compile_head;
      dev->compile_head = s->compile_next;
      if (dev->compile_head == NULL)
         dev->compile_tail = NULL;
      s->compile_next = NULL;
      s->compile_running = true;
      /* Nothing will use a destroyed program once the device goes away */
      const bool skip = s->compile_orphaned && dev->compile_shutdown;
      mtx_unlock(&dev->compile_mtx);

      /* All workers share the device's compiler: Metal compiles concurrently
       * on one MTL4Compiler as well as on several. */
      VkResult result = skip ? VK_INCOMPLETE
                             : kk_compile_program(dev, s,
                                                  dev->mtl_compiler_handle);

      mtx_lock(&dev->compile_mtx);
      s->compile_result = result;
      s->compile_done = true;
      dev->compile_pending--;
      cnd_broadcast(&dev->compile_cnd);
      if (s->compile_orphaned) {
         /* Destroyed while it was queued or compiling (kk_shader_destroy) */
         mtx_unlock(&dev->compile_mtx);
         kk_shader_free(dev, s, &dev->vk.alloc);
         mtx_lock(&dev->compile_mtx);
      }
   }
   mtx_unlock(&dev->compile_mtx);
   return 0;
}

void
kk_compile_pool_init(struct kk_device *dev)
{
   mtx_init(&dev->compile_mtx, mtx_plain);
   cnd_init(&dev->compile_cnd);

   /* A worker per core, leaving room for the ring thread and the rest of the
    * system; every VkDevice has a pool of its own (gamescope, Steam and a game
    * make three in a session), so the default stays below the core count.
    * MESA_KK_COMPILE_THREADS overrides it (A/B, and machines where that many
    * Metal compiles at once is too much), capped at 64. */
   long cpus = sysconf(_SC_NPROCESSORS_ONLN);
   uint32_t threads = cpus > 3 ? MIN2(16u, (uint32_t)(cpus - 2)) : 2u;
   const char *env = getenv("MESA_KK_COMPILE_THREADS");
   if (env != NULL)
      threads = CLAMP((uint32_t)MAX2(1, atoi(env)), 1u, 64u);

   dev->compile_threads = vk_zalloc(&dev->vk.alloc, threads * sizeof(thrd_t),
                                    8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (dev->compile_threads == NULL)
      return;

   for (uint32_t i = 0; i < threads; i++) {
      if (thrd_create(&dev->compile_threads[i], kk_compile_worker, dev) !=
          thrd_success)
         break;
      dev->compile_thread_count++;
   }

   if (KK_DEBUG(COMPILES) && dev->compile_thread_count != 0)
      mesa_logi("kk: compiling programs on %u thread(s)",
                dev->compile_thread_count);
}

void
kk_compile_pool_finish(struct kk_device *dev)
{
   /* Workers drain the queue before exiting, so no compile is running for a
    * shader that is being destroyed. */
   mtx_lock(&dev->compile_mtx);
   dev->compile_shutdown = true;
   cnd_broadcast(&dev->compile_cnd);
   mtx_unlock(&dev->compile_mtx);

   for (uint32_t i = 0; i < dev->compile_thread_count; i++)
      thrd_join(dev->compile_threads[i], NULL);
   if (dev->compile_threads != NULL)
      vk_free(&dev->vk.alloc, dev->compile_threads);

   mtx_destroy(&dev->compile_mtx);
   cnd_destroy(&dev->compile_cnd);
}

void
kk_compile_enqueue(struct kk_device *dev, struct kk_shader *s)
{
   mtx_lock(&dev->compile_mtx);
   if (!s->compile_queued && !s->compile_running && !s->compile_done) {
      s->compile_queued = true;
      s->compile_next = NULL;
      if (dev->compile_tail != NULL)
         dev->compile_tail->compile_next = s;
      else
         dev->compile_head = s;
      dev->compile_tail = s;
      dev->compile_pending++;
      cnd_signal(&dev->compile_cnd);
   }
   mtx_unlock(&dev->compile_mtx);
}

VkResult
kk_compile_wait(struct kk_device *dev, struct kk_shader *s)
{
   if (p_atomic_read(&s->pipeline_ready))
      return VK_SUCCESS;

   mtx_lock(&dev->compile_mtx);
   if (s->compile_queued && !s->compile_running && !s->compile_done) {
      /* First use of a program that is still in the queue: a title's whole
       * pipeline set can be queued ahead of it, and the thread waiting here
       * (a bind or draw on the ring thread, under virglrenderer's submission
       * lock that every guest context shares) would wait for all of it -
       * Dragon's Dogma 2's desktop froze for minutes after its compile screen.
       * Take it out of the queue and compile it now. */
      struct kk_shader **link = &dev->compile_head, *prev = NULL;
      while (*link != s) {
         prev = *link;
         link = &(*link)->compile_next;
      }
      *link = s->compile_next;
      if (dev->compile_tail == s)
         dev->compile_tail = prev;
      s->compile_next = NULL;
      s->compile_running = true;
      mtx_unlock(&dev->compile_mtx);

      const VkResult result =
         kk_compile_program(dev, s, dev->mtl_compiler_handle);

      mtx_lock(&dev->compile_mtx);
      s->compile_result = result;
      s->compile_done = true;
      dev->compile_pending--;
      cnd_broadcast(&dev->compile_cnd);
      const bool orphaned = s->compile_orphaned;
      mtx_unlock(&dev->compile_mtx);
      /* Destroyed while this thread compiled it (kk_shader_destroy): the
       * thread that compiles a program frees it, as the worker does */
      if (orphaned)
         kk_shader_free(dev, s, &dev->vk.alloc);
      return result;
   }
   while (!s->compile_done && (s->compile_queued || s->compile_running))
      cnd_wait(&dev->compile_cnd, &dev->compile_mtx);
   if (s->compile_done) {
      const VkResult result = s->compile_result;
      mtx_unlock(&dev->compile_mtx);
      return result;
   }

   /* Never queued: claim it before unlocking so another first use waits and
    * destruction knows that this thread still owns the program's handles. */
   s->compile_running = true;
   mtx_unlock(&dev->compile_mtx);
   const VkResult result = kk_compile_program(dev, s, dev->mtl_compiler_handle);
   mtx_lock(&dev->compile_mtx);
   s->compile_result = result;
   s->compile_done = true;
   cnd_broadcast(&dev->compile_cnd);
   const bool orphaned = s->compile_orphaned;
   mtx_unlock(&dev->compile_mtx);
   if (orphaned)
      kk_shader_free(dev, s, &dev->vk.alloc);
   return result;
}

/* Graphics pipeline state the VS of a program holds. */
static void
kk_gather_program(struct kk_device *dev,
                  const struct vk_graphics_pipeline_state *state,
                  struct kk_shader **shaders, uint32_t total_shaders)
{
   gather_graphics_pipeline_create_info(state, shaders, total_shaders);
   struct kk_shader *vs = shaders[0];
   const uint32_t samples =
      kk_supported_sample_count(kk_device_physical(dev), vs->info.vs.sample_count);
   if (samples != vs->info.vs.sample_count) {
      mesa_logw("kk: rasterizationSamples %u is not supported, using %u",
                vs->info.vs.sample_count, samples);
      vs->info.vs.sample_count = samples;
   }
}

/* Before Apple10 a pipeline whose samplers come from descriptor sets has two
 * programs: the VS or CS one, which emulates min/max reduction only for
 * immutable samplers, and its minmax one, which also emulates it for the
 * samplers of info.minmax_sets (a branch, LOD preparation and a call per
 * sample: about twice the GPU time of sampling-bound shaders, and 1.3-1.6x
 * the compile time). Draws and dispatches use the minmax program only while
 * one of those sets holds a reduction sampler (kk_cmd_shader_pipeline).
 * Pipelines compile the other program when they are created, unless a set of
 * one of those sets' layouts already got a reduction sampler (descriptor
 * heaps: vkd3d-proton's sampler heap is then bound with every draw; for
 * pipelines read from a pipeline cache, any set); the program not compiled at
 * creation is compiled on first use. Both programs' MSL is built at creation,
 * but only the predicted program is queued for Metal compilation. */
static bool
kk_minmax_layouts_used(uint32_t minmax_sets, uint32_t set_layout_count,
                       struct vk_descriptor_set_layout *const *set_layouts)
{
   for (uint32_t s = 0; s < set_layout_count; s++) {
      if (s < 32 && !(minmax_sets & BITFIELD_BIT(s)))
         continue;
      if (set_layouts[s] != NULL &&
          p_atomic_read(
             &vk_to_kk_descriptor_set_layout(set_layouts[s])->reduction_samplers))
         return true;
   }
   return false;
}

static VkResult
kk_compile_shaders(struct vk_device *device, uint32_t shader_count,
                   struct vk_shader_compile_info *infos,
                   const struct vk_graphics_pipeline_state *state,
                   const struct vk_features *enabled_features,
                   const VkAllocationCallbacks *pAllocator,
                   struct vk_shader **shaders_out)
{
   struct kk_device *dev = container_of(device, struct kk_device, vk);
   struct kk_physical_device *pdev = kk_device_physical(dev);
   enum kk_feature_key features = kk_make_feature_key(enabled_features);
   const bool minmax = pdev->info.gpu_apple_family < 10 &&
                       (features & KK_FEAT_SAMPLER_MINMAX);

   nir_shader *nirs[shader_count], *minmax_nirs[shader_count];
   for (uint32_t i = 0u; i < shader_count; ++i) {
      nirs[i] = infos[i].nir;
      minmax_nirs[i] = minmax ? nir_shader_clone(NULL, infos[i].nir) : NULL;
   }

   struct kk_shader *shaders[shader_count + 2u];
   bool generated[shader_count + 2u];
   uint32_t total_shaders = 0u, minmax_sets = 0u;
   bool reported_missing_descriptor = false;
   VkResult result =
      kk_build_shaders(dev, shader_count, infos, nirs, state, features,
                       pAllocator, shaders, generated, &total_shaders,
                       &minmax_sets, &reported_missing_descriptor);
   if (result != VK_SUCCESS || !minmax_sets) {
      for (uint32_t i = 0u; i < shader_count; ++i)
         ralloc_free(minmax_nirs[i]);
   }
   if (result != VK_SUCCESS) {
      memset(shaders_out, 0, shader_count * sizeof(*shaders_out));
      return result;
   }

   /* Compile pipeline:
    * 1. Compute pipeline
    * 2. Graphics with all stages (since we don't support GPL nor shader
    * objects for now). This will be addressed later.
    */
   struct kk_shader *s = shaders[0];
   s->info.minmax_sets = minmax_sets;
   if (s->vk.stage != MESA_SHADER_COMPUTE)
      kk_gather_program(dev, state, shaders, total_shaders);

   if (minmax_sets) {
      struct kk_shader *minmax_shaders[shader_count + 2u];
      bool minmax_generated[shader_count + 2u];
      uint32_t minmax_total = 0u, unused_sets = 0u;
      result = kk_build_shaders(
         dev, shader_count, infos, minmax_nirs, state,
         features | KK_FEAT_SAMPLER_MINMAX_DYNAMIC, pAllocator, minmax_shaders,
         minmax_generated, &minmax_total, &unused_sets,
         &reported_missing_descriptor);
      if (result == VK_SUCCESS) {
         if (s->vk.stage != MESA_SHADER_COMPUTE)
            kk_gather_program(dev, state, minmax_shaders, minmax_total);
         /* The VS or CS has the program now */
         for (uint32_t i = 1; i < minmax_total; i++)
            kk_shader_destroy(&dev->vk, minmax_shaders[i], pAllocator);
         s->minmax = minmax_shaders[0];
      }
   }

   /* Keep the unused variant out of the pool: descriptor heaps usually need
    * minmax first, ordinary sampler layouts need the cheaper plain program. */
   if (result == VK_SUCCESS) {
      kk_create_ds_state(dev, s);
      const bool minmax_first =
         s->minmax != NULL &&
         kk_minmax_layouts_used(minmax_sets, infos[0].set_layout_count,
                               infos[0].set_layouts);
      if (minmax_first && KK_DEBUG(MINMAX))
         mesa_logi("kk: min/max emulating program of a pipeline compiled at "
                   "creation (a set has a reduction sampler)");
      kk_compile_enqueue(dev, minmax_first ? s->minmax : s);
   }

   /* The VS has the pipeline data now, so free the generated shaders. */
   uint32_t out_count = 0u;
   for (uint32_t i = 0; i < total_shaders; i++) {
      if (generated[i] || result != VK_SUCCESS)
         kk_shader_destroy(&dev->vk, shaders[i], pAllocator);
      else
         shaders_out[out_count++] = &shaders[i]->vk;
   }

   /* Clear the output array on failure. */
   if (result != VK_SUCCESS)
      memset(shaders_out, 0, shader_count * sizeof(*shaders_out));

   return result;
}

/* A GS and the VS holding its pipeline also carry the GS compute pass. */
static bool
kk_has_gs_main(const struct kk_shader_info *info)
{
   return info->stage == MESA_SHADER_GEOMETRY ||
          (info->stage == MESA_SHADER_VERTEX &&
           (info->vs.additional_stages_bits &
            BITFIELD_BIT(MESA_SHADER_GEOMETRY)));
}

/* An XFB GS needs pre-GS, plus a count shader if its counts aren't static. */
static bool
kk_has_pre_gs(const struct kk_shader_info *info)
{
   const struct poly_gs_info *gs =
      info->stage == MESA_SHADER_GEOMETRY ? &info->gs : &info->vs.gs;
   return kk_has_gs_main(info) && gs->xfb;
}

static bool
kk_has_gs_count(const struct kk_shader_info *info)
{
   const struct poly_gs_info *gs =
      info->stage == MESA_SHADER_GEOMETRY ? &info->gs : &info->vs.gs;
   return kk_has_pre_gs(info) && gs->count_words;
}

static void
kk_msl_serialize(const struct msl_compile_data *data, struct blob *blob)
{
   uint32_t entrypoint_length = strlen(data->entrypoint_name) + 1;
   uint32_t code_length = strlen(data->code) + 1;
   blob_write_uint32(blob, entrypoint_length);
   blob_write_uint32(blob, code_length);
   blob_write_bytes(blob, data->entrypoint_name, entrypoint_length);
   blob_write_bytes(blob, data->code, code_length);
}

static void
kk_program_serialize(const struct kk_shader *shader, struct blob *blob)
{
   kk_msl_serialize(&shader->msl_data[shader->info.stage], blob);
   if (shader->info.stage == MESA_SHADER_VERTEX) {
      u_foreach_bit(stage, shader->info.vs.additional_stages_bits) {
         kk_msl_serialize(&shader->msl_data[stage], blob);
      }
   }
   if (kk_has_gs_main(&shader->info))
      kk_msl_serialize(&shader->gs_main, blob);
   if (kk_has_gs_count(&shader->info))
      kk_msl_serialize(&shader->gs_count, blob);
   if (kk_has_pre_gs(&shader->info))
      kk_msl_serialize(&shader->pre_gs, blob);
}

static bool
kk_shader_serialize(struct vk_device *vk_dev, const struct vk_shader *vk_shader,
                    struct blob *blob)
{
   struct kk_shader *shader = container_of(vk_shader, struct kk_shader, vk);

   blob_write_bytes(blob, &shader->info, sizeof(shader->info));
   kk_program_serialize(shader, blob);
   /* info.minmax_sets != 0: the minmax program follows */
   if (shader->minmax)
      kk_program_serialize(shader->minmax, blob);

   return !blob->out_of_memory;
}

static VkResult
kk_msl_deserialize(struct blob_reader *blob, struct msl_compile_data *data)
{
   const uint32_t entrypoint_length = blob_read_uint32(blob);
   const uint32_t code_length = blob_read_uint32(blob);
   data->entrypoint_name = ralloc_array(NULL, char, entrypoint_length);
   if (data->entrypoint_name == NULL)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   data->code = ralloc_array(NULL, char, code_length);
   if (data->code == NULL)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   blob_copy_bytes(blob, (void *)data->entrypoint_name, entrypoint_length);
   blob_copy_bytes(blob, (void *)data->code, code_length);

   if (blob->overrun || entrypoint_length == 0u || code_length == 0u ||
       data->entrypoint_name[entrypoint_length - 1u] != '\0' ||
       data->code[code_length - 1u] != '\0')
      return VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT;

   kk_msl_source_intern(data);
   return VK_SUCCESS;
}

/* Reads the MSL of a program whose info is set */
static VkResult
kk_program_deserialize(struct blob_reader *blob, struct kk_shader *shader)
{
   const struct kk_shader_info *info = &shader->info;
   VkResult result = kk_msl_deserialize(blob, &shader->msl_data[info->stage]);
   if (result != VK_SUCCESS)
      return result;

   if (info->stage == MESA_SHADER_VERTEX) {
      u_foreach_bit(stage, info->vs.additional_stages_bits) {
         result = kk_msl_deserialize(blob, &shader->msl_data[stage]);
         if (result != VK_SUCCESS)
            return result;
      }
   }

   if (kk_has_gs_main(info)) {
      result = kk_msl_deserialize(blob, &shader->gs_main);
      if (result != VK_SUCCESS)
         return result;
   }

   if (kk_has_gs_count(info)) {
      result = kk_msl_deserialize(blob, &shader->gs_count);
      if (result != VK_SUCCESS)
         return result;
   }

   if (kk_has_pre_gs(info))
      result = kk_msl_deserialize(blob, &shader->pre_gs);

   return result;
}

static VkResult
kk_deserialize_shader(struct vk_device *vk_dev, struct blob_reader *blob,
                      uint32_t binary_version,
                      const VkAllocationCallbacks *pAllocator,
                      struct vk_shader **shader_out)
{
   struct kk_device *dev = container_of(vk_dev, struct kk_device, vk);

   struct kk_shader_info info;
   blob_copy_bytes(blob, &info, sizeof(info));
   if (blob->overrun)
      return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);

   struct kk_shader *shader = vk_shader_zalloc(
      &dev->vk, &kk_shader_ops, info.stage, pAllocator, sizeof(*shader));
   if (shader == NULL)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   shader->info = info;
   VkResult result = kk_program_deserialize(blob, shader);
   if (result != VK_SUCCESS)
      goto fail;

   if (info.minmax_sets) {
      shader->minmax = vk_shader_zalloc(&dev->vk, &kk_shader_ops, info.stage,
                                        pAllocator, sizeof(*shader));
      if (shader->minmax == NULL) {
         result = VK_ERROR_OUT_OF_HOST_MEMORY;
         goto fail;
      }
      shader->minmax->info = info;
      shader->minmax->info.minmax_sets = 0u;
      result = kk_program_deserialize(blob, shader->minmax);
      if (result != VK_SUCCESS)
         goto fail;
   }

   /* Without set layouts, predict minmax once any set got a reduction sampler.
    * The other program is never queued and compiles at its first use. */
   if (info.stage == MESA_SHADER_COMPUTE || info.stage == MESA_SHADER_VERTEX) {
      kk_create_ds_state(dev, shader);
      const bool minmax_first =
         shader->minmax != NULL && p_atomic_read(&dev->reduction_sampler_sets);
      if (minmax_first && KK_DEBUG(MINMAX))
         mesa_logi("kk: min/max emulating program of a pipeline compiled at "
                   "creation (a set has a reduction sampler)");
      kk_compile_enqueue(dev, minmax_first ? shader->minmax : shader);
   }

   if (result != VK_SUCCESS)
      goto fail;

   *shader_out = &shader->vk;

   return VK_SUCCESS;

fail:
   kk_shader_destroy(&dev->vk, shader, pAllocator);
   return vk_error(dev, result);
}

void
kk_cmd_bind_compute_shader(struct kk_cmd_buffer *cmd, struct kk_shader *shader)
{
   cmd->state.shaders[MESA_SHADER_COMPUTE] = shader;
}

/* The program of a pipeline (its VS or CS) a draw or dispatch with the
 * descriptor sets of desc uses: the minmax program while one of the sets its
 * samplers come from holds a reduction sampler (kk_compile_programs). An
 * update-after-bind set may get one after recording, before submission: such
 * sets are recorded for kk_queue_submit, which then re-records the command
 * buffer from its command queue. One-time-submit command buffers have no
 * command queue to replay and use the minmax program with them right away.
 * NULL when the program this draw or dispatch needs has no Metal pipeline: its
 * compile failed, and the command buffer carries the error. */
const struct kk_pipeline_handles *
kk_cmd_shader_pipeline(struct kk_cmd_buffer *cmd, struct kk_shader *shader,
                       const struct kk_descriptor_state *desc)
{
   struct kk_shader *program = shader;

   /* Which program this draw uses is only meaningful when the pipeline has
    * two: the plain one and the sampler min/max emulating one. */
   if (shader->minmax != NULL) {
      bool minmax = false;
      uint32_t uab_sets = 0u;
      u_foreach_bit(s, shader->info.minmax_sets & BITFIELD_MASK(KK_MAX_SETS)) {
         const struct kk_descriptor_set *set = desc->sets[s];
         if (set == NULL) {
            minmax = desc->push[s] != NULL && desc->push[s]->reduction_samplers;
         } else if (p_atomic_read(&set->reduction_samplers)) {
            minmax = true;
         } else if (!cmd->replay &&
                    (set->layout->flags &
                     VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT)) {
            minmax = cmd->one_time_submit;
            uab_sets |= BITFIELD_BIT(s);
         }
         if (minmax)
            break;
      }

      if (!minmax) {
         u_foreach_bit(s, uab_sets) {
            struct kk_descriptor_set *set = desc->sets[s];
            bool recorded = false;
            util_dynarray_foreach(&cmd->minmax_uab_sets,
                                  struct kk_descriptor_set *, it)
               recorded |= *it == set;
            if (!recorded)
               util_dynarray_append(&cmd->minmax_uab_sets, set);
         }
      }

      program = minmax ? shader->minmax : shader;
   }

   if (unlikely(!p_atomic_read(&program->pipeline_ready))) {
      struct kk_device *dev = kk_cmd_buffer_device(cmd);
      /* The pool compiles programs in the background (kk_compile_pool_init);
       * this waits for one that is not done yet. A program whose compile failed
       * has no Metal pipeline to bind: report it on the command buffer and let
       * the caller drop its draw or dispatch, instead of binding a nil pipeline
       * (Metal validation asserts on that). */
      /* A queued creation compile may still be running: waiting for it is
       * not a lazy compile of the other program (and must not be logged as one). */
      if (KK_DEBUG(MINMAX) && !program->compile_queued)
         mesa_logi("kk: %s program of a pipeline compiled on first use",
                   program != shader ? "min/max emulating" : "plain");
      VkResult result = kk_compile_wait(dev, program);
      if (result != VK_SUCCESS) {
         /* The pipeline's other program may still be ready */
         struct kk_shader *other = program == shader ? shader->minmax : shader;
         if (other == NULL || !p_atomic_read(&other->pipeline_ready)) {
            vk_command_buffer_set_error(&cmd->vk, result);
            return NULL;
         }
         program = other;
      }
   }
   return &program->pipeline;
}

static void
kk_cmd_bind_graphics_shader(struct kk_cmd_buffer *cmd,
                            const mesa_shader_stage stage,
                            struct kk_shader *shader)
{
   cmd->state.shaders[stage] = shader;
   cmd->state.dirty_shaders |= BITFIELD_BIT(stage);

   /* Relevant pipeline data is only stored in vertex shaders */
   if (stage != MESA_SHADER_VERTEX)
      return;

   /* The program is compiled in the background (kk_compile_pool_init) and
    * draws wait for it (kk_cmd_shader_pipeline); its depth/stencil state was
    * built with the pipeline (kk_create_ds_state), so binding does not: that
    * wait ran on the ring thread under virglrenderer's submission lock, and a
    * pipeline with static depth/stencil state must not be taken for a dynamic
    * one, whose state the application never sets. */
   bool requires_dynamic_depth_stencil = shader->pipeline.gfx.ds_handle == NULL;
   if (cmd->state.gfx.is_depth_stencil_dynamic) {
      /* If we are switching from dynamic to static, we need to clean up
       * temporary state. Otherwise, leave the existing dynamic state
       * untouched.
       */
      if (!requires_dynamic_depth_stencil) {
         mtl_release(cmd->state.gfx.depth_stencil_state);
         cmd->state.gfx.depth_stencil_state = shader->pipeline.gfx.ds_handle;
      }
   } else
      cmd->state.gfx.depth_stencil_state = shader->pipeline.gfx.ds_handle;
   cmd->state.gfx.is_depth_stencil_dynamic = requires_dynamic_depth_stencil;
   cmd->state.gfx.dirty |= KK_DIRTY_VB;

   cmd->state.gfx.pipeline_sample_count = shader->info.vs.sample_count;
}

static void
kk_cmd_bind_shaders(struct vk_command_buffer *cmd_buffer, uint32_t stage_count,
                    const mesa_shader_stage *stages,
                    struct vk_shader **const shaders)
{
   struct kk_cmd_buffer *cmd =
      container_of(cmd_buffer, struct kk_cmd_buffer, vk);

   for (uint32_t i = 0; i < stage_count; i++) {
      struct kk_shader *shader = container_of(shaders[i], struct kk_shader, vk);

      if (stages[i] == MESA_SHADER_COMPUTE || stages[i] == MESA_SHADER_KERNEL)
         kk_cmd_bind_compute_shader(cmd, shader);
      else
         kk_cmd_bind_graphics_shader(cmd, stages[i], shader);
   }
}

static VkResult
kk_shader_get_executable_properties(
   UNUSED struct vk_device *device, const struct vk_shader *vk_shader,
   uint32_t *executable_count, VkPipelineExecutablePropertiesKHR *properties)
{
   VK_OUTARRAY_MAKE_TYPED(VkPipelineExecutablePropertiesKHR, out, properties,
                          executable_count);

   return vk_outarray_status(&out);
}

static VkResult
kk_shader_get_executable_statistics(
   UNUSED struct vk_device *device, const struct vk_shader *vk_shader,
   uint32_t executable_index, uint32_t *statistic_count,
   VkPipelineExecutableStatisticKHR *statistics)
{
   /* TODO_KOSMICKRISP */
   return VK_SUCCESS;
}

static VkResult
kk_shader_get_executable_internal_representations(
   UNUSED struct vk_device *device, const struct vk_shader *vk_shader,
   uint32_t executable_index, uint32_t *internal_representation_count,
   VkPipelineExecutableInternalRepresentationKHR *internal_representations)
{
   /* TODO_KOSMICKRISP */
   return VK_SUCCESS;
}

static void
kk_vk_shader_destroy(struct vk_device *vk_dev, struct vk_shader *vk_shader,
                     const VkAllocationCallbacks *pAllocator)
{
   kk_shader_destroy(vk_dev, container_of(vk_shader, struct kk_shader, vk),
                     pAllocator);
}

static const struct vk_shader_ops kk_shader_ops = {
   .destroy = kk_vk_shader_destroy,
   .serialize = kk_shader_serialize,
   .get_executable_properties = kk_shader_get_executable_properties,
   .get_executable_statistics = kk_shader_get_executable_statistics,
   .get_executable_internal_representations =
      kk_shader_get_executable_internal_representations,
};

const struct vk_device_shader_ops kk_device_shader_ops = {
   .get_nir_options = kk_get_nir_options,
   .get_spirv_options = kk_get_spirv_options,
   .preprocess_nir = kk_preprocess_nir,
   .hash_state = kk_hash_graphics_state,
   .compile =
      kk_compile_shaders, /* This will only generate the MSL string we need
                             to use for actual library generation */
   .deserialize = kk_deserialize_shader,
   .cmd_set_dynamic_graphics_state = vk_cmd_set_dynamic_graphics_state,
   .cmd_bind_shaders = kk_cmd_bind_shaders,
};
