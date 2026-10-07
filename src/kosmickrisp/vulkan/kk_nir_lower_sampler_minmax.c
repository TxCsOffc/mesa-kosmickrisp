/*
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: MIT
 */
#include "kk_descriptor_set_layout.h"
#include "kk_descriptor_types.h"
#include "kk_sampler.h"
#include "kk_shader.h"
#include "kosmickrisp/compiler/nir_to_msl.h"
#include "nir_builder.h"
#include "nir_builtin_builder.h"

struct lower_state {
   bool dynamic;
   uint32_t set_layout_count;
   struct vk_descriptor_set_layout *const *set_layouts;
   uint32_t dynamic_sets;
};

enum sampler_kind {
   /* Immutable samplers without a reduction: nothing to emulate */
   SAMPLER_REGULAR,
   /* Immutable samplers that all reduce: always emulated */
   SAMPLER_REDUCTION,
   /* Known when the descriptor set is bound */
   SAMPLER_DYNAMIC,
};

static enum sampler_kind
classify(nir_deref_instr *deref, struct lower_state *state, uint32_t *set_out)
{
   nir_variable *var = nir_deref_instr_get_variable(deref);
   if (var == NULL) {
      *set_out = ~0u; /* any set */
      return SAMPLER_DYNAMIC;
   }
   const uint32_t set = var->data.descriptor_set;
   *set_out = set;
   if (set >= state->set_layout_count || state->set_layouts[set] == NULL)
      return SAMPLER_DYNAMIC;

   const struct kk_descriptor_set_layout *layout =
      vk_to_kk_descriptor_set_layout(state->set_layouts[set]);
   if (var->data.binding >= layout->binding_count)
      return SAMPLER_DYNAMIC;
   const struct kk_descriptor_set_binding_layout *binding =
      &layout->binding[var->data.binding];
   if (binding->immutable_samplers == NULL)
      return SAMPLER_DYNAMIC;

   uint32_t first = 0, count = binding->array_size;
   if (deref->deref_type == nir_deref_type_array &&
       nir_src_is_const(deref->arr.index) &&
       nir_deref_instr_parent(deref)->deref_type == nir_deref_type_var) {
      first = MIN2(nir_src_as_uint(deref->arr.index), count - 1);
      count = 1;
   }

   bool regular = false, reduction = false;
   for (uint32_t i = first; i < first + count; i++) {
      const struct kk_sampler *sampler = binding->immutable_samplers[i];
      if (sampler && (sampler->filter_flags & (KK_SAMPLER_REDUCTION_MIN |
                                               KK_SAMPLER_REDUCTION_MAX)))
         reduction = true;
      else
         regular = true;
   }
   return !reduction ? SAMPLER_REGULAR
          : !regular ? SAMPLER_REDUCTION
                     : SAMPLER_DYNAMIC;
}

static nir_def *
src(nir_tex_instr *tex, nir_tex_src_type type)
{
   int i = nir_tex_instr_src_index(tex, type);
   return i < 0 ? NULL : tex->src[i].src.ssa;
}

static nir_def *
load_state(nir_builder *b, nir_def *sampler, unsigned components,
           unsigned bits, unsigned offset)
{
   return nir_load_sampler_state_kk(b, components, bits, sampler, .base = offset);
}

/* Software reduction of a sample through a min/max sampler, whose hardware
 * sampler is a point sampler. Only lambda preparation is per sample site;
 * footprint enumeration and reduction live in one non-inlined MSL helper per
 * texture type. */
static nir_def *
build_emulated(nir_builder *b, nir_tex_instr *tex, nir_def *sampler,
               nir_def *flags)
{
   nir_def *lod = src(tex, nir_tex_src_lod);
   if (tex->op == nir_texop_txd) {
      unsigned dims = tex->sampler_dim == GLSL_SAMPLER_DIM_3D ? 3 : 2;
      nir_def *size = nir_build_texture_query(b, tex, nir_texop_txs,
         dims + tex->is_array, nir_type_int32, false, true);
      size = nir_i2f32(b, nir_trim_vector(b, size, dims));
      nir_def *dx = nir_fmul(b, src(tex, nir_tex_src_ddx), size);
      nir_def *dy = nir_fmul(b, src(tex, nir_tex_src_ddy), size);
      lod = nir_fmul_imm(b, nir_flog2(b, nir_fmax(b,
         nir_fdot(b, dx, dx), nir_fdot(b, dy, dy))), 0.5);
   } else if (!lod) {
      nir_def *zero_bias = nir_imm_float(b, 0);
      lod = zero_bias;
      if (nir_shader_supports_implicit_lod(b->shader)) {
         nir_def *query = nir_build_texture_query(b, tex, nir_texop_lod, 2,
            nir_type_float32, true, false);
         nir_tex_instr_add_src(nir_instr_as_tex(nir_def_instr(query)),
                              nir_tex_src_bias, zero_bias);
         lod = nir_channel(b, query, 1);
      }
      if (src(tex, nir_tex_src_bias))
         lod = nir_fadd(b, lod, nir_f2f32(b, src(tex, nir_tex_src_bias)));
   }
   lod = nir_f2f32(b, lod);
   nir_def *clamp = nir_f2f32(b, load_state(b, sampler, 2, 16,
      offsetof(struct kk_sampled_image_descriptor, sampler_lod_min_fp16)));
   lod = nir_fmin(b, nir_fmax(b, lod, nir_channel(b, clamp, 0)),
                  nir_channel(b, clamp, 1));
   if (src(tex, nir_tex_src_min_lod))
      lod = nir_fmax(b, lod, nir_f2f32(b, src(tex, nir_tex_src_min_lod)));

   /* Custom borders must be reconstructed per contributing point, before
    * reduction, not by lerping the two completed reductions. */
   nir_def *border_index = nir_i2i32(b, load_state(b, sampler, 1, 16,
      offsetof(struct kk_sampled_image_descriptor,
               clamp_0_sampler_index_or_negative)));
   nir_def *has_border = nir_ige_imm(b, border_index, 0);
   nir_def *regular_index = nir_u2u32(b, load_state(b, sampler, 1, 16,
      offsetof(struct kk_sampled_image_descriptor, sampler_index)));
   nir_def *params = nir_vec3(b, flags,
      nir_bcsel(b, has_border, border_index, regular_index),
      nir_b2i32(b, has_border));
   nir_def *border = load_state(b, sampler, 4, 32,
      offsetof(struct kk_sampled_image_descriptor, border));

   nir_tex_instr *emulated = nir_instr_as_tex(
      nir_instr_clone(b->shader, &tex->instr));
   nir_builder_instr_insert(b, &emulated->instr);
   nir_steal_tex_src(emulated, nir_tex_src_bias);
   nir_steal_tex_src(emulated, nir_tex_src_lod);
   nir_steal_tex_src(emulated, nir_tex_src_ddx);
   nir_steal_tex_src(emulated, nir_tex_src_ddy);
   nir_steal_tex_src(emulated, nir_tex_src_min_lod);
   emulated->op = nir_texop_txl;
   emulated->backend_flags |= MSL_TEX_FLAG_MINMAX;
   nir_tex_instr_add_src(emulated, nir_tex_src_lod, lod);
   nir_tex_instr_add_src(emulated, nir_tex_src_backend1, params);
   nir_tex_instr_add_src(emulated, nir_tex_src_backend2, border);

   return &emulated->def;
}

static bool
lower(nir_builder *b, nir_tex_instr *tex, void *data)
{
   struct lower_state *state = data;

   if (tex->is_shadow || (tex->backend_flags & MSL_TEX_FLAG_MINMAX) ||
       nir_alu_type_get_base_type(tex->dest_type) != nir_type_float ||
       (tex->op != nir_texop_tex && tex->op != nir_texop_txb &&
        tex->op != nir_texop_txl && tex->op != nir_texop_txd))
      return false;
   nir_def *sampler = src(tex, nir_tex_src_sampler_deref);
   if (!sampler)
      return false;

   uint32_t set;
   enum sampler_kind kind = classify(nir_def_as_deref(sampler), state, &set);
   if (kind == SAMPLER_REGULAR)
      return false;

   if (kind == SAMPLER_DYNAMIC) {
      /* Without the emulation, the pipeline's draws and dispatches use its
       * emulating program when one of these sets holds a reduction sampler. */
      state->dynamic_sets |= set < 32 ? BITFIELD_BIT(set) : ~0u;
      if (!state->dynamic)
         return false;
   }

   b->cursor = nir_before_instr(&tex->instr);
   nir_def *flags = nir_u2u32(b, load_state(b, sampler, 1, 16,
      offsetof(struct kk_sampled_image_descriptor, sampler_filter_flags)));

   if (kind == SAMPLER_REDUCTION) {
      nir_def_replace(&tex->def, build_emulated(b, tex, sampler, flags));
      return true;
   }

   nir_push_if(b, nir_ine_imm(b, nir_iand_imm(b, flags,
      KK_SAMPLER_REDUCTION_MIN | KK_SAMPLER_REDUCTION_MAX), 0));
   nir_def *emulated = build_emulated(b, tex, sampler, flags);
   nir_push_else(b, NULL);
   nir_tex_instr *original = nir_instr_as_tex(
      nir_instr_clone(b->shader, &tex->instr));
   nir_builder_instr_insert(b, &original->instr);
   nir_pop_if(b, NULL);
   nir_def_replace(&tex->def, nir_if_phi(b, emulated, &original->def));
   return true;
}

/* Before Apple10, samplers with a min/max reduction are point samplers whose
 * reduction the shader computes. Immutable samplers are resolved here.
 * Samplers from descriptor sets are emulated only with dynamic: without it,
 * dynamic_sets_out gets the sets they come from, which need the emulating
 * program (compiled with dynamic) when they hold a reduction sampler. */
bool
kk_nir_lower_sampler_minmax(nir_shader *nir, bool dynamic,
                            uint32_t set_layout_count,
                            struct vk_descriptor_set_layout *const *set_layouts,
                            uint32_t *dynamic_sets_out)
{
   struct lower_state state = {
      .dynamic = dynamic,
      .set_layout_count = set_layout_count,
      .set_layouts = set_layouts,
   };
   bool progress =
      nir_shader_tex_pass(nir, lower, nir_metadata_none, &state);
   *dynamic_sets_out |= state.dynamic_sets;
   return progress;
}
