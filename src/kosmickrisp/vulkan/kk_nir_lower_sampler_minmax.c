/*
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: MIT
 */
#include "kk_descriptor_types.h"
#include "kk_shader.h"
#include "kosmickrisp/compiler/nir_to_msl.h"
#include "nir_builder.h"
#include "nir_builtin_builder.h"

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

static bool
lower(nir_builder *b, nir_tex_instr *tex, UNUSED void *data)
{
   if (tex->is_shadow ||
       nir_alu_type_get_base_type(tex->dest_type) != nir_type_float ||
       (tex->op != nir_texop_tex && tex->op != nir_texop_txb &&
        tex->op != nir_texop_txl && tex->op != nir_texop_txd))
      return false;
   nir_def *sampler = src(tex, nir_tex_src_sampler_deref);
   if (!sampler)
      return false;

   b->cursor = nir_before_instr(&tex->instr);
   nir_def *flags = nir_u2u32(b, load_state(b, sampler, 1, 16,
      offsetof(struct kk_sampled_image_descriptor, sampler_filter_flags)));
   nir_push_if(b, nir_ine_imm(b, nir_iand_imm(b, flags,
      KK_SAMPLER_REDUCTION_MIN | KK_SAMPLER_REDUCTION_MAX), 0));

   /* Only lambda preparation is per sample site. Footprint enumeration and
    * reduction live in one non-inlined MSL helper per texture type. */
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

   nir_push_else(b, NULL);
   nir_tex_instr *original = nir_instr_as_tex(
      nir_instr_clone(b->shader, &tex->instr));
   nir_builder_instr_insert(b, &original->instr);
   nir_pop_if(b, NULL);
   nir_def_replace(&tex->def, nir_if_phi(b, &emulated->def, &original->def));
   return true;
}

bool
kk_nir_lower_sampler_minmax(nir_shader *nir)
{
   return nir_shader_tex_pass(nir, lower, nir_metadata_none, NULL);
}
