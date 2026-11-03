/*
 * Copyright 2026 steamac
 * SPDX-License-Identifier: MIT
 */

#include "kk_cb_record.h"

#include "libkk_shaders.h"

#include "util/hash_table.h"
#include "util/log.h"
#include "util/macros.h"
#include "util/simple_mtx.h"

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

void
kk_cb_record_program(struct kk_cb_record *record,
                     const struct kk_cb_program *program)
{
   for (uint32_t i = 0u; i < record->program_count; i++) {
      const struct kk_cb_program *p = &record->programs[i];
      if (p->hash[0] == program->hash[0] && p->hash[1] == program->hash[1] &&
          p->compute == program->compute)
         return;
   }
   if (record->program_count < KK_CB_RECORD_PROGRAMS)
      record->programs[record->program_count++] = *program;
   else
      kk_cb_record_count(&record->programs_dropped);
}

void
kk_cb_record_log_dispatches(const struct kk_cb_record *record)
{
   const struct kk_dispatch_record *d = &record->largest_direct;
   if (record->largest_direct_groups)
      mesa_loge("kk: largest direct dispatch requested c:%08x groups=%ux%ux%u local=%ux%ux%u",
                d->hash, d->groups[0], d->groups[1], d->groups[2],
                d->local[0], d->local[1], d->local[2]);
   d = &record->last_indirect;
   if (d->indirect)
      mesa_loge("kk: last indirect dispatch c:%08x local=%ux%ux%u address=0x%llx (GPU group counts unknown)",
                d->hash, d->local[0], d->local[1], d->local[2],
                (unsigned long long)d->indirect_addr);
}

/* The patterns nir_to_msl emits for each feature; a loop is
 * "for (uint64_t no_crash ..." or, with KK_WORKAROUND_2 disabled,
 * "while (true)". */
static const struct {
   const char *pattern;
   enum kk_msl_feature feature;
} kk_msl_patterns[] = {
   {"no_crash", KK_MSL_LOOP},
   {"while (true)", KK_MSL_LOOP},
   {"discard_fragment", KK_MSL_DISCARD},
   {"simd_", KK_MSL_SIMD},
   {"quad_", KK_MSL_SIMD},
   {"atomic_", KK_MSL_ATOMIC},
   {"_barrier(", KK_MSL_BARRIER},
   {".write(", KK_MSL_WRITE},
};

void
kk_msl_digest(const char *code, uint32_t *hash, uint8_t *features)
{
   *hash = code ? _mesa_hash_string(code) : 0u;
   *features = 0u;
   if (code == NULL)
      return;
   for (uint32_t i = 0u; i < ARRAY_SIZE(kk_msl_patterns); i++) {
      if (strstr(code, kk_msl_patterns[i].pattern))
         *features |= kk_msl_patterns[i].feature;
   }
}

#define KK_MSL_EXCERPTS       128u
#define KK_MSL_EXCERPT_LENGTH 480u
#define KK_MSL_EXCERPT_BEFORE 96u

/* Programs whose MSL could spin or wait on other threads */
#define KK_MSL_EXCERPT_FEATURES (KK_MSL_LOOP | KK_MSL_ATOMIC | KK_MSL_BARRIER)

static struct {
   simple_mtx_t lock;
   uint32_t next;
   struct {
      uint32_t hash;
      bool logged;
      char text[KK_MSL_EXCERPT_LENGTH + 1u];
   } entries[KK_MSL_EXCERPTS];
} kk_msl_excerpts = {.lock = SIMPLE_MTX_INITIALIZER};

void
kk_msl_note_excerpt(uint32_t hash, uint8_t features, const char *code)
{
   if (!(features & KK_MSL_EXCERPT_FEATURES) || code == NULL)
      return;

   /* From a little before the first loop, atomic or barrier */
   const char *first = NULL;
   for (uint32_t i = 0u; i < ARRAY_SIZE(kk_msl_patterns); i++) {
      if (!(kk_msl_patterns[i].feature & KK_MSL_EXCERPT_FEATURES))
         continue;
      const char *at = strstr(code, kk_msl_patterns[i].pattern);
      if (at && (first == NULL || at < first))
         first = at;
   }
   const char *start = first - MIN2((size_t)(first - code),
                                    (size_t)KK_MSL_EXCERPT_BEFORE);

   simple_mtx_lock(&kk_msl_excerpts.lock);
   for (uint32_t i = 0u; i < KK_MSL_EXCERPTS; i++) {
      if (kk_msl_excerpts.entries[i].hash == hash &&
          kk_msl_excerpts.entries[i].text[0]) {
         simple_mtx_unlock(&kk_msl_excerpts.lock);
         return;
      }
   }
   uint32_t slot = kk_msl_excerpts.next++ % KK_MSL_EXCERPTS;
   kk_msl_excerpts.entries[slot].hash = hash;
   kk_msl_excerpts.entries[slot].logged = false;
   /* One line, runs of whitespace collapsed */
   char *out = kk_msl_excerpts.entries[slot].text;
   size_t n = 0u;
   for (const char *c = start; *c && n < KK_MSL_EXCERPT_LENGTH; c++) {
      const bool space = *c == ' ' || *c == '\n' || *c == '\t';
      if (space && (n == 0u || out[n - 1u] == ' '))
         continue;
      out[n++] = space ? ' ' : *c;
   }
   out[n] = '\0';
   simple_mtx_unlock(&kk_msl_excerpts.lock);
}

unsigned
kk_cb_record_log_excerpts(const struct kk_cb_record *record,
                          unsigned max_lines)
{
   unsigned lines = 0u;
   simple_mtx_lock(&kk_msl_excerpts.lock);
   for (uint32_t p = 0u; p < record->program_count && lines < max_lines; p++) {
      for (uint32_t h = 0u; h < 2u && lines < max_lines; h++) {
         const uint32_t hash = record->programs[p].hash[h];
         if (hash == 0u)
            continue;
         for (uint32_t i = 0u; i < KK_MSL_EXCERPTS; i++) {
            if (kk_msl_excerpts.entries[i].hash != hash ||
                !kk_msl_excerpts.entries[i].text[0] ||
                kk_msl_excerpts.entries[i].logged)
               continue;
            kk_msl_excerpts.entries[i].logged = true;
            mesa_loge("kk: MSL %08x from its first loop, atomic or barrier: %s",
                      hash, kk_msl_excerpts.entries[i].text);
            lines++;
            break;
         }
      }
   }
   simple_mtx_unlock(&kk_msl_excerpts.lock);
   return lines;
}

static const char *const kk_cb_op_names[KK_CB_OP_COUNT] = {
   [KK_CB_OP_CLEAR_ATTACHMENTS] = "clear-attachments",
   [KK_CB_OP_CLEAR_IMAGE] = "clear-image",
   [KK_CB_OP_BLIT_IMAGE] = "blit-image",
   [KK_CB_OP_RESOLVE_IMAGE] = "resolve-image",
   [KK_CB_OP_RESOLVE_RENDERING] = "resolve-rendering",
   [KK_CB_OP_COPY] = "copy",
   [KK_CB_OP_FILL_UPDATE] = "fill/update",
   [KK_CB_OP_QUERY_RESET] = "query-reset",
   [KK_CB_OP_QUERY_COPY] = "query-copy",
   [KK_CB_OP_QUERY] = "query",
   [KK_CB_OP_TIMESTAMP] = "timestamp",
   [KK_CB_OP_EVENT] = "event",
   [KK_CB_OP_GS_TESS] = "gs/tess",
   [KK_CB_OP_CONDITIONAL] = "conditional",
};

static const char *const kk_precomp_names[LIBKK_NUM_PROGRAMS] = {
   [LIBKK_PREDICATE_INDIRECT_GT_DRAW_ID] = "predicate_gt_draw_id",
   [LIBKK_PREDICATE_INDIRECT_EQ_ZERO] = "predicate_eq_zero",
   [LIBKK_PREDICATE_INDIRECT_NEQ_ZERO] = "predicate_neq_zero",
   [LIBKK_DRAW_BYTE_COUNT] = "draw_byte_count",
   [LIBKK_UNROLL_GEOMETRY] = "unroll_geometry",
   [LIBKK_UNROLL_RESTART] = "unroll_restart",
   [LIBKK_UNROLL_ADJACENCY] = "unroll_adjacency",
   [LIBKK_PREFIX_SUM_TESS] = "prefix_sum_tess",
   [LIBKK_GS_SETUP_INDIRECT] = "gs_setup_indirect",
   [LIBKK_PREFIX_SUM_GEOM] = "prefix_sum_geom",
   [LIBKK_RESET_QUERY] = "reset_query",
   [LIBKK_WRITE_U32_ARRAY] = "write_u32_array",
   [LIBKK_WRITE_U32] = "write_u32",
   [LIBKK_COPY_XFB_COUNTERS] = "copy_xfb_counters",
   [LIBKK_COPY_QUERIES] = "copy_queries",
   [LIBKK_TESS_SETUP_INDIRECT] = "tess_setup_indirect",
   [LIBKK_TESS_ISOLINE] = "tess_isoline",
   [LIBKK_TESS_TRI] = "tess_tri",
   [LIBKK_TESS_QUAD] = "tess_quad",
};
static_assert(LIBKK_NUM_PROGRAMS <= 32, "precomp is a 32-bit mask");

struct kk_out {
   char *p;
   size_t left;
};

static void PRINTFLIKE(2, 3)
kk_out_printf(struct kk_out *out, const char *fmt, ...)
{
   if (out->left <= 1u)
      return;
   va_list args;
   va_start(args, fmt);
   int n = vsnprintf(out->p, out->left, fmt, args);
   va_end(args);
   if (n < 0)
      return;
   const size_t used = MIN2((size_t)n, out->left - 1u);
   out->p += used;
   out->left -= used;
}

void
kk_cb_record_format(const struct kk_cb_record *r, char *buf, size_t size)
{
   struct kk_out out = {buf, size};
   buf[0] = '\0';

   kk_out_printf(&out, "%u render passes", r->render_passes);
   if (r->msaa_render_passes)
      kk_out_printf(&out, " (%u at %ux)", r->msaa_render_passes,
                    r->max_samples);
   kk_out_printf(&out,
                 ", %u compute encoders, %u draws (%u indirect), %u "
                 "dispatches (%u indirect), %u Metal command buffers",
                 r->compute_encoders, r->draws, r->indirect_draws,
                 r->dispatches, r->indirect_dispatches, r->metal_cmd_bufs);
   if (r->ops) {
      kk_out_printf(&out, "; ops");
      for (uint32_t i = 0u; i < KK_CB_OP_COUNT; i++) {
         if (r->ops & (1u << i))
            kk_out_printf(&out, " %s", kk_cb_op_names[i]);
      }
   }
   if (r->precomp) {
      kk_out_printf(&out, "; kernels");
      for (uint32_t i = 0u; i < LIBKK_NUM_PROGRAMS; i++) {
         if (r->precomp & (1u << i))
            kk_out_printf(&out, " %s", kk_precomp_names[i]);
      }
   }
   if (r->program_count) {
      kk_out_printf(&out, "; programs (MSL hashes, LDSABW features)");
      for (uint32_t i = 0u; i < r->program_count; i++) {
         const struct kk_cb_program *p = &r->programs[i];
         char features[KK_MSL_FEATURE_COUNT + 1u];
         uint32_t n = 0u;
         for (uint32_t f = 0u; f < KK_MSL_FEATURE_COUNT; f++) {
            if (p->features & (1u << f))
               features[n++] = "LDSABW"[f];
         }
         features[n] = '\0';
         if (p->compute)
            kk_out_printf(&out, " c:%08x", p->hash[0]);
         else
            kk_out_printf(&out, " r%s:%08x/%08x", p->pre_render ? "+" : "",
                          p->hash[0], p->hash[1]);
         if (n)
            kk_out_printf(&out, ":%s", features);
      }
      if (r->programs_dropped)
         kk_out_printf(&out, " +%u more", r->programs_dropped);
   }
}
