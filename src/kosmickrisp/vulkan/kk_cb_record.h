/*
 * Copyright 2026 steamac
 * SPDX-License-Identifier: MIT
 */
#ifndef KK_CB_RECORD_H
#define KK_CB_RECORD_H 1

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* What a command buffer contained, for command queue errors: the queue keeps
 * the record of its last commits (kk_queue_log_failure prints the failed one
 * and every command buffer of the process still in flight) and
 * MESA_KK_DEBUG=cbs logs every commit's. Recording only counts and ORs bits;
 * programs are noted when the bound Metal pipeline changes. */

/* MSL features of a program that can keep the GPU from making progress
 * (kk_msl_digest) */
enum kk_msl_feature {
   KK_MSL_LOOP = 1u << 0,    /* L: a loop */
   KK_MSL_DISCARD = 1u << 1, /* D: discard_fragment */
   KK_MSL_SIMD = 1u << 2,    /* S: SIMD-group or quad operations */
   KK_MSL_ATOMIC = 1u << 3,  /* A: atomics */
   KK_MSL_BARRIER = 1u << 4, /* B: threadgroup or SIMD-group barriers */
   KK_MSL_WRITE = 1u << 5,   /* W: texture writes */
   KK_MSL_FEATURE_COUNT = 6,
};

/* Work KosmicKrisp encodes for Vulkan commands other than draws and
 * dispatches */
enum kk_cb_op {
   KK_CB_OP_CLEAR_ATTACHMENTS,
   KK_CB_OP_CLEAR_IMAGE,
   KK_CB_OP_BLIT_IMAGE,
   KK_CB_OP_RESOLVE_IMAGE,
   KK_CB_OP_RESOLVE_RENDERING,
   KK_CB_OP_COPY,
   KK_CB_OP_FILL_UPDATE,
   KK_CB_OP_QUERY_RESET,
   KK_CB_OP_QUERY_COPY,
   KK_CB_OP_QUERY,
   KK_CB_OP_TIMESTAMP,
   KK_CB_OP_EVENT,
   KK_CB_OP_GS_TESS,
   KK_CB_OP_CONDITIONAL,
   KK_CB_OP_COUNT,
};

#define KK_CB_RECORD_PROGRAMS 8u

/* A Metal pipeline: render (vertex and fragment function MSL hashes) or
 * compute (kernel MSL hash, 0). */
struct kk_cb_program {
   uint32_t hash[2];
   uint8_t features; /* enum kk_msl_feature of all its functions */
   bool compute;
   /* Render pipelines with compute passes before them (GS, tessellation) */
   bool pre_render;
};

struct kk_cb_record {
   uint16_t render_passes;
   uint16_t msaa_render_passes;
   uint16_t compute_encoders;
   uint16_t draws;
   uint16_t indirect_draws;
   uint16_t dispatches;
   uint16_t indirect_dispatches;
   /* Metal command buffers committed for it */
   uint16_t metal_cmd_bufs;
   uint8_t max_samples;
   uint8_t program_count;
   /* Programs bound past programs[] */
   uint16_t programs_dropped;
   uint32_t ops;     /* BITFIELD_BIT(enum kk_cb_op) */
   uint32_t precomp; /* BITFIELD_BIT(enum libkk_program) */
   struct kk_cb_program programs[KK_CB_RECORD_PROGRAMS];
};

static inline void
kk_cb_record_count(uint16_t *counter)
{
   if (*counter != UINT16_MAX)
      (*counter)++;
}

static inline void
kk_cb_record_op(struct kk_cb_record *record, enum kk_cb_op op)
{
   record->ops |= 1u << op;
}

void kk_cb_record_program(struct kk_cb_record *record,
                          const struct kk_cb_program *program);

/* Hash and features of a program's MSL */
void kk_msl_digest(const char *code, uint32_t *hash, uint8_t *features);

/* Keeps an excerpt of the MSL of programs with loops, atomics or barriers
 * (from the first loop, atomic or barrier) for kk_cb_record_log_excerpts. */
void kk_msl_note_excerpt(uint32_t hash, uint8_t features, const char *code);

/* One line: the record's encoders, draws, dispatches, operations and
 * programs. */
void kk_cb_record_format(const struct kk_cb_record *record, char *out,
                         size_t size);

/* Logs the kept MSL excerpts of the record's programs not logged before in the
 * process, at most max_lines; returns the number logged. */
unsigned kk_cb_record_log_excerpts(const struct kk_cb_record *record,
                                   unsigned max_lines);

#endif /* KK_CB_RECORD_H */
