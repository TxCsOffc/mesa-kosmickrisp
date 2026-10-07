/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright © 2024 Alyssa Rosenzweig
 * Copyright © 2024 Valve Corporation
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */
#ifndef KK_QUERY_H
#define KK_QUERY_H

#include "compiler/libcl/libcl.h"

struct libkk_xfb_counter_copy {
   DEVICE(uint32_t) dest[4];
   DEVICE(uint32_t) src[4];
};

struct libkk_imm_write {
   DEVICE(uint32_t) address;
   uint32_t value;
};

#define LIBKK_QUERY_UNAVAILABLE (uint64_t)((int64_t)-1)

/* Occlusion query results live in chunks of 32768 64-bit values: Metal's
 * maximum visibility query offset is 256 KB (Metal feature set tables, Apple7
 * and later), so one visibility result buffer holds 32768 queries. A pool's
 * queries are indices into the device's chunks; libkk finds a chunk's address
 * in the device's chunk table. */
#define KK_OQ_CHUNK_QUERIES 32768u

#endif /* KK_QUERY_H */
