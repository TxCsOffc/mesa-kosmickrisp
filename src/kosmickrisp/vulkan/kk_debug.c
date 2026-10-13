/*
 * Copyright 2025 LunarG, Inc.
 * SPDX-License-Identifier: MIT
 */
#include "kk_debug.h"
#include "util/u_debug.h"

#include "kosmickrisp/bridge/mtl_compiler.h"

enum kk_debug kk_mesa_debug_flags = 0;

const struct debug_named_value debug_flags[] = {
   {"nir", KK_DEBUG_NIR},
   {"msl", KK_DEBUG_MSL},
   {"force_robustness", KK_DEBUG_FORCE_ROBUSTNESS},
   {"minmax", KK_DEBUG_MINMAX},
   {"commits", KK_DEBUG_COMMITS},
   {"cbs", KK_DEBUG_CBS},
   {"compiles", KK_DEBUG_COMPILES},
   {"noint64atomics", KK_DEBUG_NO_INT64_ATOMICS},
   {"novertexsubgroups", KK_DEBUG_NO_VERTEX_SUBGROUPS},
   {NULL, 0},
};

DEBUG_GET_ONCE_FLAGS_OPTION(mesa_kk_debug, "MESA_KK_DEBUG", debug_flags, 0);

enum kk_experimental kk_mesa_experimental_flags = 0;

const struct debug_named_value experimental_flags[] = {
   {"all", UINT64_MAX},
   {"custom_border", KK_EXPERIMENTAL_CUSTOM_BORDER},
   {"m5_optimize_e5b9g9r9", KK_EXPERIMENTAL_M5_OPTIMIZE_E5B9G9R9},
   {NULL, 0},
};

DEBUG_GET_ONCE_FLAGS_OPTION(mesa_kk_experimental, "MESA_KK_EXPERIMENTAL",
                            experimental_flags, 0);

void
kk_process_debug_variable(void)
{
   kk_mesa_debug_flags = debug_get_option_mesa_kk_debug();
   kk_mesa_experimental_flags = debug_get_option_mesa_kk_experimental();
   mtl_compiler_set_log(KK_DEBUG(COMPILES));
}
