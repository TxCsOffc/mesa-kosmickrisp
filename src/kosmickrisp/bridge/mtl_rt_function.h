#ifndef MTL_RT_FUNCTION_H
#define MTL_RT_FUNCTION_H 1

#include "mtl_types.h"

#include <stdbool.h>
#include <stdint.h>

typedef void mtl_binary_function;
typedef void mtl_visible_function_table;

mtl_binary_function *mtl_new_binary_function(mtl_compiler *compiler,
                                             mtl_library *library,
                                             const char *name);

mtl_compute_pipeline_state *
mtl_new_compute_pipeline_state_linked(mtl_compiler *compiler,
                                      mtl_function_descriptor *function,
                                      uint64_t max_total_threads_per_threadgroup,
                                      mtl_binary_function *const *functions,
                                      uint32_t function_count);

mtl_visible_function_table *
mtl_new_visible_function_table(mtl_compute_pipeline_state *pipeline,
                               uint32_t function_count);

bool mtl_visible_function_table_set(mtl_visible_function_table *table,
                                    mtl_compute_pipeline_state *pipeline,
                                    mtl_binary_function *function,
                                    uint32_t index);

uint64_t mtl_visible_function_table_resource_id(mtl_visible_function_table *table);

void mtl_set_resource_id(mtl_argument_table *table, uint64_t resource_id,
                         uint32_t binding);

#endif
