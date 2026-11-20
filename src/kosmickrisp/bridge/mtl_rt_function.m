#include "mtl_rt_function.h"

#include <Metal/MTL4ArgumentTable.h>
#include <Metal/MTL4BinaryFunction.h>
#include <Metal/MTL4BinaryFunctionDescriptor.h>
#include <Metal/MTL4Compiler.h>
#include <Metal/MTL4ComputePipeline.h>
#include <Metal/MTL4LibraryFunctionDescriptor.h>
#include <Metal/MTL4LinkingDescriptor.h>
#include <Metal/Metal.h>

#include <stdio.h>

mtl_binary_function *
mtl_new_binary_function(mtl_compiler *compiler, mtl_library *library,
                        const char *name)
{
   @autoreleasepool {
      id<MTL4Compiler> comp = (id<MTL4Compiler>)compiler;
      NSString *fn_name = [NSString stringWithUTF8String:name];
      MTL4LibraryFunctionDescriptor *fd = [[MTL4LibraryFunctionDescriptor new] autorelease];
      fd.library = (id<MTLLibrary>)library;
      fd.name = fn_name;
      MTL4BinaryFunctionDescriptor *bd = [[MTL4BinaryFunctionDescriptor new] autorelease];
      bd.name = fn_name;
      bd.functionDescriptor = fd;
      bd.options = MTL4BinaryFunctionOptionPipelineIndependent;
      NSError *error = nil;
      id<MTL4BinaryFunction> fn = [comp newBinaryFunctionWithDescriptor:bd
                                                    compilerTaskOptions:nil
                                                                  error:&error];
      if (fn == nil)
         fprintf(stderr, "kk: failed to create binary function %s: %s\n", name,
                 [error.localizedDescription UTF8String]);
      return fn;
   }
}

mtl_compute_pipeline_state *
mtl_new_compute_pipeline_state_linked(mtl_compiler *compiler,
                                      mtl_function_descriptor *function,
                                      uint64_t max_total_threads_per_threadgroup,
                                      mtl_binary_function *const *functions,
                                      uint32_t function_count)
{
   @autoreleasepool {
      id<MTL4Compiler> comp = (id<MTL4Compiler>)compiler;
      MTL4ComputePipelineDescriptor *desc = [[MTL4ComputePipelineDescriptor new] autorelease];
      desc.computeFunctionDescriptor = (MTL4FunctionDescriptor *)function;
      desc.maxTotalThreadsPerThreadgroup = max_total_threads_per_threadgroup;
      desc.supportBinaryLinking = YES;

      NSMutableArray<id<MTL4BinaryFunction>> *fns =
         [NSMutableArray arrayWithCapacity:function_count];
      for (uint32_t i = 0; i < function_count; i++)
         [fns addObject:(id<MTL4BinaryFunction>)functions[i]];
      MTL4PipelineStageDynamicLinkingDescriptor *link =
         [[MTL4PipelineStageDynamicLinkingDescriptor new] autorelease];
      link.binaryLinkedFunctions = fns;
      link.maxCallStackDepth = 1;

      NSError *error = nil;
      id<MTLComputePipelineState> pipeline =
         [comp newComputePipelineStateWithDescriptor:desc
                            dynamicLinkingDescriptor:link
                                 compilerTaskOptions:nil
                                               error:&error];
      if (pipeline == nil)
         fprintf(stderr, "kk: failed to create linked compute pipeline: %s\n",
                 [error.localizedDescription UTF8String]);
      return pipeline;
   }
}

mtl_visible_function_table *
mtl_new_visible_function_table(mtl_compute_pipeline_state *pipeline,
                               uint32_t function_count)
{
   @autoreleasepool {
      MTLVisibleFunctionTableDescriptor *td = [[MTLVisibleFunctionTableDescriptor new] autorelease];
      td.functionCount = MAX(function_count, 1u);
      return [(id<MTLComputePipelineState>)pipeline newVisibleFunctionTableWithDescriptor:td];
   }
}

bool
mtl_visible_function_table_set(mtl_visible_function_table *table,
                               mtl_compute_pipeline_state *pipeline,
                               mtl_binary_function *function, uint32_t index)
{
   @autoreleasepool {
      id<MTLFunctionHandle> handle = [(id<MTLComputePipelineState>)pipeline
         functionHandleWithBinaryFunction:(id<MTL4BinaryFunction>)function];
      if (handle == nil)
         return false;
      [(id<MTLVisibleFunctionTable>)table setFunction:handle atIndex:index];
      return true;
   }
}

uint64_t
mtl_visible_function_table_resource_id(mtl_visible_function_table *table)
{
   return ((id<MTLVisibleFunctionTable>)table).gpuResourceID._impl;
}

void
mtl_set_resource_id(mtl_argument_table *table, uint64_t resource_id,
                    uint32_t binding)
{
   MTLResourceID rid = {._impl = resource_id};
   [(id<MTL4ArgumentTable>)table setResource:rid atBufferIndex:binding];
}
