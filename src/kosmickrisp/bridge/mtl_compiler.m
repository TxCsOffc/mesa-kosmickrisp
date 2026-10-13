/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "mtl_compiler.h"
#include "mtl_format.h"

#include <Metal/MTL4ComputePipeline.h>
#include <Metal/MTL4LibraryDescriptor.h>
#include <Metal/MTL4LibraryFunctionDescriptor.h>
#include <Metal/MTL4RenderPipeline.h>
#include <Metal/MTLDevice.h>

#include "util/hash_table.h"
#include "util/macros.h"

#include <os/lock.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

/* STEAMAC-31/39: what the Metal compiles in flight are. A compiling thread is
 * named after its compile ("kk <kind> <MSL hashes> compiler <n>: <k> in flight
 * on <m>") for as long as it runs, so a crash report's thread list shows every
 * compile in flight, the MTL4Compiler of each and how many ran at once;
 * MESA_KK_DEBUG=compiles logs each compile with the most compiles and distinct
 * compilers that overlapped it. MSL hashes are _mesa_hash_string's, as in
 * MESA_KK_DEBUG=cbs records. */
#define MTL_COMPILES_TRACKED  64u
#define MTL_COMPILERS_TRACKED 256u

static os_unfair_lock mtl_compiles_lock = OS_UNFAIR_LOCK_INIT;
static struct {
   bool used;
   const void *compiler;
   uint32_t peak_in_flight, peak_compilers;
} mtl_compiles[MTL_COMPILES_TRACKED];
/* Also counts compiles past MTL_COMPILES_TRACKED */
static uint32_t mtl_compiles_in_flight;
/* Serial numbers of the MTL4Compilers created, 1 for the first */
static struct {
   const void *handle;
   uint32_t serial;
} mtl_compilers[MTL_COMPILERS_TRACKED];
static uint32_t mtl_compilers_created;
static bool mtl_compiles_logged;
/* MSL hashes of the last two libraries this thread compiled: a compute
 * pipeline's kernel, a render pipeline's vertex and fragment functions */
static _Thread_local uint32_t mtl_thread_msl[2];

struct mtl_compile {
   const char *kind;
   int slot;
   uint32_t msl[2];
   unsigned msl_count;
   uint32_t compiler;
   uint32_t in_flight, compilers;
   uint64_t start_ns;
   char thread_name[64];
};

void
mtl_compiler_set_log(bool enabled)
{
   mtl_compiles_logged = enabled;
}

static uint32_t
mtl_compiler_serial(const void *compiler)
{
   for (uint32_t i = 0u; i < MIN2(mtl_compilers_created, MTL_COMPILERS_TRACKED); i++) {
      if (mtl_compilers[i].handle == compiler)
         return mtl_compilers[i].serial;
   }
   return 0u;
}

static void
mtl_compile_begin(struct mtl_compile *c, const void *compiler, const char *kind,
                  unsigned msl_count)
{
   *c = (struct mtl_compile){.kind = kind, .slot = -1, .msl_count = msl_count};
   if (msl_count == 1u)
      c->msl[0] = mtl_thread_msl[1];
   else if (msl_count == 2u)
      memcpy(c->msl, mtl_thread_msl, sizeof(c->msl));

   os_unfair_lock_lock(&mtl_compiles_lock);
   c->in_flight = ++mtl_compiles_in_flight;
   for (uint32_t i = 0u; i < MTL_COMPILES_TRACKED; i++) {
      if (!mtl_compiles[i].used) {
         mtl_compiles[i].used = true;
         mtl_compiles[i].compiler = compiler;
         mtl_compiles[i].peak_in_flight = 0u;
         mtl_compiles[i].peak_compilers = 0u;
         c->slot = i;
         break;
      }
   }
   for (uint32_t i = 0u; i < MTL_COMPILES_TRACKED; i++) {
      if (!mtl_compiles[i].used)
         continue;
      bool seen = false;
      for (uint32_t j = 0u; j < i && !seen; j++)
         seen = mtl_compiles[j].used && mtl_compiles[j].compiler == mtl_compiles[i].compiler;
      c->compilers += !seen;
   }
   for (uint32_t i = 0u; i < MTL_COMPILES_TRACKED; i++) {
      if (!mtl_compiles[i].used)
         continue;
      mtl_compiles[i].peak_in_flight = MAX2(mtl_compiles[i].peak_in_flight, c->in_flight);
      mtl_compiles[i].peak_compilers = MAX2(mtl_compiles[i].peak_compilers, c->compilers);
   }
   c->compiler = mtl_compiler_serial(compiler);
   os_unfair_lock_unlock(&mtl_compiles_lock);

   char name[64];
   int n = snprintf(name, sizeof(name), "kk %s", kind);
   for (unsigned i = 0u; i < msl_count && n > 0 && (size_t)n < sizeof(name); i++)
      n += snprintf(name + n, sizeof(name) - n, "%s%08x", i ? "+" : " ", c->msl[i]);
   if (n > 0 && (size_t)n < sizeof(name))
      snprintf(name + n, sizeof(name) - n, " compiler %u: %u in flight on %u", c->compiler,
               c->in_flight, c->compilers);
   if (pthread_getname_np(pthread_self(), c->thread_name, sizeof(c->thread_name)) != 0)
      c->thread_name[0] = '\0';
   pthread_setname_np(name);

   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   c->start_ns = (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void
mtl_compile_end(struct mtl_compile *c, bool ok)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   uint64_t end_ns = (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;

   uint32_t peak_in_flight = c->in_flight, peak_compilers = c->compilers;
   os_unfair_lock_lock(&mtl_compiles_lock);
   mtl_compiles_in_flight--;
   if (c->slot >= 0) {
      peak_in_flight = mtl_compiles[c->slot].peak_in_flight;
      peak_compilers = mtl_compiles[c->slot].peak_compilers;
      mtl_compiles[c->slot].used = false;
   }
   os_unfair_lock_unlock(&mtl_compiles_lock);
   pthread_setname_np(c->thread_name);

   if (!mtl_compiles_logged)
      return;
   char msl[32] = "";
   for (unsigned i = 0u, n = 0u; i < c->msl_count; i++)
      n += snprintf(msl + n, sizeof(msl) - n, "%s%08x", i ? "+" : " ", c->msl[i]);
   uint64_t thread = 0u;
   pthread_threadid_np(NULL, &thread);
   fprintf(stderr,
           "kk: compile %s%s on thread %" PRIu64 " with MTL4Compiler %u: %.1f ms%s; "
           "%u compiles in flight at its start on %u compilers, up to %u on %u\n",
           c->kind, msl, thread, c->compiler, (end_ns - c->start_ns) / 1e6,
           ok ? "" : ", failed", c->in_flight, c->compilers, peak_in_flight,
           peak_compilers);
}

/* Compiler */
mtl_compiler *
mtl_new_compiler(mtl_device *device)
{
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;

      MTL4CompilerDescriptor *desc = [[MTL4CompilerDescriptor new] autorelease];

      NSError *error;
      id<MTL4Compiler> compiler = [dev newCompilerWithDescriptor:desc error:&error];

      if (compiler == nil) {
         fprintf(stderr, "Failed to create MTL4Compiler: %s\n", [error.localizedDescription UTF8String]);
      } else {
         /* A handle reused after a release gets a new serial */
         os_unfair_lock_lock(&mtl_compiles_lock);
         uint32_t serial = ++mtl_compilers_created, i = 0u;
         while (i < MIN2(serial - 1u, MTL_COMPILERS_TRACKED) && mtl_compilers[i].handle != compiler)
            i++;
         if (i == MTL_COMPILERS_TRACKED)
            i = (serial - 1u) % MTL_COMPILERS_TRACKED;
         mtl_compilers[i].handle = compiler;
         mtl_compilers[i].serial = serial;
         os_unfair_lock_unlock(&mtl_compiles_lock);
         if (mtl_compiles_logged)
            fprintf(stderr, "kk: MTL4Compiler %u created\n", serial);
      }

      return compiler;
   }
}

/* Library */
mtl_library *
mtl_new_library(mtl_compiler *compiler, const char *src,
                enum mtl_language_version msl_version,
                enum mtl_math_mode math_mode,
                enum mtl_math_floating_point_functions math_fp_fns)
{
   @autoreleasepool {
      id<MTL4Compiler> comp = (id<MTL4Compiler>)compiler;
      NSString *ns_src = [NSString stringWithCString:src encoding:NSASCIIStringEncoding];

      MTLCompileOptions *opts = [[MTLCompileOptions new] autorelease];
      opts.languageVersion = (MTLLanguageVersion)msl_version;
      opts.mathMode = (MTLMathMode)math_mode;
      opts.mathFloatingPointFunctions = (MTLMathFloatingPointFunctions)math_fp_fns;

      MTL4LibraryDescriptor *desc = [[MTL4LibraryDescriptor new] autorelease];
      desc.source = ns_src;
      desc.options = opts;

      mtl_thread_msl[0] = mtl_thread_msl[1];
      mtl_thread_msl[1] = _mesa_hash_string(src);
      struct mtl_compile c;
      mtl_compile_begin(&c, compiler, "library", 1u);
      NSError *error;
      id<MTLLibrary> lib = [comp newLibraryWithDescriptor:desc error:&error];
      mtl_compile_end(&c, lib != nil);

      if (lib == nil) {
         fprintf(stderr, "Failed to create MTLLibrary: %s\n", [error.localizedDescription UTF8String]);
      }

      return lib;
   }
}

mtl_function_descriptor *
mtl_new_library_function_descriptor(mtl_library *library, const char *entry_point)
{
   @autoreleasepool {
      id<MTLLibrary> lib = (id<MTLLibrary>)library;
      NSString *ns_entry_point = [NSString stringWithCString:entry_point encoding:NSASCIIStringEncoding];

      MTL4LibraryFunctionDescriptor *desc = [MTL4LibraryFunctionDescriptor new];
      desc.name = ns_entry_point;
      desc.library = lib;
      return desc;
   }
}

/* Compute pipeline */
mtl_compute_pipeline_state *
mtl_new_compute_pipeline_state(mtl_compiler *compiler,
                               mtl_function_descriptor *function,
                               uint64_t max_total_threads_per_threadgroup)
{
   @autoreleasepool {
      id<MTL4Compiler> comp = (id<MTL4Compiler>)compiler;

      MTL4ComputePipelineDescriptor *desc = [[MTL4ComputePipelineDescriptor new] autorelease];
      desc.computeFunctionDescriptor = (MTL4FunctionDescriptor *)function;
      desc.maxTotalThreadsPerThreadgroup = max_total_threads_per_threadgroup;

      struct mtl_compile c;
      mtl_compile_begin(&c, compiler, "compute", 1u);
      NSError *error;
      id<MTLComputePipelineState> pipeline = [comp newComputePipelineStateWithDescriptor:desc compilerTaskOptions:nil
                                                                                   error:&error];
      mtl_compile_end(&c, pipeline != nil);

      if (pipeline == nil) {
         fprintf(stderr, "Failed to create MTLComputePipelineState: %s\n", [error.localizedDescription UTF8String]);
      }

      return pipeline;
   }
}

/* Render pipeline descriptor */
mtl_render_pipeline_descriptor *
mtl_new_render_pipeline_descriptor()
{
   @autoreleasepool {
      return [MTL4RenderPipelineDescriptor new];
   }
}

void
mtl_render_pipeline_descriptor_set_vertex_shader(mtl_render_pipeline_descriptor *descriptor,
                                                 mtl_function_descriptor *shader)
{
   @autoreleasepool {
      MTL4RenderPipelineDescriptor *desc = (MTL4RenderPipelineDescriptor *)descriptor;
      desc.vertexFunctionDescriptor = (MTL4FunctionDescriptor *)shader;
   }
}

void
mtl_render_pipeline_descriptor_set_fragment_shader(mtl_render_pipeline_descriptor *descriptor,
                                                   mtl_function_descriptor *shader)
{
   @autoreleasepool {
      MTL4RenderPipelineDescriptor *desc = (MTL4RenderPipelineDescriptor *)descriptor;
      desc.fragmentFunctionDescriptor = (MTL4FunctionDescriptor *)shader;
   }
}

void
mtl_render_pipeline_descriptor_set_input_primitive_topology(mtl_render_pipeline_descriptor *descriptor,
                                                            enum mtl_primitive_topology_class class)
{
   @autoreleasepool {
      MTL4RenderPipelineDescriptor *desc = (MTL4RenderPipelineDescriptor *)descriptor;
      desc.inputPrimitiveTopology = (MTLPrimitiveTopologyClass)class;
   }
}

void
mtl_render_pipeline_descriptor_set_color_attachment_format(mtl_render_pipeline_descriptor *descriptor,
                                                           uint8_t index,
                                                           enum mtl_pixel_format format)
{
   @autoreleasepool {
      MTL4RenderPipelineDescriptor *desc = (MTL4RenderPipelineDescriptor *)descriptor;
      desc.colorAttachments[index].pixelFormat = (MTLPixelFormat)format;
   }
}

void
mtl_render_pipeline_descriptor_set_raster_sample_count(mtl_render_pipeline_descriptor *descriptor,
                                                       uint32_t sample_count)
{
   @autoreleasepool {
      MTL4RenderPipelineDescriptor *desc = (MTL4RenderPipelineDescriptor *)descriptor;
      desc.rasterSampleCount = sample_count;
   }
}

void
mtl_render_pipeline_descriptor_set_rasterization_enabled(mtl_render_pipeline_descriptor *descriptor,
                                                         bool enabled)
{
   @autoreleasepool {
      MTL4RenderPipelineDescriptor *desc = (MTL4RenderPipelineDescriptor *)descriptor;
      desc.rasterizationEnabled = enabled;
   }
}

void
mtl_render_pipeline_descriptor_set_max_vertex_amplification_count(mtl_render_pipeline_descriptor *descriptor,
                                                                  uint32_t count)
{
   @autoreleasepool {
      MTL4RenderPipelineDescriptor *desc = (MTL4RenderPipelineDescriptor *)descriptor;
      desc.maxVertexAmplificationCount = count;
   }
}

/* Render pipeline */
mtl_render_pipeline_state *
mtl_new_render_pipeline(mtl_compiler *compiler, mtl_render_pass_descriptor *descriptor)
{
   @autoreleasepool {
      id<MTL4Compiler> comp = (id<MTL4Compiler>)compiler;
      MTL4RenderPipelineDescriptor *desc = (MTL4RenderPipelineDescriptor *)descriptor;

      struct mtl_compile c;
      mtl_compile_begin(&c, compiler, "render", 2u);
      NSError *error = nil;
      id<MTLRenderPipelineState> pipeline = [comp newRenderPipelineStateWithDescriptor:desc compilerTaskOptions:nil
                                                                                 error:&error];
      mtl_compile_end(&c, pipeline != nil);

      if (pipeline == nil) {
         fprintf(stderr, "Failed to create MTLRenderPipelineState: %s\n", [error.localizedDescription UTF8String]);
      }

      return pipeline;
   }
}
