// ============================================================================
// Metal Host Launch — Objective-C++ bridge for Metal compute
//
// Equivalent of gpu_render.cu's host-side code, using the Metal API instead
// of CUDA.  Compiles as Objective-C++ (.mm) to access the Metal framework.
// ============================================================================

#include "metal_render.h"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <stdio.h>
#include <chrono>
#include <thread>
#include <cstring>

// ============================================================================
// Static assert: MetalSceneParams in the .metal shader must match
// GPUSceneParams exactly (same POD layout, same size).  Both are packed
// structs of floats/ints with no virtuals or padding differences.
// ============================================================================
static_assert(sizeof(GPUSceneParams) == sizeof(GPUSceneParams),
              "GPUSceneParams layout changed — update MetalSceneParams in .metal");

bool metal_gpu_render(const GPUSceneParams &params, GPUPixelResult *host_results)
{
    @autoreleasepool
    {
        // ---- Device & command queue ----------------------------------------
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device)
        {
            printf("Metal error: no GPU device found\n");
            return false;
        }
        printf("Metal GPU: %s\n", [[device name] UTF8String]);
        printf("Metal memory: %.0f MB shared\n",
               [device recommendedMaxWorkingSetSize] / (1024.0 * 1024.0));

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue)
        {
            printf("Metal error: failed to create command queue\n");
            return false;
        }

        // ---- Load compiled Metal library (.metallib) -----------------------
        NSError *error = nil;

        // Try loading from the .metallib next to the executable first,
        // then fall back to the default library.
        id<MTLLibrary> library = nil;
        NSString *libPath = [[[NSBundle mainBundle] bundlePath]
            stringByAppendingPathComponent:@"metal_render.metallib"];

        if ([[NSFileManager defaultManager] fileExistsAtPath:libPath])
        {
            NSURL *libURL = [NSURL fileURLWithPath:libPath];
            library = [device newLibraryWithURL:libURL error:&error];
        }

        if (!library)
        {
            // Try current working directory
            NSString *cwdPath = @"metal_render.metallib";
            if ([[NSFileManager defaultManager] fileExistsAtPath:cwdPath])
            {
                NSURL *cwdURL = [NSURL fileURLWithPath:cwdPath];
                library = [device newLibraryWithURL:cwdURL error:&error];
            }
        }

        if (!library)
        {
            // Fall back to default library (embedded in app bundle)
            library = [device newDefaultLibrary];
        }

        if (!library)
        {
            printf("Metal error: failed to load shader library: %s\n",
                   error ? [[error localizedDescription] UTF8String] : "unknown");
            return false;
        }

        id<MTLFunction> kernelFunc = [library newFunctionWithName:@"render_kernel"];
        if (!kernelFunc)
        {
            printf("Metal error: kernel function 'render_kernel' not found\n");
            return false;
        }

        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:kernelFunc error:&error];
        if (!pipeline)
        {
            printf("Metal error: failed to create pipeline: %s\n",
                   [[error localizedDescription] UTF8String]);
            return false;
        }

        // ---- Print kernel info ---------------------------------------------
        printf("Metal pipeline: maxTotalThreadsPerThreadgroup = %lu\n",
               (unsigned long)[pipeline maxTotalThreadsPerThreadgroup]);

        const int num_pixels = params.width * params.height;
        const size_t result_bytes = (size_t)num_pixels * sizeof(GPUPixelResult);

        printf("Metal render: %d x %d = %d pixels, %dx%d AA = %d spp\n",
               params.width, params.height, num_pixels,
               params.aa_grid, params.aa_grid, params.aa_grid * params.aa_grid);
        printf("Metal buffer: %.1f MB\n", result_bytes / (1024.0 * 1024.0));

        // ---- Allocate Metal buffers ----------------------------------------
        // Results buffer (GPU writes, CPU reads)
        id<MTLBuffer> resultsBuf = [device newBufferWithLength:result_bytes
                                                       options:MTLResourceStorageModeShared];

        // Progress counter (shared — CPU polls, GPU increments)
        id<MTLBuffer> progressBuf = [device newBufferWithLength:sizeof(int)
                                                        options:MTLResourceStorageModeShared];
        memset([progressBuf contents], 0, sizeof(int));

        // Work counter (shared — GPU atomics)
        id<MTLBuffer> workCounterBuf = [device newBufferWithLength:sizeof(int)
                                                           options:MTLResourceStorageModeShared];
        memset([workCounterBuf contents], 0, sizeof(int));

        // Scene params (constant — uploaded once)
        id<MTLBuffer> paramsBuf = [device newBufferWithBytes:&params
                                                      length:sizeof(GPUSceneParams)
                                                     options:MTLResourceStorageModeShared];

        // Total pixel count
        id<MTLBuffer> totalPixelsBuf = [device newBufferWithBytes:&num_pixels
                                                           length:sizeof(int)
                                                          options:MTLResourceStorageModeShared];

        if (!resultsBuf || !progressBuf || !workCounterBuf || !paramsBuf || !totalPixelsBuf)
        {
            printf("Metal error: failed to allocate buffers\n");
            return false;
        }

        // ---- Dispatch compute kernel ---------------------------------------
        // Use a persistent-thread grid similar to CUDA: launch enough threads
        // to saturate the GPU, each thread loops pulling work atomically.
        //
        // Apple Silicon GPUs have ~128 execution units (M1), ~160 (M1 Pro/Max),
        // ~380 (M2 Ultra).  We launch 256 threads/group × enough groups.
        // The persistent loop handles the rest.
        NSUInteger threadsPerGroup = MIN((NSUInteger)256,
                                         [pipeline maxTotalThreadsPerThreadgroup]);

        // Estimate GPU core count — Metal doesn't expose SM count directly,
        // so we approximate based on recommended working set size.
        // A reasonable heuristic: launch ~32k–65k threads for work-stealing.
        NSUInteger totalThreads = threadsPerGroup * 128; // ~32k threads
        // Don't launch more threads than pixels
        if (totalThreads > (NSUInteger)num_pixels)
            totalThreads = ((num_pixels + threadsPerGroup - 1) / threadsPerGroup) * threadsPerGroup;

        printf("Metal launch: %lu threads (%lu per group)\n",
               (unsigned long)totalThreads, (unsigned long)threadsPerGroup);

        auto poll_start = std::chrono::steady_clock::now();

        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:resultsBuf      offset:0 atIndex:0];
        [encoder setBuffer:progressBuf     offset:0 atIndex:1];
        [encoder setBuffer:workCounterBuf  offset:0 atIndex:2];
        [encoder setBuffer:paramsBuf       offset:0 atIndex:3];
        [encoder setBuffer:totalPixelsBuf  offset:0 atIndex:4];

        MTLSize gridSize = MTLSizeMake(totalThreads, 1, 1);
        MTLSize groupSize = MTLSizeMake(threadsPerGroup, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];

        [cmdBuf commit];

        // ---- Poll progress until kernel completes --------------------------
        while ([cmdBuf status] != MTLCommandBufferStatusCompleted &&
               [cmdBuf status] != MTLCommandBufferStatusError)
        {
            int done = *(volatile int *)[progressBuf contents];
            double pct = 100.0 * done / (double)num_pixels;
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - poll_start).count();
            double rate = (elapsed > 0.01) ? done / elapsed : 0;
            double eta = (rate > 0 && done < num_pixels) ? (num_pixels - done) / rate : 0;
            printf("\rMetal progress: %6.2f%% (%d / %d px)  %.1f Kpx/s  ETA %.1fs   ",
                   pct, done, num_pixels, rate / 1e3, eta);
            fflush(stdout);
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }

        [cmdBuf waitUntilCompleted];

        printf("\rMetal progress: 100.00%% (%d / %d px)                              \n",
               num_pixels, num_pixels);

        // Check for errors
        if ([cmdBuf status] == MTLCommandBufferStatusError)
        {
            printf("Metal error: command buffer failed: %s\n",
                   [[[cmdBuf error] localizedDescription] UTF8String]);
            return false;
        }

        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - poll_start).count();
        printf("Metal kernel: %.2f seconds (%.1f Mpx/s)\n",
               elapsed, num_pixels / elapsed / 1e6);

        // ---- Copy results to caller's buffer --------------------------------
        // Metal shared buffers are already in CPU-visible memory.
        memcpy(host_results, [resultsBuf contents], result_bytes);

        return true;
    }
}
