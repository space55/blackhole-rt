// ============================================================================
// GPU Black Hole Ray Tracer — CUDA Kernel + Launch
//
// All physics (geodesic integration, disk emission, radiative transfer) lives
// in the shared header physics.h — compiled here as __host__ __device__ via
// the BH_FUNC macro.  This file contains only the render kernel and host
// launch function.
// ============================================================================

#include "gpu_render.h"
#include "physics.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>
#include <thread>

// ============================================================================
// CUDA error checking
// ============================================================================
#define CUDA_CHECK(call)                                            \
    do                                                              \
    {                                                               \
        cudaError_t err = (call);                                   \
        if (err != cudaSuccess)                                     \
        {                                                           \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err));                        \
            return false;                                           \
        }                                                           \
    } while (0)

// ============================================================================
// Constant memory — scene parameters broadcast to every thread
// ============================================================================
__constant__ GPUSceneParams c_params;

// ============================================================================
// Render Kernel — persistent-thread work-stealing design
//
// Instead of mapping one thread to one pixel via 2D grid coordinates, we
// launch a fixed number of threads and each one pulls pixel indices from a
// global atomic counter.  Benefits:
//
//  1. Finished threads immediately pick up new work instead of idling while
//     neighbouring threads in the same warp/block are still integrating
//     expensive photon-sphere rays.  This eliminates the "tail-end" stall.
//
//  2. The work order is interleaved: thread 0 gets pixel 0, thread 1 gets
//     pixel 1, etc.  Spatially adjacent heavy pixels (around the shadow
//     edge) are spread across many warps/blocks instead of clustering in
//     one block.
//
//  3. No __syncthreads() needed anywhere — every thread is fully independent.
// ============================================================================
__global__ __launch_bounds__(256, 2) void render_kernel(GPUPixelResult *results,
                                                        int *progress,
                                                        int *work_counter,
                                                        int total_pixels)
{
    // Local copy of physics params — constant memory is cached and
    // broadcasts to all warp threads for free; copying into a local
    // struct lets the compiler keep hot fields (bh_spin, bh_mass) in
    // registers for the tight geodesic_accel inner loop.
    const PhysicsParams pp = c_params.physics;

    // Camera basis vectors (from constant memory)
    const dvec3 cam_right(c_params.cam_right[0], c_params.cam_right[1], c_params.cam_right[2]);
    const dvec3 cam_up(c_params.cam_up[0], c_params.cam_up[1], c_params.cam_up[2]);
    const dvec3 cam_fwd(c_params.cam_fwd[0], c_params.cam_fwd[1], c_params.cam_fwd[2]);
    const dvec3 cam_pos(c_params.cam_pos[0], c_params.cam_pos[1], c_params.cam_pos[2]);

    const int width = c_params.width;
    const int aa_grid = c_params.aa_grid;
    const bh_real inv_aa = 1.0 / aa_grid;
    const bh_real inv_spp = 1.0 / (bh_real)(aa_grid * aa_grid);

    const bh_real r_plus = pp.r_plus;
    const bh_real base_dt = c_params.base_dt;
    const bh_real max_affine = c_params.max_affine;
    const bh_real escape_r2 = c_params.escape_r2;
    const bh_real width_d = (bh_real)c_params.width;
    const bh_real height_d = (bh_real)c_params.height;
    const bh_real fov_x = c_params.fov_x;
    const bh_real fov_y = c_params.fov_y;

    // Hard iteration cap to bound worst-case rays near the photon sphere
    const int max_iter = c_params.max_iter;

    // Persistent work loop — each thread grabs one pixel at a time
    while (true)
    {
        const int idx = atomicAdd(work_counter, 1);
        if (idx >= total_pixels)
            break;

        const int px = idx % width;
        const int py = idx / width;

        dvec3 pixel_disk(0, 0, 0);
        bh_real pixel_sky_weight = 0;
        dvec3 pixel_exit_dir(0, 0, 0);

        for (int sy = 0; sy < aa_grid; sy++)
        {
            for (int sx = 0; sx < aa_grid; sx++)
            {
                const bh_real sub_x = (px + (sx + 0.5) * inv_aa) / width_d;
                const bh_real sub_y = (py + (sy + 0.5) * inv_aa) / height_d;

                // Initialize ray using shared physics
                dvec3 pos, vel;
                init_ray(pos, vel, cam_pos, cam_right, cam_up, cam_fwd,
                         sub_x, sub_y, fov_x, fov_y);

                dvec3 acc_color(0, 0, 0);
                bh_real acc_opacity = 0;
                bh_real cached_r = ks_radius(pos, pp.bh_spin);
                bool hit_bh = false;
                bh_real affine = 0;
                int iter = 0;

                while (affine < max_affine && iter < max_iter)
                {
                    ++iter;
                    const bh_real delta = bh_fmax(cached_r - r_plus, 0.01);
                    bh_real step_dt = base_dt * dclamp(delta * delta, 0.001, 1.0);

                    // Step size reduction near disk
                    const bh_real disk_inner_guard = pp.disk_flat_mode ? pp.disk_inner_r * 0.4 : pp.disk_inner_r * 0.8;
                    const bh_real disk_outer_guard = pp.disk_flat_mode ? pp.disk_outer_r * 1.6 : pp.disk_outer_r * 1.2;
                    if (cached_r >= disk_inner_guard && cached_r <= disk_outer_guard)
                    {
                        const bh_real h = disk_half_thickness(cached_r, pp);
                        const bh_real y_dist = fabs(pos.y);
                        if (y_dist < 5.0 * h)
                        {
                            step_dt = bh_fmin(step_dt, bh_fmax(0.15 * h, 0.001));

                            // Grazing-angle refinement
                            const bh_real v_horiz_sq = vel.x * vel.x + vel.z * vel.z;
                            const bh_real v_vert_sq = vel.y * vel.y;
                            if (v_horiz_sq > 4.0 * v_vert_sq)
                            {
                                const bh_real v_horiz = sqrt(v_horiz_sq);
                                const bh_real texture_scale = pp.disk_flat_mode ? 0.15 : 0.3;
                                step_dt = bh_fmin(step_dt, texture_scale / v_horiz);
                            }
                        }
                    }

                    const bh_real prev_y = pos.y;

                    if (!advance_ray(pos, vel, cached_r, step_dt, pp))
                    {
                        hit_bh = true;
                        break;
                    }

                    // Midplane crossing sub-step for flat mode
                    if (pp.disk_flat_mode && prev_y * pos.y < 0.0)
                    {
                        if (cached_r >= (pp.disk_inner_r * 0.4) &&
                            cached_r <= (pp.disk_outer_r * 1.6))
                        {
                            const bh_real t_cross = fabs(prev_y) /
                                                    bh_fmax(fabs(prev_y) + fabs(pos.y), 1e-12);
                            dvec3 mid_pos = (1.0 - t_cross) * (pos - step_dt * vel) +
                                            t_cross * pos;
                            mid_pos.y = 0.0;
                            sample_disk_volume(mid_pos, vel, step_dt * 0.5,
                                               acc_color, acc_opacity, cached_r, pp);
                        }
                    }

                    // Validate ray state BEFORE disk sampling.  Under
                    // --use_fast_math, NaN slips past fmax/fmin guards and
                    // NaN comparisons return false, so NaN position would
                    // bypass range checks inside sample_disk_volume and
                    // corrupt acc_opacity — poisoning the sky transmittance
                    // while leaving acc_color intact (appearing as sky-only
                    // noise).
                    if (!bh_isfinite(vel.squaredNorm()) || !bh_isfinite(cached_r))
                    {
                        hit_bh = true;
                        break;
                    }

                    // Cheap vertical check: skip disk sampling entirely when
                    // far above/below the disk plane (avoids expensive inner
                    // computation for the vast majority of ray steps)
                    const bh_real samp_inner_guard = pp.disk_flat_mode ? pp.disk_inner_r * 0.4 : pp.disk_inner_r * 0.8;
                    const bh_real samp_outer_guard = pp.disk_flat_mode ? pp.disk_outer_r * 1.6 : pp.disk_outer_r * 1.3;
                    if (fabs(pos.y) < pp.disk_thickness * 15.0 &&
                        cached_r >= samp_inner_guard &&
                        cached_r <= samp_outer_guard)
                    {
                        sample_disk_volume(pos, vel, step_dt, acc_color, acc_opacity, cached_r, pp);
                    }

                    affine += step_dt;
                    if (cached_r <= r_plus)
                    {
                        hit_bh = true;
                        break;
                    }
                    if (pos.squaredNorm() > escape_r2)
                        break;
                    if (acc_opacity > 0.95)
                        break;
                }

                pixel_disk += acc_color;
                if (!hit_bh && bh_isfinite(acc_opacity))
                {
                    bh_real transmittance = 1.0 - acc_opacity;
                    pixel_sky_weight += transmittance;
                    pixel_exit_dir += transmittance * vel;
                }
            }
        }

        pixel_disk *= inv_spp;
        pixel_sky_weight *= inv_spp;
        pixel_exit_dir *= inv_spp;

        // Sanitise outputs.  --use_fast_math can let NaN slip through in rare
        // edge cases.  For the exit direction we must validate ATOMICALLY: if
        // any component is non-finite, zero the ENTIRE direction + weight so
        // the CPU sky mapper never sees a partially-corrupted direction
        // pointing to a random sky location.
        auto is_finite_d = [](bh_real v) -> bool
        {
            return bh_isfinite(v);
        };

        if (!is_finite_d(pixel_sky_weight) || !is_finite_d(pixel_exit_dir.x) ||
            !is_finite_d(pixel_exit_dir.y) || !is_finite_d(pixel_exit_dir.z))
        {
            pixel_sky_weight = 0;
            pixel_exit_dir = dvec3(0, 0, 0);
        }

        // Disk components can be sanitised individually (no directional semantics)
        if (!is_finite_d(pixel_disk.x))
            pixel_disk.x = 0;
        if (!is_finite_d(pixel_disk.y))
            pixel_disk.y = 0;
        if (!is_finite_d(pixel_disk.z))
            pixel_disk.z = 0;

        results[idx].disk_r = (float)pixel_disk.x;
        results[idx].disk_g = (float)pixel_disk.y;
        results[idx].disk_b = (float)pixel_disk.z;
        results[idx].sky_weight = (float)pixel_sky_weight;
        results[idx].exit_vx = (float)pixel_exit_dir.x;
        results[idx].exit_vy = (float)pixel_exit_dir.y;
        results[idx].exit_vz = (float)pixel_exit_dir.z;

        // Update progress counter (visible to host via mapped pinned memory)
        atomicAdd(progress, 1);
    }
}

// ============================================================================
// Host Launch Function
// ============================================================================
bool gpu_render(const GPUSceneParams &params, GPUPixelResult *host_results)
{
    // Print GPU info
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("GPU: %s (compute %d.%d, %d SMs, %.0f MHz)\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, prop.clockRate / 1000.0);
    printf("GPU memory: %.0f MB total, bh_real-precision throughput: %s\n",
           prop.totalGlobalMem / (1024.0 * 1024.0),
           (prop.major >= 8) ? "good (Ampere+)" : (prop.major >= 7) ? "decent (Volta+)"
                                                                    : "limited");

    const int num_pixels = params.width * params.height;
    const size_t result_bytes = (size_t)num_pixels * sizeof(GPUPixelResult);

    printf("GPU render: %d x %d = %d pixels, %dx%d AA = %d spp\n",
           params.width, params.height, num_pixels,
           params.aa_grid, params.aa_grid, params.aa_grid * params.aa_grid);
    printf("GPU buffer: %.1f MB\n", result_bytes / (1024.0 * 1024.0));

    // Allocate device memory
    GPUPixelResult *d_results = nullptr;
    CUDA_CHECK(cudaMalloc(&d_results, result_bytes));

    // Copy scene parameters to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(c_params, &params, sizeof(GPUSceneParams)));

    // Allocate mapped pinned memory for progress counter (zero-copy host↔device)
    int *h_progress = nullptr;
    int *d_progress = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_progress, sizeof(int), cudaHostAllocMapped));
    *h_progress = 0;
    CUDA_CHECK(cudaHostGetDevicePointer(&d_progress, h_progress, 0));

    // Allocate device work counter for persistent-thread work-stealing
    int *d_work_counter = nullptr;
    CUDA_CHECK(cudaMalloc(&d_work_counter, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_work_counter, 0, sizeof(int)));

    // Launch persistent-thread kernel: 256 threads/block, enough blocks to
    // fill all SMs with ~2 blocks each for latency hiding.  The work-stealing
    // loop inside the kernel handles the pixel→thread mapping dynamically.
    const int threads_per_block = 256;
    const int blocks = prop.multiProcessorCount * 2;
    printf("GPU launch: %d blocks × %d threads = %d persistent threads\n",
           blocks, threads_per_block, blocks * threads_per_block);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    render_kernel<<<blocks, threads_per_block>>>(d_results, d_progress,
                                                 d_work_counter, num_pixels);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));

    // Poll progress until kernel completes
    auto poll_start = std::chrono::steady_clock::now();
    while (cudaEventQuery(stop) == cudaErrorNotReady)
    {
        int done = *h_progress; // mapped memory: no cudaMemcpy needed
        bh_real pct = 100.0 * done / (bh_real)num_pixels;
        auto now = std::chrono::steady_clock::now();
        bh_real elapsed = std::chrono::duration<bh_real>(now - poll_start).count();
        bh_real rate = (elapsed > 0.01) ? done / elapsed : 0;
        bh_real eta = (rate > 0 && done < num_pixels) ? (num_pixels - done) / rate : 0;
        printf("\rGPU progress: %6.2f%% (%d / %d px)  %.1f Kpx/s  ETA %.1fs   ",
               pct, done, num_pixels, rate / 1e3, eta);
        fflush(stdout);
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    printf("\rGPU progress: 100.00%% (%d / %d px)                              \n",
           num_pixels, num_pixels);

    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("GPU kernel: %.2f seconds (%.1f Mpx/s)\n",
           ms / 1000.0, num_pixels / ms / 1000.0);

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(host_results, d_results, result_bytes, cudaMemcpyDeviceToHost));

    // Cleanup
    cudaFree(d_results);
    cudaFree(d_work_counter);
    cudaFreeHost(h_progress);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return true;
}
