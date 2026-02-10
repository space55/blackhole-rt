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
// Render Kernel — one thread per pixel, AA loop inside
// ============================================================================
__global__ __launch_bounds__(256) void render_kernel(GPUPixelResult *results, int *progress)
{
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= c_params.width || py >= c_params.height)
        return;

    const int idx = py * c_params.width + px;

    // Local copy of physics params (fast reads from constant memory into registers)
    const PhysicsParams pp = c_params.physics;

    // Camera basis vectors (from constant memory)
    const dvec3 cam_right(c_params.cam_right[0], c_params.cam_right[1], c_params.cam_right[2]);
    const dvec3 cam_up(c_params.cam_up[0], c_params.cam_up[1], c_params.cam_up[2]);
    const dvec3 cam_fwd(c_params.cam_fwd[0], c_params.cam_fwd[1], c_params.cam_fwd[2]);
    const dvec3 cam_pos(c_params.cam_pos[0], c_params.cam_pos[1], c_params.cam_pos[2]);

    const int aa_grid = c_params.aa_grid;
    const double inv_aa = 1.0 / aa_grid;
    const double inv_spp = 1.0 / (double)(aa_grid * aa_grid);

    const double r_plus = pp.r_plus;
    const double base_dt = c_params.base_dt;
    const double max_affine = c_params.max_affine;
    const double escape_r2 = c_params.escape_r2;
    const double width_d = (double)c_params.width;
    const double height_d = (double)c_params.height;
    const double fov_x = c_params.fov_x;
    const double fov_y = c_params.fov_y;

    dvec3 pixel_disk(0, 0, 0);
    double pixel_sky_weight = 0;
    dvec3 pixel_exit_dir(0, 0, 0);

    for (int sy = 0; sy < aa_grid; sy++)
    {
        for (int sx = 0; sx < aa_grid; sx++)
        {
            const double sub_x = (px + (sx + 0.5) * inv_aa) / width_d;
            const double sub_y = (py + (sy + 0.5) * inv_aa) / height_d;

            // Initialize ray using shared physics
            dvec3 pos, vel;
            init_ray(pos, vel, cam_pos, cam_right, cam_up, cam_fwd,
                     sub_x, sub_y, fov_x, fov_y);

            dvec3 acc_color(0, 0, 0);
            double acc_opacity = 0;
            double cached_r = ks_radius(pos, pp.bh_spin);
            bool hit_bh = false;
            double affine = 0;

            while (affine < max_affine)
            {
                const double delta = fmax(cached_r - r_plus, 0.01);
                double step_dt = base_dt * dclamp(delta * delta, 0.0001, 1.0);

                // Step size reduction near disk
                if (cached_r >= pp.disk_inner_r * 0.8 && cached_r <= pp.disk_outer_r * 1.2)
                {
                    const double h = disk_half_thickness(cached_r, pp);
                    const double y_dist = fabs(pos.y);
                    if (y_dist < 5.0 * h)
                    {
                        step_dt = fmin(step_dt, fmax(0.3 * h, 0.005));
                    }
                }

                if (!advance_ray(pos, vel, cached_r, step_dt, pp))
                {
                    hit_bh = true;
                    break;
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

                sample_disk_volume(pos, vel, step_dt, acc_color, acc_opacity, pp);

                affine += step_dt;
                if (cached_r <= r_plus)
                {
                    hit_bh = true;
                    break;
                }
                if (pos.squaredNorm() > escape_r2)
                    break;
                if (acc_opacity > 0.99)
                    break;
            }

            pixel_disk += acc_color;
            if (!hit_bh && bh_isfinite(acc_opacity))
            {
                double transmittance = 1.0 - acc_opacity;
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
    auto is_finite_d = [](double v) -> bool {
        unsigned long long b = __double_as_longlong(v);
        return ((b >> 52) & 0x7FFull) != 0x7FFull;
    };

    if (!is_finite_d(pixel_sky_weight) || !is_finite_d(pixel_exit_dir.x) ||
        !is_finite_d(pixel_exit_dir.y) || !is_finite_d(pixel_exit_dir.z))
    {
        pixel_sky_weight = 0;
        pixel_exit_dir = dvec3(0, 0, 0);
    }

    // Disk components can be sanitised individually (no directional semantics)
    if (!is_finite_d(pixel_disk.x)) pixel_disk.x = 0;
    if (!is_finite_d(pixel_disk.y)) pixel_disk.y = 0;
    if (!is_finite_d(pixel_disk.z)) pixel_disk.z = 0;

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
    printf("GPU memory: %.0f MB total, double-precision throughput: %s\n",
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

    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((params.width + 15) / 16, (params.height + 15) / 16);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    render_kernel<<<grid, block>>>(d_results, d_progress);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));

    // Poll progress until kernel completes
    auto poll_start = std::chrono::steady_clock::now();
    while (cudaEventQuery(stop) == cudaErrorNotReady)
    {
        int done = *h_progress;  // mapped memory: no cudaMemcpy needed
        double pct = 100.0 * done / (double)num_pixels;
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - poll_start).count();
        double rate = (elapsed > 0.01) ? done / elapsed : 0;
        double eta = (rate > 0 && done < num_pixels) ? (num_pixels - done) / rate : 0;
        printf("\rGPU progress: %6.2f%% (%d / %d px)  %.1f Mpx/s  ETA %.1fs   ",
               pct, done, num_pixels, rate / 1e6, eta);
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
    cudaFreeHost(h_progress);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return true;
}
