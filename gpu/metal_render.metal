// ============================================================================
// GPU Black Hole Ray Tracer — Metal Compute Kernel
//
// All physics (geodesic integration, disk emission, radiative transfer) lives
// in the shared header physics.h — compiled here as plain Metal functions via
// the BH_FUNC macro (empty under __METAL_VERSION__).
//
// This file contains only the Metal compute kernel.  The host launch code is
// in metal_render.mm.
// ============================================================================

// Metal forces float precision — BH_USE_FLOAT is always active.
// (Apple Silicon has no fp64 compute capability.)
#ifndef BH_USE_FLOAT
#define BH_USE_FLOAT
#endif

#include "../inc/physics.h"

// ============================================================================
// Shared struct layout — must match GPUSceneParams / GPUPixelResult on host.
// We re-declare them here because Metal cannot include C++ headers with STL.
// The layout is trivially POD and kept in sync via static_asserts on the host.
// ============================================================================

struct MetalSceneParams
{
    PhysicsParams physics;

    // Camera (rotation pre-computed on host)
    float cam_pos[3];
    float cam_right[3];
    float cam_up[3];
    float cam_fwd[3];
    float fov_x, fov_y;

    // Ray integration
    float base_dt, max_affine, escape_r2;
    int max_iter;

    // Rendering
    int aa_grid;
    int width, height;
};

struct MetalPixelResult
{
    float disk_r, disk_g, disk_b;
    float sky_weight;
    float exit_vx, exit_vy, exit_vz;
};

// ============================================================================
// Render Kernel — persistent-thread work-stealing design
//
// Metal equivalent of the CUDA render_kernel.  Each thread grabs pixels from
// a global atomic counter, identical to the CUDA persistent-thread pattern.
// ============================================================================
kernel void render_kernel(
    device MetalPixelResult *results        [[buffer(0)]],
    device atomic_int       *progress       [[buffer(1)]],
    device atomic_int       *work_counter   [[buffer(2)]],
    constant MetalSceneParams &c_params     [[buffer(3)]],
    constant int            &total_pixels   [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    // Local copy of physics params
    const PhysicsParams pp = c_params.physics;

    // Camera basis vectors
    const dvec3 cam_right(c_params.cam_right[0], c_params.cam_right[1], c_params.cam_right[2]);
    const dvec3 cam_up(c_params.cam_up[0], c_params.cam_up[1], c_params.cam_up[2]);
    const dvec3 cam_fwd(c_params.cam_fwd[0], c_params.cam_fwd[1], c_params.cam_fwd[2]);
    const dvec3 cam_pos(c_params.cam_pos[0], c_params.cam_pos[1], c_params.cam_pos[2]);

    const int width = c_params.width;
    const int aa_grid = c_params.aa_grid;
    const bh_real inv_aa = 1.0f / aa_grid;
    const bh_real inv_spp = 1.0f / (bh_real)(aa_grid * aa_grid);

    const bh_real r_plus = pp.r_plus;
    const bh_real base_dt = c_params.base_dt;
    const bh_real max_affine = c_params.max_affine;
    const bh_real escape_r2 = c_params.escape_r2;
    const bh_real width_d = (bh_real)c_params.width;
    const bh_real height_d = (bh_real)c_params.height;
    const bh_real fov_x = c_params.fov_x;
    const bh_real fov_y = c_params.fov_y;

    const int max_iter = c_params.max_iter;

    // Persistent work loop — each thread grabs one pixel at a time
    while (true)
    {
        const int idx = atomic_fetch_add_explicit(work_counter, 1, memory_order_relaxed);
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
                const bh_real sub_x = (px + (sx + 0.5f) * inv_aa) / width_d;
                const bh_real sub_y = (py + (sy + 0.5f) * inv_aa) / height_d;

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
                    const bh_real delta = bh_fmax(cached_r - r_plus, 0.01f);
                    bh_real step_dt = base_dt * dclamp(delta * delta, 0.001f, 1.0f);

                    // Step size reduction near disk
                    const bh_real disk_inner_guard = pp.disk_flat_mode ? pp.disk_inner_r * 0.4f : pp.disk_inner_r * 0.8f;
                    const bh_real disk_outer_guard = pp.disk_flat_mode ? pp.disk_outer_r * 1.6f : pp.disk_outer_r * 1.2f;
                    if (cached_r >= disk_inner_guard && cached_r <= disk_outer_guard)
                    {
                        const bh_real h = disk_half_thickness(cached_r, pp);
                        const bh_real y_dist = bh_fabs(pos.y);
                        if (y_dist < 5.0f * h)
                        {
                            step_dt = bh_fmin(step_dt, bh_fmax(0.15f * h, 0.001f));

                            const bh_real v_horiz_sq = vel.x * vel.x + vel.z * vel.z;
                            const bh_real v_vert_sq = vel.y * vel.y;
                            if (v_horiz_sq > 4.0f * v_vert_sq)
                            {
                                const bh_real v_horiz = sqrt(v_horiz_sq);
                                const bh_real texture_scale = pp.disk_flat_mode ? 0.15f : 0.3f;
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
                    if (pp.disk_flat_mode && prev_y * pos.y < 0.0f)
                    {
                        if (cached_r >= (pp.disk_inner_r * 0.4f) &&
                            cached_r <= (pp.disk_outer_r * 1.6f))
                        {
                            const bh_real t_cross = bh_fabs(prev_y) /
                                                    bh_fmax(bh_fabs(prev_y) + bh_fabs(pos.y), 1e-12f);
                            dvec3 mid_pos = (1.0f - t_cross) * (pos - step_dt * vel) +
                                            t_cross * pos;
                            mid_pos.y = 0.0f;
                            sample_disk_volume(mid_pos, vel, step_dt * 0.5f,
                                               acc_color, acc_opacity, cached_r, pp);
                        }
                    }

                    // Validate ray state
                    if (!bh_isfinite(vel.squaredNorm()) || !bh_isfinite(cached_r))
                    {
                        hit_bh = true;
                        break;
                    }

                    // Disk sampling with cheap vertical guard
                    const bh_real samp_inner_guard = pp.disk_flat_mode ? pp.disk_inner_r * 0.4f : pp.disk_inner_r * 0.8f;
                    const bh_real samp_outer_guard = pp.disk_flat_mode ? pp.disk_outer_r * 1.6f : pp.disk_outer_r * 1.3f;
                    if (bh_fabs(pos.y) < pp.disk_thickness * 15.0f &&
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
                    if (acc_opacity > 0.95f)
                        break;
                }

                pixel_disk += acc_color;
                if (!hit_bh && bh_isfinite(acc_opacity))
                {
                    bh_real transmittance = 1.0f - acc_opacity;
                    pixel_sky_weight += transmittance;
                    pixel_exit_dir += transmittance * vel;
                }
            }
        }

        pixel_disk *= inv_spp;
        pixel_sky_weight *= inv_spp;
        pixel_exit_dir *= inv_spp;

        // Sanitise outputs
        if (!bh_isfinite(pixel_sky_weight) || !bh_isfinite(pixel_exit_dir.x) ||
            !bh_isfinite(pixel_exit_dir.y) || !bh_isfinite(pixel_exit_dir.z))
        {
            pixel_sky_weight = 0;
            pixel_exit_dir = dvec3(0, 0, 0);
        }

        if (!bh_isfinite(pixel_disk.x))
            pixel_disk.x = 0;
        if (!bh_isfinite(pixel_disk.y))
            pixel_disk.y = 0;
        if (!bh_isfinite(pixel_disk.z))
            pixel_disk.z = 0;

        results[idx].disk_r = pixel_disk.x;
        results[idx].disk_g = pixel_disk.y;
        results[idx].disk_b = pixel_disk.z;
        results[idx].sky_weight = pixel_sky_weight;
        results[idx].exit_vx = pixel_exit_dir.x;
        results[idx].exit_vy = pixel_exit_dir.y;
        results[idx].exit_vz = pixel_exit_dir.z;

        // Update progress counter
        atomic_fetch_add_explicit(progress, 1, memory_order_relaxed);
    }
}
