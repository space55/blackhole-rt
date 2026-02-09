#ifndef GPU_RENDER_H
#define GPU_RENDER_H

#include "scene.h"
#include "blackhole.h"
#include "disk.h"

// ============================================================================
// GPU Render Interface
//
// All scene parameters are packed into GPUSceneParams (passed to GPU constant
// memory).  The kernel outputs per-pixel GPUPixelResult — disk HDR emission
// plus the data needed for CPU-side sky mapping.
// ============================================================================

struct GPUSceneParams
{
    // Camera (rotation pre-computed on host via Eigen)
    double cam_pos[3];
    double cam_right[3]; // rotation matrix column 0
    double cam_up[3];    // rotation matrix column 1
    double cam_fwd[3];   // rotation matrix column 2
    double fov_x, fov_y;

    // Black hole
    double bh_mass, bh_spin;
    double r_plus; // event horizon radius
    double r_isco; // innermost stable circular orbit

    // Ray integration
    double base_dt, max_affine, escape_r2;

    // Disk geometry
    double disk_inner_r, disk_outer_r;
    double disk_thickness, disk_density0, disk_opacity0;
    double disk_r_ref; // sqrt(inner_r * outer_r), precomputed

    // Disk appearance
    double disk_emission_boost, disk_color_variation;
    double disk_turbulence, disk_time;

    // Rendering
    int aa_grid;
    int width, height;
};

struct GPUPixelResult
{
    // Averaged HDR disk emission (before tone mapping)
    float disk_r, disk_g, disk_b;

    // Sky reconstruction data for CPU-side sky mapping:
    //   sky_weight  = average transmittance of escaped sub-samples
    //   exit_v*     = transmittance-weighted average exit velocity direction
    float sky_weight;
    float exit_vx, exit_vy, exit_vz;
};

// Launch the GPU render kernel.  Writes num_pixels results to host memory.
// Returns true on success, false on CUDA error.
bool gpu_render(const GPUSceneParams &params, GPUPixelResult *host_results);

// Populate GPUSceneParams from the scene config, black hole, and disk objects,
// using a precomputed camera rotation matrix (3×3, column-major Eigen Matrix3d).
// This eliminates the manual field-by-field copy in main().
inline void fill_gpu_params(GPUSceneParams &p,
                            const scene_config_s &cfg,
                            const blackhole_s &bh,
                            const accretion_disk_s &disk,
                            const Matrix3d &cam_rot)
{
    p = {};

    // Camera
    p.cam_pos[0] = cfg.camera_x;
    p.cam_pos[1] = cfg.camera_y;
    p.cam_pos[2] = cfg.camera_z;
    for (int i = 0; i < 3; i++)
    {
        p.cam_right[i] = cam_rot(i, 0);
        p.cam_up[i]    = cam_rot(i, 1);
        p.cam_fwd[i]   = cam_rot(i, 2);
    }
    p.fov_x = cfg.fov_x;
    p.fov_y = cfg.fov_y;

    // Black hole
    p.bh_mass = cfg.bh_mass;
    p.bh_spin = cfg.bh_spin;
    p.r_plus  = bh.event_horizon_radius();
    p.r_isco  = bh.isco_radius();

    // Ray integration
    p.base_dt    = cfg.base_dt;
    p.max_affine = cfg.max_affine;
    p.escape_r2  = cfg.escape_radius * cfg.escape_radius;

    // Disk geometry
    p.disk_inner_r  = disk.inner_r;
    p.disk_outer_r  = disk.outer_r;
    p.disk_thickness = cfg.disk_thickness;
    p.disk_density0  = cfg.disk_density;
    p.disk_opacity0  = cfg.disk_opacity;
    p.disk_r_ref     = sqrt(disk.inner_r * disk.outer_r);

    // Disk appearance
    p.disk_emission_boost  = cfg.disk_emission_boost;
    p.disk_color_variation = cfg.disk_color_variation;
    p.disk_turbulence      = cfg.disk_turbulence;
    p.disk_time            = cfg.time;

    // Rendering
    p.aa_grid = std::max(cfg.aa_samples, 1);
    p.width   = cfg.output_width;
    p.height  = cfg.output_height;
}

#endif
