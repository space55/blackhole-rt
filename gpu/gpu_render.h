#ifndef GPU_RENDER_H
#define GPU_RENDER_H

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

#endif
