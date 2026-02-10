#ifndef GPU_RENDER_H
#define GPU_RENDER_H

#include "physics.h"
#include "scene.h"

#include <algorithm>

// ============================================================================
// GPU Render Interface
//
// GPUSceneParams packs PhysicsParams + camera/rendering params for GPU
// constant memory.  The kernel outputs per-pixel GPUPixelResult — disk HDR
// emission plus the data needed for CPU-side sky mapping.
// ============================================================================

struct GPUSceneParams
{
    PhysicsParams physics;

    // Camera (rotation pre-computed on host)
    double cam_pos[3];
    double cam_right[3]; // rotation matrix column 0
    double cam_up[3];    // rotation matrix column 1
    double cam_fwd[3];   // rotation matrix column 2
    double fov_x, fov_y;

    // Ray integration
    double base_dt, max_affine, escape_r2;

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

// Populate GPUSceneParams from scene config, PhysicsParams, and camera rotation.
inline void fill_gpu_params(GPUSceneParams &p,
                            const scene_config_s &cfg,
                            const PhysicsParams &pp,
                            const dmat3 &cam_rot)
{
    p = {};
    p.physics = pp;

    // Camera
    p.cam_pos[0] = cfg.camera_x;
    p.cam_pos[1] = cfg.camera_y;
    p.cam_pos[2] = cfg.camera_z;
    dvec3 right = cam_rot.col(0);
    dvec3 up    = cam_rot.col(1);
    dvec3 fwd   = cam_rot.col(2);
    p.cam_right[0] = right.x; p.cam_right[1] = right.y; p.cam_right[2] = right.z;
    p.cam_up[0]    = up.x;    p.cam_up[1]    = up.y;    p.cam_up[2]    = up.z;
    p.cam_fwd[0]   = fwd.x;   p.cam_fwd[1]   = fwd.y;   p.cam_fwd[2]   = fwd.z;
    p.fov_x = cfg.fov_x;
    p.fov_y = cfg.fov_y;

    // Ray integration
    p.base_dt    = cfg.base_dt;
    p.max_affine = cfg.max_affine;
    p.escape_r2  = cfg.escape_radius * cfg.escape_radius;

    // Rendering
    p.aa_grid = std::max(cfg.aa_samples, 1);
    p.width   = cfg.output_width;
    p.height  = cfg.output_height;
}

#endif
