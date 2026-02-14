// ============================================================================
// gpu_ghosts.h — CUDA-accelerated ghost rendering interface
//
// Flat, GPU-friendly data structures and host launch function.
// When FLARESIM_USE_CUDA is defined, render_ghosts_gpu() is available.
// The CPU fallback in ghost.cpp handles the non-CUDA case.
// ============================================================================
#pragma once

#include <vector>

// ---------------------------------------------------------------------------
// GPU-friendly flat structs (no pointers, no std::vector, POD)
// ---------------------------------------------------------------------------

// Matches Surface in lens.h but without member functions
struct GPUSurface
{
    float radius;        // signed radius of curvature (0 = flat)
    float thickness;     // axial distance to next surface
    float ior;           // d-line IOR of medium after this surface
    float abbe_v;        // Abbe number (0 = non-dispersive / air)
    float semi_aperture; // clear semi-diameter
    int coating;         // AR coating layers (0 = uncoated)
    int is_stop;         // 1 if aperture stop
    float z;             // axial position of surface vertex
};

struct GPUBrightPixel
{
    float angle_x, angle_y;
    float r, g, b;
};

struct GPUGhostPair
{
    int surf_a, surf_b;
    float area_boost; // per-pair area correction factor
};

// Per-ray trace result written by the GPU kernel
struct GPURayHit
{
    float px, py;  // pixel coordinates on sensor
    float value;   // contribution (source intensity × Fresnel × gain × ...)
    int channel;   // 0=R, 1=G, 2=B
    int source_idx; // which bright source (for adaptive splat grouping)
    int pair_idx;  // which ghost pair
};

// Configuration passed to the GPU kernel
struct GPUGhostConfig
{
    int ray_grid;            // N×N entrance pupil grid
    float wavelengths[3];    // R, G, B in nm
    float gain;              // ghost intensity multiplier
    float sensor_half_w;     // sensor half-width in mm
    float sensor_half_h;     // sensor half-height in mm
    float front_R;           // entrance pupil radius
    float start_z;           // z position for ray origins (before first surface)
    float ray_weight;        // 1.0 / valid_grid_count
    int img_width, img_height;
    int num_surfaces;
    int num_sources;
    int num_pairs;
    int valid_grid_count;
};

#ifdef FLARESIM_USE_CUDA

// Check if a CUDA-capable GPU is available at runtime.
bool gpu_ghosts_available();

// Launch CUDA ghost rendering.
//
// Traces all (pair × source × grid × wavelength) rays on GPU,
// then splats results back on CPU with adaptive tent filter.
//
// out_r/g/b: pre-allocated output buffers (width × height), will be ADDED to.
// Returns true on success, false on CUDA error (caller should fall back to CPU).
bool render_ghosts_gpu(const std::vector<GPUSurface> &surfaces,
                       const std::vector<GPUBrightPixel> &sources,
                       const std::vector<GPUGhostPair> &pairs,
                       const GPUGhostConfig &config,
                       float *out_r, float *out_g, float *out_b);

#endif // FLARESIM_USE_CUDA
