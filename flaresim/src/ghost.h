// ============================================================================
// ghost.h — Ghost reflection enumeration and rendering
// ============================================================================
#pragma once

#include "lens.h"
#include "vec3.h"

#include <vector>

// A ghost bounce pair: surfaces where light reflects instead of transmitting.
struct GhostPair
{
    int surf_a; // first bounce surface (closer to front)
    int surf_b; // second bounce surface (closer to sensor)
};

// A bright pixel extracted from the input image.
struct BrightPixel
{
    float angle_x; // horizontal angle from optical axis (radians)
    float angle_y; // vertical angle from optical axis (radians)
    float r, g, b; // HDR intensity
};

// Configuration for the ghost renderer.
struct GhostConfig
{
    int ray_grid = 64;                               // samples per dimension across entrance pupil
    float min_intensity = 1e-7f;                     // skip ghost pairs dimmer than this
    float gain = 1000.0f;                            // ghost intensity multiplier
    float wavelengths[3] = {650.0f, 550.0f, 450.0f}; // R, G, B in nm

    // Per-pair area normalization: boost defocused ghost pairs so they remain
    // visible.  Production renderers (ILM, Weta) use a similar technique.
    bool ghost_normalize = true;   // enable per-pair area correction
    float max_area_boost = 100.0f; // clamp the correction factor
};

// Enumerate all valid ghost bounce pairs for the lens system.
// Returns C(N, 2) pairs where N = number of surfaces.
std::vector<GhostPair> enumerate_ghost_pairs(const LensSystem &lens);

// Render all ghost reflections onto the output flare image.
//
// For each ghost pair, traces rays from each bright source through the
// lens system with two reflections, accumulating contributions on the
// output image via bilinear splatting.
//
// out_r/g/b: pre-allocated output buffers (width × height), zeroed.
// fov_h/fov_v: horizontal and vertical FOV in radians.
void render_ghosts(const LensSystem &lens,
                   const std::vector<BrightPixel> &sources,
                   float fov_h, float fov_v,
                   float *out_r, float *out_g, float *out_b,
                   int width, int height,
                   const GhostConfig &config);
