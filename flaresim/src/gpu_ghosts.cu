// ============================================================================
// gpu_ghosts.cu — CUDA ghost ray tracing kernel + GPU splatting
//
// Architecture:
//   1. Upload lens, sources, pairs to device memory
//   2. Trace kernel: one thread per (pair, source, grid_x, grid_y)
//      - traces 3 wavelengths, writes hits to compacted output via atomicAdd
//   3. Compute per-group adaptive splat radii on GPU (no sorting — uses
//      pair_idx × num_sources + source_idx as a perfect hash key)
//   4. Tile-based splatting:
//      a. Bin hits into image tiles based on footprint overlap
//      b. One thread block per tile, accumulate in shared memory
//      c. Write tile to global (no atomics — tiles are disjoint)
//   5. Download output image to host
//
// Everything stays on-device — no hit buffer download required.
// ============================================================================

#include "gpu_ghosts.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <vector>
#include <chrono>
#include <thread>

// ============================================================================
// CUDA error checking
// ============================================================================
#define CUDA_CHECK(call)                                          \
    do                                                            \
    {                                                             \
        cudaError_t err = (call);                                 \
        if (err != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            return false;                                         \
        }                                                         \
    } while (0)

// ============================================================================
// Device-side math helpers
// ============================================================================

struct DVec3
{
    float x, y, z;
    __device__ DVec3() : x(0), y(0), z(0) {}
    __device__ DVec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    __device__ DVec3 operator+(const DVec3 &v) const { return {x + v.x, y + v.y, z + v.z}; }
    __device__ DVec3 operator-(const DVec3 &v) const { return {x - v.x, y - v.y, z - v.z}; }
    __device__ DVec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    __device__ DVec3 operator-() const { return {-x, -y, -z}; }
    __device__ float length_sq() const { return x * x + y * y + z * z; }
    __device__ float length() const { return sqrtf(length_sq()); }
    __device__ DVec3 normalized() const
    {
        float l = length();
        return (l > 1e-12f) ? DVec3(x / l, y / l, z / l) : DVec3(0, 0, 0);
    }
};

__device__ float d_dot(const DVec3 &a, const DVec3 &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// ============================================================================
// Device-side Fresnel / dispersion (mirrors fresnel.h)
// ============================================================================

__device__ float d_dispersion_ior(float n_d, float V_d, float lambda_nm)
{
    if (V_d < 0.1f || n_d <= 1.0001f)
        return n_d;

    constexpr float lF = 486.13f, lC = 656.27f, ld = 587.56f;
    float dn = (n_d - 1.0f) / V_d;
    float inv_lF2 = 1.0f / (lF * lF);
    float inv_lC2 = 1.0f / (lC * lC);
    float inv_ld2 = 1.0f / (ld * ld);
    float B = dn / (inv_lF2 - inv_lC2);
    float A = n_d - B * inv_ld2;
    return A + B / (lambda_nm * lambda_nm);
}

__device__ float d_fresnel_reflectance(float cos_i, float n1, float n2)
{
    cos_i = fabsf(cos_i);
    float eta = n1 / n2;
    float sin2_t = eta * eta * (1.0f - cos_i * cos_i);
    if (sin2_t >= 1.0f)
        return 1.0f;
    float cos_t = sqrtf(1.0f - sin2_t);
    float rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t);
    float rp = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t);
    return 0.5f * (rs * rs + rp * rp);
}

__device__ float d_coating_reflectance(float cos_i, float n1, float n2,
                                       float coating_n, float d_nm, float lambda_nm)
{
    float sin2_c = (n1 / coating_n) * (n1 / coating_n) * (1.0f - cos_i * cos_i);
    if (sin2_c >= 1.0f)
        return d_fresnel_reflectance(cos_i, n1, n2);
    float cos_c = sqrtf(1.0f - sin2_c);

    constexpr float PI = 3.14159265358979323846f;
    float delta = 2.0f * PI * coating_n * d_nm * cos_c / lambda_nm;

    float r01 = (n1 * cos_i - coating_n * cos_c) / (n1 * cos_i + coating_n * cos_c);

    float sin2_2 = (coating_n / n2) * (coating_n / n2) * (1.0f - cos_c * cos_c);
    if (sin2_2 >= 1.0f)
        return d_fresnel_reflectance(cos_i, n1, n2);
    float cos_2 = sqrtf(1.0f - sin2_2);
    float r12 = (coating_n * cos_c - n2 * cos_2) / (coating_n * cos_c + n2 * cos_2);

    float cos_2delta = cosf(2.0f * delta);
    float num = r01 * r01 + r12 * r12 + 2.0f * r01 * r12 * cos_2delta;
    float den = 1.0f + r01 * r01 * r12 * r12 + 2.0f * r01 * r12 * cos_2delta;
    float result = num / den;
    return fminf(fmaxf(result, 0.0f), 1.0f);
}

__device__ float d_surface_reflectance(float cos_i, float n1, float n2,
                                       int coating_layers, float lambda_nm)
{
    if (coating_layers <= 0)
        return d_fresnel_reflectance(cos_i, n1, n2);

    constexpr float mgf2_n = 1.38f;
    constexpr float design_lambda = 550.0f;
    float qw_thickness = design_lambda / (4.0f * mgf2_n);

    float R = d_coating_reflectance(cos_i, n1, n2, mgf2_n, qw_thickness, lambda_nm);
    for (int i = 1; i < coating_layers; ++i)
        R *= 0.25f;
    return fminf(fmaxf(R, 0.0f), 1.0f);
}

// ============================================================================
// Device-side surface IOR helpers
// ============================================================================

__device__ float d_surf_ior_at(const GPUSurface &s, float lambda_nm)
{
    return d_dispersion_ior(s.ior, s.abbe_v, lambda_nm);
}

__device__ float d_ior_before(const GPUSurface *surfaces, int idx, float lambda_nm)
{
    if (idx <= 0)
        return 1.0f;
    return d_surf_ior_at(surfaces[idx - 1], lambda_nm);
}

__device__ float d_ior_before_d(const GPUSurface *surfaces, int idx)
{
    if (idx <= 0)
        return 1.0f;
    return surfaces[idx - 1].ior;
}

// ============================================================================
// Device-side ray–surface intersection (mirrors trace.cpp)
// ============================================================================

struct DRay
{
    DVec3 origin, dir;
};

__device__ bool d_intersect_surface(const DRay &ray, const GPUSurface &surf,
                                    DVec3 &hit_pos, DVec3 &normal)
{
    if (fabsf(surf.radius) < 1e-6f)
    {
        if (fabsf(ray.dir.z) < 1e-12f)
            return false;
        float t = (surf.z - ray.origin.z) / ray.dir.z;
        if (t < 1e-6f)
            return false;
        hit_pos = ray.origin + ray.dir * t;
        float h2 = hit_pos.x * hit_pos.x + hit_pos.y * hit_pos.y;
        if (h2 > surf.semi_aperture * surf.semi_aperture)
            return false;
        normal = DVec3(0, 0, (ray.dir.z > 0) ? -1.0f : 1.0f);
        return true;
    }

    float R = surf.radius;
    DVec3 center(0, 0, surf.z + R);
    DVec3 oc = ray.origin - center;
    float a = d_dot(ray.dir, ray.dir);
    float b = 2.0f * d_dot(oc, ray.dir);
    float c = d_dot(oc, oc) - R * R;
    float disc = b * b - 4.0f * a * c;
    if (disc < 0)
        return false;

    float sqrt_disc = sqrtf(disc);
    float inv_2a = 0.5f / a;
    float t1 = (-b - sqrt_disc) * inv_2a;
    float t2 = (-b + sqrt_disc) * inv_2a;

    float t;
    if (t1 > 1e-6f && t2 > 1e-6f)
    {
        float z1 = ray.origin.z + t1 * ray.dir.z;
        float z2 = ray.origin.z + t2 * ray.dir.z;
        t = (fabsf(z1 - surf.z) < fabsf(z2 - surf.z)) ? t1 : t2;
    }
    else if (t1 > 1e-6f)
        t = t1;
    else if (t2 > 1e-6f)
        t = t2;
    else
        return false;

    hit_pos = ray.origin + ray.dir * t;
    float h2 = hit_pos.x * hit_pos.x + hit_pos.y * hit_pos.y;
    if (h2 > surf.semi_aperture * surf.semi_aperture)
        return false;

    normal = (hit_pos - center) * (1.0f / fabsf(R));
    if (d_dot(normal, ray.dir) > 0)
        normal = -normal;
    return true;
}

// ============================================================================
// Device-side refraction / reflection
// ============================================================================

__device__ bool d_refract(const DVec3 &dir, const DVec3 &normal, float n_ratio, DVec3 &out)
{
    float cos_i = -d_dot(normal, dir);
    float sin2_t = n_ratio * n_ratio * (1.0f - cos_i * cos_i);
    if (sin2_t >= 1.0f)
        return false;
    float cos_t = sqrtf(1.0f - sin2_t);
    out = (dir * n_ratio + normal * (n_ratio * cos_i - cos_t)).normalized();
    return true;
}

__device__ DVec3 d_reflect(const DVec3 &dir, const DVec3 &normal)
{
    return (dir - normal * (2.0f * d_dot(dir, normal))).normalized();
}

// ============================================================================
// Device-side trace_ghost_ray (mirrors trace.cpp exactly)
// ============================================================================

struct DTraceResult
{
    DVec3 position;
    float weight;
    bool valid;
};

__device__ DTraceResult d_trace_ghost_ray(DRay ray, const GPUSurface *surfaces,
                                          int num_surfaces, float sensor_z,
                                          int bounce_a, int bounce_b,
                                          float lambda_nm)
{
    DTraceResult result;
    result.valid = false;
    result.weight = 1.0f;

    float current_ior = 1.0f;

    // Phase 1: forward 0..bounce_b
    for (int s = 0; s <= bounce_b; ++s)
    {
        DVec3 hit, norm;
        if (!d_intersect_surface(ray, surfaces[s], hit, norm))
            return result;

        ray.origin = hit;
        float n1 = current_ior;
        float n2 = d_surf_ior_at(surfaces[s], lambda_nm);
        float cos_i = fabsf(d_dot(norm, ray.dir));
        float R = d_surface_reflectance(cos_i, n1, n2, surfaces[s].coating, lambda_nm);

        if (s == bounce_b)
        {
            ray.dir = d_reflect(ray.dir, norm);
            result.weight *= R;
        }
        else
        {
            DVec3 new_dir;
            if (!d_refract(ray.dir, norm, n1 / n2, new_dir))
                return result;
            ray.dir = new_dir;
            result.weight *= (1.0f - R);
            current_ior = n2;
        }
    }

    // Phase 2: backward bounce_b-1..bounce_a
    for (int s = bounce_b - 1; s >= bounce_a; --s)
    {
        DVec3 hit, norm;
        if (!d_intersect_surface(ray, surfaces[s], hit, norm))
            return result;

        ray.origin = hit;
        float n1 = current_ior;
        float n2 = d_ior_before(surfaces, s, lambda_nm);
        float cos_i = fabsf(d_dot(norm, ray.dir));
        float R = d_surface_reflectance(cos_i, n1, n2, surfaces[s].coating, lambda_nm);

        if (s == bounce_a)
        {
            ray.dir = d_reflect(ray.dir, norm);
            result.weight *= R;
            current_ior = d_surf_ior_at(surfaces[bounce_a], lambda_nm);
        }
        else
        {
            DVec3 new_dir;
            if (!d_refract(ray.dir, norm, n1 / n2, new_dir))
                return result;
            ray.dir = new_dir;
            result.weight *= (1.0f - R);
            current_ior = n2;
        }
    }

    // Phase 3: forward bounce_a+1..N-1
    for (int s = bounce_a + 1; s < num_surfaces; ++s)
    {
        DVec3 hit, norm;
        if (!d_intersect_surface(ray, surfaces[s], hit, norm))
            return result;

        ray.origin = hit;
        float n1 = current_ior;
        float n2 = d_surf_ior_at(surfaces[s], lambda_nm);
        float cos_i = fabsf(d_dot(norm, ray.dir));
        float R = d_surface_reflectance(cos_i, n1, n2, surfaces[s].coating, lambda_nm);

        DVec3 new_dir;
        if (!d_refract(ray.dir, norm, n1 / n2, new_dir))
            return result;
        ray.dir = new_dir;
        result.weight *= (1.0f - R);
        current_ior = n2;
    }

    // Propagate to sensor
    if (fabsf(ray.dir.z) < 1e-12f)
        return result;
    float t = (sensor_z - ray.origin.z) / ray.dir.z;
    if (t < 0)
        return result;

    result.position = ray.origin + ray.dir * t;
    result.valid = true;
    return result;
}

// ============================================================================
// Ghost trace kernel
//
// Thread mapping: one thread per (pair_idx, source_idx, grid_x, grid_y)
// Each thread traces 3 wavelengths and writes up to 3 hits via atomic append.
// ============================================================================

__global__ void ghost_trace_kernel(
    const GPUSurface *surfaces,
    const GPUBrightPixel *sources,
    const GPUGhostPair *pairs,
    const int2 *grid_lut,
    GPURayHit *hit_buffer,
    int *hit_count,
    int max_hits,
    int *progress,
    GPUGhostConfig config)
{
    int total_work = config.num_pairs * config.num_sources * config.valid_grid_count;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_work)
        return;

    // Decompose flat index → (pair, source, grid_sample)
    int grid_sample = idx % config.valid_grid_count;
    int remainder = idx / config.valid_grid_count;
    int source_idx = remainder % config.num_sources;
    int pair_idx = remainder / config.num_sources;

    if (pair_idx >= config.num_pairs)
        return;

    int a = pairs[pair_idx].surf_a;
    int b = pairs[pair_idx].surf_b;
    float area_boost = pairs[pair_idx].area_boost;

    const GPUBrightPixel &src = sources[source_idx];
    int N = config.ray_grid;

    // O(1) grid_sample → (gx, gy) via precomputed lookup table
    int gx = grid_lut[grid_sample].x;
    int gy = grid_lut[grid_sample].y;

    // Simple hash-based jitter (deterministic, matches CPU behavior intent)
    unsigned int seed = (unsigned int)(source_idx * 7919 + a * 131 + b * 1031 + gx * 97 + gy * 53);
    seed ^= seed >> 16;
    seed *= 0x45d9f3b;
    seed ^= seed >> 16;
    float jx = (seed & 0xFFFF) / 65536.0f;
    seed *= 0x45d9f3b;
    seed ^= seed >> 16;
    float jy = (seed & 0xFFFF) / 65536.0f;

    float u = ((gx + jx) / N) * 2.0f - 1.0f;
    float v = ((gy + jy) / N) * 2.0f - 1.0f;
    if (u * u + v * v > 1.0f)
        return; // jitter pushed us outside aperture

    // Beam direction for this source
    float bx = tanf(src.angle_x);
    float by = tanf(src.angle_y);
    DVec3 beam_dir = DVec3(bx, by, 1.0f).normalized();

    DRay ray;
    ray.origin = DVec3(u * config.front_R, v * config.front_R, config.start_z);
    ray.dir = beam_dir;

    // Sensor z = last surface z + last surface thickness
    float sensor_z = surfaces[config.num_surfaces - 1].z +
                     surfaces[config.num_surfaces - 1].thickness;

    // Trace 3 wavelengths
    for (int ch = 0; ch < 3; ++ch)
    {
        DTraceResult res = d_trace_ghost_ray(ray, surfaces, config.num_surfaces,
                                             sensor_z, a, b, config.wavelengths[ch]);
        if (!res.valid)
            continue;

        float px = (res.position.x / (2.0f * config.sensor_half_w) + 0.5f) * config.img_width;
        float py = (res.position.y / (2.0f * config.sensor_half_h) + 0.5f) * config.img_height;

        float src_i = (ch == 0) ? src.r : (ch == 1) ? src.g
                                                    : src.b;
        float contribution = src_i * res.weight * config.ray_weight *
                             config.gain * area_boost;
        if (contribution < 1e-12f)
            continue;

        // Atomic append to hit buffer
        int slot = atomicAdd(hit_count, 1);
        if (slot < max_hits)
        {
            GPURayHit &h = hit_buffer[slot];
            h.px = px;
            h.py = py;
            h.value = contribution;
            h.channel = ch;
            h.source_idx = source_idx;
            h.pair_idx = pair_idx;
        }
    }

    // Update progress (one per thread regardless of hit/miss)
    atomicAdd(progress, 1);
}

// ============================================================================
// GPU splatting kernels
//
// Per-group adaptive radii are computed without sorting: we use
//   group_key = pair_idx * num_sources + source_idx
// as a perfect hash into fixed-size per-group arrays.  A single pass
// with atomicMin/Max/Add accumulates green-channel bounding boxes,
// then a tiny kernel converts those to adaptive tent-filter radii.
// Finally, hits are binned into image tiles and splatted using
// shared-memory accumulation (one thread block per tile).
// ============================================================================

// CAS-based atomicMin/Max for float (works for all float values)
__device__ float d_atomicMinFloat(float *addr, float val)
{
    int *addr_i = (int *)addr;
    int old = *addr_i, assumed;
    do
    {
        assumed = old;
        if (__int_as_float(assumed) <= val)
            return __int_as_float(assumed);
        old = atomicCAS(addr_i, assumed, __float_as_int(val));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ float d_atomicMaxFloat(float *addr, float val)
{
    int *addr_i = (int *)addr;
    int old = *addr_i, assumed;
    do
    {
        assumed = old;
        if (__int_as_float(assumed) >= val)
            return __int_as_float(assumed);
        old = atomicCAS(addr_i, assumed, __float_as_int(val));
    } while (assumed != old);
    return __int_as_float(old);
}

// Initialize per-group stats arrays
__global__ void init_group_stats_kernel(float *min_x, float *max_x,
                                        float *min_y, float *max_y,
                                        int *green_count, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    min_x[i] = 1e30f;
    max_x[i] = -1e30f;
    min_y[i] = 1e30f;
    max_y[i] = -1e30f;
    green_count[i] = 0;
}

// One pass: accumulate green-channel bbox per group
__global__ void accumulate_group_stats_kernel(const GPURayHit *hits, int num_hits,
                                              int num_sources,
                                              float *min_x, float *max_x,
                                              float *min_y, float *max_y,
                                              int *green_count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_hits)
        return;
    if (hits[i].channel != 1)
        return; // only green contributes to bbox
    int gkey = hits[i].pair_idx * num_sources + hits[i].source_idx;
    d_atomicMinFloat(&min_x[gkey], hits[i].px);
    d_atomicMaxFloat(&max_x[gkey], hits[i].px);
    d_atomicMinFloat(&min_y[gkey], hits[i].py);
    d_atomicMaxFloat(&max_y[gkey], hits[i].py);
    atomicAdd(&green_count[gkey], 1);
}

// Compute adaptive tent-filter radius per group
__global__ void compute_radii_kernel(float *radii,
                                     const float *min_x, const float *max_x,
                                     const float *min_y, const float *max_y,
                                     const int *green_count, int max_groups)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= max_groups)
        return;
    int gc = green_count[g];
    float r = 1.5f;
    if (gc >= 4)
    {
        float ex = max_x[g] - min_x[g];
        float ey = max_y[g] - min_y[g];
        float extent = fmaxf(fmaxf(ex, ey), 1.0f);
        float spacing = extent / sqrtf((float)gc);
        r = fminf(fmaxf(spacing * 1.2f, 1.5f), 80.0f);
    }
    radii[g] = r;
}

// ============================================================================
// Tile-based splatting
//
// Instead of one-thread-per-hit doing atomicAdd on global memory (extreme
// contention for spatially-clustered ghost hits), we:
//
//   1. Bin hits into image tiles based on their footprint overlap
//   2. One thread block per tile accumulates in shared memory
//   3. Write tile to global memory (no atomics — tiles are disjoint)
//
// Shared-memory atomicAdd is ~50× faster than global for float, and the
// final write-back is a plain store since each pixel belongs to exactly
// one tile.
// ============================================================================

#define SPLAT_TILE_W 16
#define SPLAT_TILE_H 16
#define SPLAT_TILE_PX (SPLAT_TILE_W * SPLAT_TILE_H)

// Helper: compute the tile range a hit's footprint overlaps
__device__ void d_hit_tile_range(float px, float py, float radius,
                                 int tiles_x, int tiles_y, int img_w, int img_h,
                                 int &tx0, int &tx1, int &ty0, int &ty1)
{
    float r = fmaxf(radius, 1.0f);
    int px0 = max((int)floorf(px - r), 0);
    int px1 = min((int)ceilf(px + r), img_w - 1);
    int py0 = max((int)floorf(py - r), 0);
    int py1 = min((int)ceilf(py + r), img_h - 1);
    tx0 = max(px0 / SPLAT_TILE_W, 0);
    tx1 = min(px1 / SPLAT_TILE_W, tiles_x - 1);
    ty0 = max(py0 / SPLAT_TILE_H, 0);
    ty1 = min(py1 / SPLAT_TILE_H, tiles_y - 1);
}

// Pass 1: count how many hits land in each tile
__global__ void count_tile_hits_kernel(const GPURayHit *hits, int num_hits,
                                       int num_sources, const float *group_radii,
                                       int *tile_counts,
                                       int tiles_x, int tiles_y,
                                       int img_w, int img_h)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_hits)
        return;

    const GPURayHit &h = hits[i];
    int gkey = h.pair_idx * num_sources + h.source_idx;
    float radius = group_radii[gkey];

    int tx0, tx1, ty0, ty1;
    d_hit_tile_range(h.px, h.py, radius, tiles_x, tiles_y, img_w, img_h,
                     tx0, tx1, ty0, ty1);

    for (int ty = ty0; ty <= ty1; ++ty)
        for (int tx = tx0; tx <= tx1; ++tx)
            atomicAdd(&tile_counts[ty * tiles_x + tx], 1);
}

// Pass 2: scatter hit indices into per-tile bins
__global__ void fill_tile_bins_kernel(const GPURayHit *hits, int num_hits,
                                      int num_sources, const float *group_radii,
                                      const int *tile_offsets, int *tile_cursors,
                                      int *tile_bins,
                                      int tiles_x, int tiles_y,
                                      int img_w, int img_h)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_hits)
        return;

    const GPURayHit &h = hits[i];
    int gkey = h.pair_idx * num_sources + h.source_idx;
    float radius = group_radii[gkey];

    int tx0, tx1, ty0, ty1;
    d_hit_tile_range(h.px, h.py, radius, tiles_x, tiles_y, img_w, img_h,
                     tx0, tx1, ty0, ty1);

    for (int ty = ty0; ty <= ty1; ++ty)
        for (int tx = tx0; tx <= tx1; ++tx)
        {
            int tidx = ty * tiles_x + tx;
            int slot = atomicAdd(&tile_cursors[tidx], 1);
            tile_bins[tile_offsets[tidx] + slot] = i;
        }
}

// Pass 3: one thread-block per tile, shared-memory accumulation
//
// Each block owns a SPLAT_TILE_W × SPLAT_TILE_H region of the output.
// Threads cooperatively iterate over all hits binned to this tile,
// computing tent/bilinear weights and accumulating via fast shared-memory
// atomicAdd.  The final write-back to global memory is a plain store.
__global__ void splat_tiles_kernel(const GPURayHit *hits,
                                   const int *tile_offsets, const int *tile_bins,
                                   int num_sources, const float *group_radii,
                                   float *out_r, float *out_g, float *out_b,
                                   int tiles_x, int tiles_y,
                                   int img_w, int img_h)
{
    __shared__ float s_r[SPLAT_TILE_PX];
    __shared__ float s_g[SPLAT_TILE_PX];
    __shared__ float s_b[SPLAT_TILE_PX];

    int tile_idx = blockIdx.x;
    if (tile_idx >= tiles_x * tiles_y)
        return;

    int tile_tx = tile_idx % tiles_x;
    int tile_ty = tile_idx / tiles_x;
    int tile_x0 = tile_tx * SPLAT_TILE_W;
    int tile_y0 = tile_ty * SPLAT_TILE_H;

    // Zero shared memory
    for (int i = threadIdx.x; i < SPLAT_TILE_PX; i += blockDim.x)
    {
        s_r[i] = 0.0f;
        s_g[i] = 0.0f;
        s_b[i] = 0.0f;
    }
    __syncthreads();

    int begin = tile_offsets[tile_idx];
    int end = tile_offsets[tile_idx + 1];

    // Cooperatively process all hits assigned to this tile
    for (int j = begin + threadIdx.x; j < end; j += blockDim.x)
    {
        int hi = tile_bins[j];
        const GPURayHit &h = hits[hi];
        int gkey = h.pair_idx * num_sources + h.source_idx;
        float radius = group_radii[gkey];
        float *s_buf = (h.channel == 0) ? s_r : (h.channel == 1) ? s_g
                                                                 : s_b;

        if (radius <= 1.5f)
        {
            // Bilinear splat — up to 4 pixels
            int x0 = (int)floorf(h.px - 0.5f);
            int y0 = (int)floorf(h.py - 0.5f);
            float fx = (h.px - 0.5f) - x0;
            float fy = (h.py - 0.5f) - y0;
            float w00 = (1.0f - fx) * (1.0f - fy);
            float w10 = fx * (1.0f - fy);
            float w01 = (1.0f - fx) * fy;
            float w11 = fx * fy;

            for (int dy = 0; dy <= 1; ++dy)
            {
                int gy = y0 + dy;
                if (gy < tile_y0 || gy >= tile_y0 + SPLAT_TILE_H ||
                    gy < 0 || gy >= img_h)
                    continue;
                int ly = gy - tile_y0;
                for (int dx = 0; dx <= 1; ++dx)
                {
                    int gx = x0 + dx;
                    if (gx < tile_x0 || gx >= tile_x0 + SPLAT_TILE_W ||
                        gx < 0 || gx >= img_w)
                        continue;
                    int lx = gx - tile_x0;
                    float w = (dx == 0) ? ((dy == 0) ? w00 : w01)
                                        : ((dy == 0) ? w10 : w11);
                    atomicAdd(&s_buf[ly * SPLAT_TILE_W + lx], h.value * w);
                }
            }
            continue;
        }

        // Tent filter — compute GLOBAL normalization (separable)
        float inv_r = 1.0f / radius;
        int gix0 = max((int)floorf(h.px - radius), 0);
        int gix1 = min((int)ceilf(h.px + radius), img_w - 1);
        int giy0 = max((int)floorf(h.py - radius), 0);
        int giy1 = min((int)ceilf(h.py + radius), img_h - 1);

        float sum_wx = 0.0f, sum_wy = 0.0f;
        for (int x = gix0; x <= gix1; ++x)
            sum_wx += fmaxf(1.0f - fabsf(x + 0.5f - h.px) * inv_r, 0.0f);
        for (int y = giy0; y <= giy1; ++y)
            sum_wy += fmaxf(1.0f - fabsf(y + 0.5f - h.py) * inv_r, 0.0f);

        float tw = sum_wx * sum_wy;
        if (tw < 1e-12f)
            continue;
        float norm = h.value / tw;

        // Splat only the tile-local portion of the footprint
        int lx0 = max(gix0 - tile_x0, 0);
        int lx1 = min(gix1 - tile_x0, SPLAT_TILE_W - 1);
        int ly0 = max(giy0 - tile_y0, 0);
        int ly1 = min(giy1 - tile_y0, SPLAT_TILE_H - 1);

        for (int ly = ly0; ly <= ly1; ++ly)
        {
            int gy = tile_y0 + ly;
            float wy = fmaxf(1.0f - fabsf(gy + 0.5f - h.py) * inv_r, 0.0f);
            for (int lx = lx0; lx <= lx1; ++lx)
            {
                int gx = tile_x0 + lx;
                float wx = fmaxf(1.0f - fabsf(gx + 0.5f - h.px) * inv_r, 0.0f);
                atomicAdd(&s_buf[ly * SPLAT_TILE_W + lx], norm * wx * wy);
            }
        }
    }

    __syncthreads();

    // Write tile to global output — no atomics, each tile owns its pixels
    for (int i = threadIdx.x; i < SPLAT_TILE_PX; i += blockDim.x)
    {
        int lx = i % SPLAT_TILE_W;
        int ly = i / SPLAT_TILE_W;
        int gx = tile_x0 + lx;
        int gy = tile_y0 + ly;
        if (gx < img_w && gy < img_h)
        {
            int gi = gy * img_w + gx;
            out_r[gi] = s_r[i];
            out_g[gi] = s_g[i];
            out_b[gi] = s_b[i];
        }
    }
}

// ============================================================================
// Public API
// ============================================================================

bool gpu_ghosts_available()
{
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
}

bool render_ghosts_gpu(const std::vector<GPUSurface> &surfaces,
                       const std::vector<GPUBrightPixel> &sources,
                       const std::vector<GPUGhostPair> &pairs,
                       const GPUGhostConfig &config,
                       float *out_r, float *out_g, float *out_b)
{
    auto t_start = std::chrono::steady_clock::now();

    // Print GPU info
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("GPU: %s (compute %d.%d, %d SMs, %.0f MHz)\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, prop.clockRate / 1000.0);

    int total_work = config.num_pairs * config.num_sources * config.valid_grid_count;
    printf("GPU ghost trace: %d pairs × %d sources × %d grid = %d threads\n",
           config.num_pairs, config.num_sources, config.valid_grid_count, total_work);

    // Estimate hit buffer size: assume up to 15% hit rate across all rays×wavelengths
    long long max_rays = (long long)total_work * 3; // 3 wavelengths
    int max_hits = std::min((long long)(max_rays * 0.15 + 1000000), (long long)200000000LL);
    size_t hit_buf_bytes = (size_t)max_hits * sizeof(GPURayHit);
    printf("GPU hit buffer: %d max hits (%.0f MB)\n",
           max_hits, hit_buf_bytes / (1024.0 * 1024.0));

    // Check available GPU memory
    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    printf("GPU memory: %.0f MB free / %.0f MB total\n",
           free_mem / (1024.0 * 1024.0), total_mem / (1024.0 * 1024.0));

    size_t needed = hit_buf_bytes +
                    surfaces.size() * sizeof(GPUSurface) +
                    sources.size() * sizeof(GPUBrightPixel) +
                    pairs.size() * sizeof(GPUGhostPair) +
                    sizeof(int) * 2 + // counters
                    // GPU splatting buffers (output images + per-group stats)
                    (size_t)config.img_width * config.img_height * sizeof(float) * 3 +
                    (size_t)config.num_pairs * config.num_sources * sizeof(float) * 5;
    if (needed > free_mem * 0.9)
    {
        fprintf(stderr, "GPU: not enough memory (need %.0f MB, have %.0f MB free)\n",
                needed / (1024.0 * 1024.0), free_mem / (1024.0 * 1024.0));
        return false;
    }

    // Allocate device memory
    GPUSurface *d_surfaces = nullptr;
    GPUBrightPixel *d_sources = nullptr;
    GPUGhostPair *d_pairs = nullptr;
    GPURayHit *d_hits = nullptr;
    int *d_hit_count = nullptr;
    int2 *d_grid_lut = nullptr;

    CUDA_CHECK(cudaMalloc(&d_surfaces, surfaces.size() * sizeof(GPUSurface)));
    CUDA_CHECK(cudaMalloc(&d_sources, sources.size() * sizeof(GPUBrightPixel)));
    CUDA_CHECK(cudaMalloc(&d_pairs, pairs.size() * sizeof(GPUGhostPair)));
    CUDA_CHECK(cudaMalloc(&d_hits, hit_buf_bytes));
    CUDA_CHECK(cudaMalloc(&d_hit_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_hit_count, 0, sizeof(int)));

    // Upload data
    CUDA_CHECK(cudaMemcpy(d_surfaces, surfaces.data(),
                          surfaces.size() * sizeof(GPUSurface), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sources, sources.data(),
                          sources.size() * sizeof(GPUBrightPixel), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pairs, pairs.data(),
                          pairs.size() * sizeof(GPUGhostPair), cudaMemcpyHostToDevice));

    // Build grid LUT: precompute (gx, gy) for each valid grid sample
    // Eliminates the O(N²) scan that every trace thread was doing
    {
        int N = config.ray_grid;
        std::vector<int2> h_grid_lut(config.valid_grid_count);
        int count = 0;
        for (int yy = 0; yy < N; ++yy)
            for (int xx = 0; xx < N; ++xx)
            {
                float u = ((xx + 0.5f) / N) * 2.0f - 1.0f;
                float v = ((yy + 0.5f) / N) * 2.0f - 1.0f;
                if (u * u + v * v <= 1.0f)
                    h_grid_lut[count++] = {xx, yy};
            }
        CUDA_CHECK(cudaMalloc(&d_grid_lut, config.valid_grid_count * sizeof(int2)));
        CUDA_CHECK(cudaMemcpy(d_grid_lut, h_grid_lut.data(),
                              config.valid_grid_count * sizeof(int2), cudaMemcpyHostToDevice));
    }

    // Mapped pinned memory for progress counter
    int *h_progress = nullptr;
    int *d_progress = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_progress, sizeof(int), cudaHostAllocMapped));
    *h_progress = 0;
    CUDA_CHECK(cudaHostGetDevicePointer(&d_progress, h_progress, 0));

    // Launch kernel
    const int threads_per_block = 256;
    const int num_blocks = (total_work + threads_per_block - 1) / threads_per_block;
    printf("GPU launch: %d blocks × %d threads\n", num_blocks, threads_per_block);

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventRecord(ev_start));

    ghost_trace_kernel<<<num_blocks, threads_per_block>>>(
        d_surfaces, d_sources, d_pairs, d_grid_lut,
        d_hits, d_hit_count, max_hits, d_progress,
        config);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ev_stop));

    // Poll progress
    auto poll_start = std::chrono::steady_clock::now();
    while (cudaEventQuery(ev_stop) == cudaErrorNotReady)
    {
        int done = *h_progress;
        double pct = 100.0 * done / (double)total_work;
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - poll_start).count();
        double rate = (elapsed > 0.01) ? done / elapsed : 0;
        double eta = (rate > 0 && done < total_work) ? (total_work - done) / rate : 0;
        int eta_m = (int)(eta / 60.0);
        int eta_s = (int)(eta) % 60;
        printf("\r  GPU trace: %6.2f%% (%d/%d)  %.1f Mray/s  ETA %dm%02ds   ",
               pct, done, total_work, rate / 1e6, eta_m, eta_s);
        fflush(stdout);
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    printf("\r  GPU trace: 100.00%% (%d/%d)                                      \n",
           total_work, total_work);

    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float kernel_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ev_start, ev_stop));
    printf("GPU kernel: %.2f s (%.1f Mray/s)\n",
           kernel_ms / 1000.0, total_work / kernel_ms / 1000.0);

    // Get hit count
    int num_hits = 0;
    CUDA_CHECK(cudaMemcpy(&num_hits, d_hit_count, sizeof(int), cudaMemcpyDeviceToHost));
    if (num_hits > max_hits)
    {
        printf("WARNING: hit buffer overflow (%d > %d), some hits lost\n", num_hits, max_hits);
        num_hits = max_hits;
    }
    printf("GPU hits: %d (%.1f%% of %lld rays)\n",
           num_hits, 100.0 * num_hits / std::max(max_rays, 1LL), max_rays);

    // Free trace-only resources (keep d_hits on device for GPU splatting)
    cudaFree(d_surfaces);
    cudaFree(d_sources);
    cudaFree(d_pairs);
    cudaFree(d_grid_lut);
    cudaFree(d_hit_count);
    cudaFreeHost(h_progress);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    // ================================================================
    // GPU-side: adaptive tent-filter splatting (tile-based)
    //
    // Uses pair_idx × num_sources + source_idx as a perfect hash key
    // into per-group arrays, avoiding the need to sort 80M+ hits.
    //
    // 1. Accumulate green-channel bbox per group (atomicMin/Max)
    // 2. Compute adaptive radius per group
    // 3. Bin hits into image tiles based on footprint overlap
    // 4. One thread block per tile, accumulate in shared memory
    // 5. Write tiles to global (no atomics — tiles are disjoint)
    // ================================================================

    int img_w = config.img_width, img_h = config.img_height;
    int num_px = img_w * img_h;
    int max_groups = config.num_pairs * config.num_sources;

    printf("GPU splatting %d hits (%d potential groups, %dx%d image)...\n",
           num_hits, max_groups, img_w, img_h);
    auto t_splat = std::chrono::steady_clock::now();

    // Allocate per-group stats (small: e.g. 55 pairs × 1000 sources = 220 KB)
    float *d_gmin_x = nullptr, *d_gmax_x = nullptr;
    float *d_gmin_y = nullptr, *d_gmax_y = nullptr;
    int *d_green_count = nullptr;
    float *d_group_radii = nullptr;

    CUDA_CHECK(cudaMalloc(&d_gmin_x, max_groups * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gmax_x, max_groups * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gmin_y, max_groups * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gmax_y, max_groups * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_green_count, max_groups * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_group_radii, max_groups * sizeof(float)));

    // Allocate device output images (zeroed)
    float *d_out_r = nullptr, *d_out_g = nullptr, *d_out_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out_r, num_px * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_g, num_px * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_b, num_px * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out_r, 0, num_px * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out_g, 0, num_px * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out_b, 0, num_px * sizeof(float)));

    const int BLK = 256;

    // ---- Step 1: init group stats ----
    {
        int nblk = (max_groups + BLK - 1) / BLK;
        init_group_stats_kernel<<<nblk, BLK>>>(
            d_gmin_x, d_gmax_x, d_gmin_y, d_gmax_y, d_green_count, max_groups);
        CUDA_CHECK(cudaGetLastError());
    }

    // ---- Step 2: accumulate green-channel bbox per group ----
    {
        int nblk = (num_hits + BLK - 1) / BLK;
        accumulate_group_stats_kernel<<<nblk, BLK>>>(
            d_hits, num_hits, config.num_sources,
            d_gmin_x, d_gmax_x, d_gmin_y, d_gmax_y, d_green_count);
        CUDA_CHECK(cudaGetLastError());
    }

    // ---- Step 3: compute adaptive radius per group ----
    {
        int nblk = (max_groups + BLK - 1) / BLK;
        compute_radii_kernel<<<nblk, BLK>>>(
            d_group_radii, d_gmin_x, d_gmax_x, d_gmin_y, d_gmax_y,
            d_green_count, max_groups);
        CUDA_CHECK(cudaGetLastError());
    }

    // Free group stats (radii array kept for splat kernel)
    cudaFree(d_gmin_x);
    cudaFree(d_gmax_x);
    cudaFree(d_gmin_y);
    cudaFree(d_gmax_y);
    cudaFree(d_green_count);

    // ---- Step 4: tile-based splatting ----
    int tiles_x = (img_w + SPLAT_TILE_W - 1) / SPLAT_TILE_W;
    int tiles_y = (img_h + SPLAT_TILE_H - 1) / SPLAT_TILE_H;
    int num_tiles = tiles_x * tiles_y;
    printf("  Tile-based splatting: %dx%d tiles (%dx%d px each)\n",
           tiles_x, tiles_y, SPLAT_TILE_W, SPLAT_TILE_H);

    // Step 4a: count hits per tile
    int *d_tile_counts = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tile_counts, num_tiles * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_tile_counts, 0, num_tiles * sizeof(int)));

    {
        int nblk = (num_hits + BLK - 1) / BLK;
        count_tile_hits_kernel<<<nblk, BLK>>>(
            d_hits, num_hits, config.num_sources, d_group_radii,
            d_tile_counts, tiles_x, tiles_y, img_w, img_h);
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 4b: prefix sum → tile offsets (small — CPU scan is fine)
    std::vector<int> h_tile_counts(num_tiles);
    CUDA_CHECK(cudaMemcpy(h_tile_counts.data(), d_tile_counts,
                          num_tiles * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_tile_counts);

    std::vector<int> h_tile_offsets(num_tiles + 1);
    h_tile_offsets[0] = 0;
    for (int t = 0; t < num_tiles; ++t)
        h_tile_offsets[t + 1] = h_tile_offsets[t] + h_tile_counts[t];
    long long total_bin_entries = h_tile_offsets[num_tiles];

    printf("  Tile bins: %lld entries (%.1f× expansion from %d hits)\n",
           total_bin_entries, (double)total_bin_entries / std::max(num_hits, 1),
           num_hits);

    // Check memory for tile bins
    size_t bin_bytes = (size_t)total_bin_entries * sizeof(int);
    {
        size_t free_now = 0, total_now = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_now, &total_now));
        size_t tile_mem = bin_bytes + (size_t)(num_tiles + 1) * sizeof(int) * 2;
        if (tile_mem > free_now * 0.85)
        {
            fprintf(stderr, "GPU: tile bins too large (%.0f MB > %.0f MB free)\n",
                    tile_mem / (1024.0 * 1024.0), free_now / (1024.0 * 1024.0));
            cudaFree(d_hits);
            cudaFree(d_group_radii);
            cudaFree(d_out_r);
            cudaFree(d_out_g);
            cudaFree(d_out_b);
            return false; // fall back to CPU
        }
    }

    int *d_tile_offsets = nullptr, *d_tile_bins = nullptr;
    int *d_tile_cursors = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tile_offsets, (num_tiles + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_tile_offsets, h_tile_offsets.data(),
                          (num_tiles + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_tile_bins, bin_bytes));
    CUDA_CHECK(cudaMalloc(&d_tile_cursors, num_tiles * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_tile_cursors, 0, num_tiles * sizeof(int)));

    // Step 4c: scatter hit indices into tile bins
    {
        int nblk = (num_hits + BLK - 1) / BLK;
        fill_tile_bins_kernel<<<nblk, BLK>>>(
            d_hits, num_hits, config.num_sources, d_group_radii,
            d_tile_offsets, d_tile_cursors, d_tile_bins,
            tiles_x, tiles_y, img_w, img_h);
        CUDA_CHECK(cudaGetLastError());
    }

    cudaFree(d_tile_cursors);

    // Step 4d: tile-parallel splat (one block per tile, 256 threads)
    printf("  Launching tile splat: %d tiles...\n", num_tiles);

    cudaEvent_t ev_splat_start, ev_splat_stop;
    CUDA_CHECK(cudaEventCreate(&ev_splat_start));
    CUDA_CHECK(cudaEventCreate(&ev_splat_stop));
    CUDA_CHECK(cudaEventRecord(ev_splat_start));

    splat_tiles_kernel<<<num_tiles, SPLAT_TILE_PX>>>(
        d_hits, d_tile_offsets, d_tile_bins,
        config.num_sources, d_group_radii,
        d_out_r, d_out_g, d_out_b,
        tiles_x, tiles_y, img_w, img_h);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(ev_splat_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_splat_stop));

    float splat_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&splat_ms, ev_splat_start, ev_splat_stop));
    printf("  GPU tile splat: %.2f s (%.1f Mhit/s)\n",
           splat_ms / 1000.0, num_hits / std::max(splat_ms, 0.001f) / 1000.0);

    cudaEventDestroy(ev_splat_start);
    cudaEventDestroy(ev_splat_stop);
    cudaFree(d_tile_offsets);
    cudaFree(d_tile_bins);

    // ---- Step 5: download output images and add to host buffers ----
    {
        std::vector<float> tmp(num_px);

        CUDA_CHECK(cudaMemcpy(tmp.data(), d_out_r, num_px * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < num_px; ++i)
            out_r[i] += tmp[i];

        CUDA_CHECK(cudaMemcpy(tmp.data(), d_out_g, num_px * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < num_px; ++i)
            out_g[i] += tmp[i];

        CUDA_CHECK(cudaMemcpy(tmp.data(), d_out_b, num_px * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < num_px; ++i)
            out_b[i] += tmp[i];
    }

    // Cleanup
    cudaFree(d_hits);
    cudaFree(d_group_radii);
    cudaFree(d_out_r);
    cudaFree(d_out_g);
    cudaFree(d_out_b);

    auto t_end = std::chrono::steady_clock::now();
    double splat_time = std::chrono::duration<double>(t_end - t_splat).count();
    double total_time = std::chrono::duration<double>(t_end - t_start).count();
    printf("GPU splat total: %.2f s (kernel %.2f s + overhead %.2f s)\n",
           splat_time, splat_ms / 1000.0, splat_time - splat_ms / 1000.0);
    printf("GPU ghost total: %.2f s (trace %.2f s + splat %.2f s)\n",
           total_time, kernel_ms / 1000.0, splat_time);

    return true;
}
