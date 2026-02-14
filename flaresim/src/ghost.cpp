// ============================================================================
// ghost.cpp — Ghost reflection rendering
//
// For each ghost bounce pair (a, b), traces a grid of rays through the
// lens system.  Each ray enters the front element as a collimated beam
// at the angle of a bright source pixel, reflects at surfaces a and b,
// and lands on the sensor plane.  The contribution is splatted onto the
// output image with bilinear weighting.
//
// Pre-filtering: a single on-axis ray is traced per pair to estimate the
// Fresnel weight.  Pairs below the min_intensity threshold are skipped.
// ============================================================================

#include "ghost.h"
#include "trace.h"

#include <cstdio>
#include <cmath>
#include <algorithm>
#include <vector>
#include <chrono>
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// Enumerate all ghost pairs: every combination of 2 surfaces.
// ---------------------------------------------------------------------------

std::vector<GhostPair> enumerate_ghost_pairs(const LensSystem &lens)
{
    std::vector<GhostPair> pairs;
    int N = lens.num_surfaces();
    for (int a = 0; a < N; ++a)
        for (int b = a + 1; b < N; ++b)
            pairs.push_back({a, b});
    return pairs;
}

// ---------------------------------------------------------------------------
// Bilinear splat: distribute a contribution to 4 neighbouring pixels.
// ---------------------------------------------------------------------------

static inline void splat_bilinear(float *buf, int w, int h,
                                  float px, float py, float value)
{
    int x0 = (int)std::floor(px - 0.5f);
    int y0 = (int)std::floor(py - 0.5f);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float fx = (px - 0.5f) - x0;
    float fy = (py - 0.5f) - y0;

    float w00 = (1.0f - fx) * (1.0f - fy);
    float w10 = fx * (1.0f - fy);
    float w01 = (1.0f - fx) * fy;
    float w11 = fx * fy;

    if (x0 >= 0 && x0 < w && y0 >= 0 && y0 < h)
        buf[y0 * w + x0] += value * w00;
    if (x1 >= 0 && x1 < w && y0 >= 0 && y0 < h)
        buf[y0 * w + x1] += value * w10;
    if (x0 >= 0 && x0 < w && y1 >= 0 && y1 < h)
        buf[y1 * w + x0] += value * w01;
    if (x1 >= 0 && x1 < w && y1 >= 0 && y1 < h)
        buf[y1 * w + x1] += value * w11;
}

// ---------------------------------------------------------------------------
// Pre-filter: trace a single on-axis ray through each ghost pair.
// Returns the average Fresnel weight across RGB wavelengths.
// ---------------------------------------------------------------------------

static float estimate_ghost_intensity(const LensSystem &lens,
                                      int bounce_a, int bounce_b,
                                      const GhostConfig &config)
{
    // On-axis ray at the centre of the entrance pupil
    Ray ray;
    ray.origin = Vec3f(0, 0, lens.surfaces[0].z - 20.0f);
    ray.dir = Vec3f(0, 0, 1);

    float total = 0;
    for (int ch = 0; ch < 3; ++ch)
    {
        TraceResult r = trace_ghost_ray(ray, lens, bounce_a, bounce_b,
                                        config.wavelengths[ch]);
        if (r.valid)
            total += r.weight;
    }
    return total / 3.0f;
}

// ---------------------------------------------------------------------------
// Estimate the ghost image spread for a bounce pair relative to the sensor.
//
// Traces a coarse grid of on-axis rays across the entrance pupil and
// measures the bounding box of sensor landing positions.  Returns a
// correction factor = ghost_area / sensor_area, clamped to [1, max_boost].
//
// Defocused ghost pairs produce images much larger than the sensor,
// diluting per-pixel brightness.  This correction factor compensates
// for that geometric dilution so all ghost pairs remain visible.
// ---------------------------------------------------------------------------

static float estimate_ghost_spread(const LensSystem &lens,
                                   int bounce_a, int bounce_b,
                                   float sensor_half_w, float sensor_half_h,
                                   const GhostConfig &config)
{
    constexpr int G = 8; // coarse grid for spread estimation
    float front_R = lens.surfaces[0].semi_aperture;
    float start_z = lens.surfaces[0].z - 20.0f;

    float min_x = 1e30f, max_x = -1e30f;
    float min_y = 1e30f, max_y = -1e30f;
    int valid_count = 0;

    for (int gy = 0; gy < G; ++gy)
    {
        for (int gx = 0; gx < G; ++gx)
        {
            float u = ((gx + 0.5f) / G) * 2.0f - 1.0f;
            float v = ((gy + 0.5f) / G) * 2.0f - 1.0f;
            if (u * u + v * v > 1.0f)
                continue;

            Ray ray;
            ray.origin = Vec3f(u * front_R, v * front_R, start_z);
            ray.dir = Vec3f(0, 0, 1); // on-axis

            // Use green wavelength for spread estimation
            TraceResult res = trace_ghost_ray(ray, lens, bounce_a, bounce_b,
                                              config.wavelengths[1]);
            if (!res.valid)
                continue;

            min_x = std::min(min_x, res.position.x);
            max_x = std::max(max_x, res.position.x);
            min_y = std::min(min_y, res.position.y);
            max_y = std::max(max_y, res.position.y);
            ++valid_count;
        }
    }

    if (valid_count < 2)
        return 1.0f; // too few hits to estimate

    float ghost_w = std::max(max_x - min_x, 0.01f);
    float ghost_h = std::max(max_y - min_y, 0.01f);
    float sensor_w = 2.0f * sensor_half_w;
    float sensor_h = 2.0f * sensor_half_h;

    // Correction = how much larger the ghost image is than the sensor.
    // Clamped to [1, max_boost] — never dim a focused ghost, and cap the boost.
    float area_ratio = (ghost_w * ghost_h) / (sensor_w * sensor_h);
    return std::clamp(area_ratio, 1.0f, config.max_area_boost);
}

// ---------------------------------------------------------------------------
// Render all ghost reflections.
// ---------------------------------------------------------------------------

void render_ghosts(const LensSystem &lens,
                   const std::vector<BrightPixel> &sources,
                   float fov_h, float fov_v,
                   float *out_r, float *out_g, float *out_b,
                   int width, int height,
                   const GhostConfig &config)
{
    auto pairs = enumerate_ghost_pairs(lens);
    printf("Total ghost pairs: %zu\n", pairs.size());

    // Sensor dimensions from focal length and FOV
    float sensor_half_w = lens.focal_length * std::tan(fov_h * 0.5f);
    float sensor_half_h = lens.focal_length * std::tan(fov_v * 0.5f);

    // Entrance pupil sampling setup
    float front_R = lens.surfaces[0].semi_aperture;
    float start_z = lens.surfaces[0].z - 20.0f;
    int N = config.ray_grid;
    size_t num_px = (size_t)width * height;

    // Pre-count valid grid samples (within circular aperture)
    int valid_grid_count = 0;
    for (int gy = 0; gy < N; ++gy)
        for (int gx = 0; gx < N; ++gx)
        {
            float u = ((gx + 0.5f) / N) * 2.0f - 1.0f;
            float v = ((gy + 0.5f) / N) * 2.0f - 1.0f;
            if (u * u + v * v <= 1.0f)
                ++valid_grid_count;
        }

    if (valid_grid_count == 0)
    {
        fprintf(stderr, "WARNING: no valid entrance pupil samples\n");
        return;
    }
    float ray_weight = 1.0f / valid_grid_count;

    printf("Entrance pupil: %.2f mm radius, %d×%d grid → %d rays/source\n",
           front_R, N, N, valid_grid_count);
    printf("Bright sources: %zu\n", sources.size());
    printf("Sensor: %.2f × %.2f mm (at z = %.2f mm)\n",
           sensor_half_w * 2, sensor_half_h * 2, lens.sensor_z);

    // Pre-filter ghost pairs
    std::vector<GhostPair> active_pairs;
    std::vector<float> pair_area_boost; // per-pair area correction factor
    for (auto &p : pairs)
    {
        // Skip pairs where either bounce surface has n1 ≈ n2 (air-to-air),
        // since Fresnel reflectance is zero and the pair contributes nothing.
        auto ior_before_a = lens.ior_before(p.surf_a);
        auto ior_after_a = lens.surfaces[p.surf_a].ior;
        auto ior_before_b = lens.ior_before(p.surf_b);
        auto ior_after_b = lens.surfaces[p.surf_b].ior;
        if (std::abs(ior_before_a - ior_after_a) < 0.001f ||
            std::abs(ior_before_b - ior_after_b) < 0.001f)
        {
            // Air-to-air surface: zero reflectance, skip.
            continue;
        }

        float est = estimate_ghost_intensity(lens, p.surf_a, p.surf_b, config);
        if (est >= config.min_intensity)
        {
            // Estimate defocus spread and compute area correction
            float boost = 1.0f;
            if (config.ghost_normalize)
            {
                boost = estimate_ghost_spread(lens, p.surf_a, p.surf_b,
                                              sensor_half_w, sensor_half_h,
                                              config);
            }
            active_pairs.push_back(p);
            pair_area_boost.push_back(boost);
        }
    }
    printf("Active ghost pairs (above %.1e threshold): %zu / %zu\n",
           config.min_intensity, active_pairs.size(), pairs.size());
    if (config.ghost_normalize)
    {
        printf("Area normalization: ON (max boost %.0f×)\n", config.max_area_boost);
        for (size_t i = 0; i < active_pairs.size(); ++i)
        {
            if (pair_area_boost[i] > 1.01f)
                printf("  pair (%d,%d): area boost %.1f×\n",
                       active_pairs[i].surf_a, active_pairs[i].surf_b,
                       pair_area_boost[i]);
        }
    }

    auto t0 = std::chrono::steady_clock::now();

    // ====================================================================
    // Parallel over ghost pairs
    //
    // Each thread gets its own full-image accumulation buffer, allocated
    // ONCE before the loop.  Threads process different ghost pairs via
    // dynamic scheduling, iterating over all bright sources within each
    // pair.  This eliminates the old pattern of:
    //   1) per-pair buffer allocation/deallocation (hundreds of MB churn)
    //   2) per-pair thread-barrier overhead
    //   3) per-pair serial reduction
    // and provides much better load balancing across pairs with wildly
    // different hit rates.
    // ====================================================================

    int num_threads = 1;
#ifdef _OPENMP
    num_threads = omp_get_max_threads();
#endif

    // Pre-allocate per-thread accumulation buffers (ONE allocation total)
    std::vector<std::vector<float>> tbuf_r(num_threads, std::vector<float>(num_px, 0));
    std::vector<std::vector<float>> tbuf_g(num_threads, std::vector<float>(num_px, 0));
    std::vector<std::vector<float>> tbuf_b(num_threads, std::vector<float>(num_px, 0));

    // Per-pair stats (written from threads, read after parallel section)
    std::vector<long long> pair_hits(active_pairs.size(), 0);
    std::vector<long long> pair_attempts(active_pairs.size(), 0);
    std::vector<double> pair_time(active_pairs.size(), 0);

    printf("Rendering %zu ghost pairs × %zu sources across %d thread(s)...\n",
           active_pairs.size(), sources.size(), num_threads);
    fflush(stdout);

#pragma omp parallel for schedule(dynamic, 1)
    for (int pi = 0; pi < (int)active_pairs.size(); ++pi)
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        int a = active_pairs[pi].surf_a;
        int b = active_pairs[pi].surf_b;
        float area_boost = pair_area_boost[pi];

        auto tp0 = std::chrono::steady_clock::now();
        long long hits = 0, attempts = 0;

        // Process every bright source for this ghost pair
        for (int si = 0; si < (int)sources.size(); ++si)
        {
            const BrightPixel &src = sources[si];

            // Per-source RNG for stratified jitter (seeded by source index
            // and ghost pair to get different patterns per pair)
            std::mt19937 rng((unsigned)(si * 7919 + a * 131 + b * 1031));
            std::uniform_real_distribution<float> jitter(0.0f, 1.0f);

            // Collimated beam direction for this source
            Vec3f beam_dir = Vec3f(std::tan(src.angle_x),
                                   std::tan(src.angle_y),
                                   1.0f)
                                 .normalized();

            // Trace grid of rays across entrance pupil with stratified jitter
            for (int gy = 0; gy < N; ++gy)
            {
                for (int gx = 0; gx < N; ++gx)
                {
                    // Stratified: random position within each grid cell
                    float u = ((gx + jitter(rng)) / N) * 2.0f - 1.0f;
                    float v = ((gy + jitter(rng)) / N) * 2.0f - 1.0f;
                    if (u * u + v * v > 1.0f)
                        continue;

                    Ray ray;
                    ray.origin = Vec3f(u * front_R, v * front_R, start_z);
                    ray.dir = beam_dir;

                    // Trace each wavelength independently (chromatic dispersion)
                    for (int ch = 0; ch < 3; ++ch)
                    {
                        ++attempts;
                        TraceResult res = trace_ghost_ray(ray, lens, a, b,
                                                          config.wavelengths[ch]);
                        if (!res.valid)
                            continue;

                        ++hits;

                        // Map sensor position to pixel coordinates
                        float px = (res.position.x / (2.0f * sensor_half_w) + 0.5f) * width;
                        float py = (res.position.y / (2.0f * sensor_half_h) + 0.5f) * height;

                        // Source intensity for this channel
                        float src_i = (ch == 0) ? src.r : (ch == 1) ? src.g
                                                                    : src.b;
                        float contribution = src_i * res.weight * ray_weight * config.gain * area_boost;

                        if (contribution < 1e-12f)
                            continue;

                        // Bilinear splat into per-thread buffer
                        auto &buf = (ch == 0)   ? tbuf_r[tid]
                                    : (ch == 1) ? tbuf_g[tid]
                                                : tbuf_b[tid];
                        splat_bilinear(buf.data(), width, height, px, py, contribution);
                    }
                }
            }
        }

        auto tp1 = std::chrono::steady_clock::now();
        pair_hits[pi] = hits;
        pair_attempts[pi] = attempts;
        pair_time[pi] = std::chrono::duration<double>(tp1 - tp0).count();
    }

    // Print per-pair diagnostics (after parallel section completes)
    for (size_t pi = 0; pi < active_pairs.size(); ++pi)
    {
        printf("  [%zu/%zu] Ghost pair (%d, %d) [area ×%.1f]  %.1f s  "
               "(%lld/%lld rays, %.1f%%)\n",
               pi + 1, active_pairs.size(),
               active_pairs[pi].surf_a, active_pairs[pi].surf_b,
               pair_area_boost[pi], pair_time[pi],
               pair_hits[pi], pair_attempts[pi],
               pair_attempts[pi] > 0
                   ? 100.0 * pair_hits[pi] / pair_attempts[pi]
                   : 0.0);
    }

    // Reduce per-thread buffers into output (parallelised over pixels)
#pragma omp parallel for
    for (int i = 0; i < (int)num_px; ++i)
    {
        for (int t = 0; t < num_threads; ++t)
        {
            out_r[i] += tbuf_r[t][i];
            out_g[i] += tbuf_g[t][i];
            out_b[i] += tbuf_b[t][i];
        }
    }

    auto t1 = std::chrono::steady_clock::now();

    long long total_hits = 0, total_attempts = 0;
    for (size_t pi = 0; pi < active_pairs.size(); ++pi)
    {
        total_hits += pair_hits[pi];
        total_attempts += pair_attempts[pi];
    }
    printf("Ghost rendering total: %.1f s  (%lld/%lld rays, %.1f%%)\n",
           std::chrono::duration<double>(t1 - t0).count(),
           total_hits, total_attempts,
           total_attempts > 0 ? 100.0 * total_hits / total_attempts : 0.0);
}
