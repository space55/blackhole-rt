
#include <stdio.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>

#include "common.h"
#include "types.h"
#include "sky.h"
#include "ray.h"
#include "disk.h"
#include "blackhole.h"
#include "scene.h"

#include "stb_image_write.h"

#ifdef USE_GPU
#include "gpu_render.h"
#endif

int main(int argc, char *argv[])
{
    // --- Load scene config ------------------------------------------------
    const char *scene_path = (argc > 1) ? argv[1] : "scene.txt";
    scene_config_s cfg;
    if (!load_scene_config(scene_path, cfg))
    {
        printf("Warning: could not open '%s', using built-in defaults\n", scene_path);
    }
    else
    {
        printf("Loaded scene: %s\n", scene_path);
    }
    print_scene_config(cfg);

    // --- Sky image -------------------------------------------------------
    sky_image_s *image = load_sky_image(cfg.sky_image.c_str());
    if (!image)
    {
        printf("Failed to load sky image: %s\n", cfg.sky_image.c_str());
        return 1;
    }
    printf("Loaded sky image: %dx%d\n", image->width, image->height);

    // --- Derived constants from config -----------------------------------
    const int out_width = cfg.output_width;
    const int out_height = cfg.output_height;
    const double base_dt = cfg.base_dt;
    const double max_affine = cfg.max_affine;
    const double escape_r2 = cfg.escape_radius * cfg.escape_radius;
    const double fov_x = cfg.fov_x;
    const double fov_y = cfg.fov_y;
    const Vector3d camera_pos(cfg.camera_x, cfg.camera_y, cfg.camera_z);
    const Vector3d camera_rot(cfg.camera_pitch, cfg.camera_yaw, cfg.camera_roll);

    // Precompute camera rotation matrix once (avoids trig per ray)
    const Matrix3d cam_rot_matrix = (AngleAxisd(cfg.camera_yaw * M_PI / 180.0, Vector3d::UnitY()) *
                                     AngleAxisd(cfg.camera_pitch * M_PI / 180.0, Vector3d::UnitX()) *
                                     AngleAxisd(cfg.camera_roll * M_PI / 180.0, Vector3d::UnitZ()))
                                        .toRotationMatrix();

    // Precompute sky rotation matrix
    const Matrix3d sky_rot = (AngleAxisd(cfg.sky_yaw * M_PI / 180.0, Vector3d::UnitY()) * AngleAxisd(cfg.sky_pitch * M_PI / 180.0, Vector3d::UnitX()) * AngleAxisd(cfg.sky_roll * M_PI / 180.0, Vector3d::UnitZ())).toRotationMatrix();

    // HDR framebuffers: disk emission separate from sky for tone mapping
    const size_t num_pixels = static_cast<size_t>(out_width) * static_cast<size_t>(out_height);
    std::vector<Vector3d> hdr_disk(num_pixels, Vector3d::Zero());
    std::vector<Vector3d> hdr_sky(num_pixels, Vector3d::Zero());
    std::vector<BH_COLOR_CHANNEL_TYPE> pixels(num_pixels * 3);

    const int total_pixels = out_width * out_height;
    std::atomic<int> pixels_done(0);

    blackhole_s bh(cfg.bh_mass, cfg.bh_spin);
    printf("Black hole: M=%.1f, a=%.2f, r+=%.4f, r_isco=%.4f\n",
           bh.mass, bh.spin, bh.event_horizon_radius(), bh.isco_radius());

    const double r_plus = bh.event_horizon_radius();

    // --- Accretion disk --------------------------------------------------
    accretion_disk_s disk(&bh, cfg.disk_outer_r, cfg.disk_thickness,
                          cfg.disk_density, cfg.disk_opacity);
    disk.emission_boost = cfg.disk_emission_boost;
    disk.color_variation = cfg.disk_color_variation;
    disk.turbulence = cfg.disk_turbulence;
    disk.time = cfg.time;

    // --- Tone mapping ----------------------------------------------------
    const double disk_tonemap_compression = cfg.tonemap_compression;
    const double tonemap_c = pow(10.0, disk_tonemap_compression * 2.0) - 1.0;
    const double tonemap_norm = 1.0 / log(1.0 + tonemap_c);
    auto tonemap_disk = [&](const Vector3d &hdr) -> Vector3d
    {
        if (tonemap_c < 1e-6)
            return hdr; // compression ≈ 0 → identity
        return Vector3d(
            log(1.0 + tonemap_c * std::max(hdr.x(), 0.0)) * tonemap_norm,
            log(1.0 + tonemap_c * std::max(hdr.y(), 0.0)) * tonemap_norm,
            log(1.0 + tonemap_c * std::max(hdr.z(), 0.0)) * tonemap_norm);
    };
    printf("Disk tone-map: compression=%.2f  (c=%.2f)\n",
           disk_tonemap_compression, tonemap_c);

    std::atomic<int> disk_samples(0);

    // --- Anti-aliasing: stratified NxN supersampling ---------------------
    const int aa_grid = std::max(cfg.aa_samples, 1);
    const int aa_spp = aa_grid * aa_grid;
    const double inv_spp = 1.0 / aa_spp;
    const double inv_aa = 1.0 / aa_grid;
    printf("Anti-aliasing: %dx%d = %d samples/pixel\n", aa_grid, aa_grid, aa_spp);

    auto render_start = std::chrono::steady_clock::now();

#ifdef USE_GPU
    // =================================================================
    // GPU render path: physics on GPU, sky mapping + post-proc on CPU
    // =================================================================
    {
        GPUSceneParams gpu_params = {};

        // Camera position
        gpu_params.cam_pos[0] = cfg.camera_x;
        gpu_params.cam_pos[1] = cfg.camera_y;
        gpu_params.cam_pos[2] = cfg.camera_z;

        // Camera rotation matrix (precomputed via Eigen)
        const double gp_pitch = cfg.camera_pitch * M_PI / 180.0;
        const double gp_yaw = cfg.camera_yaw * M_PI / 180.0;
        const double gp_roll = cfg.camera_roll * M_PI / 180.0;
        Matrix3d cam_rot = (AngleAxisd(gp_yaw, Vector3d::UnitY()) *
                            AngleAxisd(gp_pitch, Vector3d::UnitX()) *
                            AngleAxisd(gp_roll, Vector3d::UnitZ()))
                               .toRotationMatrix();

        // right = col(0), up = col(1), forward = col(2)
        for (int i = 0; i < 3; i++)
        {
            gpu_params.cam_right[i] = cam_rot(i, 0);
            gpu_params.cam_up[i] = cam_rot(i, 1);
            gpu_params.cam_fwd[i] = cam_rot(i, 2);
        }
        gpu_params.fov_x = cfg.fov_x;
        gpu_params.fov_y = cfg.fov_y;

        // Black hole
        gpu_params.bh_mass = cfg.bh_mass;
        gpu_params.bh_spin = cfg.bh_spin;
        gpu_params.r_plus = bh.event_horizon_radius();
        gpu_params.r_isco = bh.isco_radius();

        // Integration
        gpu_params.base_dt = cfg.base_dt;
        gpu_params.max_affine = cfg.max_affine;
        gpu_params.escape_r2 = escape_r2;

        // Disk
        gpu_params.disk_inner_r = disk.inner_r;
        gpu_params.disk_outer_r = disk.outer_r;
        gpu_params.disk_thickness = cfg.disk_thickness;
        gpu_params.disk_density0 = cfg.disk_density;
        gpu_params.disk_opacity0 = cfg.disk_opacity;
        gpu_params.disk_r_ref = sqrt(disk.inner_r * disk.outer_r);
        gpu_params.disk_emission_boost = cfg.disk_emission_boost;
        gpu_params.disk_color_variation = cfg.disk_color_variation;
        gpu_params.disk_turbulence = cfg.disk_turbulence;
        gpu_params.disk_time = cfg.time;

        // Rendering
        gpu_params.aa_grid = aa_grid;
        gpu_params.width = out_width;
        gpu_params.height = out_height;

        // Launch GPU render
        std::vector<GPUPixelResult> gpu_results(num_pixels);
        if (!gpu_render(gpu_params, gpu_results.data()))
        {
            printf("GPU render failed!\n");
            return 1;
        }

        // --- CPU sky mapping from GPU results ---
        printf("Mapping sky on CPU...\n");
#pragma omp parallel for schedule(dynamic, 64)
        for (int i = 0; i < (int)num_pixels; i++)
        {
            hdr_disk[i] = Vector3d(gpu_results[i].disk_r,
                                   gpu_results[i].disk_g,
                                   gpu_results[i].disk_b);

            if (gpu_results[i].sky_weight > 1e-6f)
            {
                Vector3d exit_dir(gpu_results[i].exit_vx,
                                  gpu_results[i].exit_vy,
                                  gpu_results[i].exit_vz);
                double len = exit_dir.norm();
                if (len > 1e-12)
                {
                    exit_dir /= len;
                    Vector3d dir_rot = (sky_rot * exit_dir).normalized();

                    double u = 0.5 - (atan2(dir_rot.z(), dir_rot.x()) / (2.0 * M_PI));
                    double v = 0.5 - (asin(std::clamp(dir_rot.y(), -1.0, 1.0)) / M_PI);
                    u += cfg.sky_offset_u;
                    v += cfg.sky_offset_v;
                    u -= floor(u);
                    v = std::clamp(v - floor(v), 0.0, 1.0 - 1e-9);

                    int sx = std::clamp((int)(u * image->width), 0, image->width - 1);
                    int sy = std::clamp((int)(v * image->height), 0, image->height - 1);

                    Vector3d sky_color(image->r(sx, sy) / 255.0,
                                       image->g(sx, sy) / 255.0,
                                       image->b(sx, sy) / 255.0);

                    hdr_sky[i] = sky_color * cfg.sky_brightness *
                                 (double)gpu_results[i].sky_weight;
                }
            }
        }
    }
#else
    // =================================================================
    // CPU render path (OpenMP)
    // =================================================================
#pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < out_height; ++y)
    {
        for (int x = 0; x < out_width; ++x)
        {
            const int idx = y * out_width + x;
            Vector3d pixel_color = Vector3d::Zero();
            Vector3d pixel_sky = Vector3d::Zero();

            for (int sy = 0; sy < aa_grid; ++sy)
            {
                for (int sx = 0; sx < aa_grid; ++sx)
                {
                    // Sub-pixel offset: stratified sample at center of each grid cell
                    const double sub_x = (x + (sx + 0.5) * inv_aa) / out_width;
                    const double sub_y = (y + (sy + 0.5) * inv_aa) / out_height;

                    bool hit_black_hole = false;

                    ray_s ray(&bh,
                              camera_pos,
                              cam_rot_matrix,
                              sub_x,
                              sub_y,
                              fov_x,
                              fov_y);

                    ray.cached_ks_r = bh.ks_radius(ray.pos);
                    double affine = 0.0;
                    while (affine < max_affine)
                    {
                        // Use cached Kerr-Schild radius for step sizing (updated by advance())
                        const double r_ks = ray.cached_ks_r;
                        const double delta = std::max(r_ks - r_plus, 0.01);
                        double step_dt = base_dt * std::clamp(delta * delta, 0.0001, 1.0);

                        // Reduce step size when near the disk to avoid skipping through it
                        if (r_ks >= disk.inner_r * 0.8 && r_ks <= disk.outer_r * 1.2)
                        {
                            const double h = disk.half_thickness(r_ks);
                            const double y_dist = fabs(ray.pos.y());
                            if (y_dist < 5.0 * h)
                            {
                                // Near disk plane: limit step to fraction of disk thickness
                                step_dt = std::min(step_dt, std::max(0.3 * h, 0.005));
                            }
                        }

                        if (!ray.advance(step_dt))
                        {
                            hit_black_hole = true;
                            break;
                        }

                        // Sample disk emission along the ray
                        const double prev_opacity = ray.accumulated_opacity;
                        ray.sample_disk(disk, step_dt);
                        if (ray.accumulated_opacity > prev_opacity)
                            disk_samples.fetch_add(1, std::memory_order_relaxed);

                        affine += step_dt;
                        if (ray.cached_ks_r <= r_plus || !std::isfinite(ray.vel.squaredNorm()))
                        {
                            hit_black_hole = true;
                            break;
                        }
                        if (ray.distance_from_origin_squared() > escape_r2)
                        {
                            break;
                        }
                        // Early exit if disk emission is already fully opaque
                        if (ray.accumulated_opacity > 0.99)
                        {
                            break;
                        }
                    }

                    if (hit_black_hole)
                    {
                        // Behind horizon: just the accumulated disk emission (sky is black)
                        pixel_color += ray.accumulated_color;
                    }
                    else
                    {
                        // Store disk and sky separately so tone mapping only affects disk
                        Vector3d sky_color = ray.project_to_sky(*image, sky_rot,
                                                                cfg.sky_offset_u, cfg.sky_offset_v) *
                                             cfg.sky_brightness;
                        double transmittance = 1.0 - ray.accumulated_opacity;
                        pixel_color += ray.accumulated_color;
                        pixel_sky += transmittance * sky_color;
                    }

                } // sx
            } // sy

            // Average all samples and store in HDR buffers
            pixel_color *= inv_spp;
            pixel_sky *= inv_spp;
            hdr_disk[idx] = pixel_color;
            hdr_sky[idx] = pixel_sky;

            int done = pixels_done.fetch_add(1, std::memory_order_relaxed) + 1;
            if (done % out_width == 0)
            {
                auto now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(now - render_start).count();
                double pct = 100.0 * done / total_pixels;
                double avg = elapsed / done;
                double eta = avg * (total_pixels - done);
                int e_m = (int)elapsed / 60, e_s = (int)elapsed % 60;
                int r_m = (int)eta / 60, r_s = (int)eta % 60;
                printf("Progress: %d / %d (%.1f%%)  elapsed %d:%02d  ETA %d:%02d  (%.1f px/s)\n",
                       done, total_pixels, pct, e_m, e_s, r_m, r_s, done / elapsed);
            }
        }
    }

    printf("Disk samples accumulated: %d\n", disk_samples.load());
#endif // USE_GPU

    {
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - render_start).count();
        int e_m = (int)elapsed / 60, e_s = (int)elapsed % 60;
        printf("Total render time: %d:%02d (%.1f seconds)\n", e_m, e_s, elapsed);
    }

    // =====================================================================
    // Post-processing: Bloom
    // =====================================================================
    if (cfg.bloom_strength > 1e-6)
    {
        printf("Applying bloom: strength=%.2f  threshold=%.2f  radius=%.3f\n",
               cfg.bloom_strength, cfg.bloom_threshold, cfg.bloom_radius);

        const double diag = sqrt((double)(out_width * out_width + out_height * out_height));
        const int kernel_r = std::max((int)(cfg.bloom_radius * diag), 1);
        const double sigma = kernel_r / 3.0;

        // --- Compute 3 box-blur radii that approximate a Gaussian ---------
        // Three successive box blurs converge to a Gaussian (CLT).
        // Each box of width w has variance (w²-1)/12.
        // We want 3 * (w²-1)/12 = σ²  →  w = sqrt(12σ²/3 + 1).
        // We use up to 3 passes with possibly different widths for a
        // tighter approximation (Burt & Adelson / W3C filter spec).
        const int BOX_PASSES = 3;
        int box_radii[BOX_PASSES];
        {
            double w_ideal = sqrt(12.0 * sigma * sigma / BOX_PASSES + 1.0);
            int w_lo = ((int)w_ideal) | 1; // round down to odd
            if (w_lo < 1) w_lo = 1;
            int w_hi = w_lo + 2;            // next odd
            // How many passes should use w_hi vs w_lo:
            // n_hi * (w_hi²-1)/12 + (3-n_hi) * (w_lo²-1)/12 = σ²
            double target_var = sigma * sigma;
            double var_lo = (w_lo * w_lo - 1) / 12.0;
            double var_hi = (w_hi * w_hi - 1) / 12.0;
            int n_hi = (var_hi > var_lo + 1e-12)
                           ? std::clamp((int)round((target_var - BOX_PASSES * var_lo) / (var_hi - var_lo)), 0, BOX_PASSES)
                           : 0;
            for (int i = 0; i < BOX_PASSES; ++i)
                box_radii[i] = ((i < n_hi) ? w_hi : w_lo) / 2;
        }

        printf("Bloom: σ=%.1f, box radii = [%d, %d, %d]\n",
               sigma, box_radii[0], box_radii[1], box_radii[2]);

        // Step 1: Threshold — extract bright regions (same as before)
        // Use flat float arrays for cache-friendly access in blur passes
        std::vector<float> buf_r(num_pixels, 0.0f);
        std::vector<float> buf_g(num_pixels, 0.0f);
        std::vector<float> buf_b(num_pixels, 0.0f);
        const double thr = cfg.bloom_threshold;
#pragma omp parallel for
        for (int i = 0; i < (int)num_pixels; ++i)
        {
            Vector3d tm = tonemap_disk(hdr_disk[i]);
            double lum = 0.2126 * tm.x() + 0.7152 * tm.y() + 0.0722 * tm.z();
            if (lum > thr)
            {
                double scale = (lum - thr) / std::max(lum, 1e-12);
                buf_r[i] = (float)(hdr_disk[i].x() * scale);
                buf_g[i] = (float)(hdr_disk[i].y() * scale);
                buf_b[i] = (float)(hdr_disk[i].z() * scale);
            }
        }

        // --- O(1)-per-pixel box blur via sliding window -------------------
        // Horizontal and vertical passes, repeated BOX_PASSES times.
        std::vector<float> tmp_r(num_pixels), tmp_g(num_pixels), tmp_b(num_pixels);

        for (int pass = 0; pass < BOX_PASSES; ++pass)
        {
            const int br = box_radii[pass];
            const float inv_w = 1.0f / (2 * br + 1);

            // ---- Horizontal pass ----
#pragma omp parallel for
            for (int y = 0; y < out_height; ++y)
            {
                const int row = y * out_width;
                // Initialize running sum for first output pixel (x=0)
                // Window: [-br, br] clamped to [0, out_width-1]
                float sr = 0.0f, sg = 0.0f, sb = 0.0f;
                for (int k = -br; k <= br; ++k)
                {
                    int sx = std::clamp(k, 0, out_width - 1);
                    sr += buf_r[row + sx];
                    sg += buf_g[row + sx];
                    sb += buf_b[row + sx];
                }
                tmp_r[row] = sr * inv_w;
                tmp_g[row] = sg * inv_w;
                tmp_b[row] = sb * inv_w;

                for (int x = 1; x < out_width; ++x)
                {
                    // Add pixel entering the window on the right
                    int add = std::min(x + br, out_width - 1);
                    sr += buf_r[row + add];
                    sg += buf_g[row + add];
                    sb += buf_b[row + add];
                    // Remove pixel leaving the window on the left
                    int rem = std::clamp(x - br - 1, 0, out_width - 1);
                    sr -= buf_r[row + rem];
                    sg -= buf_g[row + rem];
                    sb -= buf_b[row + rem];

                    tmp_r[row + x] = sr * inv_w;
                    tmp_g[row + x] = sg * inv_w;
                    tmp_b[row + x] = sb * inv_w;
                }
            }

            // ---- Vertical pass ----
#pragma omp parallel for
            for (int x = 0; x < out_width; ++x)
            {
                float sr = 0.0f, sg = 0.0f, sb = 0.0f;
                for (int k = -br; k <= br; ++k)
                {
                    int sy = std::clamp(k, 0, out_height - 1);
                    sr += tmp_r[sy * out_width + x];
                    sg += tmp_g[sy * out_width + x];
                    sb += tmp_b[sy * out_width + x];
                }
                buf_r[x] = sr * inv_w;
                buf_g[x] = sg * inv_w;
                buf_b[x] = sb * inv_w;

                for (int y = 1; y < out_height; ++y)
                {
                    int add = std::min(y + br, out_height - 1);
                    sr += tmp_r[add * out_width + x];
                    sg += tmp_g[add * out_width + x];
                    sb += tmp_b[add * out_width + x];
                    int rem = std::clamp(y - br - 1, 0, out_height - 1);
                    sr -= tmp_r[rem * out_width + x];
                    sg -= tmp_g[rem * out_width + x];
                    sb -= tmp_b[rem * out_width + x];

                    buf_r[y * out_width + x] = sr * inv_w;
                    buf_g[y * out_width + x] = sg * inv_w;
                    buf_b[y * out_width + x] = sb * inv_w;
                }
            }
        }

        // Step 4: Composite bloom onto disk HDR buffer
        const double strength = cfg.bloom_strength;
#pragma omp parallel for
        for (int i = 0; i < (int)num_pixels; ++i)
        {
            hdr_disk[i].x() += strength * buf_r[i];
            hdr_disk[i].y() += strength * buf_g[i];
            hdr_disk[i].z() += strength * buf_b[i];
        }

        printf("Bloom applied (kernel radius = %d px)\n", kernel_r);
    }

    // =====================================================================
    // Write optional HDR output (raw linear, pre-tonemap)
    // =====================================================================
    if (!cfg.hdr_output.empty())
    {
        std::vector<float> hdr_float(num_pixels * 3);
#pragma omp parallel for
        for (int i = 0; i < (int)num_pixels; ++i)
        {
            Vector3d combined = hdr_disk[i] + hdr_sky[i];
            hdr_float[i * 3 + 0] = static_cast<float>(combined.x());
            hdr_float[i * 3 + 1] = static_cast<float>(combined.y());
            hdr_float[i * 3 + 2] = static_cast<float>(combined.z());
        }
        if (stbi_write_hdr(cfg.hdr_output.c_str(), out_width, out_height, 3, hdr_float.data()))
            printf("Wrote HDR: %s\n", cfg.hdr_output.c_str());
        else
            printf("ERROR: failed to write HDR: %s\n", cfg.hdr_output.c_str());
    }

    // =====================================================================
    // Tone mapping + quantization
    // =====================================================================
    printf("Tone mapping and writing output...\n");
#pragma omp parallel for
    for (int i = 0; i < (int)num_pixels; ++i)
    {
        Vector3d color = tonemap_disk(hdr_disk[i]) + hdr_sky[i];
        color = color.cwiseMax(0.0).cwiseMin(1.0);
        pixels[i * 3 + 0] = static_cast<BH_COLOR_CHANNEL_TYPE>(color.x() * 255);
        pixels[i * 3 + 1] = static_cast<BH_COLOR_CHANNEL_TYPE>(color.y() * 255);
        pixels[i * 3 + 2] = static_cast<BH_COLOR_CHANNEL_TYPE>(color.z() * 255);
    }

    stbi_write_tga(cfg.output_file.c_str(), out_width, out_height, 3, pixels.data());
    printf("Wrote %s\n", cfg.output_file.c_str());

    return 0;
}