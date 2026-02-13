
#include <stdio.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <vector>

#include "physics.h"
#include "types.h"
#include "sky.h"
#include "scene.h"

#include "stb_image_write.h"

#ifdef HAS_OPENEXR
#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfHeader.h>
#include <OpenEXR/ImfFrameBuffer.h>
#endif

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

    // --- Physics parameters (replaces blackhole_s + accretion_disk_s) ----
    PhysicsParams pp = make_physics_params(
        cfg.bh_mass, cfg.bh_spin,
        cfg.disk_outer_r, cfg.disk_thickness,
        cfg.disk_density, cfg.disk_opacity,
        cfg.disk_emission_boost, cfg.disk_color_variation,
        cfg.disk_turbulence, cfg.time,
        cfg.disk_flat_mode);

    printf("Black hole: M=%.1f, a=%.2f, r+=%.4f, r_isco=%.4f\n",
           pp.bh_mass, pp.bh_spin, pp.r_plus, pp.disk_inner_r);

    // --- Derived constants from config -----------------------------------
    const int out_width = cfg.output_width;
    const int out_height = cfg.output_height;
    const double base_dt = cfg.base_dt;
    const double max_affine = cfg.max_affine;
    const double escape_r2 = cfg.escape_radius * cfg.escape_radius;
    const double fov_x = cfg.fov_x;
    const double fov_y = cfg.fov_y;
    const dvec3 camera_pos(cfg.camera_x, cfg.camera_y, cfg.camera_z);

    // Precompute camera rotation matrix once (avoids trig per ray)
    const dmat3 cam_rot_matrix = dmat3::rotation_y(cfg.camera_yaw * M_PI / 180.0) *
                                 dmat3::rotation_x(cfg.camera_pitch * M_PI / 180.0) *
                                 dmat3::rotation_z(cfg.camera_roll * M_PI / 180.0);

    // Extract camera basis vectors
    const dvec3 cam_right = cam_rot_matrix.col(0);
    const dvec3 cam_up = cam_rot_matrix.col(1);
    const dvec3 cam_fwd = cam_rot_matrix.col(2);

    // Precompute sky rotation matrix
    const dmat3 sky_rot = dmat3::rotation_y(cfg.sky_yaw * M_PI / 180.0) *
                          dmat3::rotation_x(cfg.sky_pitch * M_PI / 180.0) *
                          dmat3::rotation_z(cfg.sky_roll * M_PI / 180.0);

    // HDR framebuffers: disk emission separate from sky for tone mapping
    const size_t num_pixels = static_cast<size_t>(out_width) * static_cast<size_t>(out_height);
    std::vector<dvec3> hdr_disk(num_pixels);
    std::vector<dvec3> hdr_sky(num_pixels);
    std::vector<double> hdr_alpha(num_pixels, 0.0); // disk opacity (0 = transparent, 1 = opaque/BH)
    std::vector<BH_COLOR_CHANNEL_TYPE> pixels(num_pixels * 3);

    const int total_pixels = out_width * out_height;
    std::atomic<int> pixels_done(0);

    const double r_plus = pp.r_plus;

    // --- Tone mapping ----------------------------------------------------
    const double disk_tonemap_compression = cfg.tonemap_compression;
    const double tonemap_c = pow(10.0, disk_tonemap_compression * 2.0) - 1.0;
    const double tonemap_norm = 1.0 / log(1.0 + tonemap_c);
    auto tonemap_disk = [&](const dvec3 &hdr) -> dvec3
    {
        if (tonemap_c < 1e-6)
            return hdr; // compression ≈ 0 → identity
        return dvec3(
            log(1.0 + tonemap_c * fmax(hdr.x, 0.0)) * tonemap_norm,
            log(1.0 + tonemap_c * fmax(hdr.y, 0.0)) * tonemap_norm,
            log(1.0 + tonemap_c * fmax(hdr.z, 0.0)) * tonemap_norm);
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
        GPUSceneParams gpu_params;
        fill_gpu_params(gpu_params, cfg, pp, cam_rot_matrix);

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
            hdr_disk[i] = dvec3(gpu_results[i].disk_r,
                                gpu_results[i].disk_g,
                                gpu_results[i].disk_b);
            hdr_alpha[i] = 1.0 - (double)gpu_results[i].sky_weight;

            if (gpu_results[i].sky_weight > 1e-6f)
            {
                dvec3 exit_dir(gpu_results[i].exit_vx,
                               gpu_results[i].exit_vy,
                               gpu_results[i].exit_vz);
                if (exit_dir.squaredNorm() > 1e-24)
                {
                    dvec3 sky_color = sample_sky(*image, exit_dir, sky_rot,
                                                 cfg.sky_offset_u, cfg.sky_offset_v);
                    hdr_sky[i] = sky_color *
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
            dvec3 pixel_color;
            dvec3 pixel_sky;
            double pixel_alpha = 0.0;

            for (int sy = 0; sy < aa_grid; ++sy)
            {
                for (int sx = 0; sx < aa_grid; ++sx)
                {
                    // Sub-pixel offset: stratified sample at center of each grid cell
                    const double sub_x = (x + (sx + 0.5) * inv_aa) / out_width;
                    const double sub_y = (y + (sy + 0.5) * inv_aa) / out_height;

                    bool hit_black_hole = false;

                    dvec3 pos, vel;
                    init_ray(pos, vel, camera_pos, cam_right, cam_up, cam_fwd,
                             sub_x, sub_y, fov_x, fov_y);

                    double cached_ks_r = ks_radius(pos, pp.bh_spin);
                    dvec3 acc_color;
                    double acc_opacity = 0.0;

                    double affine = 0.0;
                    while (affine < max_affine)
                    {
                        const double r_ks = cached_ks_r;
                        const double delta = std::max(r_ks - r_plus, 0.01);
                        double step_dt = base_dt * std::clamp(delta * delta, 0.0001, 1.0);

                        // Reduce step size when near the disk
                        const double disk_inner_guard = pp.disk_flat_mode ? pp.disk_inner_r * 0.4 : pp.disk_inner_r * 0.8;
                        const double disk_outer_guard = pp.disk_flat_mode ? pp.disk_outer_r * 1.6 : pp.disk_outer_r * 1.2;
                        if (r_ks >= disk_inner_guard && r_ks <= disk_outer_guard)
                        {
                            const double h = disk_half_thickness(r_ks, pp);
                            const double y_dist = fabs(pos.y);
                            if (y_dist < 5.0 * h)
                            {
                                step_dt = std::min(step_dt, std::max(0.3 * h, 0.005));
                            }
                        }

                        if (!advance_ray(pos, vel, cached_ks_r, step_dt, pp))
                        {
                            hit_black_hole = true;
                            break;
                        }

                        // Sample disk emission along the ray — cheap guard
                        // skips the full function when clearly outside the disk
                        const double prev_opacity = acc_opacity;
                        const double samp_inner_guard = pp.disk_flat_mode ? pp.disk_inner_r * 0.4 : pp.disk_inner_r * 0.8;
                        const double samp_outer_guard = pp.disk_flat_mode ? pp.disk_outer_r * 1.6 : pp.disk_outer_r * 1.3;
                        if (fabs(pos.y) < pp.disk_thickness * 15.0 &&
                            cached_ks_r >= samp_inner_guard &&
                            cached_ks_r <= samp_outer_guard)
                        {
                            sample_disk_volume(pos, vel, step_dt, acc_color, acc_opacity, cached_ks_r, pp);
                        }
                        if (acc_opacity > prev_opacity)
                            disk_samples.fetch_add(1, std::memory_order_relaxed);

                        affine += step_dt;
                        if (cached_ks_r <= r_plus || !bh_isfinite(vel.squaredNorm()))
                        {
                            hit_black_hole = true;
                            break;
                        }
                        if (pos.squaredNorm() > escape_r2)
                        {
                            break;
                        }
                        if (acc_opacity > 0.99)
                        {
                            break;
                        }
                    }

                    if (hit_black_hole)
                    {
                        pixel_color += acc_color;
                        pixel_alpha += 1.0; // fully opaque (disk or black hole)
                    }
                    else
                    {
                        dvec3 sky_color = sample_sky(*image, vel, sky_rot,
                                                     cfg.sky_offset_u, cfg.sky_offset_v);
                        double transmittance = 1.0 - acc_opacity;
                        pixel_color += acc_color;
                        pixel_sky += transmittance * sky_color;
                        pixel_alpha += acc_opacity;
                    }

                } // sx
            } // sy

            // Average all samples and store in HDR buffers
            pixel_color *= inv_spp;
            pixel_sky *= inv_spp;
            pixel_alpha *= inv_spp;
            hdr_disk[idx] = pixel_color;
            hdr_sky[idx] = pixel_sky;
            hdr_alpha[idx] = pixel_alpha;

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
    // Write optional HDR output (raw linear, pre-tonemap, pre-bloom)
    // =====================================================================
    if (!cfg.hdr_output.empty())
    {
        std::vector<float> hdr_float(num_pixels * 3);
#pragma omp parallel for
        for (int i = 0; i < (int)num_pixels; ++i)
        {
            dvec3 combined = hdr_disk[i] + hdr_sky[i] * cfg.sky_brightness;
            hdr_float[i * 3 + 0] = static_cast<float>(combined.x);
            hdr_float[i * 3 + 1] = static_cast<float>(combined.y);
            hdr_float[i * 3 + 2] = static_cast<float>(combined.z);
        }
        if (stbi_write_hdr(cfg.hdr_output.c_str(), out_width, out_height, 3, hdr_float.data()))
            printf("Wrote HDR: %s\n", cfg.hdr_output.c_str());
        else
            printf("ERROR: failed to write HDR: %s\n", cfg.hdr_output.c_str());
    }

    // =====================================================================
    // Write optional OpenEXR output (float32, multi-layer, ZIP compressed)
    // Written before bloom so EXR contains clean, unprocessed radiance data.
    // =====================================================================
#ifdef HAS_OPENEXR
    if (!cfg.exr_output.empty())
    {
        try
        {
            Imf::Header header(out_width, out_height);
            header.compression() = Imf::ZIP_COMPRESSION;

            // Combined RGB (disk + sky) — the "beauty pass"
            header.channels().insert("R", Imf::Channel(Imf::FLOAT));
            header.channels().insert("G", Imf::Channel(Imf::FLOAT));
            header.channels().insert("B", Imf::Channel(Imf::FLOAT));
            header.channels().insert("A", Imf::Channel(Imf::FLOAT));

            // Separate disk emission layer (with its own alpha)
            header.channels().insert("disk.R", Imf::Channel(Imf::FLOAT));
            header.channels().insert("disk.G", Imf::Channel(Imf::FLOAT));
            header.channels().insert("disk.B", Imf::Channel(Imf::FLOAT));
            header.channels().insert("disk.A", Imf::Channel(Imf::FLOAT));

            // Separate sky layer (brightness-scaled)
            header.channels().insert("sky.R", Imf::Channel(Imf::FLOAT));
            header.channels().insert("sky.G", Imf::Channel(Imf::FLOAT));
            header.channels().insert("sky.B", Imf::Channel(Imf::FLOAT));

            // Build float32 scanline buffers
            std::vector<float> exr_r(num_pixels), exr_g(num_pixels), exr_b(num_pixels);
            std::vector<float> exr_a(num_pixels);
            std::vector<float> exr_disk_r(num_pixels), exr_disk_g(num_pixels), exr_disk_b(num_pixels);
            std::vector<float> exr_disk_a(num_pixels);
            std::vector<float> exr_sky_r(num_pixels), exr_sky_g(num_pixels), exr_sky_b(num_pixels);

#pragma omp parallel for
            for (int i = 0; i < (int)num_pixels; ++i)
            {
                dvec3 sky_scaled = hdr_sky[i] * cfg.sky_brightness;
                dvec3 combined = hdr_disk[i] + sky_scaled;

                exr_r[i] = static_cast<float>(combined.x);
                exr_g[i] = static_cast<float>(combined.y);
                exr_b[i] = static_cast<float>(combined.z);
                exr_a[i] = 1.0f; // beauty pass is fully composited — always opaque

                exr_disk_r[i] = static_cast<float>(hdr_disk[i].x);
                exr_disk_g[i] = static_cast<float>(hdr_disk[i].y);
                exr_disk_b[i] = static_cast<float>(hdr_disk[i].z);
                exr_disk_a[i] = static_cast<float>(hdr_alpha[i]);

                exr_sky_r[i] = static_cast<float>(sky_scaled.x);
                exr_sky_g[i] = static_cast<float>(sky_scaled.y);
                exr_sky_b[i] = static_cast<float>(sky_scaled.z);
            }

            const size_t stride = sizeof(float);
            const size_t scanline = sizeof(float) * out_width;

            Imf::FrameBuffer fb;
            fb.insert("R", Imf::Slice(Imf::FLOAT, (char *)exr_r.data(), stride, scanline));
            fb.insert("G", Imf::Slice(Imf::FLOAT, (char *)exr_g.data(), stride, scanline));
            fb.insert("B", Imf::Slice(Imf::FLOAT, (char *)exr_b.data(), stride, scanline));
            fb.insert("A", Imf::Slice(Imf::FLOAT, (char *)exr_a.data(), stride, scanline));

            fb.insert("disk.R", Imf::Slice(Imf::FLOAT, (char *)exr_disk_r.data(), stride, scanline));
            fb.insert("disk.G", Imf::Slice(Imf::FLOAT, (char *)exr_disk_g.data(), stride, scanline));
            fb.insert("disk.B", Imf::Slice(Imf::FLOAT, (char *)exr_disk_b.data(), stride, scanline));
            fb.insert("disk.A", Imf::Slice(Imf::FLOAT, (char *)exr_disk_a.data(), stride, scanline));

            fb.insert("sky.R", Imf::Slice(Imf::FLOAT, (char *)exr_sky_r.data(), stride, scanline));
            fb.insert("sky.G", Imf::Slice(Imf::FLOAT, (char *)exr_sky_g.data(), stride, scanline));
            fb.insert("sky.B", Imf::Slice(Imf::FLOAT, (char *)exr_sky_b.data(), stride, scanline));

            Imf::OutputFile file(cfg.exr_output.c_str(), header);
            file.setFrameBuffer(fb);
            file.writePixels(out_height);

            printf("Wrote EXR: %s\n", cfg.exr_output.c_str());
        }
        catch (const std::exception &e)
        {
            printf("ERROR: failed to write EXR: %s (%s)\n",
                   cfg.exr_output.c_str(), e.what());
        }
    }
#else
    if (!cfg.exr_output.empty())
    {
        printf("WARNING: EXR output requested but OpenEXR was not found at build time.\n");
        printf("         Install OpenEXR (brew install openexr) and reconfigure with cmake.\n");
    }
#endif

    // =====================================================================
    // Post-processing: Bloom (applied only to TGA/JPG output)
    // =====================================================================
    if (cfg.bloom_strength > 1e-6)
    {
        printf("Applying bloom: strength=%.2f  threshold=%.2f  radius=%.3f\n",
               cfg.bloom_strength, cfg.bloom_threshold, cfg.bloom_radius);

        const double diag = sqrt((double)(out_width * out_width + out_height * out_height));
        const int kernel_r = std::max((int)(cfg.bloom_radius * diag), 1);
        const double sigma = kernel_r / 3.0;

        const int BOX_PASSES = 3;
        int box_radii[BOX_PASSES];
        {
            double w_ideal = sqrt(12.0 * sigma * sigma / BOX_PASSES + 1.0);
            int w_lo = ((int)w_ideal) | 1;
            if (w_lo < 1)
                w_lo = 1;
            int w_hi = w_lo + 2;
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

        std::vector<float> buf_r(num_pixels, 0.0f);
        std::vector<float> buf_g(num_pixels, 0.0f);
        std::vector<float> buf_b(num_pixels, 0.0f);
        const double thr = cfg.bloom_threshold;
#pragma omp parallel for
        for (int i = 0; i < (int)num_pixels; ++i)
        {
            dvec3 tm = tonemap_disk(hdr_disk[i]);
            double lum = 0.2126 * tm.x + 0.7152 * tm.y + 0.0722 * tm.z;
            if (lum > thr)
            {
                double scale = (lum - thr) / std::max(lum, 1e-12);
                buf_r[i] = (float)(hdr_disk[i].x * scale);
                buf_g[i] = (float)(hdr_disk[i].y * scale);
                buf_b[i] = (float)(hdr_disk[i].z * scale);
            }
        }

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
                    int add = std::min(x + br, out_width - 1);
                    sr += buf_r[row + add];
                    sg += buf_g[row + add];
                    sb += buf_b[row + add];
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

        // Composite bloom onto disk HDR buffer
        const double strength = cfg.bloom_strength;
#pragma omp parallel for
        for (int i = 0; i < (int)num_pixels; ++i)
        {
            hdr_disk[i].x += strength * buf_r[i];
            hdr_disk[i].y += strength * buf_g[i];
            hdr_disk[i].z += strength * buf_b[i];
        }

        printf("Bloom applied (kernel radius = %d px)\n", kernel_r);
    }

    // =====================================================================
    // Tone mapping + quantization
    // =====================================================================
    printf("Tone mapping and writing output...\n");
#pragma omp parallel for
    for (int i = 0; i < (int)num_pixels; ++i)
    {
        dvec3 color = tonemap_disk(hdr_disk[i]) * cfg.exposure + hdr_sky[i] * cfg.sky_brightness;
        color = color.cwiseMax(0.0).cwiseMin(1.0);
        pixels[i * 3 + 0] = static_cast<BH_COLOR_CHANNEL_TYPE>(color.x * 255);
        pixels[i * 3 + 1] = static_cast<BH_COLOR_CHANNEL_TYPE>(color.y * 255);
        pixels[i * 3 + 2] = static_cast<BH_COLOR_CHANNEL_TYPE>(color.z * 255);
    }

    stbi_write_tga(cfg.output_file.c_str(), out_width, out_height, 3, pixels.data());
    printf("Wrote %s\n", cfg.output_file.c_str());

    // Write JPEG thumbnail
    if (!cfg.jpg_output.empty())
    {
        if (stbi_write_jpg(cfg.jpg_output.c_str(), out_width, out_height, 3, pixels.data(), 90))
            printf("Wrote JPEG thumbnail: %s\n", cfg.jpg_output.c_str());
        else
            printf("ERROR: failed to write JPEG thumbnail: %s\n", cfg.jpg_output.c_str());
    }

    return 0;
}