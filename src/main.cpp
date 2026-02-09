
#include <stdio.h>
#include <algorithm>
#include <atomic>
#include <cmath>

#include "common.h"
#include "types.h"
#include "sky.h"
#include "ray.h"
#include "disk.h"
#include "blackhole.h"
#include "scene.h"

#include "stb_image_write.h"

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

    // Precompute sky rotation matrix
    const Matrix3d sky_rot = (AngleAxisd(cfg.sky_yaw * M_PI / 180.0, Vector3d::UnitY()) * AngleAxisd(cfg.sky_pitch * M_PI / 180.0, Vector3d::UnitX()) * AngleAxisd(cfg.sky_roll * M_PI / 180.0, Vector3d::UnitZ())).toRotationMatrix();

    std::vector<BH_COLOR_CHANNEL_TYPE> pixels(static_cast<size_t>(out_width) * static_cast<size_t>(out_height) * 3);

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

#pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < out_height; ++y)
    {
        for (int x = 0; x < out_width; ++x)
        {
            bool hit_black_hole = false;
            const int idx = y * out_width + x;

            ray_s ray(&bh,
                      camera_pos,
                      camera_rot,
                      (static_cast<double>(x) / out_width),
                      (static_cast<double>(y) / out_height),
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

            int done = pixels_done.fetch_add(1, std::memory_order_relaxed) + 1;
            if (done % out_width == 0)
            {
                printf("Progress: %d / %d pixels (%.1f%%)\n", done, total_pixels, 100.0 * done / total_pixels);
            }

            if (hit_black_hole)
            {
                // Behind horizon: just the accumulated disk emission (sky is black)
                Vector3d color = tonemap_disk(ray.accumulated_color);
                color = color.cwiseMax(0.0).cwiseMin(1.0);
                pixels[idx * 3 + 0] = static_cast<BH_COLOR_CHANNEL_TYPE>(color.x() * 255);
                pixels[idx * 3 + 1] = static_cast<BH_COLOR_CHANNEL_TYPE>(color.y() * 255);
                pixels[idx * 3 + 2] = static_cast<BH_COLOR_CHANNEL_TYPE>(color.z() * 255);
                continue;
            }
            else
            {
                // Composite: tone-mapped disk emission + transmittance * sky color
                Vector3d sky_color = ray.project_to_sky(*image, sky_rot,
                                                        cfg.sky_offset_u, cfg.sky_offset_v) *
                                     cfg.sky_brightness;
                double transmittance = 1.0 - ray.accumulated_opacity;
                Vector3d color = tonemap_disk(ray.accumulated_color) + transmittance * sky_color;
                color = color.cwiseMax(0.0).cwiseMin(1.0);

                pixels[idx * 3 + 0] = static_cast<BH_COLOR_CHANNEL_TYPE>(color.x() * 255);
                pixels[idx * 3 + 1] = static_cast<BH_COLOR_CHANNEL_TYPE>(color.y() * 255);
                pixels[idx * 3 + 2] = static_cast<BH_COLOR_CHANNEL_TYPE>(color.z() * 255);
            }
        }
    }

    printf("Disk samples accumulated: %d\n", disk_samples.load());
    stbi_write_tga(cfg.output_file.c_str(), out_width, out_height, 3, pixels.data());
    printf("Wrote %s\n", cfg.output_file.c_str());

    return 0;
}