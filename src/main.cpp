
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

#include "stb_image_write.h"

int main()
{
    sky_image_s *image = load_sky_image("hubble-skymap.jpg");
    if (!image)
    {
        printf("Failed to load sky image\n");
        return 1;
    }

    printf("Loaded sky image: %dx%d\n", image->width, image->height);

    // const int out_width = image->width;
    // const int out_height = image->height;
    const int out_width = 1024;
    const int out_height = 512;
    const double base_dt = 0.1;
    const double max_affine = 100.0;
    const double escape_r2 = 2500.0; // r=50, where H~0.0004 (negligible curvature)
    const double fov_x = 360.0;
    const double fov_y = 180.0;
    const Vector3d camera_pos(-15, 3, 0);
    // const Vector3d camera_rot(5, 70, -20); // pitch, yaw, roll in degrees
    const Vector3d camera_rot(0, 90, 0); // pitch, yaw, roll in degrees
    const double bh_mass = 1.0;
    const double bh_spin = 0.999;

    std::vector<BH_COLOR_CHANNEL_TYPE> pixels(static_cast<size_t>(out_width) * static_cast<size_t>(out_height) * 3);

    const int total_pixels = out_width * out_height;
    std::atomic<int> pixels_done(0);

    blackhole_s bh(bh_mass, bh_spin);
    printf("Black hole: M=%.1f, a=%.2f, r+=%.4f, r_isco=%.4f\n",
           bh.mass, bh.spin, bh.event_horizon_radius(), bh.isco_radius());

    const double r_plus = bh.event_horizon_radius();

    // Accretion disk: outer_r=20M, half-thickness scale=0.5, density=20.0, opacity=0.5
    accretion_disk_s disk(&bh, 20.0, 0.5, 20.0, 0.5);
    disk.emission_boost = 10.0; // <-- Tweak this to make the disk brighter/dimmer
    printf("Disk: inner_r=%.3f M, outer_r=%.1f M, boost=%.1f\n",
           disk.inner_r, disk.outer_r, disk.emission_boost);

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

            double affine = 0.0;
            while (affine < max_affine)
            {
                // Use Kerr-Schild radius for step sizing
                const double r_ks = ray.kerr_radius();
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
                if (ray.has_crossed_event_horizon() || !std::isfinite(ray.vel.squaredNorm()))
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
                Vector3d color = ray.accumulated_color;
                color = color.cwiseMax(0.0).cwiseMin(1.0);
                pixels[idx * 3 + 0] = static_cast<BH_COLOR_CHANNEL_TYPE>(color.x() * 255);
                pixels[idx * 3 + 1] = static_cast<BH_COLOR_CHANNEL_TYPE>(color.y() * 255);
                pixels[idx * 3 + 2] = static_cast<BH_COLOR_CHANNEL_TYPE>(color.z() * 255);
                continue;
            }
            else
            {
                // Composite: disk emission + transmittance * sky color
                Vector3d sky_color = ray.project_to_sky(*image);
                double transmittance = 1.0 - ray.accumulated_opacity;
                Vector3d color = ray.accumulated_color + transmittance * sky_color;
                color = color.cwiseMax(0.0).cwiseMin(1.0);

                pixels[idx * 3 + 0] = static_cast<BH_COLOR_CHANNEL_TYPE>(color.x() * 255);
                pixels[idx * 3 + 1] = static_cast<BH_COLOR_CHANNEL_TYPE>(color.y() * 255);
                pixels[idx * 3 + 2] = static_cast<BH_COLOR_CHANNEL_TYPE>(color.z() * 255);
            }
        }
    }

    printf("Disk samples accumulated: %d\n", disk_samples.load());
    stbi_write_tga("output.tga", out_width, out_height, 3, pixels.data());

    return 0;
}