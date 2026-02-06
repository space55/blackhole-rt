
#include <stdio.h>
#include <algorithm>
#include <atomic>
#include <cmath>

#include "common.h"
#include "types.h"
#include "sky.h"
#include "ray.h"

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

    std::vector<BH_COLOR_CHANNEL_TYPE> pixels(static_cast<size_t>(out_width) * static_cast<size_t>(out_height) * 3);

    const int total_pixels = out_width * out_height;
    std::atomic<int> pixels_done(0);

#pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < out_height; ++y)
    {
        for (int x = 0; x < out_width; ++x)
        {
            bool hit_black_hole = false;
            const int idx = y * out_width + x;

            ray_s ray(Vector3d(-15, 0, 0),
                      Vector3d(0, 90, 0),
                      (static_cast<double>(x) / out_width),
                      (static_cast<double>(y) / out_height),
                      60.0,
                      30.0);

            double affine = 0.0;
            while (affine < max_affine)
            {
                const double r2 = ray.distance_from_origin_squared();
                const double step_dt = base_dt * std::clamp(r2 / 25.0, 0.01, 1.0);
                ray.advance(step_dt);
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
            }

            int done = pixels_done.fetch_add(1, std::memory_order_relaxed) + 1;
            if (done % out_width == 0)
            {
                printf("Progress: %d / %d pixels (%.1f%%)\n", done, total_pixels, 100.0 * done / total_pixels);
            }

            if (hit_black_hole)
            {
                pixels[idx * 3 + 0] = 0;
                pixels[idx * 3 + 1] = 0;
                pixels[idx * 3 + 2] = 0;
                continue;
            }
            else
            {
                Vector3d color = ray.project_to_sky(*image);

                pixels[idx * 3 + 0] = static_cast<BH_COLOR_CHANNEL_TYPE>(color.x() * 255);
                pixels[idx * 3 + 1] = static_cast<BH_COLOR_CHANNEL_TYPE>(color.y() * 255);
                pixels[idx * 3 + 2] = static_cast<BH_COLOR_CHANNEL_TYPE>(color.z() * 255);
            }
        }
    }

    stbi_write_tga("output.tga", out_width, out_height, 3, pixels.data());

    return 0;
}