
#include "sky.h"

#include <algorithm>
#include <cmath>

// ---------------------------------------------------------------------------
// Sample the sky image for a given world-space direction
// ---------------------------------------------------------------------------
Vector3d sample_sky(const sky_image_s &sky, const Vector3d &direction,
                    const Matrix3d &sky_rot, double offset_u, double offset_v)
{
    Vector3d dir_normalized = (sky_rot * direction).normalized();

    double u = 0.5 - (atan2(dir_normalized.z(), dir_normalized.x()) / (2.0 * M_PI));
    double v = 0.5 - (asin(std::clamp(dir_normalized.y(), -1.0, 1.0)) / M_PI);

    u += offset_u;
    v += offset_v;
    u -= floor(u);
    v = std::clamp(v - floor(v), 0.0, 1.0 - 1e-9);

    int x = std::clamp(static_cast<int>(u * sky.width), 0, sky.width - 1);
    int y = std::clamp(static_cast<int>(v * sky.height), 0, sky.height - 1);

    return Vector3d(static_cast<double>(sky.r(x, y)) / 255.0,
                    static_cast<double>(sky.g(x, y)) / 255.0,
                    static_cast<double>(sky.b(x, y)) / 255.0);
}

sky_image_s *load_sky_image(const char *filename)
{
    int width = 0, height = 0, channels = 0;
    unsigned char *rgba = stbi_load(filename, &width, &height, &channels, 4);
    if (!rgba)
    {
        fprintf(stderr, "Failed to load image\n");
        return NULL;
    }

    const size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
    unsigned char *red = new unsigned char[pixelCount];
    unsigned char *green = new unsigned char[pixelCount];
    unsigned char *blue = new unsigned char[pixelCount];

    for (size_t i = 0; i < pixelCount; ++i)
    {
        red[i] = rgba[i * 4 + 0];
        green[i] = rgba[i * 4 + 1];
        blue[i] = rgba[i * 4 + 2];
    }

    stbi_image_free(rgba);

    sky_image_s *image = new sky_image_s;
    image->width = width;
    image->height = height;
    image->red = red;
    image->green = green;
    image->blue = blue;

    return image;
}