
#include "sky.h"

sky_image_s *load_sky_image(const char *filename)
{
    int width = 0, height = 0, channels = 0;
    unsigned char *rgba = stbi_load("hubble-skymap.jpg", &width, &height, &channels, 4);
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