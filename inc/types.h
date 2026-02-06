

#ifndef _BH_TYPES_H
#define _BH_TYPES_H

#include "common.h"

#define BH_COLOR_CHANNEL_TYPE unsigned char

struct sky_image_s
{
    int width = 0;
    int height = 0;
    BH_COLOR_CHANNEL_TYPE *red = nullptr;
    BH_COLOR_CHANNEL_TYPE *green = nullptr;
    BH_COLOR_CHANNEL_TYPE *blue = nullptr;

    BH_COLOR_CHANNEL_TYPE r(int x, int y) const { return red[y * width + x]; }
    BH_COLOR_CHANNEL_TYPE g(int x, int y) const { return green[y * width + x]; }
    BH_COLOR_CHANNEL_TYPE b(int x, int y) const { return blue[y * width + x]; }
};

#endif