
#ifndef _BH_SKY_H
#define _BH_SKY_H

#include "types.h"
#include "common.h"

#include "stb_image.h"

sky_image_s *load_sky_image(const char *filename);

// Sample the sky image given a world-space direction, sky rotation matrix,
// and UV offsets.  Returns linear RGB in [0,1].
Vector3d sample_sky(const sky_image_s &sky, const Vector3d &direction,
                    const Matrix3d &sky_rot, double offset_u, double offset_v);

#endif