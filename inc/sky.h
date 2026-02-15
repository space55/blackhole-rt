
#ifndef _BH_SKY_H
#define _BH_SKY_H

#include "types.h"
#include "vec_math.h"

sky_image_s *load_sky_image(const char *filename);

// Sample the sky image given a world-space direction, sky rotation matrix,
// and UV offsets.  Returns linear RGB in [0,1].
dvec3 sample_sky(const sky_image_s &sky, const dvec3 &direction,
                 const dmat3 &sky_rot, bh_real offset_u, bh_real offset_v);

#endif