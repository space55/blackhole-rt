
#ifndef _BH_RAY_H
#define _BH_RAY_H

#include "common.h"
#include "types.h"
#include "sky.h"

class ray_s
{
public:
    Vector3d pos;
    Vector3d vel;

    double distance_from_origin();
    double distance_from_origin_squared() const;
    double kerr_radius() const;
    static double event_horizon_radius();
    bool has_crossed_event_horizon() const;

    ray_s();
    ray_s(const Vector3d &position, const Vector3d &rotation_deg, double x, double y, const double fov_x, const double fov_y);

    bool advance(double dt);
    Vector3d project_to_sky(sky_image_s &sky);
};

#endif