
#ifndef _BH_RAY_H
#define _BH_RAY_H

#include "common.h"
#include "types.h"
#include "sky.h"
#include "blackhole.h"

struct accretion_disk_s; // forward declaration

class ray_s
{
public:
    const blackhole_s *bh;

    Vector3d pos;
    Vector3d vel;

    // Radiative transfer accumulators
    Vector3d accumulated_color = Vector3d(0, 0, 0);
    double accumulated_opacity = 0.0;

    // Cached KS radius from last advance(), avoids redundant recomputation
    double cached_ks_r = 0.0;

    double distance_from_origin();
    double distance_from_origin_squared() const;
    double kerr_radius() const;
    bool has_crossed_event_horizon() const;

    ray_s();
    ray_s(const blackhole_s *black_hole, const Vector3d &position, const Vector3d &rotation_deg,
          double x, double y, double fov_x, double fov_y);
    ray_s(const blackhole_s *black_hole, const Vector3d &position, const Matrix3d &cam_rot,
          double x, double y, double fov_x, double fov_y);

    bool advance(double dt);
    void sample_disk(const accretion_disk_s &disk, double ds);
    Vector3d project_to_sky(sky_image_s &sky, const Matrix3d &sky_rot,
                            double offset_u, double offset_v);
};

#endif