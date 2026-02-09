
#include "ray.h"
#include "disk.h"

#include <algorithm>
#include <cmath>

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------
ray_s::ray_s()
    : bh(nullptr)
{
    pos = Vector3d(0, 0, 0);
    vel = Vector3d(0, 0, 1);
}

ray_s::ray_s(const blackhole_s *black_hole, const Vector3d &position, const Vector3d &rotation_deg,
             double x, double y, double fov_x, double fov_y)
    : bh(black_hole)
{
    pos = position;

    const double pitch = rotation_deg.x() * M_PI / 180.0;
    const double yaw = rotation_deg.y() * M_PI / 180.0;
    const double roll = rotation_deg.z() * M_PI / 180.0;

    Matrix3d rot = (AngleAxisd(yaw, Vector3d::UnitY()) * AngleAxisd(pitch, Vector3d::UnitX()) * AngleAxisd(roll, Vector3d::UnitZ())).toRotationMatrix();

    Vector3d forward = rot.col(2);
    Vector3d right = rot.col(0);
    Vector3d up = rot.col(1);

    double x_angle = (x - 0.5) * fov_x;
    double y_angle = (0.5 - y) * fov_y;

    const double x_rad = x_angle * M_PI / 180.0;
    const double y_rad = y_angle * M_PI / 180.0;

    const double local_forward = cos(y_rad) * cos(x_rad);
    const double local_up = sin(y_rad);
    const double local_right = cos(y_rad) * sin(x_rad);

    vel = forward * local_forward + up * local_up + right * local_right;
}

// ---------------------------------------------------------------------------
// Distance helpers
// ---------------------------------------------------------------------------
double ray_s::distance_from_origin()
{
    return pos.norm();
}

double ray_s::distance_from_origin_squared() const
{
    return pos.squaredNorm();
}

// ---------------------------------------------------------------------------
// Horizon / radius queries
// ---------------------------------------------------------------------------
bool ray_s::has_crossed_event_horizon() const
{
    return bh->ks_radius(pos) <= bh->event_horizon_radius();
}

double ray_s::kerr_radius() const
{
    return bh->ks_radius(pos);
}

// ---------------------------------------------------------------------------
// RK4 geodesic integrator with substep horizon checking
// ---------------------------------------------------------------------------
bool ray_s::advance(double dt)
{
    struct Deriv
    {
        Vector3d dx;
        Vector3d dv;
    };

    const double r_plus = bh->event_horizon_radius();

    auto deriv = [&](const Vector3d &x, const Vector3d &v, bool &inside) -> Deriv
    {
        if (bh->ks_radius(x) <= r_plus)
        {
            inside = true;
            return {Vector3d::Zero(), Vector3d::Zero()};
        }
        return {v, bh->geodesic_accel(x, v)};
    };

    bool inside = false;
    const Deriv k1 = deriv(pos, vel, inside);
    if (inside)
        return false;
    const Deriv k2 = deriv(pos + 0.5 * dt * k1.dx, vel + 0.5 * dt * k1.dv, inside);
    if (inside)
        return false;
    const Deriv k3 = deriv(pos + 0.5 * dt * k2.dx, vel + 0.5 * dt * k2.dv, inside);
    if (inside)
        return false;
    const Deriv k4 = deriv(pos + dt * k3.dx, vel + dt * k3.dv, inside);
    if (inside)
        return false;

    pos += (dt / 6.0) * (k1.dx + 2.0 * k2.dx + 2.0 * k3.dx + k4.dx);
    vel += (dt / 6.0) * (k1.dv + 2.0 * k2.dv + 2.0 * k3.dv + k4.dv);
    return true;
}

// ---------------------------------------------------------------------------
// Volumetric disk sampling with relativistic radiative transfer
// ---------------------------------------------------------------------------
void ray_s::sample_disk(const accretion_disk_s &disk, double ds)
{
    if (accumulated_opacity >= 0.999)
        return;

    if (!disk.contains(pos))
        return;

    const Vector3d j = disk.emissivity(pos);
    const double alpha = disk.absorption(pos);

    const double u0 = bh->compute_u0_null(pos, vel);
    const Vector4d photon_k(u0, vel.x(), vel.y(), vel.z());

    const Vector4d gas_u = disk.gas_four_velocity(pos);

    const MetricResult m = bh->metric(pos);
    const double g = accretion_disk_s::redshift_factor(photon_k, gas_u, m.g);

    const double g_clamped = std::clamp(g, 0.01, 10.0);
    const double g3 = g_clamped * g_clamped * g_clamped;

    Vector3d j_obs = g3 * j;
    if (g_clamped > 1.0)
    {
        j_obs.z() *= std::min(g_clamped, 2.0);
    }
    else
    {
        j_obs.x() *= std::min(1.0 / g_clamped, 2.0);
        j_obs.z() *= g_clamped;
    }

    const double transmittance = 1.0 - accumulated_opacity;
    const double dtau = alpha * ds;
    const double absorption_factor = 1.0 - exp(-dtau);

    accumulated_color += transmittance * absorption_factor * j_obs / std::max(alpha, 1e-12);
    accumulated_opacity += transmittance * absorption_factor;
    accumulated_opacity = std::min(accumulated_opacity, 1.0);
}

// ---------------------------------------------------------------------------
// Project ray direction onto equirectangular sky map
// ---------------------------------------------------------------------------
Vector3d ray_s::project_to_sky(sky_image_s &sky, const Matrix3d &sky_rot,
                               double offset_u, double offset_v)
{
    Vector3d dir_normalized = (sky_rot * vel).normalized();

    double u = 0.5 - (atan2(dir_normalized.z(), dir_normalized.x()) / (2.0 * M_PI));
    double v = 0.5 - (asin(dir_normalized.y()) / M_PI);

    // Apply UV offsets and wrap to [0, 1)
    u = u + offset_u;
    v = v + offset_v;
    u = u - floor(u);
    v = std::clamp(v - floor(v), 0.0, 1.0 - 1e-9);

    int x = std::clamp(static_cast<int>(u * sky.width), 0, sky.width - 1);
    int y = std::clamp(static_cast<int>(v * sky.height), 0, sky.height - 1);

    unsigned char r = sky.r(x, y);
    unsigned char g = sky.g(x, y);
    unsigned char b = sky.b(x, y);

    return Vector3d(static_cast<double>(r) / 255.0,
                    static_cast<double>(g) / 255.0,
                    static_cast<double>(b) / 255.0);
}