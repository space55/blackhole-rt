
#include "ray.h"

#include <algorithm>
#include <cmath>

namespace
{
    constexpr double kBlackHoleMass = 1.0;
    constexpr double kBlackHoleSpin = 0.8;
    constexpr double kMetricEps = 1e-4;

    struct MetricResult
    {
        Matrix4d g;
        Matrix4d g_inv;
    };

    static double kerr_schild_radius(double x, double y, double z, double a)
    {
        const double rho2 = x * x + y * y + z * z;
        const double a2 = a * a;
        const double term = rho2 - a2;
        const double disc = term * term + 4.0 * a2 * y * y;
        const double r2 = 0.5 * (term + sqrt(disc));
        return sqrt(std::max(r2, 1e-12));
    }

    static MetricResult kerr_schild_metric(const Vector3d &pos, double mass, double spin)
    {
        const double x = pos.x();
        const double y = pos.y();
        const double z = pos.z();

        const double a = spin;
        const double a2 = a * a;
        const double r = kerr_schild_radius(x, y, z, a);
        const double r2 = r * r;

        const double denom = r2 + a2;
        const double lx = (r * x + a * z) / denom;
        const double ly = y / r;
        const double lz = (r * z - a * x) / denom;

        const double H = (mass * r * r * r) / (r2 * r2 + a2 * y * y);

        Vector4d l;
        l << 1.0, lx, ly, lz;

        Matrix4d eta = Matrix4d::Zero();
        eta(0, 0) = -1.0;
        eta(1, 1) = 1.0;
        eta(2, 2) = 1.0;
        eta(3, 3) = 1.0;

        Matrix4d g = eta + 2.0 * H * (l * l.transpose());

        Vector4d l_up;
        l_up << -l(0), l(1), l(2), l(3);
        Matrix4d g_inv = eta - 2.0 * H * (l_up * l_up.transpose());

        return {g, g_inv};
    }

    static double compute_u0_null(const Vector3d &pos, const Vector3d &vel, double mass, double spin)
    {
        const MetricResult metric = kerr_schild_metric(pos, mass, spin);

        const double g00 = metric.g(0, 0);
        const double g01 = metric.g(0, 1);
        const double g02 = metric.g(0, 2);
        const double g03 = metric.g(0, 3);

        const double v1 = vel.x();
        const double v2 = vel.y();
        const double v3 = vel.z();

        const double b = 2.0 * (g01 * v1 + g02 * v2 + g03 * v3);
        const double c = metric.g(1, 1) * v1 * v1 + metric.g(2, 2) * v2 * v2 + metric.g(3, 3) * v3 * v3 + 2.0 * (metric.g(1, 2) * v1 * v2 + metric.g(1, 3) * v1 * v3 + metric.g(2, 3) * v2 * v3);

        const double disc = b * b - 4.0 * g00 * c;
        const double sqrt_disc = sqrt(std::max(disc, 0.0));

        const double u0_a = (-b + sqrt_disc) / (2.0 * g00);
        const double u0_b = (-b - sqrt_disc) / (2.0 * g00);

        return (u0_a < 0.0) ? u0_a : u0_b;
    }

    static void metric_partials(const Vector3d &pos, double mass, double spin, Matrix4d partials[3])
    {
        for (int i = 0; i < 3; ++i)
        {
            Vector3d step = Vector3d::Zero();
            step[i] = kMetricEps;
            Matrix4d g_plus = kerr_schild_metric(pos + step, mass, spin).g;
            Matrix4d g_minus = kerr_schild_metric(pos - step, mass, spin).g;
            partials[i] = (g_plus - g_minus) / (2.0 * kMetricEps);
        }
    }

    static Vector3d geodesic_accel(const Vector3d &pos, const Vector3d &vel, double mass, double spin)
    {
        const MetricResult metric = kerr_schild_metric(pos, mass, spin);
        Matrix4d partials[3];
        metric_partials(pos, mass, spin, partials);

        const double u0 = compute_u0_null(pos, vel, mass, spin);
        Vector4d u;
        u << u0, vel.x(), vel.y(), vel.z();

        Vector3d accel = Vector3d::Zero();

        for (int mu = 1; mu <= 3; ++mu)
        {
            double sum = 0.0;
            for (int alpha = 0; alpha <= 3; ++alpha)
            {
                for (int beta = 0; beta <= 3; ++beta)
                {
                    double gamma = 0.0;
                    for (int nu = 0; nu <= 3; ++nu)
                    {
                        const double d_alpha_g = (alpha == 0) ? 0.0 : partials[alpha - 1](beta, nu);
                        const double d_beta_g = (beta == 0) ? 0.0 : partials[beta - 1](alpha, nu);
                        const double d_nu_g = (nu == 0) ? 0.0 : partials[nu - 1](alpha, beta);
                        gamma += 0.5 * metric.g_inv(mu, nu) * (d_alpha_g + d_beta_g - d_nu_g);
                    }
                    sum += gamma * u(alpha) * u(beta);
                }
            }
            accel[mu - 1] = -sum;
        }

        return accel;
    }
}

ray_s::ray_s()
{
    pos = Vector3d(0, 0, 0);
    vel = Vector3d(0, 0, 1);
}

ray_s::ray_s(const Vector3d &position, const Vector3d &rotation_deg, double x, double y, const double fov_x, const double fov_y)
{
    pos = position;

    // rotation_deg: (pitch, yaw, roll) in degrees
    const double pitch = rotation_deg.x() * M_PI / 180.0;
    const double yaw = rotation_deg.y() * M_PI / 180.0;
    const double roll = rotation_deg.z() * M_PI / 180.0;

    // Build camera basis via yaw (Y) -> pitch (X) -> roll (Z)
    Matrix3d rot = (AngleAxisd(yaw, Vector3d::UnitY()) * AngleAxisd(pitch, Vector3d::UnitX()) * AngleAxisd(roll, Vector3d::UnitZ())).toRotationMatrix();

    Vector3d forward = rot.col(2); // local Z -> forward
    Vector3d right = rot.col(0);   // local X -> right
    Vector3d up = rot.col(1);      // local Y -> up

    double x_angle = (x - 0.5) * fov_x;
    double y_angle = (0.5 - y) * fov_y;
    double velocity = 1.0;

    const double x_rad = x_angle * M_PI / 180.0;
    const double y_rad = y_angle * M_PI / 180.0;

    const double local_forward = cos(y_rad) * cos(x_rad);
    const double local_up = sin(y_rad);
    const double local_right = cos(y_rad) * sin(x_rad);

    vel = velocity * (forward * local_forward + up * local_up + right * local_right);
}

double ray_s::distance_from_origin()
{
    return pos.norm();
}

double ray_s::distance_from_origin_squared() const
{
    return pos.squaredNorm();
}

bool ray_s::has_crossed_event_horizon() const
{
    const double r = kerr_schild_radius(pos.x(), pos.y(), pos.z(), kBlackHoleSpin);
    const double mass = kBlackHoleMass;
    const double spin = kBlackHoleSpin;
    const double r_plus = mass + sqrt(std::max(mass * mass - spin * spin, 0.0));
    // Small buffer catches rays in the numerically unstable zone
    // just outside the horizon where finite-difference Christoffels blow up
    return r <= r_plus * 1.02;
}

void ray_s::advance(double dt)
{
    struct Deriv
    {
        Vector3d dx;
        Vector3d dv;
    };

    auto deriv = [&](const Vector3d &x, const Vector3d &v) -> Deriv
    {
        Deriv d;
        d.dx = v;
        d.dv = geodesic_accel(x, v, kBlackHoleMass, kBlackHoleSpin);
        return d;
    };

    const Deriv k1 = deriv(pos, vel);
    const Deriv k2 = deriv(pos + 0.5 * dt * k1.dx, vel + 0.5 * dt * k1.dv);
    const Deriv k3 = deriv(pos + 0.5 * dt * k2.dx, vel + 0.5 * dt * k2.dv);
    const Deriv k4 = deriv(pos + dt * k3.dx, vel + dt * k3.dv);

    pos += (dt / 6.0) * (k1.dx + 2.0 * k2.dx + 2.0 * k3.dx + k4.dx);
    vel += (dt / 6.0) * (k1.dv + 2.0 * k2.dv + 2.0 * k3.dv + k4.dv);
}

Vector3d ray_s::project_to_sky(sky_image_s &sky)
{
    // This is sure as hell to cause an issue if the ray is pointing through a black hole. Come back to this.
    Vector3d dir_normalized = vel.normalized();

    double u = 0.5 + (atan2(dir_normalized.z(), dir_normalized.x()) / (2.0 * M_PI));
    double v = 0.5 - (asin(dir_normalized.y()) / M_PI);

    int x = std::clamp(static_cast<int>(u * sky.width), 0, sky.width - 1);
    int y = std::clamp(static_cast<int>(v * sky.height), 0, sky.height - 1);

    unsigned char r = sky.r(x, y);
    unsigned char g = sky.g(x, y);
    unsigned char b = sky.b(x, y);

    return Vector3d(static_cast<double>(r) / 255.0,
                    static_cast<double>(g) / 255.0,
                    static_cast<double>(b) / 255.0);
}