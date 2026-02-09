
#include "blackhole.h"

#include <algorithm>
#include <cmath>

// ---------------------------------------------------------------------------
// Static ISCO formula (prograde orbit)
// ---------------------------------------------------------------------------
double blackhole_s::compute_isco(double M, double a)
{
    const double astar = a / M;
    const double Z1 = 1.0 + std::cbrt(1.0 - astar * astar) *
                                (std::cbrt(1.0 + astar) + std::cbrt(1.0 - astar));
    const double Z2 = sqrt(3.0 * astar * astar + Z1 * Z1);
    return M * (3.0 + Z2 - sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2)));
}

// ---------------------------------------------------------------------------
// Constructor: precompute derived quantities
// ---------------------------------------------------------------------------
blackhole_s::blackhole_s(double M, double a)
    : mass(M), spin(a),
      r_event_horizon(M + sqrt(std::max(M * M - a * a, 0.0))),
      r_isco(compute_isco(M, a))
{
}

// ---------------------------------------------------------------------------
// Accessors for precomputed radii
// ---------------------------------------------------------------------------
double blackhole_s::event_horizon_radius() const
{
    return r_event_horizon;
}

double blackhole_s::isco_radius() const
{
    return r_isco;
}

// ---------------------------------------------------------------------------
// Kerr-Schild radius from Cartesian position (spin axis = y)
// ---------------------------------------------------------------------------
double blackhole_s::ks_radius(const Vector3d &pos) const
{
    const double x = pos.x(), y = pos.y(), z = pos.z();
    const double a2 = spin * spin;
    const double rho2 = x * x + y * y + z * z;
    const double term = rho2 - a2;
    const double disc = term * term + 4.0 * a2 * y * y;
    const double r2 = 0.5 * (term + sqrt(disc));
    return sqrt(std::max(r2, 1e-12));
}

// ---------------------------------------------------------------------------
// Full Kerr-Schild metric and inverse in Cartesian coordinates
// ---------------------------------------------------------------------------
MetricResult blackhole_s::metric(const Vector3d &pos) const
{
    const double x = pos.x();
    const double y = pos.y();
    const double z = pos.z();

    const double a = spin;
    const double a2 = a * a;
    const double r = ks_radius(pos);
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

// ---------------------------------------------------------------------------
// Solve null condition g_mu_nu u^mu u^nu = 0 for u^0
// ---------------------------------------------------------------------------
double blackhole_s::compute_u0_null(const Vector3d &pos, const Vector3d &vel) const
{
    const MetricResult m = metric(pos);

    const double g00 = m.g(0, 0);
    const double g01 = m.g(0, 1);
    const double g02 = m.g(0, 2);
    const double g03 = m.g(0, 3);

    const double v1 = vel.x();
    const double v2 = vel.y();
    const double v3 = vel.z();

    const double b = 2.0 * (g01 * v1 + g02 * v2 + g03 * v3);
    const double c = m.g(1, 1) * v1 * v1 + m.g(2, 2) * v2 * v2 + m.g(3, 3) * v3 * v3 + 2.0 * (m.g(1, 2) * v1 * v2 + m.g(1, 3) * v1 * v3 + m.g(2, 3) * v2 * v3);

    const double disc = b * b - 4.0 * g00 * c;
    const double sqrt_disc = sqrt(std::max(disc, 0.0));

    const double u0_a = (-b + sqrt_disc) / (2.0 * g00);
    const double u0_b = (-b - sqrt_disc) / (2.0 * g00);

    return (u0_a < 0.0) ? u0_a : u0_b;
}

// ---------------------------------------------------------------------------
// Analytical partial derivatives of the metric dg_ab/dx^i (i=1,2,3)
// ---------------------------------------------------------------------------
void blackhole_s::metric_partials(const Vector3d &pos, Matrix4d partials[3]) const
{
    const double x = pos.x();
    const double y = pos.y();
    const double z = pos.z();
    const double a = spin;
    const double a2 = a * a;

    const double r = ks_radius(pos);
    const double r2 = r * r;
    const double r3 = r2 * r;
    const double P = r2 + a2;
    const double P2 = P * P;
    const double Sigma = r2 + a2 * y * y / r2;
    const double Sigma2 = Sigma * Sigma;

    const double lx = (r * x + a * z) / P;
    const double ly = y / r;
    const double lz = (r * z - a * x) / P;
    Vector4d l;
    l << 1.0, lx, ly, lz;

    const double H = mass * r / Sigma;

    const double dr[3] = {
        x * r / Sigma,
        y * P / (r * Sigma),
        z * r / Sigma};

    const double sig_r_coeff = 2.0 * r - 2.0 * a2 * y * y / r3;
    double dSigma[3];
    for (int i = 0; i < 3; ++i)
        dSigma[i] = sig_r_coeff * dr[i];
    dSigma[1] += 2.0 * a2 * y / r2;

    double dH[3];
    for (int i = 0; i < 3; ++i)
        dH[i] = mass * (dr[i] * Sigma - r * dSigma[i]) / Sigma2;

    double dlx[3];
    for (int i = 0; i < 3; ++i)
    {
        double num = dr[i] * x;
        if (i == 0)
            num += r;
        if (i == 2)
            num += a;
        dlx[i] = (num * P - (r * x + a * z) * 2.0 * r * dr[i]) / P2;
    }

    double dly[3];
    for (int i = 0; i < 3; ++i)
    {
        double num = -y * dr[i];
        if (i == 1)
            num += r;
        dly[i] = num / r2;
    }

    double dlz[3];
    for (int i = 0; i < 3; ++i)
    {
        double num = dr[i] * z;
        if (i == 2)
            num += r;
        if (i == 0)
            num -= a;
        dlz[i] = (num * P - (r * z - a * x) * 2.0 * r * dr[i]) / P2;
    }

    for (int i = 0; i < 3; ++i)
    {
        Vector4d dl_i;
        dl_i << 0.0, dlx[i], dly[i], dlz[i];
        partials[i] = 2.0 * dH[i] * (l * l.transpose()) + 2.0 * H * (dl_i * l.transpose() + l * dl_i.transpose());
    }
}

// ---------------------------------------------------------------------------
// Geodesic acceleration: d^2 x^mu / dlambda^2 = -Gamma^mu_ab u^a u^b
// ---------------------------------------------------------------------------
Vector3d blackhole_s::geodesic_accel(const Vector3d &pos, const Vector3d &vel) const
{
    const MetricResult m = metric(pos);
    Matrix4d partials[3];
    metric_partials(pos, partials);

    const double u0 = compute_u0_null(pos, vel);
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
                    gamma += 0.5 * m.g_inv(mu, nu) * (d_alpha_g + d_beta_g - d_nu_g);
                }
                sum += gamma * u(alpha) * u(beta);
            }
        }
        accel[mu - 1] = -sum;
    }

    return accel;
}
