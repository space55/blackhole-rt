
#include "disk.h"

#include <algorithm>
#include <cmath>

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
accretion_disk_s::accretion_disk_s(const blackhole_s *black_hole, double r_outer,
                                   double h, double rho0, double kappa0)
    : bh(black_hole), inner_r(bh->isco_radius()), outer_r(r_outer),
      thickness(h), density0(rho0), opacity0(kappa0)
{
}

// ---------------------------------------------------------------------------
// Disk half-thickness: linearly flared, h(r) = thickness * (r / r_ref)
// r_ref is the geometric mean of inner and outer radii
// ---------------------------------------------------------------------------
double accretion_disk_s::half_thickness(double r_ks) const
{
    const double r_ref = sqrt(inner_r * outer_r);
    return thickness * (r_ks / r_ref);
}

// ---------------------------------------------------------------------------
// Is this point inside the disk volume?
// ---------------------------------------------------------------------------
bool accretion_disk_s::contains(const Vector3d &pos) const
{
    const double r_ks = bh->ks_radius(pos);
    if (r_ks < inner_r || r_ks > outer_r)
        return false;

    const double h = half_thickness(r_ks);
    const double height = fabs(pos.y());
    return height <= 3.0 * h;
}

// ---------------------------------------------------------------------------
// Gas density: power-law radial profile, Gaussian vertical profile
// ---------------------------------------------------------------------------
double accretion_disk_s::density(const Vector3d &pos) const
{
    const double r_ks = bh->ks_radius(pos);
    if (r_ks < inner_r || r_ks > outer_r)
        return 0.0;

    const double h = half_thickness(r_ks);
    if (h < 1e-12)
        return 0.0;

    const double height = pos.y();
    const double radial = pow(inner_r / r_ks, 1.5);
    const double vertical = exp(-0.5 * (height * height) / (h * h));

    return density0 * radial * vertical;
}

// ---------------------------------------------------------------------------
// Temperature: simplified Novikov-Thorne profile
// ---------------------------------------------------------------------------
double accretion_disk_s::temperature(const Vector3d &pos) const
{
    const double r_ks = bh->ks_radius(pos);
    if (r_ks < inner_r || r_ks > outer_r)
        return 0.0;

    const double x_ratio = r_ks / inner_r;

    const double factor = (1.0 / (x_ratio * x_ratio * x_ratio)) *
                          std::max(1.0 - sqrt(1.0 / x_ratio), 0.0);

    const double T_max = 1.0;
    const double peak_x = 49.0 / 36.0;
    const double peak_val = (1.0 / (peak_x * peak_x * peak_x)) *
                            (1.0 - sqrt(1.0 / peak_x));
    const double T4 = T_max * factor / peak_val;

    return std::max(pow(std::max(T4, 0.0), 0.25), 0.0);
}

// ---------------------------------------------------------------------------
// Blackbody temperature to RGB
// ---------------------------------------------------------------------------
Vector3d accretion_disk_s::temperature_to_rgb(double T)
{
    T = std::clamp(T, 0.0, 2.0);

    double r, g, b;
    r = std::clamp(1.5 * T, 0.0, 1.0);
    g = std::clamp(1.5 * (T - 0.2), 0.0, 1.0);
    b = std::clamp(2.0 * (T - 0.4), 0.0, 1.0);

    return Vector3d(r, g, b);
}

// ---------------------------------------------------------------------------
// Emissivity: thermal emission j = emission_boost * rho * T^4 * color(T)
// ---------------------------------------------------------------------------
Vector3d accretion_disk_s::emissivity(const Vector3d &pos) const
{
    const double rho = density(pos);
    if (rho < 1e-12)
        return Vector3d(0, 0, 0);

    const double T = temperature(pos);
    if (T < 1e-12)
        return Vector3d(0, 0, 0);

    const double T4 = T * T * T * T;
    const Vector3d color = temperature_to_rgb(T);

    return emission_boost * rho * T4 * color;
}

// ---------------------------------------------------------------------------
// Absorption coefficient: alpha = opacity0 * density
// ---------------------------------------------------------------------------
double accretion_disk_s::absorption(const Vector3d &pos) const
{
    return opacity0 * density(pos);
}

// ---------------------------------------------------------------------------
// Keplerian 4-velocity of gas in prograde circular orbit
// ---------------------------------------------------------------------------
Vector4d accretion_disk_s::gas_four_velocity(const Vector3d &pos) const
{
    const double x = pos.x(), z = pos.z();
    const double r_ks = bh->ks_radius(pos);

    if (r_ks < inner_r)
        return Vector4d(-1, 0, 0, 0);

    const double sqrtM = sqrt(bh->mass);
    const double Omega = sqrtM / (r_ks * sqrt(r_ks) + bh->spin * sqrtM);

    const double r_cyl = sqrt(x * x + z * z);
    if (r_cyl < 1e-12)
        return Vector4d(-1, 0, 0, 0);

    const double cos_phi = x / r_cyl;
    const double sin_phi = z / r_cyl;
    const double vx = -sin_phi * Omega * r_cyl;
    const double vy = 0.0;
    const double vz = cos_phi * Omega * r_cyl;

    const double a2 = bh->spin * bh->spin;
    const double r2 = r_ks * r_ks;
    const double P = r2 + a2;
    const double lx_ks = (r_ks * x + bh->spin * z) / P;
    const double ly_ks = pos.y() / r_ks;
    const double lz_ks = (r_ks * z - bh->spin * x) / P;
    const double H = (bh->mass * r_ks * r_ks * r_ks) / (r2 * r2 + a2 * pos.y() * pos.y());

    const double g0x = 2.0 * H * lx_ks;
    const double g0y = 2.0 * H * ly_ks;
    const double g0z = 2.0 * H * lz_ks;
    const double g00 = -1.0 + 2.0 * H;

    const double b_coeff = 2.0 * (g0x * vx + g0y * vy + g0z * vz);
    const double lv = lx_ks * vx + ly_ks * vy + lz_ks * vz;
    const double v_sq = vx * vx + vy * vy + vz * vz;
    const double gij_vivj = v_sq + 2.0 * H * lv * lv;

    const double c_coeff = gij_vivj + 1.0;
    const double disc = b_coeff * b_coeff - 4.0 * g00 * c_coeff;
    const double sqrt_disc = sqrt(std::max(disc, 0.0));

    const double u0a = (-b_coeff + sqrt_disc) / (2.0 * g00);
    const double u0b = (-b_coeff - sqrt_disc) / (2.0 * g00);
    const double u0 = (u0a < 0.0) ? u0a : u0b;

    return Vector4d(u0, vx, vy, vz);
}

// ---------------------------------------------------------------------------
// Redshift factor
// ---------------------------------------------------------------------------
double accretion_disk_s::redshift_factor(const Vector4d &photon_k,
                                         const Vector4d &gas_u,
                                         const Matrix4d &g_metric)
{
    Vector4d k_lower = g_metric * photon_k;
    double k_dot_u_emit = k_lower.dot(gas_u);

    if (fabs(k_dot_u_emit) < 1e-15)
        return 1.0;

    return -k_lower(0) / k_dot_u_emit;
}
