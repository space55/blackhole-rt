
#include "disk.h"

#include <algorithm>
#include <cmath>

// ---------------------------------------------------------------------------
// ISCO radius for Kerr (prograde orbit)
// ---------------------------------------------------------------------------
double accretion_disk_s::isco_radius(double M, double a)
{
    // Standard Kerr ISCO formula (prograde)
    const double astar = a / M;
    const double Z1 = 1.0 + std::cbrt(1.0 - astar * astar) *
                                (std::cbrt(1.0 + astar) + std::cbrt(1.0 - astar));
    const double Z2 = sqrt(3.0 * astar * astar + Z1 * Z1);
    return M * (3.0 + Z2 - sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2)));
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
accretion_disk_s::accretion_disk_s(double M, double a, double r_outer,
                                   double h, double rho0, double kappa0)
    : mass(M), spin(a), inner_r(isco_radius(M, a)), outer_r(r_outer),
      thickness(h), density0(rho0), opacity0(kappa0)
{
}

// ---------------------------------------------------------------------------
// Kerr-Schild radius from Cartesian position (spin axis = y)
// ---------------------------------------------------------------------------
double accretion_disk_s::ks_radius(const Vector3d &pos) const
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
// The disk lives in the xz plane (spin axis = y), so "height" = y coordinate
// but we use the oblate spheroidal height: h_eff = y * r_ks / sqrt(r_ks^2 + a^2)
// ---------------------------------------------------------------------------
bool accretion_disk_s::contains(const Vector3d &pos) const
{
    const double r_ks = ks_radius(pos);
    if (r_ks < inner_r || r_ks > outer_r)
        return false;

    const double h = half_thickness(r_ks);
    // Height above the equatorial plane (y in our convention)
    const double height = fabs(pos.y());
    return height <= 3.0 * h; // 3-sigma cutoff for Gaussian profile
}

// ---------------------------------------------------------------------------
// Gas density: power-law radial profile, Gaussian vertical profile
// rho(r, y) = rho0 * (r_inner / r)^alpha * exp(-y^2 / (2*h(r)^2))
// ---------------------------------------------------------------------------
double accretion_disk_s::density(const Vector3d &pos) const
{
    const double r_ks = ks_radius(pos);
    if (r_ks < inner_r || r_ks > outer_r)
        return 0.0;

    const double h = half_thickness(r_ks);
    if (h < 1e-12)
        return 0.0;

    const double height = pos.y();
    const double radial = pow(inner_r / r_ks, 1.5); // alpha = 1.5
    const double vertical = exp(-0.5 * (height * height) / (h * h));

    return density0 * radial * vertical;
}

// ---------------------------------------------------------------------------
// Temperature: simplified Novikov-Thorne profile
// T(r) ~ T_max * (r_inner / r)^(3/4) * f(r)
// where f(r) captures the zero-torque inner boundary
// ---------------------------------------------------------------------------
double accretion_disk_s::temperature(const Vector3d &pos) const
{
    const double r_ks = ks_radius(pos);
    if (r_ks < inner_r || r_ks > outer_r)
        return 0.0;

    const double x_ratio = r_ks / inner_r;

    // Simplified Novikov-Thorne: T^4 ~ (1/r^3) * (1 - sqrt(r_in/r))
    const double factor = (1.0 / (x_ratio * x_ratio * x_ratio)) *
                          std::max(1.0 - sqrt(1.0 / x_ratio), 0.0);

    // T_max is set so the peak temperature is ~1.0 (normalized units)
    // The peak of r^{-3}(1-sqrt(r_in/r)) occurs at r = (49/36)*r_in
    const double T_max = 1.0;
    const double peak_x = 49.0 / 36.0;
    const double peak_val = (1.0 / (peak_x * peak_x * peak_x)) *
                            (1.0 - sqrt(1.0 / peak_x));
    const double T4 = T_max * factor / peak_val;

    return std::max(pow(std::max(T4, 0.0), 0.25), 0.0);
}

// ---------------------------------------------------------------------------
// Blackbody temperature to RGB (peaking hot-white around T=1)
// This is a simplified mapping where:
//   T < 0.3 : deep red/infrared
//   T ~ 0.5 : orange
//   T ~ 0.7 : yellow-white
//   T ~ 1.0 : white-blue
//   T > 1.2 : blue-white
// ---------------------------------------------------------------------------
Vector3d accretion_disk_s::temperature_to_rgb(double T)
{
    // Clamp temperature to useful range
    T = std::clamp(T, 0.0, 2.0);

    double r, g, b;

    // Red channel: rises quickly, saturates
    r = std::clamp(1.5 * T, 0.0, 1.0);

    // Green channel: rises slower
    g = std::clamp(1.5 * (T - 0.2), 0.0, 1.0);

    // Blue channel: rises latest
    b = std::clamp(2.0 * (T - 0.4), 0.0, 1.0);

    return Vector3d(r, g, b);
}

// ---------------------------------------------------------------------------
// Emissivity: thermal emission j = rho * sigma * T^4 * color(T)
// Returns RGB emissivity per unit length
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

    // Emissivity proportional to density * T^4
    return rho * T4 * color;
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
// In Kerr-Schild Cartesian coordinates (spin axis = y), for a circular
// orbit in the xz equatorial plane at radius r:
//
// The angular velocity is Omega = M^{1/2} / (r^{3/2} + a*M^{1/2})
// and the gas moves in the xz plane with velocity v_phi = Omega * r_cyl
// The direction of motion is tangent to the orbit: (-sin(phi), 0, cos(phi))
// where phi = atan2(z, x) (azimuthal angle in xz plane)
// ---------------------------------------------------------------------------
Vector4d accretion_disk_s::gas_four_velocity(const Vector3d &pos) const
{
    const double x = pos.x(), z = pos.z();
    const double r_ks = ks_radius(pos);

    if (r_ks < inner_r)
        return Vector4d(-1, 0, 0, 0); // fallback

    // Keplerian angular velocity (prograde, spin axis y)
    const double sqrtM = sqrt(mass);
    const double Omega = sqrtM / (r_ks * sqrt(r_ks) + spin * sqrtM);

    // Cylindrical radius in the equatorial plane
    const double r_cyl = sqrt(x * x + z * z);
    if (r_cyl < 1e-12)
        return Vector4d(-1, 0, 0, 0); // on the axis

    // Tangent direction in xz plane (prograde = counterclockwise when viewed from +y)
    const double cos_phi = x / r_cyl;
    const double sin_phi = z / r_cyl;
    // v_tangent = (-sin_phi, 0, cos_phi) * Omega * r_cyl
    const double vx = -sin_phi * Omega * r_cyl;
    const double vy = 0.0;
    const double vz = cos_phi * Omega * r_cyl;

    // Now we need u^0 such that g_mu_nu u^mu u^nu = -1 (timelike normalization)
    // For the spatial velocity (vx, vy, vz), solve g_00 (u0)^2 + 2 g_0i u0 vi + g_ij vi vj = -1
    // This is the same quadratic as for null rays but with c = g_ij vi vj + 1
    const double a2 = spin * spin;
    const double r2 = r_ks * r_ks;
    const double P = r2 + a2;
    const double lx_ks = (r_ks * x + spin * z) / P;
    const double ly_ks = pos.y() / r_ks;
    const double lz_ks = (r_ks * z - spin * x) / P;
    const double H = (mass * r_ks * r_ks * r_ks) / (r2 * r2 + a2 * pos.y() * pos.y());

    // g_0i = 2H * l_0 * l_i = 2H * l_i  (since l_0 = 1)
    const double g0x = 2.0 * H * lx_ks;
    const double g0y = 2.0 * H * ly_ks;
    const double g0z = 2.0 * H * lz_ks;
    const double g00 = -1.0 + 2.0 * H;

    const double b_coeff = 2.0 * (g0x * vx + g0y * vy + g0z * vz);
    // g_ij v^i v^j : eta_ij v^i v^j + 2H (l_i v^i)^2
    const double lv = lx_ks * vx + ly_ks * vy + lz_ks * vz;
    const double v_sq = vx * vx + vy * vy + vz * vz;
    const double gij_vivj = v_sq + 2.0 * H * lv * lv;

    // g_00 u0^2 + b u0 + (gij vivj + 1) = 0
    const double c_coeff = gij_vivj + 1.0;
    const double disc = b_coeff * b_coeff - 4.0 * g00 * c_coeff;
    const double sqrt_disc = sqrt(std::max(disc, 0.0));

    // Pick the future-pointing solution (u0 should be negative in -+++ convention
    // with KS coords where g_00 < 0 for large r)
    const double u0a = (-b_coeff + sqrt_disc) / (2.0 * g00);
    const double u0b = (-b_coeff - sqrt_disc) / (2.0 * g00);
    const double u0 = (u0a < 0.0) ? u0a : u0b;

    return Vector4d(u0, vx, vy, vz);
}

// ---------------------------------------------------------------------------
// Redshift factor
// g = (k_mu u^mu)_observer / (k_mu u^mu)_emitter
// where k is the photon 4-momentum (covariant) and u is the 4-velocity
// For observer at infinity: (k_mu u^mu)_obs = -E (the photon energy = -k_0)
// ---------------------------------------------------------------------------
double accretion_disk_s::redshift_factor(const Vector4d &photon_k,
                                         const Vector4d &gas_u,
                                         const Matrix4d &g_metric)
{
    // Lower the photon 4-momentum: k_mu = g_mu_nu k^nu
    Vector4d k_lower = g_metric * photon_k;

    // k_mu u^mu at emitter
    double k_dot_u_emit = k_lower.dot(gas_u);

    // At observer at infinity, k_mu u^mu_obs = k_0 (u^0_obs = 1, ui_obs = 0)
    // and k_0 at infinity is just the conserved energy = -k_lower(0) at emission
    // But we want the ratio, so: g = k_lower(0) / k_dot_u_emit
    // Actually, for a static observer at infinity: g_mu_nu -> eta, u_obs = (1,0,0,0)
    // so (k_mu u^mu)_obs = k_0_obs which we don't know.
    // Instead use: the photon energy is conserved along the geodesic in KS coords
    // (stationary spacetime), so E_obs = -k_t = -k_lower(0) at any point.
    // g = E_obs / E_emit = -k_lower(0) / k_dot_u_emit
    // But k_dot_u_emit is already negative (timelike dot lightlike), so:

    if (fabs(k_dot_u_emit) < 1e-15)
        return 1.0;

    return -k_lower(0) / k_dot_u_emit;
}
