// ============================================================================
// Shared Physics — all black-hole, accretion-disk, and ray-integration maths
//
// Every function is BH_FUNC inline so it compiles under both g++ (host) and
// nvcc (host + device).  The physics is parameterised by PhysicsParams,
// eliminating the class-based duplication that previously existed between
// the CPU (blackhole_s, accretion_disk_s, ray_s) and GPU (gpu_*) paths.
// ============================================================================

#ifndef _BH_PHYSICS_H
#define _BH_PHYSICS_H

#include "vec_math.h"

// ============================================================================
// Physics parameters — everything the integrator and disk model need
// ============================================================================

struct PhysicsParams
{
    // Black hole
    double bh_mass;
    double bh_spin;
    double r_plus; // event horizon radius (precomputed)

    // Disk geometry
    double disk_inner_r;   // ISCO (or custom inner edge)
    double disk_outer_r;
    double disk_thickness; // half-thickness scale height at reference radius
    double disk_r_ref;     // sqrt(inner_r * outer_r), precomputed

    // Disk physical properties
    double disk_density0;
    double disk_opacity0;

    // Disk appearance
    double disk_emission_boost;
    double disk_color_variation;
    double disk_turbulence;
    double disk_time;
};

// ============================================================================
// Black Hole — derived quantities
// ============================================================================

BH_FUNC inline double compute_event_horizon(double M, double a)
{
    return M + sqrt(fmax(M * M - a * a, 0.0));
}

BH_FUNC inline double compute_isco(double M, double a)
{
    const double astar = a / M;
    const double Z1 = 1.0 + cbrt(1.0 - astar * astar) *
                                (cbrt(1.0 + astar) + cbrt(1.0 - astar));
    const double Z2 = sqrt(3.0 * astar * astar + Z1 * Z1);
    return M * (3.0 + Z2 - sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2)));
}

// ============================================================================
// Kerr-Schild radius from Cartesian coordinates (spin axis = y)
// ============================================================================

BH_FUNC inline double ks_radius(const dvec3 &pos, double spin)
{
    const double a2 = spin * spin;
    const double rho2 = pos.x * pos.x + pos.y * pos.y + pos.z * pos.z;
    const double term = rho2 - a2;
    const double disc = term * term + 4.0 * a2 * pos.y * pos.y;
    const double r2 = 0.5 * (term + sqrt(disc));
    return sqrt(fmax(r2, 1e-12));
}

// ============================================================================
// Geodesic acceleration: d²x^μ/dλ² = -Γ^μ_αβ u^α u^β
//
// Exploits the rank-2 Kerr-Schild structure — no 4×4 matrices needed.
// Optionally outputs the KS radius via out_r.
// ============================================================================

BH_FUNC inline dvec3 geodesic_accel(const dvec3 &pos, const dvec3 &vel,
                                    const PhysicsParams &pp, double *out_r = nullptr)
{
    const double px = pos.x, py = pos.y, pz = pos.z;
    const double a = pp.bh_spin, a2 = a * a;
    const double M = pp.bh_mass;

    // KS radius
    const double rho2 = px * px + py * py + pz * pz;
    const double term = rho2 - a2;
    const double r2 = 0.5 * (term + sqrt(term * term + 4.0 * a2 * py * py));
    const double r = sqrt(fmax(r2, 1e-12));
    const double inv_r = 1.0 / r;
    const double inv_r2 = inv_r * inv_r;

    if (out_r)
        *out_r = r;

    // Null vector l, Sigma, H
    const double P = r2 + a2;
    const double inv_P = 1.0 / P;
    const double inv_P2 = inv_P * inv_P;
    const double lx = (r * px + a * pz) * inv_P;
    const double ly = py * inv_r;
    const double lz = (r * pz - a * px) * inv_P;

    const double Sigma = r2 + a2 * py * py * inv_r2;
    const double inv_Sigma = 1.0 / Sigma;
    const double H = M * r * inv_Sigma;

    // u^0 from null condition
    const double twoH = 2.0 * H;
    const double g00 = -1.0 + twoH;
    const double v1 = vel.x, v2 = vel.y, v3 = vel.z;
    const double lv = lx * v1 + ly * v2 + lz * v3;
    const double b_u0 = 2.0 * twoH * lv;
    const double v_sq = v1 * v1 + v2 * v2 + v3 * v3;
    const double c_u0 = v_sq + twoH * lv * lv;

    const double disc_u0 = b_u0 * b_u0 - 4.0 * g00 * c_u0;
    const double sqrt_disc = sqrt(fmax(disc_u0, 0.0));
    const double inv_2g00 = 0.5 / g00;
    const double u0a = (-b_u0 + sqrt_disc) * inv_2g00;
    const double u0b = (-b_u0 - sqrt_disc) * inv_2g00;
    const double u0 = (u0a < 0.0) ? u0a : u0b;

    // L = l · u
    const double L = u0 + lx * v1 + ly * v2 + lz * v3;

    // Partial derivative intermediates
    const double inv_Sigma2 = inv_Sigma * inv_Sigma;
    const double r_inv_Sigma = r * inv_Sigma;

    const double dr0 = px * r_inv_Sigma;
    const double dr1 = py * P * inv_r * inv_Sigma;
    const double dr2 = pz * r_inv_Sigma;

    const double inv_r3 = inv_r * inv_r2;
    const double sig_r_coeff = 2.0 * r - 2.0 * a2 * py * py * inv_r3;
    const double dSigma0 = sig_r_coeff * dr0;
    const double dSigma1 = sig_r_coeff * dr1 + 2.0 * a2 * py * inv_r2;
    const double dSigma2 = sig_r_coeff * dr2;

    const double mass_inv_S2 = M * inv_Sigma2;
    const double dH0 = mass_inv_S2 * (dr0 * Sigma - r * dSigma0);
    const double dH1 = mass_inv_S2 * (dr1 * Sigma - r * dSigma1);
    const double dH2 = mass_inv_S2 * (dr2 * Sigma - r * dSigma2);

    // dl_i factored form
    const double rxaz = r * px + a * pz;
    const double rzax = r * pz - a * px;
    const double r_inv_P = r * inv_P;
    const double a_inv_P = a * inv_P;
    const double K_x = (px * P - 2.0 * r * rxaz) * inv_P2;
    const double K_y = -py * inv_r2;
    const double K_z = (pz * P - 2.0 * r * rzax) * inv_P2;

    const double dlx0 = dr0 * K_x + r_inv_P;
    const double dly0 = K_y * dr0;
    const double dlz0 = dr0 * K_z - a_inv_P;

    const double dlx1 = dr1 * K_x;
    const double dly1 = K_y * dr1 + inv_r;
    const double dlz1 = dr1 * K_z;

    const double dlx2 = dr2 * K_x + a_inv_P;
    const double dly2 = K_y * dr2;
    const double dlz2 = dr2 * K_z + r_inv_P;

    // D[i] = dl_i · u
    const double D0 = dlx0 * v1 + dly0 * v2 + dlz0 * v3;
    const double D1 = dlx1 * v1 + dly1 * v2 + dlz1 * v3;
    const double D2 = dlx2 * v1 + dly2 * v2 + dlz2 * v3;

    // P_i = (2*dH_i*L + 2*H*D_i) * l + 2*H*L * dl_i
    const double twoHL = twoH * L;
    const double c0 = 2.0 * dH0 * L + twoH * D0;
    const double c1 = 2.0 * dH1 * L + twoH * D1;
    const double c2 = 2.0 * dH2 * L + twoH * D2;

    const double P0_0 = c0, P0_1 = c0 * lx + twoHL * dlx0;
    const double P0_2 = c0 * ly + twoHL * dly0, P0_3 = c0 * lz + twoHL * dlz0;

    const double P1_0 = c1, P1_1 = c1 * lx + twoHL * dlx1;
    const double P1_2 = c1 * ly + twoHL * dly1, P1_3 = c1 * lz + twoHL * dlz1;

    const double P2_0 = c2, P2_1 = c2 * lx + twoHL * dlx2;
    const double P2_2 = c2 * ly + twoHL * dly2, P2_3 = c2 * lz + twoHL * dlz2;

    // Q_i = 2*dH_i*L² + 4*H*L*D_i
    const double L2 = L * L;
    const double fourHL = 2.0 * twoHL;
    const double Q0 = 2.0 * dH0 * L2 + fourHL * D0;
    const double Q1 = 2.0 * dH1 * L2 + fourHL * D1;
    const double Q2 = 2.0 * dH2 * L2 + fourHL * D2;

    // F(ν) = 2*T1(ν) - T3(ν)
    const double F0 = 2.0 * (v1 * P0_0 + v2 * P1_0 + v3 * P2_0);
    const double F1 = 2.0 * (v1 * P0_1 + v2 * P1_1 + v3 * P2_1) - Q0;
    const double F2 = 2.0 * (v1 * P0_2 + v2 * P1_2 + v3 * P2_2) - Q1;
    const double F3 = 2.0 * (v1 * P0_3 + v2 * P1_3 + v3 * P2_3) - Q2;

    // a^μ = -½ g_inv^{μν} F_ν using KS form
    const double S = -F0 + lx * F1 + ly * F2 + lz * F3;
    const double twoHS = twoH * S;

    return dvec3(
        -0.5 * (F1 - lx * twoHS),
        -0.5 * (F2 - ly * twoHS),
        -0.5 * (F3 - lz * twoHS));
}

// ============================================================================
// Accretion Disk Physics
// ============================================================================

// Half-thickness: linearly flared, h(r) = thickness * (r / r_ref)
BH_FUNC inline double disk_half_thickness(double r_ks, const PhysicsParams &pp)
{
    return pp.disk_thickness * (r_ks / pp.disk_r_ref);
}

// Turbulence-warped half-thickness (azimuthal lumps, gaps, warps)
BH_FUNC inline double disk_warped_half_thickness(const dvec3 &pos, double r_ks,
                                                 const PhysicsParams &pp)
{
    const double h0 = disk_half_thickness(fmin(r_ks, pp.disk_outer_r), pp);
    const double turb = pp.disk_turbulence;
    if (turb < 1e-6)
        return h0;

    const double phi = atan2(pos.z, pos.x);
    const double log_r = log(fmax(r_ks, 1e-6));

    const double omega = 1.0 / (r_ks * sqrt(r_ks));
    const double tp = pp.disk_time * omega;

    double warp = 0.0;
    warp += 0.55 * sin(2.0 * (phi + tp) - 3.0 * log_r + 1.2);
    warp += 0.35 * sin(3.0 * (phi + tp) - 5.0 * log_r + 4.1);
    warp += 0.25 * sin(5.0 * (phi + tp) - 9.0 * log_r + 0.7);
    warp += 0.18 * sin(7.0 * (phi + tp) - 11.0 * log_r + 2.9);
    warp += 0.12 * sin(11.0 * (phi + tp) - 17.0 * log_r + 5.3);
    warp += 0.08 * sin(17.0 * (phi + tp) - 23.0 * log_r + 3.1);

    // Narrow gaps
    auto dip = [](double phase, double sharpness) -> double
    {
        double s = sin(phase);
        return pow(s * s, sharpness);
    };
    warp -= 0.70 * dip(3.0 * (phi + tp) - 4.0 * log_r + 2.5, 4.0);
    warp -= 0.40 * dip(5.0 * (phi + tp) - 8.0 * log_r + 0.3, 5.0);

    const double factor = 1.0 + turb * warp;
    return h0 * fmax(factor, 0.05);
}

// Procedural clump / streak modulation
BH_FUNC inline double disk_clump_factor(const dvec3 &pos, double r_ks,
                                        double warped_h, const PhysicsParams &pp)
{
    const double inner_r = pp.disk_inner_r;
    const double outer_r = pp.disk_outer_r;
    if (r_ks < inner_r || r_ks > outer_r)
        return 1.0;

    const double phi = atan2(pos.z, pos.x);
    const double log_r = log(r_ks);
    const double r_norm = (r_ks - inner_r) / (outer_r - inner_r);

    const double omega = 1.0 / (r_ks * sqrt(r_ks));
    const double t_phase = pp.disk_time * omega;

    const double y_over_h = (warped_h > 1e-12) ? (pos.y / warped_h) : 0.0;
    const double y_phase = 3.0 * y_over_h;

    auto streak = [](double phase, double sharpness) -> double
    {
        const double s = sin(phase);
        return pow(s * s, sharpness);
    };

    double mod = 0.0;

    // Large-scale spirals
    mod += 1.4 * streak(2.0 * (phi + t_phase) - 4.5 * log_r + 0.5 + y_phase, 3.0);
    mod += 0.7 * streak(3.0 * (phi + t_phase) - 7.0 * log_r + 2.1 + 1.5 * y_phase, 2.5);

    // Medium-scale
    mod += 0.45 * streak(7.0 * (phi + t_phase) - 13.0 * log_r + 1.3 + 2.0 * y_phase, 2.0);
    mod += 0.35 * streak(11.0 * (phi + t_phase) - 19.0 * log_r - 0.8 + 2.5 * y_phase, 2.0);

    // Fine-scale
    mod += 0.25 * sin(17.0 * (phi + t_phase) - 25.0 * log_r + 3.7 + 3.0 * y_phase);
    mod += 0.15 * sin(23.0 * (phi + t_phase) - 33.0 * log_r + 0.4 + 4.0 * y_phase);
    mod += 0.10 * sin(31.0 * (phi + t_phase) - 45.0 * log_r + 5.2 + 5.0 * y_phase);

    // Radial hot-spot rings
    mod += 0.40 * streak(5.0 * r_ks + 1.0 + 0.5 * y_phase, 2.0) *
           cos(3.0 * (phi + t_phase) - 5.0 * log_r + y_phase);

    // Bright knots
    mod += 0.35 * streak(4.0 * M_PI * r_norm, 2.0) *
           streak(9.0 * (phi + t_phase) - 14.0 * log_r + 1.9 + 2.0 * y_phase, 2.0);

    // Dark lanes
    mod -= 0.50 * streak(5.0 * (phi + t_phase) - 8.0 * log_r + 4.0 + 1.8 * y_phase, 4.0);
    mod -= 0.30 * streak(8.0 * (phi + t_phase) - 15.0 * log_r + 0.7 + 2.2 * y_phase, 5.0);

    // Radial envelope
    const double envelope = 4.0 * r_norm * (1.0 - r_norm);
    mod *= (0.4 + 0.6 * envelope);

    const double factor = 0.35 + 0.65 * fmax(mod, 0.0);
    return dclamp(factor, 0.02, 3.0);
}

// Gas density: power-law radial, Gaussian vertical
BH_FUNC inline double disk_density(const dvec3 &pos, double r_ks,
                                   double warped_h, const PhysicsParams &pp)
{
    const double inner_r = pp.disk_inner_r;
    const double outer_r = pp.disk_outer_r;
    const double fade_limit = outer_r * 1.2;
    if (r_ks < inner_r || r_ks > fade_limit)
        return 0.0;
    if (warped_h < 1e-12)
        return 0.0;

    const double height = pos.y;
    const double radial = pow(inner_r / r_ks, 1.5);
    const double vertical = exp(-0.5 * (height * height) / (warped_h * warped_h));

    double outer_fade = 1.0;
    if (r_ks > outer_r)
    {
        const double fw = 0.1 * outer_r;
        const double d = r_ks - outer_r;
        outer_fade = exp(-(d * d) / (2.0 * fw * fw));
    }

    return pp.disk_density0 * radial * vertical *
           disk_clump_factor(pos, r_ks, warped_h, pp) * outer_fade;
}

// Temperature: simplified Novikov-Thorne profile
BH_FUNC inline double disk_temperature(double r_ks, const PhysicsParams &pp)
{
    const double inner_r = pp.disk_inner_r;
    const double outer_r = pp.disk_outer_r;
    const double fade_limit = outer_r * 1.2;
    if (r_ks < inner_r || r_ks > fade_limit)
        return 0.0;

    const double x_ratio = r_ks / inner_r;
    const double factor = (1.0 / (x_ratio * x_ratio * x_ratio)) *
                          fmax(1.0 - sqrt(1.0 / x_ratio), 0.0);
    const double peak_x = 49.0 / 36.0;
    const double peak_val = (1.0 / (peak_x * peak_x * peak_x)) *
                            (1.0 - sqrt(1.0 / peak_x));
    const double T4 = factor / peak_val;

    double outer_fade = 1.0;
    if (r_ks > outer_r)
    {
        const double fw = 0.1 * outer_r;
        const double d = r_ks - outer_r;
        outer_fade = exp(-(d * d) / (2.0 * fw * fw));
    }

    return fmax(pow(fmax(T4 * outer_fade, 0.0), 0.25), 0.0);
}

// Blackbody temperature to RGB (5-zone piecewise)
BH_FUNC inline dvec3 temperature_to_rgb(double T)
{
    T = dclamp(T, 0.0, 2.0);
    double r, g, b;

    if (T < 0.15)
    {
        r = dclamp(T / 0.15 * 0.4, 0.0, 0.4);
        g = 0.0;
        b = 0.0;
    }
    else if (T < 0.4)
    {
        double t = (T - 0.15) / 0.25;
        r = 0.4 + 0.6 * t;
        g = 0.25 * t * t;
        b = 0.0;
    }
    else if (T < 0.7)
    {
        double t = (T - 0.4) / 0.3;
        r = 1.0;
        g = 0.25 + 0.55 * t;
        b = 0.05 * t;
    }
    else if (T < 1.0)
    {
        double t = (T - 0.7) / 0.3;
        r = 1.0;
        g = 0.8 + 0.2 * t;
        b = 0.05 + 0.95 * t;
    }
    else
    {
        double t = fmin((T - 1.0) / 1.0, 1.0);
        r = 1.0 - 0.15 * t;
        g = 1.0 - 0.05 * t;
        b = 1.0;
    }

    return dvec3(r, g, b);
}

// Emissivity + optional absorption output
BH_FUNC inline dvec3 disk_emissivity(const dvec3 &pos, double r_ks,
                                     double warped_h, double *alpha_out,
                                     const PhysicsParams &pp)
{
    const double rho = disk_density(pos, r_ks, warped_h, pp);
    if (alpha_out)
        *alpha_out = pp.disk_opacity0 * rho;
    if (rho < 1e-12)
        return dvec3(0, 0, 0);

    const double T = disk_temperature(r_ks, pp);
    if (T < 1e-12)
        return dvec3(0, 0, 0);

    const double T4 = T * T * T * T;
    dvec3 color = temperature_to_rgb(T);

    // Cinematic color variation
    const double cv = pp.disk_color_variation;
    if (cv > 1e-6)
    {
        const double inner_r = pp.disk_inner_r;
        const double outer_r = pp.disk_outer_r;
        const double phi = atan2(pos.z, pos.x);
        const double log_r = log(fmax(r_ks, 1e-6));
        const double r_norm = dclamp((r_ks - inner_r) / (outer_r - inner_r), 0.0, 1.0);

        const double hue = 0.6 * sin(2.0 * phi - 4.5 * log_r + 0.5) +
                           0.4 * sin(3.0 * phi - 7.0 * log_r + 2.1) +
                           0.3 * cos(5.0 * phi - 10.0 * log_r + 1.7);
        const double hue2 = 0.5 * sin(4.0 * phi - 6.0 * log_r + 3.3) +
                            0.3 * cos(7.0 * phi - 12.0 * log_r + 0.9);

        const double radial_hue = 1.0 - 2.0 * r_norm;
        const double mat = 0.5 + 0.5 * hue2;

        double dr = 0.0, dg = 0.0, db = 0.0;

        dr += cv * (0.20 * (-radial_hue) + 0.15 * hue);
        dg += cv * (0.10 * hue * radial_hue);
        db += cv * (0.25 * radial_hue + 0.12 * (-hue));

        dr += cv * 0.18 * (1.0 - mat) * (0.5 + 0.5 * hue);
        db += cv * 0.12 * (1.0 - mat) * (0.5 + 0.5 * hue);

        const double ice = exp(-8.0 * (mat - 0.5) * (mat - 0.5));
        dg += cv * 0.20 * ice;
        db += cv * 0.25 * ice;
        dr -= cv * 0.10 * ice;

        dr += cv * 0.15 * mat * (0.5 - 0.5 * hue);
        dg += cv * 0.06 * mat;
        db -= cv * 0.12 * mat;

        color.x = dclamp(color.x + dr, 0.0, 1.0);
        color.y = dclamp(color.y + dg, 0.0, 1.0);
        color.z = dclamp(color.z + db, 0.0, 1.0);
    }

    return dvec3(pp.disk_emission_boost * rho * T4 * color.x,
                 pp.disk_emission_boost * rho * T4 * color.y,
                 pp.disk_emission_boost * rho * T4 * color.z);
}

// Keplerian gas 4-velocity (prograde circular orbit)
// Takes pre-computed KS intermediates (H, lx, ly, lz) to avoid redundancy
BH_FUNC inline dvec4 disk_gas_four_velocity(const dvec3 &pos, double r_ks,
                                            double H, double lx, double ly, double lz,
                                            const PhysicsParams &pp)
{
    const double inner_r = pp.disk_inner_r;
    if (r_ks < inner_r)
        return dvec4(-1, 0, 0, 0);

    const double a = pp.bh_spin;
    const double M = pp.bh_mass;
    const double sqrtM = sqrt(M);
    const double Omega = sqrtM / (r_ks * sqrt(r_ks) + a * sqrtM);

    const double r_cyl = sqrt(pos.x * pos.x + pos.z * pos.z);
    if (r_cyl < 1e-12)
        return dvec4(-1, 0, 0, 0);

    const double cos_phi = pos.x / r_cyl;
    const double sin_phi = pos.z / r_cyl;
    const double vx = -sin_phi * Omega * r_cyl;
    const double vy = 0.0;
    const double vz = cos_phi * Omega * r_cyl;

    // Solve for u^0 of timelike 4-velocity: g_μν u^μ u^ν = -1
    const double twoH = 2.0 * H;
    const double g00 = -1.0 + twoH;
    const double lv = lx * vx + ly * vy + lz * vz;
    const double b = 2.0 * twoH * lv;
    const double v_sq = vx * vx + vy * vy + vz * vz;
    const double c = v_sq + twoH * lv * lv + 1.0; // +1 for timelike

    const double disc = b * b - 4.0 * g00 * c;
    const double sqrt_disc = sqrt(fmax(disc, 0.0));
    const double inv_2g00 = 0.5 / g00;
    const double u0a = (-b + sqrt_disc) * inv_2g00;
    const double u0b = (-b - sqrt_disc) * inv_2g00;
    const double u0 = (u0a < 0.0) ? u0a : u0b;

    return dvec4(u0, vx, vy, vz);
}

// ============================================================================
// Ray Integration
// ============================================================================

// Initialize ray position and velocity from camera parameters
BH_FUNC inline void init_ray(dvec3 &pos, dvec3 &vel,
                              const dvec3 &cam_pos,
                              const dvec3 &cam_right,
                              const dvec3 &cam_up,
                              const dvec3 &cam_fwd,
                              double sub_x, double sub_y,
                              double fov_x, double fov_y)
{
    pos = cam_pos;
    const double x_rad = ((sub_x - 0.5) * fov_x) * (M_PI / 180.0);
    const double y_rad = ((0.5 - sub_y) * fov_y) * (M_PI / 180.0);

    const double cy = cos(y_rad), cx = cos(x_rad);
    const double local_fwd = cy * cx;
    const double local_up = sin(y_rad);
    const double local_right = cy * sin(x_rad);

    vel = local_fwd * cam_fwd + local_up * cam_up + local_right * cam_right;
}

// RK2 (far-field) / RK4 (near-field) geodesic integrator
BH_FUNC inline bool advance_ray(dvec3 &pos, dvec3 &vel, double &cached_r,
                                double dt, const PhysicsParams &pp)
{
    const double r_plus = pp.r_plus;

    if (cached_r > 30.0 * r_plus)
    {
        // Far-field: RK2 midpoint (2 evaluations)
        double r1;
        dvec3 a1 = geodesic_accel(pos, vel, pp, &r1);
        if (r1 <= r_plus)
            return false;

        dvec3 mid_pos = pos + 0.5 * dt * vel;
        dvec3 mid_vel = vel + 0.5 * dt * a1;

        double r2;
        dvec3 a2 = geodesic_accel(mid_pos, mid_vel, pp, &r2);
        if (r2 <= r_plus)
            return false;

        pos = pos + dt * mid_vel;
        vel = vel + dt * a2;
    }
    else
    {
        // Near-field: full RK4 (4 evaluations)
        double rk;

        // k1
        dvec3 a_k = geodesic_accel(pos, vel, pp, &rk);
        if (rk <= r_plus)
            return false;
        dvec3 k1_dx = vel, k1_dv = a_k;

        // k2
        a_k = geodesic_accel(pos + 0.5 * dt * k1_dx, vel + 0.5 * dt * k1_dv, pp, &rk);
        if (rk <= r_plus)
            return false;
        dvec3 k2_dx = vel + 0.5 * dt * k1_dv;
        dvec3 k2_dv = a_k;

        // k3
        a_k = geodesic_accel(pos + 0.5 * dt * k2_dx, vel + 0.5 * dt * k2_dv, pp, &rk);
        if (rk <= r_plus)
            return false;
        dvec3 k3_dx = vel + 0.5 * dt * k2_dv;
        dvec3 k3_dv = a_k;

        // k4
        a_k = geodesic_accel(pos + dt * k3_dx, vel + dt * k3_dv, pp, &rk);
        if (rk <= r_plus)
            return false;
        dvec3 k4_dx = vel + dt * k3_dv;
        dvec3 k4_dv = a_k;

        double s = dt / 6.0;
        pos = pos + s * (k1_dx + 2.0 * k2_dx + 2.0 * k3_dx + k4_dx);
        vel = vel + s * (k1_dv + 2.0 * k2_dv + 2.0 * k3_dv + k4_dv);
    }

    cached_r = ks_radius(pos, pp.bh_spin);
    return true;
}

// Volumetric disk sampling with relativistic radiative transfer
BH_FUNC inline void sample_disk_volume(const dvec3 &pos, const dvec3 &vel, double ds,
                                       dvec3 &acc_color, double &acc_opacity,
                                       const PhysicsParams &pp)
{
    if (acc_opacity >= 0.999)
        return;

    // Compute KS radius
    const double r_ks = ks_radius(pos, pp.bh_spin);

    // Contains check
    const double fade_limit = pp.disk_outer_r * 1.2;
    if (r_ks < pp.disk_inner_r || r_ks > fade_limit)
        return;

    const double warped_h = disk_warped_half_thickness(pos, r_ks, pp);
    if (fabs(pos.y) > 3.0 * warped_h)
        return;

    // Emissivity + absorption
    double alpha;
    dvec3 j = disk_emissivity(pos, r_ks, warped_h, &alpha, pp);

    // KS metric intermediates (computed once, shared)
    const double a = pp.bh_spin, a2 = a * a;
    const double M = pp.bh_mass;
    const double r2 = r_ks * r_ks;
    const double Pk = r2 + a2;
    const double inv_Pk = 1.0 / Pk;
    const double lx = (r_ks * pos.x + a * pos.z) * inv_Pk;
    const double ly = pos.y / r_ks;
    const double lz = (r_ks * pos.z - a * pos.x) * inv_Pk;
    const double H = (M * r_ks * r2) / (r2 * r2 + a2 * pos.y * pos.y);
    const double twoH = 2.0 * H;

    // u^0 from null condition (photon)
    const double v1 = vel.x, v2 = vel.y, v3 = vel.z;
    const double lv_phot = lx * v1 + ly * v2 + lz * v3;
    const double g00 = -1.0 + twoH;
    const double b_u0 = 2.0 * twoH * lv_phot;
    const double v_sq = v1 * v1 + v2 * v2 + v3 * v3;
    const double c_u0 = v_sq + twoH * lv_phot * lv_phot;

    const double disc_u0 = b_u0 * b_u0 - 4.0 * g00 * c_u0;
    const double sqrt_disc = sqrt(fmax(disc_u0, 0.0));
    const double inv_2g00 = 0.5 / g00;
    const double u0a = (-b_u0 + sqrt_disc) * inv_2g00;
    const double u0b = (-b_u0 - sqrt_disc) * inv_2g00;
    const double u0 = (u0a < 0.0) ? u0a : u0b;

    // Gas 4-velocity (shares H, l from metric)
    dvec4 gas_u = disk_gas_four_velocity(pos, r_ks, H, lx, ly, lz, pp);

    // Redshift via g = -k_lower[0] / (k_lower · gas_u)
    // k_lower = η_μν k^ν + 2H l_μ (l · k)   [KS form]
    const double l_dot_k = u0 + lx * v1 + ly * v2 + lz * v3;
    const double k_lower0 = -u0 + twoH * l_dot_k;
    const double k_lower1 = v1 + twoH * lx * l_dot_k;
    const double k_lower2 = v2 + twoH * ly * l_dot_k;
    const double k_lower3 = v3 + twoH * lz * l_dot_k;

    double k_dot_u = k_lower0 * gas_u.t + k_lower1 * gas_u.x +
                     k_lower2 * gas_u.y + k_lower3 * gas_u.z;
    if (fabs(k_dot_u) < 1e-15)
        k_dot_u = 1e-15;

    const double g_red = -k_lower0 / k_dot_u;
    const double g_clamped = dclamp(g_red, 0.01, 10.0);
    const double g3 = g_clamped * g_clamped * g_clamped;

    dvec3 j_obs = g3 * j;
    if (g_clamped > 1.0)
    {
        j_obs.z *= fmin(g_clamped, 2.0);
    }
    else
    {
        j_obs.x *= fmin(1.0 / g_clamped, 2.0);
        j_obs.z *= g_clamped;
    }

    // Radiative transfer accumulation
    const double transmittance = 1.0 - acc_opacity;
    const double dtau = alpha * ds;
    const double absorption_factor = 1.0 - exp(-dtau);
    const double inv_alpha = 1.0 / fmax(alpha, 1e-12);

    acc_color += transmittance * absorption_factor * inv_alpha * j_obs;
    acc_opacity += transmittance * absorption_factor;
    acc_opacity = fmin(acc_opacity, 1.0);
}

// ============================================================================
// Convenience: construct PhysicsParams from scene parameters
// ============================================================================

inline PhysicsParams make_physics_params(double M, double a,
                                         double disk_outer_r, double disk_thickness,
                                         double disk_density, double disk_opacity,
                                         double emission_boost, double color_variation,
                                         double turbulence, double time)
{
    PhysicsParams pp = {};
    pp.bh_mass = M;
    pp.bh_spin = a;
    pp.r_plus = compute_event_horizon(M, a);
    pp.disk_inner_r = compute_isco(M, a);
    pp.disk_outer_r = disk_outer_r;
    pp.disk_thickness = disk_thickness;
    pp.disk_r_ref = sqrt(pp.disk_inner_r * disk_outer_r);
    pp.disk_density0 = disk_density;
    pp.disk_opacity0 = disk_opacity;
    pp.disk_emission_boost = emission_boost;
    pp.disk_color_variation = color_variation;
    pp.disk_turbulence = turbulence;
    pp.disk_time = time;
    return pp;
}

#endif // _BH_PHYSICS_H
