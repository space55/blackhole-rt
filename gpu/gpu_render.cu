// ============================================================================
// GPU Black Hole Ray Tracer — CUDA Implementation
//
// Complete port of the CPU physics (geodesic integration, volumetric RT,
// accretion disk procedural generation) to CUDA device code.  Uses plain
// double-precision scalar math instead of Eigen.  All scene parameters
// live in __constant__ memory for fast broadcast reads.
// ============================================================================

#include "gpu_render.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <chrono>
#include <thread>

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279
#endif

// ============================================================================
// CUDA error checking
// ============================================================================
#define CUDA_CHECK(call)                                            \
    do                                                              \
    {                                                               \
        cudaError_t err = (call);                                   \
        if (err != cudaSuccess)                                     \
        {                                                           \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err));                        \
            return false;                                           \
        }                                                           \
    } while (0)

// ============================================================================
// Constant memory — scene parameters broadcast to every thread
// ============================================================================
__constant__ GPUSceneParams c_params;

// ============================================================================
// Lightweight double-precision vector types (replaces Eigen on device)
// ============================================================================
struct dvec3
{
    double x, y, z;
    __host__ __device__ dvec3() : x(0), y(0), z(0) {}
    __host__ __device__ dvec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    __device__ dvec3 operator+(const dvec3 &b) const { return {x + b.x, y + b.y, z + b.z}; }
    __device__ dvec3 operator-(const dvec3 &b) const { return {x - b.x, y - b.y, z - b.z}; }
    __device__ dvec3 operator*(double s) const { return {x * s, y * s, z * s}; }
    __device__ dvec3 &operator+=(const dvec3 &b)
    {
        x += b.x;
        y += b.y;
        z += b.z;
        return *this;
    }
    __device__ dvec3 &operator*=(double s)
    {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }
    __device__ double dot(const dvec3 &b) const { return x * b.x + y * b.y + z * b.z; }
    __device__ double squaredNorm() const { return x * x + y * y + z * z; }
    __device__ double norm() const { return sqrt(squaredNorm()); }
};

__device__ inline dvec3 operator*(double s, const dvec3 &v)
{
    return {s * v.x, s * v.y, s * v.z};
}

struct dvec4
{
    double t, x, y, z;
    __device__ dvec4() : t(0), x(0), y(0), z(0) {}
    __device__ dvec4(double t_, double x_, double y_, double z_) : t(t_), x(x_), y(y_), z(z_) {}
};

// ============================================================================
// Device helper: clamp
// ============================================================================
__device__ inline double dclamp(double x, double lo, double hi)
{
    return fmin(fmax(x, lo), hi);
}

// ============================================================================
// Black Hole Physics
// ============================================================================

// Kerr-Schild radius from Cartesian coordinates (spin axis = y)
__device__ double gpu_ks_radius(double px, double py, double pz)
{
    const double a2 = c_params.bh_spin * c_params.bh_spin;
    const double rho2 = px * px + py * py + pz * pz;
    const double term = rho2 - a2;
    const double disc = term * term + 4.0 * a2 * py * py;
    const double r2 = 0.5 * (term + sqrt(disc));
    return sqrt(fmax(r2, 1e-12));
}

// ---------------------------------------------------------------------------
// Geodesic acceleration: d²x^μ/dλ² = -Γ^μ_αβ u^α u^β
//
// Direct port of the optimized CPU geodesic_accel — pure scalar math,
// exploits rank-2 KS structure, no 4×4 matrices.
// ---------------------------------------------------------------------------
__device__ dvec3 gpu_geodesic_accel(dvec3 pos, dvec3 vel, double *out_r)
{
    const double px = pos.x, py = pos.y, pz = pos.z;
    const double a = c_params.bh_spin, a2 = a * a;
    const double M = c_params.bh_mass;

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
__device__ double gpu_half_thickness(double r_ks)
{
    return c_params.disk_thickness * (r_ks / c_params.disk_r_ref);
}

// Turbulence-warped half-thickness (azimuthal lumps, gaps, warps)
__device__ double gpu_warped_half_thickness(dvec3 pos, double r_ks)
{
    const double h0 = gpu_half_thickness(fmin(r_ks, c_params.disk_outer_r));
    const double turb = c_params.disk_turbulence;
    if (turb < 1e-6)
        return h0;

    const double phi = atan2(pos.z, pos.x);
    const double log_r = log(fmax(r_ks, 1e-6));

    const double omega = 1.0 / (r_ks * sqrt(r_ks));
    const double tp = c_params.disk_time * omega;

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
__device__ double gpu_clump_factor(dvec3 pos, double r_ks, double warped_h)
{
    const double inner_r = c_params.disk_inner_r;
    const double outer_r = c_params.disk_outer_r;
    if (r_ks < inner_r || r_ks > outer_r)
        return 1.0;

    const double phi = atan2(pos.z, pos.x);
    const double log_r = log(r_ks);
    const double r_norm = (r_ks - inner_r) / (outer_r - inner_r);

    const double omega = 1.0 / (r_ks * sqrt(r_ks));
    const double t_phase = c_params.disk_time * omega;

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
__device__ double gpu_density(dvec3 pos, double r_ks, double warped_h)
{
    const double inner_r = c_params.disk_inner_r;
    const double outer_r = c_params.disk_outer_r;
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

    return c_params.disk_density0 * radial * vertical *
           gpu_clump_factor(pos, r_ks, warped_h) * outer_fade;
}

// Temperature: simplified Novikov-Thorne profile
__device__ double gpu_temperature(double r_ks)
{
    const double inner_r = c_params.disk_inner_r;
    const double outer_r = c_params.disk_outer_r;
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
__device__ dvec3 gpu_temperature_to_rgb(double T)
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

// Emissivity + optional absorption output (avoids redundant density computation)
__device__ dvec3 gpu_emissivity(dvec3 pos, double r_ks, double warped_h, double *alpha_out)
{
    const double rho = gpu_density(pos, r_ks, warped_h);
    if (alpha_out)
        *alpha_out = c_params.disk_opacity0 * rho;
    if (rho < 1e-12)
        return dvec3(0, 0, 0);

    const double T = gpu_temperature(r_ks);
    if (T < 1e-12)
        return dvec3(0, 0, 0);

    const double T4 = T * T * T * T;
    dvec3 color = gpu_temperature_to_rgb(T);

    // Cinematic color variation
    const double cv = c_params.disk_color_variation;
    if (cv > 1e-6)
    {
        const double inner_r = c_params.disk_inner_r;
        const double outer_r = c_params.disk_outer_r;
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

    return dvec3(c_params.disk_emission_boost * rho * T4 * color.x,
                 c_params.disk_emission_boost * rho * T4 * color.y,
                 c_params.disk_emission_boost * rho * T4 * color.z);
}

// Keplerian gas 4-velocity (prograde circular orbit)
// Takes pre-computed KS intermediates (r, H, lx, ly, lz) to avoid redundancy
__device__ dvec4 gpu_gas_four_velocity(dvec3 pos, double r_ks,
                                       double H, double lx, double ly, double lz)
{
    const double inner_r = c_params.disk_inner_r;
    if (r_ks < inner_r)
        return dvec4(-1, 0, 0, 0);

    const double a = c_params.bh_spin;
    const double M = c_params.bh_mass;
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
// Ray Marching
// ============================================================================

// RK2 (far-field) / RK4 (near-field) geodesic integrator
__device__ bool gpu_advance(dvec3 &pos, dvec3 &vel, double &cached_r, double dt)
{
    const double r_plus = c_params.r_plus;

    if (cached_r > 30.0 * r_plus)
    {
        // Far-field: RK2 midpoint (2 evaluations)
        double r1;
        dvec3 a1 = gpu_geodesic_accel(pos, vel, &r1);
        if (r1 <= r_plus)
            return false;

        dvec3 mid_pos = pos + 0.5 * dt * vel;
        dvec3 mid_vel = vel + 0.5 * dt * a1;

        double r2;
        dvec3 a2 = gpu_geodesic_accel(mid_pos, mid_vel, &r2);
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
        dvec3 a_k = gpu_geodesic_accel(pos, vel, &rk);
        if (rk <= r_plus)
            return false;
        dvec3 k1_dx = vel, k1_dv = a_k;

        // k2
        a_k = gpu_geodesic_accel(pos + 0.5 * dt * k1_dx, vel + 0.5 * dt * k1_dv, &rk);
        if (rk <= r_plus)
            return false;
        dvec3 k2_dx = vel + 0.5 * dt * k1_dv;
        dvec3 k2_dv = a_k;

        // k3
        a_k = gpu_geodesic_accel(pos + 0.5 * dt * k2_dx, vel + 0.5 * dt * k2_dv, &rk);
        if (rk <= r_plus)
            return false;
        dvec3 k3_dx = vel + 0.5 * dt * k2_dv;
        dvec3 k3_dv = a_k;

        // k4
        a_k = gpu_geodesic_accel(pos + dt * k3_dx, vel + dt * k3_dv, &rk);
        if (rk <= r_plus)
            return false;
        dvec3 k4_dx = vel + dt * k3_dv;
        dvec3 k4_dv = a_k;

        double s = dt / 6.0;
        pos = pos + s * (k1_dx + 2.0 * k2_dx + 2.0 * k3_dx + k4_dx);
        vel = vel + s * (k1_dv + 2.0 * k2_dv + 2.0 * k3_dv + k4_dv);
    }

    cached_r = gpu_ks_radius(pos.x, pos.y, pos.z);
    return true;
}

// Volumetric disk sampling with relativistic radiative transfer
__device__ void gpu_sample_disk(dvec3 pos, dvec3 vel, double ds,
                                dvec3 &acc_color, double &acc_opacity)
{
    if (acc_opacity >= 0.999)
        return;

    // Compute KS radius once (shared by all sub-functions)
    const double r_ks = gpu_ks_radius(pos.x, pos.y, pos.z);

    // Contains check
    const double fade_limit = c_params.disk_outer_r * 1.2;
    if (r_ks < c_params.disk_inner_r || r_ks > fade_limit)
        return;

    const double warped_h = gpu_warped_half_thickness(pos, r_ks);
    if (fabs(pos.y) > 3.0 * warped_h)
        return;

    // Emissivity + absorption (shares r_ks and warped_h)
    double alpha;
    dvec3 j = gpu_emissivity(pos, r_ks, warped_h, &alpha);

    // KS metric intermediates (computed once, shared by u0 + gas_u + redshift)
    const double a = c_params.bh_spin, a2 = a * a;
    const double M = c_params.bh_mass;
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
    dvec4 gas_u = gpu_gas_four_velocity(pos, r_ks, H, lx, ly, lz);

    // Redshift via g = -k_lower[0] / (k_lower · gas_u)
    // k_lower = g_μν k^ν = η_μν k^ν + 2H l_μ (l · k)
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
// Render Kernel — one thread per pixel, AA loop inside
// ============================================================================
__global__ __launch_bounds__(256) void render_kernel(GPUPixelResult *results, int *progress)
{
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= c_params.width || py >= c_params.height)
        return;

    const int idx = py * c_params.width + px;

    // Camera basis vectors (from constant memory)
    const dvec3 cam_right(c_params.cam_right[0], c_params.cam_right[1], c_params.cam_right[2]);
    const dvec3 cam_up(c_params.cam_up[0], c_params.cam_up[1], c_params.cam_up[2]);
    const dvec3 cam_fwd(c_params.cam_fwd[0], c_params.cam_fwd[1], c_params.cam_fwd[2]);
    const dvec3 cam_pos(c_params.cam_pos[0], c_params.cam_pos[1], c_params.cam_pos[2]);

    const int aa_grid = c_params.aa_grid;
    const double inv_aa = 1.0 / aa_grid;
    const double inv_spp = 1.0 / (double)(aa_grid * aa_grid);

    const double r_plus = c_params.r_plus;
    const double base_dt = c_params.base_dt;
    const double max_affine = c_params.max_affine;
    const double escape_r2 = c_params.escape_r2;
    const double disk_inner_r = c_params.disk_inner_r;
    const double disk_outer_r = c_params.disk_outer_r;
    const double width_d = (double)c_params.width;
    const double height_d = (double)c_params.height;
    const double fov_x = c_params.fov_x;
    const double fov_y = c_params.fov_y;

    dvec3 pixel_disk(0, 0, 0);
    double pixel_sky_weight = 0;
    dvec3 pixel_exit_dir(0, 0, 0);

    for (int sy = 0; sy < aa_grid; sy++)
    {
        for (int sx = 0; sx < aa_grid; sx++)
        {
            const double sub_x = (px + (sx + 0.5) * inv_aa) / width_d;
            const double sub_y = (py + (sy + 0.5) * inv_aa) / height_d;

            // Ray direction (same math as CPU ray_s constructor)
            const double x_rad = ((sub_x - 0.5) * fov_x) * (M_PI / 180.0);
            const double y_rad = ((0.5 - sub_y) * fov_y) * (M_PI / 180.0);

            const double cy = cos(y_rad), cx = cos(x_rad);
            const double local_fwd = cy * cx;
            const double local_up = sin(y_rad);
            const double local_right = cy * sin(x_rad);

            dvec3 vel = local_fwd * cam_fwd + local_up * cam_up + local_right * cam_right;
            dvec3 pos = cam_pos;

            dvec3 acc_color(0, 0, 0);
            double acc_opacity = 0;
            double cached_r = gpu_ks_radius(pos.x, pos.y, pos.z);
            bool hit_bh = false;
            double affine = 0;

            while (affine < max_affine)
            {
                const double delta = fmax(cached_r - r_plus, 0.01);
                double step_dt = base_dt * dclamp(delta * delta, 0.0001, 1.0);

                // Step size reduction near disk
                if (cached_r >= disk_inner_r * 0.8 && cached_r <= disk_outer_r * 1.2)
                {
                    const double h = gpu_half_thickness(cached_r);
                    const double y_dist = fabs(pos.y);
                    if (y_dist < 5.0 * h)
                    {
                        step_dt = fmin(step_dt, fmax(0.3 * h, 0.005));
                    }
                }

                if (!gpu_advance(pos, vel, cached_r, step_dt))
                {
                    hit_bh = true;
                    break;
                }

                gpu_sample_disk(pos, vel, step_dt, acc_color, acc_opacity);

                affine += step_dt;
                if (cached_r <= r_plus || !isfinite(vel.squaredNorm()))
                {
                    hit_bh = true;
                    break;
                }
                if (pos.squaredNorm() > escape_r2)
                    break;
                if (acc_opacity > 0.99)
                    break;
            }

            pixel_disk += acc_color;
            if (!hit_bh)
            {
                double transmittance = 1.0 - acc_opacity;
                pixel_sky_weight += transmittance;
                pixel_exit_dir += transmittance * vel;
            }
        }
    }

    pixel_disk *= inv_spp;
    pixel_sky_weight *= inv_spp;
    pixel_exit_dir *= inv_spp;

    results[idx].disk_r = (float)pixel_disk.x;
    results[idx].disk_g = (float)pixel_disk.y;
    results[idx].disk_b = (float)pixel_disk.z;
    results[idx].sky_weight = (float)pixel_sky_weight;
    results[idx].exit_vx = (float)pixel_exit_dir.x;
    results[idx].exit_vy = (float)pixel_exit_dir.y;
    results[idx].exit_vz = (float)pixel_exit_dir.z;

    // Update progress counter (visible to host via mapped pinned memory)
    atomicAdd(progress, 1);
}

// ============================================================================
// Host Launch Function
// ============================================================================
bool gpu_render(const GPUSceneParams &params, GPUPixelResult *host_results)
{
    // Print GPU info
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("GPU: %s (compute %d.%d, %d SMs, %.0f MHz)\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, prop.clockRate / 1000.0);
    printf("GPU memory: %.0f MB total, double-precision throughput: %s\n",
           prop.totalGlobalMem / (1024.0 * 1024.0),
           (prop.major >= 8) ? "good (Ampere+)" : (prop.major >= 7) ? "decent (Volta+)"
                                                                    : "limited");

    const int num_pixels = params.width * params.height;
    const size_t result_bytes = (size_t)num_pixels * sizeof(GPUPixelResult);

    printf("GPU render: %d x %d = %d pixels, %dx%d AA = %d spp\n",
           params.width, params.height, num_pixels,
           params.aa_grid, params.aa_grid, params.aa_grid * params.aa_grid);
    printf("GPU buffer: %.1f MB\n", result_bytes / (1024.0 * 1024.0));

    // Allocate device memory
    GPUPixelResult *d_results = nullptr;
    CUDA_CHECK(cudaMalloc(&d_results, result_bytes));

    // Copy scene parameters to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(c_params, &params, sizeof(GPUSceneParams)));

    // Allocate mapped pinned memory for progress counter (zero-copy host↔device)
    int *h_progress = nullptr;
    int *d_progress = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_progress, sizeof(int), cudaHostAllocMapped));
    *h_progress = 0;
    CUDA_CHECK(cudaHostGetDevicePointer(&d_progress, h_progress, 0));

    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((params.width + 15) / 16, (params.height + 15) / 16);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    render_kernel<<<grid, block>>>(d_results, d_progress);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));

    // Poll progress until kernel completes
    auto poll_start = std::chrono::steady_clock::now();
    while (cudaEventQuery(stop) == cudaErrorNotReady)
    {
        int done = *h_progress;  // mapped memory: no cudaMemcpy needed
        double pct = 100.0 * done / (double)num_pixels;
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - poll_start).count();
        double rate = (elapsed > 0.01) ? done / elapsed : 0;
        double eta = (rate > 0 && done < num_pixels) ? (num_pixels - done) / rate : 0;
        printf("\rGPU progress: %6.2f%% (%d / %d px)  %.1f Mpx/s  ETA %.1fs   ",
               pct, done, num_pixels, rate / 1e6, eta);
        fflush(stdout);
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    printf("\rGPU progress: 100.00%% (%d / %d px)                              \n",
           num_pixels, num_pixels);

    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("GPU kernel: %.2f seconds (%.1f Mpx/s)\n",
           ms / 1000.0, num_pixels / ms / 1000.0);

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(host_results, d_results, result_bytes, cudaMemcpyDeviceToHost));

    // Cleanup
    cudaFree(d_results);
    cudaFreeHost(h_progress);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return true;
}
