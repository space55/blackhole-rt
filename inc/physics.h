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
    bh_real bh_mass;
    bh_real bh_spin;
    bh_real r_plus; // event horizon radius (precomputed)

    // Disk geometry
    bh_real disk_inner_r; // visible inner edge (may differ from ISCO)
    bh_real disk_isco;    // physical ISCO — used for temperature profile
    bh_real disk_outer_r;
    bh_real disk_thickness; // half-thickness scale height at reference radius
    bh_real disk_r_ref;     // sqrt(inner_r * outer_r), precomputed

    // Disk physical properties
    bh_real disk_density0;
    bh_real disk_opacity0;

    // Disk appearance
    bh_real disk_emission_boost;
    bh_real disk_color_variation;
    bh_real disk_turbulence;
    bh_real disk_time;

    // Stipple / particle texture
    bh_real disk_stipple; // 0 = smooth, 1 = fully particulate (specs & clumps)

    // Flat disk mode: 0 = normal volumetric, 1 = thin/flat with extra texture & opacity
    int disk_flat_mode;

    // LOD / anti-aliasing: camera info for procedural texture frequency clamping.
    // When pixel_angle > 0, high-frequency texture terms are faded to prevent moiré.
    bh_real cam_x, cam_y, cam_z;
    bh_real pixel_angle; // radians per pixel (fov_x / output_width), 0 = no LOD
};

// ============================================================================
// Black Hole — derived quantities
// ============================================================================

BH_FUNC inline bh_real compute_event_horizon(bh_real M, bh_real a)
{
    return M + sqrt(fmax(M * M - a * a, 0.0));
}

BH_FUNC inline bh_real compute_isco(bh_real M, bh_real a)
{
    const bh_real astar = a / M;
    const bh_real Z1 = 1.0 + cbrt(1.0 - astar * astar) *
                                (cbrt(1.0 + astar) + cbrt(1.0 - astar));
    const bh_real Z2 = sqrt(3.0 * astar * astar + Z1 * Z1);
    return M * (3.0 + Z2 - sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2)));
}

// ============================================================================
// Kerr-Schild radius from Cartesian coordinates (spin axis = y)
// ============================================================================

BH_FUNC inline bh_real ks_radius(const dvec3 &pos, bh_real spin)
{
    const bh_real a2 = spin * spin;
    const bh_real rho2 = pos.x * pos.x + pos.y * pos.y + pos.z * pos.z;
    const bh_real term = rho2 - a2;
    const bh_real disc = term * term + 4.0 * a2 * pos.y * pos.y;
    const bh_real r2 = 0.5 * (term + sqrt(disc));
    return sqrt(fmax(r2, 1e-12));
}

// ============================================================================
// LOD helpers — procedural texture anti-aliasing
//
// Computes the effective pixel footprint on the disk surface at a given
// sample point, then provides fade functions for high-frequency terms.
// This prevents moiré when the camera is close to / at grazing angle
// with the disk.
// ============================================================================

// Effective pixel footprint on disk surface (world-space units)
// Returns 0 if LOD is disabled (pixel_angle == 0).
BH_FUNC inline bh_real compute_texel_size(const dvec3 &pos, const PhysicsParams &pp)
{
    if (pp.pixel_angle <= 0.0)
        return 0.0;
    const bh_real dx = pos.x - pp.cam_x;
    const bh_real dy = pos.y - pp.cam_y;
    const bh_real dz = pos.z - pp.cam_z;
    const bh_real cam_dist = sqrt(dx * dx + dy * dy + dz * dz);
    if (cam_dist < 1e-6)
        return 0.0;
    const bh_real texel_base = cam_dist * pp.pixel_angle;
    // Foreshortening: disk normal = (0,1,0), view dir ≈ (pos-cam)/|pos-cam|
    const bh_real sin_incidence = fabs(dy) / cam_dist;
    return texel_base / fmax(sin_incidence, 0.002); // cap at 500× stretch
}

// Frequency fade for sin(k * log_r) terms: wavelength = 2π * r / k
// Returns 1.0 when well-resolved, 0.0 when aliased.
BH_FUNC inline bh_real lod_fade_logr(bh_real k, bh_real r_ks, bh_real texel_size)
{
    if (texel_size <= 0.0)
        return 1.0;
    const bh_real wavelength = 2.0 * M_PI * r_ks / k;
    const bh_real nyquist = 2.0 * texel_size;
    if (wavelength >= 4.0 * nyquist)
        return 1.0;
    if (wavelength <= nyquist)
        return 0.0;
    const bh_real t = (wavelength - nyquist) / (3.0 * nyquist);
    return t * t * (3.0 - 2.0 * t); // smoothstep
}

// Frequency fade for streak(k * log_r, sharpness) terms.
// pow(sin², sharpness) creates features sqrt(sharpness) times narrower than
// the base wavelength.  Effective frequency ≈ k * min(sqrt(sharpness), 8).
BH_FUNC inline bh_real lod_fade_streak(bh_real k, bh_real sharpness, bh_real r_ks, bh_real texel_size)
{
    const bh_real eff_k = k * fmin(sqrt(sharpness), 8.0);
    return lod_fade_logr(eff_k, r_ks, texel_size);
}

// LOD-aware arc window: product of sin² at incommensurate radial frequencies
// creates quasi-random azimuthal windows (fragmented arcs).
// When the camera is close, the 2D lattice pattern of the window itself
// becomes visible as a warped checkerboard.  Fade toward 1.0 (full rings,
// no fragmentation) when the window's own spatial frequency aliases.
BH_FUNC inline bh_real lod_arc_window(bh_real phi, bh_real log_r, bh_real seed,
                                     bh_real brevity, bh_real r_ks, bh_real texel_size)
{
    const bh_real w1 = sin(1.0 * phi + 2.71 * log_r + seed);
    const bh_real w2 = sin(2.0 * phi - 4.33 * log_r + seed * 1.7 + 1.1);
    const bh_real w3 = sin(3.0 * phi + 7.19 * log_r + seed * 0.6 + 3.4);
    const bh_real window = pow(fmax(w1 * w1 * w2 * w2 * w3 * w3, 1e-12), brevity);

    // The highest radial freq component is 7.19 in log_r space.
    // brevity sharpens all three sin² terms jointly.
    const bh_real eff_k = 7.19 * fmin(sqrt(fmax(brevity, 1.0)), 4.0);
    const bh_real fade = lod_fade_logr(eff_k, r_ks, texel_size);
    // fade=1 → well-resolved → return window; fade=0 → aliased → return 1.0
    return 1.0 - fade * (1.0 - window);
}

// Frequency fade for sin(k * r) or streak(k * r, ...) terms: wavelength ≈ π/k
BH_FUNC inline bh_real lod_fade_linear(bh_real k, bh_real texel_size)
{
    if (texel_size <= 0.0)
        return 1.0;
    const bh_real wavelength = M_PI / k;
    const bh_real nyquist = 2.0 * texel_size;
    if (wavelength >= 4.0 * nyquist)
        return 1.0;
    if (wavelength <= nyquist)
        return 0.0;
    const bh_real t = (wavelength - nyquist) / (3.0 * nyquist);
    return t * t * (3.0 - 2.0 * t);
}

// ============================================================================
// Geodesic acceleration: d²x^μ/dλ² = -Γ^μ_αβ u^α u^β
//
// Exploits the rank-2 Kerr-Schild structure — no 4×4 matrices needed.
// Optionally outputs the KS radius via out_r.
// ============================================================================

BH_FUNC inline dvec3 geodesic_accel(const dvec3 &pos, const dvec3 &vel,
                                    const PhysicsParams &pp, bh_real *out_r = nullptr)
{
    const bh_real px = pos.x, py = pos.y, pz = pos.z;
    const bh_real a = pp.bh_spin, a2 = a * a;
    const bh_real M = pp.bh_mass;

    // KS radius
    const bh_real rho2 = px * px + py * py + pz * pz;
    const bh_real term = rho2 - a2;
    const bh_real r2 = 0.5 * (term + sqrt(term * term + 4.0 * a2 * py * py));
    const bh_real r = sqrt(fmax(r2, 1e-12));
    const bh_real inv_r = 1.0 / r;
    const bh_real inv_r2 = inv_r * inv_r;

    if (out_r)
        *out_r = r;

    // Null vector l, Sigma, H
    const bh_real P = r2 + a2;
    const bh_real inv_P = 1.0 / P;
    const bh_real inv_P2 = inv_P * inv_P;
    const bh_real lx = (r * px + a * pz) * inv_P;
    const bh_real ly = py * inv_r;
    const bh_real lz = (r * pz - a * px) * inv_P;

    const bh_real Sigma = r2 + a2 * py * py * inv_r2;
    const bh_real inv_Sigma = 1.0 / Sigma;
    const bh_real H = M * r * inv_Sigma;

    // u^0 from null condition
    const bh_real twoH = 2.0 * H;
    const bh_real g00 = -1.0 + twoH;
    const bh_real v1 = vel.x, v2 = vel.y, v3 = vel.z;
    const bh_real lv = lx * v1 + ly * v2 + lz * v3;
    const bh_real b_u0 = 2.0 * twoH * lv;
    const bh_real v_sq = v1 * v1 + v2 * v2 + v3 * v3;
    const bh_real c_u0 = v_sq + twoH * lv * lv;

    const bh_real disc_u0 = b_u0 * b_u0 - 4.0 * g00 * c_u0;
    const bh_real sqrt_disc = sqrt(fmax(disc_u0, 0.0));
    const bh_real inv_2g00 = 0.5 / g00;
    const bh_real u0a = (-b_u0 + sqrt_disc) * inv_2g00;
    const bh_real u0b = (-b_u0 - sqrt_disc) * inv_2g00;
    const bh_real u0 = (u0a < 0.0) ? u0a : u0b;

    // L = l · u
    const bh_real L = u0 + lx * v1 + ly * v2 + lz * v3;

    // Partial derivative intermediates
    const bh_real inv_Sigma2 = inv_Sigma * inv_Sigma;
    const bh_real r_inv_Sigma = r * inv_Sigma;

    const bh_real dr0 = px * r_inv_Sigma;
    const bh_real dr1 = py * P * inv_r * inv_Sigma;
    const bh_real dr2 = pz * r_inv_Sigma;

    const bh_real inv_r3 = inv_r * inv_r2;
    const bh_real sig_r_coeff = 2.0 * r - 2.0 * a2 * py * py * inv_r3;
    const bh_real dSigma0 = sig_r_coeff * dr0;
    const bh_real dSigma1 = sig_r_coeff * dr1 + 2.0 * a2 * py * inv_r2;
    const bh_real dSigma2 = sig_r_coeff * dr2;

    const bh_real mass_inv_S2 = M * inv_Sigma2;
    const bh_real dH0 = mass_inv_S2 * (dr0 * Sigma - r * dSigma0);
    const bh_real dH1 = mass_inv_S2 * (dr1 * Sigma - r * dSigma1);
    const bh_real dH2 = mass_inv_S2 * (dr2 * Sigma - r * dSigma2);

    // dl_i factored form
    const bh_real rxaz = r * px + a * pz;
    const bh_real rzax = r * pz - a * px;
    const bh_real r_inv_P = r * inv_P;
    const bh_real a_inv_P = a * inv_P;
    const bh_real K_x = (px * P - 2.0 * r * rxaz) * inv_P2;
    const bh_real K_y = -py * inv_r2;
    const bh_real K_z = (pz * P - 2.0 * r * rzax) * inv_P2;

    const bh_real dlx0 = dr0 * K_x + r_inv_P;
    const bh_real dly0 = K_y * dr0;
    const bh_real dlz0 = dr0 * K_z - a_inv_P;

    const bh_real dlx1 = dr1 * K_x;
    const bh_real dly1 = K_y * dr1 + inv_r;
    const bh_real dlz1 = dr1 * K_z;

    const bh_real dlx2 = dr2 * K_x + a_inv_P;
    const bh_real dly2 = K_y * dr2;
    const bh_real dlz2 = dr2 * K_z + r_inv_P;

    // D[i] = dl_i · u
    const bh_real D0 = dlx0 * v1 + dly0 * v2 + dlz0 * v3;
    const bh_real D1 = dlx1 * v1 + dly1 * v2 + dlz1 * v3;
    const bh_real D2 = dlx2 * v1 + dly2 * v2 + dlz2 * v3;

    // P_i = (2*dH_i*L + 2*H*D_i) * l + 2*H*L * dl_i
    const bh_real twoHL = twoH * L;
    const bh_real c0 = 2.0 * dH0 * L + twoH * D0;
    const bh_real c1 = 2.0 * dH1 * L + twoH * D1;
    const bh_real c2 = 2.0 * dH2 * L + twoH * D2;

    const bh_real P0_0 = c0, P0_1 = c0 * lx + twoHL * dlx0;
    const bh_real P0_2 = c0 * ly + twoHL * dly0, P0_3 = c0 * lz + twoHL * dlz0;

    const bh_real P1_0 = c1, P1_1 = c1 * lx + twoHL * dlx1;
    const bh_real P1_2 = c1 * ly + twoHL * dly1, P1_3 = c1 * lz + twoHL * dlz1;

    const bh_real P2_0 = c2, P2_1 = c2 * lx + twoHL * dlx2;
    const bh_real P2_2 = c2 * ly + twoHL * dly2, P2_3 = c2 * lz + twoHL * dlz2;

    // Q_i = 2*dH_i*L² + 4*H*L*D_i
    const bh_real L2 = L * L;
    const bh_real fourHL = 2.0 * twoHL;
    const bh_real Q0 = 2.0 * dH0 * L2 + fourHL * D0;
    const bh_real Q1 = 2.0 * dH1 * L2 + fourHL * D1;
    const bh_real Q2 = 2.0 * dH2 * L2 + fourHL * D2;

    // F(ν) = 2*T1(ν) - T3(ν)
    const bh_real F0 = 2.0 * (v1 * P0_0 + v2 * P1_0 + v3 * P2_0);
    const bh_real F1 = 2.0 * (v1 * P0_1 + v2 * P1_1 + v3 * P2_1) - Q0;
    const bh_real F2 = 2.0 * (v1 * P0_2 + v2 * P1_2 + v3 * P2_2) - Q1;
    const bh_real F3 = 2.0 * (v1 * P0_3 + v2 * P1_3 + v3 * P2_3) - Q2;

    // a^μ = -½ g_inv^{μν} F_ν using KS form
    const bh_real S = -F0 + lx * F1 + ly * F2 + lz * F3;
    const bh_real twoHS = twoH * S;

    return dvec3(
        -0.5 * (F1 - lx * twoHS),
        -0.5 * (F2 - ly * twoHS),
        -0.5 * (F3 - lz * twoHS));
}

// ============================================================================
// Accretion Disk Physics
// ============================================================================

// Half-thickness: linearly flared, h(r) = thickness * (r / r_ref)
BH_FUNC inline bh_real disk_half_thickness(bh_real r_ks, const PhysicsParams &pp)
{
    bh_real h = pp.disk_thickness * (r_ks / pp.disk_r_ref);
    if (pp.disk_flat_mode)
        h *= 0.15; // flatten to ~15% of normal thickness
    return h;
}

// Turbulence-warped half-thickness (azimuthal lumps, gaps, warps)
BH_FUNC inline bh_real disk_warped_half_thickness(const dvec3 &pos, bh_real r_ks,
                                                 const PhysicsParams &pp)
{
    const bh_real h0 = disk_half_thickness(fmin(r_ks, pp.disk_outer_r), pp);
    bh_real turb = pp.disk_turbulence;
    if (pp.disk_flat_mode)
        turb *= 0.65; // allow substantial geometric warping even when flat
    if (turb < 1e-6)
        return h0;

    const bh_real phi = atan2(pos.z, pos.x);
    const bh_real log_r = log(fmax(r_ks, 1e-6));

    const bh_real omega = 1.0 / (r_ks * sqrt(r_ks));
    const bh_real tp = pp.disk_time * omega;

    bh_real warp = 0.0;
    // Circumferential warps — high radial frequency, low azimuthal → concentric ripples
    warp += 0.55 * sin(1.0 * (phi + tp) - 3.0 * log_r + 1.2);
    warp += 0.35 * sin(1.0 * (phi + tp) - 5.0 * log_r + 4.1);
    warp += 0.25 * sin(1.0 * (phi + tp) - 8.0 * log_r + 0.7);
    warp += 0.18 * sin(2.0 * (phi + tp) - 12.0 * log_r + 2.9);
    warp += 0.12 * sin(2.0 * (phi + tp) - 18.0 * log_r + 5.3);
    warp += 0.08 * sin(3.0 * (phi + tp) - 25.0 * log_r + 3.1);

    // Narrow gaps — concentric dark rings
    auto dip = [](bh_real phase, bh_real sharpness) -> bh_real
    {
        bh_real s = sin(phase);
        return pow(s * s, sharpness);
    };
    warp -= 0.70 * dip(1.0 * (phi + tp) - 4.0 * log_r + 2.5, 40.0);
    warp -= 0.40 * dip(1.0 * (phi + tp) - 7.0 * log_r + 0.3, 50.0);

    // Flat-mode extra high-frequency geometric turbulence
    if (pp.disk_flat_mode)
    {
        warp += 0.45 * sin(2.0 * (phi + tp) - 16.0 * log_r + 0.9);
        warp += 0.35 * sin(3.0 * (phi + tp) - 22.0 * log_r + 3.6);
        warp += 0.25 * sin(4.0 * (phi + tp) - 30.0 * log_r + 5.8);
        warp -= 0.50 * dip(1.0 * (phi + tp) - 10.0 * log_r + 1.7, 50.0);
        warp -= 0.35 * dip(2.0 * (phi + tp) - 15.0 * log_r + 4.3, 60.0);
    }

    const bh_real factor = 1.0 + turb * warp;
    return h0 * fmax(factor, 0.05);
}

// ============================================================================
// Procedural hash & noise for stipple / particle texture
// ============================================================================

// Integer hash — CPU + GPU compatible (Muller finalizer)
BH_FUNC inline unsigned int bh_hash(unsigned int x)
{
    x = ((x >> 16) ^ x) * 0x45d9f3bu;
    x = ((x >> 16) ^ x) * 0x45d9f3bu;
    x = (x >> 16) ^ x;
    return x;
}

// Hash 3D integer coords to [0,1]
BH_FUNC inline bh_real bh_hash_01(int ix, int iy, int iz)
{
    unsigned int h = bh_hash(
        (unsigned int)(ix * 374761393 + iy * 668265263 + iz * 1274126177));
    return (h & 0x00FFFFFFu) * (1.0 / 16777215.0);
}

// Quintic smooth interpolant (C2 continuous — no grid artifacts)
BH_FUNC inline bh_real bh_smootherstep(bh_real t)
{
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// 3D value noise in [0,1] — trilinear interpolation of hashed lattice values
BH_FUNC inline bh_real bh_value_noise(bh_real x, bh_real y, bh_real z)
{
    int ix = (int)floor(x), iy = (int)floor(y), iz = (int)floor(z);
    bh_real fx = x - ix, fy = y - iy, fz = z - iz;
    bh_real sx = bh_smootherstep(fx), sy = bh_smootherstep(fy), sz = bh_smootherstep(fz);

    bh_real c000 = bh_hash_01(ix, iy, iz), c100 = bh_hash_01(ix + 1, iy, iz);
    bh_real c010 = bh_hash_01(ix, iy + 1, iz), c110 = bh_hash_01(ix + 1, iy + 1, iz);
    bh_real c001 = bh_hash_01(ix, iy, iz + 1), c101 = bh_hash_01(ix + 1, iy, iz + 1);
    bh_real c011 = bh_hash_01(ix, iy + 1, iz + 1), c111 = bh_hash_01(ix + 1, iy + 1, iz + 1);

    bh_real x00 = c000 + sx * (c100 - c000), x10 = c010 + sx * (c110 - c010);
    bh_real x01 = c001 + sx * (c101 - c001), x11 = c011 + sx * (c111 - c011);
    bh_real xy0 = x00 + sy * (x10 - x00), xy1 = x01 + sy * (x11 - x01);
    return xy0 + sz * (xy1 - xy0);
}

// Stipple / particle factor: creates discrete bright specks and dim voids.
// Returns a multiplicative modulation on disk density.
//   stipple = 0 → returns 1.0 (no effect)
//   stipple = 1 → full particulate texture (bright specks, dark gaps)
BH_FUNC inline bh_real disk_stipple_factor(const dvec3 &pos, bh_real r_ks,
                                          bh_real warped_h, const PhysicsParams &pp)
{
    if (pp.disk_stipple < 1e-6)
        return 1.0;

    const bh_real inner_r = pp.disk_inner_r;
    const bh_real outer_r = pp.disk_outer_r;
    const bh_real stip_inner = pp.disk_flat_mode ? inner_r * 0.5 : inner_r;
    const bh_real stip_outer = pp.disk_flat_mode ? outer_r * 1.4 : outer_r;
    if (r_ks < stip_inner || r_ks > stip_outer)
        return 1.0;

    const bh_real y_h = (warped_h > 1e-12) ? (pos.y / warped_h) : 0.0;

    // Keplerian rotation so particles orbit with the disk.
    // Use rotated Cartesian coordinates instead of atan2(phi) to avoid
    // the hard seam at the ±π wrap-around of atan2.
    const bh_real omega = 1.0 / (r_ks * sqrt(r_ks));
    const bh_real rot_angle = pp.disk_time * omega;
    const bh_real cos_rot = cos(rot_angle);
    const bh_real sin_rot = sin(rot_angle);
    const bh_real rx = pos.x * cos_rot - pos.z * sin_rot;
    const bh_real rz = pos.x * sin_rot + pos.z * cos_rot;

    // Multi-octave thresholded value noise → discrete specks at each scale
    bh_real particle = 0.0;

    // LOD: fade high-frequency stipple octaves to prevent moiré
    const bh_real texel = compute_texel_size(pos, pp);

    // Octave 1: medium specks (dominant visible particle texture)
    // Noise cell size = 1/4 = 0.25 units → fade when texel > 0.125
    {
        const bh_real fade = lod_fade_linear(4.0, texel);
        if (fade > 1e-6)
        {
            bh_real u = rx * 4.0;
            bh_real v = rz * 4.0;
            bh_real w = y_h * 4.0;
            bh_real n = bh_value_noise(u, v, w);
            bh_real spec = dclamp((n - 0.55) / 0.14, 0.0, 1.0);
            particle += fade * 0.55 * spec * spec;
        }
    }

    // Octave 2: fine dust (dense small particles)
    // Noise cell size = 1/10 = 0.1 units → fade when texel > 0.05
    {
        const bh_real fade = lod_fade_linear(10.0, texel);
        if (fade > 1e-6)
        {
            bh_real u = rx * 10.0 + 7.3;
            bh_real v = rz * 10.0 + 3.1;
            bh_real w = y_h * 10.0 + 1.7;
            bh_real n = bh_value_noise(u, v, w);
            bh_real spec = dclamp((n - 0.50) / 0.16, 0.0, 1.0);
            particle += fade * 0.30 * spec;
        }
    }

    // Octave 3: coarse bright clumps (sparse, large, eye-catching)
    {
        bh_real u = rx * 1.5 + 2.9;
        bh_real v = rz * 1.5 + 5.7;
        bh_real w = y_h * 2.0 + 0.4;
        bh_real n = bh_value_noise(u, v, w);
        bh_real spec = dclamp((n - 0.62) / 0.10, 0.0, 1.0);
        particle += 0.15 * spec * spec * spec; // cubed: very sparse bright knots
    }

    particle = fmin(particle, 1.0);

    // Factor: voids between particles are dim, particles are bright
    const bh_real void_level = 0.08; // material between specks (very dim)
    const bh_real peak_level = 3.0;  // particles are brighter than the smooth disk
    const bh_real factor = void_level + (peak_level - void_level) * particle;

    // Blend by stipple amount: 0 → 1.0 (no change), 1 → full particle texture
    return 1.0 + pp.disk_stipple * (factor - 1.0);
}

// Procedural clump / streak modulation
BH_FUNC inline bh_real disk_clump_factor(const dvec3 &pos, bh_real r_ks,
                                        bh_real warped_h, const PhysicsParams &pp)
{
    const bh_real inner_r = pp.disk_inner_r;
    const bh_real outer_r = pp.disk_outer_r;

    // In flat mode: extend texture well inside ISCO (infalling streaks)
    // and beyond outer edge (trailing filaments)
    const bh_real tex_inner = pp.disk_flat_mode ? inner_r * 0.5 : inner_r;
    const bh_real tex_outer = pp.disk_flat_mode ? outer_r * 1.4 : outer_r;
    if (r_ks < tex_inner || r_ks > tex_outer)
        return 1.0;

    const bh_real phi = atan2(pos.z, pos.x);
    const bh_real log_r = log(r_ks);
    const bh_real r_norm = dclamp((r_ks - tex_inner) / (tex_outer - tex_inner), 0.0, 1.0);

    const bh_real omega = 1.0 / (r_ks * sqrt(r_ks));
    const bh_real t_phase = pp.disk_time * omega;

    const bh_real y_over_h = (warped_h > 1e-12) ? (pos.y / warped_h) : 0.0;
    const bh_real y_phase = 3.0 * y_over_h;

    // LOD: compute texel footprint for frequency clamping
    const bh_real texel = compute_texel_size(pos, pp);

    auto streak = [](bh_real phase, bh_real sharpness) -> bh_real
    {
        const bh_real s = sin(phase);
        return pow(s * s, sharpness);
    };

    // Arc-length window: product of sin² at incommensurate radial frequencies
    // creates quasi-random azimuthal windows (fragmented arcs).
    // brevity controls arc shortness: 0 = full rings, 1 = ~half-orbit, 3+ = ~1/10th orbit
    // seed offsets ensure each streak fragments independently.
    // LOD: fades toward 1.0 (full rings) when pattern would alias.
    auto arc_window = [r_ks, texel](bh_real phi, bh_real log_r, bh_real seed, bh_real brevity) -> bh_real
    {
        return lod_arc_window(phi, log_r, seed, brevity, r_ks, texel);
    };

    bh_real mod = 0.0;

    // Large-scale concentric arcs — radial freq dominates for Gargantua look
    mod += lod_fade_streak(4.0, 30.0, r_ks, texel) * 1.4 * streak(1.0 * (phi + t_phase) - 4.0 * log_r + 0.5 + y_phase, 30.0) * arc_window(phi, log_r, 0.5, 0.4);
    mod += lod_fade_streak(6.0, 25.0, r_ks, texel) * 0.7 * streak(1.0 * (phi + t_phase) - 6.0 * log_r + 2.1 + 1.5 * y_phase, 25.0) * arc_window(phi, log_r, 2.1, 0.5);

    // Medium-scale concentric bands
    mod += lod_fade_streak(10.0, 20.0, r_ks, texel) * 0.45 * streak(2.0 * (phi + t_phase) - 10.0 * log_r + 1.3 + 2.0 * y_phase, 20.0) * arc_window(phi, log_r, 1.3, 0.6);
    mod += lod_fade_streak(14.0, 20.0, r_ks, texel) * 0.35 * streak(2.0 * (phi + t_phase) - 14.0 * log_r - 0.8 + 2.5 * y_phase, 20.0) * arc_window(phi, log_r, 3.8, 0.7);

    // Fine-scale concentric filaments
    mod += lod_fade_logr(20.0, r_ks, texel) * 0.25 * sin(3.0 * (phi + t_phase) - 20.0 * log_r + 3.7 + 3.0 * y_phase) * arc_window(phi, log_r, 3.7, 0.8);
    mod += lod_fade_logr(28.0, r_ks, texel) * 0.15 * sin(3.0 * (phi + t_phase) - 28.0 * log_r + 0.4 + 4.0 * y_phase) * arc_window(phi, log_r, 0.4, 0.9);
    mod += lod_fade_logr(36.0, r_ks, texel) * 0.10 * sin(4.0 * (phi + t_phase) - 36.0 * log_r + 5.2 + 5.0 * y_phase) * arc_window(phi, log_r, 5.2, 1.0);

    // Radial hot-spot rings
    mod += lod_fade_linear(5.0, texel) * 0.40 * streak(5.0 * r_ks + 1.0 + 0.5 * y_phase, 20.0) *
           cos(1.0 * (phi + t_phase) - 3.0 * log_r + y_phase) * arc_window(phi, log_r, 1.0, 0.8);

    // Bright knots
    mod += lod_fade_streak(10.0, 20.0, r_ks, texel) * 0.35 * streak(4.0 * M_PI * r_norm, 20.0) *
           streak(2.0 * (phi + t_phase) - 10.0 * log_r + 1.9 + 2.0 * y_phase, 20.0) * arc_window(phi, log_r, 1.9, 1.2);

    // Dark lanes — concentric gaps
    mod -= lod_fade_streak(6.0, 40.0, r_ks, texel) * 0.50 * streak(1.0 * (phi + t_phase) - 6.0 * log_r + 4.0 + 1.8 * y_phase, 40.0) * arc_window(phi, log_r, 4.0, 0.6);
    mod -= lod_fade_streak(10.0, 50.0, r_ks, texel) * 0.30 * streak(1.0 * (phi + t_phase) - 10.0 * log_r + 0.7 + 2.2 * y_phase, 50.0) * arc_window(phi, log_r, 0.7, 0.7);

    // --- Flat-mode extra turbulence: heavy procedural detail at all scales ---
    if (pp.disk_flat_mode)
    {
        // Large-scale concentric arcs (dominant structure)
        mod += lod_fade_streak(5.0, 20.0, r_ks, texel) * 1.0 * streak(1.0 * (phi + t_phase) - 5.0 * log_r + 3.3 + y_phase, 20.0) * arc_window(phi, log_r, 3.3, 0.5);
        mod += lod_fade_streak(8.0, 25.0, r_ks, texel) * 0.80 * streak(1.0 * (phi + t_phase) - 8.0 * log_r + 0.8 + 1.2 * y_phase, 25.0) * arc_window(phi, log_r, 0.8, 0.6);

        // Fine concentric filaments — many closely-spaced rings
        mod += lod_fade_streak(15.0, 30.0, r_ks, texel) * 0.65 * streak(2.0 * (phi + t_phase) - 15.0 * log_r + 2.3 + 1.5 * y_phase, 30.0) * arc_window(phi, log_r, 2.3, 0.8);
        mod += lod_fade_streak(22.0, 25.0, r_ks, texel) * 0.50 * streak(3.0 * (phi + t_phase) - 22.0 * log_r + 4.7 + 2.0 * y_phase, 25.0) * arc_window(phi, log_r, 4.7, 0.9);
        mod += lod_fade_streak(30.0, 20.0, r_ks, texel) * 0.40 * streak(3.0 * (phi + t_phase) - 30.0 * log_r + 1.1 + 3.0 * y_phase, 20.0) * arc_window(phi, log_r, 1.1, 1.0);
        mod += lod_fade_logr(40.0, r_ks, texel) * 0.30 * sin(4.0 * (phi + t_phase) - 40.0 * log_r + 3.9 + 2.5 * y_phase) * arc_window(phi, log_r, 3.9, 1.1);
        mod += lod_fade_logr(50.0, r_ks, texel) * 0.25 * sin(5.0 * (phi + t_phase) - 50.0 * log_r + 0.6 + 4.0 * y_phase) * arc_window(phi, log_r, 0.6, 1.2);
        mod += lod_fade_logr(60.0, r_ks, texel) * 0.20 * sin(6.0 * (phi + t_phase) - 60.0 * log_r + 1.4 + 5.0 * y_phase) * arc_window(phi, log_r, 1.4, 1.3);
        mod += lod_fade_logr(70.0, r_ks, texel) * 0.15 * sin(7.0 * (phi + t_phase) - 70.0 * log_r + 4.2 + 6.0 * y_phase) * arc_window(phi, log_r, 4.2, 1.4);

        // Dense concentric ring structure (Saturn-like, many bands)
        // Rings get arc-windowed too — not perfect circles
        mod += lod_fade_linear(6.0, texel) * 0.70 * streak(6.0 * r_ks + 0.3, 30.0) * arc_window(phi, log_r, 0.3, 0.3);
        mod += lod_fade_linear(11.0, texel) * 0.55 * streak(11.0 * r_ks + 1.7, 25.0) * arc_window(phi, log_r, 1.7, 0.4);
        mod += lod_fade_linear(18.0, texel) * 0.45 * streak(18.0 * r_ks + 4.1, 20.0) * arc_window(phi, log_r, 4.1, 0.5);
        mod += lod_fade_linear(27.0, texel) * 0.35 * streak(27.0 * r_ks + 2.9, 25.0) * arc_window(phi, log_r, 2.9, 0.6);
        mod += lod_fade_linear(40.0, texel) * 0.25 * streak(40.0 * r_ks + 0.6, 20.0) * arc_window(phi, log_r, 0.6, 0.7);

        // Sharp dark concentric gaps / ring divisions (lightly windowed)
        mod -= lod_fade_streak(14.0, 60.0, r_ks, texel) * 0.70 * streak(2.0 * (phi + t_phase) - 14.0 * log_r + 1.4 + 2.0 * y_phase, 60.0) * arc_window(phi, log_r, 1.4, 0.3);
        mod -= lod_fade_streak(20.0, 70.0, r_ks, texel) * 0.55 * streak(3.0 * (phi + t_phase) - 20.0 * log_r + 3.8 + 1.5 * y_phase, 70.0) * arc_window(phi, log_r, 3.8, 0.35);
        mod -= lod_fade_streak(28.0, 50.0, r_ks, texel) * 0.45 * streak(3.0 * (phi + t_phase) - 28.0 * log_r + 0.5 + 2.5 * y_phase, 50.0) * arc_window(phi, log_r, 0.5, 0.4);
        mod -= lod_fade_streak(38.0, 80.0, r_ks, texel) * 0.35 * streak(4.0 * (phi + t_phase) - 38.0 * log_r + 2.7 + 1.8 * y_phase, 80.0) * arc_window(phi, log_r, 2.7, 0.45);

        // Dense mottling / granularity (2D — radial-dominant)
        mod += lod_fade_logr(45.0, r_ks, texel) * 0.40 * sin(5.0 * phi - 35.0 * log_r + 2.2) *
               sin(6.0 * phi - 45.0 * log_r + 5.1);
        mod += lod_fade_logr(65.0, r_ks, texel) * 0.30 * sin(8.0 * phi - 55.0 * log_r + 1.0) *
               cos(10.0 * phi - 65.0 * log_r + 3.7);

        // Caustic-like bright concentric filaments
        mod += fmin(lod_fade_streak(12.0, 50.0, r_ks, texel), lod_fade_linear(15.0, texel)) * 0.50 * streak(2.0 * (phi + t_phase) - 12.0 * log_r + 2.0, 50.0) *
               streak(15.0 * r_ks + 3.3, 30.0) * arc_window(phi, log_r, 2.0, 1.0);
        mod += fmin(lod_fade_streak(18.0, 40.0, r_ks, texel), lod_fade_linear(21.0, texel)) * 0.35 * streak(2.0 * (phi + t_phase) - 18.0 * log_r + 4.5, 40.0) *
               streak(21.0 * r_ks + 1.1, 25.0) * arc_window(phi, log_r, 4.5, 1.2);

        // Turbulent "froth" — radial-dominant fine texture
        const bh_real froth_fade = lod_fade_logr(55.0, r_ks, texel);
        const bh_real froth1 = sin(6.0 * phi + 40.0 * log_r + 1.3 + t_phase * 2.0);
        const bh_real froth2 = sin(8.0 * phi - 55.0 * log_r + 4.6 + t_phase * 3.0);
        mod += froth_fade * 0.35 * froth1 * froth2;

        // ---- Edge streaks: azimuthally narrow, radially elongated bright
        //      filaments at both inner and outer edges (Gargantua hallmark) ----
        // Inner-edge plunging streaks: short fragmented arcs near ISCO
        // High brevity (2.0-3.5) → arcs span only ~1/10th of orbit
        const bh_real inner_prox = exp(-4.0 * (r_norm - 0.0) * (r_norm - 0.0));
        mod += lod_fade_streak(6.0, 40.0, r_ks, texel) * 1.8 * inner_prox * streak(1.0 * (phi + t_phase * 1.5) - 6.0 * log_r + 0.7, 40.0) * arc_window(phi, log_r, 0.7, 2.5);
        mod += lod_fade_streak(10.0, 50.0, r_ks, texel) * 1.2 * inner_prox * streak(1.0 * (phi + t_phase * 1.3) - 10.0 * log_r + 2.9, 50.0) * arc_window(phi, log_r, 2.9, 3.0);
        mod += lod_fade_streak(15.0, 60.0, r_ks, texel) * 0.8 * inner_prox * streak(2.0 * (phi + t_phase * 1.1) - 15.0 * log_r + 4.4, 60.0) * arc_window(phi, log_r, 4.4, 3.5);
        mod += lod_fade_streak(20.0, 30.0, r_ks, texel) * 0.5 * inner_prox * streak(2.0 * (phi + t_phase * 0.9) - 20.0 * log_r + 1.2, 30.0) * arc_window(phi, log_r, 1.2, 2.0);

        // Outer-edge trailing arcs: short fragments fading at disk edge
        // High brevity (2.0-3.0) → scattered short arcs
        const bh_real outer_prox = exp(-4.0 * (r_norm - 1.0) * (r_norm - 1.0));
        mod += lod_fade_streak(4.0, 30.0, r_ks, texel) * 1.4 * outer_prox * streak(1.0 * (phi + t_phase) - 4.0 * log_r + 1.5, 30.0) * arc_window(phi, log_r, 1.5, 2.0);
        mod += lod_fade_streak(7.0, 40.0, r_ks, texel) * 1.0 * outer_prox * streak(1.0 * (phi + t_phase) - 7.0 * log_r + 3.8, 40.0) * arc_window(phi, log_r, 3.8, 2.5);
        mod += lod_fade_streak(11.0, 50.0, r_ks, texel) * 0.7 * outer_prox * streak(2.0 * (phi + t_phase) - 11.0 * log_r + 0.3, 50.0) * arc_window(phi, log_r, 0.3, 3.0);
        mod += lod_fade_streak(16.0, 30.0, r_ks, texel) * 0.4 * outer_prox * streak(2.0 * (phi + t_phase) - 16.0 * log_r + 5.6, 30.0) * arc_window(phi, log_r, 5.6, 2.5);

        // Mid-radius bright arcs (gravitational lensing caustic lines)
        // Moderate brevity (1.5) → medium-length arcs
        const bh_real mid_prox = exp(-6.0 * (r_norm - 0.3) * (r_norm - 0.3));
        mod += lod_fade_streak(4.0, 60.0, r_ks, texel) * 0.9 * mid_prox * streak(1.0 * (phi + t_phase) - 4.0 * log_r + 2.2, 60.0) * arc_window(phi, log_r, 2.2, 1.5);
        mod += lod_fade_streak(7.0, 50.0, r_ks, texel) * 0.6 * mid_prox * streak(1.0 * (phi + t_phase) - 7.0 * log_r + 0.8, 50.0) * arc_window(phi, log_r, 0.8, 1.8);
    }

    // Radial envelope
    if (pp.disk_flat_mode)
    {
        // Edge-preserving envelope: bright at inner edge (infalling),
        // gentle taper at outer edge, never kills texture completely
        const bh_real inner_edge = dclamp((r_ks - tex_inner) / (0.15 * (tex_outer - tex_inner)), 0.0, 1.0);
        const bh_real outer_edge = dclamp((tex_outer - r_ks) / (0.25 * (tex_outer - tex_inner)), 0.0, 1.0);
        const bh_real edge_env = sqrt(inner_edge) * sqrt(outer_edge);
        // Inner-edge brightening: streaks near ISCO glow hotter
        const bh_real inner_boost = 1.0 + 2.0 * exp(-8.0 * r_norm * r_norm);
        mod *= (0.5 + 0.5 * edge_env) * inner_boost;
    }
    else
    {
        const bh_real envelope = 4.0 * r_norm * (1.0 - r_norm);
        mod *= (0.4 + 0.6 * envelope);
    }

    // Flat mode: wider dynamic range for deeper contrast
    if (pp.disk_flat_mode)
    {
        const bh_real factor = 0.20 + 0.80 * fmax(mod, 0.0);
        return dclamp(factor, 0.01, 5.0);
    }

    const bh_real factor = 0.35 + 0.65 * fmax(mod, 0.0);
    return dclamp(factor, 0.02, 3.0);
}

// Gas density: power-law radial, Gaussian vertical
BH_FUNC inline bh_real disk_density(const dvec3 &pos, bh_real r_ks,
                                   bh_real warped_h, const PhysicsParams &pp)
{
    const bh_real inner_r = pp.disk_inner_r;
    const bh_real outer_r = pp.disk_outer_r;

    // In flat mode: extend material inside ISCO and further beyond outer edge
    const bh_real dens_inner = pp.disk_flat_mode ? inner_r * 0.5 : inner_r;
    const bh_real fade_limit = pp.disk_flat_mode ? outer_r * 1.5 : outer_r * 1.2;

    if (r_ks < dens_inner || r_ks > fade_limit)
        return 0.0;
    if (warped_h < 1e-12)
        return 0.0;

    const bh_real height = pos.y;
    const bh_real radial = pow(pp.disk_isco / r_ks, 1.5);
    const bh_real vertical = exp(-0.5 * (height * height) / (warped_h * warped_h));

    bh_real outer_fade = 1.0;
    if (r_ks > outer_r)
    {
        // In flat mode: wider outer fade so streaks trail off gradually
        const bh_real fw = pp.disk_flat_mode ? 0.2 * outer_r : 0.1 * outer_r;
        const bh_real d = r_ks - outer_r;
        outer_fade = exp(-(d * d) / (2.0 * fw * fw));
    }

    // Inner-edge plunging region: material inside ISCO falls inward with
    // rapidly decreasing density — cubic falloff keeps it dim
    bh_real inner_fade = 1.0;
    if (pp.disk_flat_mode && r_ks < inner_r)
    {
        const bh_real t = (r_ks - dens_inner) / (inner_r - dens_inner);
        inner_fade = t * t * t; // cubic ramp: drops off fast toward BH
    }

    // In flat mode: boost density significantly so the thin disk appears
    // more opaque and solid.  The vertical Gaussian is much sharper (thin
    // disk), so we compensate by increasing the coefficient.
    bh_real density_scale = 1.0;
    if (pp.disk_flat_mode)
        density_scale = 8.0;

    return density_scale * pp.disk_density0 * radial * vertical *
           disk_clump_factor(pos, r_ks, warped_h, pp) *
           disk_stipple_factor(pos, r_ks, warped_h, pp) *
           outer_fade * inner_fade;
}

// Temperature: simplified Novikov-Thorne profile
BH_FUNC inline bh_real disk_temperature(bh_real r_ks, const PhysicsParams &pp)
{
    const bh_real inner_r = pp.disk_inner_r;
    const bh_real isco = pp.disk_isco; // physical ISCO for temperature profile
    const bh_real outer_r = pp.disk_outer_r;

    // In flat mode: extend temperature profile inside ISCO and further past outer edge
    const bh_real temp_inner = pp.disk_flat_mode ? inner_r * 0.5 : inner_r;
    const bh_real fade_limit = pp.disk_flat_mode ? outer_r * 1.5 : outer_r * 1.2;
    if (r_ks < temp_inner || r_ks > fade_limit)
        return 0.0;

    // Inside ISCO (plunging region): temperature fades as material falls in.
    // Use a lower peak (0.6 instead of 1.0) and cube the ramp for faster falloff.
    if (pp.disk_flat_mode && r_ks < inner_r)
    {
        const bh_real t = (r_ks - temp_inner) / (inner_r - temp_inner);
        return 0.6 * t * t * t;
    }

    // Novikov-Thorne profile always references the physical ISCO
    const bh_real x_ratio = r_ks / isco;
    const bh_real factor = (1.0 / (x_ratio * x_ratio * x_ratio)) *
                          fmax(1.0 - sqrt(1.0 / x_ratio), 0.0);
    const bh_real peak_x = 49.0 / 36.0;
    const bh_real peak_val = (1.0 / (peak_x * peak_x * peak_x)) *
                            (1.0 - sqrt(1.0 / peak_x));
    const bh_real T4 = factor / peak_val;

    bh_real outer_fade = 1.0;
    if (r_ks > outer_r)
    {
        const bh_real fw = pp.disk_flat_mode ? 0.2 * outer_r : 0.1 * outer_r;
        const bh_real d = r_ks - outer_r;
        outer_fade = exp(-(d * d) / (2.0 * fw * fw));
    }

    bh_real T_base = fmax(pow(fmax(T4 * outer_fade, 0.0), 0.25), 0.0);
    return T_base;
}

// Azimuthal temperature perturbation for flat mode — creates hot/cool streaks
// that shift the blackbody color across the disk surface.
// Returns a multiplier on temperature (centered around 1.0).
BH_FUNC inline bh_real disk_temperature_perturbation(const dvec3 &pos, bh_real r_ks,
                                                    const PhysicsParams &pp)
{
    if (!pp.disk_flat_mode)
        return 1.0;

    const bh_real inner_r = pp.disk_inner_r;
    const bh_real outer_r = pp.disk_outer_r;
    const bh_real tex_inner = inner_r * 0.5;
    const bh_real tex_outer = outer_r * 1.4;
    if (r_ks < tex_inner || r_ks > tex_outer)
        return 1.0;

    const bh_real phi = atan2(pos.z, pos.x);
    const bh_real log_r = log(fmax(r_ks, 1e-6));
    const bh_real r_norm = dclamp((r_ks - tex_inner) / (tex_outer - tex_inner), 0.0, 1.0);

    const bh_real omega = 1.0 / (r_ks * sqrt(r_ks));
    const bh_real tp = pp.disk_time * omega;

    // LOD: fade high-frequency temperature terms
    const bh_real texel = compute_texel_size(pos, pp);

    auto streak = [](bh_real phase, bh_real sharpness) -> bh_real
    {
        const bh_real s = sin(phase);
        return pow(s * s, sharpness);
    };

    auto arc_window = [r_ks, texel](bh_real phi, bh_real log_r, bh_real seed, bh_real brevity) -> bh_real
    {
        return lod_arc_window(phi, log_r, seed, brevity, r_ks, texel);
    };

    // Large-scale concentric temperature arcs — subtle
    bh_real dT = 0.0;
    dT += 0.10 * sin(1.0 * (phi + tp) - 4.0 * log_r + 0.5) * arc_window(phi, log_r, 7.1, 0.4);
    dT += 0.08 * sin(1.0 * (phi + tp) - 6.0 * log_r + 2.1) * arc_window(phi, log_r, 8.3, 0.5);
    dT += 0.06 * sin(1.0 * (phi + tp) - 8.0 * log_r + 0.7) * arc_window(phi, log_r, 9.7, 0.6);

    // Medium-scale concentric temperature filaments
    dT += lod_fade_logr(12.0, r_ks, texel) * 0.10 * sin(2.0 * (phi + tp) - 12.0 * log_r + 1.3) * arc_window(phi, log_r, 10.3, 0.7);
    dT += lod_fade_logr(16.0, r_ks, texel) * 0.09 * sin(2.0 * (phi + tp) - 16.0 * log_r - 0.8) * arc_window(phi, log_r, 11.8, 0.8);
    dT += lod_fade_logr(22.0, r_ks, texel) * 0.08 * sin(3.0 * (phi + tp) - 22.0 * log_r + 3.7) * arc_window(phi, log_r, 12.7, 0.9);

    // Fine-scale: rich micro-turbulence temperature jitter (concentric)
    dT += lod_fade_logr(30.0, r_ks, texel) * 0.10 * sin(3.0 * (phi + tp) - 30.0 * log_r + 1.1) * arc_window(phi, log_r, 13.1, 1.0);
    dT += lod_fade_logr(40.0, r_ks, texel) * 0.09 * sin(4.0 * phi - 40.0 * log_r + 2.2) * arc_window(phi, log_r, 14.2, 1.1);
    dT += lod_fade_logr(50.0, r_ks, texel) * 0.08 * sin(5.0 * (phi + tp) - 50.0 * log_r + 3.5) * arc_window(phi, log_r, 15.5, 1.2);
    dT += lod_fade_logr(60.0, r_ks, texel) * 0.07 * sin(6.0 * phi - 60.0 * log_r + 0.9) * arc_window(phi, log_r, 16.9, 1.3);
    dT += lod_fade_logr(70.0, r_ks, texel) * 0.06 * sin(7.0 * (phi + tp) - 70.0 * log_r + 4.8) * arc_window(phi, log_r, 17.8, 1.4);
    dT += lod_fade_logr(80.0, r_ks, texel) * 0.05 * sin(8.0 * phi - 80.0 * log_r + 2.6) * arc_window(phi, log_r, 18.6, 1.5);
    // 2D micro-mottling (product terms — radial-dominant)
    dT += lod_fade_logr(40.0, r_ks, texel) * 0.08 * sin(5.0 * phi + 30.0 * log_r + 1.7 + tp * 2.0) *
          sin(7.0 * phi - 40.0 * log_r + 4.2 + tp * 1.5);
    dT += lod_fade_logr(60.0, r_ks, texel) * 0.06 * sin(8.0 * phi - 50.0 * log_r + 0.3 + tp * 3.0) *
          cos(10.0 * phi + 60.0 * log_r + 3.1 + tp * 2.5);

    // Edge-specific: short hot arc fragments near inner edge
    const bh_real inner_prox = exp(-5.0 * r_norm * r_norm);
    dT += lod_fade_streak(6.0, 30.0, r_ks, texel) * 0.12 * inner_prox * streak(1.0 * (phi + tp * 1.5) - 6.0 * log_r + 0.7, 30.0) * arc_window(phi, log_r, 19.7, 2.5);
    dT += lod_fade_streak(10.0, 40.0, r_ks, texel) * 0.10 * inner_prox * streak(1.0 * (phi + tp * 1.3) - 10.0 * log_r + 2.9, 40.0) * arc_window(phi, log_r, 20.9, 3.0);
    dT -= lod_fade_streak(14.0, 50.0, r_ks, texel) * 0.08 * inner_prox * streak(2.0 * (phi + tp * 1.1) - 14.0 * log_r + 4.4, 50.0) * arc_window(phi, log_r, 21.4, 3.5);

    // Outer edge: short cool concentric trailing arc fragments
    const bh_real outer_prox = exp(-5.0 * (r_norm - 1.0) * (r_norm - 1.0));
    dT -= lod_fade_streak(4.0, 1.0, r_ks, texel) * 0.10 * outer_prox * (0.5 + 0.5 * sin(1.0 * (phi + tp) - 4.0 * log_r + 1.5)) * arc_window(phi, log_r, 22.5, 2.0);
    dT += lod_fade_streak(7.0, 50.0, r_ks, texel) * 0.12 * outer_prox * streak(1.0 * (phi + tp) - 7.0 * log_r + 3.8, 50.0) * arc_window(phi, log_r, 23.8, 2.5);

    // Clamp perturbation: temperature swings ±25%
    return dclamp(1.0 + dT, 0.75, 1.25);
}

// Blackbody temperature to RGB (5-zone piecewise)
BH_FUNC inline dvec3 temperature_to_rgb(bh_real T)
{
    T = dclamp(T, 0.0, 2.0);
    bh_real r, g, b;

    if (T < 0.15)
    {
        r = dclamp(T / 0.15 * 0.4, 0.0, 0.4);
        g = 0.0;
        b = 0.0;
    }
    else if (T < 0.4)
    {
        bh_real t = (T - 0.15) / 0.25;
        r = 0.4 + 0.6 * t;
        g = 0.25 * t * t;
        b = 0.0;
    }
    else if (T < 0.7)
    {
        bh_real t = (T - 0.4) / 0.3;
        r = 1.0;
        g = 0.25 + 0.55 * t;
        b = 0.05 * t;
    }
    else if (T < 1.0)
    {
        bh_real t = (T - 0.7) / 0.3;
        r = 1.0;
        g = 0.8 + 0.2 * t;
        b = 0.05 + 0.95 * t;
    }
    else
    {
        bh_real t = fmin((T - 1.0) / 1.0, 1.0);
        r = 1.0 - 0.15 * t;
        g = 1.0 - 0.05 * t;
        b = 1.0;
    }

    return dvec3(r, g, b);
}

// Emissivity + optional absorption output
BH_FUNC inline dvec3 disk_emissivity(const dvec3 &pos, bh_real r_ks,
                                     bh_real warped_h, bh_real *alpha_out,
                                     const PhysicsParams &pp)
{
    const bh_real rho = disk_density(pos, r_ks, warped_h, pp);
    if (alpha_out)
        *alpha_out = pp.disk_opacity0 * rho;
    if (rho < 1e-12)
        return dvec3(0, 0, 0);

    // Apply azimuthal temperature perturbation (flat mode: hot/cool streaks)
    const bh_real T_perturb = disk_temperature_perturbation(pos, r_ks, pp);
    const bh_real T = disk_temperature(r_ks, pp) * T_perturb;
    if (T < 1e-12)
        return dvec3(0, 0, 0);

    const bh_real T4 = T * T * T * T;
    dvec3 color = temperature_to_rgb(T);

    // Cinematic color variation
    const bh_real cv = pp.disk_color_variation;
    if (cv > 1e-6)
    {
        const bh_real inner_r = pp.disk_inner_r;
        const bh_real outer_r = pp.disk_outer_r;
        const bh_real phi = atan2(pos.z, pos.x);
        const bh_real log_r = log(fmax(r_ks, 1e-6));
        const bh_real r_norm = dclamp((r_ks - inner_r) / (outer_r - inner_r), 0.0, 1.0);

        const bh_real hue = 0.6 * sin(1.0 * phi - 4.0 * log_r + 0.5) +
                           0.4 * sin(1.0 * phi - 6.0 * log_r + 2.1) +
                           0.3 * cos(1.0 * phi - 8.0 * log_r + 1.7);
        const bh_real hue2 = 0.5 * sin(1.0 * phi - 5.0 * log_r + 3.3) +
                            0.3 * cos(1.0 * phi - 9.0 * log_r + 0.9);

        const bh_real radial_hue = 1.0 - 2.0 * r_norm;
        const bh_real mat = 0.5 + 0.5 * hue2;

        bh_real dr = 0.0, dg = 0.0, db = 0.0;

        dr += cv * (0.20 * (-radial_hue) + 0.15 * hue);
        dg += cv * (0.10 * hue * radial_hue);
        db += cv * (0.25 * radial_hue + 0.12 * (-hue));

        dr += cv * 0.18 * (1.0 - mat) * (0.5 + 0.5 * hue);
        db += cv * 0.12 * (1.0 - mat) * (0.5 + 0.5 * hue);

        const bh_real ice = exp(-8.0 * (mat - 0.5) * (mat - 0.5));
        dg += cv * 0.20 * ice;
        db += cv * 0.25 * ice;
        dr -= cv * 0.10 * ice;

        dr += cv * 0.15 * mat * (0.5 - 0.5 * hue);
        dg += cv * 0.06 * mat;
        db -= cv * 0.12 * mat;

        // ---- Flat mode: strong edge-specific color streaks ----
        if (pp.disk_flat_mode)
        {
            const bh_real omega = 1.0 / (r_ks * sqrt(r_ks));
            const bh_real tp = pp.disk_time * omega;

            auto streak_fn = [](bh_real phase, bh_real sharpness) -> bh_real
            {
                const bh_real s = sin(phase);
                return pow(s * s, sharpness);
            };

            // LOD: fade high-frequency color terms
            const bh_real texel = compute_texel_size(pos, pp);

            auto arc_win = [r_ks, texel](bh_real phi, bh_real log_r, bh_real seed, bh_real brevity) -> bh_real
            {
                return lod_arc_window(phi, log_r, seed, brevity, r_ks, texel);
            };

            // Extended radial range for edge color
            const bh_real tex_inner = inner_r * 0.5;
            const bh_real tex_outer = outer_r * 1.4;
            const bh_real r_ext = dclamp((r_ks - tex_inner) / (tex_outer - tex_inner), 0.0, 1.0);

            // Subtle large-scale concentric hue variation (arc-windowed)
            const bh_real hue3 = sin(1.0 * (phi + tp) - 6.0 * log_r + 1.7) * arc_win(phi, log_r, 30.1, 0.5);
            const bh_real hue4 = sin(1.0 * (phi + tp) - 10.0 * log_r + 3.2) * arc_win(phi, log_r, 31.2, 0.6);

            dr += cv * 0.12 * hue3;
            dg += cv * 0.08 * hue4;

            // Rich micro-turbulence color jitter — arc-windowed (LOD-faded)
            dr += lod_fade_logr(16.0, r_ks, texel) * cv * 0.10 * sin(2.0 * (phi + tp) - 16.0 * log_r + 0.4) * arc_win(phi, log_r, 32.4, 0.8);
            dg += lod_fade_logr(22.0, r_ks, texel) * cv * 0.09 * sin(3.0 * (phi + tp) - 22.0 * log_r + 5.1) * arc_win(phi, log_r, 33.1, 0.9);
            dr += lod_fade_logr(28.0, r_ks, texel) * cv * 0.08 * sin(3.0 * (phi + tp) - 28.0 * log_r + 2.8) * arc_win(phi, log_r, 34.8, 1.0);
            dg += lod_fade_logr(34.0, r_ks, texel) * cv * 0.08 * sin(4.0 * (phi + tp) - 34.0 * log_r + 1.3) * arc_win(phi, log_r, 35.3, 1.0);
            db += lod_fade_logr(18.0, r_ks, texel) * cv * 0.06 * sin(2.0 * (phi + tp) - 18.0 * log_r + 4.3) * arc_win(phi, log_r, 36.3, 1.1);
            dr -= lod_fade_logr(40.0, r_ks, texel) * cv * 0.07 * sin(4.0 * (phi + tp) - 40.0 * log_r + 3.9) * arc_win(phi, log_r, 37.9, 1.1);
            dg += lod_fade_logr(50.0, r_ks, texel) * cv * 0.07 * sin(5.0 * (phi + tp) - 50.0 * log_r + 0.7) * arc_win(phi, log_r, 38.7, 1.2);
            db += lod_fade_logr(55.0, r_ks, texel) * cv * 0.05 * sin(6.0 * (phi + tp) - 55.0 * log_r + 2.4) * arc_win(phi, log_r, 39.4, 1.3);
            dr += lod_fade_logr(60.0, r_ks, texel) * cv * 0.06 * sin(6.0 * phi - 60.0 * log_r + 5.5) * arc_win(phi, log_r, 40.5, 1.3);
            dg -= lod_fade_logr(70.0, r_ks, texel) * cv * 0.06 * sin(7.0 * phi - 70.0 * log_r + 1.1) * arc_win(phi, log_r, 41.1, 1.4);
            db += lod_fade_logr(80.0, r_ks, texel) * cv * 0.05 * sin(8.0 * phi - 80.0 * log_r + 3.2) * arc_win(phi, log_r, 42.2, 1.4);
            dr += lod_fade_logr(90.0, r_ks, texel) * cv * 0.05 * sin(9.0 * phi - 90.0 * log_r + 0.8) * arc_win(phi, log_r, 43.8, 1.5);

            // 2D micro-mottling: product terms — radial-dominant (LOD-faded)
            dr += lod_fade_logr(40.0, r_ks, texel) * cv * 0.08 * sin(5.0 * phi + 30.0 * log_r + 1.7 + tp * 2.0) *
                  sin(6.0 * phi - 40.0 * log_r + 4.2 + tp * 1.5);
            dg += lod_fade_logr(45.0, r_ks, texel) * cv * 0.07 * sin(5.0 * phi - 35.0 * log_r + 0.3 + tp * 3.0) *
                  cos(7.0 * phi + 45.0 * log_r + 3.1 + tp * 2.5);
            db += lod_fade_logr(60.0, r_ks, texel) * cv * 0.05 * sin(8.0 * phi + 50.0 * log_r + 2.9 + tp * 1.8) *
                  sin(9.0 * phi - 60.0 * log_r + 5.3 + tp * 2.2);

            // Inner-edge color: short arc fragments (high brevity)
            const bh_real inner_prox = exp(-5.0 * r_ext * r_ext);
            const bh_real inner_fade = lod_fade_streak(6.0, 30.0, r_ks, texel);
            const bh_real inner_streak = streak_fn(1.0 * (phi + tp * 1.5) - 6.0 * log_r + 0.7, 30.0) * arc_win(phi, log_r, 44.7, 2.5);
            dg += inner_fade * cv * 0.18 * inner_prox * inner_streak;
            db += inner_fade * cv * 0.08 * inner_prox * inner_streak;

            // Outer-edge color: short arc fragments toward deeper red
            const bh_real outer_prox = exp(-5.0 * (r_ext - 1.0) * (r_ext - 1.0));
            const bh_real outer_fade = lod_fade_streak(4.0, 30.0, r_ks, texel);
            const bh_real outer_streak = streak_fn(1.0 * (phi + tp) - 4.0 * log_r + 1.5, 30.0) * arc_win(phi, log_r, 45.5, 2.0);
            dr += outer_fade * cv * 0.15 * outer_prox * outer_streak;
            dg -= outer_fade * cv * 0.10 * outer_prox * (1.0 - outer_streak);

            // Subtle hot-spot accents (medium arc length)
            const bh_real knot_fade = lod_fade_streak(4.0, 60.0, r_ks, texel);
            const bh_real knot = streak_fn(1.0 * (phi + tp) - 4.0 * log_r + 2.2, 60.0) * arc_win(phi, log_r, 46.2, 1.5);
            const bh_real mid_prox = exp(-6.0 * (r_ext - 0.3) * (r_ext - 0.3));
            dg += knot_fade * cv * 0.15 * mid_prox * knot;
            db += knot_fade * cv * 0.07 * mid_prox * knot;
        }

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
BH_FUNC inline dvec4 disk_gas_four_velocity(const dvec3 &pos, bh_real r_ks,
                                            bh_real H, bh_real lx, bh_real ly, bh_real lz,
                                            const PhysicsParams &pp)
{
    const bh_real inner_r = pp.disk_inner_r;
    const bh_real a = pp.bh_spin;
    const bh_real M = pp.bh_mass;
    const bh_real sqrtM = sqrt(M);

    const bh_real r_cyl = sqrt(pos.x * pos.x + pos.z * pos.z);
    if (r_cyl < 1e-12)
        return dvec4(-1, 0, 0, 0);

    const bh_real cos_phi = pos.x / r_cyl;
    const bh_real sin_phi = pos.z / r_cyl;

    // Inside ISCO (plunging region): gas falls inward with increasing
    // radial velocity while conserving angular momentum from the ISCO.
    // This produces a strong redshift that dims the inner emission.
    if (pp.disk_flat_mode && r_ks < inner_r)
    {
        // Angular velocity: conserve ISCO angular momentum L = Omega_isco * r_isco^2
        const bh_real Omega_isco = sqrtM / (inner_r * sqrt(inner_r) + a * sqrtM);
        const bh_real L_isco = Omega_isco * inner_r * inner_r;
        const bh_real Omega_plunge = L_isco / (r_ks * r_ks);

        // Radial infall velocity: approximate free-fall from ISCO
        // v_r ~ -sqrt(2M * (1/r - 1/r_isco)) (Newtonian approximation)
        const bh_real v_r_mag = sqrt(fmax(2.0 * M * (1.0 / r_ks - 1.0 / inner_r), 0.0));

        // Spatial velocity in Cartesian coords (inward radial + orbital)
        const bh_real vx = -cos_phi * v_r_mag - sin_phi * Omega_plunge * r_cyl;
        const bh_real vy = 0.0;
        const bh_real vz = -sin_phi * v_r_mag + cos_phi * Omega_plunge * r_cyl;

        // Solve for u^0
        const bh_real twoH = 2.0 * H;
        const bh_real g00 = -1.0 + twoH;
        const bh_real lv = lx * vx + ly * vy + lz * vz;
        const bh_real b = 2.0 * twoH * lv;
        const bh_real v_sq = vx * vx + vy * vy + vz * vz;
        const bh_real c = v_sq + twoH * lv * lv + 1.0;
        const bh_real disc = b * b - 4.0 * g00 * c;
        const bh_real sqrt_disc = sqrt(fmax(disc, 0.0));
        const bh_real inv_2g00 = 0.5 / g00;
        const bh_real u0a = (-b + sqrt_disc) * inv_2g00;
        const bh_real u0b = (-b - sqrt_disc) * inv_2g00;
        const bh_real u0 = (u0a < 0.0) ? u0a : u0b;

        return dvec4(u0, vx, vy, vz);
    }

    if (r_ks < inner_r)
        return dvec4(-1, 0, 0, 0);

    const bh_real Omega = sqrtM / (r_ks * sqrt(r_ks) + a * sqrtM);
    const bh_real vx = -sin_phi * Omega * r_cyl;
    const bh_real vy = 0.0;
    const bh_real vz = cos_phi * Omega * r_cyl;

    // Solve for u^0 of timelike 4-velocity: g_μν u^μ u^ν = -1
    const bh_real twoH = 2.0 * H;
    const bh_real g00 = -1.0 + twoH;
    const bh_real lv = lx * vx + ly * vy + lz * vz;
    const bh_real b = 2.0 * twoH * lv;
    const bh_real v_sq = vx * vx + vy * vy + vz * vz;
    const bh_real c = v_sq + twoH * lv * lv + 1.0; // +1 for timelike

    const bh_real disc = b * b - 4.0 * g00 * c;
    const bh_real sqrt_disc = sqrt(fmax(disc, 0.0));
    const bh_real inv_2g00 = 0.5 / g00;
    const bh_real u0a = (-b + sqrt_disc) * inv_2g00;
    const bh_real u0b = (-b - sqrt_disc) * inv_2g00;
    const bh_real u0 = (u0a < 0.0) ? u0a : u0b;

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
                             bh_real sub_x, bh_real sub_y,
                             bh_real fov_x, bh_real fov_y)
{
    pos = cam_pos;
    const bh_real x_rad = ((sub_x - 0.5) * fov_x) * (M_PI / 180.0);
    const bh_real y_rad = ((0.5 - sub_y) * fov_y) * (M_PI / 180.0);

    const bh_real cy = cos(y_rad), cx = cos(x_rad);
    const bh_real local_fwd = cy * cx;
    const bh_real local_up = sin(y_rad);
    const bh_real local_right = cy * sin(x_rad);

    vel = local_fwd * cam_fwd + local_up * cam_up + local_right * cam_right;
}

// RK2 (far-field) / RK4 (near-field) geodesic integrator
BH_FUNC inline bool advance_ray(dvec3 &pos, dvec3 &vel, bh_real &cached_r,
                                bh_real dt, const PhysicsParams &pp)
{
    const bh_real r_plus = pp.r_plus;

    if (cached_r > 15.0 * r_plus)
    {
        // Far-field: RK2 midpoint (2 evaluations)
        bh_real r1;
        dvec3 a1 = geodesic_accel(pos, vel, pp, &r1);
        if (r1 <= r_plus)
            return false;

        dvec3 mid_pos = pos + 0.5 * dt * vel;
        dvec3 mid_vel = vel + 0.5 * dt * a1;

        bh_real r2;
        dvec3 a2 = geodesic_accel(mid_pos, mid_vel, pp, &r2);
        if (r2 <= r_plus)
            return false;

        pos = pos + dt * mid_vel;
        vel = vel + dt * a2;
    }
    else
    {
        // Near-field: full RK4 (4 evaluations)
        bh_real rk;

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

        bh_real s = dt / 6.0;
        pos = pos + s * (k1_dx + 2.0 * k2_dx + 2.0 * k3_dx + k4_dx);
        vel = vel + s * (k1_dv + 2.0 * k2_dv + 2.0 * k3_dv + k4_dv);
    }

    cached_r = ks_radius(pos, pp.bh_spin);
    return true;
}

// Volumetric disk sampling with relativistic radiative transfer
// r_ks is the precomputed Kerr-Schild radius (avoids redundant ks_radius call)
BH_FUNC inline void sample_disk_volume(const dvec3 &pos, const dvec3 &vel, bh_real ds,
                                       dvec3 &acc_color, bh_real &acc_opacity,
                                       bh_real r_ks, const PhysicsParams &pp)
{
    if (acc_opacity >= 0.999)
        return;

    // Contains check — wider bounds in flat mode for edge streaks
    const bh_real samp_inner = pp.disk_flat_mode ? pp.disk_inner_r * 0.5 : pp.disk_inner_r;
    const bh_real fade_limit = pp.disk_flat_mode ? pp.disk_outer_r * 1.5 : pp.disk_outer_r * 1.2;
    if (r_ks < samp_inner || r_ks > fade_limit)
        return;

    const bh_real warped_h = disk_warped_half_thickness(pos, r_ks, pp);
    if (fabs(pos.y) > 3.0 * warped_h)
        return;

    // Emissivity + absorption
    bh_real alpha;
    dvec3 j = disk_emissivity(pos, r_ks, warped_h, &alpha, pp);

    // KS metric intermediates (computed once, shared)
    const bh_real a = pp.bh_spin, a2 = a * a;
    const bh_real M = pp.bh_mass;
    const bh_real r2 = r_ks * r_ks;
    const bh_real Pk = r2 + a2;
    const bh_real inv_Pk = 1.0 / Pk;
    const bh_real lx = (r_ks * pos.x + a * pos.z) * inv_Pk;
    const bh_real ly = pos.y / r_ks;
    const bh_real lz = (r_ks * pos.z - a * pos.x) * inv_Pk;
    const bh_real H = (M * r_ks * r2) / (r2 * r2 + a2 * pos.y * pos.y);
    const bh_real twoH = 2.0 * H;

    // u^0 from null condition (photon)
    const bh_real v1 = vel.x, v2 = vel.y, v3 = vel.z;
    const bh_real lv_phot = lx * v1 + ly * v2 + lz * v3;
    const bh_real g00 = -1.0 + twoH;
    const bh_real b_u0 = 2.0 * twoH * lv_phot;
    const bh_real v_sq = v1 * v1 + v2 * v2 + v3 * v3;
    const bh_real c_u0 = v_sq + twoH * lv_phot * lv_phot;

    const bh_real disc_u0 = b_u0 * b_u0 - 4.0 * g00 * c_u0;
    const bh_real sqrt_disc = sqrt(fmax(disc_u0, 0.0));
    const bh_real inv_2g00 = 0.5 / g00;
    const bh_real u0a = (-b_u0 + sqrt_disc) * inv_2g00;
    const bh_real u0b = (-b_u0 - sqrt_disc) * inv_2g00;
    const bh_real u0 = (u0a < 0.0) ? u0a : u0b;

    // Gas 4-velocity (shares H, l from metric)
    dvec4 gas_u = disk_gas_four_velocity(pos, r_ks, H, lx, ly, lz, pp);

    // Redshift via g = -k_lower[0] / (k_lower · gas_u)
    // k_lower = η_μν k^ν + 2H l_μ (l · k)   [KS form]
    const bh_real l_dot_k = u0 + lx * v1 + ly * v2 + lz * v3;
    const bh_real k_lower0 = -u0 + twoH * l_dot_k;
    const bh_real k_lower1 = v1 + twoH * lx * l_dot_k;
    const bh_real k_lower2 = v2 + twoH * ly * l_dot_k;
    const bh_real k_lower3 = v3 + twoH * lz * l_dot_k;

    bh_real k_dot_u = k_lower0 * gas_u.t + k_lower1 * gas_u.x +
                     k_lower2 * gas_u.y + k_lower3 * gas_u.z;
    if (fabs(k_dot_u) < 1e-15)
        k_dot_u = 1e-15;

    const bh_real g_red = -k_lower0 / k_dot_u;
    const bh_real g_clamped = dclamp(g_red, 0.01, 10.0);
    const bh_real g3 = g_clamped * g_clamped * g_clamped;

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
    const bh_real transmittance = 1.0 - acc_opacity;
    const bh_real dtau = alpha * ds;
    const bh_real absorption_factor = 1.0 - exp(-dtau);
    const bh_real inv_alpha = 1.0 / fmax(alpha, 1e-12);

    acc_color += transmittance * absorption_factor * inv_alpha * j_obs;
    acc_opacity += transmittance * absorption_factor;
    acc_opacity = fmin(acc_opacity, 1.0);
}

// ============================================================================
// Convenience: construct PhysicsParams from scene parameters
// ============================================================================

inline PhysicsParams make_physics_params(bh_real M, bh_real a,
                                         bh_real disk_outer_r, bh_real disk_thickness,
                                         bh_real disk_density, bh_real disk_opacity,
                                         bh_real emission_boost, bh_real color_variation,
                                         bh_real turbulence, bh_real time,
                                         int flat_mode = 0,
                                         bh_real disk_inner_override = -1.0,
                                         bh_real stipple = 0.0)
{
    PhysicsParams pp = {};
    pp.bh_mass = M;
    pp.bh_spin = a;
    pp.r_plus = compute_event_horizon(M, a);
    pp.disk_isco = compute_isco(M, a);
    pp.disk_inner_r = (disk_inner_override > 0) ? disk_inner_override : pp.disk_isco;
    pp.disk_outer_r = disk_outer_r;
    pp.disk_thickness = disk_thickness;
    pp.disk_r_ref = sqrt(pp.disk_isco * disk_outer_r);
    pp.disk_density0 = disk_density;
    pp.disk_opacity0 = disk_opacity;
    pp.disk_emission_boost = emission_boost;
    pp.disk_color_variation = color_variation;
    pp.disk_turbulence = turbulence;
    pp.disk_time = time;
    pp.disk_stipple = stipple;
    pp.disk_flat_mode = flat_mode;
    return pp;
}

#endif // _BH_PHYSICS_H
