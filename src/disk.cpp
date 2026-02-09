
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
// Turbulence-warped half-thickness: azimuthal lumps, gaps, and warps
// Creates a debris-ring look where some directions are thick/puffy and
// others are nearly absent — like a planet was ripped apart.
// ---------------------------------------------------------------------------
double accretion_disk_s::warped_half_thickness(const Vector3d &pos) const
{
    const double r_ks = bh->ks_radius(pos);
    const double h0 = half_thickness(std::min(r_ks, outer_r));

    if (turbulence < 1e-6)
        return h0;

    const double x = pos.x(), z = pos.z();
    const double phi = atan2(z, x);
    const double log_r = log(std::max(r_ks, 1e-6));
    const double t = turbulence;

    // Keplerian time offset so thickness warps co-rotate with clump pattern
    const double omega = 1.0 / (r_ks * sqrt(r_ks));
    const double tp = time * omega;

    // Large-scale azimuthal lumps (2-3 dominant thick/thin regions)
    double warp = 0.0;
    warp += 0.55 * sin(2.0 * (phi + tp) - 3.0 * log_r + 1.2);
    warp += 0.35 * sin(3.0 * (phi + tp) - 5.0 * log_r + 4.1);

    // Medium-scale: turbulent puffiness variations
    warp += 0.25 * sin(5.0 * (phi + tp) - 9.0 * log_r + 0.7);
    warp += 0.18 * sin(7.0 * (phi + tp) - 11.0 * log_r + 2.9);

    // Fine-scale: small billows and wisps
    warp += 0.12 * sin(11.0 * (phi + tp) - 17.0 * log_r + 5.3);
    warp += 0.08 * sin(17.0 * (phi + tp) - 23.0 * log_r + 3.1);

    // Narrow gaps / tears in the disk (sharp dips)
    auto dip = [](double phase, double sharpness) -> double
    {
        double s = sin(phase);
        return pow(s * s, sharpness);
    };
    warp -= 0.70 * dip(3.0 * (phi + tp) - 4.0 * log_r + 2.5, 4.0);
    warp -= 0.40 * dip(5.0 * (phi + tp) - 8.0 * log_r + 0.3, 5.0);

    // Apply: factor range from near-zero (gap) to ~2x (puffy clump)
    // At turbulence=1, full effect; at turbulence=0.5, half effect
    const double factor = 1.0 + t * warp;
    return h0 * std::max(factor, 0.05);
}

// ---------------------------------------------------------------------------
// Is this point inside the disk volume?
// ---------------------------------------------------------------------------
bool accretion_disk_s::contains(const Vector3d &pos) const
{
    const double r_ks = bh->ks_radius(pos);
    // Allow a fade zone extending 20% past outer_r for smooth falloff
    const double fade_limit = outer_r * 1.2;
    if (r_ks < inner_r || r_ks > fade_limit)
        return false;

    const double h = warped_half_thickness(pos);
    const double height = fabs(pos.y());
    return height <= 3.0 * h;
}

// ---------------------------------------------------------------------------
// Procedural clump / streak modulation
// Multi-scale trailing logarithmic spirals simulate turbulent clumps that
// have been sheared into elongated streaks by differential rotation, giving
// an Interstellar-style inhomogeneous disk appearance.
// ---------------------------------------------------------------------------
double accretion_disk_s::clump_factor(const Vector3d &pos) const
{
    const double r_ks = bh->ks_radius(pos);
    if (r_ks < inner_r || r_ks > outer_r)
        return 1.0;

    const double x = pos.x(), z = pos.z();
    const double y = pos.y();
    const double phi = atan2(z, x);
    const double log_r = log(r_ks);
    const double r_norm = (r_ks - inner_r) / (outer_r - inner_r);

    // --- Time-dependent rotation -----------------------------------------
    // Each radial shell rotates at a Keplerian angular velocity ∝ r^{-3/2}.
    // The time parameter shifts the azimuthal phase so that sequential
    // frames show differential rotation (inner material leads outer).
    const double omega = 1.0 / (r_ks * sqrt(r_ks)); // ~ Keplerian
    const double t_phase = time * omega;

    // --- Height-dependent phase offset -----------------------------------
    // Without this, the clump pattern is constant along vertical columns,
    // producing vertical striping when viewed edge-on.  Adding a phase
    // shift proportional to height makes the pattern tilt so that streaks
    // are coherent *along the orbital plane* (horizontal from the side).
    // Physically: turbulent eddies at different heights are at different
    // azimuthal phases due to vertical shear.
    const double h = warped_half_thickness(pos);
    const double y_over_h = (h > 1e-12) ? (y / h) : 0.0;
    // Large tilt so the pattern is clearly horizontal, not vertical
    const double y_phase = 3.0 * y_over_h;

    // --- Helper: sharp streak function -----------------------------------
    // Raises sin^2 to a power to create narrow bright streaks on a darker
    // background, rather than gentle sinusoidal undulations.
    auto streak = [](double phase, double sharpness) -> double
    {
        const double s = sin(phase);
        return pow(s * s, sharpness); // 0..1, narrow peaks
    };

    double mod = 0.0;

    // === LARGE-SCALE: 2 dominant trailing spiral arms ====================
    // Use a sharpened streak so these read as distinct bright arcs
    mod += 1.4 * streak(2.0 * (phi + t_phase) - 4.5 * log_r + 0.5 + y_phase, 3.0);

    // === SECONDARY: 3-arm spiral, offset phase ===========================
    mod += 0.7 * streak(3.0 * (phi + t_phase) - 7.0 * log_r + 2.1 + 1.5 * y_phase, 2.5);

    // === MEDIUM: sheared turbulent clumps → azimuthal streaks =============
    mod += 0.45 * streak(7.0 * (phi + t_phase) - 13.0 * log_r + 1.3 + 2.0 * y_phase, 2.0);
    mod += 0.35 * streak(11.0 * (phi + t_phase) - 19.0 * log_r - 0.8 + 2.5 * y_phase, 2.0);

    // === FINE: high-frequency turbulent streaks ===========================
    mod += 0.25 * sin(17.0 * (phi + t_phase) - 25.0 * log_r + 3.7 + 3.0 * y_phase);
    mod += 0.15 * sin(23.0 * (phi + t_phase) - 33.0 * log_r + 0.4 + 4.0 * y_phase);
    mod += 0.10 * sin(31.0 * (phi + t_phase) - 45.0 * log_r + 5.2 + 5.0 * y_phase);

    // === RADIAL hot-spot rings × spiral ==================================
    mod += 0.40 * streak(5.0 * r_ks + 1.0 + 0.5 * y_phase, 2.0) *
           cos(3.0 * (phi + t_phase) - 5.0 * log_r + y_phase);

    // === BRIGHT KNOTS: cross-term localised clumps =======================
    mod += 0.35 * streak(4.0 * M_PI * r_norm, 2.0) *
           streak(9.0 * (phi + t_phase) - 14.0 * log_r + 1.9 + 2.0 * y_phase, 2.0);

    // === DARK LANES: subtract narrow dark gaps between bright streaks =====
    mod -= 0.50 * streak(5.0 * (phi + t_phase) - 8.0 * log_r + 4.0 + 1.8 * y_phase, 4.0);
    mod -= 0.30 * streak(8.0 * (phi + t_phase) - 15.0 * log_r + 0.7 + 2.2 * y_phase, 5.0);

    // === RADIAL BRIGHTNESS ENVELOPE: fade inner/outer edges ==============
    // Boost mid-disk contrast; let edges be calmer
    const double envelope = 4.0 * r_norm * (1.0 - r_norm); // parabola 0..1..0
    mod *= (0.4 + 0.6 * envelope);

    // Shift baseline down so the "background" disk is dimmer and streaks
    // punch up from it, giving much more obvious contrast
    const double factor = 0.35 + 0.65 * std::max(mod, 0.0);

    // Clamp so we never go fully black (some residual glow)
    return std::clamp(factor, 0.02, 3.0);
}

// ---------------------------------------------------------------------------
// Gas density: power-law radial profile, Gaussian vertical profile
// ---------------------------------------------------------------------------
double accretion_disk_s::density(const Vector3d &pos) const
{
    const double r_ks = bh->ks_radius(pos);
    const double fade_limit = outer_r * 1.2;
    if (r_ks < inner_r || r_ks > fade_limit)
        return 0.0;

    const double h = warped_half_thickness(pos);
    if (h < 1e-12)
        return 0.0;

    const double height = pos.y();
    const double radial = pow(inner_r / r_ks, 1.5);
    const double vertical = exp(-0.5 * (height * height) / (h * h));

    // Smooth exponential taper beyond outer_r
    double outer_fade = 1.0;
    if (r_ks > outer_r)
    {
        const double fade_width = 0.1 * outer_r; // fade over ~10% of outer_r
        outer_fade = exp(-(r_ks - outer_r) * (r_ks - outer_r) / (2.0 * fade_width * fade_width));
    }

    return density0 * radial * vertical * clump_factor(pos) * outer_fade;
}

// ---------------------------------------------------------------------------
// Temperature: simplified Novikov-Thorne profile
// ---------------------------------------------------------------------------
double accretion_disk_s::temperature(const Vector3d &pos) const
{
    const double r_ks = bh->ks_radius(pos);
    const double fade_limit = outer_r * 1.2;
    if (r_ks < inner_r || r_ks > fade_limit)
        return 0.0;

    const double x_ratio = r_ks / inner_r;

    const double factor = (1.0 / (x_ratio * x_ratio * x_ratio)) *
                          std::max(1.0 - sqrt(1.0 / x_ratio), 0.0);

    const double T_max = 1.0;
    const double peak_x = 49.0 / 36.0;
    const double peak_val = (1.0 / (peak_x * peak_x * peak_x)) *
                            (1.0 - sqrt(1.0 / peak_x));
    const double T4 = T_max * factor / peak_val;

    // Smooth fade beyond outer_r (matches density fade)
    double outer_fade = 1.0;
    if (r_ks > outer_r)
    {
        const double fade_width = 0.1 * outer_r;
        outer_fade = exp(-(r_ks - outer_r) * (r_ks - outer_r) / (2.0 * fade_width * fade_width));
    }

    return std::max(pow(std::max(T4 * outer_fade, 0.0), 0.25), 0.0);
}

// ---------------------------------------------------------------------------
// Blackbody temperature to RGB — enriched palette
// Low T: deep red/crimson → orange → gold → white-hot → blue-white at high T
// ---------------------------------------------------------------------------
Vector3d accretion_disk_s::temperature_to_rgb(double T)
{
    T = std::clamp(T, 0.0, 2.0);

    // Piecewise spline-ish mapping for a richer colour gradient
    double r, g, b;

    if (T < 0.15)
    {
        // Very cool: deep crimson / ember
        r = std::clamp(T / 0.15 * 0.4, 0.0, 0.4);
        g = 0.0;
        b = 0.0;
    }
    else if (T < 0.4)
    {
        // Warm: crimson → orange
        double t = (T - 0.15) / 0.25;
        r = 0.4 + 0.6 * t;
        g = 0.25 * t * t;
        b = 0.0;
    }
    else if (T < 0.7)
    {
        // Hot: orange → gold / yellow
        double t = (T - 0.4) / 0.3;
        r = 1.0;
        g = 0.25 + 0.55 * t;
        b = 0.05 * t;
    }
    else if (T < 1.0)
    {
        // Very hot: gold → white
        double t = (T - 0.7) / 0.3;
        r = 1.0;
        g = 0.8 + 0.2 * t;
        b = 0.05 + 0.95 * t;
    }
    else
    {
        // Superhot: white → blue-white
        double t = std::min((T - 1.0) / 1.0, 1.0);
        r = 1.0 - 0.15 * t;
        g = 1.0 - 0.05 * t;
        b = 1.0;
    }

    return Vector3d(r, g, b);
}

// ---------------------------------------------------------------------------
// Emissivity: thermal emission with optional cinematic color variation
//   color_variation = 0.0 → physically based (temperature color only)
//   color_variation = 1.0 → full cinematic (azimuthal & radial hue shifts)
// ---------------------------------------------------------------------------
Vector3d accretion_disk_s::emissivity(const Vector3d &pos, double *alpha_out) const
{
    const double rho = density(pos);
    if (alpha_out)
        *alpha_out = opacity0 * rho;
    if (rho < 1e-12)
        return Vector3d(0, 0, 0);

    const double T = temperature(pos);
    if (T < 1e-12)
        return Vector3d(0, 0, 0);

    const double T4 = T * T * T * T;
    Vector3d color = temperature_to_rgb(T);

    // --- Cinematic color tint (controlled by color_variation) -------------
    if (color_variation > 1e-6)
    {
        const double r_ks = bh->ks_radius(pos);
        const double x = pos.x(), z = pos.z();
        const double phi = atan2(z, x);
        const double log_r = log(std::max(r_ks, 1e-6));
        const double r_norm = std::clamp((r_ks - inner_r) / (outer_r - inner_r), 0.0, 1.0);
        const double cv = color_variation;

        // Hue angle: sweeps through colour wheel based on position
        // Spiral-following so colour bands align with the streak pattern
        const double hue = 0.6 * sin(2.0 * phi - 4.5 * log_r + 0.5) + 0.4 * sin(3.0 * phi - 7.0 * log_r + 2.1) + 0.3 * cos(5.0 * phi - 10.0 * log_r + 1.7);

        // Secondary hue axis — orthogonal variation for richer palette
        const double hue2 = 0.5 * sin(4.0 * phi - 6.0 * log_r + 3.3) + 0.3 * cos(7.0 * phi - 12.0 * log_r + 0.9);

        // Radial colour gradient: inner = blue-white boost, outer = deep amber
        const double radial_hue = 1.0 - 2.0 * r_norm; // +1 inner, -1 outer

        // Debris composition bands — some streaks are metal (blue-steel),
        // some are rocky (ochre/brown), some are icy (cyan), some are
        // molten (deep orange/magenta).  hue2 drives the "material type".
        const double mat = 0.5 + 0.5 * hue2; // 0..1 material selector

        double dr = 0.0, dg = 0.0, db = 0.0;

        // Base radial + spiral tint (same as before but stronger)
        dr += cv * (0.20 * (-radial_hue) + 0.15 * hue);
        dg += cv * (0.10 * hue * radial_hue);
        db += cv * (0.25 * radial_hue + 0.12 * (-hue));

        // Material-type color shifts
        // Molten/magenta streaks (mat near 0)
        dr += cv * 0.18 * (1.0 - mat) * (0.5 + 0.5 * hue);
        db += cv * 0.12 * (1.0 - mat) * (0.5 + 0.5 * hue);

        // Cyan/ice streaks (mat near 0.5)
        const double ice = exp(-8.0 * (mat - 0.5) * (mat - 0.5));
        dg += cv * 0.20 * ice;
        db += cv * 0.25 * ice;
        dr -= cv * 0.10 * ice;

        // Ochre/brown rocky debris (mat near 1)
        dr += cv * 0.15 * mat * (0.5 - 0.5 * hue);
        dg += cv * 0.06 * mat;
        db -= cv * 0.12 * mat;

        color.x() = std::clamp(color.x() + dr, 0.0, 1.0);
        color.y() = std::clamp(color.y() + dg, 0.0, 1.0);
        color.z() = std::clamp(color.z() + db, 0.0, 1.0);
    }

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
