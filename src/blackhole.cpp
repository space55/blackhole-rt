
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
//
// Fully self-contained: computes KS radius, metric components, partials,
// u^0, and the Christoffel contraction in a single pass with no redundant
// work.  Exploits the rank-2 structure of the Kerr-Schild metric partials
// and the algebraic form of g_inv to avoid forming any 4×4 matrices.
//
// Key identities used:
//   partials[i] = 2 dH_i (l⊗l) + 2H (dl_i⊗l + l⊗dl_i)   (rank-2)
//   ⇒ P_i = partials[i]·u = (2 dH_i L + 2H D_i) l + 2H L dl_i
//   ⇒ Q_i = u·P_i = 2 dH_i L² + 4H L D_i
//   where L = l·u, D_i = dl_i·u
//
//   g_inv = η - 2H l_up⊗l_up   (Kerr-Schild form)
//   ⇒ g_inv · F = F(spatial) - 2H l_up · (l_up·F)
// ---------------------------------------------------------------------------
#if defined(__GNUC__) || defined(__clang__)
__attribute__((hot))
#endif
Vector3d blackhole_s::geodesic_accel(const Vector3d &pos, const Vector3d &vel,
                                     double *out_ks_r) const
{
    const double x = pos.x(), y = pos.y(), z = pos.z();
    const double a = spin, a2 = a * a;

    // --- KS radius (computed once) ---------------------------------------
    const double rho2 = x * x + y * y + z * z;
    const double term = rho2 - a2;
    const double r2 = 0.5 * (term + sqrt(term * term + 4.0 * a2 * y * y));
    const double r = sqrt(std::max(r2, 1e-12));
    const double inv_r = 1.0 / r;
    const double inv_r2 = inv_r * inv_r;

    if (out_ks_r)
        *out_ks_r = r;

    // --- Null vector l, Sigma, H (shared) --------------------------------
    const double P = r2 + a2;
    const double inv_P = 1.0 / P;
    const double inv_P2 = inv_P * inv_P;
    const double lx = (r * x + a * z) * inv_P;
    const double ly = y * inv_r;
    const double lz = (r * z - a * x) * inv_P;

    const double Sigma = r2 + a2 * y * y * inv_r2;
    const double inv_Sigma = 1.0 / Sigma;
    const double H = mass * r * inv_Sigma;

    // --- Inline u^0 from null condition (no separate metric() call) ------
    const double twoH = 2.0 * H;
    const double g00 = -1.0 + twoH;

    const double v1 = vel.x(), v2 = vel.y(), v3 = vel.z();
    const double lv = lx * v1 + ly * v2 + lz * v3;
    const double b_u0 = 2.0 * twoH * lv;
    const double v_sq = v1 * v1 + v2 * v2 + v3 * v3;
    const double c_u0 = v_sq + twoH * lv * lv;

    const double disc_u0 = b_u0 * b_u0 - 4.0 * g00 * c_u0;
    const double sqrt_disc = sqrt(std::max(disc_u0, 0.0));
    const double inv_2g00 = 0.5 / g00;
    const double u0a = (-b_u0 + sqrt_disc) * inv_2g00;
    const double u0b = (-b_u0 - sqrt_disc) * inv_2g00;
    const double u0 = (u0a < 0.0) ? u0a : u0b;

    // --- 4-velocity u and key dot products with l ------------------------
    // u = (u0, v1, v2, v3),  l = (1, lx, ly, lz)
    const double L = u0 + lx * v1 + ly * v2 + lz * v3; // l · u

    // --- Partial derivative intermediates --------------------------------
    const double inv_Sigma2 = inv_Sigma * inv_Sigma;
    const double r_inv_Sigma = r * inv_Sigma;

    const double dr0 = x * r_inv_Sigma;
    const double dr1 = y * P * inv_r * inv_Sigma;
    const double dr2 = z * r_inv_Sigma;

    const double inv_r3 = inv_r * inv_r2;
    const double sig_r_coeff = 2.0 * r - 2.0 * a2 * y * y * inv_r3;
    const double dSigma0 = sig_r_coeff * dr0;
    const double dSigma1 = sig_r_coeff * dr1 + 2.0 * a2 * y * inv_r2;
    const double dSigma2 = sig_r_coeff * dr2;

    const double mass_inv_S2 = mass * inv_Sigma2;
    const double dH0 = mass_inv_S2 * (dr0 * Sigma - r * dSigma0);
    const double dH1 = mass_inv_S2 * (dr1 * Sigma - r * dSigma1);
    const double dH2 = mass_inv_S2 * (dr2 * Sigma - r * dSigma2);

    // dl_i vectors: dl_i = (0, dlx_i, dly_i, dlz_i)
    // Factored form: dl[i]_x = dr[i]*K_x + addend, saving ~23 multiplies
    const double rxaz = r * x + a * z;
    const double rzax = r * z - a * x;
    const double r_inv_P = r * inv_P;
    const double a_inv_P = a * inv_P;
    const double K_x = (x * P - 2.0 * r * rxaz) * inv_P2;
    const double K_y = -y * inv_r2;
    const double K_z = (z * P - 2.0 * r * rzax) * inv_P2;

    // i=0 (d/dx)
    const double dlx0 = dr0 * K_x + r_inv_P;
    const double dly0 = K_y * dr0;
    const double dlz0 = dr0 * K_z - a_inv_P;

    // i=1 (d/dy)
    const double dlx1 = dr1 * K_x;
    const double dly1 = K_y * dr1 + inv_r;
    const double dlz1 = dr1 * K_z;

    // i=2 (d/dz)
    const double dlx2 = dr2 * K_x + a_inv_P;
    const double dly2 = K_y * dr2;
    const double dlz2 = dr2 * K_z + r_inv_P;

    // D[i] = dl_i · u = dlx_i*v1 + dly_i*v2 + dlz_i*v3  (dl_i(0) = 0)
    const double D0 = dlx0 * v1 + dly0 * v2 + dlz0 * v3;
    const double D1 = dlx1 * v1 + dly1 * v2 + dlz1 * v3;
    const double D2 = dlx2 * v1 + dly2 * v2 + dlz2 * v3;

    // --- P_i = partials[i] * u  (without forming the 4×4 matrix) --------
    // P_i = (2*dH_i*L + 2*H*D_i) * l + 2*H*L * dl_i
    const double twoHL = twoH * L;
    const double c0 = 2.0 * dH0 * L + twoH * D0;
    const double c1 = 2.0 * dH1 * L + twoH * D1;
    const double c2 = 2.0 * dH2 * L + twoH * D2;

    // P_i components: P_i = c_i * l + twoHL * dl_i
    // P_i(0) = c_i * 1   + twoHL * 0      = c_i
    // P_i(j) = c_i * l_j + twoHL * dl_ij
    const double P0_0 = c0;
    const double P0_1 = c0 * lx + twoHL * dlx0;
    const double P0_2 = c0 * ly + twoHL * dly0;
    const double P0_3 = c0 * lz + twoHL * dlz0;

    const double P1_0 = c1;
    const double P1_1 = c1 * lx + twoHL * dlx1;
    const double P1_2 = c1 * ly + twoHL * dly1;
    const double P1_3 = c1 * lz + twoHL * dlz1;

    const double P2_0 = c2;
    const double P2_1 = c2 * lx + twoHL * dlx2;
    const double P2_2 = c2 * ly + twoHL * dly2;
    const double P2_3 = c2 * lz + twoHL * dlz2;

    // --- Q_i = u · P_i = 2*dH_i*L² + 4*H*L*D_i -------------------------
    const double L2 = L * L;
    const double fourHL = 2.0 * twoHL;
    const double Q0 = 2.0 * dH0 * L2 + fourHL * D0;
    const double Q1 = 2.0 * dH1 * L2 + fourHL * D1;
    const double Q2 = 2.0 * dH2 * L2 + fourHL * D2;

    // --- F(ν) = 2*T1(ν) - T3(ν) -----------------------------------------
    // T1(ν) = u(1)*P0(ν) + u(2)*P1(ν) + u(3)*P2(ν)
    // T3(ν) = Q[ν-1] for ν≥1, else 0
    const double F0 = 2.0 * (v1 * P0_0 + v2 * P1_0 + v3 * P2_0);
    const double F1 = 2.0 * (v1 * P0_1 + v2 * P1_1 + v3 * P2_1) - Q0;
    const double F2 = 2.0 * (v1 * P0_2 + v2 * P1_2 + v3 * P2_2) - Q1;
    const double F3 = 2.0 * (v1 * P0_3 + v2 * P1_3 + v3 * P2_3) - Q2;

    // --- a^μ = -½ g_inv^{μν} F_ν  using KS form -------------------------
    // g_inv = η - 2H (l_up ⊗ l_up),  l_up = (-1, lx, ly, lz)
    // ⇒ g_inv^{μν} F_ν = F_μ - 2H * l_up_μ * S
    //   where S = l_up · F = -F0 + lx*F1 + ly*F2 + lz*F3
    const double S = -F0 + lx * F1 + ly * F2 + lz * F3;
    const double twoHS = twoH * S;

    return Vector3d(
        -0.5 * (F1 - lx * twoHS),
        -0.5 * (F2 - ly * twoHS),
        -0.5 * (F3 - lz * twoHS));
}
