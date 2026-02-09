
#ifndef _BH_BLACKHOLE_H
#define _BH_BLACKHOLE_H

#include "common.h"

struct MetricResult
{
    Matrix4d g;
    Matrix4d g_inv;
};

struct blackhole_s
{
    double mass;
    double spin;

    // Precomputed derived quantities
    double r_event_horizon;
    double r_isco;

    blackhole_s(double M, double a);

    // Kerr-Schild radius from Cartesian position (spin axis = y)
    double ks_radius(const Vector3d &pos) const;

    // Event horizon radius: r+ = M + sqrt(M^2 - a^2)
    double event_horizon_radius() const;

    // ISCO radius (prograde)
    double isco_radius() const;

    // Full Kerr-Schild metric and inverse
    MetricResult metric(const Vector3d &pos) const;

    // Solve null condition g_mu_nu u^mu u^nu = 0 for u^0
    double compute_u0_null(const Vector3d &pos, const Vector3d &vel) const;

    // Analytical partial derivatives dg_ab/dx^i (i=1,2,3)
    void metric_partials(const Vector3d &pos, Matrix4d partials[3]) const;

    // Geodesic acceleration: d^2 x^mu / dlambda^2 = -Gamma^mu_ab u^a u^b
    Vector3d geodesic_accel(const Vector3d &pos, const Vector3d &vel) const;

    // Static ISCO formula (usable without an instance)
    static double compute_isco(double M, double a);
};

#endif
