
#ifndef _BH_DISK_H
#define _BH_DISK_H

#include "common.h"

struct accretion_disk_s
{
    double mass;      // Black hole mass M
    double spin;      // Black hole spin a
    double inner_r;   // Inner edge (ISCO or custom)
    double outer_r;   // Outer edge
    double thickness; // Half-thickness scale height at reference radius
    double density0;  // Base density scale
    double opacity0;  // Base opacity scale (absorption per unit density)

    // Construct with ISCO as default inner radius
    accretion_disk_s(double M, double a, double r_outer, double h, double rho0, double kappa0);

    // Compute the ISCO radius for Kerr
    static double isco_radius(double M, double a);

    // Kerr-Schild radius from Cartesian position (spin axis = y)
    double ks_radius(const Vector3d &pos) const;

    // Disk half-thickness at a given KS radius (flared disk: h ~ r)
    double half_thickness(double r_ks) const;

    // Is this Cartesian point inside the disk volume?
    bool contains(const Vector3d &pos) const;

    // Gas density at a point (power-law in r, Gaussian falloff in height)
    double density(const Vector3d &pos) const;

    // Temperature at a point (Novikov-Thorne-like profile)
    double temperature(const Vector3d &pos) const;

    // Thermal emissivity as RGB (blackbody-ish color from temperature)
    Vector3d emissivity(const Vector3d &pos) const;

    // Absorption coefficient at a point
    double absorption(const Vector3d &pos) const;

    // Keplerian 4-velocity of gas (prograde circular orbit in Kerr-Schild coords, spin axis = y)
    Vector4d gas_four_velocity(const Vector3d &pos) const;

    // Redshift factor g = (k_mu u^mu)_obs / (k_mu u^mu)_emit
    // k is the photon 4-momentum, u_emit is the gas 4-velocity
    // observer is assumed static at infinity (u_obs = (-1,0,0,0) in KS)
    static double redshift_factor(const Vector4d &photon_k, const Vector4d &gas_u,
                                  const Matrix4d &g_metric);

    // Convert temperature to approximate RGB via blackbody
    static Vector3d temperature_to_rgb(double T);
};

#endif
