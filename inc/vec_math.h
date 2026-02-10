// ============================================================================
// Lightweight vector / matrix types for CPU + GPU shared physics code.
//
// Replaces Eigen's Vector3d, Vector4d, Matrix3d with simple types that
// compile under both g++ and nvcc (__host__ __device__).
// ============================================================================

#ifndef _BH_VEC_MATH_H
#define _BH_VEC_MATH_H

#include <math.h>

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279
#endif

#ifdef __CUDACC__
#define BH_FUNC __host__ __device__
#else
#define BH_FUNC
#endif

// ============================================================================
// Utility
// ============================================================================

BH_FUNC inline double dclamp(double x, double lo, double hi)
{
    return fmin(fmax(x, lo), hi);
}

BH_FUNC inline bool bh_isfinite(double x)
{
#ifdef __CUDA_ARCH__
    // Bit-level IEEE 754 check — immune to --use_fast_math which may
    // optimise away NaN comparisons.  Exponent bits [52:62] all-ones → Inf/NaN.
    unsigned long long bits = __double_as_longlong(x);
    return ((bits >> 52) & 0x7FFull) != 0x7FFull;
#else
    return (x == x) && (x - x == 0.0);
#endif
}

// ============================================================================
// dvec3 — 3-component double vector
// ============================================================================

struct dvec3
{
    double x, y, z;

    BH_FUNC dvec3() : x(0), y(0), z(0) {}
    BH_FUNC dvec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    BH_FUNC dvec3 operator+(const dvec3 &b) const { return {x + b.x, y + b.y, z + b.z}; }
    BH_FUNC dvec3 operator-(const dvec3 &b) const { return {x - b.x, y - b.y, z - b.z}; }
    BH_FUNC dvec3 operator*(double s) const { return {x * s, y * s, z * s}; }
    BH_FUNC dvec3 operator/(double s) const { double inv = 1.0 / s; return {x * inv, y * inv, z * inv}; }

    BH_FUNC dvec3 &operator+=(const dvec3 &b) { x += b.x; y += b.y; z += b.z; return *this; }
    BH_FUNC dvec3 &operator-=(const dvec3 &b) { x -= b.x; y -= b.y; z -= b.z; return *this; }
    BH_FUNC dvec3 &operator*=(double s) { x *= s; y *= s; z *= s; return *this; }

    BH_FUNC double dot(const dvec3 &b) const { return x * b.x + y * b.y + z * b.z; }
    BH_FUNC dvec3 cross(const dvec3 &b) const
    {
        return {y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x};
    }
    BH_FUNC double squaredNorm() const { return x * x + y * y + z * z; }
    BH_FUNC double norm() const { return sqrt(squaredNorm()); }
    BH_FUNC dvec3 normalized() const
    {
        double n = norm();
        return (n > 1e-15) ? dvec3(x / n, y / n, z / n) : dvec3(0, 0, 0);
    }
    BH_FUNC dvec3 cwiseMax(double v) const { return {fmax(x, v), fmax(y, v), fmax(z, v)}; }
    BH_FUNC dvec3 cwiseMin(double v) const { return {fmin(x, v), fmin(y, v), fmin(z, v)}; }
};

BH_FUNC inline dvec3 operator*(double s, const dvec3 &v) { return {s * v.x, s * v.y, s * v.z}; }

// ============================================================================
// dvec4 — 4-component double vector (t, x, y, z)
// ============================================================================

struct dvec4
{
    double t, x, y, z;

    BH_FUNC dvec4() : t(0), x(0), y(0), z(0) {}
    BH_FUNC dvec4(double t_, double x_, double y_, double z_) : t(t_), x(x_), y(y_), z(z_) {}
};

// ============================================================================
// dmat3 — 3×3 double matrix (column-major), for camera / sky rotation
// ============================================================================

struct dmat3
{
    double m[9]; // column-major: element(row, col) = m[col * 3 + row]

    BH_FUNC dmat3()
    {
        m[0] = 1; m[1] = 0; m[2] = 0;
        m[3] = 0; m[4] = 1; m[5] = 0;
        m[6] = 0; m[7] = 0; m[8] = 1;
    }

    BH_FUNC double  operator()(int row, int col) const { return m[col * 3 + row]; }
    BH_FUNC double &operator()(int row, int col)       { return m[col * 3 + row]; }

    BH_FUNC dvec3 col(int c) const { return {m[c * 3], m[c * 3 + 1], m[c * 3 + 2]}; }

    BH_FUNC dvec3 operator*(const dvec3 &v) const
    {
        return {
            m[0] * v.x + m[3] * v.y + m[6] * v.z,
            m[1] * v.x + m[4] * v.y + m[7] * v.z,
            m[2] * v.x + m[5] * v.y + m[8] * v.z};
    }

    BH_FUNC dmat3 operator*(const dmat3 &b) const
    {
        dmat3 r;
        for (int c = 0; c < 3; ++c)
            for (int row = 0; row < 3; ++row)
                r.m[c * 3 + row] = m[0 * 3 + row] * b.m[c * 3 + 0] +
                                   m[1 * 3 + row] * b.m[c * 3 + 1] +
                                   m[2 * 3 + row] * b.m[c * 3 + 2];
        return r;
    }

    // --- Rotation matrices around principal axes -------------------------

    static BH_FUNC dmat3 rotation_x(double angle)
    {
        dmat3 r;
        double c = cos(angle), s = sin(angle);
        r.m[0] = 1; r.m[3] = 0; r.m[6] = 0;
        r.m[1] = 0; r.m[4] = c; r.m[7] = -s;
        r.m[2] = 0; r.m[5] = s; r.m[8] = c;
        return r;
    }

    static BH_FUNC dmat3 rotation_y(double angle)
    {
        dmat3 r;
        double c = cos(angle), s = sin(angle);
        r.m[0] = c;  r.m[3] = 0; r.m[6] = s;
        r.m[1] = 0;  r.m[4] = 1; r.m[7] = 0;
        r.m[2] = -s; r.m[5] = 0; r.m[8] = c;
        return r;
    }

    static BH_FUNC dmat3 rotation_z(double angle)
    {
        dmat3 r;
        double c = cos(angle), s = sin(angle);
        r.m[0] = c; r.m[3] = -s; r.m[6] = 0;
        r.m[1] = s; r.m[4] = c;  r.m[7] = 0;
        r.m[2] = 0; r.m[5] = 0;  r.m[8] = 1;
        return r;
    }
};

#endif // _BH_VEC_MATH_H
