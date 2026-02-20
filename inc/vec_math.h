// ============================================================================
// Lightweight vector / matrix types for CPU + GPU shared physics code.
//
// Replaces Eigen's Vector3d, Vector4d, Matrix3d with simple types that
// compile under both g++ and nvcc (__host__ __device__).
//
// Precision: controlled by BH_USE_FLOAT.
//   -DBH_USE_FLOAT  -> bh_real = float  (fast, GPU-friendly)
//   (default)       -> bh_real = double  (reference quality)
// ============================================================================

#ifndef _BH_VEC_MATH_H
#define _BH_VEC_MATH_H

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
#else
#include <math.h>
#endif

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279
#endif

#ifdef __CUDACC__
#define BH_FUNC __host__ __device__
#elif defined(__METAL_VERSION__)
#define BH_FUNC
#else
#define BH_FUNC
#endif

// Metal Shading Language requires explicit address space qualifiers on all
// reference and pointer types.  All physics data lives in the thread (local)
// address space — BH_THREAD expands to `thread` under Metal, empty otherwise.
#ifdef __METAL_VERSION__
#define BH_THREAD thread
#else
#define BH_THREAD
#endif

// ============================================================================
// Precision typedef
// ============================================================================

#ifdef BH_USE_FLOAT
using bh_real = float;
#else
using bh_real = double;
#endif

// ============================================================================
// Utility
// ============================================================================

BH_FUNC inline bh_real dclamp(bh_real x, bh_real lo, bh_real hi)
{
    return fmin(fmax(x, lo), hi);
}

// --------------------------------------------------------------------------
// Type-safe wrappers for two-arg math functions.
//
// When bh_real = float, bare double literals (0.01, 1e-12 …) cause mixed
// float/double overload resolution that picks the <cmath> constexpr
// __host__-only version — illegal in CUDA __device__ code.  These wrappers
// force both arguments to bh_real at the call site, so the correct
// same-type CUDA intrinsic is selected.
// --------------------------------------------------------------------------
BH_FUNC inline bh_real bh_fmax(bh_real a, bh_real b) { return fmax(a, b); }
BH_FUNC inline bh_real bh_fmin(bh_real a, bh_real b) { return fmin(a, b); }
BH_FUNC inline bh_real bh_pow(bh_real a, bh_real b) { return pow(a, b); }

BH_FUNC inline bool bh_isfinite(bh_real x)
{
#ifdef __CUDA_ARCH__
    // Bit-level IEEE 754 check -- immune to --use_fast_math which may
    // optimise away NaN comparisons.  Exponent bits all-ones -> Inf/NaN.
#ifdef BH_USE_FLOAT
    unsigned int bits = __float_as_uint(x);
    return ((bits >> 23) & 0xFFu) != 0xFFu;
#else
    unsigned long long bits = __double_as_longlong(x);
    return ((bits >> 52) & 0x7FFull) != 0x7FFull;
#endif
#elif defined(__METAL_VERSION__)
    return metal::isfinite(x);
#else
    return (x == x) && (x - x == bh_real(0));
#endif
}

// cbrt: Metal Shading Language has no cbrt — use pow(x, 1/3) with sign handling
BH_FUNC inline bh_real bh_cbrt(bh_real x)
{
#ifdef __METAL_VERSION__
    return (x >= 0.0) ? pow(x, bh_real(1.0 / 3.0)) : -pow(-x, bh_real(1.0 / 3.0));
#else
    return cbrt(x);
#endif
}

// fabs: Metal uses metal::abs for floating-point types
BH_FUNC inline bh_real bh_fabs(bh_real x)
{
#ifdef __METAL_VERSION__
    return metal::abs(x);
#else
    return fabs(x);
#endif
}

// ============================================================================
// dvec3 -- 3-component vector
// ============================================================================

struct dvec3
{
    bh_real x, y, z;

    BH_FUNC dvec3() : x(0), y(0), z(0) {}
    BH_FUNC dvec3(bh_real x_, bh_real y_, bh_real z_) : x(x_), y(y_), z(z_) {}

    BH_FUNC dvec3 operator+(BH_THREAD const dvec3 &b) const { return {x + b.x, y + b.y, z + b.z}; }
    BH_FUNC dvec3 operator-(BH_THREAD const dvec3 &b) const { return {x - b.x, y - b.y, z - b.z}; }
    BH_FUNC dvec3 operator*(bh_real s) const { return {x * s, y * s, z * s}; }
    BH_FUNC dvec3 operator/(bh_real s) const
    {
        bh_real inv = bh_real(1) / s;
        return {x * inv, y * inv, z * inv};
    }

    BH_FUNC BH_THREAD dvec3 &operator+=(BH_THREAD const dvec3 &b)
    {
        x += b.x;
        y += b.y;
        z += b.z;
        return *this;
    }
    BH_FUNC BH_THREAD dvec3 &operator-=(BH_THREAD const dvec3 &b)
    {
        x -= b.x;
        y -= b.y;
        z -= b.z;
        return *this;
    }
    BH_FUNC BH_THREAD dvec3 &operator*=(bh_real s)
    {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }

    BH_FUNC bh_real dot(BH_THREAD const dvec3 &b) const { return x * b.x + y * b.y + z * b.z; }
    BH_FUNC dvec3 cross(BH_THREAD const dvec3 &b) const
    {
        return {y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x};
    }
    BH_FUNC bh_real squaredNorm() const { return x * x + y * y + z * z; }
    BH_FUNC bh_real norm() const { return sqrt(squaredNorm()); }
    BH_FUNC dvec3 normalized() const
    {
        bh_real n = norm();
        return (n > bh_real(1e-15)) ? dvec3(x / n, y / n, z / n) : dvec3(0, 0, 0);
    }
    BH_FUNC dvec3 cwiseMax(bh_real v) const { return {fmax(x, v), fmax(y, v), fmax(z, v)}; }
    BH_FUNC dvec3 cwiseMin(bh_real v) const { return {fmin(x, v), fmin(y, v), fmin(z, v)}; }
};

BH_FUNC inline dvec3 operator*(bh_real s, BH_THREAD const dvec3 &v) { return {s * v.x, s * v.y, s * v.z}; }

// ============================================================================
// dvec4 -- 4-component vector (t, x, y, z)
// ============================================================================

struct dvec4
{
    bh_real t, x, y, z;

    BH_FUNC dvec4() : t(0), x(0), y(0), z(0) {}
    BH_FUNC dvec4(bh_real t_, bh_real x_, bh_real y_, bh_real z_) : t(t_), x(x_), y(y_), z(z_) {}
};

// ============================================================================
// dmat3 -- 3x3 matrix (column-major), for camera / sky rotation
// ============================================================================

struct dmat3
{
    bh_real m[9]; // column-major: element(row, col) = m[col * 3 + row]

    BH_FUNC dmat3()
    {
        m[0] = 1;
        m[1] = 0;
        m[2] = 0;
        m[3] = 0;
        m[4] = 1;
        m[5] = 0;
        m[6] = 0;
        m[7] = 0;
        m[8] = 1;
    }

    BH_FUNC bh_real operator()(int row, int col) const { return m[col * 3 + row]; }
    BH_FUNC BH_THREAD bh_real &operator()(int row, int col) { return m[col * 3 + row]; }

    BH_FUNC dvec3 col(int c) const { return {m[c * 3], m[c * 3 + 1], m[c * 3 + 2]}; }

    BH_FUNC dvec3 operator*(BH_THREAD const dvec3 &v) const
    {
        return {
            m[0] * v.x + m[3] * v.y + m[6] * v.z,
            m[1] * v.x + m[4] * v.y + m[7] * v.z,
            m[2] * v.x + m[5] * v.y + m[8] * v.z};
    }

    BH_FUNC dmat3 operator*(BH_THREAD const dmat3 &b) const
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

    static BH_FUNC dmat3 rotation_x(bh_real angle)
    {
        dmat3 r;
        bh_real c = cos(angle), s = sin(angle);
        r.m[0] = 1;
        r.m[3] = 0;
        r.m[6] = 0;
        r.m[1] = 0;
        r.m[4] = c;
        r.m[7] = -s;
        r.m[2] = 0;
        r.m[5] = s;
        r.m[8] = c;
        return r;
    }

    static BH_FUNC dmat3 rotation_y(bh_real angle)
    {
        dmat3 r;
        bh_real c = cos(angle), s = sin(angle);
        r.m[0] = c;
        r.m[3] = 0;
        r.m[6] = s;
        r.m[1] = 0;
        r.m[4] = 1;
        r.m[7] = 0;
        r.m[2] = -s;
        r.m[5] = 0;
        r.m[8] = c;
        return r;
    }

    static BH_FUNC dmat3 rotation_z(bh_real angle)
    {
        dmat3 r;
        bh_real c = cos(angle), s = sin(angle);
        r.m[0] = c;
        r.m[3] = -s;
        r.m[6] = 0;
        r.m[1] = s;
        r.m[4] = c;
        r.m[7] = 0;
        r.m[2] = 0;
        r.m[5] = 0;
        r.m[8] = 1;
        return r;
    }
};

#endif // _BH_VEC_MATH_H
