// dual.cuh — minimal forward-mode AD via dual numbers for CUDA device code.
//
// Each Dual<N> carries a value `v` and a derivative vector `d[N]`. Standard
// arithmetic operators propagate derivatives via the chain rule. Used by the
// feature-pair walk kernel to compute crossing-point Jacobians w.r.t. the
// underlying backbone vertices (delta>0 case, where flat-segment endpoints
// and arc centers depend non-trivially on neighbor vertices).
//
// Convention: pass values + slot indices into helpers; helpers assign
// d[slot] = 1.0 for the corresponding input variable.
//
// Stack budget: a Dual<16> = 17 doubles = 136 bytes. Keep intermediate
// Dual<N>s tight; inline when possible.

#ifndef DUAL_CUH
#define DUAL_CUH

#include <cuda_runtime.h>
#include <math.h>

// Fixed N=16 here. Slot layout for the feature-pair walk (edge-edge):
//   A-side: 0..1 = prev_A, 2..3 = vA, 4..5 = next_A, 6..7 = next2_A
//   B-side: 8..9 = prev_B, 10..11 = vB, 12..13 = next_B, 14..15 = next2_B
#define DUAL_N 16

struct Dual {
    double v;
    double d[DUAL_N];
};

__device__ static void dual_zero(Dual& r) {
    r.v = 0.0;
    for (int i = 0; i < DUAL_N; ++i) r.d[i] = 0.0;
}

__device__ static void dual_const(Dual& r, double v) {
    r.v = v;
    for (int i = 0; i < DUAL_N; ++i) r.d[i] = 0.0;
}

__device__ static void dual_var(Dual& r, double v, int slot) {
    r.v = v;
    for (int i = 0; i < DUAL_N; ++i) r.d[i] = 0.0;
    r.d[slot] = 1.0;
}

__device__ static void dual_add(const Dual& a, const Dual& b, Dual& r) {
    r.v = a.v + b.v;
    for (int i = 0; i < DUAL_N; ++i) r.d[i] = a.d[i] + b.d[i];
}

__device__ static void dual_sub(const Dual& a, const Dual& b, Dual& r) {
    r.v = a.v - b.v;
    for (int i = 0; i < DUAL_N; ++i) r.d[i] = a.d[i] - b.d[i];
}

__device__ static void dual_neg(const Dual& a, Dual& r) {
    r.v = -a.v;
    for (int i = 0; i < DUAL_N; ++i) r.d[i] = -a.d[i];
}

__device__ static void dual_mul(const Dual& a, const Dual& b, Dual& r) {
    double av = a.v, bv = b.v;
    r.v = av * bv;
    for (int i = 0; i < DUAL_N; ++i) r.d[i] = av * b.d[i] + a.d[i] * bv;
}

__device__ static void dual_scale(const Dual& a, double s, Dual& r) {
    r.v = a.v * s;
    for (int i = 0; i < DUAL_N; ++i) r.d[i] = a.d[i] * s;
}

__device__ static void dual_div(const Dual& a, const Dual& b, Dual& r) {
    double bv = b.v, inv = 1.0 / bv;
    double av = a.v;
    r.v = av * inv;
    double inv2 = inv * inv;
    for (int i = 0; i < DUAL_N; ++i) r.d[i] = (a.d[i] * bv - av * b.d[i]) * inv2;
}

__device__ static void dual_sqrt(const Dual& a, Dual& r) {
    double s = sqrt(a.v);
    r.v = s;
    double inv2s = (s > 1e-300) ? 0.5 / s : 0.0;
    for (int i = 0; i < DUAL_N; ++i) r.d[i] = a.d[i] * inv2s;
}

__device__ static void dual_fabs(const Dual& a, Dual& r) {
    double s = (a.v >= 0.0) ? 1.0 : -1.0;
    r.v = s * a.v;
    for (int i = 0; i < DUAL_N; ++i) r.d[i] = s * a.d[i];
}

__device__ static void dual_sin(const Dual& a, Dual& r) {
    double c = cos(a.v);
    r.v = sin(a.v);
    for (int i = 0; i < DUAL_N; ++i) r.d[i] = a.d[i] * c;
}

__device__ static void dual_cos(const Dual& a, Dual& r) {
    double s = sin(a.v);
    r.v = cos(a.v);
    for (int i = 0; i < DUAL_N; ++i) r.d[i] = -a.d[i] * s;
}

// d(atan2(y, x))/d(*) = (x * dy - y * dx) / (x^2 + y^2)
__device__ static void dual_atan2(const Dual& y, const Dual& x, Dual& r) {
    double denom = x.v*x.v + y.v*y.v;
    double inv = (denom > 1e-300) ? 1.0 / denom : 0.0;
    r.v = atan2(y.v, x.v);
    for (int i = 0; i < DUAL_N; ++i) r.d[i] = (x.v * y.d[i] - y.v * x.d[i]) * inv;
}

// dwrap/dx = 1 almost everywhere (round() is locally constant). For the
// purposes of force computation the discontinuities are measure-zero.
__device__ static void dual_wrap_box(const Dual& a, Dual& r) {
    r.v = a.v - round(a.v);
    for (int i = 0; i < DUAL_N; ++i) r.d[i] = a.d[i];
}

// 2D point with per-component Duals.
struct DPoint {
    Dual x, y;
};

__device__ static void dpoint_var(DPoint& p, double x, double y, int slot_x, int slot_y) {
    dual_var(p.x, x, slot_x);
    dual_var(p.y, y, slot_y);
}

#endif // DUAL_CUH
