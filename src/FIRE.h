#pragma once
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cub/device/device_reduce.cuh>
#include "cuda_check.h"

__global__ void updatePositionAndVelocityFIREKernel(int numVertices, double* positions, double* velocities, const double* force, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVertices) return;
    for (int dim = 0; dim < 2; dim++) {
        double p = positions[i * 2 + dim]
                 + velocities[i * 2 + dim] * dt
                 + 0.5 * dt * dt * force[i * 2 + dim];
        positions[i * 2 + dim] = p - floor(p);
        velocities[i * 2 + dim] += 0.5 * force[i * 2 + dim] * dt;
    }
}

__global__ void updateVelocityFIREKernel(int numVertices, double* velocities, const double* force, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVertices) return;
    for (int dim = 0; dim < 2; dim++) {
        velocities[i * 2 + dim] += 0.5 * force[i * 2 + dim] * dt;
    }
}

// Welcome back to this detail
__global__ void bendVelocityTowardsForceFIREKernel(int numVertices, double* velocities, const double* force, double alpha, double vnorm, double fnorm) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVertices) return;
    double ratio = (fnorm > 1e-14) ? (vnorm / fnorm) : 0.0;
    for (int dim = 0; dim < 2; dim++) {
        velocities[i * 2 + dim] = (1.0 - alpha) * velocities[i * 2 + dim] + alpha * ratio * force[i * 2 + dim];
    }
}

// Rederive velocity from actual displacement (accounts for XPBD position correction).
// vel[i] = (posNew[i] - posOld[i]) / dt, with periodic wrapping to [-0.5, 0.5).
__global__ void rederiveVelocityFromDisplacementKernel(int numVertices, double* vel, const double* posNew, const double* posOld, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVertices) return;
    for (int dim = 0; dim < 2; dim++) {
        double d = posNew[i * 2 + dim] - posOld[i * 2 + dim];
        d = d - floor(d + 0.5);
        vel[i * 2 + dim] = d / dt;
    }
}

// Element-wise product: out[i] = a[i] * b[i]
__global__ void multiplyFIREKernel(const double* a, const double* b, double* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}

// Dot product sum(a[i]*b[i]) using pre-allocated scratch and result buffers.
// Caches the CUB temp buffer across calls to avoid per-call cudaMalloc overhead.
static double dotProductFIRE(const double* a, const double* b, int n,
                              double* scratch, double* result) {
    static void* s_cubTMP = nullptr;
    static size_t s_cubBytes = 0;

    int blocks = (n + 255) / 256;
    multiplyFIREKernel<<<blocks, 256>>>(a, b, scratch, n);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    size_t needed = 0;
    CUDA_CHECK(cub::DeviceReduce::Sum(static_cast<void*>(nullptr), needed, scratch, result, n));
    if (needed > s_cubBytes) {
        if (s_cubTMP) { CUDA_CHECK(cudaFree(s_cubTMP)); }
        CUDA_CHECK(cudaMalloc(&s_cubTMP, needed));
        s_cubBytes = needed;
    }
    CUDA_CHECK(cub::DeviceReduce::Sum(s_cubTMP, s_cubBytes, scratch, result, n));
    CUDA_CHECK(cudaDeviceSynchronize());

    double h_result;
    CUDA_CHECK(cudaMemcpy(&h_result, result, sizeof(double), cudaMemcpyDeviceToHost));
    return h_result;
}
