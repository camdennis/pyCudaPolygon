#pragma once
#include <cuda_runtime.h>

// SHAKE simultaneous constraint projection
//
// For each polygon p (n vertices), enforces:
//   C_k(x) = |x_{k+1} - x_k| - L_k = 0   for k = 0..n-1   (edge constraints)
//   A(x) - A_0 = 0                                           (area constraint)
//
// One Newton step per iteration:
//   Assemble  M = J Jᵀ   ((n+1) × (n+1))
//   Solve     M λ = C(x)
//   Apply     Δx = -Jᵀ λ
//
// Launch: one block per polygon, blockDim.x = n.
// Threads 0..n-1 handle vertex-level parallel work.
// Thread 0 handles the serial n×n assembly and LU solve.

__device__ __forceinline__ double shakeWrap(double x) {
    return x - floor(x + 0.5);
}

// In-place LU factorization (no pivoting) of a nc×nc row-major matrix.
// On exit the lower unit-triangular L and upper triangular U are packed into mat.
// Returns false if a zero pivot is encountered.
__device__ bool shakeLUFactor(double* mat, int nc) {
    for (int col = 0; col < nc; col++) {
        double pivot = mat[col * nc + col];
        if (fabs(pivot) < 1e-30) return false;
        double inv = 1.0 / pivot;
        for (int row = col + 1; row < nc; row++) {
            double m = mat[row * nc + col] * inv;
            mat[row * nc + col] = m;
            for (int k = col + 1; k < nc; k++)
                mat[row * nc + k] -= m * mat[col * nc + k];
        }
    }
    return true;
}

// Solve L U x = b in-place (b overwritten with x).
// L is unit lower triangular, U is upper triangular, both packed in mat.
__device__ void shakeLUSolve(const double* mat, double* b, int nc) {
    // Forward: L y = b
    for (int i = 0; i < nc; i++)
        for (int j = 0; j < i; j++)
            b[i] -= mat[i * nc + j] * b[j];
    // Back: U x = y
    for (int i = nc - 1; i >= 0; i--) {
        for (int j = i + 1; j < nc; j++)
            b[i] -= mat[i * nc + j] * b[j];
        b[i] /= mat[i * nc + i];
    }
}

// One block per polygon.
__global__ void shakeProjectKernel(
        int numPolygons,
        const int* startIndices,
        const int* next,
        const int* prev,
        double* positions,
        const double* targetEdgeLengths,
        const double* targetAreas,
        int maxIter,
        double tol,
        int* maxIterOut)
{
    int p = blockIdx.x;
    if (p >= numPolygons) return;

    int s  = startIndices[p];
    int n  = startIndices[p + 1] - s;
    int nc = n + 1;

    extern __shared__ double smem[];
    double* ehat = smem;              // n*2
    double* gA   = ehat + n * 2;     // n*2
    double* mat  = gA   + n * 2;     // nc*nc
    double* rhs  = mat  + nc * nc;   // nc
    double* flag = rhs  + nc;        // 1: convergence flag (0 = keep going, 1 = done)

    int tid = threadIdx.x;
    int actualIter = maxIter;

    for (int iter = 0; iter < maxIter; iter++) {
        if (tid < n) {
            int k  = s + tid;
            int nk = next[k];
            int pk = prev[k];

            double ex  = shakeWrap(positions[2*nk]   - positions[2*k]);
            double ey  = shakeWrap(positions[2*nk+1] - positions[2*k+1]);
            double len = sqrt(ex*ex + ey*ey);
            double inv = (len > 1e-14) ? 1.0 / len : 0.0;
            ehat[tid*2]   = ex * inv;
            ehat[tid*2+1] = ey * inv;
            rhs[tid]      = len - targetEdgeLengths[p];

            double xs  = positions[2*s],   ys  = positions[2*s+1];
            double ynk = shakeWrap(positions[2*nk+1] - ys);
            double ypk = shakeWrap(positions[2*pk+1] - ys);
            double xnk = shakeWrap(positions[2*nk]   - xs);
            double xpk = shakeWrap(positions[2*pk]   - xs);
            gA[tid*2]   = 0.5 * (ynk - ypk);
            gA[tid*2+1] = 0.5 * (xpk - xnk);
        }
        __syncthreads();

        // ── Serial (thread 0): area residual, convergence check, build JJᵀ, LU-solve ──
        if (tid == 0) {
            // area residual via shoelace
            double xs   = positions[2*s],   ys = positions[2*s+1];
            double area = 0.0;
            for (int i = 0; i < n; i++) {
                int k  = s + i,       nk = next[k];
                double xk  = shakeWrap(positions[2*k]   - xs);
                double yk  = shakeWrap(positions[2*k+1] - ys);
                double xnk = shakeWrap(positions[2*nk]  - xs);
                double ynk = shakeWrap(positions[2*nk+1]- ys);
                area += xk * ynk - xnk * yk;
            }
            rhs[n] = 0.5 * area - targetAreas[p];

            // convergence check: if all violations < tol, zero lambdas and signal done
            flag[0] = 0.0;
            if (tol > 0.0) {
                double maxViol = 0.0;
                for (int i = 0; i < nc; i++) maxViol = fmax(maxViol, fabs(rhs[i]));
                if (maxViol < tol) {
                    for (int i = 0; i < nc; i++) rhs[i] = 0.0;
                    flag[0] = 1.0;
                }
            }

            if (flag[0] == 0.0) {
                // zero mat
                for (int i = 0; i < nc * nc; i++) mat[i] = 0.0;

                // diagonal: edges = 2 (each edge has two unit-vector contributions)
                for (int i = 0; i < n; i++) mat[i*nc + i] = 2.0;

                // diagonal: area = ||grad A||²
                double gnorm2 = 0.0;
                for (int i = 0; i < n; i++)
                    gnorm2 += gA[i*2]*gA[i*2] + gA[i*2+1]*gA[i*2+1];
                mat[n*nc + n] = gnorm2;

                // off-diagonal: adjacent edge pairs share one vertex
                for (int i = 0; i < n; i++) {
                    int j = (i + 1) % n;
                    double d = -(ehat[i*2]*ehat[j*2] + ehat[i*2+1]*ehat[j*2+1]);
                    mat[i*nc + j] = d;
                    mat[j*nc + i] = d;
                }

                // off-diagonal: edge-area coupling
                for (int i = 0; i < n; i++) {
                    int j = (i + 1) % n;
                    double c = ehat[i*2]  *(gA[j*2]   - gA[i*2])
                             + ehat[i*2+1]*(gA[j*2+1] - gA[i*2+1]);
                    mat[i*nc + n] = c;
                    mat[n*nc + i] = c;
                }

                // LU factorize and solve; rhs becomes lambda
                if (shakeLUFactor(mat, nc))
                    shakeLUSolve(mat, rhs, nc);
                else
                    for (int i = 0; i < nc; i++) rhs[i] = 0.0;
            }
        }
        __syncthreads();

        // Parallel: apply (rhs = lambdas, or zeros if converged)
        if (tid < n) {
            int k    = s + tid;
            int prev_local = (tid + n - 1) % n;

            double dx = rhs[tid] * ehat[tid*2]     - rhs[prev_local] * ehat[prev_local*2]     - rhs[n] * gA[tid*2];
            double dy = rhs[tid] * ehat[tid*2+1]   - rhs[prev_local] * ehat[prev_local*2+1]   - rhs[n] * gA[tid*2+1];

            positions[2*k]   += dx;
            positions[2*k+1] += dy;
        }
        __syncthreads();

        if (flag[0] != 0.0) {
            actualIter = iter + 1;
            break;
        }
    }
    if (tid == 0) atomicMax(maxIterOut, actualIter);
}
