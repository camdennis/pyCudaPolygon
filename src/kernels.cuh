#include <cuda_runtime.h>
#include <iostream>
#include <cufft.h>
#include <complex>
#include <curand_kernel.h>
#include <float.h>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/iterator/transform_input_iterator.cuh>
#include <cub/device/device_run_length_encode.cuh>

// Feature-pair refactor headers. features.cuh and intersect.cuh have
// concrete content; integrate.cuh and parityWalk.cuh fill in over
// Phase 1-3. geom.cuh holds the periodic-wrap helpers that all the
// others share.
#include "geom.cuh"
#include "features.cuh"
#include "intersect.cuh"
#include "integrate.cuh"
#include "parityWalk.cuh"
#include "dual.cuh"

static const dim3 myBlockDim(16, 16);
static const int blockSize = 256;
static const double pi = 3.141592653589793238462643383279;

// init

__global__ void initShapeRangesKernel(int* shapeStart, int* shapeEnd, int numPolygons, int sentinel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPolygons) {
        shapeStart[idx] = sentinel;
        shapeEnd[idx]   = -1;
    }
}

// structs:

struct ExtractHigh32 {
    __host__ __device__ unsigned int operator()(uint64_t x) const {
        return static_cast<unsigned int>(x >> 32);
    }
};

// Helpers:

__global__ void fillSequenceKernel(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = idx;
    }
}

__global__ void fillScalarKernel(double* data, int n, double value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = value;
}

__global__ void lowerBoundKernel(const int* __restrict__ haystack, int haystack_size, const int* __restrict__ needles, int needles_size, int* __restrict__ results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= needles_size) return;
    
    int needle = needles[idx];
    int left = 0, right = haystack_size;
    
    // Binary search for lower bound
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (haystack[mid] < needle) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    results[idx] = left;
}

__global__ void maxReduceKernel(const int* __restrict__ data, int n, int* __restrict__ out) {
    extern __shared__ int sdata[];              // size: blockDim.x * sizeof(int)
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    int localMax = INT_MIN;
    for (int i = idx; i < n; i += stride) {
        int v = data[i];
        if (v > localMax) localMax = v;
    }
    sdata[tid] = localMax;
    __syncthreads();

    // in-block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            int a = sdata[tid];
            int b = sdata[tid + s];
            sdata[tid] = (a > b) ? a : b;
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax(out, sdata[0]);
    }
}

__global__ void gatherKernel_double2(const double2* input, const uint32_t* perm, double2* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[perm[idx]];
    }
}

__global__ void gatherKernel_int64(const uint64_t* input, const uint32_t* perm, uint64_t* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[perm[idx]];
    }
}

// wrap, wrap2, wrapPeriodic now live in geom.cuh, included above.

__global__ void computeNextPrevKernel(int* next, int* prev, const int* startIndices, const int* shapeId, int numVertices) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVertices) return;
    int s = shapeId[i];
    int start = startIndices[s];
    int end   = startIndices[s+1];
    next[i] = (i + 1 == end) ? start : i + 1;
    prev[i] = (i == start)   ? end - 1 : i - 1;
}

// functions:

// h(pt1, pt2, startPoint) and g12(...) now live in integrate.cuh as
// integrate::chordArea and integrate::chordGradients. Callers in this
// file use the namespaced names directly.

__device__ void getDf(double2 vi, double2 vzi, double2 vj, double2 vzj, double* df) {
    // Edge vectors with periodic wrap
    double2 dj = wrap2({vzj.x - vj.x, vzj.y - vj.y});
    double2 di = wrap2({vzi.x - vi.x, vzi.y - vi.y});
    double2 dij = wrap2({vj.x - vi.x, vj.y - vi.y});

    double w = dj.x * di.y - dj.y * di.x;
    double diNorm2 = di.x*di.x + di.y*di.y;
    double djNorm2 = dj.x*dj.x + dj.y*dj.y;
    if (w*w < 1e-12 * diNorm2 * djNorm2 || diNorm2 < 1e-28 || djNorm2 < 1e-28) {
        for (int i = 0; i < 16; i++) df[i] = 0.0;
        return;
    }
    double k = dj.x * dij.y - dj.y * dij.x;
    double u = k / w;

    // dk matrix (2 rows, 4 columns) – only columns 0,2,3 are non‑zero
    double dk[8] = {0};   // [col*2 + row]
    // col 0 (vi)
    dk[0*2 + 0] =  dj.y;   // row0
    dk[0*2 + 1] = -dj.x;   // row1
    // col 2 (vj)
    dk[2*2 + 0] = -dij.y - dj.y;
    dk[2*2 + 1] =  dj.x + dij.x;
    // col 3 (vzj)
    dk[3*2 + 0] =  dij.y;
    dk[3*2 + 1] = -dij.x;

    // dw matrix (all columns)
    double dw[8];
    // col 0 (vi)
    dw[0*2 + 0] =  dj.y;
    dw[0*2 + 1] = -dj.x;
    // col 1 (vzi)
    dw[1*2 + 0] = -dj.y;
    dw[1*2 + 1] =  dj.x;
    // col 2 (vj)
    dw[2*2 + 0] = -di.y;
    dw[2*2 + 1] =  di.x;
    // col 3 (vzj)
    dw[3*2 + 0] =  di.y;
    dw[3*2 + 1] = -di.x;

    // Initialise df to zero
    for (int i = 0; i < 16; ++i) df[i] = 0.0;

    // Main loops – exactly as in Python
    for (int alpha = 0; alpha < 2; ++alpha) {          // output component (x,y)
        for (int beta = 0; beta < 2; ++beta) {         // input component (x,y)
            for (int p = 0; p < 4; ++p) {              // vertex index (vi, vzi, vj, vzj)
                double du = dk[p*2 + beta] / w - u * dw[p*2 + beta] / w;
                int col = 2*p + beta;
                df[col*2 + alpha] += (alpha == 0 ? di.x : di.y) * du;
            }
            if (alpha == beta) {
                // linear terms for vi and vzi
                df[beta*2 + alpha]     += 1.0 - u;      // col = beta (vi)
                df[(2+beta)*2 + alpha] += u;            // col = 2+beta (vzi)
            }
        }
    }
}

// initializers

__global__ void initStatesKernel(curandState *globalState, unsigned long long seed, int gridSize) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Flatten 2D index to 1D index
    int idx = row * gridSize + col;
    if (row < gridSize && col < gridSize) {
       curand_init(seed, idx, 0, &globalState[idx]);
    }
}

__global__ void initIndicesKernel(uint32_t* indices, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) indices[idx] = idx;
}

// updaters

__global__ void updateAreasCOMKernel(double* areas, double* positions, int* startIndices, int numPolygons) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPolygons) {
        int start = startIndices[idx];
        int end = startIndices[idx + 1];
        double startY = positions[2 * start + 1];
        double dx, dy1, dy2;
        for (int i = start; i < end - 1; i++) {
            dx = (positions[2 * i] - positions[2 * i + 2] + 0.5);
            while (dx < 0.0) {
                dx += 1;
            }
            while (dx > 1.0) {
                dx -= 1.0;
            }
            dx -= 0.5;
            dy1 = (positions[2 * i + 1] - startY + 0.5);
            while (dy1 < 0.0) {
                dy1 += 1;
            }
            while (dy1 > 1.0) {
                dy1 -= 1.0;
            }
            dy2 = (positions[2 * i + 3] - startY + 0.5);
            while (dy2 < 0.0) {
                dy2 += 1;
            }
            while (dy2 > 1.0) {
                dy2 -= 1.0;
            }
            areas[idx] += dx * (dy1 + dy2 - 1.0 + 2.0 * startY) / 2.0;
        }
        dx = (positions[2 * end - 2] - positions[2 * start] + 0.5);
        while (dx < 0.0) {
            dx += 1;
        }
        while (dx > 1.0) {
            dx -= 1.0;
        }
        dx -= 0.5;
        dy1 = (positions[2 * end - 1] - startY + 0.5);
        while (dy1 < 0.0) {
            dy1 += 1;
        }
        while (dy1 > 1.0) {
            dy1 -= 1.0;
        }
        areas[idx] += dx * (dy1 - 0.5 + 2.0 * startY) / 2.0;
    }
}

__global__ void updatePolygonGeometryKernel(int numVertices, int numPolygons, double* positions, int* startIndices, int* shapeId, int* next, int* prev, double* edgeLengths, double* areaParts, double* comParts, double* constraints, double* constraintNormSq) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVertices * 2) return;
    int k = idx / 2;
    int alpha = idx % 2;
    int startIndex = startIndices[shapeId[k]];

    double x1 = wrap(positions[idx] - positions[startIndex * 2 + alpha]);
    double x4 = wrap(positions[prev[k] * 2 + 1 - alpha] - positions[next[k] * 2 + 1 - alpha]);
    double dxPrev = wrap(positions[2 * k] - positions[prev[k] * 2]);
    double dyPrev = wrap(positions[2 * k + 1] - positions[prev[k] * 2 + 1]);
    double dxNext = wrap(positions[2 * k] - positions[next[k] * 2]);
    double dyNext = wrap(positions[2 * k + 1] - positions[next[k] * 2 + 1]);
    comParts[k + numVertices * alpha] = x1;

    double lenPrev = sqrt(dxPrev * dxPrev + dyPrev * dyPrev);
    double lenNext = sqrt(dxNext * dxNext + dyNext * dyNext);

    double areaComp  = (2 * alpha - 1) * x4;
    double edge2Comp = alpha ? (dyNext + dyPrev) : (dxNext + dxPrev);
    double edgeComp = alpha ? (dyNext / lenNext) + (dyPrev / lenPrev) : (dxNext / lenNext) + (dxPrev / lenPrev);

    constraints[6 * k + alpha] = areaComp;
    constraints[6 * k + 2 + alpha] = edge2Comp;
    constraints[6 * k + 4 + alpha] = edgeComp;

    int p = shapeId[k];
    atomicAdd(&constraintNormSq[3 * p + 0], areaComp * areaComp);
    atomicAdd(&constraintNormSq[3 * p + 1], edge2Comp * edge2Comp);
    atomicAdd(&constraintNormSq[3 * p + 2], edgeComp * edgeComp);

    if (alpha) return;
    double dx = wrap(positions[2 * next[k]] - positions[2 * k]);
    double dy = wrap(positions[2 * next[k] + 1] - positions[2 * k + 1]);
    edgeLengths[k] = sqrt(dx * dx + dy * dy);
    double dy2 = wrap(positions[next[k] * 2 + 1] - positions[idx + 1]);
    double x2 = wrap(positions[next[k] * 2 + alpha] - positions[startIndex * 2 + alpha]);
    areaParts[k] = dy2 * (x2 + x1) / 2.0;
}

__global__ void normalizeConstraintsKernel(int numVertices, int* shapeId, double* constraints, const double* constraintNormSq) {
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVertices * 2) return;
    int k     = idx / 2;
    int alpha = idx % 2;
    int p     = shapeId[k];
    for (int c = 0; c < 3; c++) {
        double ns = constraintNormSq[3 * p + c];
        if (ns > 0.0) constraints[6 * k + 2 * c + alpha] /= sqrt(ns);
    }
}

__global__ void mgsInnerProductKernel(int numVertices, int* shapeId, const double* constraints, double* ip, int cA, int cB) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVertices * 2) return;
    int k = idx / 2;
    int alpha = idx % 2;
    atomicAdd(&ip[shapeId[k]], constraints[6 * k + 2 * cA + alpha] * constraints[6 * k + 2 * cB + alpha]);
}

__global__ void mgsSubtractKernel(int numVertices, int* shapeId, double* constraints, const double* ip, double* normSq, int cA, int cB) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVertices * 2) return;
    int k = idx / 2;
    int alpha = idx % 2;
    int p = shapeId[k];
    double val = constraints[6 * k + 2 * cB + alpha] - ip[p] * constraints[6 * k + 2 * cA + alpha];
    constraints[6 * k + 2 * cB + alpha] = val;
    atomicAdd(&normSq[3 * p + cB], val * val);
}

__global__ void forceInnerProductKernel(int numVertices, int* shapeId, const double* constraints, const double* force, double* fp) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVertices * 2) return;
    int k = idx / 2;
    int alpha = idx % 2;
    int p = shapeId[k];
    for (int c = 0; c < 3; c++) atomicAdd(&fp[3 * p + c], constraints[6 * k + 2 * c + alpha] * force[idx]);
}

__global__ void forceProjectionKernel(int numVertices, int* shapeId, const double* constraints, double* force, const double* fp) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVertices * 2) return;
    int k = idx / 2;
    int alpha = idx % 2;
    int p = shapeId[k];
    for (int c = 0; c < 3; c++) force[idx] -= fp[3 * p + c] * constraints[6 * k + 2 * c + alpha];
}


// ---------------------------------------------------------------------------
// n+1 individual constraint force projection
// ---------------------------------------------------------------------------

__global__ void buildEdgeGradMatrixKernel(
        int numVertices, const int* shapeId, const int* startIndices,
        const int* next, const double* positions, double* edgeGradTMP, int n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= numVertices) return;
    int p = shapeId[k];
    int k_loc = k - startIndices[p];
    double dx = wrap(positions[2*next[k]]   - positions[2*k]);
    double dy = wrap(positions[2*next[k]+1] - positions[2*k+1]);
    double len = sqrt(dx*dx + dy*dy);
    if (len < 1e-14) return;
    double tx = dx / len, ty = dy / len;
    double* col = edgeGradTMP + (long long)p * 2*n*n + (long long)k_loc * 2*n;
    col[2*k_loc]               = -tx;
    col[2*k_loc + 1]           = -ty;
    col[2*((k_loc+1)%n)]       =  tx;
    col[2*((k_loc+1)%n) + 1]   =  ty;
}

// One block per polygon. blockDim.x must be >= 2n.
// smem layout: [0..2n-1] = area gradient loaded from constraints,
//              then overwritten with v after projection.
__global__ void gramSchmidtAreaKernel(
        int numPolygons, int n,
        const int* startIndices, const int* shapeId,
        const double* uMat,         // Q: (2n x n) per polygon, stride 2n*n
        const double* constraints,  // normalized area gradient at [6*k+0,1]
        double* qAreaVec) {         // output: 2 doubles per vertex
    extern __shared__ double smem[];
    int p = blockIdx.x;
    if (p >= numPolygons) return;
    int start = startIndices[p];
    int tid = threadIdx.x;

    // Load normalized area gradient — direction is all that matters for GS
    for (int i = tid; i < 2*n; i += blockDim.x) {
        int k = start + i/2;
        int alpha = i % 2;
        smem[i] = constraints[6*k + alpha];
    }
    __syncthreads();

    // Project out each column of Q: smem -= (Q[:,j] . smem) * Q[:,j]
    const double* Q = uMat + (long long)p * 2*n*n;
    for (int j = 0; j < n; j++) {
        // Reduce: dot = Q[:,j] . smem
        double dot = 0.0;
        for (int i = tid; i < 2*n; i += blockDim.x) dot += Q[j*2*n + i] * smem[i];
        // Warp + block reduction
        for (int off = warpSize/2; off > 0; off >>= 1) dot += __shfl_down_sync(0xffffffff, dot, off);
        if (tid % warpSize == 0) smem[2*n + tid/warpSize] = dot;
        __syncthreads();
        if (tid == 0) {
            double sum = 0.0;
            for (int w = 0; w < (blockDim.x + warpSize - 1)/warpSize; w++) sum += smem[2*n + w];
            smem[2*n] = sum;
        }
        __syncthreads();
        dot = smem[2*n];
        for (int i = tid; i < 2*n; i += blockDim.x) smem[i] -= dot * Q[j*2*n + i];
        __syncthreads();
    }

    // Normalize
    double mag = 0.0;
    for (int i = tid; i < 2*n; i += blockDim.x) mag += smem[i] * smem[i];
    for (int off = warpSize/2; off > 0; off >>= 1) mag += __shfl_down_sync(0xffffffff, mag, off);
    if (tid % warpSize == 0) smem[2*n + tid/warpSize] = mag;
    __syncthreads();
    if (tid == 0) {
        double sum = 0.0;
        for (int w = 0; w < (blockDim.x + warpSize - 1)/warpSize; w++) sum += smem[2*n + w];
        smem[2*n] = sqrt(sum);
    }
    __syncthreads();
    double invMag = (smem[2*n] > 1e-14) ? 1.0 / smem[2*n] : 0.0;
    for (int i = tid; i < 2*n; i += blockDim.x)
        qAreaVec[2*start + i] = smem[i] * invMag;
}

// One block per polygon. blockDim.x must be >= 2n.
// smem layout: [0..2n-1] = force slice, [2n..2n+n-1] = fp scalars.
__global__ void forceProjectFullKernel(
        int numPolygons, int n,
        const int* startIndices,
        const double* uMat,     // Q: (2n x n) per polygon, stride 2n*n
        const double* qAreaVec, // 2 doubles per vertex
        double* force) {
    extern __shared__ double smem[];
    int p = blockIdx.x;
    if (p >= numPolygons) return;
    int start = startIndices[p];
    int tid = threadIdx.x;

    // Load force slice into smem
    for (int i = tid; i < 2*n; i += blockDim.x) smem[i] = force[2*start + i];
    __syncthreads();

    // Remove each edge constraint direction
    const double* Q = uMat + (long long)p * 2*n*n;
    for (int j = 0; j < n; j++) {
        double dot = 0.0;
        for (int i = tid; i < 2*n; i += blockDim.x) dot += Q[j*2*n + i] * smem[i];
        for (int off = warpSize/2; off > 0; off >>= 1) dot += __shfl_down_sync(0xffffffff, dot, off);
        if (tid % warpSize == 0) smem[2*n + tid/warpSize] = dot;
        __syncthreads();
        if (tid == 0) {
            double sum = 0.0;
            for (int w = 0; w < (blockDim.x + warpSize - 1)/warpSize; w++) sum += smem[2*n + w];
            smem[2*n] = sum;
        }
        __syncthreads();
        dot = smem[2*n];
        for (int i = tid; i < 2*n; i += blockDim.x) smem[i] -= dot * Q[j*2*n + i];
        __syncthreads();
    }

    // Remove area constraint direction
    double dot = 0.0;
    for (int i = tid; i < 2*n; i += blockDim.x) dot += qAreaVec[2*start + i] * smem[i];
    for (int off = warpSize/2; off > 0; off >>= 1) dot += __shfl_down_sync(0xffffffff, dot, off);
    if (tid % warpSize == 0) smem[2*n + tid/warpSize] = dot;
    __syncthreads();
    if (tid == 0) {
        double sum = 0.0;
        for (int w = 0; w < (blockDim.x + warpSize - 1)/warpSize; w++) sum += smem[2*n + w];
        smem[2*n] = sum;
    }
    __syncthreads();
    dot = smem[2*n];
    for (int i = tid; i < 2*n; i += blockDim.x) smem[i] -= dot * qAreaVec[2*start + i];

    // Write back
    for (int i = tid; i < 2*n; i += blockDim.x) force[2*start + i] = smem[i];
}

// ---------------------------------------------------------------------------
// XPBD position-space constraint projection
// ---------------------------------------------------------------------------

// Project edges of a single color (0 = even local indices, 1 = odd) back to
// their target length.  For even n, within each color no two active edges
// share a vertex, so direct (non-atomic) writes are safe.
__global__ void xpbdEdgeProjectKernel(
        int numVertices, const int* startIndices, const int* shapeId,
        const int* next, double* positions,
        const double* targetEdgeLengths, int color) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= numVertices) return;
    int p = shapeId[k];
    if ((k - startIndices[p]) % 2 != color) return;

    int nk = next[k];
    double ex = wrap(positions[2*nk]   - positions[2*k]);
    double ey = wrap(positions[2*nk+1] - positions[2*k+1]);
    double len = sqrt(ex*ex + ey*ey);
    if (len < 1e-14) return;

    double c = 0.5 * (len - targetEdgeLengths[p]) / len;
    positions[2*k]     += c * ex;
    positions[2*k+1]   += c * ey;
    positions[2*nk]    -= c * ex;
    positions[2*nk+1]  -= c * ey;
}

// Pass 1 of area projection: accumulate current signed area and ||grad A||^2
// per polygon using atomic adds into pre-zeroed scratch buffers.
__global__ void xpbdAreaReductionKernel(
        int numVertices, const int* shapeId, const int* startIndices,
        const int* next, const int* prev, const double* positions,
        double* d_area, double* d_gradNormSq) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= numVertices) return;
    int p  = shapeId[k];
    int s  = startIndices[p];
    int nk = next[k], pk = prev[k];

    // MIC-wrapped positions relative to polygon start vertex
    double xs  = positions[2*s],     ys  = positions[2*s+1];
    double xk  = wrap(positions[2*k]   - xs), yk  = wrap(positions[2*k+1]   - ys);
    double xnk = wrap(positions[2*nk]  - xs), ynk = wrap(positions[2*nk+1]  - ys);
    double xpk = wrap(positions[2*pk]  - xs), ypk = wrap(positions[2*pk+1]  - ys);

    // Shoelace contribution: 0.5*(x_k * y_{k+1} - x_{k+1} * y_k)
    atomicAdd(&d_area[p], 0.5 * (xk * ynk - xnk * yk));

    // Gradient: dA/dx_k = 0.5*(y_{k+1}' - y_{k-1}'), dA/dy_k = 0.5*(x_{k-1}' - x_{k+1}')
    double gx = 0.5 * (ynk - ypk);
    double gy = 0.5 * (xpk - xnk);
    atomicAdd(&d_gradNormSq[p], gx*gx + gy*gy);
}

// Pass 2 of area projection: apply correction delta_x_k = -alpha * grad_k A
// where alpha = (A - A0) / ||grad A||^2.
__global__ void xpbdAreaCorrectionKernel(
        int numVertices, const int* shapeId, const int* startIndices,
        const int* next, const int* prev, double* positions,
        const double* d_area, const double* d_gradNormSq,
        const double* targetAreas) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= numVertices) return;
    int p  = shapeId[k];
    int s  = startIndices[p];
    int nk = next[k], pk = prev[k];

    double alpha = (d_area[p] - targetAreas[p]) / fmax(d_gradNormSq[p], 1e-24);

    double xs  = positions[2*s],     ys  = positions[2*s+1];
    double xnk = wrap(positions[2*nk]  - xs), ynk = wrap(positions[2*nk+1]  - ys);
    double xpk = wrap(positions[2*pk]  - xs), ypk = wrap(positions[2*pk+1]  - ys);
    double gx  = 0.5 * (ynk - ypk);
    double gy  = 0.5 * (xpk - xnk);

    positions[2*k]   -= alpha * gx;
    positions[2*k+1] -= alpha * gy;
}

__global__ void xpbdEdgeDeviationKernel(
        int numVertices, const int* shapeId, const int* next,
        const double* positions, const double* targetEdgeLengths, double* dev) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= numVertices) return;
    int p = shapeId[k];
    int nk = next[k];
    double ex = wrap(positions[2*nk]   - positions[2*k]);
    double ey = wrap(positions[2*nk+1] - positions[2*k+1]);
    dev[k] = fabs(sqrt(ex*ex + ey*ey) - targetEdgeLengths[p]);
}

__global__ void xpbdAreaDeviationKernel(
        int numPolygons, const double* area, const double* targetAreas, double* dev) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= numPolygons) return;
    dev[p] = fabs(area[p] - targetAreas[p]);
}

__global__ void effectiveForceMagKernel(
        int numVertices, const double* positions, const double* tentPos,
        const double* force, double scale, double* mag) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= numVertices) return;
    double fx = force[2*k]   + scale * (positions[2*k]   - tentPos[2*k]);
    double fy = force[2*k+1] + scale * (positions[2*k+1] - tentPos[2*k+1]);
    mag[k] = sqrt(fx*fx + fy*fy);
}

__global__ void normalizeKernel(int numPolygons, double* comX, double* comY, double* positions, int* startIndices) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPolygons) return;
    int n = startIndices[i + 1] - startIndices[i];
    int startIdx = startIndices[i];
    double comx = comX[i];
    comx /= ((double) n);
    comx += positions[2 * startIdx];
    comx -= floor(comx);
    comX[i] = comx;
    double comy = comY[i];
    comy /= ((double) n);
    comy += positions[2 * startIdx + 1];
    comy -= floor(comy);
    comY[i] = comy;
}

__global__ void updateShapeIdKernel(int* shapeId, int* startIndices, int numPolygons) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPolygons) {
        for (int i = startIndices[idx]; i < startIndices[idx + 1]; i++) {
            shapeId[i] = idx;
        }
    }
}

__global__ void updateNeighborCellsKernel(double* positions, int* startIndices, int* shapeId, int numPolygons, int size, int boxSize, int* cellLocation) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int v2 = idx + 1;
        if (idx == size - 1 || shapeId[idx] != shapeId[idx + 1]) {
            v2 = startIndices[shapeId[idx]];
        }
        double x = positions[v2 * 2] - positions[idx * 2] + 0.5;
        double y = positions[v2 * 2 + 1] - positions[idx * 2 + 1] + 0.5;
        while (x < 0.0) {
            x += 1.0;
        }
        while (y < 0.0) {
            y += 1.0;
        }
        while (x >= 1.0) {
            x -= 1.0;
        }
        while (y >= 1.0) {
            y -= 1.0;
        }
        x -= 0.5;
        y -= 0.5;
        x /= 2.0;
        y /= 2.0;
        x += positions[idx * 2];
        y += positions[idx * 2 + 1];
        while (x < 0.0) {
            x += 1.0;
        }
        while (y < 0.0) {
            y += 1.0;
        }
        while (x >= 1.0) {
            x -= 1.0;
        }
        while (y >= 1.0) {
            y -= 1.0;
        }
        int ix = (int)floor(x * boxSize);
        int iy = (int)floor(y * boxSize);
        // Clamp: NaN/Inf positions produce INT_MIN from (int); clamp to valid cell range.
        if (ix < 0) ix = 0; else if (ix >= boxSize) ix = boxSize - 1;
        if (iy < 0) iy = 0; else if (iy >= boxSize) iy = boxSize - 1;
        cellLocation[idx] = iy * boxSize + ix;
    }
}

// -----------------------------------------------------------------------------
// Ball-style (Verlet) neighbor list: per-vertex flat array of vertex ids within
// ball_radius = searchFactor * maxEdgeLength. Rebuilt only when max displacement
// since last build exceeds (ball_radius - 2*maxEdgeLength)/2 (the standard MD
// skin rule), so it carries across many force evaluations.
//
// Brute-force O(N^2) build. Replaces the cell-grid's 3x3 candidate stencil for
// kernels that have a ball-iteration variant.
// -----------------------------------------------------------------------------
__global__ void buildBallListKernel(
    const double* __restrict__ positions,
    const int*    __restrict__ shapeId,
    int   numVertices,
    double ballRadius2,
    int   ballMaxNeighbors,
    int*  __restrict__ ballNeighbors,
    int*  __restrict__ numBallNeighbors)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVertices) return;

    double xi = positions[2*i];
    double yi = positions[2*i+1];
    int    sA = shapeId[i];

    int count = 0;
    int writeBase = i * ballMaxNeighbors;
    for (int j = 0; j < numVertices; j++) {
        if (j == i) continue;
        if (shapeId[j] == sA) continue;     // skip own polygon (matches cell-iter convention)
        double dx = wrap(positions[2*j]   - xi);
        double dy = wrap(positions[2*j+1] - yi);
        if (dx*dx + dy*dy > ballRadius2) continue;
        if (count < ballMaxNeighbors) {
            ballNeighbors[writeBase + count] = j;
        }
        count++;     // always count; overflow detected by host via max-reduce
    }
    numBallNeighbors[i] = count;
}

__global__ void ballMaxDisplacementSqKernel(
    const double* __restrict__ positions,
    const double* __restrict__ refPositions,
    int   numVertices,
    double* __restrict__ dispSq)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVertices) return;
    double dx = wrap(positions[2*i]   - refPositions[2*i]);
    double dy = wrap(positions[2*i+1] - refPositions[2*i+1]);
    dispSq[i] = dx*dx + dy*dy;
}

// Convert the cell grid into the same per-vertex flat candidate list that
// buildBallListKernel produces. After this runs, downstream force kernels
// can iterate ballNeighbors/numBallNeighbors regardless of whether candidates
// were discovered by the cell stencil or by a Verlet ball search.
__global__ void populateCandidatesFromCellsKernel(
    const int* __restrict__ shapeId,
    const int* __restrict__ cellLocation,
    const int* __restrict__ neighborIndices,
    const int* __restrict__ countPerBox,
    int   boxSize,
    int   numVertices,
    int   ballMaxNeighbors,
    int*  __restrict__ ballNeighbors,
    int*  __restrict__ numBallNeighbors)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVertices) return;

    int sA = shapeId[i];
    int box = cellLocation[i];
    int bx  = box % boxSize;
    int by  = box / boxSize;
    int nBoxes = boxSize * boxSize;

    int processedBoxes[16];
    int numProcessedBoxes = 0;

    int count = 0;
    int base  = i * ballMaxNeighbors;

    for (int ddx = -1; ddx <= 1; ddx++) {
        int cx = bx + ddx;
        if (cx < 0) cx += boxSize; else if (cx >= boxSize) cx -= boxSize;
        for (int ddy = -1; ddy <= 1; ddy++) {
            int cy = by + ddy;
            if (cy < 0) cy += boxSize; else if (cy >= boxSize) cy -= boxSize;
            int nbox = cy * boxSize + cx;

            bool already = false;
            for (int b = 0; b < numProcessedBoxes; b++)
                if (processedBoxes[b] == nbox) { already = true; break; }
            if (already) continue;
            if (numProcessedBoxes < 16) processedBoxes[numProcessedBoxes++] = nbox;

            int si = countPerBox[nbox];
            int sf = (nbox + 1 < nBoxes) ? countPerBox[nbox + 1] : numVertices;
            for (int idx = si; idx < sf; idx++) {
                int j = neighborIndices[idx];
                if (j == i) continue;
                if (shapeId[j] == sA) continue;     // match ball builder: skip same polygon
                if (count < ballMaxNeighbors) {
                    ballNeighbors[base + count] = j;
                }
                count++;     // always count; overflow detected by host via max-reduce
            }
        }
    }
    numBallNeighbors[i] = count;
}

// =============================================================
// Feature-pair refactor: Phase A (emitFeaturePairsKernel).
//
// Walks the per-vertex ball list and emits feature-pair candidates that
// Phase B will probe for crossings. Each candidate is a pair of feature
// IDs (one feature of polygon A, one of polygon B) encoded via feat::encode.
//
// Canonical ordering by polygon: emit only when shapeId[i] < shapeId[j],
// so each cross-polygon pair is emitted exactly once (from the thread
// associated with the lower-shape vertex). Assumes the ball list is
// symmetric (it is by construction; auto-resize handles overflow).
//
// Per (i, j) candidate, emits featureSetSize^2 pairs:
//   featureSetSize = 1 : 1 pair (EDGE x EDGE)         -- normal model
//   featureSetSize = 2 : 4 pairs ({EDGE, ARC1}^2)      -- single-arc rounded
//   featureSetSize = 3 : 9 pairs ({EDGE, ARC1, ARC2}^2) -- biarc
//
// Output slots are claimed via a global atomic counter. Caller pre-allocates
// outPairs with `outCapacity` slots and checks the final count against
// capacity for overflow (resize and retry, like the ball list).
// =============================================================
__global__ void emitFeaturePairsKernel(
    int                  numVertices,
    const int* __restrict__ shapeId,
    const int* __restrict__ ballNeighbors,
    const int* __restrict__ numBallNeighbors,
    int                  ballMaxNeighbors,
    int                  featureSetSize,
    feat::FeaturePair* __restrict__ outPairs,
    int* __restrict__    outCount,
    int                  outCapacity)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVertices) return;

    int sA = shapeId[i];
    int nN = numBallNeighbors[i];
    if (nN > ballMaxNeighbors) nN = ballMaxNeighbors;
    int base = i * ballMaxNeighbors;

    for (int k = 0; k < nN; ++k) {
        int j  = ballNeighbors[base + k];
        int sB = shapeId[j];
        if (sA >= sB) continue;             // canonical: emit each shape-pair once

        // Emit featureSetSize^2 pairs (Cartesian product of i's and j's feature
        // sets). Unroll across small fixed counts so the inner loops fold out
        // when featureSetSize is a compile-time-known small value.
        for (int tA = 0; tA < featureSetSize; ++tA) {
            for (int tB = 0; tB < featureSetSize; ++tB) {
                int slot = atomicAdd(outCount, 1);
                if (slot < outCapacity) {
                    outPairs[slot].fA = feat::encode(i, static_cast<feat::FeatureType>(tA));
                    outPairs[slot].fB = feat::encode(j, static_cast<feat::FeatureType>(tB));
                }
                // Always increment so the host can detect overflow.
            }
        }
    }
}

// =============================================================
// Per-polygon uniform-delta mode (area-preserving rounded).
//
// One radius per polygon (uniform across that polygon's corners), recomputed
// each step from current corner angles so that the rounded polygon area
// stays at its stored target. Concretely, for polygon s with corners
// {phi_k}, define
//
//     S(s) = sum_k [ tan(phi_k/2) - phi_k/2 ].
//
// The total per-corner area loss is delta(s)^2 * S(s). Pinning this to a
// per-polygon target dA_target(s) gives
//
//     delta(s) = sqrt( dA_target(s) / S(s) ).
//
// The target is set once (initPolyDeltaTargetsKernel) from the user's
// nominal delta and the initial S(s); updatePolyDeltaKernel then maintains
// the relation as positions evolve. SHAKE preserves backbone area, so the
// rounded area = backbone area - dA_target stays constant by construction.
// Sharpening a corner increases that corner's S contribution, which shrinks
// delta(s) for the whole polygon -- no runaway pressure toward sharper
// corners.
// =============================================================

// Compute the per-corner area-loss sum S(s) for polygon s. Helper used by
// both init and update kernels. Returns 0.0 if any corner is invalid
// (degenerate edges or reflex).
__device__ inline double computePolygonS(int start, int n,
                                          const double* __restrict__ positions)
{
    double S = 0.0;
    for (int k = 0; k < n; ++k) {
        int kp = (k == 0)     ? (n - 1) : (k - 1);
        int kn = (k == n - 1) ? 0       : (k + 1);
        double px = positions[2*(start + kp)],     py = positions[2*(start + kp) + 1];
        double cx = positions[2*(start + k)],      cy = positions[2*(start + k)  + 1];
        double nx = positions[2*(start + kn)],     ny = positions[2*(start + kn) + 1];
        double ux = wrap(cx - px), uy = wrap(cy - py);
        double vx = wrap(nx - cx), vy = wrap(ny - cy);
        double u_len = sqrt(ux*ux + uy*uy);
        double v_len = sqrt(vx*vx + vy*vy);
        if (u_len < 1e-14 || v_len < 1e-14) continue;
        double cr = (ux*vy - uy*vx) / (u_len * v_len);   // sin(phi)
        double dt = (ux*vx + uy*vy) / (u_len * v_len);   // cos(phi)
        double phi = atan2(cr, dt);                       // exterior angle
        if (phi <= 1e-14 || phi >= M_PI - 1e-14) continue;
        S += tan(0.5*phi) - 0.5*phi;
    }
    return S;
}

// One-time setup: given an initial scalar delta, compute the per-polygon
// area-loss target dA_target(s) = delta^2 * S_initial(s).
__global__ void initPolyDeltaTargetsKernel(
    int numPolygons,
    const double* __restrict__ positions,
    const int*    __restrict__ startIndices,
    double deltaInitial,
    double* __restrict__ polyDeltaTargets)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= numPolygons) return;
    int start = startIndices[s];
    int n     = startIndices[s + 1] - start;
    double S  = computePolygonS(start, n, positions);
    polyDeltaTargets[s] = deltaInitial * deltaInitial * S;
}

// Per-step update: given the current backbone state and the stored
// per-polygon targets, compute delta(s) = sqrt(target / S(s)).
__global__ void updatePolyDeltaKernel(
    int numPolygons,
    const double* __restrict__ positions,
    const int*    __restrict__ startIndices,
    const double* __restrict__ polyDeltaTargets,
    double* __restrict__ polyDelta)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= numPolygons) return;
    int start = startIndices[s];
    int n     = startIndices[s + 1] - start;
    double S  = computePolygonS(start, n, positions);
    double t  = polyDeltaTargets[s];
    polyDelta[s] = (S > 1e-15 && t > 0.0) ? sqrt(t / S) : 0.0;
}

// Forward declaration: computeArcGeom is defined later in this file (in the
// rounded-model section). Phase B's per-pair thread calls it to derive arc
// geometry and edge tangent points on-the-fly.
__device__ void computeArcGeom(
    double ax, double ay,
    double ux, double uy,
    double vx, double vy,
    double delta,
    double& uh_x, double& uh_y,
    double& vh_x, double& vh_y,
    double& u_len, double& v_len,
    double& cr, double& dt, double& dphi, double& ell,
    double& am_x, double& am_y,
    double& ap_x, double& ap_y,
    double& zx,   double& zy,
    double& phi0);

// =============================================================
// Feature-pair refactor: Phase B (detectCrossingsKernel) -- installment 2.
//
// Per feature-pair thread. Decodes the two feature IDs and dispatches one of
// the four type combos:
//   (EDGE, EDGE) -> isect::edgeEdgeCross on flat segments between tangent
//                   points (a_v^+ to a_{next(v)}^-)
//   (EDGE, ARC1) -> isect::edgeArcCross
//   (ARC1, EDGE) -> isect::edgeArcCross (with roles swapped)
//   (ARC1, ARC1) -> isect::arcArcCross
//
// Arc geometry (centre, span, tangent points) is computed on-the-fly via
// computeArcGeom; no scratch buffer needed.
//
// Frame convention: both A's and B's geometry is expressed in vA's MIC frame.
// That is, vA's absolute position is used as the anchor for vA's arc/edge
// derivation; vB is first shifted to its image nearest vA, and the same
// derivation is then run from that shifted anchor. This lets the intersection
// routines treat all input coordinates as if they live in a single
// non-periodic frame.
// =============================================================

// Compute arc geometry at vertex v using `anchor*` as the effective absolute
// position of v in the desired output frame. Neighbour vertex offsets are
// MIC-wrapped relative to v's true absolute position (frame-independent
// since wrap operates on differences), so the output (z, am, ap, etc.) is
// in the anchor frame.
__device__ inline void computeArcAtVertex(
    int v, double anchorX, double anchorY,
    const double* positions, const int* next, const int* prev,
    double delta,
    double& zx, double& zy, double& phi0, double& dphi,
    double& amx, double& amy, double& apx, double& apy)
{
    int nv = next[v];
    int pv = prev[v];
    double vAbsX = positions[2*v], vAbsY = positions[2*v+1];
    double ux = wrap(vAbsX - positions[2*pv]);
    double uy = wrap(vAbsY - positions[2*pv+1]);
    double vx = wrap(positions[2*nv]   - vAbsX);
    double vy = wrap(positions[2*nv+1] - vAbsY);

    double uh_x, uh_y, vh_x, vh_y, u_len, v_len, cr, dt, ell;
    computeArcGeom(anchorX, anchorY, ux, uy, vx, vy, delta,
                   uh_x, uh_y, vh_x, vh_y, u_len, v_len,
                   cr, dt, dphi, ell,
                   amx, amy, apx, apy, zx, zy, phi0);
}

// Return the flat-segment edge endpoints (a_v^+ and a_{next(v)}^-) for the
// EDGE feature at vertex v, expressed in the anchor frame for v.
__device__ inline void getFlatEdgeEndpoints(
    int v, double anchorX, double anchorY,
    const double* positions, const int* next, const int* prev,
    double delta,
    double& px, double& py, double& qx, double& qy)
{
    // Arc at v -> a_v^+
    double zx_v, zy_v, phi0_v, dphi_v, amx_v, amy_v, apx_v, apy_v;
    computeArcAtVertex(v, anchorX, anchorY, positions, next, prev, delta,
                       zx_v, zy_v, phi0_v, dphi_v,
                       amx_v, amy_v, apx_v, apy_v);
    px = apx_v;  py = apy_v;

    // Anchor for next(v): shift along the MIC-wrapped edge vector from v.
    int nv = next[v];
    double vAbsX = positions[2*v], vAbsY = positions[2*v+1];
    double offX  = wrap(positions[2*nv]   - vAbsX);
    double offY  = wrap(positions[2*nv+1] - vAbsY);
    double nvAnchorX = anchorX + offX;
    double nvAnchorY = anchorY + offY;

    // Arc at next(v) -> a_{next(v)}^-
    double zx_n, zy_n, phi0_n, dphi_n, amx_n, amy_n, apx_n, apy_n;
    computeArcAtVertex(nv, nvAnchorX, nvAnchorY, positions, next, prev, delta,
                       zx_n, zy_n, phi0_n, dphi_n,
                       amx_n, amy_n, apx_n, apy_n);
    qx = amx_n;  qy = amy_n;
}

__global__ void detectCrossingsKernel(
    int                                       numPairs,
    const feat::FeaturePair* __restrict__     pairs,
    const double*            __restrict__     positions,
    const int*               __restrict__     next,
    const int*               __restrict__     prev,
    const int*               __restrict__     shapeId,
    const double*            __restrict__     polyDelta,
    feat::Crossing*          __restrict__     outCrossings,
    int*                     __restrict__     outCount,
    int                                       outCapacity)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPairs) return;

    feat::FeaturePair p = pairs[idx];
    int vA = feat::vertexOf(p.fA);
    int vB = feat::vertexOf(p.fB);
    feat::FeatureType tA = feat::typeOf(p.fA);
    feat::FeatureType tB = feat::typeOf(p.fB);

    // Phase 2 supports EDGE and ARC1; ARC2 dispatch lands with biarc (Phase 3).
    if (tA == feat::ARC2 || tB == feat::ARC2) return;

    // Per-polygon radius for the arc/tangent-point computations: each side
    // uses its own polygon's delta. For uniform-delta runs polyDelta[s] is
    // the same value for every polygon and this reduces to the scalar case.
    double deltaA = polyDelta[shapeId[vA]];
    double deltaB = polyDelta[shapeId[vB]];

    // Frames: anchor A at vA's absolute position; B at vB's image nearest vA.
    double anchorAx = positions[2*vA];
    double anchorAy = positions[2*vA+1];
    double offBx    = wrap(positions[2*vB]   - anchorAx);
    double offBy    = wrap(positions[2*vB+1] - anchorAy);
    double anchorBx = anchorAx + offBx;
    double anchorBy = anchorAy + offBy;

    // Emit a single crossing record into a slot claimed by atomicAdd.
    auto emit = [&](unsigned int fa, unsigned int fb, float pa, float pb,
                     double Xx, double Xy) {
        int slot = atomicAdd(outCount, 1);
        if (slot < outCapacity) {
            feat::Crossing c;
            c.fA = fa; c.fB = fb;
            c.paramA = pa; c.paramB = pb;
            c.Xx = Xx; c.Xy = Xy;
            outCrossings[slot] = c;
        }
    };

    // -------- Dispatch on (tA, tB). --------
    if (tA == feat::EDGE && tB == feat::EDGE) {
        double pAx, pAy, qAx, qAy;
        double pBx, pBy, qBx, qBy;
        getFlatEdgeEndpoints(vA, anchorAx, anchorAy, positions, next, prev, deltaA,
                             pAx, pAy, qAx, qAy);
        getFlatEdgeEndpoints(vB, anchorBx, anchorBy, positions, next, prev, deltaB,
                             pBx, pBy, qBx, qBy);

        double rx = qAx - pAx, ry = qAy - pAy;
        double sx = qBx - pBx, sy = qBy - pBy;
        double gx = pBx - pAx, gy = pBy - pAy;
        isect::EdgeEdgeResult cr = isect::edgeEdgeCross(rx, ry, sx, sy, gx, gy, 0.0);
        if (cr.crossed) {
            double Xx = pAx + cr.tt * rx;
            double Xy = pAy + cr.tt * ry;
            emit(p.fA, p.fB, (float)cr.tt, (float)cr.uu, Xx, Xy);
        }
    }
    else if (tA == feat::EDGE && tB == feat::ARC1) {
        double pAx, pAy, qAx, qAy;
        getFlatEdgeEndpoints(vA, anchorAx, anchorAy, positions, next, prev, deltaA,
                             pAx, pAy, qAx, qAy);
        double zBx, zBy, phi0B, dphiB, amBx, amBy, apBx, apBy;
        computeArcAtVertex(vB, anchorBx, anchorBy, positions, next, prev, deltaB,
                           zBx, zBy, phi0B, dphiB, amBx, amBy, apBx, apBy);
        isect::ArcCrossingPair r = isect::edgeArcCross(pAx, pAy, qAx, qAy,
                                                       zBx, zBy, deltaB,
                                                       phi0B, dphiB);
        for (int k = 0; k < r.count; ++k) {
            emit(p.fA, p.fB, (float)r.hits[k].tEdge, (float)r.hits[k].tauArc,
                 r.hits[k].Xx, r.hits[k].Xy);
        }
    }
    else if (tA == feat::ARC1 && tB == feat::EDGE) {
        double zAx, zAy, phi0A, dphiA, amAx, amAy, apAx, apAy;
        computeArcAtVertex(vA, anchorAx, anchorAy, positions, next, prev, deltaA,
                           zAx, zAy, phi0A, dphiA, amAx, amAy, apAx, apAy);
        double pBx, pBy, qBx, qBy;
        getFlatEdgeEndpoints(vB, anchorBx, anchorBy, positions, next, prev, deltaB,
                             pBx, pBy, qBx, qBy);
        // Call edgeArcCross with the edge as first feature, arc as second;
        // swap the param fields when emitting so paramA tracks the A-feature.
        isect::ArcCrossingPair r = isect::edgeArcCross(pBx, pBy, qBx, qBy,
                                                       zAx, zAy, deltaA,
                                                       phi0A, dphiA);
        for (int k = 0; k < r.count; ++k) {
            emit(p.fA, p.fB, (float)r.hits[k].tauArc, (float)r.hits[k].tEdge,
                 r.hits[k].Xx, r.hits[k].Xy);
        }
    }
    else { // ARC1 x ARC1
        double zAx, zAy, phi0A, dphiA, amAx, amAy, apAx, apAy;
        double zBx, zBy, phi0B, dphiB, amBx, amBy, apBx, apBy;
        computeArcAtVertex(vA, anchorAx, anchorAy, positions, next, prev, deltaA,
                           zAx, zAy, phi0A, dphiA, amAx, amAy, apAx, apAy);
        computeArcAtVertex(vB, anchorBx, anchorBy, positions, next, prev, deltaB,
                           zBx, zBy, phi0B, dphiB, amBx, amBy, apBx, apBy);
        isect::ArcCrossingPair r = isect::arcArcCross(zAx, zAy, deltaA, phi0A, dphiA,
                                                       zBx, zBy, deltaB, phi0B, dphiB);
        for (int k = 0; k < r.count; ++k) {
            // arcArcCross returns tauArc (A's parameter) in .tauArc and B's
            // parameter in .tEdge by convention.
            emit(p.fA, p.fB, (float)r.hits[k].tauArc, (float)r.hits[k].tEdge,
                 r.hits[k].Xx, r.hits[k].Xy);
        }
    }
}

// =============================================================
// Feature-pair refactor: Phase C scaffolding (sort + run-lengths).
//
// Phase B emits a flat, unordered array of feat::Crossing records. Phase C
// needs them grouped by A-feature (fA) and, within each group, sorted by
// paramA. That gives one contiguous slice per A-feature for the per-feature
// walk to consume.
//
// Strategy:
//   1. buildCrossingKeysKernel  -- build a 64-bit composite sort key from
//      (fA, paramA): key = (uint64(fA) << 32) | floatBits(paramA). Since
//      paramA is non-negative (in [0,1] for edges, mapped likewise for arcs),
//      its IEEE 754 bit pattern preserves numerical order under unsigned
//      lex sort, so the composite key gives (fA, paramA) lex order.
//   2. CUB DeviceRadixSort::SortPairs(keys -> values=index) yields a
//      permutation. A gather kernel then permutes the crossing array.
//   3. extractFAKernel  -- after sorting, write each crossing's fA into a
//      flat uint32 array. CUB DeviceRunLengthEncode::Encode on that gives
//      (uniqueFA[], lengths[], numUniqueFA).
// =============================================================

__global__ void buildCrossingKeysKernel(
    int numCrossings,
    const feat::Crossing* __restrict__ crossings,
    uint64_t* __restrict__ keysOut,
    uint32_t* __restrict__ idxOut)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numCrossings) return;
    const feat::Crossing& c = crossings[i];
    // paramA is non-negative by construction in detectCrossingsKernel, so
    // raw IEEE 754 bits already sort numerically under unsigned compare.
    uint32_t pbits = __float_as_uint(c.paramA);
    keysOut[i] = (uint64_t(c.fA) << 32) | uint64_t(pbits);
    idxOut[i] = (uint32_t)i;
}

__global__ void gatherCrossingsKernel(
    int numCrossings,
    const feat::Crossing* __restrict__ in,
    const uint32_t* __restrict__ perm,
    feat::Crossing* __restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numCrossings) return;
    out[i] = in[perm[i]];
}

__global__ void extractFAKernel(
    int numCrossings,
    const feat::Crossing* __restrict__ crossings,
    uint32_t* __restrict__ fAOut)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numCrossings) return;
    fAOut[i] = crossings[i].fA;
}

// =============================================================
// Per-A-polygon sort + walk (fixes the cross-feature chord-pair bug).
//
// The per-feature walk misses inside chord pairs whose two crossings sit on
// different A-features (the typical case for non-trivial overlaps). The fix
// is to sort crossings by (sA, globalParamAlongPolygonBoundary) and walk all
// of polygon A's crossings in boundary order, tracking parity per partner
// shape across feature boundaries.
//
// Boundary order at vertex v: ARC at v -> EDGE at v -> ARC at v+1 -> EDGE
// at v+1 -> ...   So globalParam = 2*localVertex + (ARC ? 0 : 1) + paramA.
// =============================================================

// Duplicate each crossing with (fA, fB) swapped. Phase A emits pairs only in
// the canonical sA < sB direction; without this duplication, only the lower-
// id polygon of each pair has crossings in its slice and the per-polygon walk
// is missing the partner polygon's path integral.
__global__ void duplicateCrossingsKernel(
    int                              numCrossings,
    const feat::Crossing* __restrict__ in,
    feat::Crossing*       __restrict__ out)   // size 2 * numCrossings
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numCrossings) return;
    feat::Crossing c = in[i];
    out[i] = c;
    feat::Crossing cs;
    cs.fA = c.fB; cs.fB = c.fA;
    cs.paramA = c.paramB; cs.paramB = c.paramA;
    cs.Xx = c.Xx; cs.Xy = c.Xy;
    out[numCrossings + i] = cs;
}

__global__ void buildPolygonKeysKernel(
    int numCrossings,
    const feat::Crossing* __restrict__ crossings,
    const int* __restrict__ shapeId,
    const int* __restrict__ startIndices,
    uint64_t* __restrict__ keysOut,
    uint32_t* __restrict__ idxOut)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numCrossings) return;
    const feat::Crossing& c = crossings[i];
    int vA = (int)(c.fA >> 2);
    int tA = (int)(c.fA & 3);
    int sA = shapeId[vA];
    int localV = vA - startIndices[sA];
    // ARC at v comes before EDGE at v in boundary order.
    int featureIdx = 2 * localV + ((tA == (int)feat::ARC1) ? 0 : 1);
    float globalParam = (float)featureIdx + c.paramA;
    uint32_t pbits = __float_as_uint(globalParam);
    keysOut[i] = ((uint64_t)(uint32_t)sA << 32) | (uint64_t)pbits;
    idxOut[i] = (uint32_t)i;
}

__global__ void extractShapeIdKernel(
    int numCrossings,
    const feat::Crossing* __restrict__ crossings,
    const int* __restrict__ shapeId,
    uint32_t* __restrict__ shapeOut)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numCrossings) return;
    int vA = (int)(crossings[i].fA >> 2);
    shapeOut[i] = (uint32_t)shapeId[vA];
}

// Per-A-polygon walk. One thread per polygon. Walks crossings in
// (sA, globalParam) order, maintaining a per-partner-shape parity bitmap
// and a per-partner segment-start position. On each crossing:
//   - If parity[partner] was 0 -> 1: this is an entry. Remember position.
//   - If parity[partner] was 1 -> 0: this is an exit. Add chordArea from
//     stored entry to current exit.
// Final per-polygon chord-area sum is half of that polygon's contribution to
// the overlap area integral (the other half comes from the partner's walk).
// Total overlap area = (sum over polygons of areaOut) / 2.
//
// Per-thread state: bool parity[MAX_PARTNERS] + 2 * double segStart[MAX_PARTNERS].
// At MAX_PARTNERS=256 this is ~4.3 KB per thread; the launcher bumps the CUDA
// stack limit to 16 KB to accommodate it.
#define PHASEC_MAX_PARTNERS 256

// Compute the tangent points am, ap at a polygon vertex v given delta and
// neighbor vertex positions (matches the existing computeArcGeom convention).
__device__ inline void tangentPointsAtVertex(
    int v, int sA,
    const int* startIndices,
    const double* positions,
    double delta,
    int polygonN,
    double2& am, double2& ap)
{
    int start = startIndices[sA];
    int localV = v - start;
    int prevV = start + ((localV - 1 + polygonN) % polygonN);
    int nextV = start + ((localV + 1) % polygonN);
    double pvx = positions[2*prevV], pvy = positions[2*prevV + 1];
    double vx  = positions[2*v],     vy  = positions[2*v + 1];
    double nvx = positions[2*nextV], nvy = positions[2*nextV + 1];
    double ux = wrap(vx - pvx),  uy = wrap(vy - pvy);
    double dx = wrap(nvx - vx),  dy = wrap(nvy - vy);
    double u_len = sqrt(ux*ux + uy*uy);
    double v_len = sqrt(dx*dx + dy*dy);
    if (u_len < 1e-14 || v_len < 1e-14) {
        am.x = vx; am.y = vy; ap.x = vx; ap.y = vy; return;
    }
    double uhx = ux/u_len, uhy = uy/u_len;
    double vhx = dx/v_len, vhy = dy/v_len;
    double cr = uhx*vhy - uhy*vhx;
    double dt = uhx*vhx + uhy*vhy;
    double abs_cr = fabs(cr);
    double denom = 1.0 + dt;
    double ell = (abs_cr > 1e-14 && denom > 1e-14) ? delta * abs_cr / denom : 0.0;
    am.x = vx - ell*uhx;  am.y = vy - ell*uhy;
    ap.x = vx + ell*vhx;  ap.y = vy + ell*vhy;
}

__global__ void phaseCWalkAreaPerPolygonKernel(
    int                              numUniqueShapes,
    const uint32_t*  __restrict__    uniqueShape,
    const uint32_t*  __restrict__    lengths,
    const uint32_t*  __restrict__    sliceStart,
    const feat::Crossing* __restrict__ crossings,
    const int*       __restrict__    shapeId,
    const int*       __restrict__    startIndices,
    const double*    __restrict__    positions,
    const double*    __restrict__    polyDelta,
    double*          __restrict__    areaOut)
{
    int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sIdx >= numUniqueShapes) return;

    int sA = (int)uniqueShape[sIdx];
    int start = (int)sliceStart[sIdx];
    int n     = (int)lengths[sIdx];
    int polyStart = startIndices[sA];
    int polygonN = startIndices[sA + 1] - polyStart;
    double deltaA = polyDelta[sA];

    bool parity[PHASEC_MAX_PARTNERS];
    for (int p = 0; p < PHASEC_MAX_PARTNERS; ++p) parity[p] = false;

    // We walk all events in boundary order: feature boundaries (am, ap at
    // each vertex) interleaved with the polygon's crossings. At every
    // consecutive event pair, for every active-parity partner, we accumulate
    // chord-area with that partner's anchor.
    //
    // Boundary order = ARC at v0 (param [0,1]) -> EDGE at v0 (param [1,2])
    // -> ARC at v1 (param [2,3]) -> ... -> ARC at v_{n-1} -> EDGE at v_{n-1}
    // -> wrap. So integer globalParam k corresponds to:
    //   k even (k=2v):     am(v)  (entry tangent of ARC at vertex v)
    //   k odd  (k=2v+1):   ap(v)  (exit tangent of ARC at vertex v, also
    //                              entry of EDGE at vertex v)
    //
    // Algorithm:
    //   Set last_event = (am(v0), globalParam=0).
    //   For each crossing K in slice (sorted by globalParam ascending):
    //     For each integer g in [floor(last_event.gp)+1 .. floor(K.gp)] :
    //       Add chord-area from last_event.pos to vertex_event(g).
    //       last_event = vertex_event(g)
    //     Add chord-area from last_event to K.
    //     last_event = K, toggle parity[K.partner]
    //   After loop: walk remaining vertex events through global wrap back to 0.
    //
    // For active-parity accumulation we use the anchor of min(sA, sB).

    auto vertexEventPos = [&](int g) -> double2 {
        int gMod = ((g % (2 * polygonN)) + 2 * polygonN) % (2 * polygonN);
        int v = polyStart + (gMod / 2);
        double2 am, ap;
        tangentPointsAtVertex(v, sA, startIndices, positions, deltaA, polygonN, am, ap);
        if ((gMod & 1) == 0) return am;
        return ap;
    };

    auto addActiveChord = [&](double2 P0, double2 P1) {
        // NOTE: a sign correction based on (B > sA) helps the 64-polygon dense
        // case (~94% match) but breaks the simple 2-polygon case (which works
        // ~100% without the sign). The relationship between each polygon's
        // CCW direction and the CCW direction around the OVERLAP depends on
        // the specific geometry (which polygon contains which), and isn't
        // captured by a simple comparison of polygon IDs. Leaving unsigned
        // for now; the right fix likely requires per-pair winding-number
        // analysis or a different formulation entirely.
        for (int B = 0; B < PHASEC_MAX_PARTNERS; ++B) {
            if (!parity[B]) continue;
            int sRef = (sA < B) ? sA : B;
            double2 anchor;
            anchor.x = positions[2 * startIndices[sRef]];
            anchor.y = positions[2 * startIndices[sRef] + 1];
            areaOut[sIdx] += integrate::chordArea(P0, P1, anchor);
        }
    };

    areaOut[sIdx] = 0.0;

    if (n == 0) return;

    // Initialize: read the first crossing to know where to start the walk.
    feat::Crossing c0 = crossings[start];
    int vA0_global = (int)(c0.fA >> 2);
    int tA0 = (int)(c0.fA & 3);
    int localV0 = vA0_global - polyStart;
    int firstFeatureIdx = 2 * localV0 + ((tA0 == (int)feat::ARC1) ? 0 : 1);
    float firstGlobalParam = (float)firstFeatureIdx + c0.paramA;

    // Walk forward through the slice. Each integer globalParam between
    // consecutive events is a vertex event.
    double2 last_pos = vertexEventPos(0);
    double last_gp = 0.0;

    for (int i = 0; i < n; ++i) {
        feat::Crossing c = crossings[start + i];
        int vA_global = (int)(c.fA >> 2);
        int tA = (int)(c.fA & 3);
        int localV = vA_global - polyStart;
        int featureIdx = 2 * localV + ((tA == (int)feat::ARC1) ? 0 : 1);
        float gp = (float)featureIdx + c.paramA;

        // Walk integer ticks in (last_gp, gp]
        int gStart = (int)floor((double)last_gp) + 1;
        int gEnd   = (int)floor((double)gp);
        for (int g = gStart; g <= gEnd; ++g) {
            double2 vp = vertexEventPos(g);
            addActiveChord(last_pos, vp);
            last_pos = vp;
        }
        // From last_pos (now at integer gp = gEnd) to crossing c.
        double2 cpos = make_double2(c.Xx, c.Xy);
        addActiveChord(last_pos, cpos);
        last_pos = cpos;
        last_gp  = gp;

        // Toggle parity for this crossing's partner.
        int vB = (int)(c.fB >> 2);
        int sB = shapeId[vB];
        if (sB >= 0 && sB < PHASEC_MAX_PARTNERS) {
            parity[sB] = !parity[sB];
        }
    }
    // Walk remaining integer ticks from last_gp up through wrap to 2*polygonN.
    int gStart = (int)floor((double)last_gp) + 1;
    int gEnd   = 2 * polygonN;
    for (int g = gStart; g <= gEnd; ++g) {
        double2 vp = vertexEventPos(g);
        addActiveChord(last_pos, vp);
        last_pos = vp;
    }
}

// Phase C, installment 2: per-feature walk -- area-only accumulator.
//
// One thread per unique A-feature. Given the (fA, paramA)-sorted crossing
// array and the RLE output (uniqueFA[], lengths[]) from installment 1, each
// thread walks its A-feature's slice and accumulates the Green's-theorem
// chord-area contribution of every "inside" sub-segment along that feature.
//
// A sub-segment between consecutive same-shapeB crossings is "inside" partner
// shape sB iff its starting crossing is at an even rank within the (fA, sB)
// sub-group (i.e., 0, 2, 4, ... -- an entry/exit pair). The per-shape rank is
// computed in O(n^2) per thread by counting prior same-shape crossings; with
// n <= ~50 per A-feature in practice this is cheap.
//
// Anchor for chord-area: positions of the first vertex of min(sA, sB) -- same
// convention as updateForceEnergyExteriorKernel so the per-pair sum telescopes
// correctly across the periodic box.
//
// Output: one double per unique A-feature (areaOut[fA_slot]).
__global__ void phaseCWalkAreaPerFeatureKernel(
    int                              numUniqueFA,
    const uint32_t*  __restrict__    uniqueFA,
    const uint32_t*  __restrict__    lengths,
    const uint32_t*  __restrict__    sliceStart,    // prefix-sum of lengths
    const feat::Crossing* __restrict__ crossings,
    const int*       __restrict__    shapeId,
    const int*       __restrict__    startIndices,
    const double*    __restrict__    positions,
    double*          __restrict__    areaOut)
{
    int fAIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (fAIdx >= numUniqueFA) return;

    int start = (int)sliceStart[fAIdx];
    int n     = (int)lengths[fAIdx];
    unsigned int fA = uniqueFA[fAIdx];
    int vA = feat::vertexOf(fA);
    int sA = shapeId[vA];

    double chordSum = 0.0;
    for (int k = 1; k < n; ++k) {
        feat::Crossing ck = crossings[start + k];
        int vBk = feat::vertexOf(ck.fB);
        int sB  = shapeId[vBk];

        // Walk backwards to find prior same-shape crossing and rank.
        int rank = 0;
        int prev = -1;
        for (int j = 0; j < k; ++j) {
            feat::Crossing cj = crossings[start + j];
            int sBj = shapeId[feat::vertexOf(cj.fB)];
            if (sBj == sB) { rank++; prev = j; }
        }
        // rank counts same-shape crossings strictly before k. k pairs with
        // the most recent prior same-shape crossing iff rank is odd:
        // (entry@rank-1, exit@rank) -> inside segment.
        if ((rank & 1) == 1 && prev >= 0) {
            feat::Crossing cp = crossings[start + prev];

            int sRef = (sA < sB) ? sA : sB;
            double2 anchor;
            anchor.x = positions[2 * startIndices[sRef]];
            anchor.y = positions[2 * startIndices[sRef] + 1];

            double2 P0 = make_double2(cp.Xx, cp.Xy);
            double2 P1 = make_double2(ck.Xx, ck.Xy);
            chordSum += integrate::chordArea(P0, P1, anchor);
        }
    }
    areaOut[fAIdx] = chordSum;
}

// =============================================================
// Dual-traced tangent-point computation. Inputs are 3 backbone vertices (as
// DPoints with derivatives wired to their slots); outputs are a_v^+ and a_v^-
// (the two tangent points at v) as DPoints carrying full Jacobians w.r.t.
// the 6 input components.
//
// Mirrors computeArcGeom's tangent-point math (no atan2/sin/cos needed since
// we only want am, ap, not phi0/dphi). For arcs themselves, dual versions of
// computeArcGeom etc. would extend this.
// =============================================================
// Dual-traced arc center. Mirrors computeArcGeom's z = a + delta*(vh - uh)/cr
// (using non-abs cr in the original; for convex corners cr > 0 so |cr| = cr).
// Inputs are 3 backbone vertices as DPoints with slot derivatives wired up.
// Output: z as a DPoint with derivatives w.r.t. the same 6 slots.
__device__ static void arcCenter_dual(
    const DPoint& p_prev, const DPoint& p_v, const DPoint& p_next,
    double delta,
    DPoint& z)
{
    Dual ux, uy, vx, vy, t1, t2;
    Dual u_lenSq, v_lenSq, u_len, v_len;
    Dual uhx, uhy, vhx, vhy, cr;
    dual_sub(p_v.x, p_prev.x, t1); dual_wrap_box(t1, ux);
    dual_sub(p_v.y, p_prev.y, t1); dual_wrap_box(t1, uy);
    dual_sub(p_next.x, p_v.x, t1); dual_wrap_box(t1, vx);
    dual_sub(p_next.y, p_v.y, t1); dual_wrap_box(t1, vy);
    dual_mul(ux, ux, t1); dual_mul(uy, uy, t2); dual_add(t1, t2, u_lenSq);
    dual_mul(vx, vx, t1); dual_mul(vy, vy, t2); dual_add(t1, t2, v_lenSq);
    dual_sqrt(u_lenSq, u_len);
    dual_sqrt(v_lenSq, v_len);
    dual_div(ux, u_len, uhx); dual_div(uy, u_len, uhy);
    dual_div(vx, v_len, vhx); dual_div(vy, v_len, vhy);
    dual_mul(uhx, vhy, t1); dual_mul(uhy, vhx, t2); dual_sub(t1, t2, cr);
    // diff = vh - uh
    Dual dxh, dyh;
    dual_sub(vhx, uhx, dxh);
    dual_sub(vhy, uhy, dyh);
    // z = a + delta * diff / cr
    Dual sx, sy;
    dual_scale(dxh, delta, sx);
    dual_scale(dyh, delta, sy);
    Dual qx, qy;
    dual_div(sx, cr, qx);
    dual_div(sy, cr, qy);
    dual_add(p_v.x, qx, z.x);
    dual_add(p_v.y, qy, z.y);
}

__device__ static void tangentPoints_dual(
    const DPoint& p_prev, const DPoint& p_v, const DPoint& p_next,
    double delta,
    DPoint& am, DPoint& ap)
{
    // u_vec = wrap(p_v - p_prev), v_vec = wrap(p_next - p_v)
    Dual ux, uy, vx, vy;
    {
        Dual t;
        dual_sub(p_v.x, p_prev.x, t); dual_wrap_box(t, ux);
        dual_sub(p_v.y, p_prev.y, t); dual_wrap_box(t, uy);
        dual_sub(p_next.x, p_v.x, t); dual_wrap_box(t, vx);
        dual_sub(p_next.y, p_v.y, t); dual_wrap_box(t, vy);
    }
    // u_len, v_len
    Dual u_lenSq, v_lenSq, u_len, v_len, t1, t2;
    dual_mul(ux, ux, t1); dual_mul(uy, uy, t2); dual_add(t1, t2, u_lenSq);
    dual_mul(vx, vx, t1); dual_mul(vy, vy, t2); dual_add(t1, t2, v_lenSq);
    dual_sqrt(u_lenSq, u_len);
    dual_sqrt(v_lenSq, v_len);
    // uh = u / u_len, vh = v / v_len
    Dual uhx, uhy, vhx, vhy;
    dual_div(ux, u_len, uhx); dual_div(uy, u_len, uhy);
    dual_div(vx, v_len, vhx); dual_div(vy, v_len, vhy);
    // cr = uhx*vhy - uhy*vhx, dt = uhx*vhx + uhy*vhy
    Dual cr, dt;
    dual_mul(uhx, vhy, t1); dual_mul(uhy, vhx, t2); dual_sub(t1, t2, cr);
    dual_mul(uhx, vhx, t1); dual_mul(uhy, vhy, t2); dual_add(t1, t2, dt);
    // |cr|, (1 + dt), ell = delta * |cr| / (1 + dt)
    Dual abs_cr, one_plus_dt, ell, num;
    dual_fabs(cr, abs_cr);
    Dual one; dual_const(one, 1.0);
    dual_add(one, dt, one_plus_dt);
    dual_scale(abs_cr, delta, num);
    dual_div(num, one_plus_dt, ell);
    // am = p_v - ell * uh, ap = p_v + ell * vh
    Dual ell_ux, ell_uy, ell_vx, ell_vy;
    dual_mul(ell, uhx, ell_ux); dual_mul(ell, uhy, ell_uy);
    dual_mul(ell, vhx, ell_vx); dual_mul(ell, vhy, ell_vy);
    dual_sub(p_v.x, ell_ux, am.x); dual_sub(p_v.y, ell_uy, am.y);
    dual_add(p_v.x, ell_vx, ap.x); dual_add(p_v.y, ell_vy, ap.y);
}

// =============================================================
// Phase C installment 4: edge-edge force with delta>0 via dual-traced
// tangent points + chain rule through getDf. This is the new force kernel.
//
// For each EDGE-EDGE inside chord pair (P_prev, P_k):
//   1. Compute pA, qA, pB, qB as DPoints (value + 16-dim Jacobian w.r.t. the
//      8 backbone vertices that influence them).
//   2. Run plain getDf on the .v fields to get dP / d(pA, qA, pB, qB) -- the
//      4-chord-endpoint Jacobian (2x8).
//   3. Chain: dP / d(input_k) = sum_col df[col*2+alpha] * (the col-th
//      .v-component's derivative w.r.t. input_k).
//   4. chordGradients gives g1, g2 (the per-endpoint chord-area gradient).
//   5. atomicAdd -g . dPdInput onto the 8 backbone vertices.
// =============================================================

// Dual-traced arc-arc crossing point. Two circles of equal radius delta
// centered at z_A and z_B. Intersection points are at
//   mid + (+/-) half_chord * perp_unit
// where mid = (z_A + z_B)/2, d = z_B - z_A, perp_unit = (-d.y, d.x)/|d|,
// and half_chord = sqrt(delta^2 - (|d|/2)^2). Sign chosen to match the
// stored (Xx_known, Xy_known) so we pick the correct branch.
__device__ static void arcArcCross_dual(
    const DPoint& z_A, const DPoint& z_B,
    double delta,
    double Xx_known, double Xy_known,
    DPoint& P)
{
    Dual dx, dy, d_sq, d_len;
    dual_sub(z_B.x, z_A.x, dx);
    dual_sub(z_B.y, z_A.y, dy);
    Dual tmp1, tmp2;
    dual_mul(dx, dx, tmp1); dual_mul(dy, dy, tmp2); dual_add(tmp1, tmp2, d_sq);
    dual_sqrt(d_sq, d_len);
    Dual mid_x, mid_y;
    dual_add(z_A.x, z_B.x, tmp1); dual_scale(tmp1, 0.5, mid_x);
    dual_add(z_A.y, z_B.y, tmp1); dual_scale(tmp1, 0.5, mid_y);
    // half_chord_sq = delta^2 - 0.25 * d_sq
    Dual delta_sq_const, hcs;
    dual_const(delta_sq_const, delta * delta);
    dual_scale(d_sq, 0.25, tmp1);
    dual_sub(delta_sq_const, tmp1, hcs);
    Dual hc;
    dual_sqrt(hcs, hc);
    // perp = (-dy, dx) / d_len
    Dual perp_x_num, perp_x, perp_y;
    dual_neg(dy, perp_x_num);
    dual_div(perp_x_num, d_len, perp_x);
    dual_div(dx, d_len, perp_y);
    // candidate +: mid + hc * perp
    Dual ex, ey;
    dual_mul(hc, perp_x, tmp1);
    dual_add(mid_x, tmp1, ex);
    dual_mul(hc, perp_y, tmp1);
    dual_add(mid_y, tmp1, ey);
    double dx_plus = ex.v - Xx_known, dy_plus = ey.v - Xy_known;
    double err_plus = dx_plus*dx_plus + dy_plus*dy_plus;
    // candidate -: mid - hc * perp
    Dual fx, fy;
    dual_mul(hc, perp_x, tmp1);
    dual_sub(mid_x, tmp1, fx);
    dual_mul(hc, perp_y, tmp1);
    dual_sub(mid_y, tmp1, fy);
    double dx_minus = fx.v - Xx_known, dy_minus = fy.v - Xy_known;
    double err_minus = dx_minus*dx_minus + dy_minus*dy_minus;
    // Pick the branch whose value matches the stored crossing.
    if (err_plus <= err_minus) {
        P.x = ex; P.y = ey;
    } else {
        P.x = fx; P.y = fy;
    }
}

__device__ static void chainCrossingJacobian(
    const DPoint& pA, const DPoint& qA, const DPoint& pB, const DPoint& qB,
    const double* df,           // 16 doubles, layout df[col*2+alpha]
    double dPdInput[DUAL_N * 2]) // dPdInput[slot*2+alpha]
{
    for (int k = 0; k < DUAL_N; ++k) {
        for (int alpha = 0; alpha < 2; ++alpha) {
            double s = 0.0;
            s += df[0*2+alpha] * pA.x.d[k];
            s += df[1*2+alpha] * pA.y.d[k];
            s += df[2*2+alpha] * qA.x.d[k];
            s += df[3*2+alpha] * qA.y.d[k];
            s += df[4*2+alpha] * pB.x.d[k];
            s += df[5*2+alpha] * pB.y.d[k];
            s += df[6*2+alpha] * qB.x.d[k];
            s += df[7*2+alpha] * qB.y.d[k];
            dPdInput[k*2+alpha] = s;
        }
    }
}

__global__ void phaseCWalkAreaForceDualPerFeatureKernel(
    int                              numUniqueFA,
    const uint32_t*  __restrict__    uniqueFA,
    const uint32_t*  __restrict__    lengths,
    const uint32_t*  __restrict__    sliceStart,
    const feat::Crossing* __restrict__ crossings,
    const int*       __restrict__    shapeId,
    const int*       __restrict__    startIndices,
    const int*       __restrict__    next_arr,
    const int*       __restrict__    prev_arr,
    const double*    __restrict__    positions,
    double                           delta,
    double*          __restrict__    areaOut,
    double*          __restrict__    forceOut)
{
    int fAIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (fAIdx >= numUniqueFA) return;

    int start = (int)sliceStart[fAIdx];
    int n     = (int)lengths[fAIdx];
    unsigned int fA = uniqueFA[fAIdx];
    int vA = feat::vertexOf(fA);
    int sA = shapeId[vA];
    feat::FeatureType tA = feat::typeOf(fA);

    double chordSum = 0.0;
    for (int k = 1; k < n; ++k) {
        feat::Crossing ck = crossings[start + k];
        int vBk = feat::vertexOf(ck.fB);
        int sBk = shapeId[vBk];
        feat::FeatureType tBk = feat::typeOf(ck.fB);

        int rank = 0;
        int pj   = -1;
        for (int j = 0; j < k; ++j) {
            int sBj = shapeId[feat::vertexOf(crossings[start + j].fB)];
            if (sBj == sBk) { rank++; pj = j; }
        }

        if ((rank & 1) == 1 && pj >= 0) {
            feat::Crossing cp = crossings[start + pj];

            int sRef = (sA < sBk) ? sA : sBk;
            double2 anchor;
            anchor.x = positions[2*startIndices[sRef]];
            anchor.y = positions[2*startIndices[sRef] + 1];

            double2 P0 = make_double2(cp.Xx, cp.Xy);
            double2 P1 = make_double2(ck.Xx, ck.Xy);
            chordSum += integrate::chordArea(P0, P1, anchor);

            // Dispatch on (tA, tBp, tBk). ARC-ARC: pure circle-circle, derivatives
            // come directly from arcCenter_dual + arcArcCross_dual. EDGE-EDGE
            // (handled below) chains through getDf. Mixed EDGE/ARC cases skipped
            // here (need implicit-function-theorem treatment; deferred).
            feat::FeatureType tBp = feat::typeOf(cp.fB);
            bool isAA_prev = (tA == feat::ARC1 && tBp == feat::ARC1);
            bool isAA_k    = (tA == feat::ARC1 && tBk == feat::ARC1);
            bool isEE_prev = (tA == feat::EDGE && tBp == feat::EDGE);
            bool isEE_k    = (tA == feat::EDGE && tBk == feat::EDGE);
            // For now, only handle pairs where BOTH crossings are the same type
            // (both EE or both AA). Mixed-type pairs skipped.
            if (!((isEE_prev && isEE_k) || (isAA_prev && isAA_k))) continue;

            double2 g1, g2;
            integrate::chordGradients(P0, P1, anchor, g1, g2);

            // Anchor-vertex contribution: d(chordArea)/d(anchor) = -(g1+g2), so
            // force on the anchor vertex picks up +(g1+g2).
            {
                int anchorV = startIndices[sRef];
                atomicAdd(&forceOut[2*anchorV    ], g1.x + g2.x);
                atomicAdd(&forceOut[2*anchorV + 1], g1.y + g2.y);
            }

            if (isAA_prev) {
                // ===== ARC-ARC: derivatives flow directly through duals. =====
                // Slot layout: A-arc deps in 0-5, B-arc deps in 8-13. Numerical
                // guard: arcArcCross_dual divides by d_len = |z_B - z_A|, which
                // blows up for near-coincident arc centers. Compute the value-
                // only d_len from positions first and skip force (keep chord-
                // area) if the configuration is degenerate (near-coincident or
                // near-tangent circles).
                int vBp = feat::vertexOf(cp.fB);
                int pvA = prev_arr[vA],  nvA  = next_arr[vA];
                int pvBp= prev_arr[vBp], nvBp = next_arr[vBp];

                auto arcCenterValue = [&] (int pv, int v, int nv,
                                             double& zx, double& zy) {
                    double pvX  = positions[2*pv];
                    double pvY  = positions[2*pv + 1];
                    double vX   = positions[2*v];
                    double vY   = positions[2*v + 1];
                    double nvX  = positions[2*nv];
                    double nvY  = positions[2*nv + 1];
                    double ux = wrap(vX - pvX), uy = wrap(vY - pvY);
                    double vx = wrap(nvX - vX), vy = wrap(nvY - vY);
                    double ul = sqrt(ux*ux + uy*uy);
                    double vl = sqrt(vx*vx + vy*vy);
                    double uhx = (ul > 1e-14) ? ux/ul : 0.0;
                    double uhy = (ul > 1e-14) ? uy/ul : 0.0;
                    double vhx = (vl > 1e-14) ? vx/vl : 0.0;
                    double vhy = (vl > 1e-14) ? vy/vl : 0.0;
                    double cr  = uhx*vhy - uhy*vhx;
                    if (fabs(cr) < 1e-14) { zx = vX; zy = vY; return; }
                    zx = vX + delta*(vhx - uhx)/cr;
                    zy = vY + delta*(vhy - uhy)/cr;
                };

                auto safeForAA = [&](int pvA_, int vA_, int nvA_,
                                      int pvB_, int vB_, int nvB_) {
                    double zAx, zAy, zBx, zBy;
                    arcCenterValue(pvA_, vA_, nvA_, zAx, zAy);
                    arcCenterValue(pvB_, vB_, nvB_, zBx, zBy);
                    double dx = zBx - zAx, dy = zBy - zAy;
                    double d_len = sqrt(dx*dx + dy*dy);
                    // Want d_len comfortably > 0 and comfortably < 2*delta.
                    double margin_low  = 0.02 * delta;
                    double margin_high = 1.98 * delta;
                    return (d_len > margin_low) && (d_len < margin_high);
                };

                bool ok_prev = safeForAA(pvA, vA, nvA, pvBp, vBp, nvBp);

                if (ok_prev) {
                    DPoint dPrevA, dvA_, dnvA_, dPrevBp, dvBp, dnvBp;
                    dpoint_var(dPrevA, positions[2*pvA],  positions[2*pvA+1],  0, 1);
                    dpoint_var(dvA_,   positions[2*vA],   positions[2*vA+1],   2, 3);
                    dpoint_var(dnvA_,  positions[2*nvA],  positions[2*nvA+1],  4, 5);
                    dpoint_var(dPrevBp,positions[2*pvBp], positions[2*pvBp+1], 8, 9);
                    dpoint_var(dvBp,   positions[2*vBp],  positions[2*vBp+1],  10, 11);
                    dpoint_var(dnvBp,  positions[2*nvBp], positions[2*nvBp+1], 12, 13);
                    DPoint zA_d, zBp_d, Pprev_d;
                    arcCenter_dual(dPrevA,  dvA_, dnvA_, delta, zA_d);
                    arcCenter_dual(dPrevBp, dvBp, dnvBp, delta, zBp_d);
                    arcArcCross_dual(zA_d, zBp_d, delta, cp.Xx, cp.Xy, Pprev_d);

                    int vlist_prev[6] = { pvA, vA, nvA, pvBp, vBp, nvBp };
                    int slot_off[6]   = { 0, 2, 4, 8, 10, 12 };
                    for (int s = 0; s < 6; ++s) {
                        int sx = slot_off[s], sy = slot_off[s] + 1;
                        double incx = g1.x * Pprev_d.x.d[sx] + g1.y * Pprev_d.y.d[sx];
                        double incy = g1.x * Pprev_d.x.d[sy] + g1.y * Pprev_d.y.d[sy];
                        atomicAdd(&forceOut[2*vlist_prev[s]    ], -incx);
                        atomicAdd(&forceOut[2*vlist_prev[s] + 1], -incy);
                    }
                }

                int pvBk = prev_arr[vBk], nvBk = next_arr[vBk];
                bool ok_k = safeForAA(pvA, vA, nvA, pvBk, vBk, nvBk);

                if (ok_k) {
                    DPoint dPrevA, dvA_, dnvA_, dPrevBk, dvBk, dnvBk;
                    dpoint_var(dPrevA, positions[2*pvA],  positions[2*pvA+1],  0, 1);
                    dpoint_var(dvA_,   positions[2*vA],   positions[2*vA+1],   2, 3);
                    dpoint_var(dnvA_,  positions[2*nvA],  positions[2*nvA+1],  4, 5);
                    dpoint_var(dPrevBk,positions[2*pvBk], positions[2*pvBk+1], 8, 9);
                    dpoint_var(dvBk,   positions[2*vBk],  positions[2*vBk+1],  10, 11);
                    dpoint_var(dnvBk,  positions[2*nvBk], positions[2*nvBk+1], 12, 13);
                    DPoint zA_d, zBk_d, Pk_d;
                    arcCenter_dual(dPrevA,  dvA_, dnvA_, delta, zA_d);
                    arcCenter_dual(dPrevBk, dvBk, dnvBk, delta, zBk_d);
                    arcArcCross_dual(zA_d, zBk_d, delta, ck.Xx, ck.Xy, Pk_d);

                    int vlist_k[6] = { pvA, vA, nvA, pvBk, vBk, nvBk };
                    int slot_off[6]= { 0, 2, 4, 8, 10, 12 };
                    for (int s = 0; s < 6; ++s) {
                        int sx = slot_off[s], sy = slot_off[s] + 1;
                        double incx = g2.x * Pk_d.x.d[sx] + g2.y * Pk_d.y.d[sx];
                        double incy = g2.x * Pk_d.x.d[sy] + g2.y * Pk_d.y.d[sy];
                        atomicAdd(&forceOut[2*vlist_k[s]    ], -incx);
                        atomicAdd(&forceOut[2*vlist_k[s] + 1], -incy);
                    }
                }
                continue;
            }

            // ===== prev crossing: edges (fA, fB_prev) =====
            int vBp = feat::vertexOf(cp.fB);
            int pvA = prev_arr[vA], nvA = next_arr[vA], nnvA = next_arr[nvA];
            int pvBp= prev_arr[vBp], nvBp= next_arr[vBp], nnvBp= next_arr[nvBp];

            DPoint dPrevA, dvA, dnvA, dnnvA, dPrevBp, dvBp, dnvBp, dnnvBp;
            dpoint_var(dPrevA, positions[2*pvA],  positions[2*pvA+1],  0, 1);
            dpoint_var(dvA,    positions[2*vA],   positions[2*vA+1],   2, 3);
            dpoint_var(dnvA,   positions[2*nvA],  positions[2*nvA+1],  4, 5);
            dpoint_var(dnnvA,  positions[2*nnvA], positions[2*nnvA+1], 6, 7);
            dpoint_var(dPrevBp,positions[2*pvBp], positions[2*pvBp+1], 8, 9);
            dpoint_var(dvBp,   positions[2*vBp],  positions[2*vBp+1],  10, 11);
            dpoint_var(dnvBp,  positions[2*nvBp], positions[2*nvBp+1], 12, 13);
            dpoint_var(dnnvBp, positions[2*nnvBp],positions[2*nnvBp+1],14, 15);

            DPoint pA_d, qA_d, pB_d, qB_d, tmpAm;
            // pA = a_v^+ at vA: from (prev_A, vA, next_A)
            tangentPoints_dual(dPrevA, dvA,  dnvA,  delta, tmpAm, pA_d);
            // qA = a_v^- at next_A: from (vA, next_A, next2_A)
            tangentPoints_dual(dvA,    dnvA, dnnvA, delta, qA_d, tmpAm);
            // pB = a_v^+ at vB_prev
            tangentPoints_dual(dPrevBp, dvBp, dnvBp, delta, tmpAm, pB_d);
            // qB = a_v^- at next_B_prev
            tangentPoints_dual(dvBp, dnvBp, dnnvBp, delta, qB_d, tmpAm);

            double2 vPA = make_double2(pA_d.x.v, pA_d.y.v);
            double2 vQA = make_double2(qA_d.x.v, qA_d.y.v);
            double2 vPB = make_double2(pB_d.x.v, pB_d.y.v);
            double2 vQB = make_double2(qB_d.x.v, qB_d.y.v);
            double dfp[16];
            getDf(vPA, vQA, vPB, vQB, dfp);

            double dPdIn[DUAL_N * 2];
            chainCrossingJacobian(pA_d, qA_d, pB_d, qB_d, dfp, dPdIn);

            int vlist_prev[8] = { pvA, vA, nvA, nnvA, pvBp, vBp, nvBp, nnvBp };
            for (int s = 0; s < 8; ++s) {
                int slot_x = 2*s, slot_y = 2*s + 1;
                double incx = g1.x * dPdIn[slot_x*2+0] + g1.y * dPdIn[slot_x*2+1];
                double incy = g1.x * dPdIn[slot_y*2+0] + g1.y * dPdIn[slot_y*2+1];
                atomicAdd(&forceOut[2*vlist_prev[s]    ], -incx);
                atomicAdd(&forceOut[2*vlist_prev[s] + 1], -incy);
            }

            // ===== k crossing: edges (fA, fB_k) =====
            int pvBk = prev_arr[vBk], nvBk = next_arr[vBk], nnvBk = next_arr[nvBk];
            DPoint dPrevBk, dvBk, dnvBk, dnnvBk;
            dpoint_var(dPrevBk, positions[2*pvBk], positions[2*pvBk+1], 8, 9);
            dpoint_var(dvBk,    positions[2*vBk],  positions[2*vBk+1],  10, 11);
            dpoint_var(dnvBk,   positions[2*nvBk], positions[2*nvBk+1], 12, 13);
            dpoint_var(dnnvBk,  positions[2*nnvBk],positions[2*nnvBk+1],14, 15);

            // pA/qA are the SAME as for prev crossing (same A-feature). Recompute
            // to be safe (cheap) since dPoint dual state isn't preserved.
            tangentPoints_dual(dPrevA, dvA,  dnvA,  delta, tmpAm, pA_d);
            tangentPoints_dual(dvA,    dnvA, dnnvA, delta, qA_d, tmpAm);
            tangentPoints_dual(dPrevBk, dvBk, dnvBk, delta, tmpAm, pB_d);
            tangentPoints_dual(dvBk, dnvBk, dnnvBk, delta, qB_d, tmpAm);

            vPA = make_double2(pA_d.x.v, pA_d.y.v);
            vQA = make_double2(qA_d.x.v, qA_d.y.v);
            vPB = make_double2(pB_d.x.v, pB_d.y.v);
            vQB = make_double2(qB_d.x.v, qB_d.y.v);
            double dfk[16];
            getDf(vPA, vQA, vPB, vQB, dfk);

            chainCrossingJacobian(pA_d, qA_d, pB_d, qB_d, dfk, dPdIn);

            int vlist_k[8] = { pvA, vA, nvA, nnvA, pvBk, vBk, nvBk, nnvBk };
            for (int s = 0; s < 8; ++s) {
                int slot_x = 2*s, slot_y = 2*s + 1;
                double incx = g2.x * dPdIn[slot_x*2+0] + g2.y * dPdIn[slot_x*2+1];
                double incy = g2.x * dPdIn[slot_y*2+0] + g2.y * dPdIn[slot_y*2+1];
                atomicAdd(&forceOut[2*vlist_k[s]    ], -incx);
                atomicAdd(&forceOut[2*vlist_k[s] + 1], -incy);
            }
        }
    }
    areaOut[fAIdx] = chordSum;
}

// Phase C installment 3 (CANONICAL EE FORCE KERNEL): walk + accumulate
// EDGE-EDGE force contributions.
//
// IMPORTANT: this kernel is CORRECT at delta>0 too, despite using only the
// 4-vertex (V_A, N_A, V_B, N_B) getDf Jacobian. The reason is line-
// intersection cancellation: with the rounded model, pA = V_A + ell_v * e_A
// and qA = N_A - ell_n * e_A both shift along the same line direction
// e_A = (N_A - V_A)/|N_A - V_A|. The flat-segment edge therefore lies on the
// SAME infinite line as the backbone edge V_A-to-N_A, and the line-line
// intersection point depends only on the lines, not on where the segment
// endpoints sit along them. So dP/d(prev_A) = 0 through the tangent-point
// chain, and the plain 4-vertex Jacobian is exact at any delta.
// (FD-validated at delta=0 and delta=0.04 to ~3e-10 abs error.)
//
// The dual-traced variant (phaseCWalkAreaForceDualPerFeatureKernel below)
// computes the same answer in principle but accumulates 1/u_len, 1/v_len
// factors that cancel analytically — at dense configs with short edges,
// these spurious intermediate magnitudes (1e3-1e9) overwhelm double precision
// and produce wrong forces. PREFER THE PLAIN KERNEL FOR EE FORCES. The dual
// kernel remains for ARC-ARC where no analogous cancellation exists.
//
// Arc-containing crossings are still counted in the chord-area sum but their
// force gradients are skipped here (arc Jacobian is future work).
//
// For each inside chord pair (P0=X[prev], P1=X[k]) on A-feature fA, partner
// shape sB:
//   - chordGradients(P0, P1, anchor) returns g1 = d(chord)/dP0, g2 = d(chord)/dP1
//   - getDf(pA, pnA, pBprev, pnBprev) gives dP0/d(4 vertices) -> force on those
//   - getDf(pA, pnA, pBk,    pnBk   ) gives dP1/d(4 vertices) -> force on those
//   - atomicAdd negated dot products onto forceOut[2v..2v+1]
// =============================================================
// Route 2 (AA): analytical Jacobian for ARC-ARC crossings.
//
// Two helpers:
//   arcCenterAndJacobian -- given (V_prev, V_v, V_next) and delta, compute
//     z = V_v + delta*(v_hat - u_hat)/cr and its 2x2 Jacobians w.r.t. each of
//     the three input vertices. Closed form derived from u, v, u_hat, v_hat,
//     and cr derivatives via chain rule.
//   aaCrossingJacobian -- given (z_A, z_B, delta) and the stored crossing
//     position (to pick the branch), compute the 2x2 dP/dz_A and dP/dz_B
//     from the closed-form circle-circle intersection.
//
// Composing them (dP/dV = dP/dz_X . dz_X/dV) gives a stable per-input
// Jacobian. The 1/u_len, 1/cr, 1/|D|, 1/hc factors appear ONCE each and
// stay bounded for non-degenerate geometry; no dual-chain compounding.
// =============================================================

__device__ inline void arcCenterAndJacobian(
    double V_prev_x, double V_prev_y,
    double V_v_x,    double V_v_y,
    double V_next_x, double V_next_y,
    double delta,
    double& zx, double& zy,
    double dzd_prev[4], double dzd_v[4], double dzd_next[4])
{
    double ux = wrap(V_v_x - V_prev_x), uy = wrap(V_v_y - V_prev_y);
    double vx = wrap(V_next_x - V_v_x), vy = wrap(V_next_y - V_v_y);
    double u_len = sqrt(ux*ux + uy*uy);
    double v_len = sqrt(vx*vx + vy*vy);
    if (u_len < 1e-14 || v_len < 1e-14) {
        zx = V_v_x; zy = V_v_y;
        for (int i = 0; i < 4; ++i) {
            dzd_prev[i] = 0.0; dzd_next[i] = 0.0;
            dzd_v[i]    = (i == 0 || i == 3) ? 1.0 : 0.0;
        }
        return;
    }
    double inv_u = 1.0 / u_len, inv_v = 1.0 / v_len;
    double uhx = ux * inv_u, uhy = uy * inv_u;
    double vhx = vx * inv_v, vhy = vy * inv_v;
    double cr  = uhx * vhy - uhy * vhx;
    if (fabs(cr) < 1e-14) {
        zx = V_v_x; zy = V_v_y;
        for (int i = 0; i < 4; ++i) {
            dzd_prev[i] = 0.0; dzd_next[i] = 0.0;
            dzd_v[i]    = (i == 0 || i == 3) ? 1.0 : 0.0;
        }
        return;
    }
    double inv_cr  = 1.0 / cr;
    double inv_cr2 = inv_cr * inv_cr;
    zx = V_v_x + delta * (vhx - uhx) * inv_cr;
    zy = V_v_y + delta * (vhy - uhy) * inv_cr;

    // d(u_hat)/d(u) = (I - u_hat ⊗ u_hat) / |u|. Row-major 2x2.
    double duh_du[4] = {
        (1.0 - uhx*uhx) * inv_u, -uhx*uhy * inv_u,
        -uhx*uhy * inv_u,        (1.0 - uhy*uhy) * inv_u
    };
    double dvh_dv[4] = {
        (1.0 - vhx*vhx) * inv_v, -vhx*vhy * inv_v,
        -vhx*vhy * inv_v,        (1.0 - vhy*vhy) * inv_v
    };

    // d(cr)/d(u_hat) = (vh.y, -vh.x). d(cr)/d(v_hat) = (-uh.y, uh.x).
    // Chain to d(cr)/d(u) and d(cr)/d(v) (each a row vector, 2 components).
    double dcr_du[2] = {
        vhy * duh_du[0] + (-vhx) * duh_du[2],   // /d(u.x)
        vhy * duh_du[1] + (-vhx) * duh_du[3]    // /d(u.y)
    };
    double dcr_dv[2] = {
        (-uhy) * dvh_dv[0] + uhx * dvh_dv[2],   // /d(v.x)
        (-uhy) * dvh_dv[1] + uhx * dvh_dv[3]    // /d(v.y)
    };

    double diff_x = vhx - uhx, diff_y = vhy - uhy;
    double k1 = delta * inv_cr;
    double k2 = delta * inv_cr2;

    // dzd_prev = -d(z)/d(u) = +δ * [duh/du / cr + (vh-uh) ⊗ dcr/du / cr²]
    dzd_prev[0] = k1 * duh_du[0] + k2 * diff_x * dcr_du[0];
    dzd_prev[1] = k1 * duh_du[1] + k2 * diff_x * dcr_du[1];
    dzd_prev[2] = k1 * duh_du[2] + k2 * diff_y * dcr_du[0];
    dzd_prev[3] = k1 * duh_du[3] + k2 * diff_y * dcr_du[1];

    // dzd_next = +d(z)/d(v) = +δ * [dvh/dv / cr - (vh-uh) ⊗ dcr/dv / cr²]
    dzd_next[0] = k1 * dvh_dv[0] - k2 * diff_x * dcr_dv[0];
    dzd_next[1] = k1 * dvh_dv[1] - k2 * diff_x * dcr_dv[1];
    dzd_next[2] = k1 * dvh_dv[2] - k2 * diff_y * dcr_dv[0];
    dzd_next[3] = k1 * dvh_dv[3] - k2 * diff_y * dcr_dv[1];

    // dzd_v = I - dzd_prev - dzd_next (direct + chain through u + through v)
    dzd_v[0] = 1.0 - dzd_prev[0] - dzd_next[0];
    dzd_v[1] = 0.0 - dzd_prev[1] - dzd_next[1];
    dzd_v[2] = 0.0 - dzd_prev[2] - dzd_next[2];
    dzd_v[3] = 1.0 - dzd_prev[3] - dzd_next[3];
}

// Closed-form dP/dz_A, dP/dz_B for arc-arc intersection. Generalized to
// unequal radii (rA, rB) so per-polygon delta can place arcs of different
// sizes on each side. Equal-radius case reduces to the original formula.
// Picks the ± branch matching (Xx_known, Xy_known).
//
// Math: with D = z_B - z_A, |D|, a = (|D|² + rA² - rB²)/(2|D|),
// h = sqrt(rA² - a²), m = z_A + (a/|D|)·D, perp = (-Dy, Dx)/|D|,
// P = m ± h·perp. Chain rules give:
//   df/dD = (rB² - rA²)/|D|³  (where f = a/|D|)
//   df/dz_A = (rA² - rB²)/|D|⁴ · D
//   K_outer = (a·K_a)/(h·|D|) + h/|D|²,   K_a = (|D|² - rA² + rB²)/(2|D|²)
//   K_f     = (rA² - rB²)/|D|⁴
//   K_R     = h/|D|
// Then:
//   dP/dz_A = (1-f) I + K_f · (D⊗D) ± [K_outer · (perp⊗D) - K_R · R]
//   dP/dz_B =    f  I - K_f · (D⊗D) ± [-K_outer · (perp⊗D) + K_R · R]
// Translation invariance: dP/dz_A + dP/dz_B = I.
//
// Returns false if circles don't intersect (|D| outside [|rA-rB|, rA+rB]).
__device__ inline bool aaCrossingJacobian(
    double zAx, double zAy, double zBx, double zBy,
    double rA,  double rB,
    double Xx_known, double Xy_known,
    double dPdzA[4], double dPdzB[4])
{
    double Dx = zBx - zAx, Dy = zBy - zAy;
    double D2 = Dx*Dx + Dy*Dy;
    double D  = sqrt(D2);
    double rsum  = rA + rB;
    double rdiff = fabs(rA - rB);
    double margin = 0.01 * rsum;
    if (D < rdiff + margin || D > rsum - margin) return false;
    double inv_D = 1.0 / D;

    double a  = (D2 + rA*rA - rB*rB) / (2.0 * D);
    double h2 = rA*rA - a*a;
    if (h2 <= 0.0) return false;
    double h  = sqrt(h2);
    double inv_h = 1.0 / h;

    double f   = a * inv_D;
    double K_a = (D2 - rA*rA + rB*rB) / (2.0 * D2);
    double K_f = (rA*rA - rB*rB) / (D2 * D2);
    double K_outer = (a * K_a) * (inv_h * inv_D) + h * inv_D * inv_D;
    double K_R = h * inv_D;

    double perpx = -Dy * inv_D;
    double perpy =  Dx * inv_D;
    double midx = zAx + f * Dx;
    double midy = zAy + f * Dy;

    double e_plus  = (midx + h*perpx - Xx_known)*(midx + h*perpx - Xx_known)
                   + (midy + h*perpy - Xy_known)*(midy + h*perpy - Xy_known);
    double e_minus = (midx - h*perpx - Xx_known)*(midx - h*perpx - Xx_known)
                   + (midy - h*perpy - Xy_known)*(midy - h*perpy - Xy_known);
    double sign = (e_plus <= e_minus) ? 1.0 : -1.0;

    // (perp ⊗ D), (D ⊗ D), R as row-major 2x2.
    double pdt00 = perpx * Dx, pdt01 = perpx * Dy;
    double pdt10 = perpy * Dx, pdt11 = perpy * Dy;
    double dot00 = Dx*Dx, dot01 = Dx*Dy;
    double dot10 = Dy*Dx, dot11 = Dy*Dy;
    double R00 = 0.0,  R01 = -1.0;
    double R10 = 1.0,  R11 =  0.0;

    dPdzA[0] = (1.0 - f) + K_f * dot00 + sign * (K_outer * pdt00 - K_R * R00);
    dPdzA[1] =       0.0 + K_f * dot01 + sign * (K_outer * pdt01 - K_R * R01);
    dPdzA[2] =       0.0 + K_f * dot10 + sign * (K_outer * pdt10 - K_R * R10);
    dPdzA[3] = (1.0 - f) + K_f * dot11 + sign * (K_outer * pdt11 - K_R * R11);

    dPdzB[0] = f         - K_f * dot00 + sign * (-K_outer * pdt00 + K_R * R00);
    dPdzB[1] = 0.0       - K_f * dot01 + sign * (-K_outer * pdt01 + K_R * R01);
    dPdzB[2] = 0.0       - K_f * dot10 + sign * (-K_outer * pdt10 + K_R * R10);
    dPdzB[3] = f         - K_f * dot11 + sign * (-K_outer * pdt11 + K_R * R11);
    return true;
}

// Accumulate force on one backbone vertex from the chain
// dP/dV = dPdz · dzd, then dot with the chord-gradient g.
__device__ inline void aaApplyForce(
    int vertex, const double dzd[4], const double dPdz[4], double2 g,
    double* forceOut)
{
    // dP/dV (row-major 2x2) = dPdz · dzd
    double dPdV0 = dPdz[0]*dzd[0] + dPdz[1]*dzd[2];
    double dPdV1 = dPdz[0]*dzd[1] + dPdz[1]*dzd[3];
    double dPdV2 = dPdz[2]*dzd[0] + dPdz[3]*dzd[2];
    double dPdV3 = dPdz[2]*dzd[1] + dPdz[3]*dzd[3];
    double incx = g.x * dPdV0 + g.y * dPdV2;
    double incy = g.x * dPdV1 + g.y * dPdV3;
    atomicAdd(&forceOut[2*vertex    ], -incx);
    atomicAdd(&forceOut[2*vertex + 1], -incy);
}

// EDGE-ARC implicit-function-theorem Jacobian. A is an edge (line through
// V_A, N_A); B is an arc (circle of radius delta around z_B). Two constraints
// pin P. dP/d(input) = -J_P^{-1} · (dF1/d(input), dF2/d(input))^T where
//   F1 = (P - V_A) × (N_A - V_A)  (P on line)
//   F2 = |P - z_B|^2 - delta^2     (P on circle)
//
// The closed-form dP/d(input) is propagated to each involved backbone vertex
// (V_A, N_A on the A side; prev_B, V_B, N_B on the B side via z_B's chain).
// Returns false if the line is tangent to the circle (det J_P ≈ 0).
__device__ inline bool edgeArcCrossingJacobian(
    double VAx, double VAy, double NAx, double NAy,
    double zBx, double zBy,
    const double dzB_prev[4], const double dzB_v[4], const double dzB_next[4],
    double Xx, double Xy,
    double dPdVA[4], double dPdNA[4],
    double dPd_prevB[4], double dPd_vB[4], double dPd_nextB[4])
{
    // dF1/dP = (N_A.y - V_A.y, -(N_A.x - V_A.x)) -- rotated edge vector
    // dF2/dP = 2(P - z_B)
    double epx = NAy - VAy, epy = -(NAx - VAx);
    double rx  = 2.0*(Xx - zBx), ry = 2.0*(Xy - zBy);
    double det = epx * ry - epy * rx;
    if (fabs(det) < 1e-14) {
        for (int i = 0; i < 4; ++i) {
            dPdVA[i] = 0.0; dPdNA[i] = 0.0;
            dPd_prevB[i] = 0.0; dPd_vB[i] = 0.0; dPd_nextB[i] = 0.0;
        }
        return false;
    }
    double inv = 1.0 / det;
    // J_P^{-1} = (1/det) * [[ry, -epy], [-rx, epx]]
    double Jinv00 = ry * inv,   Jinv01 = -epy * inv;
    double Jinv10 = -rx * inv,  Jinv11 = epx * inv;

    // dF1/dV_A = (P.y - N_A.y, N_A.x - P.x);  dF1/dN_A = (-(P.y - V_A.y), P.x - V_A.x)
    double dF1_dVAx = Xy - NAy;
    double dF1_dVAy = NAx - Xx;
    double dF1_dNAx = -(Xy - VAy);
    double dF1_dNAy = Xx - VAx;

    // dF2/dz_B = (-rx, -ry)
    double dF2_dzBx = -rx, dF2_dzBy = -ry;

    // For a 2-component input (V.x, V.y): dF = (dF1_x, dF2_x) for V.x and
    // (dF1_y, dF2_y) for V.y; then dP/dV = -J_P^{-1} . dF.
    // Output dPd: row-major 2x2: dPd[i*2+j] = dP[i]/dV[j].
    auto pack = [&](double dF1x, double dF1y, double dF2x, double dF2y, double dPd[4]) {
        dPd[0] = -(Jinv00 * dF1x + Jinv01 * dF2x);
        dPd[1] = -(Jinv00 * dF1y + Jinv01 * dF2y);
        dPd[2] = -(Jinv10 * dF1x + Jinv11 * dF2x);
        dPd[3] = -(Jinv10 * dF1y + Jinv11 * dF2y);
    };

    pack(dF1_dVAx, dF1_dVAy, 0.0, 0.0, dPdVA);
    pack(dF1_dNAx, dF1_dNAy, 0.0, 0.0, dPdNA);

    // B-side inputs: dF1 = 0; dF2 chains through z_B's Jacobian.
    auto packB = [&](const double dzd[4], double dPd[4]) {
        double dF2x = dF2_dzBx * dzd[0] + dF2_dzBy * dzd[2];   // d/dV.x
        double dF2y = dF2_dzBx * dzd[1] + dF2_dzBy * dzd[3];   // d/dV.y
        pack(0.0, 0.0, dF2x, dF2y, dPd);
    };
    packB(dzB_prev, dPd_prevB);
    packB(dzB_v,    dPd_vB);
    packB(dzB_next, dPd_nextB);
    return true;
}

// Direct application: -g · dPdV onto vertex, atomic add.
__device__ inline void applyForceDirect(
    int vertex, const double dPdV[4], double2 g, double* forceOut)
{
    double incx = g.x * dPdV[0] + g.y * dPdV[2];
    double incy = g.x * dPdV[1] + g.y * dPdV[3];
    atomicAdd(&forceOut[2*vertex    ], -incx);
    atomicAdd(&forceOut[2*vertex + 1], -incy);
}

// Per-crossing force helpers. Each one computes dP/d(input) for ONE crossing
// (using whichever Jacobian machinery fits its (tA, tB) type) and atomic-adds
// the negated chord-gradient dot products onto the involved backbone vertices.
__device__ inline void applyCrossingForceEE(
    int vA, int nA, int vB, int nB,
    const double* positions, double2 g, double* forceOut)
{
    double2 pA  = make_double2(positions[2*vA], positions[2*vA+1]);
    double2 pnA = make_double2(positions[2*nA], positions[2*nA+1]);
    double2 pB  = make_double2(positions[2*vB], positions[2*vB+1]);
    double2 pnB = make_double2(positions[2*nB], positions[2*nB+1]);
    double df[16];
    getDf(pA, pnA, pB, pnB, df);
    int vlist[4] = { vA, nA, vB, nB };
    for (int p = 0; p < 4; ++p) {
        int cx = 2*p, cy = 2*p + 1;
        double incx = g.x * df[cx*2+0] + g.y * df[cx*2+1];
        double incy = g.x * df[cy*2+0] + g.y * df[cy*2+1];
        atomicAdd(&forceOut[2*vlist[p]    ], -incx);
        atomicAdd(&forceOut[2*vlist[p] + 1], -incy);
    }
}

__device__ inline void applyCrossingForceAAOne(
    int pvA, int vA, int nvA, int pvB, int vB, int nvB,
    const double* positions, double deltaA, double deltaB, double2 g,
    double Xx, double Xy, double* forceOut)
{
    double zAx, zAy; double dzA_prev[4], dzA_v[4], dzA_next[4];
    arcCenterAndJacobian(
        positions[2*pvA], positions[2*pvA+1],
        positions[2*vA],  positions[2*vA+1],
        positions[2*nvA], positions[2*nvA+1],
        deltaA, zAx, zAy, dzA_prev, dzA_v, dzA_next);
    double zBx, zBy; double dzB_prev[4], dzB_v[4], dzB_next[4];
    arcCenterAndJacobian(
        positions[2*pvB], positions[2*pvB+1],
        positions[2*vB],  positions[2*vB+1],
        positions[2*nvB], positions[2*nvB+1],
        deltaB, zBx, zBy, dzB_prev, dzB_v, dzB_next);
    double dPdzA[4], dPdzB[4];
    if (!aaCrossingJacobian(zAx, zAy, zBx, zBy, deltaA, deltaB, Xx, Xy, dPdzA, dPdzB)) return;
    aaApplyForce(pvA, dzA_prev, dPdzA, g, forceOut);
    aaApplyForce(vA,  dzA_v,    dPdzA, g, forceOut);
    aaApplyForce(nvA, dzA_next, dPdzA, g, forceOut);
    aaApplyForce(pvB, dzB_prev, dPdzB, g, forceOut);
    aaApplyForce(vB,  dzB_v,    dPdzB, g, forceOut);
    aaApplyForce(nvB, dzB_next, dPdzB, g, forceOut);
}

__device__ inline void applyCrossingForceEdgeArc(
    int vA, int nA, int pvB, int vB, int nvB,
    const double* positions, double deltaB, double2 g,
    double Xx, double Xy, double* forceOut)
{
    double zBx, zBy; double dzB_prev[4], dzB_v[4], dzB_next[4];
    arcCenterAndJacobian(
        positions[2*pvB], positions[2*pvB+1],
        positions[2*vB],  positions[2*vB+1],
        positions[2*nvB], positions[2*nvB+1],
        deltaB, zBx, zBy, dzB_prev, dzB_v, dzB_next);
    double VAx = positions[2*vA], VAy = positions[2*vA+1];
    double NAx = positions[2*nA], NAy = positions[2*nA+1];
    double dPdVA[4], dPdNA[4], dPd_prevB[4], dPd_vB[4], dPd_nextB[4];
    if (!edgeArcCrossingJacobian(
            VAx, VAy, NAx, NAy, zBx, zBy,
            dzB_prev, dzB_v, dzB_next, Xx, Xy,
            dPdVA, dPdNA, dPd_prevB, dPd_vB, dPd_nextB)) return;
    applyForceDirect(vA,  dPdVA,     g, forceOut);
    applyForceDirect(nA,  dPdNA,     g, forceOut);
    applyForceDirect(pvB, dPd_prevB, g, forceOut);
    applyForceDirect(vB,  dPd_vB,    g, forceOut);
    applyForceDirect(nvB, dPd_nextB, g, forceOut);
}

__device__ inline void applyCrossingForceArcEdge(
    int pvA, int vA, int nvA, int vB, int nB,
    const double* positions, double deltaA, double2 g,
    double Xx, double Xy, double* forceOut)
{
    // Mirror of edgeArc: A is now the arc, B is now the line. Reuse the same
    // helper with (line=B, circle=A); just relabel output slots.
    double zAx, zAy; double dzA_prev[4], dzA_v[4], dzA_next[4];
    arcCenterAndJacobian(
        positions[2*pvA], positions[2*pvA+1],
        positions[2*vA],  positions[2*vA+1],
        positions[2*nvA], positions[2*nvA+1],
        deltaA, zAx, zAy, dzA_prev, dzA_v, dzA_next);
    double VBx = positions[2*vB], VBy = positions[2*vB+1];
    double NBx = positions[2*nB], NBy = positions[2*nB+1];
    double dPdVB[4], dPdNB[4], dPd_prevA[4], dPd_vA[4], dPd_nextA[4];
    if (!edgeArcCrossingJacobian(
            VBx, VBy, NBx, NBy, zAx, zAy,
            dzA_prev, dzA_v, dzA_next, Xx, Xy,
            dPdVB, dPdNB, dPd_prevA, dPd_vA, dPd_nextA)) return;
    applyForceDirect(vB,  dPdVB,     g, forceOut);
    applyForceDirect(nB,  dPdNB,     g, forceOut);
    applyForceDirect(pvA, dPd_prevA, g, forceOut);
    applyForceDirect(vA,  dPd_vA,    g, forceOut);
    applyForceDirect(nvA, dPd_nextA, g, forceOut);
}

// (aaApplyForce moved above per-crossing helpers so they can call it.)

__global__ void phaseCWalkAreaForcePerFeatureKernel(
    int                              numUniqueFA,
    const uint32_t*  __restrict__    uniqueFA,
    const uint32_t*  __restrict__    lengths,
    const uint32_t*  __restrict__    sliceStart,
    const feat::Crossing* __restrict__ crossings,
    const int*       __restrict__    shapeId,
    const int*       __restrict__    startIndices,
    const int*       __restrict__    prev_arr,
    const double*    __restrict__    polyDelta,   // per-polygon delta (size numPolygons)
    const int*       __restrict__    next_arr,
    const double*    __restrict__    positions,
    double*          __restrict__    areaOut,
    double*          __restrict__    forceOut)
{
    int fAIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (fAIdx >= numUniqueFA) return;

    int start = (int)sliceStart[fAIdx];
    int n     = (int)lengths[fAIdx];
    unsigned int fA = uniqueFA[fAIdx];
    int vA = feat::vertexOf(fA);
    int sA = shapeId[vA];
    feat::FeatureType tA = feat::typeOf(fA);

    int nA = next_arr[vA];
    double2 pA  = make_double2(positions[2*vA],  positions[2*vA+1]);
    double2 pnA = make_double2(positions[2*nA],  positions[2*nA+1]);

    double chordSum = 0.0;
    for (int k = 1; k < n; ++k) {
        feat::Crossing ck = crossings[start + k];
        int vBk = feat::vertexOf(ck.fB);
        int sBk = shapeId[vBk];
        feat::FeatureType tBk = feat::typeOf(ck.fB);

        int rank = 0;
        int pj   = -1;
        for (int j = 0; j < k; ++j) {
            int sBj = shapeId[feat::vertexOf(crossings[start + j].fB)];
            if (sBj == sBk) { rank++; pj = j; }
        }

        if ((rank & 1) == 1 && pj >= 0) {
            feat::Crossing cp = crossings[start + pj];

            int sRef = (sA < sBk) ? sA : sBk;
            double2 anchor;
            anchor.x = positions[2*startIndices[sRef]];
            anchor.y = positions[2*startIndices[sRef] + 1];

            double2 P0 = make_double2(cp.Xx, cp.Xy);
            double2 P1 = make_double2(ck.Xx, ck.Xy);

            chordSum += integrate::chordArea(P0, P1, anchor);

            // Per-crossing force dispatch. Each crossing's dP/d(input) is
            // computed by the helper matching its (tA, tB) type:
            //   EE: getDf (line-intersection cancellation makes it exact at
            //       any delta)
            //   AA: arcCenter + closed-form circle-circle dP/d(z_A, z_B)
            //   EDGE-ARC / ARC-EDGE: implicit function on the 2-constraint
            //       line+circle system (edgeArcCrossingJacobian)
            // The chord-area and anchor-vertex contributions apply to every
            // inside pair regardless of type.
            double2 g1, g2;
            integrate::chordGradients(P0, P1, anchor, g1, g2);

            int anchorV = startIndices[sRef];
            atomicAdd(&forceOut[2*anchorV    ], g1.x + g2.x);
            atomicAdd(&forceOut[2*anchorV + 1], g1.y + g2.y);

            feat::FeatureType tBp = feat::typeOf(cp.fB);
            int vBp = feat::vertexOf(cp.fB);
            int pvA = prev_arr[vA];
            // Per-polygon delta: A-side polygon is sA (constant for this fA's
            // slice); B-side polygon is sBk (same for prev and k since same-shape
            // crossings get paired). All arc-related radii read from polyDelta.
            double deltaA = polyDelta[sA];
            double deltaB = polyDelta[sBk];

            // prev crossing -> g1
            if (tA == feat::EDGE && tBp == feat::EDGE) {
                applyCrossingForceEE(vA, nA, vBp, next_arr[vBp],
                                       positions, g1, forceOut);
            } else if (tA == feat::EDGE && tBp == feat::ARC1) {
                applyCrossingForceEdgeArc(vA, nA, prev_arr[vBp], vBp, next_arr[vBp],
                                            positions, deltaB, g1, cp.Xx, cp.Xy, forceOut);
            } else if (tA == feat::ARC1 && tBp == feat::EDGE) {
                applyCrossingForceArcEdge(pvA, vA, nA, vBp, next_arr[vBp],
                                            positions, deltaA, g1, cp.Xx, cp.Xy, forceOut);
            } else if (tA == feat::ARC1 && tBp == feat::ARC1) {
                applyCrossingForceAAOne(pvA, vA, nA, prev_arr[vBp], vBp, next_arr[vBp],
                                          positions, deltaA, deltaB, g1, cp.Xx, cp.Xy, forceOut);
            }

            // k crossing -> g2
            if (tA == feat::EDGE && tBk == feat::EDGE) {
                applyCrossingForceEE(vA, nA, vBk, next_arr[vBk],
                                       positions, g2, forceOut);
            } else if (tA == feat::EDGE && tBk == feat::ARC1) {
                applyCrossingForceEdgeArc(vA, nA, prev_arr[vBk], vBk, next_arr[vBk],
                                            positions, deltaB, g2, ck.Xx, ck.Xy, forceOut);
            } else if (tA == feat::ARC1 && tBk == feat::EDGE) {
                applyCrossingForceArcEdge(pvA, vA, nA, vBk, next_arr[vBk],
                                            positions, deltaA, g2, ck.Xx, ck.Xy, forceOut);
            } else if (tA == feat::ARC1 && tBk == feat::ARC1) {
                applyCrossingForceAAOne(pvA, vA, nA, prev_arr[vBk], vBk, next_arr[vBk],
                                          positions, deltaA, deltaB, g2, ck.Xx, ck.Xy, forceOut);
            }
        }
    }
    areaOut[fAIdx] = chordSum;
}

// Prefix-sum kernel: convert per-feature lengths to per-feature start offsets.
// Tiny scan; we run it on a single block with parallel exclusive scan via cub.
__global__ void exclusiveScanU32Kernel(
    const uint32_t* __restrict__ lengths,
    uint32_t*       __restrict__ starts,
    int n)
{
    // Single-thread version is fine here -- n is at most ~hundreds and this
    // runs once per Phase C call.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx != 0) return;
    uint32_t acc = 0;
    for (int i = 0; i < n; ++i) {
        starts[i] = acc;
        acc += lengths[i];
    }
}

__global__ void updateNeighborsKernel(const int* __restrict__ shapeId, const int* __restrict__ startIndices, const double* __restrict__ positions, const int* __restrict__ cellLocation, const int* __restrict__ neighborIndices, const int size, int* __restrict__ neighbors, int* __restrict__ numNeighbors, int maxNeighbors, int boxSize, int* __restrict__ countPerBox, double2* __restrict__ tu,  bool* __restrict__ inside, double delta) {
    int id1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (id1 >= size) return;

    // cached per-thread values
    int boxCount = boxSize * boxSize;

    int shape = shapeId[id1];
    int st = startIndices[shape + 1] - 1;
    int id2 = (id1 == st) ? startIndices[shape] : id1 + 1;

    const double px = positions[2 * id1];
    const double py = positions[2 * id1 + 1];

    double rx = wrap(positions[2 * id2] - px);
    double ry = wrap(positions[2 * id2 + 1] - py);

    int box = cellLocation[id1];
    int bx = box % boxSize;
    int by = box / boxSize;

    int neighborCount = 0;

    int processedBoxes[16];
    int numProcessedBoxes = 0;

    // iterate 3x3 neighborhood of boxes (with wrap)
    for (int dx = -1; dx <= 1; dx++) {
        int nx = bx + dx;
        if (nx < 0) nx += boxSize; else if (nx >= boxSize) nx -= boxSize;
        for (int dy = -1; dy <= 1; dy++) {
            int ny = by + dy;
            if (ny < 0) ny += boxSize; else if (ny >= boxSize) ny -= boxSize;

            int newBox = ny * boxSize + nx;

            // Skip if we've already processed this box
            bool alreadyProcessed = false;
            for (int b = 0; b < numProcessedBoxes; b++) {
                if (processedBoxes[b] == newBox) {
                    alreadyProcessed = true;
                    break;
                }
            }
            if (alreadyProcessed) continue;
            if (numProcessedBoxes < 16) {
                processedBoxes[numProcessedBoxes++] = newBox;
            }

            int si = countPerBox[newBox];
            int sf = (newBox + 1 < boxCount) ? countPerBox[newBox + 1] : size;
            if (si >= sf) continue;

            // iterate entries in the neighborIndices array for this box
            for (int idx = si; idx < sf; idx++) {
                int nid = neighborIndices[idx];  // actual vertex index
                if (nid <= id1) {
                    continue;
                }

                int shape2 = shapeId[nid];
                int st2 = startIndices[shape2 + 1] - 1;
                int nid2 = (nid == st2) ? startIndices[shape2] : nid + 1;

                // skip trivial/adjacent edges
                if (nid == id1 || nid == id2 || nid2 == id1 || nid2 == id2) continue;

                // positions of neighbor edge
                double qx = positions[2 * nid];
                double qy = positions[2 * nid + 1];

                double sx = wrap(positions[2 * nid2] - qx);
                double sy = wrap(positions[2 * nid2 + 1] - qy);
                double gx = wrap(qx - px);
                double gy = wrap(qy - py);

                isect::EdgeEdgeResult cr = isect::edgeEdgeCross(rx, ry, sx, sy, gx, gy, delta);
                if (!cr.crossed) continue;
                if (neighborCount < maxNeighbors) {
                    neighbors[id1 * maxNeighbors + neighborCount] = nid;
                    if (cr.positiveDenom) {
                        inside[id1 * maxNeighbors + neighborCount] = true;
                        tu[id1 * maxNeighbors + neighborCount].x = cr.tt;
                        tu[id1 * maxNeighbors + neighborCount].y = cr.uu;
                    } else {
                        inside[id1 * maxNeighbors + neighborCount] = false;
                        tu[id1 * maxNeighbors + neighborCount].x = cr.uu;
                        tu[id1 * maxNeighbors + neighborCount].y = cr.tt;
                    }
                }
                neighborCount++;    // ...always count, so numNeighbors reflects the true total
            }
        }
    }
    numNeighbors[id1] = neighborCount;
}

__global__ void updateValidAndCountsKernel(const int numVertices, const int* __restrict__ neighbors, const int* __restrict__ numNeighbors, const int maxNeighbors, const bool* __restrict__ insideFlag, const int* __restrict__ shapeIds, const int numShapes, int* __restrict__ valid, int* __restrict__ shapeCounts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalNeighbors = numVertices * maxNeighbors;
    if (idx >= totalNeighbors) return;

    int n1 = idx / maxNeighbors;
    int i = idx % maxNeighbors;

    // Check if this contact slot is actually used
    if (i < numNeighbors[n1]) {
        valid[idx] = 1;

        int n2 = neighbors[idx];
        int s1 = shapeIds[n1];
        int s2 = shapeIds[n2];
        bool inside = insideFlag[idx];

        int targetShape = inside ? s2 : s1;
        atomicAdd(&shapeCounts[targetShape], 1);
    } else {
        valid[idx] = 0;
    }
}

__global__ void updateCompactedIntersectionsKernel(const int numVertices, const int maxNeighbors, const int* __restrict__ neighbors, const bool* __restrict__ insideFlag, const int* __restrict__ shapeIds, const int* __restrict__ startIndices, const int* __restrict__ valid, const uint64_t* __restrict__ outputIdx, uint64_t* __restrict__ intersections, const double2* __restrict__ tuSrc, double2* __restrict__ tuOut) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalNeighbors = numVertices * maxNeighbors;
    if (idx >= totalNeighbors) return;
    if (!valid[idx]) return;

    uint64_t outPos = outputIdx[idx];

    int n1 = idx / maxNeighbors;
    int n2 = neighbors[idx];
    int s1 = shapeIds[n1];
    int s2 = shapeIds[n2];
    n1 -= startIndices[s1];
    n2 -= startIndices[s2];
    bool inside = insideFlag[idx];
    // Read source tu before any writes (tuSrc != tuOut: separate buffers required).
    // Convention (established in updateNeighborsKernel and relied on throughout):
    //   tu.y = parameter along si's edge (inner shape, bits [47:32] of packed)
    //   tu.x = parameter along sj's edge (outer shape, bits [63:48] of packed)
    float tVal = tuSrc[idx].x;
    float uVal = tuSrc[idx].y;

    // Pack 64-bit intersection exactly as Python's pack()
    uint64_t packed;
    if (inside) {
        // numbers = [s2, s1, n1, n2]
        packed = ((uint64_t)(uint16_t)s1 << 48) |
                 ((uint64_t)(uint16_t)s2 << 32) |
                 ((uint64_t)(uint16_t)n2 << 16) |
                 (uint64_t)(uint16_t)n1;
    } else {
        // numbers = [s1, s2, n2, n1]
        packed = ((uint64_t)(uint16_t)s2 << 48) |
                 ((uint64_t)(uint16_t)s1 << 32) |
                 ((uint64_t)(uint16_t)n1 << 16) |
                 (uint64_t)(uint16_t)n2;
    }
    intersections[outPos] = packed;
    tuOut[outPos] = make_double2(tVal, uVal);
}

__global__ void updateOverlapAreaKernel(const int* __restrict__ shapeId, const int* __restrict__ startIndices, int pointDensity, int* __restrict__ intersectionsCounter, const int* __restrict__ neighborIndices, int size, int boxSize, const int* __restrict__ countPerBox, const double* __restrict__ positions) {
    const double eps = 1e-12;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPoints = pointDensity * pointDensity;
    if (idx >= totalPoints) return;

    int ix = idx % pointDensity;
    int iy = idx / pointDensity;

    // Center the point in its grid cell
    double px = (ix + 0.5) / pointDensity;
    double py = (iy + 0.5) / pointDensity;

    intersectionsCounter[idx] = 0;
    int numIntersections = 0;

    // Map to box
    int cellX = int(px * boxSize);
    int cellY = int(py * boxSize);
    int boxCount = boxSize * boxSize;

    // Track polygons we've already checked for this point.
    // 512 slots handles very dense packings; if this overflows we break early
    // rather than silently double-counting polygons.
    int trackedPolys[512];
    int numTrackedPolys = 0;

    // Iterate over 3x3 neighboring boxes
    for (int dx = -1; dx <= 1; dx++) {
        int nx = cellX + dx;
        if (nx < 0) nx += boxSize;
        else if (nx >= boxSize) nx -= boxSize;

        for (int dy = -1; dy <= 1; dy++) {
            int ny = cellY + dy;
            if (ny < 0) ny += boxSize;
            else if (ny >= boxSize) ny -= boxSize;

            int boxId = ny * boxSize + nx;
            int si = countPerBox[boxId];
            int sf = (boxId + 1 < boxCount) ? countPerBox[boxId + 1] : size;

            for (int nidx = si; nidx < sf; nidx++) {
                int vid = neighborIndices[nidx];
                int polyId = shapeId[vid];

                // Skip polygons we've already checked
                bool alreadyTracked = false;
                for (int t = 0; t < numTrackedPolys; t++) {
                    if (trackedPolys[t] == polyId) {
                        alreadyTracked = true;
                        break;
                    }
                }
                if (alreadyTracked) continue;

                int start = startIndices[polyId];
                int end = startIndices[polyId + 1];
                if (end - start < 3) continue; // Not a polygon

                // Angle-sum test
                double angleSum = 0.0;
                double vxPrev = wrapPeriodic(positions[2 * start] - px);
                double vyPrev = wrapPeriodic(positions[2 * start + 1] - py);
                double vx0 = vxPrev + 0.0;
                double vy0 = vyPrev + 0.0;

                for (int e = start + 1; e < end; e++) {
                    double vx = vxPrev + wrapPeriodic(positions[2 * e] - positions[2 * e - 2]);
                    double vy = vyPrev + wrapPeriodic(positions[2 * e + 1] - positions[2 * e - 1]);

                    double cross = vxPrev * vy - vyPrev * vx;
                    double dot = vxPrev * vx + vyPrev * vy;
                    angleSum += atan2(cross, dot);

                    vxPrev = vx;
                    vyPrev = vy;
                }

                // Close the polygon loop
                angleSum += atan2(vxPrev * vy0 - vyPrev * vx0, vxPrev * vx0 + vyPrev * vy0);
                if (fabs(angleSum - 2.0 * pi) < eps) {
                    numIntersections++;
                }

                // Track this polygon; break if the deduplication array is full
                // to avoid double-counting polygons seen in multiple cells.
                if (numTrackedPolys < 512) {
                    trackedPolys[numTrackedPolys++] = polyId;
                } else {
                    // Array full — stop searching to preserve correctness.
                    goto done;
                }
            }
        }
    }

    done:
    intersectionsCounter[idx] = numIntersections * (numIntersections - 1) / 2;
//    intersectionsCounter[idx] = numIntersections;
}

__global__ void updateOutersectionsKernel(const uint64_t* __restrict__ intersections, const double2* __restrict__ tu, double2* __restrict__ ut, const int* __restrict__ startIndices, int numIntersections, uint64_t* __restrict__ outersections) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numIntersections) return;

    uint64_t inter = intersections[idx];
    int sj = (inter >> 48) & 0xFFFF;
    int si = (inter >> 32) & 0xFFFF;
    int i = ((inter >> 16) & 0xFFFF);
    int ni = startIndices[si + 1] - startIndices[si];
    uint64_t sij = ((uint64_t)si << 48) | ((uint64_t)sj << 32);
        
    float tVal = tu[idx].y;
    int start = 0;
    int end = numIntersections - 1;
    int mid;
    while (end > start) {
        mid = (end + start) / 2;
        if (intersections[mid] < sij) start = mid + 1;
        else end = mid;
    }
    uint64_t ub = ((uint64_t)si << 48) | ((uint64_t)(sj + 1) << 32);
    

    int bestDist = -1;
    float bestU = FLT_MAX;
    int bestIdx = -1;
    float fallbackU = FLT_MAX;
    int fallbackIdx = -1;
    int k = start;
    while (k < numIntersections && intersections[k] < ub) {
        uint64_t kInter = intersections[k];
        int l = kInter & 0xFFFF;
        float uVal = tu[k].x;

        int d = (l - i + ni) % ni;

        // Skip self-pairing if condition fails
        if (l == i && tVal >= uVal) {
            k++;
            continue;
        }

        // New minimum distance found
        if (bestDist == -1 || d < bestDist) {
            bestDist = d;
            bestU = FLT_MAX;
            bestIdx = -1;
            fallbackU = FLT_MAX;
            fallbackIdx = -1;
        }

        // If this distance matches current best
        if (d == bestDist) {
            if (d == 0) {
                // same edge. Now check t stuff
                // Prefer candidates with u >= tVal
                if (uVal >= tVal && uVal < bestU) {
                    bestU = uVal;
                    bestIdx = k;
                }
                // Keep fallback for those violating TU
                if (uVal < fallbackU) {
                    fallbackU = uVal;
                    fallbackIdx = k;
                }
            }
            else {
                if (uVal < bestU) {
                    bestU = uVal;
                    bestIdx = k;
                }
            }
        }
        k++;
    }
    int player = (bestIdx != -1) ? bestIdx : fallbackIdx;
    if (player == -1) {
        // no valid partners. Falling back
        outersections[idx] = intersections[idx];
        ut[idx] = tu[idx];
        return;
    }
    outersections[idx] = intersections[player];
    ut[idx] = tu[player];
}

__global__ void updateForceEnergyExteriorKernel(int numIntersections, const uint64_t* __restrict__ intersections, const uint64_t* __restrict__ outersections, const double2* __restrict__ tu, const double2* __restrict__ ut, const double* __restrict__ positions, const int* __restrict__ next, const int* __restrict__ prev, const int* __restrict__ shapeId, const int* __restrict__ startIndices, double* __restrict__ force, double* __restrict__ energyGlobal) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double localEnergy = 0.0;

    if (idx < numIntersections) {
        uint64_t inter = intersections[idx];
        uint64_t outer = outersections[idx];
        if (inter != outer) {

        uint16_t s1    = (inter >> 32) & 0xFFFF;   // shape index for i
        uint16_t s2    = (inter >> 48) & 0xFFFF;   // shape index for j
        uint16_t iLoc = (inter >> 16) & 0xFFFF;
        uint16_t jLoc =  inter        & 0xFFFF;

        uint16_t kLoc = (outer >> 16) & 0xFFFF;   // local index in shape s2
        uint16_t lLoc =  outer        & 0xFFFF;   // local index in shape s1

        int start1 = startIndices[s1];
        int start2 = startIndices[s2];

        int i = start1 + iLoc;
        int j = start2 + jLoc;
        int k = start2 + kLoc;          // using start2 for k
        int l = start1 + lLoc;          // using start1 for l

        double2 pi = *reinterpret_cast<const double2*>(&positions[2*i]);
        double2 pj = *reinterpret_cast<const double2*>(&positions[2*j]);
        double2 pk = *reinterpret_cast<const double2*>(&positions[2*k]);
        double2 pl = *reinterpret_cast<const double2*>(&positions[2*l]);

        int zi = next[i];
        int zj = next[j];
        int zk = next[k];
        int zl = next[l];

        double2 pzi = *reinterpret_cast<const double2*>(&positions[2*zi]);
        double2 pzj = *reinterpret_cast<const double2*>(&positions[2*zj]);
        double2 pzk = *reinterpret_cast<const double2*>(&positions[2*zk]);
        double2 pzl = *reinterpret_cast<const double2*>(&positions[2*zl]);

        // ---- Edge vectors and parameters ----
        double2 r1 = wrap2({pzi.x - pi.x, pzi.y - pi.y});
        double2 r2 = wrap2({pzk.x - pk.x, pzk.y - pk.y});

        double2 tuf = tu[idx];
        double2 utf = ut[idx];
        double t1 = tuf.y;          // second component
        double t2 = utf.y;

        double2 fij = wrap2({pi.x + t1 * r1.x, pi.y + t1 * r1.y});
        double2 fkl = wrap2({pk.x + t2 * r2.x, pk.y + t2 * r2.y});

        // Shared per-pair anchor for the Green's-theorem terms. Every thread
        // (this kernel AND the interior kernel) that contributes to pair
        // (s1,s2) must use the SAME startPoint, otherwise the anchored sum
        // does not telescope to the closed-contour area (especially when a
        // polygon straddles the periodic box).
        int sRef = (s1 < s2) ? s1 : s2;
        double2 startPoint;
        startPoint.x = positions[2*startIndices[sRef]];
        startPoint.y = positions[2*startIndices[sRef] + 1];

        if (i == l) {
            // Branch 1: i == l
            localEnergy += integrate::chordArea(fij, fkl, startPoint);

            double2 g1, g2;
            integrate::chordGradients(fij, fkl, startPoint, g1, g2);

            double dfij[16], dfki[16];
            getDf(pi, pzi, pj, pzj, dfij);
            getDf(pk, pzk, pi, pzi, dfki);   // note order: (pk,pzk,pi,pzi)

            // Helper lambda to add dot product of g with a column of df to a vertex
            auto addContrib = [&](int v, int col_start, double2 g) {
                double2 inc = {0.0, 0.0};
                // col_start is the first column of the two belonging to the vertex
                for (int beta = 0; beta < 2; ++beta) {
                    int col = col_start + beta;
                    double dot = g.x * dfij[col*2 + 0] + g.y * dfij[col*2 + 1];
                    if (beta == 0) inc.x = dot;
                    else inc.y = dot;
                }
                atomicAdd(&force[2*v],   -inc.x);
                atomicAdd(&force[2*v+1], -inc.y);
            };

            // Contributions from dfij with g1
            // vertex i (col 0-1)
            addContrib(i, 0, g1);
            // vertex zi (col 2-3)
            addContrib(zi, 2, g1);
            // vertex j (col 4-5)
            addContrib(j, 4, g1);
            // vertex zj (col 6-7)
            addContrib(zj, 6, g1);

            // Contributions from dfki with g2
            auto addContrib2 = [&](int v, int col_start, double2 g) {
                double2 inc = {0.0, 0.0};
                for (int beta = 0; beta < 2; beta++) {
                    int col = col_start + beta;
                    double dot = g.x * dfki[col*2 + 0] + g.y * dfki[col*2 + 1];
                    if (beta == 0) inc.x = dot;
                    else inc.y = dot;
                }
                atomicAdd(&force[2*v],   -inc.x);
                atomicAdd(&force[2*v+1], -inc.y);
            };
            // vertex k (col 0-1)
            addContrib2(k, 0, g2);
            // vertex zk (col 2-3)
            addContrib2(zk, 2, g2);
            // vertex i (col 4-5) – note: i already updated, we add again
            addContrib2(i, 4, g2);
            // vertex zi (col 6-7)
            addContrib2(zi, 6, g2);
        }
        else {
            // Branch 2: i != l
            localEnergy += integrate::chordArea(fij, pzi, startPoint);
            localEnergy += integrate::chordArea(pl, fkl, startPoint);

            // First chord gradients (fij, pzi)
            double2 g1a, g2a;
            integrate::chordGradients(fij, pzi, startPoint, g1a, g2a);

            double dfij[16], dfkl[16];
            getDf(pi, pzi, pj, pzj, dfij);
            getDf(pk, pzk, pl, pzl, dfkl);

            // Contributions from first g12 with dfij
            auto addContribDfij = [&](int v, int col_start, double2 g) {
                double2 inc = {0.0, 0.0};
                for (int beta = 0; beta < 2; ++beta) {
                    int col = col_start + beta;
                    double dot = g.x * dfij[col*2 + 0] + g.y * dfij[col*2 + 1];
                    if (beta == 0) inc.x = dot;
                    else inc.y = dot;
                }
                atomicAdd(&force[2*v],   -inc.x);
                atomicAdd(&force[2*v+1], -inc.y);
            };
            // i from dfij col 0-1 with g1a
            addContribDfij(i, 0, g1a);
            // j from dfij col 4-5 with g1a
            addContribDfij(j, 4, g1a);
            // zi from dfij col 2-3 with g1a, plus direct g2a
            {
                double2 inc = {0.0, 0.0};
                for (int beta = 0; beta < 2; ++beta) {
                    int col = 2 + beta;   // col 2-3
                    double dot = g1a.x * dfij[col*2 + 0] + g1a.y * dfij[col*2 + 1];
                    if (beta == 0) inc.x = dot + g2a.x;
                    else inc.y = dot + g2a.y;
                }
                atomicAdd(&force[2*zi],   -inc.x);
                atomicAdd(&force[2*zi+1], -inc.y);
            }
            // zj from dfij col 6-7 with g1a
            addContribDfij(zj, 6, g1a);

            // Second chord gradients (pl, fkl)
            double2 g1b, g2b;
            integrate::chordGradients(pl, fkl, startPoint, g1b, g2b);

            auto addContribDfkl = [&](int v, int col_start, double2 g) {
                double2 inc = {0.0, 0.0};
                for (int beta = 0; beta < 2; ++beta) {
                    int col = col_start + beta;
                    double dot = g.x * dfkl[col*2 + 0] + g.y * dfkl[col*2 + 1];
                    if (beta == 0) inc.x = dot;
                    else inc.y = dot;
                }
                atomicAdd(&force[2*v],   -inc.x);
                atomicAdd(&force[2*v+1], -inc.y);
            };

            {
                double2 inc = {g1b.x, g1b.y};
                for (int beta = 0; beta < 2; ++beta) {
                    int col = 4 + beta;   // col 4-5
                    double dot = g2b.x * dfkl[col*2 + 0] + g2b.y * dfkl[col*2 + 1];
                    if (beta == 0) inc.x += dot;
                    else inc.y += dot;
                }
                atomicAdd(&force[2*l],   -inc.x);
                atomicAdd(&force[2*l+1], -inc.y);
            }
            // k from dfkl col 0-1 with g2b
            addContribDfkl(k, 0, g2b);
            // zl from dfkl col 6-7 with g2b
            addContribDfkl(zl, 6, g2b);
            // zk from dfkl col 2-3 with g2b
            addContribDfkl(zk, 2, g2b);
        }
        } // if (inter != outer)
    }
    // ---- Energy reduction ----
    // Use shared memory for block‑wise sum, then atomicAdd to global
    extern __shared__ double smem[];
    smem[threadIdx.x] = localEnergy;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] += smem[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(energyGlobal, smem[0]);
    }
}

__global__ void updateShapeRangesKernel(const uint64_t* intersections, int numIntersections, int* shapeStart, int* shapeEnd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numIntersections) return;
    uint64_t inter = intersections[idx];
    int s = (inter >> 32) & 0xFFFF;          // shape of the first vertex (i)
    atomicMin(&shapeStart[s], idx);
    atomicMax(&shapeEnd[s], idx);
}

__global__ void updateForceEnergyEdgeKernel(int numVertices, const double* positions, const double* targetEdgeLengths, const double* edgeLengths, const int* next, const int* prev, const int* shapeId, double* force, double* energy, double stiffness) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= numVertices) return;
    // get the edge ids
    int prv = prev[m];
    int nxt = next[m];
    double l0 = targetEdgeLengths[shapeId[m]];
    double l0prv = targetEdgeLengths[shapeId[prv]];

    // use pre-computed edge lengths from updatePolygonGeometryKernel
    double l = edgeLengths[m];
    double prvl = edgeLengths[prv];

    double2 dvmzm, dvzpmm;
    dvmzm.x = positions[2 * m] - positions[2 * nxt];
    dvmzm.y = positions[2 * m + 1] - positions[2 * nxt + 1];
    dvzpmm.x = positions[2 * prv] - positions[2 * m];
    dvzpmm.y = positions[2 * prv + 1] - positions[2 * m + 1];

    dvmzm.x += 1.5;
    dvmzm.y += 1.5;
    dvzpmm.x += 1.5;
    dvzpmm.y += 1.5;

    while (dvmzm.x  >= 1.0) dvmzm.x -= 1.0;
    while (dvmzm.y  >= 1.0) dvmzm.y -= 1.0;
    while (dvzpmm.x >= 1.0) dvzpmm.x -= 1.0;
    while (dvzpmm.y >= 1.0) dvzpmm.y -= 1.0;

    dvmzm.x -= 0.5;
    dvmzm.y -= 0.5;
    dvzpmm.x -= 0.5;
    dvzpmm.y -= 0.5;

    // energy: stiffness * (1 - l/l0)^2
    // dE/dr_m = stiffness * 2*(1-l/l0) * (-1/l0) * (-dvmzm/l)
    //         = stiffness * 2*(1-l/l0) / (l0*l) * dvmzm
    // force = -dE/dr_m  => coeff * dvmzm where coeff = -2*(1-l/l0)/(l0*l)
    double coeff1 = (l > 1e-14) ? (1.0 - l0 / l) : 0.0;
    double coeff2 = (prvl > 1e-14) ? (1.0 - l0prv / prvl) : 0.0;

    double localEnergy = stiffness * (l - l0) * (l - l0) / 2.0;
    double2 localForceM;
    localForceM.x = -(coeff1 * dvmzm.x - coeff2 * dvzpmm.x) * stiffness;
    localForceM.y = -(coeff1 * dvmzm.y - coeff2 * dvzpmm.y) * stiffness;

    if (localEnergy != 0.0) atomicAdd(energy, localEnergy);
    if (localForceM.x != 0.0 || localForceM.y != 0.0) {
        atomicAdd(&force[2 * m], localForceM.x);
        atomicAdd(&force[2 * m + 1], localForceM.y);
    }
}

__global__ void updateForceEnergyAreaKernel(int numVertices, const int* shapeId, const int* next, const int* prev, const double* positions, const double* areas, const double* targetAreas, const int* startIndices, double* force, double* energy, double compressibility) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= numVertices) return;
    int s = shapeId[m];
    double A  = areas[s];
    double A0 = targetAreas[s];
    if (A0 < 1e-14) return;
    double coeff = 0.5 * compressibility * (A - A0);
    double fx = -coeff * wrap(positions[2 * next[m] + 1] - positions[2 * prev[m] + 1]);
    double fy = -coeff * wrap(positions[2 * prev[m]] - positions[2 * next[m]]);
    atomicAdd(&force[2 * m], fx);
    atomicAdd(&force[2 * m + 1], fy);
    if (startIndices[s] == m) {
        atomicAdd(energy, compressibility * (A - A0) * (A - A0) / 2.0);
    }
}

__global__ void updateForceEnergyInteriorKernel(int numVertices, const uint64_t* intersections, const uint64_t* outersections, const double2* tu, const double2* ut, const double* positions, const int* next, const int* prev, const int* shapeId, const int* startIndices, double* force, double* energy, int* shapeStart, int* shapeEnd) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= numVertices) return;

    int s = shapeId[m];
    int startIdx = shapeStart[s];
    int endIdx   = shapeEnd[s];
    if (startIdx > endIdx) return;            // no intersections for this shape

    int n = startIndices[s+1] - startIndices[s];
    int mLoc = m - startIndices[s];

    // Cache positions of edge (m -> next(m))
    double2 pm    = *reinterpret_cast<const double2*>(&positions[2*m]);
    int nxt       = next[m];
    double2 pnxt  = *reinterpret_cast<const double2*>(&positions[2*nxt]);

    double localEnergy = 0.0;
    double2 localForceM  = {0.0, 0.0};
    double2 localForceNxt = {0.0, 0.0};

    for (int k = startIdx; k <= endIdx; ++k) {
        uint64_t inter = intersections[k];
        uint64_t outer = outersections[k];
        if (inter == outer) continue;

        int si = (inter >> 32) & 0xFFFF;
        int iLoc = (inter >> 16) & 0xFFFF;
        int lLoc = outer & 0xFFFF;

        // Cyclic distances from iLoc to mLoc and to lLoc
        int iDist = (mLoc - iLoc + n) % n;
        int lDist = (lLoc - iLoc + n) % n;

        // Condition from Python: edge lies strictly between i and l
        if (iDist > 0 && iDist < lDist) {
            // Shared per-pair anchor for the Green's-theorem terms (must match
            // the exterior kernel's anchor for the same polygon pair so the
            // contributions telescope correctly).
            int sj = (inter >> 48) & 0xFFFF;
            int sRef = (si < sj) ? si : sj;
            double2 startPoint;
            startPoint.x = positions[2*startIndices[sRef]];
            startPoint.y = positions[2*startIndices[sRef] + 1];

            // Energy contribution
            localEnergy += integrate::chordArea(pm, pnxt, startPoint);

            // Gradient
            double2 g1, g2;
            integrate::chordGradients(pm, pnxt, startPoint, g1, g2);

            // Accumulate force (negative gradient, as in Python)
            localForceM.x  -= g1.x;
            localForceM.y  -= g1.y;
            localForceNxt.x -= g2.x;
            localForceNxt.y -= g2.y;
        }
    }

    // Write back with atomics
    if (localEnergy != 0.0)
        atomicAdd(energy, localEnergy);
    if (localForceM.x != 0.0 || localForceM.y != 0.0) {
        atomicAdd(&force[2*m],   localForceM.x);
        atomicAdd(&force[2*m+1], localForceM.y);
    }
    if (localForceNxt.x != 0.0 || localForceNxt.y != 0.0) {
        atomicAdd(&force[2*nxt],   localForceNxt.x);
        atomicAdd(&force[2*nxt+1], localForceNxt.y);
    }
}

// ============================================================
// Rounded-polygon intersection energy/forces
// Implements Area(Ã∩B̃) = I(A→B) + I(B→A) via Green's theorem
// boundary walk over smoothed features (arcs + straight edges).
//
// Scratch buffer layout (per thread, stride = 30 * polygonSize doubles):
//   [0 .. n-1]       bx_pos
//   [n .. 2n-1]      by_pos
//   [2n .. 3n-1]     bam_x
//   [3n .. 4n-1]     bam_y
//   [4n .. 5n-1]     bap_x
//   [5n .. 6n-1]     bap_y
//   [6n .. 7n-1]     bz_x
//   [7n .. 8n-1]     bz_y
//   [8n .. 9n-1]     bdphi
//   [9n .. 10n-1]    bphi0
//   [10n .. 14n]     arc/edge cross tau/t  (4n slots, max 2n+2)
//   [14n .. 18n]     arc/edge cross Xx
//   [18n .. 22n]     arc/edge cross Xy
//   [22n .. 26n]     edge cross eB_x  (B-feature tangent, 0 for arc crossings)
//   [26n .. 30n]     edge cross eB_y
// ============================================================

// ---- arc geometry at one backbone vertex ----------------------------------------
__device__ void computeArcGeom(
    double ax, double ay,
    double ux, double uy,   // u = a_k - a_{k-1}, MIC-wrapped
    double vx, double vy,   // v = a_{k+1} - a_k, MIC-wrapped
    double delta,
    double& uh_x, double& uh_y,
    double& vh_x, double& vh_y,
    double& u_len, double& v_len,
    double& cr, double& dt, double& dphi, double& ell,
    double& am_x, double& am_y,
    double& ap_x, double& ap_y,
    double& zx,   double& zy,
    double& phi0
) {
    u_len = sqrt(ux*ux + uy*uy);
    v_len = sqrt(vx*vx + vy*vy);
    uh_x = (u_len > 1e-14) ? ux/u_len : 0.0;
    uh_y = (u_len > 1e-14) ? uy/u_len : 0.0;
    vh_x = (v_len > 1e-14) ? vx/v_len : 0.0;
    vh_y = (v_len > 1e-14) ? vy/v_len : 0.0;

    cr   = uh_x*vh_y - uh_y*vh_x;
    dt   = uh_x*vh_x + uh_y*vh_y;
    dphi = atan2(cr, dt);

    double abs_cr = fabs(cr);
    double denom  = 1.0 + dt;
    ell = (abs_cr > 1e-14 && denom > 1e-14) ? delta*abs_cr/denom : 0.0;

    am_x = ax - ell*uh_x;  am_y = ay - ell*uh_y;
    ap_x = ax + ell*vh_x;  ap_y = ay + ell*vh_y;

    double inv_cr = (abs_cr > 1e-14) ? 1.0/abs_cr : 0.0;
    zx   = ax + delta*(vh_x - uh_x)*inv_cr;
    zy   = ay + delta*(vh_y - uh_y)*inv_cr;
    phi0 = atan2(am_y - zy, am_x - zx);
}

// ---- wrap angle difference to (-π, π] ------------------------------------------
__device__ double wrapAngleDiff(double x) {
    x = fmod(x + 3.0*M_PI, 2.0*M_PI) - M_PI;
    if (x <= -M_PI) x += 2.0*M_PI;
    return x;
}

// ---- check whether angle phi lies inside arc [phi0, phi0+dphi] -----------------
__device__ double tauInArc(double phi, double phi0, double dphi) {
    if (fabs(dphi) < 1e-12) return -1.0;
    double diff = wrapAngleDiff(phi - phi0);
    double tau  = diff / dphi;
    return (tau > 1e-9 && tau < 1.0 - 1e-9) ? tau : -1.0;
}

// ---- point-in-smoothed-polygon (horizontal +x ray casting) ---------------------
// B geometry arrays are stored in the scratch buffer; accessed via base pointers.
__device__ bool pipSmoothed(
    double px, double py,
    const double* bam_x, const double* bam_y,
    const double* bap_x, const double* bap_y,
    const double* bdphi,  const double* bphi0,
    const double* bzx,    const double* bzy,
    double delta, int nb
) {
    int cnt = 0;
    for (int j = 0; j < nb; j++) {
        int jn = (j + 1) % nb;
        if (fabs(bdphi[j]) > 1e-12) {
            double dy   = py - bzy[j];
            double disc = delta*delta - dy*dy;
            if (disc > 0.0) {
                double sq = sqrt(disc);
                for (int sg = -1; sg <= 1; sg += 2) {
                    double xc = bzx[j] + sg*sq;
                    if (xc > px) {
                        double phi = atan2(dy, xc - bzx[j]);
                        if (tauInArc(phi, bphi0[j], bdphi[j]) >= 0.0) cnt++;
                    }
                }
            }
        }
        double x1 = bap_x[j], y1 = bap_y[j];
        double x2 = bam_x[jn], y2 = bam_y[jn];
        if ((y1 <= py && py < y2) || (y2 <= py && py < y1)) {
            double t = (py - y1) / (y2 - y1);
            if (x1 + t*(x2 - x1) > px) cnt++;
        }
    }
    return (cnt & 1) != 0;
}

// ---- arc-edge intersections (arc first, edge second) ---------------------------
__device__ int isectArcEdge(
    double zx,  double zy,  double delta, double phi0, double dphi,
    double px,  double py,  double qx,   double qy,
    double* tau_a, double* t_seg, double* Xx, double* Xy, int maxIsect
) {
    double ex = qx-px, ey = qy-py;
    double fx = px-zx, fy = py-zy;
    double a = ex*ex + ey*ey;
    if (a < 1e-28) return 0;
    double b = 2.0*(fx*ex + fy*ey);
    double c = fx*fx + fy*fy - delta*delta;
    double disc = b*b - 4.0*a*c;
    if (disc < 0.0) return 0;
    double sq = sqrt(disc);
    int cnt = 0;
    for (int sg = -1; sg <= 1; sg += 2) {
        double t = (-b + sg*sq) / (2.0*a);
        if (t <= 1e-9 || t >= 1.0-1e-9) continue;
        double Xxv = px + t*ex, Xyv = py + t*ey;
        double phi  = atan2(Xyv - zy, Xxv - zx);
        double tau  = tauInArc(phi, phi0, dphi);
        if (tau < 0.0) continue;
        if (cnt < maxIsect - 1) {
            tau_a[cnt] = tau;  t_seg[cnt] = t;
            Xx[cnt] = Xxv;    Xy[cnt] = Xyv;
            cnt++;
        }
    }
    return cnt;
}

// ---- arc-arc intersections -----------------------------------------------------
__device__ int isectArcArc(
    double zax, double zay, double da, double phi0a, double dphia,
    double zbx, double zby, double db, double phi0b, double dphib,
    double* taua, double* taub, double* Xx, double* Xy, int maxIsect
) {
    double sx = zbx-zax, sy = zby-zay;
    double dist2 = sx*sx + sy*sy;
    if (dist2 < 1e-28) return 0;
    double dist  = sqrt(dist2);
    double inv_d = 1.0/dist;
    double a2 = (dist2 + da*da - db*db) / (2.0*dist);
    double h2 = da*da - a2*a2;
    if (h2 < 0.0) return 0;
    double h   = sqrt(fmax(h2, 0.0));
    double mx  = zax + a2*sx*inv_d, my = zay + a2*sy*inv_d;
    double px2 = -sy*inv_d,         py2 = sx*inv_d;
    int cnt = 0;
    for (int sg = -1; sg <= 1; sg += 2) {
        if (h < 1e-14 && sg < 0) break;
        double Xxv = mx + sg*h*px2, Xyv = my + sg*h*py2;
        double pha = atan2(Xyv-zay, Xxv-zax);
        double phb = atan2(Xyv-zby, Xxv-zbx);
        double ta  = tauInArc(pha, phi0a, dphia);
        double tb  = tauInArc(phb, phi0b, dphib);
        if (ta < 0.0 || tb < 0.0) continue;
        if (cnt < maxIsect - 1) {
            taua[cnt] = ta;  taub[cnt] = tb;
            Xx[cnt] = Xxv;   Xy[cnt] = Xyv;
            cnt++;
        }
    }
    return cnt;
}

// ---- edge-edge intersection ----------------------------------------------------
__device__ bool isectEdgeEdge(
    double px, double py, double qx, double qy,
    double rx, double ry, double sx, double sy,
    double& t_out, double& s_out, double& Xx, double& Xy
) {
    double dx = qx-px, dy = qy-py;
    double ex = sx-rx, ey = sy-ry;
    double den = dx*ey - dy*ex;
    if (fabs(den) < 1e-14) return false;
    double fx = rx-px, fy = ry-py;
    t_out = (fx*ey - fy*ex) / den;
    s_out = (fx*dy - fy*dx) / den;
    if (t_out > 1e-9 && t_out < 1.0-1e-9 && s_out > 1e-9 && s_out < 1.0-1e-9) {
        Xx = px + t_out*dx;  Xy = py + t_out*dy;
        return true;
    }
    return false;
}

// ---- in-place insertion sort of 3 arrays by key --------------------------------
__device__ void sortByKey3(double* key, double* d1, double* d2, int n) {
    for (int i = 1; i < n; i++) {
        double k = key[i], v1 = d1[i], v2 = d2[i];
        int j = i - 1;
        while (j >= 0 && key[j] > k) {
            key[j+1] = key[j]; d1[j+1] = d1[j]; d2[j+1] = d2[j];
            j--;
        }
        key[j+1] = k; d1[j+1] = v1; d2[j+1] = v2;
    }
}

__device__ void sortByKey5(double* key, double* d1, double* d2, double* d3, double* d4, int n) {
    for (int i = 1; i < n; i++) {
        double k = key[i], v1 = d1[i], v2 = d2[i], v3 = d3[i], v4 = d4[i];
        int j = i - 1;
        while (j >= 0 && key[j] > k) {
            key[j+1] = key[j]; d1[j+1] = d1[j]; d2[j+1] = d2[j];
            d3[j+1] = d3[j]; d4[j+1] = d4[j];
            j--;
        }
        key[j+1] = k; d1[j+1] = v1; d2[j+1] = v2;
        d3[j+1] = v3; d4[j+1] = v4;
    }
}

__device__ void sortByKey7(double* key, double* d1, double* d2, double* d3, double* d4,
                           double* d5, double* d6, int n) {
    for (int i = 1; i < n; i++) {
        double k = key[i], v1 = d1[i], v2 = d2[i], v3 = d3[i], v4 = d4[i];
        double v5 = d5[i], v6 = d6[i];
        int j = i - 1;
        while (j >= 0 && key[j] > k) {
            key[j+1] = key[j]; d1[j+1] = d1[j]; d2[j+1] = d2[j];
            d3[j+1] = d3[j]; d4[j+1] = d4[j]; d5[j+1] = d5[j]; d6[j+1] = d6[j];
            j--;
        }
        key[j+1] = k; d1[j+1] = v1; d2[j+1] = v2;
        d3[j+1] = v3; d4[j+1] = v4; d5[j+1] = v5; d6[j+1] = v6;
    }
}

// ---- gradient of energy through arc geometry → 3 backbone vertices -------------
// Given dE/d(am_k), dE/d(ap_k), dE/d(dphi_k), compute forces on
// a_{k-1}(fp), a_k(fa), a_{k+1}(fn).
__device__ void arcGeomGrad(
    double uh_x, double uh_y, double vh_x, double vh_y,
    double u_len, double v_len,
    double cr, double dt, double ell, double delta,
    double fam_x, double fam_y,
    double fap_x, double fap_y,
    double f_dphi,
    double& fp_x, double& fp_y,
    double& fa_x, double& fa_y,
    double& fn_x, double& fn_y
) {
    fp_x = fp_y = fa_x = fa_y = fn_x = fn_y = 0.0;
    double abs_cr = fabs(cr);
    if (abs_cr < 1e-14 || u_len < 1e-14 || v_len < 1e-14) {
        fa_x = fam_x + fap_x;
        fa_y = fam_y + fap_y;
        return;
    }

    // d(ell)/d(u_hat) and d(v_hat)
    double sign_cr  = (cr >= 0.0) ? 1.0 : -1.0;
    double inv_1pdt = 1.0 / (1.0 + dt);
    double dell_du_x = inv_1pdt * ( sign_cr*delta*vh_y  - ell*vh_x);
    double dell_du_y = inv_1pdt * (-sign_cr*delta*vh_x  - ell*vh_y);
    double dell_dv_x = inv_1pdt * (-sign_cr*delta*uh_y  - ell*uh_x);
    double dell_dv_y = inv_1pdt * ( sign_cr*delta*uh_x  - ell*uh_y);

    // d(dphi)/d(u_hat), d(v_hat)  [using cr^2+dt^2=1 for unit vectors]
    double ddphi_duh_x =  dt*vh_y   - cr*vh_x;
    double ddphi_duh_y =  dt*(-vh_x) - cr*vh_y;
    double ddphi_dvh_x =  dt*(-uh_y) - cr*uh_x;
    double ddphi_dvh_y =  dt*uh_x    - cr*uh_y;

    double g_ell = -(fam_x*uh_x + fam_y*uh_y) + (fap_x*vh_x + fap_y*vh_y);

    double dE_duh_x = -ell*fam_x + g_ell*dell_du_x + f_dphi*ddphi_duh_x;
    double dE_duh_y = -ell*fam_y + g_ell*dell_du_y + f_dphi*ddphi_duh_y;
    double dE_dvh_x =  ell*fap_x + g_ell*dell_dv_x + f_dphi*ddphi_dvh_x;
    double dE_dvh_y =  ell*fap_y + g_ell*dell_dv_y + f_dphi*ddphi_dvh_y;

    double dot_u = dE_duh_x*uh_x + dE_duh_y*uh_y;
    double dE_du_x = (dE_duh_x - dot_u*uh_x) / u_len;
    double dE_du_y = (dE_duh_y - dot_u*uh_y) / u_len;

    double dot_v = dE_dvh_x*vh_x + dE_dvh_y*vh_y;
    double dE_dv_x = (dE_dvh_x - dot_v*vh_x) / v_len;
    double dE_dv_y = (dE_dvh_y - dot_v*vh_y) / v_len;

    fp_x = -dE_du_x;
    fp_y = -dE_du_y;
    fa_x = fam_x + fap_x + dE_du_x - dE_dv_x;
    fa_y = fam_y + fap_y + dE_du_y - dE_dv_y;
    fn_x = dE_dv_x;
    fn_y = dE_dv_y;
}

// ---- Backpropagate ∂E/∂z (arc center) through arc geometry to the three -------
// vertices (prev = i-1, cur = i, next = i+1) of the arc. Used to propagate the
// chord-term gradient at interior arc-edge crossings: after the implicit-
// function-theorem projection, dX_c/dv = dz/dv − T·(eB×dz/dv)/(eB·n_c), and the
// φ₀/dφ contributions cancel, so only dz/dv enters. Mirrors arcGeomGrad.
__device__ void zGrad(
    double uh_x, double uh_y, double vh_x, double vh_y,
    double u_len, double v_len,
    double cr, double delta,
    double fz_x, double fz_y,
    double& fp_x, double& fp_y,
    double& fa_x, double& fa_y,
    double& fn_x, double& fn_y
) {
    fp_x = fp_y = fa_x = fa_y = fn_x = fn_y = 0.0;
    double abs_cr = fabs(cr);
    if (abs_cr < 1e-14 || u_len < 1e-14 || v_len < 1e-14) {
        // degenerate (near-straight vertex): z collapses onto v
        fa_x = fz_x; fa_y = fz_y;
        return;
    }
    double sign_cr = (cr >= 0.0) ? 1.0 : -1.0;
    double inv  = 1.0 / abs_cr;
    double inv2 = inv * inv;

    // z = v + δ·(v̂ − û)·inv, so ∂z/∂v = I, ∂z/∂û and ∂z/∂v̂ via product rule.
    double fz_dot_dv = fz_x*(vh_x - uh_x) + fz_y*(vh_y - uh_y);

    double dE_duh_x = -delta*inv*fz_x - delta*sign_cr*vh_y*inv2*fz_dot_dv;
    double dE_duh_y = -delta*inv*fz_y + delta*sign_cr*vh_x*inv2*fz_dot_dv;
    double dE_dvh_x =  delta*inv*fz_x + delta*sign_cr*uh_y*inv2*fz_dot_dv;
    double dE_dvh_y =  delta*inv*fz_y - delta*sign_cr*uh_x*inv2*fz_dot_dv;

    // Project onto the unit-norm tangent space (perpendicular to û, v̂)
    double dot_u = dE_duh_x*uh_x + dE_duh_y*uh_y;
    double dE_du_x = (dE_duh_x - dot_u*uh_x) / u_len;
    double dE_du_y = (dE_duh_y - dot_u*uh_y) / u_len;

    double dot_v = dE_dvh_x*vh_x + dE_dvh_y*vh_y;
    double dE_dv_x = (dE_dvh_x - dot_v*vh_x) / v_len;
    double dE_dv_y = (dE_dvh_y - dot_v*vh_y) / v_len;

    // Chain to vertices: u = v_cur − v_prev, v = v_next − v_cur
    fp_x = -dE_du_x;
    fp_y = -dE_du_y;
    fa_x = fz_x + dE_du_x - dE_dv_x;
    fa_y = fz_y + dE_du_y - dE_dv_y;
    fn_x = dE_dv_x;
    fn_y = dE_dv_y;
}

// ---- Backpropagate a tangent-point gradient through B's arc geometry -----------
// Gradient w.r.t. bap_jj (is_ap=true) or bam_jj (is_ap=false) is propagated
// to the three backbone vertices (prev, jj, next) of polygon B.
// Forces (= -gradient) are added to the global force array via atomicAdd.
__device__ void applyTangentGradToB(
    bool is_ap,
    int jj, int n_B, int startB,
    double ftan_x, double ftan_y,
    const double* bx_pos, const double* by_pos,
    double delta, double* force
) {
    int jjp = (jj - 1 + n_B) % n_B;
    int jjn = (jj + 1) % n_B;
    double bux = bx_pos[jj] - bx_pos[jjp], buy = by_pos[jj] - by_pos[jjp];
    double bvx = bx_pos[jjn] - bx_pos[jj], bvy = by_pos[jjn] - by_pos[jj];
    double _uh, _uhy, _vh, _vhy, _ul, _vl, _cr, _dt, _dp, _el;
    double _amx, _amy, _apx, _apy, _zx, _zy, _phi0;
    computeArcGeom(bx_pos[jj], by_pos[jj], bux, buy, bvx, bvy, delta,
                   _uh, _uhy, _vh, _vhy, _ul, _vl, _cr, _dt, _dp, _el,
                   _amx, _amy, _apx, _apy, _zx, _zy, _phi0);
    double fp_x, fp_y, fa_x, fa_y, fn_x, fn_y;
    if (is_ap)
        arcGeomGrad(_uh, _uhy, _vh, _vhy, _ul, _vl, _cr, _dt, _el, delta,
                    0.0, 0.0, ftan_x, ftan_y, 0.0,
                    fp_x, fp_y, fa_x, fa_y, fn_x, fn_y);
    else
        arcGeomGrad(_uh, _uhy, _vh, _vhy, _ul, _vl, _cr, _dt, _el, delta,
                    ftan_x, ftan_y, 0.0, 0.0, 0.0,
                    fp_x, fp_y, fa_x, fa_y, fn_x, fn_y);
    if (fp_x != 0.0) atomicAdd(&force[2*(startB+jjp)],   -fp_x);
    if (fp_y != 0.0) atomicAdd(&force[2*(startB+jjp)+1], -fp_y);
    if (fa_x != 0.0) atomicAdd(&force[2*(startB+jj)],    -fa_x);
    if (fa_y != 0.0) atomicAdd(&force[2*(startB+jj)+1],  -fa_y);
    if (fn_x != 0.0) atomicAdd(&force[2*(startB+jjn)],   -fn_x);
    if (fn_y != 0.0) atomicAdd(&force[2*(startB+jjn)+1], -fn_y);
}

// ---- Main rounded-polygon kernel -----------------------------------------------
// Thread i owns arc K_i (a_i^- → a_i^+) and edge E_i (a_i^+ → a_{next[i]}^-).
// scratch: pre-allocated device buffer, stride (doubles) = 30 * polygonSize.
// polygonSize: max vertices per polygon (runtime parameter).
// maxProc: max distinct neighbour polygons tracked per thread.
__global__ void updateForceEnergyRoundedKernel(
    int numVertices,
    const double* __restrict__ positions,
    const int*    __restrict__ shapeId,
    const int*    __restrict__ next,
    const int*    __restrict__ prev,
    const int*    __restrict__ startIndices,
    const int*    __restrict__ ballNeighbors,    // per-vertex flat candidate list
    const int*    __restrict__ numBallNeighbors, // per-vertex count
    int ballMaxNeighbors,                        // candidate-list stride
    double delta,
    int polygonSize,     // max vertices per polygon (= scratch stride / 30)
    int maxProc,         // max distinct neighbour shapes per thread (≤ polygonSize)
    double* __restrict__ scratch,  // size = numVertices * 30 * polygonSize doubles
    double* __restrict__ force,
    double* __restrict__ energy,
    double* __restrict__ pairAreaAccum,   // pass 1: accumulate area per (sA,sB); null = skip
    const double* __restrict__ pairAreaScale, // pass 2: scale forces by area; null = skip
    int numPolygons)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVertices) return;

    int sA    = shapeId[i];
    int pi_i  = prev[i];
    int ni_i  = next[i];
    int nni_i = next[ni_i];

    // --- Arc geometry at vertex i ---
    double ax = positions[2*i],     ay = positions[2*i+1];
    double px = positions[2*pi_i],  py = positions[2*pi_i+1];
    double nx = ax + wrap(positions[2*ni_i]   - ax);
    double ny = ay + wrap(positions[2*ni_i+1] - ay);
    double ux_i = wrap(ax - px), uy_i = wrap(ay - py);
    double vx_i = nx - ax, vy_i = ny - ay;

    double uh_x, uh_y, vh_x, vh_y, u_len, v_len;
    double cr_i, dt_i, dphi_i, ell_i;
    double am_x, am_y, ap_x, ap_y, z_x, z_y, phi0_i;
    computeArcGeom(ax, ay, ux_i, uy_i, vx_i, vy_i, delta,
                   uh_x, uh_y, vh_x, vh_y, u_len, v_len,
                   cr_i, dt_i, dphi_i, ell_i,
                   am_x, am_y, ap_x, ap_y, z_x, z_y, phi0_i);

    // --- Arc geometry at vertex next[i] (for edge E_i's far end a_{i+1}^-) ---
    double nnx = nx + wrap(positions[2*nni_i]   - positions[2*ni_i]);
    double nny = ny + wrap(positions[2*nni_i+1] - positions[2*ni_i+1]);
    double vx_n = nnx - nx, vy_n = nny - ny;

    double uh_nx, uh_ny, vh_nx, vh_ny, u_len_n, v_len_n;
    double cr_n, dt_n, dphi_n, ell_n;
    double am_nx, am_ny, ap_nx, ap_ny, z_nx, z_ny, phi0_n;
    computeArcGeom(nx, ny, vx_i, vy_i, vx_n, vy_n, delta,
                   uh_nx, uh_ny, vh_nx, vh_ny, u_len_n, v_len_n,
                   cr_n, dt_n, dphi_n, ell_n,
                   am_nx, am_ny, ap_nx, ap_ny, z_nx, z_ny, phi0_n);

    // Thread-global force accumulators: [0]=prev[i], [1]=i, [2]=next[i], [3]=next[next[i]]
    double glf_x[4] = {0,0,0,0}, glf_y[4] = {0,0,0,0};
    double localEnergy = 0.0;

    // Scratch pointer for this thread
    int n = polygonSize;
    int maxIsect = 2*n + 2;
    double* base = scratch + (long long)i * 38 * n;

    double* bx_pos = base + 0*n;
    double* by_pos = base + 1*n;
    double* bam_xS = base + 2*n;
    double* bam_yS = base + 3*n;
    double* bap_xS = base + 4*n;
    double* bap_yS = base + 5*n;
    double* bz_xS  = base + 6*n;
    double* bz_yS  = base + 7*n;
    double* bdphiS = base + 8*n;
    double* bphi0S = base + 9*n;
    // Arc crossing storage: maxIsect = 2n+2 ≤ 4n (fits in 4n slots)
    double* arc_tau = base + 10*n;
    double* arc_Xx  = base + 14*n;
    double* arc_Xy  = base + 18*n;
    // Edge crossing storage (reuse arc slots, processed sequentially)
    double* edg_t   = base + 10*n;
    double* edg_Xx  = base + 14*n;
    double* edg_Xy  = base + 18*n;
    // B-feature tangent at each edge crossing (0 for arc crossings / sentinels)
    double* edg_eBx = base + 22*n;
    double* edg_eBy = base + 26*n;
    // Coupling data: parameter on B's edge (s_B) and B vertex index (jj, as double; -1 for non-edge)
    double* edg_sB  = base + 30*n;
    double* edg_jj  = base + 34*n;

    // Neighbour-shape deduplication
    int processedShapes[32];   // stack-local: small, bounded by typical fan-out
    int numProcessed = 0;
    // We bound this to min(maxProc, 32) at runtime
    int maxProcBound = (maxProc < 32) ? maxProc : 32;

    // Walk the per-vertex candidate list (populated by either updateNeighborCells
    // via populateCandidatesFromCellsKernel, or by buildBallListKernel directly).
    int nCand = numBallNeighbors[i];
    if (nCand > ballMaxNeighbors) nCand = ballMaxNeighbors;
    int candBase = i * ballMaxNeighbors;
    for (int k_n = 0; k_n < nCand; k_n++) {
            int j  = ballNeighbors[candBase + k_n];
            int sB = shapeId[j];
            if (sB == sA) continue;

            bool seen = false;
            for (int k = 0; k < numProcessed; k++)
                if (processedShapes[k] == sB) { seen = true; break; }
            if (seen) continue;
            if (numProcessed < maxProcBound)
                processedShapes[numProcessed++] = sB;

                // Per-(sA,sB) accumulators for area² mode
                double lf_x[4] = {0,0,0,0}, lf_y[4] = {0,0,0,0};
                double pairEnergy = 0.0;
                bool skip_forces = (pairAreaAccum != nullptr);
                double area_scale = (pairAreaScale != nullptr)
                    ? pairAreaScale[sA * numPolygons + sB] + pairAreaScale[sB * numPolygons + sA]
                    : 1.0;
                // Per-pair anchor for the Green's-theorem terms. Every vertex
                // thread of A and of B uses this same reference, so the
                // origin-anchored chord terms telescope correctly even when a
                // polygon straddles the periodic box (MIC via wrap()).
                int sRef = (sA < sB) ? sA : sB;
                double sx = positions[2*startIndices[sRef]];
                double sy = positions[2*startIndices[sRef] + 1];

                int startB = startIndices[sB];
                int endB   = startIndices[sB + 1];
                int n_B    = endB - startB;
                if (n_B > n || n_B < 3) continue;

                // Load B's positions: place all vertices in the periodic image
                // nearest to vertex i (ax,ay).  B's first vertex is the MIC
                // reference; each subsequent vertex is offset by a MIC-wrapped
                // difference so that polygons stored with per-vertex % 1.0
                // wrapping (i.e. split across the PBC) are reassembled correctly.
                {
                    double ref_bx = positions[2*startB];
                    double ref_by = positions[2*startB+1];
                    double bx_ref_mic = ax + wrap(ref_bx - ax);
                    double by_ref_mic = ay + wrap(ref_by - ay);
                    for (int jj = 0; jj < n_B; jj++) {
                        int vid = startB + jj;
                        bx_pos[jj] = bx_ref_mic + wrap(positions[2*vid]   - ref_bx);
                        by_pos[jj] = by_ref_mic + wrap(positions[2*vid+1] - ref_by);
                    }
                }

                // Compute arc geometry for B
                for (int jj = 0; jj < n_B; jj++) {
                    int jjp = (jj - 1 + n_B) % n_B;
                    int jjn = (jj + 1) % n_B;
                    // Edge vectors within the consistent MIC frame (no further wrapping needed)
                    double bux = bx_pos[jj] - bx_pos[jjp];
                    double buy = by_pos[jj] - by_pos[jjp];
                    double bvx = bx_pos[jjn] - bx_pos[jj];
                    double bvy = by_pos[jjn] - by_pos[jj];
                    double _uh, _uh2, _vh, _vh2, _ul, _vl, _cr, _dt, _dp, _el;
                    computeArcGeom(bx_pos[jj], by_pos[jj], bux, buy, bvx, bvy, delta,
                                   _uh, _uh2, _vh, _vh2, _ul, _vl,
                                   _cr, _dt, _dp, _el,
                                   bam_xS[jj], bam_yS[jj], bap_xS[jj], bap_yS[jj],
                                   bz_xS[jj], bz_yS[jj], bphi0S[jj]);
                    bdphiS[jj] = _dp;
                }

                // ==================================================
                // Feature 1: ARC K_i  (a_i^- to a_i^+ via circle)
                // ==================================================
                if (fabs(dphi_i) > 1e-12) {
                    // Reuse Feature 2's scratch slots for per-crossing metadata:
                    //   arc_eB     = B-boundary tangent at X_c (for the A-side projection)
                    //   arc_sB     = parameter on partner's feature (s for edge, τ for arc)
                    //   arc_jj_enc = partner-feature index, encoded:
                    //                  [0, n_B)        → partner-edge index jj
                    //                  [n_B, 2·n_B)    → partner-arc index (jj_enc − n_B)
                    //                  < 0             → sentinel / no-propagate
                    double* arc_eBx     = edg_eBx;
                    double* arc_eBy     = edg_eBy;
                    double* arc_sB      = edg_sB;
                    double* arc_jj_enc  = edg_jj;
                    int nc = 0;
                    for (int jj = 0; jj < n_B && nc < maxIsect - 3; jj++) {
                        int jjn = (jj + 1) % n_B;
                        double ts2[2], ta2[2];
                        // K_i vs edge E_{jj}: bap[jj] → bam[jjn]
                        int c1 = isectArcEdge(z_x, z_y, delta, phi0_i, dphi_i,
                                              bap_xS[jj], bap_yS[jj],
                                              bam_xS[jjn], bam_yS[jjn],
                                              arc_tau+nc, ts2, arc_Xx+nc, arc_Xy+nc,
                                              maxIsect - nc);
                        double eBx_jj = bam_xS[jjn] - bap_xS[jj];
                        double eBy_jj = bam_yS[jjn] - bap_yS[jj];
                        for (int kc = 0; kc < c1; kc++) {
                            arc_eBx[nc+kc]    = eBx_jj;
                            arc_eBy[nc+kc]    = eBy_jj;
                            arc_sB[nc+kc]     = ts2[kc];        // s on partner edge
                            arc_jj_enc[nc+kc] = (double)jj;     // partner-edge index
                        }
                        nc += c1;
                        // K_i vs arc K_{jj}
                        if (fabs(bdphiS[jj]) > 1e-12 && nc < maxIsect - 3) {
                            int c2 = isectArcArc(z_x, z_y, delta, phi0_i, dphi_i,
                                                 bz_xS[jj], bz_yS[jj], delta,
                                                 bphi0S[jj], bdphiS[jj],
                                                 arc_tau+nc, ta2, arc_Xx+nc, arc_Xy+nc,
                                                 maxIsect - nc);
                            for (int kc = 0; kc < c2; kc++) {
                                double dx = arc_Xx[nc+kc] - bz_xS[jj];
                                double dy = arc_Xy[nc+kc] - bz_yS[jj];
                                arc_eBx[nc+kc]    = -dy;
                                arc_eBy[nc+kc]    =  dx;
                                arc_sB[nc+kc]     = ta2[kc];                  // τ on partner arc
                                arc_jj_enc[nc+kc] = (double)(n_B + jj);        // partner-arc index, offset
                            }
                            nc += c2;
                        }
                    }
                    // Sentinel endpoints (t=0: am_i, t=1: ap_i)
                    arc_tau[nc] = 0.0; arc_Xx[nc] = am_x; arc_Xy[nc] = am_y;
                    arc_eBx[nc] = 0.0; arc_eBy[nc] = 0.0;
                    arc_sB[nc]  = 0.0; arc_jj_enc[nc] = -1.0; nc++;
                    arc_tau[nc] = 1.0; arc_Xx[nc] = ap_x; arc_Xy[nc] = ap_y;
                    arc_eBx[nc] = 0.0; arc_eBy[nc] = 0.0;
                    arc_sB[nc]  = 0.0; arc_jj_enc[nc] = -1.0; nc++;

                    sortByKey7(arc_tau, arc_Xx, arc_Xy, arc_eBx, arc_eBy,
                               arc_sB, arc_jj_enc, nc);

                    for (int kk = 0; kk < nc - 1; kk++) {
                        double t0 = arc_tau[kk],    t1 = arc_tau[kk+1];
                        if (t1 - t0 < 1e-12) continue;
                        double X0x = arc_Xx[kk],   X0y = arc_Xy[kk];
                        double X1x = arc_Xx[kk+1], X1y = arc_Xy[kk+1];

                        double tm    = 0.5*(t0 + t1);
                        double phi_m = phi0_i + tm*dphi_i;
                        double sig   = (dphi_i > 0.0) ? 1.0 : -1.0;
                        double mid_x = z_x + delta*cos(phi_m);
                        double mid_y = z_y + delta*sin(phi_m);
                        // Perturb midpoint inward (toward z for convex arc)
                        double inw_x = sig*(z_x - mid_x);
                        double inw_y = sig*(z_y - mid_y);
                        double inw_n = sqrt(inw_x*inw_x + inw_y*inw_y);
                        if (inw_n > 1e-14) { inw_x /= inw_n; inw_y /= inw_n; }

                        if (!pipSmoothed(mid_x + 1e-8*inw_x, mid_y + 1e-8*inw_y,
                                         bam_xS, bam_yS, bap_xS, bap_yS,
                                         bdphiS, bphi0S, bz_xS, bz_yS, delta, n_B))
                            continue;

                        double dpsi  = (t1 - t0)*dphi_i;
                        double r0x = wrap(X0x - sx), r0y = wrap(X0y - sy);
                        double r1x = wrap(X1x - sx), r1y = wrap(X1y - sy);
                        pairEnergy += 0.5*(r0x*r1y - r0y*r1x)
                                     + 0.5*delta*delta*(dpsi - sin(dpsi));

                        double f0x =  0.5*r1y, f0y = -0.5*r1x;
                        double f1x = -0.5*r0y, f1y =  0.5*r0x;
                        double fdp  = 0.5*delta*delta*(1.0 - cos(dpsi));

                        // Sentinel endpoint t≈0 (X0 = am): standard arcGeomGrad chain.
                        // Interior arc-edge crossing at t0: use zGrad with the
                        // implicit-function-theorem projection f_eff = f0 −
                        // (f0·T)·perp(eB)/(eB·n_c).  φ₀ / dφ contributions cancel,
                        // leaving only the ∂z/∂v chain (see zGrad).
                        if (!skip_forces && t0 < 1e-10) {
                            double gp, gq, ga, gb, gn, gm;
                            arcGeomGrad(uh_x, uh_y, vh_x, vh_y, u_len, v_len,
                                        cr_i, dt_i, ell_i, delta,
                                        f0x, f0y, 0.0, 0.0, 0.0,
                                        gp, gq, ga, gb, gn, gm);
                            lf_x[0] += gp; lf_y[0] += gq;
                            lf_x[1] += ga; lf_y[1] += gb;
                            lf_x[2] += gn; lf_y[2] += gm;
                        } else if (!skip_forces && (arc_eBx[kk] != 0.0 || arc_eBy[kk] != 0.0)) {
                            // Interior arc-edge or arc-arc crossing at X0.
                            double eBx = arc_eBx[kk], eBy = arc_eBy[kk];
                            double phi_c = phi0_i + t0*dphi_i;
                            double cos_c = cos(phi_c), sin_c = sin(phi_c);
                            double Tx = -sin_c, Ty = cos_c;     // arc tangent
                            double nx =  cos_c, ny = sin_c;     // arc outward normal
                            double eBdotn = eBx*nx + eBy*ny;
                            if (fabs(eBdotn) > 1e-14) {
                                // ── A-side: propagate to A's arc vertices via zGrad ──
                                double scale = (f0x*Tx + f0y*Ty) / eBdotn;
                                double fz_x = f0x + scale*eBy;
                                double fz_y = f0y - scale*eBx;
                                double gp, gq, ga, gb, gn, gm;
                                zGrad(uh_x, uh_y, vh_x, vh_y, u_len, v_len,
                                      cr_i, delta, fz_x, fz_y,
                                      gp, gq, ga, gb, gn, gm);
                                lf_x[0] += gp; lf_y[0] += gq;
                                lf_x[1] += ga; lf_y[1] += gb;
                                lf_x[2] += gn; lf_y[2] += gm;

                                // ── B-side: propagate to partner's vertices ──
                                double jj_enc = arc_jj_enc[kk];
                                if (jj_enc >= 0.0 && jj_enc < (double)n_B) {
                                    // Partner-edge propagation: ∂E/∂P_B = (1−sB)·f_pe,
                                    //   ∂E/∂Q_B = sB·f_pe,  f_pe = f0 − g·n_c_A,
                                    //   g = (f0·eB)/(eB·n_c_A).
                                    double g_pe = (f0x*eBx + f0y*eBy) / eBdotn;
                                    double fpe_x = f0x - g_pe * nx;
                                    double fpe_y = f0y - g_pe * ny;
                                    double sB    = arc_sB[kk];
                                    int jj_pe  = (int)jj_enc;
                                    int jj_pen = (jj_pe + 1) % n_B;
                                    applyTangentGradToB(true,  jj_pe,  n_B, startB,
                                                        area_scale*(1.0-sB)*fpe_x,
                                                        area_scale*(1.0-sB)*fpe_y,
                                                        bx_pos, by_pos, delta, force);
                                    applyTangentGradToB(false, jj_pen, n_B, startB,
                                                        area_scale*sB*fpe_x,
                                                        area_scale*sB*fpe_y,
                                                        bx_pos, by_pos, delta, force);
                                } else if (jj_enc >= (double)n_B) {
                                    // Partner-arc propagation: zGrad on partner's K_jj_p.
                                    int jj_p  = (int)(jj_enc - (double)n_B);
                                    int jjp_p = (jj_p - 1 + n_B) % n_B;
                                    int jjn_p = (jj_p + 1) % n_B;
                                    double bux = bx_pos[jj_p]  - bx_pos[jjp_p];
                                    double buy = by_pos[jj_p]  - by_pos[jjp_p];
                                    double bvx = bx_pos[jjn_p] - bx_pos[jj_p];
                                    double bvy = by_pos[jjn_p] - by_pos[jj_p];
                                    double bul = sqrt(bux*bux + buy*buy);
                                    double bvl = sqrt(bvx*bvx + bvy*bvy);
                                    double TA_dot_n_p = Tx*eBy - Ty*eBx;
                                    if (bul > 1e-14 && bvl > 1e-14 && fabs(TA_dot_n_p) > 1e-14) {
                                        double buh_x = bux/bul, buh_y = buy/bul;
                                        double bvh_x = bvx/bvl, bvh_y = bvy/bvl;
                                        double bcr   = buh_x*bvh_y - buh_y*bvh_x;
                                        double g0_p  = (f0x*Tx + f0y*Ty) / TA_dot_n_p;
                                        double fzp_x = area_scale * g0_p *  eBy;
                                        double fzp_y = area_scale * g0_p * (-eBx);
                                        double pgp, pgq, pga, pgb, pgn, pgm;
                                        zGrad(buh_x, buh_y, bvh_x, bvh_y, bul, bvl,
                                              bcr, delta, fzp_x, fzp_y,
                                              pgp, pgq, pga, pgb, pgn, pgm);
                                        atomicAdd(&force[2*(startB + jjp_p)],     -pgp);
                                        atomicAdd(&force[2*(startB + jjp_p) + 1], -pgq);
                                        atomicAdd(&force[2*(startB + jj_p)],      -pga);
                                        atomicAdd(&force[2*(startB + jj_p) + 1],  -pgb);
                                        atomicAdd(&force[2*(startB + jjn_p)],     -pgn);
                                        atomicAdd(&force[2*(startB + jjn_p) + 1], -pgm);
                                    }
                                }
                            }
                        }
                        if (!skip_forces && t1 > 1.0-1e-10) {
                            double full = (t0 < 1e-10) ? fdp : 0.0;
                            double gp, gq, ga, gb, gn, gm;
                            arcGeomGrad(uh_x, uh_y, vh_x, vh_y, u_len, v_len,
                                        cr_i, dt_i, ell_i, delta,
                                        0.0, 0.0, f1x, f1y, full,
                                        gp, gq, ga, gb, gn, gm);
                            lf_x[0] += gp; lf_y[0] += gq;
                            lf_x[1] += ga; lf_y[1] += gb;
                            lf_x[2] += gn; lf_y[2] += gm;
                        } else if (!skip_forces && (arc_eBx[kk+1] != 0.0 || arc_eBy[kk+1] != 0.0)) {
                            // Interior arc-edge or arc-arc crossing at X1.
                            double eBx = arc_eBx[kk+1], eBy = arc_eBy[kk+1];
                            double phi_c = phi0_i + t1*dphi_i;
                            double cos_c = cos(phi_c), sin_c = sin(phi_c);
                            double Tx = -sin_c, Ty = cos_c;
                            double nx =  cos_c, ny = sin_c;
                            double eBdotn = eBx*nx + eBy*ny;
                            if (fabs(eBdotn) > 1e-14) {
                                // ── A-side ──
                                double scale = (f1x*Tx + f1y*Ty) / eBdotn;
                                double fz_x = f1x + scale*eBy;
                                double fz_y = f1y - scale*eBx;
                                double gp, gq, ga, gb, gn, gm;
                                zGrad(uh_x, uh_y, vh_x, vh_y, u_len, v_len,
                                      cr_i, delta, fz_x, fz_y,
                                      gp, gq, ga, gb, gn, gm);
                                lf_x[0] += gp; lf_y[0] += gq;
                                lf_x[1] += ga; lf_y[1] += gb;
                                lf_x[2] += gn; lf_y[2] += gm;

                                // ── B-side ──
                                double jj_enc = arc_jj_enc[kk+1];
                                if (jj_enc >= 0.0 && jj_enc < (double)n_B) {
                                    // Partner-edge propagation
                                    double g_pe = (f1x*eBx + f1y*eBy) / eBdotn;
                                    double fpe_x = f1x - g_pe * nx;
                                    double fpe_y = f1y - g_pe * ny;
                                    double sB    = arc_sB[kk+1];
                                    int jj_pe  = (int)jj_enc;
                                    int jj_pen = (jj_pe + 1) % n_B;
                                    applyTangentGradToB(true,  jj_pe,  n_B, startB,
                                                        area_scale*(1.0-sB)*fpe_x,
                                                        area_scale*(1.0-sB)*fpe_y,
                                                        bx_pos, by_pos, delta, force);
                                    applyTangentGradToB(false, jj_pen, n_B, startB,
                                                        area_scale*sB*fpe_x,
                                                        area_scale*sB*fpe_y,
                                                        bx_pos, by_pos, delta, force);
                                } else if (jj_enc >= (double)n_B) {
                                    // Partner-arc propagation
                                    int jj_p  = (int)(jj_enc - (double)n_B);
                                    int jjp_p = (jj_p - 1 + n_B) % n_B;
                                    int jjn_p = (jj_p + 1) % n_B;
                                    double bux = bx_pos[jj_p]  - bx_pos[jjp_p];
                                    double buy = by_pos[jj_p]  - by_pos[jjp_p];
                                    double bvx = bx_pos[jjn_p] - bx_pos[jj_p];
                                    double bvy = by_pos[jjn_p] - by_pos[jj_p];
                                    double bul = sqrt(bux*bux + buy*buy);
                                    double bvl = sqrt(bvx*bvx + bvy*bvy);
                                    double TA_dot_n_p = Tx*eBy - Ty*eBx;
                                    if (bul > 1e-14 && bvl > 1e-14 && fabs(TA_dot_n_p) > 1e-14) {
                                        double buh_x = bux/bul, buh_y = buy/bul;
                                        double bvh_x = bvx/bvl, bvh_y = bvy/bvl;
                                        double bcr   = buh_x*bvh_y - buh_y*bvh_x;
                                        double g1_p  = (f1x*Tx + f1y*Ty) / TA_dot_n_p;
                                        double fzp_x = area_scale * g1_p *  eBy;
                                        double fzp_y = area_scale * g1_p * (-eBx);
                                        double pgp, pgq, pga, pgb, pgn, pgm;
                                        zGrad(buh_x, buh_y, bvh_x, bvh_y, bul, bvl,
                                              bcr, delta, fzp_x, fzp_y,
                                              pgp, pgq, pga, pgb, pgn, pgm);
                                        atomicAdd(&force[2*(startB + jjp_p)],     -pgp);
                                        atomicAdd(&force[2*(startB + jjp_p) + 1], -pgq);
                                        atomicAdd(&force[2*(startB + jj_p)],      -pga);
                                        atomicAdd(&force[2*(startB + jj_p) + 1],  -pgb);
                                        atomicAdd(&force[2*(startB + jjn_p)],     -pgn);
                                        atomicAdd(&force[2*(startB + jjn_p) + 1], -pgm);
                                    }
                                }
                            }
                        }
                    }
                }

                // ==================================================
                // Feature 2: EDGE E_i  (a_i^+ to a_{i+1}^-)
                // ==================================================
                {
                    double P_x = ap_x, P_y = ap_y;
                    double Q_x = am_nx, Q_y = am_ny;
                    int nc2 = 0;

                    for (int jj = 0; jj < n_B && nc2 < maxIsect - 3; jj++) {
                        int jjn = (jj + 1) % n_B;
                        double t_ee, s_ee, Xx_ee, Xy_ee;
                        if (isectEdgeEdge(P_x, P_y, Q_x, Q_y,
                                          bap_xS[jj], bap_yS[jj],
                                          bam_xS[jjn], bam_yS[jjn],
                                          t_ee, s_ee, Xx_ee, Xy_ee)) {
                            edg_t[nc2] = t_ee;
                            edg_Xx[nc2] = Xx_ee; edg_Xy[nc2] = Xy_ee;
                            edg_eBx[nc2] = bam_xS[jjn] - bap_xS[jj];
                            edg_eBy[nc2] = bam_yS[jjn] - bap_yS[jj];
                            edg_sB[nc2]  = s_ee;
                            edg_jj[nc2]  = (double)jj;
                            nc2++;
                        }
                        if (fabs(bdphiS[jj]) > 1e-12 && nc2 < maxIsect - 3) {
                            double ta3[2], ts3[2], Xxa3[2], Xya3[2];
                            // Segment E_i vs arc K_{jj}: call isectArcEdge with arc=K_{jj}, seg=E_i
                            int c3 = isectArcEdge(bz_xS[jj], bz_yS[jj], delta,
                                                  bphi0S[jj], bdphiS[jj],
                                                  P_x, P_y, Q_x, Q_y,
                                                  ta3, ts3, Xxa3, Xya3,
                                                  maxIsect - nc2);
                            for (int kk = 0; kk < c3 && nc2 < maxIsect - 2; kk++) {
                                edg_t[nc2] = ts3[kk]; // ts3 is the segment param
                                edg_Xx[nc2] = Xxa3[kk]; edg_Xy[nc2] = Xya3[kk];
                                // For edge-arc crossings the B-side tangent at X_c
                                // is perp(X_c − bz_B). The A-side β-correction
                                // uses this in place of the B-edge vector; the
                                // formula is otherwise unchanged.
                                double dxB = Xxa3[kk] - bz_xS[jj];
                                double dyB = Xya3[kk] - bz_yS[jj];
                                edg_eBx[nc2] = -dyB;
                                edg_eBy[nc2] =  dxB;
                                edg_sB[nc2]  = 0.0;
                                // Encode partner-arc index with +n_B offset so the
                                // dispatcher in the β-correction can route to the
                                // partner-arc propagation (zGrad on partner's K_jj)
                                // instead of applyTangentGradToB.
                                edg_jj[nc2]  = (double)(n_B + jj);
                                nc2++;
                            }
                        }
                    }
                    edg_t[nc2] = 0.0; edg_Xx[nc2] = P_x; edg_Xy[nc2] = P_y;
                    edg_eBx[nc2] = 0.0; edg_eBy[nc2] = 0.0;
                    edg_sB[nc2]  = 0.0; edg_jj[nc2]  = -1.0; nc2++;
                    edg_t[nc2] = 1.0; edg_Xx[nc2] = Q_x; edg_Xy[nc2] = Q_y;
                    edg_eBx[nc2] = 0.0; edg_eBy[nc2] = 0.0;
                    edg_sB[nc2]  = 0.0; edg_jj[nc2]  = -1.0; nc2++;

                    sortByKey7(edg_t, edg_Xx, edg_Xy, edg_eBx, edg_eBy, edg_sB, edg_jj, nc2);

                    double eVx = Q_x-P_x, eVy = Q_y-P_y;
                    double eLen = sqrt(eVx*eVx + eVy*eVy);
                    double lnx = 0.0, lny = 0.0;
                    if (eLen > 1e-14) { lnx = -eVy/eLen; lny = eVx/eLen; }

                    for (int kk = 0; kk < nc2 - 1; kk++) {
                        double t0 = edg_t[kk],    t1 = edg_t[kk+1];
                        if (t1 - t0 < 1e-12) continue;
                        double X0x = edg_Xx[kk],   X0y = edg_Xy[kk];
                        double X1x = edg_Xx[kk+1], X1y = edg_Xy[kk+1];
                        double mid_x = 0.5*(X0x+X1x) + 1e-8*lnx;
                        double mid_y = 0.5*(X0y+X1y) + 1e-8*lny;

                        if (!pipSmoothed(mid_x, mid_y,
                                         bam_xS, bam_yS, bap_xS, bap_yS,
                                         bdphiS, bphi0S, bz_xS, bz_yS, delta, n_B))
                            continue;

                        double r0x = wrap(X0x - sx), r0y = wrap(X0y - sy);
                        double r1x = wrap(X1x - sx), r1y = wrap(X1y - sy);
                        pairEnergy += 0.5*(r0x*r1y - r0y*r1x);

                        if (skip_forces) continue;

                        double f0x =  0.5*r1y, f0y = -0.5*r1x;
                        double f1x = -0.5*r0y, f1y =  0.5*r0x;

                        // Direct term: derivative holding integration limits fixed.
                        // Force on ap_i (t=0 end): weight = (1-t0) for f0, (1-t1) for f1
                        {
                            double gfap_x = (1.0-t0)*f0x + (1.0-t1)*f1x;
                            double gfap_y = (1.0-t0)*f0y + (1.0-t1)*f1y;
                            double gp, gq, ga, gb, gn, gm;
                            arcGeomGrad(uh_x, uh_y, vh_x, vh_y, u_len, v_len,
                                        cr_i, dt_i, ell_i, delta,
                                        0.0, 0.0, gfap_x, gfap_y, 0.0,
                                        gp, gq, ga, gb, gn, gm);
                            lf_x[0] += gp; lf_y[0] += gq;
                            lf_x[1] += ga; lf_y[1] += gb;
                            lf_x[2] += gn; lf_y[2] += gm;
                        }
                        // Force on am_{i+1} (t=1 end): weight = t0 for f0, t1 for f1
                        {
                            double gfam_x = t0*f0x + t1*f1x;
                            double gfam_y = t0*f0y + t1*f1y;
                            double gp, gq, ga, gb, gn, gm;
                            arcGeomGrad(uh_nx, uh_ny, vh_nx, vh_ny, u_len_n, v_len_n,
                                        cr_n, dt_n, ell_n, delta,
                                        gfam_x, gfam_y, 0.0, 0.0, 0.0,
                                        gp, gq, ga, gb, gn, gm);
                            lf_x[1] += gp; lf_y[1] += gq;
                            lf_x[2] += ga; lf_y[2] += gb;
                            lf_x[3] += gn; lf_y[3] += gm;
                        }

                        // β-correction: derivative from moving the integration limits
                        // (crossing parameters t0, t1 depend on A's tangent points).
                        // Only applies for edge-edge crossings (eB != 0).
                        // At t0: dI/dt0 = 0.5*(eA × X1); dt0/d(ap_i) = -(1-t0)*eB_y/C, t0/C for am_{i+1}
                        {
                            double eB0x = edg_eBx[kk], eB0y = edg_eBy[kk];
                            double C0 = eVx*eB0y - eVy*eB0x; // eA × eB0
                            if (fabs(C0) > 1e-14) {
                                double g0 = 0.5*(eVx*r1y - eVy*r1x) / C0; // dI/dt0 / C0
                                double fap_x = g0*(1.0-t0)*(-eB0y);
                                double fap_y = g0*(1.0-t0)*(eB0x);
                                double fam_x = g0*t0*(-eB0y);
                                double fam_y = g0*t0*(eB0x);
                                double gp, gq, ga, gb, gn, gm;
                                arcGeomGrad(uh_x, uh_y, vh_x, vh_y, u_len, v_len,
                                            cr_i, dt_i, ell_i, delta,
                                            0.0, 0.0, fap_x, fap_y, 0.0,
                                            gp, gq, ga, gb, gn, gm);
                                lf_x[0] += gp; lf_y[0] += gq;
                                lf_x[1] += ga; lf_y[1] += gb;
                                lf_x[2] += gn; lf_y[2] += gm;
                                arcGeomGrad(uh_nx, uh_ny, vh_nx, vh_ny, u_len_n, v_len_n,
                                            cr_n, dt_n, ell_n, delta,
                                            fam_x, fam_y, 0.0, 0.0, 0.0,
                                            gp, gq, ga, gb, gn, gm);
                                lf_x[1] += gp; lf_y[1] += gq;
                                lf_x[2] += ga; lf_y[2] += gb;
                                lf_x[3] += gn; lf_y[3] += gm;

                                // Coupling: force on B's vertices.
                                //   Edge-edge case (0 <= edg_jj < n_B): dt0/d(B's edge endpoints)
                                //     via applyTangentGradToB.
                                //   Edge-arc case (edg_jj >= n_B): partner is B's arc K_jj_p; the
                                //     crossing X_c moves with bz_p, so propagate fz_p = g0·n_c_p
                                //     through partner's arc geometry to its three backbone vertices.
                                double jj_enc0 = edg_jj[kk];
                                if (jj_enc0 >= 0.0 && jj_enc0 < (double)n_B) {
                                    double sB0 = edg_sB[kk];
                                    int jj0  = (int)jj_enc0;
                                    int jj0n = (jj0 + 1) % n_B;
                                    applyTangentGradToB(true,  jj0,  n_B, startB,
                                                        area_scale*g0*(1.0-sB0)*eB0y, -area_scale*g0*(1.0-sB0)*eB0x,
                                                        bx_pos, by_pos, delta, force);
                                    applyTangentGradToB(false, jj0n, n_B, startB,
                                                        area_scale*g0*sB0*eB0y, -area_scale*g0*sB0*eB0x,
                                                        bx_pos, by_pos, delta, force);
                                } else if (jj_enc0 >= (double)n_B) {
                                    int jj_p = (int)(jj_enc0 - (double)n_B);
                                    int jjp_p = (jj_p - 1 + n_B) % n_B;
                                    int jjn_p = (jj_p + 1) % n_B;
                                    double bux = bx_pos[jj_p] - bx_pos[jjp_p];
                                    double buy = by_pos[jj_p] - by_pos[jjp_p];
                                    double bvx = bx_pos[jjn_p] - bx_pos[jj_p];
                                    double bvy = by_pos[jjn_p] - by_pos[jj_p];
                                    double bul = sqrt(bux*bux + buy*buy);
                                    double bvl = sqrt(bvx*bvx + bvy*bvy);
                                    if (bul > 1e-14 && bvl > 1e-14) {
                                        double buh_x = bux/bul, buh_y = buy/bul;
                                        double bvh_x = bvx/bvl, bvh_y = bvy/bvl;
                                        double bcr = buh_x*bvh_y - buh_y*bvh_x;
                                        // fz_partner = g0·n_c_partner (unnormalised n_c
                                        // matches the unnormalised eB stored here)
                                        double fz_x = area_scale * g0 *  eB0y;
                                        double fz_y = area_scale * g0 * (-eB0x);
                                        double gp, gq, ga, gb, gn, gm;
                                        zGrad(buh_x, buh_y, bvh_x, bvh_y, bul, bvl,
                                              bcr, delta, fz_x, fz_y,
                                              gp, gq, ga, gb, gn, gm);
                                        atomicAdd(&force[2*(startB + jjp_p)],     -gp);
                                        atomicAdd(&force[2*(startB + jjp_p) + 1], -gq);
                                        atomicAdd(&force[2*(startB + jj_p)],      -ga);
                                        atomicAdd(&force[2*(startB + jj_p) + 1],  -gb);
                                        atomicAdd(&force[2*(startB + jjn_p)],     -gn);
                                        atomicAdd(&force[2*(startB + jjn_p) + 1], -gm);
                                    }
                                }
                            }
                        }
                        // At t1: dI/dt1 = 0.5*(X0 × eA)
                        {
                            double eB1x = edg_eBx[kk+1], eB1y = edg_eBy[kk+1];
                            double C1 = eVx*eB1y - eVy*eB1x; // eA × eB1
                            if (fabs(C1) > 1e-14) {
                                double g1 = 0.5*(r0x*eVy - r0y*eVx) / C1; // dI/dt1 / C1
                                double fap_x = g1*(1.0-t1)*(-eB1y);
                                double fap_y = g1*(1.0-t1)*(eB1x);
                                double fam_x = g1*t1*(-eB1y);
                                double fam_y = g1*t1*(eB1x);
                                double gp, gq, ga, gb, gn, gm;
                                arcGeomGrad(uh_x, uh_y, vh_x, vh_y, u_len, v_len,
                                            cr_i, dt_i, ell_i, delta,
                                            0.0, 0.0, fap_x, fap_y, 0.0,
                                            gp, gq, ga, gb, gn, gm);
                                lf_x[0] += gp; lf_y[0] += gq;
                                lf_x[1] += ga; lf_y[1] += gb;
                                lf_x[2] += gn; lf_y[2] += gm;
                                arcGeomGrad(uh_nx, uh_ny, vh_nx, vh_ny, u_len_n, v_len_n,
                                            cr_n, dt_n, ell_n, delta,
                                            fam_x, fam_y, 0.0, 0.0, 0.0,
                                            gp, gq, ga, gb, gn, gm);
                                lf_x[1] += gp; lf_y[1] += gq;
                                lf_x[2] += ga; lf_y[2] += gb;
                                lf_x[3] += gn; lf_y[3] += gm;

                                // Coupling: force on B's vertices from dt1/d(B's vertex).
                                double jj_enc1 = edg_jj[kk+1];
                                if (jj_enc1 >= 0.0 && jj_enc1 < (double)n_B) {
                                    double sB1 = edg_sB[kk+1];
                                    int jj1  = (int)jj_enc1;
                                    int jj1n = (jj1 + 1) % n_B;
                                    applyTangentGradToB(true,  jj1,  n_B, startB,
                                                        area_scale*g1*(1.0-sB1)*eB1y, -area_scale*g1*(1.0-sB1)*eB1x,
                                                        bx_pos, by_pos, delta, force);
                                    applyTangentGradToB(false, jj1n, n_B, startB,
                                                        area_scale*g1*sB1*eB1y, -area_scale*g1*sB1*eB1x,
                                                        bx_pos, by_pos, delta, force);
                                } else if (jj_enc1 >= (double)n_B) {
                                    int jj_p = (int)(jj_enc1 - (double)n_B);
                                    int jjp_p = (jj_p - 1 + n_B) % n_B;
                                    int jjn_p = (jj_p + 1) % n_B;
                                    double bux = bx_pos[jj_p] - bx_pos[jjp_p];
                                    double buy = by_pos[jj_p] - by_pos[jjp_p];
                                    double bvx = bx_pos[jjn_p] - bx_pos[jj_p];
                                    double bvy = by_pos[jjn_p] - by_pos[jj_p];
                                    double bul = sqrt(bux*bux + buy*buy);
                                    double bvl = sqrt(bvx*bvx + bvy*bvy);
                                    if (bul > 1e-14 && bvl > 1e-14) {
                                        double buh_x = bux/bul, buh_y = buy/bul;
                                        double bvh_x = bvx/bvl, bvh_y = bvy/bvl;
                                        double bcr = buh_x*bvh_y - buh_y*bvh_x;
                                        double fz_x = area_scale * g1 *  eB1y;
                                        double fz_y = area_scale * g1 * (-eB1x);
                                        double gp, gq, ga, gb, gn, gm;
                                        zGrad(buh_x, buh_y, bvh_x, bvh_y, bul, bvl,
                                              bcr, delta, fz_x, fz_y,
                                              gp, gq, ga, gb, gn, gm);
                                        atomicAdd(&force[2*(startB + jjp_p)],     -gp);
                                        atomicAdd(&force[2*(startB + jjp_p) + 1], -gq);
                                        atomicAdd(&force[2*(startB + jj_p)],      -ga);
                                        atomicAdd(&force[2*(startB + jj_p) + 1],  -gb);
                                        atomicAdd(&force[2*(startB + jjn_p)],     -gn);
                                        atomicAdd(&force[2*(startB + jjn_p) + 1], -gm);
                                    }
                                }
                            }
                        }
                    }
                }

                // Merge per-B results into thread-global accumulators
                if (pairAreaAccum != nullptr) {
                    if (pairEnergy != 0.0)
                        atomicAdd(&pairAreaAccum[sA * numPolygons + sB], pairEnergy);
                } else {
                    double escale = (pairAreaScale != nullptr) ? 0.5 * area_scale : 1.0;
                    localEnergy += escale * pairEnergy;
                    for (int k = 0; k < 4; k++) {
                        glf_x[k] += area_scale * lf_x[k];
                        glf_y[k] += area_scale * lf_y[k];
                    }
                }

    } // end candidate-list walk

    if (localEnergy != 0.0)
        atomicAdd(energy, localEnergy);

    // Force = -dE/d(vertex)
    int verts[4] = {pi_i, i, ni_i, nni_i};
    for (int k = 0; k < 4; k++) {
        if (glf_x[k] != 0.0) atomicAdd(&force[2*verts[k]],   -glf_x[k]);
        if (glf_y[k] != 0.0) atomicAdd(&force[2*verts[k]+1], -glf_y[k]);
    }
}

// Vertex-disk rounded-polygon contact energy.
// For each vertex v_i, iterates over edges of neighboring polygons.
// When the disk of radius `delta` at v_i overlaps the edge's half-plane (|d| < delta,
// foot parameter t in [0,1]), adds the circular-cap area as energy and the
// corresponding forces on v_i and the two edge endpoints.
__global__ void updateForceEnergyVertexDiskKernel(
    int numVertices,
    const double* __restrict__ positions,
    const int*    __restrict__ shapeId,
    const int*    __restrict__ next,
    const int*    __restrict__ cellLocation,
    const int*    __restrict__ neighborIndices,
    const int*    __restrict__ countPerBox,
    int boxSize,
    double delta,
    double* __restrict__ force,
    double* __restrict__ energy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVertices) return;

    int  sA     = shapeId[i];
    double vx   = positions[2*i];
    double vy   = positions[2*i+1];
    int  box    = cellLocation[i];
    int  bx     = box % boxSize;
    int  by     = box / boxSize;
    int  boxCount = boxSize * boxSize;
    double delta2 = delta * delta;

    double localEnergy = 0.0;
    double localFx = 0.0, localFy = 0.0;

    int processedBoxes[16];
    int numProcessedBoxes = 0;

    for (int ddx = -1; ddx <= 1; ddx++) {
        int nx = bx + ddx;
        if (nx < 0) nx += boxSize; else if (nx >= boxSize) nx -= boxSize;
        for (int ddy = -1; ddy <= 1; ddy++) {
            int ny = by + ddy;
            if (ny < 0) ny += boxSize; else if (ny >= boxSize) ny -= boxSize;
            int nbox = ny * boxSize + nx;

            bool alreadyProcessed = false;
            for (int b = 0; b < numProcessedBoxes; b++) {
                if (processedBoxes[b] == nbox) { alreadyProcessed = true; break; }
            }
            if (alreadyProcessed) continue;
            if (numProcessedBoxes < 16) processedBoxes[numProcessedBoxes++] = nbox;

            int si = countPerBox[nbox];
            int sf = (nbox + 1 < boxCount) ? countPerBox[nbox + 1] : numVertices;

            for (int idx = si; idx < sf; idx++) {
                int j = neighborIndices[idx];
                if (shapeId[j] == sA) continue;

                int    zj  = next[j];
                double qx  = positions[2*j];
                double qy  = positions[2*j+1];
                double qzx = positions[2*zj];
                double qzy = positions[2*zj+1];

                double ex = wrap(qzx - qx);
                double ey = wrap(qzy - qy);
                double L2 = ex*ex + ey*ey;
                if (L2 < 1e-28) continue;
                double L = sqrt(L2);

                double dxe = wrap(vx - qx);
                double dye = wrap(vy - qy);

                double t = (dxe*ex + dye*ey) / L2;
                if (t < 0.0 || t > 1.0) continue;

                // Signed distance: positive = outside polygon (outward-normal side of CCW edge)
                double d = (dxe*ey - dye*ex) / L;
                if (d >= delta || d <= -delta) continue;

                double ratio     = fmax(-1.0, fmin(1.0, d / delta));
                double sqrt_term = sqrt(fmax(0.0, delta2 - d*d));
                double cap_area  = delta2 * acos(ratio) - d * sqrt_term;
                localEnergy += cap_area;

                // F = 2*sqrt_term * d(d)/d(pos);  dE/dd = -2*sqrt_term
                double F_cap   = 2.0 * sqrt_term;
                double inv_L   = 1.0 / L;
                double inv_L2  = inv_L * inv_L;

                // Force on v_i
                localFx += F_cap * ( ey * inv_L);
                localFy += F_cap * (-ex * inv_L);

                // Force on edge-start vertex j
                atomicAdd(&force[2*j],   F_cap * ((dye - ey) * inv_L + d * ex * inv_L2));
                atomicAdd(&force[2*j+1], F_cap * ((ex - dxe) * inv_L + d * ey * inv_L2));

                // Force on edge-end vertex zj
                atomicAdd(&force[2*zj],   F_cap * (-dye * inv_L - d * ex * inv_L2));
                atomicAdd(&force[2*zj+1], F_cap * ( dxe * inv_L - d * ey * inv_L2));
            }
        }
    }

    if (localFx != 0.0 || localFy != 0.0) {
        atomicAdd(&force[2*i],   localFx);
        atomicAdd(&force[2*i+1], localFy);
    }
    if (localEnergy != 0.0)
        atomicAdd(energy, localEnergy);
}

// Ball-iteration variant: candidates come from a per-vertex Verlet list instead
// of the 3x3 cell stencil. Per-pair physics is identical to the cell version.
__global__ void updateForceEnergyVertexDiskBallKernel(
    int numVertices,
    const double* __restrict__ positions,
    const int*    __restrict__ shapeId,
    const int*    __restrict__ next,
    const int*    __restrict__ ballNeighbors,
    const int*    __restrict__ numBallNeighbors,
    int ballMaxNeighbors,
    double delta,
    double* __restrict__ force,
    double* __restrict__ energy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVertices) return;

    int  sA     = shapeId[i];
    double vx   = positions[2*i];
    double vy   = positions[2*i+1];
    double delta2 = delta * delta;

    double localEnergy = 0.0;
    double localFx = 0.0, localFy = 0.0;

    int nN = numBallNeighbors[i];
    if (nN > ballMaxNeighbors) nN = ballMaxNeighbors;
    int base = i * ballMaxNeighbors;

    for (int k = 0; k < nN; k++) {
        int j = ballNeighbors[base + k];
        if (shapeId[j] == sA) continue;   // (already excluded at build time; redundant guard)

        int    zj  = next[j];
        double qx  = positions[2*j];
        double qy  = positions[2*j+1];
        double qzx = positions[2*zj];
        double qzy = positions[2*zj+1];

        double ex = wrap(qzx - qx);
        double ey = wrap(qzy - qy);
        double L2 = ex*ex + ey*ey;
        if (L2 < 1e-28) continue;
        double L = sqrt(L2);

        double dxe = wrap(vx - qx);
        double dye = wrap(vy - qy);

        double t = (dxe*ex + dye*ey) / L2;
        if (t < 0.0 || t > 1.0) continue;

        double d = (dxe*ey - dye*ex) / L;
        if (d >= delta || d <= -delta) continue;

        double ratio     = fmax(-1.0, fmin(1.0, d / delta));
        double sqrt_term = sqrt(fmax(0.0, delta2 - d*d));
        double cap_area  = delta2 * acos(ratio) - d * sqrt_term;
        localEnergy += cap_area;

        double F_cap   = 2.0 * sqrt_term;
        double inv_L   = 1.0 / L;
        double inv_L2  = inv_L * inv_L;

        localFx += F_cap * ( ey * inv_L);
        localFy += F_cap * (-ex * inv_L);

        atomicAdd(&force[2*j],   F_cap * ((dye - ey) * inv_L + d * ex * inv_L2));
        atomicAdd(&force[2*j+1], F_cap * ((ex - dxe) * inv_L + d * ey * inv_L2));

        atomicAdd(&force[2*zj],   F_cap * (-dye * inv_L - d * ex * inv_L2));
        atomicAdd(&force[2*zj+1], F_cap * ( dxe * inv_L - d * ey * inv_L2));
    }

    if (localFx != 0.0 || localFy != 0.0) {
        atomicAdd(&force[2*i],   localFx);
        atomicAdd(&force[2*i+1], localFy);
    }
    if (localEnergy != 0.0)
        atomicAdd(energy, localEnergy);
}

__global__ void translateVertexKernel(int numVertices, double* positions, double* delta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVertices * 2) return;
    double p = positions[idx] + delta[idx];
    positions[idx] = p - floor(p);
}

__global__ void updatePositionsKernel(int numVertices, double* positions, const double* force, double dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVertices * 2) return;
    double p = positions[idx] + force[idx] * dt;
    positions[idx] = p - floor(p);
}

__global__ void resetAreasKernel(const int numVertices, const int* shapeId, double* positions, const double* areas, const double* targetAreas, const double* comX, const double* comY) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVertices * 2) return;
    // which polygon, sir?
    double x = positions[idx];
    int s = shapeId[idx / 2];
    double currentArea = areas[s];
    double targetArea = targetAreas[s];
    // Guard: degenerate polygon (area <= 0 or NaN) → skip rescaling to prevent NaN propagation
    if (!(currentArea > 0.0)) return;
    double rescale = sqrt(targetArea / currentArea);
    double u = (idx % 2) ? wrap(x - comY[s]) : wrap(x - comX[s]);
    u *= rescale;
    double p = (idx % 2) ? u + comY[s] : u + comX[s];
    positions[idx] = p - floor(p);
}

__global__ void negateArrayKernel(double* arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] = -arr[idx];
}
