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

static const dim3 myBlockDim(16, 16);
static const int blockSize = 256;
static const double pi = 3.141592653589793238462643383279;

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

__global__ void gatherKernel_float2(const float2* input, const uint32_t* perm, float2* output, int n) {
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

__device__ inline double wrapPeriodic(double x) {
    // Wrap x into [-0.5, 0.5)
    x += 1.5;
    while (x > 1.0) x -= 1.0;
    return x - 0.5;
}

__device__ double wrap(double x) {
    double y = x + 1.5;
    y = y - floor(y);      // fractional part in [0,1)
    return y - 0.5;
}

__device__ double2 wrap2(double2 v) {
    v.x = wrap(v.x);
    v.y = wrap(v.y);
    return v;
}

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

__device__ double h(double2 pt1, double2 pt2, double2 startPoint) {
    double2 r1 = {pt1.x - startPoint.x, pt1.y - startPoint.y};
    double2 r2 = {pt2.x - startPoint.x, pt2.y - startPoint.y};
    r1 = wrap2(r1);
    r2 = wrap2(r2);
    double2 d = {r2.x - r1.x, r2.y - r1.y};
    return (r1.x + r2.x) * d.y;
}

__device__ void g12(double2 pt1, double2 pt2, double2 startPoint, double2& g1, double2& g2) {
    double2 r1 = {pt1.x - startPoint.x, pt1.y - startPoint.y};
    double2 r2 = {pt2.x - startPoint.x, pt2.y - startPoint.y};
    r1 = wrap2(r1);
    r2 = wrap2(r2);
    double2 d = {pt2.x - pt1.x, pt2.y - pt1.y};
    d = wrap2(d);
    g1.x = d.y;
    g1.y = -(r1.x + r2.x);
    g2.x = d.y;
    g2.y = (r1.x + r2.x);
}

__device__ void getDf(double2 vi, double2 vzi, double2 vj, double2 vzj, double* df) {
    // Edge vectors with periodic wrap
    double2 dj = wrap2({vzj.x - vj.x, vzj.y - vj.y});
    double2 di = wrap2({vzi.x - vi.x, vzi.y - vi.y});
    double2 dij = wrap2({vj.x - vi.x, vj.y - vi.y});

    double w = dj.x * di.y - dj.y * di.x;
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

__global__ void updateAreasKernel(double* areas, double* positions, int* startIndices, int numPolygons) {
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

__global__ void updatePerimetersKernel(double* perimeters, double* positions, int* startIndices, int numPolygons) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPolygons) {
        int start = startIndices[idx];
        int end = startIndices[idx + 1];
        double dx, dy;
        for (int i = start; i < end - 1; i++) {
            dx = (positions[2 * i] - positions[2 * i + 2] + 0.5);
            while (dx < 0.0) {
                dx += 1;
            }
            while (dx > 1.0) {
                dx -= 1.0;
            }
            dx -= 0.5;
            dy = (positions[2 * i + 1] - positions[2 * i + 3] + 0.5);
            while (dy < 0.0) dy += 1.0;
            while (dy > 1.0) dy -= 1.0;
            perimeters[idx] += sqrt(dx * dx + dy * dy);
        }
    }
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
        while (x > 1.0) {
            x -= 1.0;
        }
        while (y > 1.0) {
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
        while (x > 1.0) {
            x -= 1.0;
        }
        while (y > 1.0) {
            y -= 1.0;
        }
        x = floor(x * boxSize);
        y = floor(y * boxSize);
        cellLocation[idx] = int(y) * boxSize + int(x);
    }
}

__global__ void updateNeighborsKernel(const int* __restrict__ shapeId, const int* __restrict__ startIndices, const double* __restrict__ positions, const int* __restrict__ cellLocation, const int* __restrict__ neighborIndices, const int size, int* __restrict__ neighbors, int* __restrict__ numNeighbors, int maxNeighbors, int boxSize, int* __restrict__ countPerBox, double a) {
    const double eps = 1e-12;
    int id1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (id1 >= size) return;

    // cached per-thread values
    int boxCount = boxSize * boxSize;

    int shape = shapeId[id1];
    int st = startIndices[shape + 1] - 1;
    int id2 = (id1 == st) ? startIndices[shape] : id1 + 1;

    const double px = positions[2 * id1];
    const double py = positions[2 * id1 + 1];

    double rx = positions[2 * id2] - px + 1.5;
    double ry = positions[2 * id2 + 1] - py + 1.5;
    // wrap once
    while (rx > 1.0) rx -= 1.0;
    while (ry > 1.0) ry -= 1.0;
    rx -= 0.5;
    ry -= 0.5;

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
                if (nid == id1 || nid == id2 || nid2 == id1) continue;

                // positions of neighbor edge
                double qx = positions[2 * nid];
                double qy = positions[2 * nid + 1];

                double sx = positions[2 * nid2] - qx + 1.5;
                double sy = positions[2 * nid2 + 1] - qy + 1.5;
                double gx = qx - px + 1.5;
                double gy = qy - py + 1.5;

                // wrap once for these differences
                while (sx > 1.0) sx -= 1.0;
                while (sy > 1.0) sy -= 1.0;
                while (gx > 1.0) gx -= 1.0;
                while (gy > 1.0) gy -= 1.0;

                sx -= 0.5;
                sy -= 0.5;
                gx -= 0.5;
                gy -= 0.5;

                double rSize = sqrt(rx * rx + ry * ry);
                double sSize = sqrt(sx * sx + sy * sy);

                double denom = rx * sy - ry * sx;
                if (fabs(denom) < eps) continue; // parallel or degenerate

                double tt = (gx * sy - gy * sx) / denom;
                double uu = (gx * ry - gy * rx) / denom;

                if (tt >= -a / rSize && tt <= 1.0 + a / rSize && uu >= -a / sSize && uu <= 1.0 + a / sSize) {
                    if (neighborCount < maxNeighbors) {
                        neighbors[id1 * maxNeighbors + neighborCount] = nid;
                    }
                    neighborCount++;
                }
            }
        }
    }
    numNeighbors[id1] = neighborCount;
}

__global__ void updateContactsKernel(const int* __restrict__ shapeId, const int* __restrict__ startIndices, const double* __restrict__ positions, const int size, const int* __restrict__ neighbors, const int* __restrict__ numNeighbors, const int maxNeighbors, bool* __restrict__ inside, int* __restrict__ contacts, int* __restrict__ numContacts, float2* __restrict__ tu) {
    const double eps = 1e-12;
    int id1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (id1 >= size) return;
    int shape = shapeId[id1];
    int st = startIndices[shape + 1] - 1;
    const double px = positions[2 * id1];
    const double py = positions[2 * id1 + 1];
    int id2 = (id1 == st) ? startIndices[shape] : id1 + 1;
    double rx = positions[2 * id2] - px + 1.5;
    double ry = positions[2 * id2 + 1] - py + 1.5;
    // wrap once
    while (rx > 1.0) rx -= 1.0;
    while (ry > 1.0) ry -= 1.0;
    rx -= 0.5;
    ry -= 0.5;
    int neighborStart = id1 * maxNeighbors;
    int neighborEnd = id1 * maxNeighbors + numNeighbors[id1];
    int contactCount = 0;
    for (int index = neighborStart; index < neighborEnd; index++) {
        int nid = neighbors[index];
        int shape2 = shapeId[nid];
        int st2 = startIndices[shape2 + 1] - 1;
        int nid2 = (nid == st2) ? startIndices[shape2] : nid + 1;
        // skip trivial/adjacent edges
        if (nid == id1 || nid == id2 || nid2 == id1) continue;
       // positions of neighbor edge
        double qx = positions[2 * nid];
        double qy = positions[2 * nid + 1];

        double sx = positions[2 * nid2] - qx + 1.5;
        double sy = positions[2 * nid2 + 1] - qy + 1.5;
        double gx = qx - px + 1.5;
        double gy = qy - py + 1.5;

        // wrap once for these differences
        while (sx > 1.0) sx -= 1.0;
        while (sy > 1.0) sy -= 1.0;
        while (gx > 1.0) gx -= 1.0;
        while (gy > 1.0) gy -= 1.0;

        sx -= 0.5;
        sy -= 0.5;
        gx -= 0.5;
        gy -= 0.5;

        double rSize = sqrt(rx * rx + ry * ry);
        double sSize = sqrt(sx * sx + sy * sy);

        double denom = rx * sy - ry * sx;
        if (fabs(denom) < eps) continue; // parallel or degenerate

        double tt = (gx * sy - gy * sx) / denom;
        double uu = (gx * ry - gy * rx) / denom;

        if (tt > 0.0 && tt < 1.0 && uu > 0.0 && uu < 1.0) {
            contacts[id1 * maxNeighbors + contactCount] = nid;
            if (denom > 0) {
                inside[id1 * maxNeighbors + contactCount] = true;
                tu[id1 * maxNeighbors + contactCount].x = tt;
                tu[id1 * maxNeighbors + contactCount].y = uu;
            }
            else {
                inside[id1 * maxNeighbors + contactCount] = false;
                tu[id1 * maxNeighbors + contactCount].x = uu;
                tu[id1 * maxNeighbors + contactCount].y = tt;
            }
            contactCount++;
        }
    }
    numContacts[id1] = contactCount;
}

__global__ void updateValidAndCountsKernel(const int numVertices, const int* __restrict__ contacts, const int* __restrict__ numContacts, const int maxNeighbors, const bool* __restrict__ insideFlag, const int* __restrict__ shapeIds, const int numShapes, int* __restrict__ valid, int* __restrict__ shapeCounts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalContacts = numVertices * maxNeighbors;
    if (idx >= totalContacts) return;

    int n1 = idx / maxNeighbors;
    int i = idx % maxNeighbors;

    // Check if this contact slot is actually used
    if (i < numContacts[n1]) {
        valid[idx] = 1;

        int n2 = contacts[idx];
        int s1 = shapeIds[n1];
        int s2 = shapeIds[n2];
        bool inside = insideFlag[idx];

        int targetShape = inside ? s2 : s1;
        atomicAdd(&shapeCounts[targetShape], 1);
    } else {
        valid[idx] = 0;
    }
}

__global__ void updateCompactedIntersectionsKernel(const int numVertices, const int maxNeighbors, const int* __restrict__ contacts, const bool* __restrict__ insideFlag, const int* __restrict__ shapeIds, const int* __restrict__ startIndices, const int* __restrict__ valid, const uint64_t* __restrict__ outputIdx, uint64_t* __restrict__ intersections, float2* __restrict__ tu) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalContacts = numVertices * maxNeighbors;
    if (idx >= totalContacts) return;
    if (!valid[idx]) return;

    uint64_t outPos = outputIdx[idx];

    int n1 = idx / maxNeighbors;
    int n2 = contacts[idx];
    int s1 = shapeIds[n1];
    int s2 = shapeIds[n2];
    n1 -= startIndices[s1];
    n2 -= startIndices[s2];
    bool inside = insideFlag[idx];
    float tVal = tu[idx].x;
    float uVal = tu[idx].y;

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

    // Store TU as float2 in required order
    // TODO: Check if this is right
    tu[outPos] = make_float2((float)tVal, (float)uVal);
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

    // Track polygons we've already checked for this point
    int trackedPolys[256];
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

                // Track this polygon
                if (numTrackedPolys < 256) {
                    trackedPolys[numTrackedPolys++] = polyId;
                }
            }
        }
    }

    intersectionsCounter[idx] = numIntersections * (numIntersections - 1) / 2;
//    intersectionsCounter[idx] = numIntersections;
}

__global__ void updateOutersectionsKernel(const uint64_t* __restrict__ intersections, const float2* __restrict__ tu, float2* __restrict__ ut, const int* __restrict__ startIndices, int numIntersections, uint64_t* __restrict__ outersections) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numIntersections) return;

    uint64_t inter = intersections[idx];
    int sj = (inter >> 48) & 0xFFFF;
    int si = (inter >> 32) & 0xFFFF;
    int i = ((inter >> 16) & 0xFFFF) + startIndices[si];
//    int j = inter & 0xFFFF;
    int ni = startIndices[si + 1] - startIndices[si];
    uint64_t sij = ((uint64_t)si << 48) | ((uint64_t)sj << 32);
        
    float tVal = tu[idx].x;
    // next we need to find idj
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
        float uVal = tu[k].y;

        int d = (l - i + ni) % ni;

        // Skip self-pairing if condition fails
        if (l == i && tVal >= uVal) {
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
        k++;
    }
    int player = (bestIdx != -1) ? bestIdx : fallbackIdx;
    outersections[idx] = intersections[player];
    ut[idx] = tu[player];
}

__global__ void updateForceEnergyExteriorKernel(int numIntersections, const uint64_t* __restrict__ intersections, const uint64_t* __restrict__ outersections, const float2* __restrict__ tu, const float2* __restrict__ ut, const double* __restrict__ positions, const int* __restrict__ next, const int* __restrict__ prev, const int* __restrict__ shapeId, const int* __restrict__ startIndices, double* __restrict__ force, double* __restrict__ energyGlobal) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numIntersections) return;

    uint64_t inter = intersections[idx];
    uint64_t outer = outersections[idx];

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

    // startPoint = first vertex of the earlier shape
    int startID = (start1 < start2) ? start1 : start2;
    double2 startPoint = *reinterpret_cast<const double2*>(&positions[2*startID]);

    // ---- Edge vectors and parameters ----
    double2 r1 = wrap2({pzi.x - pi.x, pzi.y - pi.y});
    double2 r2 = wrap2({pzk.x - pk.x, pzk.y - pk.y});

    float2 tuf = tu[idx];
    float2 utf = ut[idx];
    double t1 = tuf.y;          // second component
    double t2 = utf.y;

    // fij = (pi + t1*r1) mod 1
    double2 fij;
    fij.x = pi.x + t1 * r1.x;
    fij.y = pi.y + t1 * r1.y;
    fij.x = fmod(fij.x + 1.0, 1.0);
    fij.y = fmod(fij.y + 1.0, 1.0);

    double2 fkl;
    fkl.x = pk.x + t2 * r2.x;
    fkl.y = pk.y + t2 * r2.y;
    fkl.x = fmod(fkl.x + 1.0, 1.0);
    fkl.y = fmod(fkl.y + 1.0, 1.0);

    double localEnergy = 0.0;

    if (i == l) {
        // Branch 1: i == l
        localEnergy += h(fij, fkl, startPoint);

        double2 g1, g2;
        g12(fij, fkl, startPoint, g1, g2);

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
        localEnergy += h(fij, pzi, startPoint);
        localEnergy += h(pl, fkl, startPoint);

        // First g12 (fij, pzi)
        double2 g1a, g2a;
        g12(fij, pzi, startPoint, g1a, g2a);

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

        // Second g12 (pl, fkl)
        double2 g1b, g2b;
        g12(pl, fkl, startPoint, g1b, g2b);

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

