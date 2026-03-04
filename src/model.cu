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
#include <cub/iterator/transform_input_iterator.cuh>   // defines TransformInputIterator
#include <cub/device/device_run_length_encode.cuh>

static const dim3 myBlockDim(16, 16);
static const int blockSize = 256;
static const double pi = 3.141592653589793238462643383279;

// Helper kernel to fill sequence [0, 1, 2, ..., n-1]
__global__ void fillSequenceKernel(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = idx;
    }
}

// Helper kernel to perform lower_bound for each needle in haystack
// For each needle, finds the position of the first element >= needle
__global__ void lowerBoundKernel(
    const int* __restrict__ haystack,
    int haystack_size,
    const int* __restrict__ needles,
    int needles_size,
    int* __restrict__ results
) {
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


__global__ void initStatesKernel(curandState *globalState, unsigned long long seed, int gridSize) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Flatten 2D index to 1D index
    int idx = row * gridSize + col;
    if (row < gridSize && col < gridSize) {
       curand_init(seed, idx, 0, &globalState[idx]);
    }
}

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

extern "C" void initializeRandomStates(curandState *globalState, unsigned long long int seed, int gridSize) {
    dim3 gridDim((gridSize + myBlockDim.x - 1) / myBlockDim.x, (gridSize + myBlockDim.y - 1) / myBlockDim.y);
    initStatesKernel<<<gridDim, myBlockDim>>>(globalState, seed, gridSize);
    cudaDeviceSynchronize();
}

extern "C" void updateAreasCUDA(double* areas, double* positions, int* startIndices, int numPolygons) {
    int numBlocks = (numPolygons + blockSize - 1) / blockSize;
    updateAreasKernel<<<numBlocks, blockSize>>>(areas, positions, startIndices, numPolygons);
}

extern "C" void updatePerimetersCUDA(double* perimeters, double* positions, int* startIndices, int numPolygons) {
    int numBlocks = (numPolygons + blockSize - 1) / blockSize;
    updatePerimetersKernel<<<numBlocks, blockSize>>>(perimeters, positions, startIndices, numPolygons);
}

extern "C" void updateNeighborCellsCUDA(double* positions, int* startIndices, int* shapeId, int numPolygons, int size, int boxSize, int* cellLocation, int* countPerBox, int* boxId, int& boxesUsed, int* neighborIndices) {
    int numBlocks = (size + blockSize - 1) / blockSize;
    updateNeighborCellsKernel<<<numBlocks, blockSize>>>(positions, startIndices, shapeId, numPolygons, size, boxSize, cellLocation);
    
    // Allocate temporary device memory for sorting
    int* d_cellLocation_sorted;
    int* d_neighborIndices_sorted;
    cudaMalloc(&d_cellLocation_sorted, size * sizeof(int));
    cudaMalloc(&d_neighborIndices_sorted, size * sizeof(int));
    
    // Copy cellLocation to sorted version
    cudaMemcpy(d_cellLocation_sorted, cellLocation, size * sizeof(int), cudaMemcpyDeviceToDevice);
    
    // Fill sequence [0, 1, 2, ..., size-1] for sorting
    int seqBlocks = (size + blockSize - 1) / blockSize;
    fillSequenceKernel<<<seqBlocks, blockSize>>>(d_neighborIndices_sorted, size);
    cudaDeviceSynchronize();
    
    // CUB sort_by_key: sort d_neighborIndices_sorted by d_cellLocation_sorted
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    // First pass: determine temp storage requirements
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_cellLocation_sorted, d_cellLocation_sorted, // key input/output
        d_neighborIndices_sorted, d_neighborIndices_sorted, // value input/output
        size
    );
    
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    // Second pass: perform the sort
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_cellLocation_sorted, d_cellLocation_sorted,
        d_neighborIndices_sorted, d_neighborIndices_sorted,
        size
    );
    cudaDeviceSynchronize();
    
    // Perform lower bound search to find where each box index starts
    int* d_search_results;
    cudaMalloc(&d_search_results, boxSize * boxSize * sizeof(int));
    
    // Create needle array [0, 1, ..., boxSize*boxSize-1]
    int* d_needles;
    cudaMalloc(&d_needles, boxSize * boxSize * sizeof(int));
    
    int needleBlocks = (boxSize * boxSize + blockSize - 1) / blockSize;
    fillSequenceKernel<<<needleBlocks, blockSize>>>(d_needles, boxSize * boxSize);
    cudaDeviceSynchronize();
    
    // Perform lower bound search using custom kernel on sorted cellLocation
    int searchBlocks = (boxSize * boxSize + blockSize - 1) / blockSize;
    lowerBoundKernel<<<searchBlocks, blockSize>>>(
        d_cellLocation_sorted, size,
        d_needles, boxSize * boxSize,
        d_search_results
    );
    cudaDeviceSynchronize();
    
    // Copy results back
    cudaMemcpy(countPerBox, d_search_results, boxSize * boxSize * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(neighborIndices, d_neighborIndices_sorted, size * sizeof(int), cudaMemcpyDeviceToDevice);
    
    // Cleanup
    cudaFree(d_cellLocation_sorted);
    cudaFree(d_neighborIndices_sorted);
    cudaFree(d_temp_storage);
    cudaFree(d_search_results);
    cudaFree(d_needles);
}

extern "C" void updateShapeIdCUDA(int* shapeId, int* startIndices, int size, int numPolygons) {
    int numBlocks = (size + blockSize - 1) / blockSize;
    updateShapeIdKernel<<<numBlocks, blockSize>>>(shapeId, startIndices, numPolygons);
}

__global__ void updateNeighborsKernel(const int* __restrict__ shapeId,
    const int* __restrict__ startIndices,
    const double* __restrict__ positions,
    const int* __restrict__ cellLocation,
    const int* __restrict__ neighborIndices,
    const int size,
    int* __restrict__ neighbors,
    int* __restrict__ numNeighbors,
    int maxNeighbors,
    int boxSize,
    int* __restrict__ countPerBox,
    double a
    ) {
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

__global__ void updateContactsKernel(
    const int* __restrict__ shapeId,
    const int* __restrict__ startIndices,
    const double* __restrict__ positions,
    const int size,
    const int* __restrict__ neighbors,
    const int* __restrict__ numNeighbors,
    const int maxNeighbors,
    bool* __restrict__ inside,
    int* __restrict__ contacts,
    int* __restrict__ numContacts,
    float2* __restrict__ tu
    ) {
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

extern "C" int updateNeighborsCUDA(
    int* shapeId,
    int* startIndices, 
    double* positions, 
    int* cellLocation, 
    int* neighborIndices,
    int size,
    int* neighbors,
    int* numNeighbors,
    int maxNeighbors,
    int boxSize,
    int* countPerBox,
    double a,
    int* maxActualNeighbors
) {
    int numBlocks = (size + blockSize - 1) / blockSize;
    updateNeighborsKernel<<<numBlocks, blockSize>>>(shapeId, startIndices, positions, cellLocation, neighborIndices, size, neighbors, numNeighbors, maxNeighbors, boxSize, countPerBox, a);
    cudaDeviceSynchronize();

    // initialize device accumulator for reduction
    int initVal = INT_MIN;
    cudaMemcpy(maxActualNeighbors, &initVal, sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    maxReduceKernel<<<blocks, threads, threads * sizeof(int)>>>(numNeighbors, size, maxActualNeighbors);
    cudaDeviceSynchronize();
    int newMaxActualNeighbors;
    cudaMemcpy(&newMaxActualNeighbors, maxActualNeighbors, sizeof(int), cudaMemcpyDeviceToHost);
    return newMaxActualNeighbors;
}

extern "C" void updateContactsCUDA(
    const int* shapeId, 
    const int* startIndices, 
    const double* positions, 
    const int size, 
    const int* neighbors, 
    const int* numNeighbors, 
    const int maxNeighbors,
    bool* inside, 
    float2* tu,
    int* contacts, 
    int* numContacts
) {
    int numBlocks = (size + blockSize - 1) / blockSize;
    updateContactsKernel<<<numBlocks, blockSize>>>(shapeId, startIndices, positions, size, neighbors, numNeighbors, maxNeighbors, inside, contacts, numContacts, tu);
    cudaDeviceSynchronize();
}

__global__ void markValidAndCountsKernel(
    const int numVertices,
    const int* __restrict__ contacts,
    const int* __restrict__ numContacts,
    const int maxNeighbors,
    const bool* __restrict__ insideFlag,
    const int* __restrict__ shapeIds,
    const int numShapes,
    int* __restrict__ valid,
    int* __restrict__ shapeCounts
) {
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

extern "C" int markValidAndCountsCUDA(
    int numVertices,
    int* contacts,
    int* numContacts,
    int maxNeighbors,
    bool* insideFlag,
    int* shapeIds,
    int numShapes,
    int* valid,
    int* shapeCounts,
    uint64_t* outputIdx
) {
    int numThreads = numVertices * maxNeighbors;
    int numBlocks = (numThreads + blockSize - 1) / blockSize;
    cudaMemset(shapeCounts, 0, numShapes * sizeof(int));
    markValidAndCountsKernel<<<numBlocks, blockSize>>>(numVertices, contacts, numContacts, maxNeighbors, insideFlag, shapeIds, numShapes, valid, shapeCounts);
    cudaDeviceSynchronize();
    
    // CUB exclusive scan
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    // First pass: determine temp storage requirements
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes,
        valid, outputIdx, numThreads
    );
    
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    // Second pass: perform the scan
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes,
        valid, outputIdx, numThreads
    );
    cudaDeviceSynchronize();
    
    int lastValid;
    cudaMemcpy(&lastValid, valid + numThreads - 1, sizeof(int), cudaMemcpyDeviceToHost);
    uint64_t lastOutputIdx;
    cudaMemcpy(&lastOutputIdx, outputIdx + numThreads - 1, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_temp_storage);
    
    return lastOutputIdx + lastValid;
}

__global__ void writeCompactedKernel(
    const int numVertices,
    const int maxNeighbors,
    const int* __restrict__ contacts,
    const bool* __restrict__ insideFlag,
    const int* __restrict__ shapeIds,
    const int* __restrict__ valid,
    const uint64_t* __restrict__ outputIdx,
    uint64_t* __restrict__ intersections,
    float2* __restrict__ tu
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalContacts = numVertices * maxNeighbors;
    if (idx >= totalContacts) return;
    if (!valid[idx]) return;

    uint64_t outPos = outputIdx[idx];

    int n1 = idx / maxNeighbors;
    int n2 = contacts[idx];
    int s1 = shapeIds[n1];
    int s2 = shapeIds[n2];
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

extern "C" void writeCompactedCUDA(
    int numVertices,
    int maxNeighbors,
    int* contacts,
    bool* insideFlag,
    int* shapeIds,
    int* valid,
    uint64_t* outputIdx,
    uint64_t* intersections,
    int numIntersections,
    float2* tu
) {
    int numThreads = numVertices * maxNeighbors;
    int numBlocks = (numThreads + blockSize - 1) / blockSize;
    writeCompactedKernel<<<numBlocks, blockSize>>>(numVertices, maxNeighbors, contacts, insideFlag, shapeIds, valid, outputIdx, intersections, tu);
    cudaDeviceSynchronize();
}

__global__ void initIndicesKernel(uint32_t* indices, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) indices[idx] = idx;
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

extern "C" void applyPermutationCUDA_float2(const float2* d_input, const uint32_t* d_perm, float2* d_output, int numItems) {
    int threads = 256;
    int blocks = (numItems + threads - 1) / threads;
    gatherKernel_float2<<<blocks, threads>>>(d_input, d_perm, d_output, numItems);
    cudaDeviceSynchronize();
}

extern "C" void applyPermutationCUDA_int64(const uint64_t* d_input, const uint32_t* d_perm, uint64_t* d_output, int numItems) {
    int threads = 256;
    int blocks = (numItems + threads - 1) / threads;
    gatherKernel_int64<<<blocks, threads>>>(d_input, d_perm, d_output, numItems);
    cudaDeviceSynchronize();
}

extern "C" void sortKeysCUDA(
    uint64_t* d_keys,          // input/output keys, device pointer
    int numItems,
    int beginBit,              // e.g. 0
    int endBit,                 // e.g. 48
    uint32_t* d_perm_out        // output permutation (size numItems), device pointer
) {
    // 1. Create device array for initial indices [0,1,2,...]
    uint32_t* d_indices_in = nullptr;
    cudaMalloc(&d_indices_in, numItems * sizeof(uint32_t));

    // Launch kernel to fill indices
    int threads = 256;
    int blocks = (numItems + threads - 1) / threads;
    initIndicesKernel<<<blocks, threads>>>(d_indices_in, numItems);

    // 2. Determine temporary storage size for CUB sort
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_keys, d_keys,                 // keys: input = output (in‑place)
        d_indices_in, d_perm_out,       // values: input indices, output permutation
        numItems,
        beginBit, endBit
    );

    // 3. Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // 4. Perform actual sort
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_keys, d_keys,
        d_indices_in, d_perm_out,
        numItems,
        beginBit, endBit
    );

    // 5. Clean up temporary buffers
    cudaFree(d_temp_storage);
    cudaFree(d_indices_in);

    // (d_perm_out now holds the permutation and can be used for reordering)
}

__device__ inline double wrapPeriodic(double x) {
    // Wrap x into [-0.5, 0.5)
    x += 1.5;
    while (x > 1.0) x -= 1.0;
    return x - 0.5;
}

__global__ void updateOverlapAreaKernel(
    const int* __restrict__ shapeId,
    const int* __restrict__ startIndices,
    int pointDensity,
    int* __restrict__ intersectionsCounter,
    const int* __restrict__ neighborIndices,
    int size,
    int boxSize,
    const int* __restrict__ countPerBox,
    const double* __restrict__ positions
) {
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

extern "C" void updateOverlapAreaCUDA(
    int* shapeId,
    int* startIndices,
    int pointDensity,
    int* intersectionsCounter,
    int* neighborIndices,
    int size,
    int boxSize,
    int* countPerBox,
    double* positions,
    double& overlapArea
) {
    int total = pointDensity * pointDensity;
    int numBlocks = (total + blockSize - 1) / blockSize;
    updateOverlapAreaKernel<<<numBlocks, blockSize>>>(
        shapeId, startIndices, pointDensity, intersectionsCounter,
        neighborIndices, size, boxSize, countPerBox, positions
    );
    cudaDeviceSynchronize();

    // CUB reduce: sum all values in intersectionsCounter
    long long* d_result;
    cudaMalloc(&d_result, sizeof(long long));
    
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    // First pass: determine temp storage requirements
    cub::DeviceReduce::Sum(
        d_temp_storage, temp_storage_bytes,
        intersectionsCounter, d_result, total
    );
    
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    // Second pass: perform the reduction
    cub::DeviceReduce::Sum(
        d_temp_storage, temp_storage_bytes,
        intersectionsCounter, d_result, total
    );
    cudaDeviceSynchronize();
    
    // Copy result back
    long long sum;
    cudaMemcpy(&sum, d_result, sizeof(long long), cudaMemcpyDeviceToHost);
    overlapArea = (double)sum;
    
    // Cleanup
    cudaFree(d_temp_storage);
    cudaFree(d_result);
}

struct ExtractHigh32 {
    __host__ __device__ unsigned int operator()(uint64_t x) const {
        return static_cast<unsigned int>(x >> 32);
    }
};

__global__ void fillGroupRangesKernel(
    const int* runStarts,
    const int* runLengths,
    int numGroups,
    int* groupStart,
    int* groupLength)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= numGroups) return;
    int start = runStarts[g];
    int len = runLengths[g];
    for (int i = 0; i < len; ++i) {
        int idx = start + i;
        groupStart[idx] = start;
        groupLength[idx] = len;
    }
}

extern "C" void markGroupBoundariesCUDA(
    const uint64_t* intersections,
    int numIntersections,
    int* groupStart,
    int* groupLength,
    int& numGroups)
{
    if (numIntersections <= 0) {
        numGroups = 0;
        return;
    }

    // Transform iterator to extract high 32 bits (shape pair)
    cub::TransformInputIterator<unsigned int, ExtractHigh32, const uint64_t*>
        itr_high(intersections, ExtractHigh32());

    // Temporary device arrays (size = numIntersections, enough for worst case)
    unsigned int* d_unique = nullptr;
    int* d_runLengths = nullptr;
    int* d_runStarts = nullptr;
    int* d_numGroups = nullptr;
    cudaMalloc(&d_unique, numIntersections * sizeof(unsigned int));
    cudaMalloc(&d_runLengths, numIntersections * sizeof(int));
    cudaMalloc(&d_runStarts, numIntersections * sizeof(int));
    cudaMalloc(&d_numGroups, sizeof(int));

    // 1st pass: get temp storage size
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceRunLengthEncode::Encode(
        d_temp_storage, temp_storage_bytes,
        itr_high, d_unique, d_runLengths, d_numGroups,
        numIntersections);

    // Allocate temp storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // 2nd pass: perform run‑length encoding
    cub::DeviceRunLengthEncode::Encode(
        d_temp_storage, temp_storage_bytes,
        itr_high, d_unique, d_runLengths, d_numGroups,
        numIntersections);

    // Copy number of groups to host
    cudaMemcpy(&numGroups, d_numGroups, sizeof(int), cudaMemcpyDeviceToHost);

    // Compute exclusive sum of run lengths to obtain run start indices
    // (now we only need the first numGroups elements)
    size_t scan_temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        nullptr, scan_temp_bytes,
        d_runLengths, d_runStarts, numGroups);

    // Reallocate temp storage if needed (or reuse d_temp_storage)
    cudaFree(d_temp_storage);
    cudaMalloc(&d_temp_storage, scan_temp_bytes);

    cub::DeviceScan::ExclusiveSum(
        d_temp_storage, scan_temp_bytes,
        d_runLengths, d_runStarts, numGroups);

    // Launch kernel to fill per‑intersection groupStart and groupLength
    int blockSize = 256;
    int gridSize = (numGroups + blockSize - 1) / blockSize;
    fillGroupRangesKernel<<<gridSize, blockSize>>>(
        d_runStarts, d_runLengths, numGroups,
        groupStart, groupLength);
    cudaDeviceSynchronize();

    // Cleanup temporary memory
    cudaFree(d_temp_storage);
    cudaFree(d_unique);
    cudaFree(d_runLengths);
    cudaFree(d_runStarts);
    cudaFree(d_numGroups);
}

__global__ void updateOutersectionsKernel(
    const uint64_t* __restrict__ intersections,
    const float2* __restrict__ tu,
    float2* __restrict__ ut,
    const int* __restrict__ startIndices,
    int numIntersections,
    uint64_t* __restrict__ outersections)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numIntersections) return;

    uint64_t inter = intersections[idx];
    int sj = (inter >> 48) & 0xFFFF;
    int si = (inter >> 32) & 0xFFFF;
    int i = (inter >> 16) & 0xFFFF;
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

extern "C" void updateOutersectionsCUDA(
    const uint64_t* intersections,
    const float2* tu,
    float2* ut,
    int* startIndices,
    int numIntersections,
    uint64_t* outersections)
{
    if (numIntersections <= 0) return;
    
    int blockSize = 256;
    int gridSize = (numIntersections + blockSize - 1) / blockSize;
    updateOutersectionsKernel<<<gridSize, blockSize>>>(intersections, tu, ut, startIndices, numIntersections, outersections);
    // After kernel launch, check for errors
    cudaDeviceSynchronize();
    // update the outersections
}

