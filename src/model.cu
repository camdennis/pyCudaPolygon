#include <cuda_runtime.h>
#include <iostream>
#include <cufft.h>
#include <complex>
#include <curand_kernel.h>
#include <float.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/binary_search.h>

static const dim3 myBlockDim(16, 16);
static const int blockSize = 256;


__global__ void initStatesKernel(curandState *globalState, unsigned long long seed, int gridSize) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Flatten 2D index to 1D index
    int idx = row * gridSize + col;
    if (row < gridSize && col < gridSize) {
       curand_init(seed, idx, 0, &globalState[idx]);
    }
}

__global__ void updateAreasKernel(double* areas, double* positions, int* startIndices, int numShapes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numShapes) {
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

__global__ void updateShapeIdKernel(int* shapeId, int* startIndices, int numShapes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numShapes) {
        for (int i = startIndices[idx]; i < startIndices[idx + 1]; i++) {
            shapeId[i] = idx;
        }
    }
}

__global__ void updateNeighborCellsKernel(double* positions, int* startIndices, int* shapeId, int numShapes, int size, int boxSize, int* cellLocation) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int shape = shapeId[idx];
        int v2 = idx + 1;
        if (idx == size - 1 || shapeId[idx] != shapeId[idx + 1]) {
            v2 = startIndices[shapeId[idx]];
        }
        double x = positions[v2 * 2] - positions[idx * 2] + 0.5;
        double y = positions[v2 * 2 + 1] - positions[idx * 2 + 1] + 0.5;
        while (x < 1.0) {
            x += 1.0;
        }
        while (y < 1.0) {
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
        while (x < 1.0) {
            x += 1.0;
        }
        while (y < 1.0) {
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

extern "C" void updateAreasCUDA(double* areas, double* positions, int* startIndices, int numShapes) {
    int numBlocks = (numShapes + blockSize - 1) / blockSize;
    updateAreasKernel<<<numBlocks, blockSize>>>(areas, positions, startIndices, numShapes);
}

extern "C" void updateNeighborCellsCUDA(double* positions, int* startIndices, int* shapeId, int numShapes, int size, int boxSize, int* cellLocation, int* countPerBox, int* boxId, int& boxesUsed, int* neighborIndices) {
    int numBlocks = (size + blockSize - 1) / blockSize;
    updateNeighborCellsKernel<<<numBlocks, blockSize>>>(positions, startIndices, shapeId, numShapes, size, boxSize, cellLocation);
    thrust::device_vector<int> d_cellLocation(cellLocation, cellLocation + size);
    thrust::device_vector<int> d_neighborIndices(neighborIndices, neighborIndices + size);
    thrust::device_vector<int> d_countPerBox(countPerBox, countPerBox + boxSize * boxSize);

    thrust::sequence(d_neighborIndices.begin(), d_neighborIndices.end());
    thrust::sort_by_key(d_cellLocation.begin(), d_cellLocation.end(), d_neighborIndices.begin());

    thrust::counting_iterator<int> box_begin(0);
    thrust::counting_iterator<int> box_end(boxSize * boxSize);

    thrust::lower_bound(
        d_cellLocation.begin(), d_cellLocation.end(),
        box_begin, box_end,
        d_countPerBox.begin()
    );

    thrust::copy(
        d_countPerBox.begin(), d_countPerBox.end(),
        thrust::device_pointer_cast(countPerBox)
    );

    thrust::copy(
        d_cellLocation.begin(), d_cellLocation.end(),
        thrust::device_pointer_cast(cellLocation)
    );

    thrust::copy(
        d_neighborIndices.begin(), d_neighborIndices.end(),
        thrust::device_pointer_cast(neighborIndices)
    );
}

extern "C" void updateShapeIdCUDA(int* shapeId, int* startIndices, int size, int numShapes) {
    int numBlocks = (size + blockSize - 1) / blockSize;
    updateShapeIdKernel<<<numBlocks, blockSize>>>(shapeId, startIndices, numShapes);
}

__global__ void updateNeighborsKernel(const int* shapeId, 
    const int* startIndices, 
    const double* positions, 
    const int* cellLocation, 
    const int* neighborIndices, 
    const int size,
    int* neighbors,
    int* numNeighbors,
    int maxNeighbors,
    int boxSize,
    int* countPerBox,
    double a
    ) {
    int id1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (id1 < size) {
        double t, u, denom;
        // get the edges
        int shape = shapeId[id1];
        int st = startIndices[shape + 1] - 1;
        int beginning = startIndices[shape];
        int id2 = id1 + 1;
        if (id1 == st) {
            id2 = startIndices[shape];
        }
        double px = positions[2 * id1];
        double py = positions[2 * id1 + 1];
        double rx = positions[2 * id2] - px + 1.5;
        double ry = positions[2 * id2 + 1] - py + 1.5;
        while (rx > 1.0) {
            rx -= 1.0;
        }
        while (ry > 1.0) {
            ry -= 1.0;
        }
        rx -= 0.5;
        ry -= 0.5;
        // What box are you in?
        int box = cellLocation[id1];
        int x, y, newBox;
        // What are the adjacent boxes?
        int neighborCount = 0;
        for (int dx = -1; dx < 2; dx++) {
            for (int dy = -1; dy < 2; dy++) {
                y = box / boxSize;
                x = box % boxSize;
                x += boxSize + dx;
                x %= boxSize;
                y += boxSize + dy;
                y %= boxSize;
                newBox = y * boxSize + x;
                // okay, so you got the right box:
                // Now loop through the edges. But first, what edges?
                int si = countPerBox[newBox];
                int sf = countPerBox[newBox + 1];
                int siHigh = sf - 1;
                // Make si be the index value that is less than the middle
                int mid;
                while (si != siHigh) {
                    mid = si + (sf - si + 1) / 2;
                    if (neighborIndices[mid] < id1) {
                        si = mid + 1;
                    }
                    else {
                        siHigh = mid;
                    }
                }
                for (int nid = si; nid < sf; nid++) {
                    // Make sure this neighbor isn't an adjacent edge:
                    if ((id1 == beginning && nid == beginning + boxSize - 1) || nid == id1 + 1) {
                        continue;
                    }
                    // Get nid2:
                    int shape2 = shapeId[nid];
                    int st2 = startIndices[shape2 + 1] - 1;
                    int nid2 = nid + 1;
                    if (nid == st2) {
                        nid2 = startIndices[shape2];
                    }
                    // Now we check if they are in contact:
                    double qx = positions[2 * nid];
                    double qy = positions[2 * nid + 1];
                    double sx = qx - positions[2 * nid2] + 1.5;
                    double sy = qy - positions[2 * nid2 + 1] + 1.5;
                    double pqx = qx - px + 1.5;
                    double pqy = qy - py + 1.5;
                    while (sx > 1.0) {
                        sx -= 1.0;
                    }
                    while (sy > 1.0) {
                        sy -= 1.0;
                    }
                    while (pqx > 1.0) {
                        pqx -= 1.0;
                    }
                    while (pqy > 1.0) {
                        pqy -= 1.0;
                    }
                    sx -= 0.5;
                    sy -= 0.5;
                    pqx -= 0.5;
                    pqy -= 0.5;
                    denom = rx * sy - ry * sx;
                    t = (pqx * sy - pqy * sx) / denom;
                    u = (pqx * ry - pqy * rx) / denom;
                    if (t >= -a && t <= 1 + a && u >= -a && u <= 1 + a) {
                        // It intersects so put it in
                        if (neighborCount <= maxNeighbors) {
                            neighbors[id1 * maxNeighbors + neighborCount] = nid;
                        }
                        neighborCount++;
                    }
                }
                numNeighbors[id1] = neighborCount;
            }
        }
    }
}

extern "C" void updateNeighborsCUDA(
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
    double a
    ) {
    int numBlocks = (size + blockSize - 1) / blockSize;
    updateNeighborsKernel<<<numBlocks, blockSize>>>(shapeId, startIndices, positions, cellLocation, neighborIndices, size, neighbors, numNeighbors, maxNeighbors, boxSize, countPerBox, a);
}

