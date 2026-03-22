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
#include "kernels.cuh"


#include <iostream>
#include <vector>
#include "model.hpp"
#include <cufft.h>
#include <cuda_runtime.h>
#include <complex>
#include <math.h>
#include <algorithm>
#include <curand_kernel.h>
#include "enumTypes.h"

using namespace std;



// initializers

extern "C" void initializeRandomStates(curandState *globalState, unsigned long long int seed, int gridSize) {
    dim3 gridDim((gridSize + myBlockDim.x - 1) / myBlockDim.x, (gridSize + myBlockDim.y - 1) / myBlockDim.y);
    initStatesKernel<<<gridDim, myBlockDim>>>(globalState, seed, gridSize);
    cudaDeviceSynchronize();
}

// helpers

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

extern "C" void sortKeysCUDA(uint64_t* d_keys, int numItems, int beginBit, int endBit, uint32_t* d_perm_out) {
    uint32_t* d_indices_in = nullptr;
    cudaMalloc(&d_indices_in, numItems * sizeof(uint32_t));

    int threads = 256;
    int blocks = (numItems + threads - 1) / threads;
    initIndicesKernel<<<blocks, threads>>>(d_indices_in, numItems);

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_keys, d_indices_in, d_perm_out, numItems, beginBit, endBit);

    // 3. Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // 4. Perform actual sort
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_keys, d_indices_in, d_perm_out, numItems, beginBit, endBit);

    // 5. Clean up temporary buffers
    cudaFree(d_temp_storage);
    cudaFree(d_indices_in);

    // (d_perm_out now holds the permutation and can be used for reordering)
}

extern "C" void computeNextPrevCUDA(int* next, int* prev, int* startIndices, int* shapeId, int size) {
    int numBlocks = (size + blockSize - 1) / blockSize;
    computeNextPrevKernel<<<numBlocks, blockSize>>>(next, prev, startIndices, shapeId, size);
}

struct Square {
    __host__ __device__ double operator()(double x) const { return x * x; }
};

// updaters

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

extern "C" int updateNeighborsCUDA(int* shapeId, int* startIndices, double* positions, int* cellLocation, int* neighborIndices,int size,int* neighbors,int* numNeighbors,int maxNeighbors,int boxSize,int* countPerBox, int* maxActualNeighbors, float2* tu, bool* inside) {
    int numBlocks = (size + blockSize - 1) / blockSize;
    updateNeighborsKernel<<<numBlocks, blockSize>>>(shapeId, startIndices, positions, cellLocation, neighborIndices, size, neighbors, numNeighbors, maxNeighbors, boxSize, countPerBox, tu, inside);
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

extern "C" int updateValidAndCountsCUDA(int numVertices, int* contacts, int* numContacts, int maxNeighbors, bool* insideFlag, int* shapeIds, int numShapes, int* valid, int* shapeCounts, uint64_t* outputIdx) {
    int numThreads = numVertices * maxNeighbors;
    int numBlocks = (numThreads + blockSize - 1) / blockSize;
    cudaMemset(shapeCounts, 0, numShapes * sizeof(int));
    updateValidAndCountsKernel<<<numBlocks, blockSize>>>(numVertices, contacts, numContacts, maxNeighbors, insideFlag, shapeIds, numShapes, valid, shapeCounts);
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

extern "C" void updateCompactedIntersectionsCUDA(int numVertices, int maxNeighbors, int* contacts, bool* insideFlag, int* shapeIds, int* startIndices, int* valid, uint64_t* outputIdx, uint64_t* intersections, int numIntersections, float2* tu) {
    int numThreads = numVertices * maxNeighbors;
    int numBlocks = (numThreads + blockSize - 1) / blockSize;
    updateCompactedIntersectionsKernel<<<numBlocks, blockSize>>>(numVertices, maxNeighbors, contacts, insideFlag, shapeIds, startIndices, valid, outputIdx, intersections, tu);
    cudaDeviceSynchronize();
}

extern "C" void updateOverlapAreaCUDA(int* shapeId, int* startIndices, int pointDensity, int* intersectionsCounter, int* neighborIndices, int size, int boxSize, int* countPerBox, double* positions, double& overlapArea) {
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

extern "C" void updateOutersectionsCUDA(const uint64_t* intersections, const float2* tu, float2* ut, int* startIndices, int numIntersections, uint64_t* outersections) {
    if (numIntersections <= 0) return;
    
    int blockSize = 256;
    int gridSize = (numIntersections + blockSize - 1) / blockSize;
    updateOutersectionsKernel<<<gridSize, blockSize>>>(intersections, tu, ut, startIndices, numIntersections, outersections);
    cudaDeviceSynchronize();
}

extern "C" void updateForceEnergyExteriorCUDA(int numVertices, int numIntersections, const uint64_t* intersections, const uint64_t* outersections, const float2* tu, const float2* ut, const double* positions, const int* next, const int* prev, const int* shapeId, const int* startIndices, double* force, double* energy) {
    if (numIntersections <= 0) {
        return;
    }
    int threads = 256;
    int blocks = (numIntersections + threads - 1) / threads;
    size_t smem = threads * sizeof(double);
    updateForceEnergyExteriorKernel<<<blocks, threads, smem>>>(numIntersections, intersections, outersections, tu, ut, positions, next, prev, shapeId, startIndices, force, energy);
    cudaDeviceSynchronize();
}

extern "C" void updateShapeRangesCUDA(int numPolygons, int numVertices, int numIntersections, const uint64_t* intersections, int* shapeStart, int* shapeEnd) {
    // Initialise with sentinel values
    int initBlocks = (numPolygons + blockSize - 1) / blockSize;
    initShapeRangesKernel<<<initBlocks, blockSize>>>(shapeStart, shapeEnd, numPolygons, numIntersections);
    cudaDeviceSynchronize();
    // Find actual ranges using atomicMin/Max
    int blocks = (numIntersections + blockSize - 1) / blockSize;
    updateShapeRangesKernel<<<blocks, blockSize>>>(intersections, numIntersections, shapeStart, shapeEnd);
    cudaDeviceSynchronize();
}

extern "C" void updateForceEnergyInteriorCUDA(int numVertices, int numIntersections, const uint64_t* intersections, const uint64_t* outersections, const float2* tu, const float2* ut, const double* positions, const int* next, const int* prev, const int* shapeId, const int* startIndices, double* force, double* energy, int numPolygons, int* shapeStart, int* shapeEnd) {
    if (numIntersections == 0) return;
    // Launch interior kernel
    int threads = 256;
    int grid = (numVertices + threads - 1) / threads;
    updateForceEnergyInteriorKernel<<<grid, threads>>>(numVertices, intersections, outersections, tu, ut, positions, next, prev, shapeId, startIndices, force, energy, shapeStart, shapeEnd);
    cudaDeviceSynchronize();
}

extern "C" void updateForceEnergyEdgeCUDA(int numVertices, const double* positions, const double* edgeLengths, const int* next, const int* prev, double* force, double* energy, double stiffness) {
    int threads = 256;
    int grid = (numVertices + threads - 1) / threads;
    updateForceEnergyEdgeKernel<<<grid, threads>>>(numVertices, positions, edgeLengths, next, prev, force, energy, stiffness);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

extern "C" void updatePositionsCUDA(int numVertices, double* positions, const double* force, double dt) {
    int initBlocks = (numVertices * 2 + blockSize - 1) / blockSize;
    updatePositionsKernel<<<initBlocks, blockSize>>>(numVertices, positions, force, dt);
    cudaDeviceSynchronize();
}

extern "C" void updateConstraintForcesCUDA(int numVertices, int numPolygons, int* shapeId, double* positions, int* next, int* prev, int* startDOF, int* endDOF, double* constraints, double* norm2, double** norm2TMP, size_t* norm2TMPStorageBytes, double* force, double* proj, double* constraintForce) {
    int initBlocks = (numVertices * 2 + blockSize - 1) / blockSize;
    updateConstraintsKernel<<<initBlocks, blockSize>>>(numVertices, positions, next, prev, constraints);
    cudaDeviceSynchronize();

    cub::TransformInputIterator<double, Square, double*> squaredConstraints(constraints, Square{});

    size_t requiredBytes = 0;
    cub::DeviceSegmentedReduce::Sum(nullptr, requiredBytes, squaredConstraints, norm2,
                                    numPolygons, startDOF, endDOF);
    if (requiredBytes > *norm2TMPStorageBytes) {
        if (*norm2TMP) cudaFree(*norm2TMP);
        cudaMalloc(norm2TMP, requiredBytes);
        *norm2TMPStorageBytes = requiredBytes;
    }
    cub::DeviceSegmentedReduce::Sum(*norm2TMP, *norm2TMPStorageBytes, squaredConstraints, norm2,
                                    numPolygons, startDOF, endDOF);
    cudaDeviceSynchronize();

    cudaMemset(proj, 0, numPolygons * sizeof(double));
    updateProjectionKernel<<<initBlocks, blockSize>>>(numVertices, numPolygons, shapeId,
                                                      constraints, norm2, force, proj);
    cudaDeviceSynchronize();

    updateConstraintForcesKernel<<<initBlocks, blockSize>>>(numVertices, numPolygons, shapeId,
                                                            constraints, force, proj, constraintForce);
    cudaDeviceSynchronize();
}