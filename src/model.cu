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
#include "cuda_check.h"

// initializers

extern "C" void initializeRandomStates(curandState *globalState, unsigned long long int seed, int gridSize) {
    dim3 gridDim((gridSize + myBlockDim.x - 1) / myBlockDim.x, (gridSize + myBlockDim.y - 1) / myBlockDim.y);
    initStatesKernel<<<gridDim, myBlockDim>>>(globalState, seed, gridSize);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}

// helpers

extern "C" void applyPermutationCUDA_float2(const float2* d_input, const uint32_t* d_perm, float2* d_output, int numItems) {
    int threads = 256;
    int blocks = (numItems + threads - 1) / threads;
    gatherKernel_float2<<<blocks, threads>>>(d_input, d_perm, d_output, numItems);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C" void applyPermutationCUDA_int64(const uint64_t* d_input, const uint32_t* d_perm, uint64_t* d_output, int numItems) {
    int threads = 256;
    int blocks = (numItems + threads - 1) / threads;
    gatherKernel_int64<<<blocks, threads>>>(d_input, d_perm, d_output, numItems);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C" void sortKeysCUDA(uint64_t* d_keys, int numItems, int beginBit, int endBit, uint32_t* d_perm_out) {
    uint32_t* d_indices_in = nullptr;
    CUDA_CHECK(cudaMalloc(&d_indices_in, numItems * sizeof(uint32_t)));

    int threads = 256;
    int blocks = (numItems + threads - 1) / threads;
    initIndicesKernel<<<blocks, threads>>>(d_indices_in, numItems);
    CUDA_CHECK_KERNEL();

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_keys, d_indices_in, d_perm_out, numItems, beginBit, endBit));

    // 3. Allocate temporary storage
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // 4. Perform actual sort
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_keys, d_indices_in, d_perm_out, numItems, beginBit, endBit));

    // 5. Clean up temporary buffers
    CUDA_CHECK(cudaFree(d_temp_storage));
    CUDA_CHECK(cudaFree(d_indices_in));

    // (d_perm_out now holds the permutation and can be used for reordering)
}

extern "C" void computeNextPrevCUDA(int* next, int* prev, int* startIndices, int* shapeId, int size) {
    int numBlocks = (size + blockSize - 1) / blockSize;
    computeNextPrevKernel<<<numBlocks, blockSize>>>(next, prev, startIndices, shapeId, size);
    CUDA_CHECK_KERNEL();
}

struct Square {
    __host__ __device__ double operator()(double x) const { return x * x; }
};

struct AbsVal {
    __host__ __device__ double operator()(double x) const { return fabs(x); }
};

double maxAbsValue(double* d_data, int n) {
    cub::TransformInputIterator<double, AbsVal, double*> absIter(d_data, AbsVal{});

    double* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(double)));

    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    CUDA_CHECK(cub::DeviceReduce::Max(d_temp, temp_bytes, absIter, d_result, n));
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
    CUDA_CHECK(cub::DeviceReduce::Max(d_temp, temp_bytes, absIter, d_result, n));
    CUDA_CHECK(cudaDeviceSynchronize());

    double result;
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_result));
    return result;
}

// updaters

extern "C" void updateAreasCUDA(double* areas, double* positions, int* startIndices, int numPolygons) {
    int numBlocks = (numPolygons + blockSize - 1) / blockSize;
    updateAreasKernel<<<numBlocks, blockSize>>>(areas, positions, startIndices, numPolygons);
    CUDA_CHECK_KERNEL();
}

extern "C" void updatePerimetersCUDA(double* perimeters, double* positions, int* startIndices, int numPolygons) {
    int numBlocks = (numPolygons + blockSize - 1) / blockSize;
    updatePerimetersKernel<<<numBlocks, blockSize>>>(perimeters, positions, startIndices, numPolygons);
    CUDA_CHECK_KERNEL();
}

extern "C" void updateNeighborCellsCUDA(double* positions, int* startIndices, int* shapeId, int numPolygons, int size, int boxSize, int* cellLocation, int* countPerBox, int* boxId, int& boxesUsed, int* neighborIndices) {
    int numBlocks = (size + blockSize - 1) / blockSize;
    updateNeighborCellsKernel<<<numBlocks, blockSize>>>(positions, startIndices, shapeId, numPolygons, size, boxSize, cellLocation);
    CUDA_CHECK_KERNEL();

    // Allocate temporary device memory for sorting
    int* d_cellLocation_sorted;
    int* d_neighborIndices_sorted;
    CUDA_CHECK(cudaMalloc(&d_cellLocation_sorted, size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_neighborIndices_sorted, size * sizeof(int)));

    // Copy cellLocation to sorted version
    CUDA_CHECK(cudaMemcpy(d_cellLocation_sorted, cellLocation, size * sizeof(int), cudaMemcpyDeviceToDevice));

    // Fill sequence [0, 1, 2, ..., size-1] for sorting
    int seqBlocks = (size + blockSize - 1) / blockSize;
    fillSequenceKernel<<<seqBlocks, blockSize>>>(d_neighborIndices_sorted, size);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    // CUB sort_by_key: sort d_neighborIndices_sorted by d_cellLocation_sorted
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // First pass: determine temp storage requirements
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_cellLocation_sorted, d_cellLocation_sorted, // key input/output
        d_neighborIndices_sorted, d_neighborIndices_sorted, // value input/output
        size
    ));

    // Allocate temporary storage
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // Second pass: perform the sort
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_cellLocation_sorted, d_cellLocation_sorted,
        d_neighborIndices_sorted, d_neighborIndices_sorted,
        size
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Perform lower bound search to find where each box index starts
    int* d_search_results;
    CUDA_CHECK(cudaMalloc(&d_search_results, boxSize * boxSize * sizeof(int)));

    // Create needle array [0, 1, ..., boxSize*boxSize-1]
    int* d_needles;
    CUDA_CHECK(cudaMalloc(&d_needles, boxSize * boxSize * sizeof(int)));

    int needleBlocks = (boxSize * boxSize + blockSize - 1) / blockSize;
    fillSequenceKernel<<<needleBlocks, blockSize>>>(d_needles, boxSize * boxSize);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Perform lower bound search using custom kernel on sorted cellLocation
    int searchBlocks = (boxSize * boxSize + blockSize - 1) / blockSize;
    lowerBoundKernel<<<searchBlocks, blockSize>>>(
        d_cellLocation_sorted, size,
        d_needles, boxSize * boxSize,
        d_search_results
    );
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back
    CUDA_CHECK(cudaMemcpy(countPerBox, d_search_results, boxSize * boxSize * sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(neighborIndices, d_neighborIndices_sorted, size * sizeof(int), cudaMemcpyDeviceToDevice));

    // Cleanup
    CUDA_CHECK(cudaFree(d_cellLocation_sorted));
    CUDA_CHECK(cudaFree(d_neighborIndices_sorted));
    CUDA_CHECK(cudaFree(d_temp_storage));
    CUDA_CHECK(cudaFree(d_search_results));
    CUDA_CHECK(cudaFree(d_needles));
}

extern "C" void updateShapeIdCUDA(int* shapeId, int* startIndices, int size, int numPolygons) {
    int numBlocks = (size + blockSize - 1) / blockSize;
    updateShapeIdKernel<<<numBlocks, blockSize>>>(shapeId, startIndices, numPolygons);
    CUDA_CHECK_KERNEL();
}

extern "C" int updateNeighborsCUDA(int* shapeId, int* startIndices, double* positions, int* cellLocation, int* neighborIndices,int size,int* neighbors,int* numNeighbors,int maxNeighbors,int boxSize,int* countPerBox, int* maxActualNeighbors, float2* tu, bool* inside) {
    int numBlocks = (size + blockSize - 1) / blockSize;
    updateNeighborsKernel<<<numBlocks, blockSize>>>(shapeId, startIndices, positions, cellLocation, neighborIndices, size, neighbors, numNeighbors, maxNeighbors, boxSize, countPerBox, tu, inside);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    // initialize device accumulator for reduction
    int initVal = INT_MIN;
    CUDA_CHECK(cudaMemcpy(maxActualNeighbors, &initVal, sizeof(int), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    maxReduceKernel<<<blocks, threads, threads * sizeof(int)>>>(numNeighbors, size, maxActualNeighbors);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
    int newMaxActualNeighbors;
    CUDA_CHECK(cudaMemcpy(&newMaxActualNeighbors, maxActualNeighbors, sizeof(int), cudaMemcpyDeviceToHost));
    return newMaxActualNeighbors;
}

extern "C" int updateValidAndCountsCUDA(int numVertices, int* contacts, int* numContacts, int maxNeighbors, bool* insideFlag, int* shapeIds, int numShapes, int* valid, int* shapeCounts, uint64_t* outputIdx) {
    int numThreads = numVertices * maxNeighbors;
    int numBlocks = (numThreads + blockSize - 1) / blockSize;
    CUDA_CHECK(cudaMemset(shapeCounts, 0, numShapes * sizeof(int)));
    updateValidAndCountsKernel<<<numBlocks, blockSize>>>(numVertices, contacts, numContacts, maxNeighbors, insideFlag, shapeIds, numShapes, valid, shapeCounts);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    // CUB exclusive scan
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // First pass: determine temp storage requirements
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes,
        valid, outputIdx, numThreads
    ));

    // Allocate temporary storage
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // Second pass: perform the scan
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes,
        valid, outputIdx, numThreads
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    int lastValid;
    CUDA_CHECK(cudaMemcpy(&lastValid, valid + numThreads - 1, sizeof(int), cudaMemcpyDeviceToHost));
    uint64_t lastOutputIdx;
    CUDA_CHECK(cudaMemcpy(&lastOutputIdx, outputIdx + numThreads - 1, sizeof(uint64_t), cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(d_temp_storage));

    return lastOutputIdx + lastValid;
}

extern "C" void updateCompactedIntersectionsCUDA(int numVertices, int maxNeighbors, int* contacts, bool* insideFlag, int* shapeIds, int* startIndices, int* valid, uint64_t* outputIdx, uint64_t* intersections, int numIntersections, float2* tu, float2* tuTMP) {
    int numThreads = numVertices * maxNeighbors;
    int numBlocks = (numThreads + blockSize - 1) / blockSize;
    // Write compacted tu into tuTMP (separate buffer) to avoid the in-place scatter
    // race where outPos[b] == idx[a] causes thread b's store to corrupt thread a's load.
    updateCompactedIntersectionsKernel<<<numBlocks, blockSize>>>(numVertices, maxNeighbors, contacts, insideFlag, shapeIds, startIndices, valid, outputIdx, intersections, tu, tuTMP);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
    // Copy compacted result back into tu
    if (numIntersections > 0) {
        CUDA_CHECK(cudaMemcpy(tu, tuTMP, numIntersections * sizeof(float2), cudaMemcpyDeviceToDevice));
    }
}

extern "C" void updateOverlapAreaCUDA(int* shapeId, int* startIndices, int pointDensity, int* intersectionsCounter, int* neighborIndices, int size, int boxSize, int* countPerBox, double* positions, double& overlapArea) {
    int total = pointDensity * pointDensity;
    int numBlocks = (total + blockSize - 1) / blockSize;
    updateOverlapAreaKernel<<<numBlocks, blockSize>>>(
        shapeId, startIndices, pointDensity, intersectionsCounter,
        neighborIndices, size, boxSize, countPerBox, positions
    );
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    // CUB reduce: sum all values in intersectionsCounter
    long long* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(long long)));

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // First pass: determine temp storage requirements
    CUDA_CHECK(cub::DeviceReduce::Sum(
        d_temp_storage, temp_storage_bytes,
        intersectionsCounter, d_result, total
    ));

    // Allocate temporary storage
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // Second pass: perform the reduction
    CUDA_CHECK(cub::DeviceReduce::Sum(
        d_temp_storage, temp_storage_bytes,
        intersectionsCounter, d_result, total
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    long long sum;
    CUDA_CHECK(cudaMemcpy(&sum, d_result, sizeof(long long), cudaMemcpyDeviceToHost));
    overlapArea = (double)sum;

    // Cleanup
    CUDA_CHECK(cudaFree(d_temp_storage));
    CUDA_CHECK(cudaFree(d_result));
}

extern "C" void updateOutersectionsCUDA(const uint64_t* intersections, const float2* tu, float2* ut, int* startIndices, int numIntersections, uint64_t* outersections) {
    if (numIntersections <= 0) return;

    int blockSize = 256;
    int gridSize = (numIntersections + blockSize - 1) / blockSize;
    updateOutersectionsKernel<<<gridSize, blockSize>>>(intersections, tu, ut, startIndices, numIntersections, outersections);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C" void updateForceEnergyExteriorCUDA(int numVertices, int numIntersections, const uint64_t* intersections, const uint64_t* outersections, const float2* tu, const float2* ut, const double* positions, const int* next, const int* prev, const int* shapeId, const int* startIndices, double* force, double* energy) {
    if (numIntersections <= 0) {
        return;
    }
    int threads = 256;
    int blocks = (numIntersections + threads - 1) / threads;
    size_t smem = threads * sizeof(double);
    updateForceEnergyExteriorKernel<<<blocks, threads, smem>>>(numIntersections, intersections, outersections, tu, ut, positions, next, prev, shapeId, startIndices, force, energy);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C" void updateShapeRangesCUDA(int numPolygons, int numVertices, int numIntersections, const uint64_t* intersections, int* shapeStart, int* shapeEnd) {
    // Initialise with sentinel values
    int initBlocks = (numPolygons + blockSize - 1) / blockSize;
    initShapeRangesKernel<<<initBlocks, blockSize>>>(shapeStart, shapeEnd, numPolygons, numIntersections);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
    // Find actual ranges using atomicMin/Max
    int blocks = (numIntersections + blockSize - 1) / blockSize;
    updateShapeRangesKernel<<<blocks, blockSize>>>(intersections, numIntersections, shapeStart, shapeEnd);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C" void updateForceEnergyInteriorCUDA(int numVertices, int numIntersections, const uint64_t* intersections, const uint64_t* outersections, const float2* tu, const float2* ut, const double* positions, const int* next, const int* prev, const int* shapeId, const int* startIndices, double* force, double* energy, int numPolygons, int* shapeStart, int* shapeEnd) {
    if (numIntersections == 0) return;
    // Launch interior kernel
    int threads = 256;
    int grid = (numVertices + threads - 1) / threads;
    updateForceEnergyInteriorKernel<<<grid, threads>>>(numVertices, intersections, outersections, tu, ut, positions, next, prev, shapeId, startIndices, force, energy, shapeStart, shapeEnd);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C" void updateForceEnergyEdgeCUDA(int numVertices, const double* positions, const double* edgeLengths, const int* next, const int* prev, double* force, double* energy, double stiffness) {
    int threads = 256;
    int grid = (numVertices + threads - 1) / threads;
    updateForceEnergyEdgeKernel<<<grid, threads>>>(numVertices, positions, edgeLengths, next, prev, force, energy, stiffness);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C" void updatePositionsCUDA(int numVertices, double* positions, const double* force, double dt) {
    int initBlocks = (numVertices * 2 + blockSize - 1) / blockSize;
    updatePositionsKernel<<<initBlocks, blockSize>>>(numVertices, positions, force, dt);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C" void updateConstraintForcesCUDA(int numVertices, int numPolygons, int* shapeId, double* positions, int* next, int* prev, int* startDOF, int* endDOF, double* constraints, double* norm2, double** norm2TMP, size_t* norm2TMPStorageBytes, double* force, double* proj, double* constraintForce) {
    int initBlocks = (numVertices * 2 + blockSize - 1) / blockSize;
    updateConstraintsKernel<<<initBlocks, blockSize>>>(numVertices, positions, next, prev, constraints);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    cub::TransformInputIterator<double, Square, double*> squaredConstraints(constraints, Square{});
    CUDA_CHECK(cudaDeviceSynchronize());

    size_t requiredBytes = 0;
    CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(nullptr, requiredBytes, squaredConstraints, norm2, numPolygons, startDOF, endDOF));
    if (requiredBytes > *norm2TMPStorageBytes) {
        if (*norm2TMP) CUDA_CHECK_NOABORT(cudaFree(*norm2TMP));
        CUDA_CHECK(cudaMalloc(norm2TMP, requiredBytes));
        *norm2TMPStorageBytes = requiredBytes;
    }
    CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(*norm2TMP, *norm2TMPStorageBytes, squaredConstraints, norm2, numPolygons, startDOF, endDOF));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemset(proj, 0, numPolygons * sizeof(double)));
    CUDA_CHECK(cudaDeviceSynchronize());

    updateProjectionKernel<<<initBlocks, blockSize>>>(numVertices, numPolygons, shapeId, constraints, norm2, force, proj);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    updateConstraintForcesKernel<<<initBlocks, blockSize>>>(numVertices, numPolygons, shapeId, constraints, force, proj, constraintForce);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C" double getMaxUnbalancedForceCUDA(int numVertices, double* constraintForce) {
    return maxAbsValue(constraintForce, numVertices * 2);
}
