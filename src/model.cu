#include <cuda_runtime.h>
#include <cusolverDn.h>
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
#include "FIRE.h"
#include "shake.h"
#include "cuda_check.h"

using namespace std;

// initializers

extern "C" void initializeRandomStates(curandState *globalState, unsigned long long int seed, int gridSize) {
    dim3 gridDim((gridSize + myBlockDim.x - 1) / myBlockDim.x, (gridSize + myBlockDim.y - 1) / myBlockDim.y);
    initStatesKernel<<<gridDim, myBlockDim>>>(globalState, seed, gridSize);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}

// helpers

extern "C" void applyPermutationCUDA_double2(const double2* d_input, const uint32_t* d_perm, double2* d_output, int numItems) {
    if (numItems == 0) return;
    int threads = 256;
    int blocks = (numItems + threads - 1) / threads;
    gatherKernel_double2<<<blocks, threads>>>(d_input, d_perm, d_output, numItems);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C" void applyPermutationCUDA_int64(const uint64_t* d_input, const uint32_t* d_perm, uint64_t* d_output, int numItems) {
    if (numItems == 0) return;
    int threads = 256;
    int blocks = (numItems + threads - 1) / threads;
    gatherKernel_int64<<<blocks, threads>>>(d_input, d_perm, d_output, numItems);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C" void sortKeysCUDA(uint64_t* d_keys, int numItems, int beginBit, int endBit, uint32_t* d_perm_out) {
    if (numItems == 0) return;
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

extern "C" void updatePolygonGeometryCUDA(int numVertices, int numPolygons, double* positions, int* startIndices, int* shapeId, int* next, int* prev, double* edgeLengths, double* areaParts, double* comParts, double* area, double* comX, double* comY, double* maxEdgeLength, double* constraints, double* constraintNormSq) {
    int numBlocks = (numVertices * 2 + blockSize - 1) / blockSize;
    // zero per-polygon squared-norm accumulators before the geometry kernel fills them
    CUDA_CHECK(cudaMemset(constraintNormSq, 0, 3 * numPolygons * sizeof(double)));
    updatePolygonGeometryKernel<<<numBlocks, blockSize>>>(numVertices, numPolygons, positions, startIndices, shapeId, next, prev, edgeLengths, areaParts, comParts, constraints, constraintNormSq);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, areaParts, area, numPolygons, startIndices, startIndices + 1));
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, areaParts, area, numPolygons, startIndices, startIndices + 1));

    size_t temp_storage_bytes2 = 0;
    CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(nullptr, temp_storage_bytes2, comParts, comX, numPolygons, startIndices, startIndices + 1));

    // resize if needed
    if (temp_storage_bytes2 > temp_storage_bytes) {
        CUDA_CHECK(cudaFree(d_temp_storage));
        CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes2));
        temp_storage_bytes = temp_storage_bytes2;
    }

    CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, comParts, comX, numPolygons, startIndices, startIndices + 1));

    CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, comParts + numVertices, comY, numPolygons, startIndices, startIndices + 1));

    // maxEdgeLength is NOT recomputed here. Under SHAKE the constraint manifold
    // pins edge lengths, so the user-set value (from setMaxEdgeLength) is the
    // physical max for the entire simulation. Recomputing live would silently
    // shrink the Verlet ball radius after polygons relax and start missing
    // overlap pairs.

    CUDA_CHECK(cudaDeviceSynchronize());

    normalizeKernel<<<numBlocks, blockSize>>>(numPolygons, comX, comY, positions, startIndices);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    normalizeConstraintsKernel<<<numBlocks, blockSize>>>(numVertices, shapeId, constraints, constraintNormSq);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_temp_storage));
}

extern "C" void projectForceCUDA(
        int numVertices, int numPolygons, int n,
        const int* shapeId, const int* startIndices, const int* next,
        const double* positions, const double* constraints,
        double* edgeGradTMP, double* uMat,
        double* singularValuesTMP, double* vMatTMP, int* solverInfoTMP,
        double* qAreaVec, cusolverDnHandle_t handle,
        double* workspace, int workspaceSize, double* hRnrmF, double* force) {
    int numBlocks = (numVertices + blockSize - 1) / blockSize;
    long long matStride = (long long)2*n * n;

    CUDA_CHECK(cudaMemset(edgeGradTMP, 0, matStride * numPolygons * sizeof(double)));
    buildEdgeGradMatrixKernel<<<numBlocks, blockSize>>>(
        numVertices, shapeId, startIndices, next, positions, edgeGradTMP, n);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUSOLVER_CHECK(cusolverDnDgesvdaStridedBatched(
        handle, CUSOLVER_EIG_MODE_VECTOR, n,
        2*n, n,
        edgeGradTMP, 2*n, matStride,
        singularValuesTMP, (long long)n,
        uMat, 2*n, matStride,
        vMatTMP, n, (long long)n*n,
        workspace, workspaceSize,
        solverInfoTMP, hRnrmF, numPolygons));
    CUDA_CHECK(cudaDeviceSynchronize());

    int gsBlock = min(1024, ((2*n + 31) / 32) * 32);
    int numWarps = (gsBlock + 31) / 32;
    size_t gsSmem = (size_t)(2*n + numWarps) * sizeof(double);
    gramSchmidtAreaKernel<<<numPolygons, gsBlock, gsSmem>>>(
        numPolygons, n, startIndices, shapeId, uMat, constraints, qAreaVec);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    size_t projSmem = (size_t)(2*n + numWarps) * sizeof(double);
    forceProjectFullKernel<<<numPolygons, gsBlock, projSmem>>>(
        numPolygons, n, startIndices, uMat, qAreaVec, force);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}


extern "C" int xpbdProjectCUDA(
        int numVertices, int numPolygons, int nIter,
        int* startIndices, int* shapeId, int* next, int* prev,
        double* positions, const double* targetEdgeLengths, const double* targetAreas,
        double* d_area, double* d_gradNormSq, double tol, double* convTMP) {
    int numBlocks = (numVertices + blockSize - 1) / blockSize;

    double* dResult = nullptr;
    void* dTemp = nullptr;
    size_t tempBytes = 0;
    if (tol > 0) {
        CUDA_CHECK(cudaMalloc(&dResult, sizeof(double)));
        CUDA_CHECK(cub::DeviceReduce::Max(dTemp, tempBytes, convTMP, dResult, numVertices));
        CUDA_CHECK(cudaMalloc(&dTemp, tempBytes));
    }

    double maxEdgeDev = 1e9;
    double maxAreaDev = 1e9;
    int iter = 0;
    for (; iter < nIter; ++iter) {
        xpbdEdgeProjectKernel<<<numBlocks, blockSize>>>(
            numVertices, startIndices, shapeId, next, positions, targetEdgeLengths, 0);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());

        xpbdEdgeProjectKernel<<<numBlocks, blockSize>>>(
            numVertices, startIndices, shapeId, next, positions, targetEdgeLengths, 1);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemset(d_area,       0, numPolygons * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_gradNormSq, 0, numPolygons * sizeof(double)));
        xpbdAreaReductionKernel<<<numBlocks, blockSize>>>(
            numVertices, shapeId, startIndices, next, prev, positions, d_area, d_gradNormSq);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());

        xpbdAreaCorrectionKernel<<<numBlocks, blockSize>>>(
            numVertices, shapeId, startIndices, next, prev, positions,
            d_area, d_gradNormSq, targetAreas);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());

        if (tol > 0) {
            xpbdEdgeDeviationKernel<<<numBlocks, blockSize>>>(
                numVertices, shapeId, next, positions, targetEdgeLengths, convTMP);
            CUDA_CHECK_KERNEL();
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cub::DeviceReduce::Max(dTemp, tempBytes, convTMP, dResult, numVertices));
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(&maxEdgeDev, dResult, sizeof(double), cudaMemcpyDeviceToHost));

            int polyBlocks = (numPolygons + blockSize - 1) / blockSize;
            xpbdAreaDeviationKernel<<<polyBlocks, blockSize>>>(
                numPolygons, d_area, targetAreas, convTMP);
            CUDA_CHECK_KERNEL();
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cub::DeviceReduce::Max(dTemp, tempBytes, convTMP, dResult, numPolygons));
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(&maxAreaDev, dResult, sizeof(double), cudaMemcpyDeviceToHost));
            if (maxEdgeDev < tol && maxAreaDev < tol) break;
        }
    }

    if (tol > 0) {
        CUDA_CHECK(cudaFree(dResult));
        CUDA_CHECK(cudaFree(dTemp));
    }
    return iter;
}

extern "C" void saveTentativePositionsCUDA(int numVertices, const double* positions, double* tentPos) {
    CUDA_CHECK(cudaMemcpy(tentPos, positions, 2 * numVertices * sizeof(double), cudaMemcpyDeviceToDevice));
}

extern "C" double getMaxEffectiveForceCUDA(
        int numVertices, const double* positions, const double* tentPos,
        const double* force, double scale, double* scratch) {
    int numBlocks = (numVertices + blockSize - 1) / blockSize;
    effectiveForceMagKernel<<<numBlocks, blockSize>>>(
        numVertices, positions, tentPos, force, scale, scratch);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    double* dResult;
    CUDA_CHECK(cudaMalloc(&dResult, sizeof(double)));
    void* dTemp = nullptr;
    size_t tempBytes = 0;
    CUDA_CHECK(cub::DeviceReduce::Max(dTemp, tempBytes, scratch, dResult, numVertices));
    CUDA_CHECK(cudaMalloc(&dTemp, tempBytes));
    CUDA_CHECK(cub::DeviceReduce::Max(dTemp, tempBytes, scratch, dResult, numVertices));
    CUDA_CHECK(cudaDeviceSynchronize());
    double result;
    CUDA_CHECK(cudaMemcpy(&result, dResult, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dTemp));
    CUDA_CHECK(cudaFree(dResult));
    return result;
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

extern "C" int updateNeighborsCUDA(int* shapeId, int* startIndices, double* positions, int* cellLocation, int* neighborIndices,int size,int* neighbors,int* numNeighbors,int maxNeighbors,int boxSize,int* countPerBox, int* maxActualNeighbors, double2* tu, bool* inside, double delta) {
    int numBlocks = (size + blockSize - 1) / blockSize;
    updateNeighborsKernel<<<numBlocks, blockSize>>>(shapeId, startIndices, positions, cellLocation, neighborIndices, size, neighbors, numNeighbors, maxNeighbors, boxSize, countPerBox, tu, inside, delta);
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

extern "C" void updateCompactedIntersectionsCUDA(int numVertices, int maxNeighbors, int* contacts, bool* insideFlag, int* shapeIds, int* startIndices, int* valid, uint64_t* outputIdx, uint64_t* intersections, int numIntersections, double2* tu, double2* tuTMP) {
    int numThreads = numVertices * maxNeighbors;
    int numBlocks = (numThreads + blockSize - 1) / blockSize;
    // Write compacted tu into tuTMP (separate buffer) to avoid the in-place scatter
    // race where outPos[b] == idx[a] causes thread b's store to corrupt thread a's load.
    updateCompactedIntersectionsKernel<<<numBlocks, blockSize>>>(numVertices, maxNeighbors, contacts, insideFlag, shapeIds, startIndices, valid, outputIdx, intersections, tu, tuTMP);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
    // Copy compacted result back into tu
    if (numIntersections > 0) {
        CUDA_CHECK(cudaMemcpy(tu, tuTMP, numIntersections * sizeof(double2), cudaMemcpyDeviceToDevice));
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

extern "C" void updateOutersectionsCUDA(const uint64_t* intersections, const double2* tu, double2* ut, int* startIndices, int numIntersections, uint64_t* outersections) {
    if (numIntersections <= 0) return;

    int blockSize = 256;
    int gridSize = (numIntersections + blockSize - 1) / blockSize;
    updateOutersectionsKernel<<<gridSize, blockSize>>>(intersections, tu, ut, startIndices, numIntersections, outersections);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C" void updateForceEnergyExteriorCUDA(int numVertices, int numIntersections, const uint64_t* intersections, const uint64_t* outersections, const double2* tu, const double2* ut, const double* positions, const int* next, const int* prev, const int* shapeId, const int* startIndices, double* force, double* energy) {
    if (numIntersections <= 0) return;
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
    if (numIntersections == 0) return;
    int blocks = (numIntersections + blockSize - 1) / blockSize;
    updateShapeRangesKernel<<<blocks, blockSize>>>(intersections, numIntersections, shapeStart, shapeEnd);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C" void updateForceEnergyInteriorCUDA(int numVertices, int numIntersections, const uint64_t* intersections, const uint64_t* outersections, const double2* tu, const double2* ut, const double* positions, const int* next, const int* prev, const int* shapeId, const int* startIndices, double* force, double* energy, int numPolygons, int* shapeStart, int* shapeEnd) {
    if (numIntersections == 0) return;
    // Launch interior kernel
    int threads = 256;
    int grid = (numVertices + threads - 1) / threads;
    updateForceEnergyInteriorKernel<<<grid, threads>>>(numVertices, intersections, outersections, tu, ut, positions, next, prev, shapeId, startIndices, force, energy, shapeStart, shapeEnd);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C" void updateForceEnergyVertexDiskCUDA(int numVertices, const double* positions, const int* shapeId, const int* next, const int* cellLocation, const int* neighborIndices, const int* countPerBox, int boxSize, double delta, double* force, double* energy) {
    if (delta <= 0.0) return;
    int threads = blockSize;
    int blocks  = (numVertices + threads - 1) / threads;
    updateForceEnergyVertexDiskKernel<<<blocks, threads>>>(numVertices, positions, shapeId, next, cellLocation, neighborIndices, countPerBox, boxSize, delta, force, energy);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C" void updateForceEnergyVertexDiskBallCUDA(int numVertices, const double* positions, const int* shapeId, const int* next, const int* ballNeighbors, const int* numBallNeighbors, int ballMaxNeighbors, double delta, double* force, double* energy) {
    if (delta <= 0.0) return;
    int threads = blockSize;
    int blocks  = (numVertices + threads - 1) / threads;
    updateForceEnergyVertexDiskBallKernel<<<blocks, threads>>>(numVertices, positions, shapeId, next, ballNeighbors, numBallNeighbors, ballMaxNeighbors, delta, force, energy);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C" int buildBallListCUDA(int numVertices, const double* positions, const int* shapeId, double ballRadius, int ballMaxNeighbors, int* ballNeighbors, int* numBallNeighbors, int* maxActualBallNeighbors) {
    int threads = blockSize;
    int blocks  = (numVertices + threads - 1) / threads;
    double ballRadius2 = ballRadius * ballRadius;
    buildBallListKernel<<<blocks, threads>>>(positions, shapeId, numVertices, ballRadius2, ballMaxNeighbors, ballNeighbors, numBallNeighbors);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    int initVal = INT_MIN;
    CUDA_CHECK(cudaMemcpy(maxActualBallNeighbors, &initVal, sizeof(int), cudaMemcpyHostToDevice));
    int reduceThreads = 256;
    int reduceBlocks = (numVertices + reduceThreads - 1) / reduceThreads;
    maxReduceKernel<<<reduceBlocks, reduceThreads, reduceThreads * sizeof(int)>>>(numBallNeighbors, numVertices, maxActualBallNeighbors);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
    int newMaxActual;
    CUDA_CHECK(cudaMemcpy(&newMaxActual, maxActualBallNeighbors, sizeof(int), cudaMemcpyDeviceToHost));
    return newMaxActual;
}

// Feature-pair refactor (Phase A): launch the candidate emitter.
//
// outPairs   : device buffer of feat::FeaturePair, length >= outCapacity
// outCount   : single device int, used as an atomic counter (reset to 0 here)
// outCapacity: size of outPairs in entries
// featureSetSize: 1 (normal), 2 (single-arc rounded), 3 (biarc)
//
// Returns the host-visible count of pairs emitted. If > outCapacity, caller
// should resize and re-run.
extern "C" int emitFeaturePairsCUDA(
    int numVertices,
    const int* shapeId,
    const int* ballNeighbors,
    const int* numBallNeighbors,
    int ballMaxNeighbors,
    int featureSetSize,
    unsigned int* outPairsRaw,    // 2 * outCapacity uints, treated as feat::FeaturePair*
    int* outCount,
    int outCapacity)
{
    feat::FeaturePair* outPairs = reinterpret_cast<feat::FeaturePair*>(outPairsRaw);
    CUDA_CHECK(cudaMemset(outCount, 0, sizeof(int)));
    int threads = blockSize;
    int blocks  = (numVertices + threads - 1) / threads;
    emitFeaturePairsKernel<<<blocks, threads>>>(
        numVertices, shapeId, ballNeighbors, numBallNeighbors, ballMaxNeighbors,
        featureSetSize, outPairs, outCount, outCapacity);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
    int count;
    CUDA_CHECK(cudaMemcpy(&count, outCount, sizeof(int), cudaMemcpyDeviceToHost));
    return count;
}

// Phase B launcher: per-pair crossing detection (edge-edge only in this
// installment). pairsRaw is reinterpreted as feat::FeaturePair*, and
// outCrossingsRaw as feat::Crossing* (each crossing is sizeof(feat::Crossing)
// bytes -- the host gets that via crossingByteSize() below).
extern "C" int detectCrossingsCUDA(
    int numPairs,
    const unsigned int* pairsRaw,
    const double* positions,
    const int* next,
    const int* prev,
    const int* shapeId,
    const double* polyDelta,
    void* outCrossingsRaw,
    int* outCount,
    int outCapacity)
{
    const feat::FeaturePair* pairs = reinterpret_cast<const feat::FeaturePair*>(pairsRaw);
    feat::Crossing* outCrossings   = reinterpret_cast<feat::Crossing*>(outCrossingsRaw);

    CUDA_CHECK(cudaMemset(outCount, 0, sizeof(int)));
    if (numPairs <= 0) return 0;
    int threads = blockSize;
    int blocks  = (numPairs + threads - 1) / threads;
    detectCrossingsKernel<<<blocks, threads>>>(
        numPairs, pairs, positions, next, prev, shapeId, polyDelta,
        outCrossings, outCount, outCapacity);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
    int count;
    CUDA_CHECK(cudaMemcpy(&count, outCount, sizeof(int), cudaMemcpyDeviceToHost));
    return count;
}

// Phase C scaffolding: sort the crossing buffer in place by (fA, paramA) and
// run-length-encode the sorted fA stream so the host (and per-feature walk
// kernels) can find each A-feature's contiguous slice in O(1).
//
// Memory contract:
//   - crossingsInOut : crossing array of length numCrossings (modified in place)
//   - crossingsTMP   : scratch of the same length (caller-owned)
//   - keys / idx     : numCrossings uint64 / uint32 scratch
//   - uniqueFAOut / lengthsOut : at least numCrossings uint32 each (RLE output)
//   - numUniqueOut   : single int (RLE output count)
// Returns numUniqueFA (number of distinct A-features with at least one crossing).
extern "C" int sortAndRleCrossingsCUDA(
    int numCrossings,
    void* crossingsInOut,
    void* crossingsTMP,
    uint64_t* keys,
    uint32_t* idx,
    uint32_t* uniqueFAOut,
    uint32_t* lengthsOut,
    int* numUniqueOut)
{
    if (numCrossings <= 0) {
        CUDA_CHECK(cudaMemset(numUniqueOut, 0, sizeof(int)));
        return 0;
    }
    feat::Crossing* crossings    = reinterpret_cast<feat::Crossing*>(crossingsInOut);
    feat::Crossing* crossingsTmp = reinterpret_cast<feat::Crossing*>(crossingsTMP);

    int threads = blockSize;
    int blocks  = (numCrossings + threads - 1) / threads;

    // 1) build (key, identity-perm).
    buildCrossingKeysKernel<<<blocks, threads>>>(numCrossings, crossings, keys, idx);
    CUDA_CHECK_KERNEL();

    // 2) sort idx by key. Need a scratch keys-out buffer; reuse end of TMP via
    // a small extra alloc (numCrossings * uint64 = ~tens of MB worst case; OK
    // for now -- could pool later).
    uint64_t* keysOut = nullptr;
    uint32_t* idxOut  = nullptr;
    CUDA_CHECK(cudaMalloc(&keysOut, (size_t)numCrossings * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&idxOut,  (size_t)numCrossings * sizeof(uint32_t)));

    void*  d_temp = nullptr;
    size_t d_temp_bytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        d_temp, d_temp_bytes, keys, keysOut, idx, idxOut, numCrossings));
    CUDA_CHECK(cudaMalloc(&d_temp, d_temp_bytes));
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        d_temp, d_temp_bytes, keys, keysOut, idx, idxOut, numCrossings));
    CUDA_CHECK(cudaFree(d_temp));

    // 3) gather crossings -> tmp, then memcpy back to in-place buffer.
    gatherCrossingsKernel<<<blocks, threads>>>(numCrossings, crossings, idxOut, crossingsTmp);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(crossings, crossingsTmp,
                          (size_t)numCrossings * sizeof(feat::Crossing),
                          cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaFree(keysOut));
    CUDA_CHECK(cudaFree(idxOut));

    // 4) extract fA into a flat uint32 stream, then RLE-encode.
    // Reuse idx scratch for the fA stream (same size, same type).
    extractFAKernel<<<blocks, threads>>>(numCrossings, crossings, idx);
    CUDA_CHECK_KERNEL();

    void*  d_temp2 = nullptr;
    size_t d_temp2_bytes = 0;
    CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(
        d_temp2, d_temp2_bytes, idx, uniqueFAOut, lengthsOut, numUniqueOut, numCrossings));
    CUDA_CHECK(cudaMalloc(&d_temp2, d_temp2_bytes));
    CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(
        d_temp2, d_temp2_bytes, idx, uniqueFAOut, lengthsOut, numUniqueOut, numCrossings));
    CUDA_CHECK(cudaFree(d_temp2));

    CUDA_CHECK(cudaDeviceSynchronize());
    int nUnique = 0;
    CUDA_CHECK(cudaMemcpy(&nUnique, numUniqueOut, sizeof(int), cudaMemcpyDeviceToHost));
    return nUnique;
}

// Per-A-polygon walk: sorts crossings by (sA, boundary-order param), runs
// RLE on the sA stream, then dispatches one thread per polygon to walk all
// of A's crossings in boundary order with global per-partner parity tracking.
// This fixes the cross-feature chord-pair bug in phaseCWalkAreaPerFeatureCUDA.
//
// Input: crossingsInOut is the unsorted crossing buffer from Phase B
//        (this routine sorts in-place using crossingsTMP scratch).
// Output: areaOut[s] = chord-area sum for polygon s (signed). Caller sums
//         across polygons and divides by 2 to get the total overlap area
//         (each pair contributes from both sides of the walk).
//
// Returns numUniqueShapes (the number of polygons that participated).
extern "C" int phaseCWalkAreaPerPolygonCUDA(
    int numCrossings,
    void* crossingsInOut,
    void* crossingsTMP,
    uint64_t* keys,
    uint32_t* idx,
    uint32_t* uniqueShape_d,
    uint32_t* lengths_d,
    uint32_t* sliceStart_d,
    int* numUniqueOut_d,
    const int* shapeId,
    const int* startIndices,
    const double* positions,
    const double* polyDelta,
    double* areaOut)
{
    if (numCrossings <= 0) {
        CUDA_CHECK(cudaMemset(numUniqueOut_d, 0, sizeof(int)));
        return 0;
    }
    // PHASEC_MAX_PARTNERS = 256, ~4.3 KB per thread of stack; lift limit once.
    static bool stack_lifted = false;
    if (!stack_lifted) {
        CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 16 * 1024));
        stack_lifted = true;
    }
    feat::Crossing* crossings    = reinterpret_cast<feat::Crossing*>(crossingsInOut);
    feat::Crossing* crossingsTmp = reinterpret_cast<feat::Crossing*>(crossingsTMP);

    int threads = blockSize;

    // 0) Duplicate crossings with swapped (fA, fB) so each polygon-pair (A, B)
    // contributes one record to A's slice AND one to B's slice. Phase A only
    // emits pairs in the sA < sB direction, so without this only the lower-id
    // polygon would have crossings; the partner's path integral would be
    // missing, breaking the anchor-term cancellation in Green's theorem.
    int blocks_orig = (numCrossings + threads - 1) / threads;
    duplicateCrossingsKernel<<<blocks_orig, threads>>>(numCrossings, crossings, crossingsTmp);
    CUDA_CHECK_KERNEL();
    int totalCrossings = 2 * numCrossings;
    CUDA_CHECK(cudaMemcpy(crossings, crossingsTmp,
                          (size_t)totalCrossings * sizeof(feat::Crossing),
                          cudaMemcpyDeviceToDevice));
    int blocks = (totalCrossings + threads - 1) / threads;

    // 1) Build per-polygon keys (sA << 32) | float_bits(boundary-order param).
    buildPolygonKeysKernel<<<blocks, threads>>>(totalCrossings, crossings,
                                                  shapeId, startIndices, keys, idx);
    CUDA_CHECK_KERNEL();

    // 2) Sort keys -> idx permutation.
    uint64_t* keysOut = nullptr;
    uint32_t* idxOut  = nullptr;
    CUDA_CHECK(cudaMalloc(&keysOut, (size_t)totalCrossings * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&idxOut,  (size_t)totalCrossings * sizeof(uint32_t)));
    void* d_temp = nullptr; size_t d_temp_bytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(d_temp, d_temp_bytes,
                                                 keys, keysOut, idx, idxOut, totalCrossings));
    CUDA_CHECK(cudaMalloc(&d_temp, d_temp_bytes));
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(d_temp, d_temp_bytes,
                                                 keys, keysOut, idx, idxOut, totalCrossings));
    CUDA_CHECK(cudaFree(d_temp));

    // 3) Gather crossings by permutation into TMP, then copy back in place.
    gatherCrossingsKernel<<<blocks, threads>>>(totalCrossings, crossings, idxOut, crossingsTmp);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(crossings, crossingsTmp,
                          (size_t)totalCrossings * sizeof(feat::Crossing),
                          cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaFree(keysOut));
    CUDA_CHECK(cudaFree(idxOut));

    // 4) Extract sA stream for RLE.
    uint32_t* shapeStream = nullptr;
    CUDA_CHECK(cudaMalloc(&shapeStream, (size_t)totalCrossings * sizeof(uint32_t)));
    extractShapeIdKernel<<<blocks, threads>>>(totalCrossings, crossings, shapeId, shapeStream);
    CUDA_CHECK_KERNEL();

    // 5) RLE: produces (uniqueShape_d, lengths_d, numUnique).
    void* d_temp2 = nullptr; size_t d_temp2_bytes = 0;
    CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(d_temp2, d_temp2_bytes,
                                                    shapeStream, uniqueShape_d,
                                                    lengths_d, numUniqueOut_d,
                                                    totalCrossings));
    CUDA_CHECK(cudaMalloc(&d_temp2, d_temp2_bytes));
    CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(d_temp2, d_temp2_bytes,
                                                    shapeStream, uniqueShape_d,
                                                    lengths_d, numUniqueOut_d,
                                                    totalCrossings));
    CUDA_CHECK(cudaFree(d_temp2));
    CUDA_CHECK(cudaFree(shapeStream));

    CUDA_CHECK(cudaDeviceSynchronize());
    int numUnique = 0;
    CUDA_CHECK(cudaMemcpy(&numUnique, numUniqueOut_d, sizeof(int), cudaMemcpyDeviceToHost));
    if (numUnique <= 0) return 0;

    // 6) Prefix-sum lengths into sliceStart.
    exclusiveScanU32Kernel<<<1, 1>>>(lengths_d, sliceStart_d, numUnique);
    CUDA_CHECK_KERNEL();

    // 7) Per-polygon walk.
    int threads2 = blockSize;
    int blocks2  = (numUnique + threads2 - 1) / threads2;
    phaseCWalkAreaPerPolygonKernel<<<blocks2, threads2>>>(
        numUnique, uniqueShape_d, lengths_d, sliceStart_d,
        crossings, shapeId, startIndices, positions, polyDelta, areaOut);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
    return numUnique;
}

// Phase C installment 2 launcher: per-feature area walk. Given the sorted
// crossings and RLE output from installment 1, compute the per-A-feature
// inside chord-area sum. Caller passes scratch (sliceStart of length
// numUniqueFA uint32) and output buffer (areaOut of length numUniqueFA
// double).
extern "C" void phaseCWalkAreaPerFeatureCUDA(
    int numUniqueFA,
    const uint32_t* uniqueFA,
    const uint32_t* lengths,
    uint32_t*       sliceStart,
    const void*     crossingsRaw,
    const int*      shapeId,
    const int*      startIndices,
    const double*   positions,
    double*         areaOut)
{
    if (numUniqueFA <= 0) return;
    const feat::Crossing* crossings = reinterpret_cast<const feat::Crossing*>(crossingsRaw);

    // Build exclusive-scan of lengths -> sliceStart (single-thread; tiny).
    exclusiveScanU32Kernel<<<1, 1>>>(lengths, sliceStart, numUniqueFA);
    CUDA_CHECK_KERNEL();

    int threads = blockSize;
    int blocks  = (numUniqueFA + threads - 1) / threads;
    phaseCWalkAreaPerFeatureKernel<<<blocks, threads>>>(
        numUniqueFA, uniqueFA, lengths, sliceStart,
        crossings, shapeId, startIndices, positions, areaOut);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Phase C installment 4 launcher: per-feature area + edge-edge force walk
// with dual-traced tangent-point Jacobian (delta>0 supported).
extern "C" void phaseCWalkAreaForceDualPerFeatureCUDA(
    int numUniqueFA,
    int numVertices,
    const uint32_t* uniqueFA,
    const uint32_t* lengths,
    uint32_t*       sliceStart,
    const void*     crossingsRaw,
    const int*      shapeId,
    const int*      startIndices,
    const int*      next_arr,
    const int*      prev_arr,
    const double*   positions,
    double          delta,
    double*         areaOut,
    double*         forceOut)
{
    if (numUniqueFA <= 0) return;
    const feat::Crossing* crossings = reinterpret_cast<const feat::Crossing*>(crossingsRaw);

    // Dual<16> + ~25 intermediates ~ 5-6 KB per thread; default 1 KB stack
    // overflows and silently corrupts adjacent threads' state. Lift the
    // stack limit once.
    static bool stack_lifted = false;
    if (!stack_lifted) {
        CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 16 * 1024));
        stack_lifted = true;
    }

    exclusiveScanU32Kernel<<<1, 1>>>(lengths, sliceStart, numUniqueFA);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemset(forceOut, 0, (size_t)numVertices * 2 * sizeof(double)));

    int threads = blockSize;
    int blocks  = (numUniqueFA + threads - 1) / threads;
    phaseCWalkAreaForceDualPerFeatureKernel<<<blocks, threads>>>(
        numUniqueFA, uniqueFA, lengths, sliceStart,
        crossings, shapeId, startIndices, next_arr, prev_arr, positions, delta,
        areaOut, forceOut);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Phase C installment 3 launcher: per-feature area walk + edge-edge force
// accumulation. forceOut must be pre-zeroed of length 2*numVertices doubles.
extern "C" void phaseCWalkAreaForcePerFeatureCUDA(
    int numUniqueFA,
    int numVertices,
    const uint32_t* uniqueFA,
    const uint32_t* lengths,
    uint32_t*       sliceStart,
    const void*     crossingsRaw,
    const int*      shapeId,
    const int*      startIndices,
    const int*      next_arr,
    const int*      prev_arr,
    const double*   polyDelta,
    const double*   positions,
    double*         areaOut,
    double*         forceOut)
{
    if (numUniqueFA <= 0) return;
    const feat::Crossing* crossings = reinterpret_cast<const feat::Crossing*>(crossingsRaw);

    // Build exclusive-scan of lengths -> sliceStart.
    exclusiveScanU32Kernel<<<1, 1>>>(lengths, sliceStart, numUniqueFA);
    CUDA_CHECK_KERNEL();

    // Zero forceOut.
    CUDA_CHECK(cudaMemset(forceOut, 0, (size_t)numVertices * 2 * sizeof(double)));

    int threads = blockSize;
    int blocks  = (numUniqueFA + threads - 1) / threads;
    phaseCWalkAreaForcePerFeatureKernel<<<blocks, threads>>>(
        numUniqueFA, uniqueFA, lengths, sliceStart,
        crossings, shapeId, startIndices, prev_arr, polyDelta,
        next_arr, positions, areaOut, forceOut);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Lets the host query sizeof(feat::Crossing) without including features.cuh
// in pure-C++ TUs.
extern "C" int crossingByteSize() {
    return (int)sizeof(feat::Crossing);
}

// Per-polygon uniform-delta mode: one-time target initialization and per-step
// update. See the kernel comments in kernels.cuh for the math.
extern "C" void fillScalarCUDA(double* data, int n, double value) {
    int threads = blockSize;
    int blocks  = (n + threads - 1) / threads;
    fillScalarKernel<<<blocks, threads>>>(data, n, value);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C" void initPolyDeltaTargetsCUDA(int numPolygons,
                                          const double* positions,
                                          const int* startIndices,
                                          double deltaInitial,
                                          double* polyDeltaTargets) {
    int threads = blockSize;
    int blocks  = (numPolygons + threads - 1) / threads;
    initPolyDeltaTargetsKernel<<<blocks, threads>>>(numPolygons, positions, startIndices,
                                                     deltaInitial, polyDeltaTargets);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C" void updatePolyDeltaCUDA(int numPolygons,
                                     const double* positions,
                                     const int* startIndices,
                                     const double* polyDeltaTargets,
                                     double* polyDelta) {
    int threads = blockSize;
    int blocks  = (numPolygons + threads - 1) / threads;
    updatePolyDeltaKernel<<<blocks, threads>>>(numPolygons, positions, startIndices,
                                                polyDeltaTargets, polyDelta);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C" double maxDisplacementCUDA(int numVertices, const double* positions, const double* refPositions, double* dispSq) {
    int threads = blockSize;
    int blocks  = (numVertices + threads - 1) / threads;
    ballMaxDisplacementSqKernel<<<blocks, threads>>>(positions, refPositions, numVertices, dispSq);
    CUDA_CHECK_KERNEL();

    double* dResult;
    CUDA_CHECK(cudaMalloc(&dResult, sizeof(double)));
    void* dTemp = nullptr;
    size_t tempBytes = 0;
    CUDA_CHECK(cub::DeviceReduce::Max(dTemp, tempBytes, dispSq, dResult, numVertices));
    CUDA_CHECK(cudaMalloc(&dTemp, tempBytes));
    CUDA_CHECK(cub::DeviceReduce::Max(dTemp, tempBytes, dispSq, dResult, numVertices));
    CUDA_CHECK(cudaDeviceSynchronize());
    double maxSq;
    CUDA_CHECK(cudaMemcpy(&maxSq, dResult, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dTemp));
    CUDA_CHECK(cudaFree(dResult));
    return sqrt(maxSq);
}

extern "C" void updateForceEnergyRoundedCUDA(
    int numVertices,
    const double* positions, const int* shapeId,
    const int* next, const int* prev, const int* startIndices,
    const int* ballNeighbors, const int* numBallNeighbors, int ballMaxNeighbors,
    double delta, int polygonSize,
    double* scratch, double* force, double* energy)
{
    if (delta <= 0.0) return;
    // Cap at 32 (size of the per-thread processedShapes[] stack array in the kernel).
    // Was previously polygonSize, but order of candidate encounter then determined
    // which neighbor shapes get processed when numPolygons > maxProc; we want the
    // result to be order-independent so cells and balls candidate paths agree.
    int maxProc = 32;
    int threads = blockSize;
    int blocks  = (numVertices + threads - 1) / threads;
    updateForceEnergyRoundedKernel<<<blocks, threads>>>(
        numVertices, positions, shapeId, next, prev, startIndices,
        ballNeighbors, numBallNeighbors, ballMaxNeighbors,
        delta, polygonSize, maxProc, scratch, force, energy,
        nullptr, nullptr, 0);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C" int populateCandidatesFromCellsCUDA(int numVertices, const int* shapeId,
    const int* cellLocation, const int* neighborIndices, const int* countPerBox,
    int boxSize, int ballMaxNeighbors,
    int* ballNeighbors, int* numBallNeighbors, int* maxActualBallNeighbors)
{
    int threads = blockSize;
    int blocks  = (numVertices + threads - 1) / threads;
    populateCandidatesFromCellsKernel<<<blocks, threads>>>(
        shapeId, cellLocation, neighborIndices, countPerBox,
        boxSize, numVertices, ballMaxNeighbors,
        ballNeighbors, numBallNeighbors);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    int initVal = INT_MIN;
    CUDA_CHECK(cudaMemcpy(maxActualBallNeighbors, &initVal, sizeof(int), cudaMemcpyHostToDevice));
    int reduceThreads = 256;
    int reduceBlocks = (numVertices + reduceThreads - 1) / reduceThreads;
    maxReduceKernel<<<reduceBlocks, reduceThreads, reduceThreads * sizeof(int)>>>(numBallNeighbors, numVertices, maxActualBallNeighbors);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
    int newMaxActual;
    CUDA_CHECK(cudaMemcpy(&newMaxActual, maxActualBallNeighbors, sizeof(int), cudaMemcpyDeviceToHost));
    return newMaxActual;
}

extern "C" void updateForceEnergyAreaSquaredCUDA(
    int numVertices, int numPolygons,
    const double* positions, const int* shapeId,
    const int* next, const int* prev, const int* startIndices,
    const int* ballNeighbors, const int* numBallNeighbors, int ballMaxNeighbors,
    double delta, int polygonSize,
    double* scratch, double* force, double* energy,
    double* pairArea)
{
    if (delta <= 0.0) return;
    int maxProc = 32;
    int threads = blockSize;
    int blocks  = (numVertices + threads - 1) / threads;

    // Pass 1: accumulate per-(sA,sB) areas into pairArea
    CUDA_CHECK(cudaMemset(pairArea, 0, (long long)numPolygons * numPolygons * sizeof(double)));
    updateForceEnergyRoundedKernel<<<blocks, threads>>>(
        numVertices, positions, shapeId, next, prev, startIndices,
        ballNeighbors, numBallNeighbors, ballMaxNeighbors,
        delta, polygonSize, maxProc, scratch, force, energy,
        pairArea, nullptr, numPolygons);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Pass 2: compute forces and energy scaled by area
    updateForceEnergyRoundedKernel<<<blocks, threads>>>(
        numVertices, positions, shapeId, next, prev, startIndices,
        ballNeighbors, numBallNeighbors, ballMaxNeighbors,
        delta, polygonSize, maxProc, scratch, force, energy,
        nullptr, pairArea, numPolygons);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C" void updateForceEnergyEdgeCUDA(int numVertices, const double* positions, const double* targetEdgeLengths, const double* edgeLengths, const int* next, const int* prev, const int* shapeId, double* force, double* energy, double stiffness) {
    int threads = 256;
    int grid = (numVertices + threads - 1) / threads;
    updateForceEnergyEdgeKernel<<<grid, threads>>>(numVertices, positions, targetEdgeLengths, edgeLengths, next, prev, shapeId, force, energy, stiffness);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C" void updateForceEnergyAreaCUDA(int numVertices, const int* shapeId, const int* next, const int* prev, const double* positions, const double* areas, const double* targetAreas, const int* startIndices, double* force, double* energy, double compressibility) {
    int threads = 256;
    int grid = (numVertices + threads - 1) / threads;
    updateForceEnergyAreaKernel<<<grid, threads>>>(numVertices, shapeId, next, prev, positions, areas, targetAreas, startIndices, force, energy, compressibility);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C" void updatePositionsCUDA(int numVertices, double* positions, const double* force, double dt) {
    int initBlocks = (numVertices * 2 + blockSize - 1) / blockSize;
    updatePositionsKernel<<<initBlocks, blockSize>>>(numVertices, positions, force, dt);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C" void updatePositionAndVelocityFIRECUDA(int numVertices, double* positions, double* velocities, const double* force, double dt) {
    int blocks = (numVertices + blockSize - 1) / blockSize;
    updatePositionAndVelocityFIREKernel<<<blocks, blockSize>>>(numVertices, positions, velocities, force, dt);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C" void updateVelocityFIRECUDA(int numVertices, double* velocities, const double* force, double dt) {
    int blocks = (numVertices + blockSize - 1) / blockSize;
    updateVelocityFIREKernel<<<blocks, blockSize>>>(numVertices, velocities, force, dt);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Computes P = F·v, bends velocity toward force direction, returns P.
extern "C" double bendVelocityTowardsForceFIRECUDA(int numVertices, double* velocities, const double* force, double alpha, double* scratch, double* result) {
    int n = numVertices * 2;
    double P = dotProductFIRE(force, velocities, n, scratch, result);
    double vnorm = sqrt(dotProductFIRE(velocities, velocities, n, scratch, result));
    double fnorm = sqrt(dotProductFIRE(force, force, n, scratch, result));
    int blocks = (numVertices + blockSize - 1) / blockSize;
    bendVelocityTowardsForceFIREKernel<<<blocks, blockSize>>>(numVertices, velocities, force, alpha, vnorm, fnorm);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
    return P;
}

extern "C" void rederiveVelocityFromDisplacementFIRECUDA(int numVertices, double* vel, const double* posNew, const double* posOld, double dt) {
    int blocks = (numVertices + blockSize - 1) / blockSize;
    rederiveVelocityFromDisplacementKernel<<<blocks, blockSize>>>(numVertices, vel, posNew, posOld, dt);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Per-vertex force magnitude clamp (preserves direction)
__global__ void clampForceMagKernel(int n, double* force, double maxMag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double fx = force[2*i], fy = force[2*i+1];
    double mag2 = fx*fx + fy*fy;
    if (mag2 > maxMag * maxMag) {
        double scale = maxMag / sqrt(mag2);
        force[2*i]   = fx * scale;
        force[2*i+1] = fy * scale;
    }
}

extern "C" void clampForceMagCUDA(int numVertices, double* force, double maxMag) {
    int block = 256;
    int grid  = (numVertices + block - 1) / block;
    clampForceMagKernel<<<grid, block>>>(numVertices, force, maxMag);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}

// getters

extern "C" double getMaxUnbalancedForceCUDA(int numVertices, double* force) {
    return maxAbsValue(force, numVertices * 2);
}

// SHAKE constraint projection
// One block per polygon, blockDim.x = n (vertices per polygon).
// Shared memory size = (n*2 + n*2 + (n+1)*(n+1) + (n+1)) * sizeof(double)
extern "C" int shakeProjectCUDA(int numPolygons, int n,
                                 const int* startIndices, const int* next, const int* prev,
                                 double* positions,
                                 const double* targetEdgeLengths, const double* targetAreas,
                                 int maxIter, double tol, int* maxIterOut) {
    if (numPolygons == 0 || maxIter == 0) return 0;
    CUDA_CHECK(cudaMemset(maxIterOut, 0, sizeof(int)));
    int nc = n + 1;
    size_t smem = (size_t)(n*2 + n*2 + nc*nc + nc + 1) * sizeof(double);
    shakeProjectKernel<<<numPolygons, n, smem>>>(
        numPolygons, startIndices, next, prev,
        positions, targetEdgeLengths, targetAreas, maxIter, tol, maxIterOut);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
    int result;
    CUDA_CHECK(cudaMemcpy(&result, maxIterOut, sizeof(int), cudaMemcpyDeviceToHost));
    return result;
}

// misc
extern "C" void resetAreasCUDA(const int numVertices, const int* shapeId, double* positions, const double* areas, const double* targetAreas, const double* comX, const double* comY) {
    int threads = 256;
    int grid = (numVertices * 2 + threads - 1) / threads;
    resetAreasKernel<<<grid, threads>>>(numVertices, shapeId, positions, areas, targetAreas, comX, comY);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C" void negateArrayCUDA(double* arr, int n) {
    int threads = blockSize;
    int grid = (n + threads - 1) / threads;
    negateArrayKernel<<<grid, threads>>>(arr, n);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}