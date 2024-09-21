//
// Created by agalex on 6/14/24.
//
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/system/omp/execution_policy.h>
#include <thrust/system/omp/vector.h>
#include <thrust/unique.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include "GridMeshCuda.cuh"

#define INF ~0

namespace {

    struct PointCloud{
        float *pnts_x;
        float *pnts_y;
        float *pnts_z;
    };

    struct NeighborHelper{
        int pointsOnCard;
        int k;
        float* knnDistances;
        int* knn;
        uint32_t* triangleIndexes;
        uint32_t* flag;
        int* counters;
        uint32_t* triangleIndexesVisited;
    };

    __device__
    int FindPosition(float array[], float x, int low, int high) {
        // Repeat until the pointers low and high meet each other
        while (low <= high) {
            int mid = low + (high - low) / 2;

            if (array[mid] == x)
                return mid;

            if (array[mid] < x)
                low = mid + 1;

            else
                high = mid - 1;
        }

        return low + (high - low) / 2;
    }

    __device__
    bool appendDistanceGrid(float *knnDistances, int *knn, float distQuery, uint32_t idxNeighbor, uint32_t indexReference, int k,
                            int &counter, uint offset) {
        indexReference -= offset;
        float maxRadius = knnDistances[k - 1 + k * (size_t)indexReference];
        if (distQuery < maxRadius) {
            bool found = false;
            for (int i = 0; i < counter; i++) {
                if (knn[i + k * (size_t)indexReference] == idxNeighbor) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                int pos = FindPosition(knnDistances + k * (size_t)indexReference, distQuery,
                                       0, counter - 1);
                if (counter == k) counter--;
                for (int i = counter - 1; i >= pos; i--) {
                    knnDistances[i + 1 + k * (size_t)indexReference] = knnDistances[i + k * (size_t)indexReference];
                    knn[i + 1 + k * (size_t)indexReference] = knn[i + k * (size_t)indexReference];
                }
                knnDistances[pos + k * (size_t)indexReference] = distQuery;
                knn[pos + k * (size_t)indexReference] = (int)idxNeighbor;
                counter++;
                return true;
            }
        }
        return false;
    }

    __device__ __host__
    float EucledianDistance(uint32_t i, uint32_t j, float *pnts_x, float *pnts_y, float *pnts_z) {
        float xi = pnts_x[i];
        float yi = pnts_y[i];
        float zi = pnts_z[i];
        float xj = pnts_x[j];
        float yj = pnts_y[j];
        float zj = pnts_z[j];
        return (xi - xj) * (xi - xj) + (yi - yj) * (yi - yj) + (zi - zj) * (zi - zj);
    }

    __global__
    void getMeshNeihbors(PointCloud referencePC, NeighborHelper helper, uint32_t advance, uint32_t trianglesCount){
        unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i < trianglesCount){
            uint32_t i1 = helper.triangleIndexes[3 * (size_t)i];
            uint32_t i2 = helper.triangleIndexes[3 * (size_t)i + 1];
            uint32_t i3 = helper.triangleIndexes[3 * (size_t)i + 2];
            bool valid_triangle = (i1 != INF && i2 != INF && i3 != INF);
            uint32_t indexes[3] = {i1, i2, i3};
            for (int ii1 = 0; ii1 < 3; ii1++){
                uint32_t realIdxReference = indexes[ii1];
                int indexInCard = (int)realIdxReference - advance;
                if (indexInCard >= 0 && indexInCard < helper.pointsOnCard) {
                    if (valid_triangle) {
                        if (!helper.triangleIndexesVisited[3 * (size_t) i + ii1]) {
                            uint32_t newValue = 1;
                            if (!atomicExch(helper.flag + indexInCard, newValue)) {
                                int counter = helper.counters[indexInCard];
                                for (int ii2 = 0; ii2 < 3; ii2++) {
                                    if (ii1 != ii2) {
                                        uint32_t realIdxNeighbor = indexes[ii2];
                                        float dist = EucledianDistance(realIdxReference, realIdxNeighbor,
                                                                       referencePC.pnts_x,
                                                                       referencePC.pnts_y, referencePC.pnts_z);
                                        appendDistanceGrid(helper.knnDistances, helper.knn, dist, realIdxNeighbor,
                                                           realIdxReference,
                                                           helper.k, counter,
                                                           advance);
                                    }
                                }
                                helper.counters[indexInCard] = counter;
                                helper.triangleIndexesVisited[3 * (size_t) i + ii1] = 1;
                            }
                        }
                    }
                    else helper.triangleIndexesVisited[3 * (size_t) i + ii1] = 1;
                }
            }
        }
    }

    __device__ void lock(uint32_t * mutex) {
        while(atomicCAS(mutex, 0, 1) != 0);
    }

    __device__ void unlock(uint32_t * mutex) {
        atomicExch(mutex, 0);
    }

    __global__
    void getMeshNeihborsSecond(PointCloud referencePC, NeighborHelper helper, uint32_t advance, uint32_t trianglesCount){
        unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i < trianglesCount){
            uint32_t i1 = helper.triangleIndexes[3 * (size_t)i];
            uint32_t i2 = helper.triangleIndexes[3 * (size_t)i + 1];
            uint32_t i3 = helper.triangleIndexes[3 * (size_t)i + 2];
            bool valid_triangle = (i1 != INF && i2 != INF && i3 != INF);
            uint32_t indexes[3] = {i1, i2, i3};
            for (int ii1 = 0; ii1 < 3; ii1++){
                uint32_t realIdxReference = indexes[ii1];
                int indexInCard = (int)realIdxReference - advance;
                if (indexInCard >= 0 && indexInCard < helper.pointsOnCard) {
                    if (valid_triangle) {
                        lock(helper.flag + indexInCard);
                        int counter = helper.counters[indexInCard];
                        for (int ii2 = 0; ii2 < 3; ii2++) {
                            if (ii1 != ii2) {
                                uint32_t realIdxNeighbor = indexes[ii2];
                                float dist = EucledianDistance(realIdxReference, realIdxNeighbor,
                                                               referencePC.pnts_x,
                                                               referencePC.pnts_y, referencePC.pnts_z);
                                appendDistanceGrid(helper.knnDistances, helper.knn, dist, realIdxNeighbor,
                                                   realIdxReference,
                                                   helper.k, counter,
                                                   advance);
                            }
                        }
                        helper.counters[indexInCard] = counter;
                        helper.triangleIndexesVisited[3 * (size_t) i + ii1] = 1;
                        unlock(helper.flag + indexInCard);
                    }
                    else helper.triangleIndexesVisited[3 * (size_t) i + ii1] = 1;
                }
            }
        }
    }

    template<typename T>
    __global__
    void get_buckets_no_binary(float *pnts_x, float *pnts_y, float *pnts_z,
                               T *bucket_indexes, int pointsCount, int bucketsCount, float a, float b) {
        unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i < pointsCount) {
            float x = pnts_x[i];
            float y = pnts_y[i];
            float z = pnts_z[i];
            int xi = (int) ((x - a) / (b - a) * bucketsCount);
            int yi = (int) ((y - a) / (b - a) * bucketsCount);
            int zi = (int) ((z - a) / (b - a) * bucketsCount);
            bucket_indexes[i] = xi + yi * bucketsCount + zi * bucketsCount * bucketsCount;
        }
    }

    __global__
    void normalize_points(float *x, float *y, float *z, float min, float max, int pointCount) {
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i < pointCount) {
            float d = max - min;
            x[i] = (x[i] - min) / d;
            y[i] = (y[i] - min) / d;
            z[i] = (z[i] - min) / d;
        }
    }

    __global__
    void getBucketSizes(int *bucketsCount, int *pointBucketIndexes, int pointBucketCount, int pointsCount) {
        unsigned int bucketPosition = threadIdx.x + blockDim.x * blockIdx.x;
        if (bucketPosition < pointBucketCount) {
            int startPointIndex = pointBucketIndexes[bucketPosition];
            int endPointIndex =
                    bucketPosition < pointBucketCount - 1 ? pointBucketIndexes[bucketPosition + 1] : pointsCount;
            bucketsCount[bucketPosition] = endPointIndex - startPointIndex;
        }
    }

    __global__
    void reverseTriangleIndexes(uint32_t* triangleIndexes, int* reverseIndexes, int trianglesCount){
        unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i < trianglesCount) {
            for (int j = 0; j < 3; j++) {
                triangleIndexes[3 * (size_t)i + j] = reverseIndexes[triangleIndexes[3 * (size_t)i + j]];
            }
        }
    }

    __global__
    void rearrangeReferenceNeighbors(int* knnDest, int *knnSource, int* indexesReference, int pointsCount, int k){
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i < pointsCount) {
            for (int j = 0; j < k; j++){
                int knnIdx  = knnSource[i * k + j];
                int trueIndex = knnIdx >= 0 ? indexesReference[knnIdx] : -1;
                knnDest[i * k + j] = trueIndex;
            }
        }
    }
}

GridMeshCuda::GridMeshCuda(const PLG::PLGMesh& mesh, int num_cards, float eps, int bucketCount):
        dpoints_x(num_cards, nullptr),
        dpoints_y(num_cards, nullptr),
        dpoints_z(num_cards, nullptr),
        dknn(num_cards, nullptr),
        doriginal_indexes(num_cards, nullptr),
        dtriangleIndexes(num_cards, nullptr),
        num_cards(num_cards),
        pointsCard(num_cards),
        pointsCount(mesh.nV()),
        finalBucketCount(bucketCount),
        trianglesCount(mesh.nF()),
        eps(eps)
{
    std::vector<float> points_x(pointsCount);
    std::vector<float> points_y(pointsCount);
    std::vector<float> points_z(pointsCount);
#pragma omp parallel for
    for (int i = 0; i < pointsCount; i++){
        points_x[i] = mesh.mV[i].x;
        points_y[i] = mesh.mV[i].y;
        points_z[i] = mesh.mV[i].z;
    }
    std::cout << "Init Structure" << std::endl;
    InitStructure(points_x.data(), points_y.data(), points_z.data(), mesh.mF.data());
}

template<typename T>
void GridMeshCuda::getBuckets(int dev_id, int bucketsCount, float min, float max, int &maxBucketSize) {
    int threads = 1024;
    int blocks = (int) ceil((1.0 * pointsCount) / threads);
    T* dbucketIndexes;
    cudaMalloc((void**)&dbucketIndexes, (size_t)pointsCount * sizeof(T));
    get_buckets_no_binary<T><<<blocks, threads>>>(dpoints_x[dev_id], dpoints_y[dev_id], dpoints_z[dev_id],
                                                  dbucketIndexes,
                                                  pointsCount, bucketsCount, min, max);
    cudaDeviceSynchronize();
    thrust::sequence(thrust::device, doriginal_indexes[dev_id], doriginal_indexes[dev_id] + pointsCount, 0);

    thrust::sort_by_key(thrust::device, dbucketIndexes, dbucketIndexes + pointsCount,
                        doriginal_indexes[dev_id], thrust::less<T>());
    int* dpointBucketIndexes;
    cudaMalloc((void**)&dpointBucketIndexes, (size_t)pointsCount * sizeof(int));
    thrust::sequence(thrust::device, dpointBucketIndexes, dpointBucketIndexes + pointsCount, 0);

    thrust::pair<T *, int *> end;
    end = thrust::unique_by_key(thrust::device, dbucketIndexes, dbucketIndexes + pointsCount,
                                dpointBucketIndexes, thrust::equal_to<T>());
    T vertexPartitionSize = end.first - dbucketIndexes;

    int *dpointBucketsCount;
    cudaMalloc((void **) &dpointBucketsCount, (size_t)vertexPartitionSize * sizeof(int));
    threads = 1024;
    blocks = (int) ceil((1.0 * vertexPartitionSize) / threads);
    getBucketSizes<<<blocks, threads>>>(dpointBucketsCount, dpointBucketIndexes, vertexPartitionSize, pointsCount);
    cudaDeviceSynchronize();
    maxBucketSize = thrust::reduce(thrust::device, dpointBucketsCount,
                                   dpointBucketsCount + vertexPartitionSize, -1e8f,
                                   thrust::maximum<int>());
    cudaFree(dpointBucketsCount);
    cudaFree(dpointBucketIndexes);
    cudaFree(dbucketIndexes);
}

void GridMeshCuda::InitStructure(const float *points_x,const float *points_y,const float *points_z,const PLG::I3* indexes) {
    std::vector<int> bucketCount(num_cards);
#pragma omp parallel for num_threads(num_cards)
    for (int dev_id = 0; dev_id < num_cards; dev_id++) {
        bucketCount[dev_id] = finalBucketCount;
        cudaSetDevice(dev_id);
        cudaMalloc((void **) &dpoints_x[dev_id], (size_t)pointsCount * sizeof(float));
        cudaMemcpy(dpoints_x[dev_id], points_x, (size_t)pointsCount * sizeof(float),
                   cudaMemcpyHostToDevice);

        cudaMalloc((void **) &dpoints_y[dev_id], (size_t)pointsCount * sizeof(float));
        cudaMemcpy(dpoints_y[dev_id], points_y, (size_t)pointsCount * sizeof(float),
                   cudaMemcpyHostToDevice);

        cudaMalloc((void **) &dpoints_z[dev_id], (size_t)pointsCount * sizeof(float));
        cudaMemcpy(dpoints_z[dev_id], points_z, (size_t)pointsCount * sizeof(float),
                   cudaMemcpyHostToDevice);

        float minx = thrust::reduce(thrust::device, dpoints_x[dev_id],
                                    dpoints_x[dev_id] + pointsCount, 1e8f,
                                    thrust::minimum<float>()) - 1e-6f;
        float maxx = thrust::reduce(thrust::device, dpoints_x[dev_id],
                                    dpoints_x[dev_id] + pointsCount, -1e8f,
                                    thrust::maximum<float>()) + 1e-6f;
        float miny = thrust::reduce(thrust::device, dpoints_y[dev_id],
                                    dpoints_y[dev_id] + pointsCount, 1e8f,
                                    thrust::minimum<float>()) - 1e-6f;
        float maxy = thrust::reduce(thrust::device, dpoints_y[dev_id],
                                    dpoints_y[dev_id] + pointsCount, -1e8f,
                                    thrust::maximum<float>()) + 1e-6f;
        float minz = thrust::reduce(thrust::device, dpoints_z[dev_id],
                                    dpoints_z[dev_id] + pointsCount, 1e8f,
                                    thrust::minimum<float>()) - 1e-6f;
        float maxz = thrust::reduce(thrust::device, dpoints_z[dev_id],
                                    dpoints_z[dev_id] + pointsCount, -1e8f,
                                    thrust::maximum<float>()) + 1e-6f;

        float min = std::min(minx, miny);
        min = std::min(min, minz);

        float max = std::max(maxx, maxy);
        max = std::max(max, maxz);

        int threads = 1024;
        int blocks = (int) ceil((1.0 * pointsCount) / threads);
        normalize_points<<<blocks, threads>>>(dpoints_x[dev_id],
                                              dpoints_y[dev_id],
                                              dpoints_z[dev_id], min, max, pointsCount);
        cudaDeviceSynchronize();

        if (dev_id == 0) {
            maxExtent = max;
            minExtent = min;
        }

        //allow the query points to be a little further...
        //this is very useful and avoids parts that are completely away
        //from the reference points. Since we are talking about ICP
        //the later should be avoided.
        min = -eps;
        max = 1.0 + eps;
        int maxPntBucketSize = 0;

        bucketCount[dev_id] *= 2;
        cudaMalloc((void**)&doriginal_indexes[dev_id], (size_t)pointsCount * sizeof(int));
        while (maxPntBucketSize <= 0) {
            bucketCount[dev_id] /= 2;
            getBuckets<long64>(dev_id, bucketCount[dev_id], min, max, maxPntBucketSize);
        }

        float *temp_pnts_x;
        cudaMalloc((void **) &temp_pnts_x, (size_t)pointsCount * sizeof(float));
        float *temp_pnts_y;
        cudaMalloc((void **) &temp_pnts_y, (size_t)pointsCount * sizeof(float));
        float *temp_pnts_z;
        cudaMalloc((void **) &temp_pnts_z, (size_t)pointsCount * sizeof(float));

        thrust::gather(thrust::device, doriginal_indexes[dev_id], doriginal_indexes[dev_id] + pointsCount,
                       dpoints_x[dev_id], temp_pnts_x);
        thrust::gather(thrust::device, doriginal_indexes[dev_id], doriginal_indexes[dev_id] + pointsCount,
                       dpoints_y[dev_id], temp_pnts_y);
        thrust::gather(thrust::device, doriginal_indexes[dev_id], doriginal_indexes[dev_id] + pointsCount,
                       dpoints_z[dev_id], temp_pnts_z);
        std::swap(temp_pnts_x, dpoints_x[dev_id]);
        std::swap(temp_pnts_y, dpoints_y[dev_id]);
        std::swap(temp_pnts_z, dpoints_z[dev_id]);
        cudaFree(temp_pnts_x);
        cudaFree(temp_pnts_y);
        cudaFree(temp_pnts_z);
        int* dreverse_indexes;
        int* dtemp_indexes;
        cudaMalloc((void**)& dtemp_indexes, (size_t)pointsCount * sizeof(int));
        cudaMemcpy(dtemp_indexes, doriginal_indexes[dev_id], (size_t)pointsCount * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMalloc((void**)& dreverse_indexes, (size_t)pointsCount * sizeof(int));
        thrust::sequence(thrust::device, dreverse_indexes, dreverse_indexes + pointsCount, 0);
        thrust::sort_by_key(thrust::device, dtemp_indexes, dtemp_indexes + pointsCount,
                            dreverse_indexes, thrust::less<int>());
        cudaFree(dtemp_indexes);
        std::cout << "HERE?" << std::endl;
        cudaMalloc((void**)& dtriangleIndexes[dev_id], (size_t)trianglesCount * sizeof(PLG::I3));
        cudaMemcpy(dtriangleIndexes[dev_id], indexes, (size_t)trianglesCount * sizeof(PLG::I3), cudaMemcpyHostToDevice);
        std::cout << "HERE?" << std::endl;
        threads = 1024;
        blocks = (int) ceil((1.0 * trianglesCount) / threads);
        reverseTriangleIndexes<<<blocks, threads>>>(dtriangleIndexes[dev_id], dreverse_indexes, trianglesCount);
        cudaDeviceSynchronize();
        cudaFree(dreverse_indexes);
    }
    finalBucketCount = bucketCount[0];
    std::cout << "Succeeded!" << std::endl;
    exit(0);
}

void GridMeshCuda::GetNeighborsFromTriangles(int k){
    std::cout << "Into the function!" << std::endl;
    neighborCount = k;
    std::vector<PointCloud> pcReference(num_cards);
    std::vector<int*> dcounters(num_cards, nullptr);
    std::vector<float*> dknndistances(num_cards, nullptr);
    std::vector<uint32_t*> dflag(num_cards, nullptr);
    std::vector<uint32_t*> dtriangleIndexesVisited(num_cards, nullptr);

    std::vector<NeighborHelper> neighborHelpers(num_cards);
    pointPartitionSize = pointsCount / num_cards;
    for (int i = 0; i < num_cards - 1; i++){
        pointsCard[i] = pointPartitionSize;
    }
    pointsCard[num_cards - 1] = pointsCount - (num_cards - 1) * pointPartitionSize;

    auto countNumber = [](uint32_t* numbers, int num_cards){
#pragma omp barrier
        uint32_t counter = 0;
        for (int i = 0; i < num_cards; i++){
            counter += numbers[i];
        }
#pragma omp barrier
        return counter;
    };

    std::vector<uint32_t> counters(num_cards);
#pragma omp parallel for num_threads(num_cards)
    for (int dev_id = 0; dev_id < num_cards; dev_id++) {
        cudaSetDevice(dev_id);
        neighborHelpers[dev_id].k = k;
        neighborHelpers[dev_id].pointsOnCard = pointsCard[dev_id];
        neighborHelpers[dev_id].triangleIndexes = dtriangleIndexes[dev_id];
        pcReference[dev_id].pnts_x = dpoints_x[dev_id];
        pcReference[dev_id].pnts_y = dpoints_y[dev_id];
        pcReference[dev_id].pnts_z = dpoints_z[dev_id];
        cudaMalloc((void **) &dcounters[dev_id], pointsCard[dev_id] * sizeof(int));
        cudaMemset(dcounters[dev_id], 0, pointsCard[dev_id] * sizeof(int));
        neighborHelpers[dev_id].counters = dcounters[dev_id];
        cudaMalloc((void **) &dflag[dev_id], pointsCard[dev_id] * sizeof(uint32_t));
        neighborHelpers[dev_id].flag = dflag[dev_id];
        cudaMalloc((void **) &dtriangleIndexesVisited[dev_id], 3 * (size_t) trianglesCount * sizeof(uint32_t));
        cudaMemset(dtriangleIndexesVisited[dev_id], 0, 3 * (size_t) trianglesCount * sizeof(uint32_t));
        neighborHelpers[dev_id].triangleIndexesVisited = dtriangleIndexesVisited[dev_id];
        //////////////KNEAREST NEIGHBOURS////////////////////////////////////////////////////
        cudaMalloc((void **) &dknndistances[dev_id], (size_t) pointsCard[dev_id] * k * sizeof(float));
        cudaMalloc((void **) &dknn[dev_id], (size_t) pointsCard[dev_id] * k * sizeof(int));
        neighborHelpers[dev_id].knn = dknn[dev_id];
        neighborHelpers[dev_id].knnDistances = dknndistances[dev_id];
        /////////////////////////////////////////////////////////////////////////////////////
        thrust::fill(thrust::device, dknn[dev_id], dknn[dev_id] + k * (size_t) pointsCard[dev_id], -1.f);
        thrust::fill(thrust::device, dknndistances[dev_id], dknndistances[dev_id] + k * (size_t) pointsCard[dev_id],
                     1e10f);
        std::cout << "Getting Neighbors..." << std::endl;
        while(true){
            cudaMemset(dflag[dev_id], 0, (size_t) pointsCard[dev_id] * sizeof(uint32_t));
            uint32_t threads = 1024;
            uint32_t blocks = (int) ceil((1.0 * trianglesCount) / threads);
            getMeshNeihbors<<<blocks, threads>>>(pcReference[dev_id],
                                                 neighborHelpers[dev_id], dev_id * pointPartitionSize,
                                                 trianglesCount);
            cudaDeviceSynchronize();
            uint32_t count = thrust::reduce(thrust::device, dtriangleIndexesVisited[dev_id],
                                            dtriangleIndexesVisited[dev_id] + 3 * (size_t) trianglesCount, (uint32_t)0);
            counters[dev_id] = count;
            uint32_t counterTotal = countNumber(counters.data(), num_cards);
            if (counterTotal == 3 * (size_t) trianglesCount) break;
        }
        cudaFree(dcounters[dev_id]);
        cudaFree(dknndistances[dev_id]);
        cudaFree(dflag[dev_id]);
        cudaFree(dtriangleIndexesVisited[dev_id]);
    }
}

void GridMeshCuda::GetKNNIndexes(int* knnIndexes){
    int* knnIndexesTemp = new int[(size_t)neighborCount * (size_t)pointsCount];
#pragma omp parallel for num_threads(num_cards)
    for (int dev_id = 0; dev_id < num_cards; dev_id++) {
        cudaSetDevice(dev_id);
        int* knnTemp;
        cudaMalloc((void**)& knnTemp, neighborCount * (size_t)pointsCard[dev_id] * sizeof(int));
        int threads = 1024;
        int blocks = ceil((1.0 * pointsCard[dev_id]) / threads);
        rearrangeReferenceNeighbors<<<blocks, threads>>>(knnTemp, dknn[dev_id], doriginal_indexes[dev_id],
                                                         pointsCard[dev_id], neighborCount);
        cudaDeviceSynchronize();
        cudaMemcpy(knnIndexesTemp + dev_id * neighborCount * (size_t)pointPartitionSize, knnTemp,
                   (size_t)neighborCount * pointsCard[dev_id] * sizeof(int),
                   cudaMemcpyDeviceToHost);
        cudaFree(knnTemp);
    }
    cudaSetDevice(0);
    std::vector<int> originalIndexes(pointsCount);
    cudaMemcpy(originalIndexes.data(), GetOriginalQuerIndexes()[0], (size_t)pointsCount * sizeof(int), cudaMemcpyDeviceToHost);
#pragma omp parallel for
    for (int i = 0; i < pointsCount; i++){
        int idx = originalIndexes[i];
        memcpy(knnIndexes + neighborCount * (size_t)idx, knnIndexesTemp + neighborCount * (size_t)i,
               neighborCount * sizeof(int));
    }
    delete[] knnIndexesTemp;
}

GridMeshCuda::~GridMeshCuda(){
    for (int dev_id = 0; dev_id < num_cards; dev_id++){
        cudaSetDevice(dev_id);
        cudaFree(dpoints_x[dev_id]);
        cudaFree(dpoints_y[dev_id]);
        cudaFree(dpoints_z[dev_id]);
        cudaFree(doriginal_indexes[dev_id]);
        cudaFree(dknn[dev_id]);
        cudaFree(dtriangleIndexes[dev_id]);
    }
}