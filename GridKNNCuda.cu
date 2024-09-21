#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/system/omp/execution_policy.h>
#include <thrust/system/omp/vector.h>
#include <thrust/unique.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <string>
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include "GridKNNCuda.cuh"

#define OLDCODE 0
#define NEWCODE 1

//#define MAX_PNTS_IN_CUBE 500

namespace {
    template<typename Tkey, typename Tvalue>
    bool exist(std::map<Tkey, Tvalue> mapIn, Tvalue value) {
        return (mapIn.find(value) != mapIn.end());
    }

    template<typename T>
    __device__
    int binarySearchInt(T array[], T x, int low, int high) {
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

        return -1;
    }

    __device__
    int binarySearchFloat(float value, float a, float h, int bucketsCount) {
        int lower = 0;
        int upper = bucketsCount;
        int middle = (lower + upper) / 2;
        while (middle > lower) {
            if (a + middle * h > value) upper = middle;
            else lower = middle;
            middle = (lower + upper) / 2;
        }
        return middle;
    }

    template<typename T>
    __global__
    void setValue(T *vector, T value, int elementCount) {
        unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i < elementCount) {
            vector[i] = value;
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

    __device__ __host__
    float EucledianDistance(int i, int j, float *pnts_x, float *pnts_y, float *pnts_z) {
        float xi = pnts_x[i];
        float yi = pnts_y[i];
        float zi = pnts_z[i];
        float xj = pnts_x[j];
        float yj = pnts_y[j];
        float zj = pnts_z[j];
        return (xi - xj) * (xi - xj) + (yi - yj) * (yi - yj) + (zi - zj) * (zi - zj);
    }

    __device__ __host__
    float EucledianDistance(int i, int j, float *pntsi_x, float *pntsi_y, float *pntsi_z,
                            float *pntsj_x, float *pntsj_y, float *pntsj_z) {
        float xi = pntsi_x[i];
        float yi = pntsi_y[i];
        float zi = pntsi_z[i];
        float xj = pntsj_x[j];
        float yj = pntsj_y[j];
        float zj = pntsj_z[j];
        return (xi - xj) * (xi - xj) + (yi - yj) * (yi - yj) + (zi - zj) * (zi - zj);
    }

    __device__
    void GetMaximumGrid(float *knnDistances, float &maximum, int &index, int indexReference, int k) {
        maximum = knnDistances[k * indexReference];
        index = 0;
        for (int i = 1; i < k; i++) {
            float dist = knnDistances[indexReference * k + i];
            if (maximum < dist) {
                index = i;
                maximum = dist;
            }
        }
    }

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
    bool appendDistanceGrid(float *knnDistances, int *knn, float distQuery, int idxNeighbor, int indexReference, int k,
                            int pointCount, int counter,
                            float &maximum, int &indexmaximum) {
        if (counter < k) {
            bool found = false;
            for (int i = 0; i < counter; i++) {
                if (knn[i + k * indexReference] == idxNeighbor) {
                    found = true;
                    break;
                }
            }

            if (!found) {
                int pos = 0;
                if (counter > 0) {
                    pos = FindPosition(knnDistances + k * indexReference, distQuery, 0, counter - 1);
                    for (int i = counter - 1; i >= pos; i--) {
                        knnDistances[i + 1 + k * indexReference] = knnDistances[i + k * indexReference];
                        knn[i + 1 + k * indexReference] = knn[i + k * indexReference];
                    }
                }
                knnDistances[pos + k * indexReference] = distQuery;
                knn[pos + k * indexReference] = idxNeighbor;
                if (counter == k - 1) {
                    maximum = knnDistances[k - 1 + k * indexReference];
                    indexmaximum = knn[k - 1 + k * indexReference];
                }
                return true;
            }
            return false;
        } else {
            if (distQuery < knnDistances[k - 1 + k * indexReference]) {
                bool found = false;
                for (int i = 0; i < k; i++) {
                    if (knn[i + k * indexReference] == idxNeighbor) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    int pos = FindPosition(knnDistances + k * indexReference, distQuery, 0, k - 1);
                    for (int i = k - 2; i >= pos; i--) {
                        knnDistances[i + 1 + k * indexReference] = knnDistances[i + k * indexReference];
                        knn[i + 1 + k * indexReference] = knn[i + k * indexReference];
                    }

                    knnDistances[pos + k * indexReference] = distQuery;
                    knn[pos + k * indexReference] = idxNeighbor;
                    maximum = knnDistances[k - 1 + k * indexReference];
                    indexmaximum = knn[k - 1 + k * indexReference];
                    return true;
                }
            }
        }
        return false;
    }

    __global__
    void add2vector(int *dvector, int *dtoaddvector, int pointsCount) {
        unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i < pointsCount) {
            dvector[i] |= dtoaddvector[i];
        }
    }

    __device__
    bool inbounds(int x, int min, int max) {
        if ((x >= min) && (x < max)) return true;
        return false;
    }

    __global__
    void invalidateIndexesNotValid(int *indexesQuery, int *indexesContinue, int *invalidIndexes, int pointsLeft) {
        unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i < pointsLeft) {
            int realIdxQuery = indexesQuery[i];
            if (indexesContinue[i]) {
                if (invalidIndexes[realIdxQuery]) indexesContinue[i] = !indexesContinue[i];
            }
        }
    }

    __global__
    void invalidateNotFinishedNeighbors(int *indexesQuery, int *indexesContinue, int *knn, int pointsLeft, int offset) {
        unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i < pointsLeft) {
            int realIdxQuery = indexesQuery[i];
            realIdxQuery -= offset;
            if (indexesContinue[i]) {
                knn[realIdxQuery] = -1;
            }
        }
    }
}

//namespace{
    struct PointCloud{
        float *pnts_x;
        float *pnts_y;
        float *pnts_z;
    };

    template <typename T>
    struct NeighborHelper{
        int pointBucketCount;
        int pointsReference;
        int bucketCount;
        int k;
        float xmin;
        float xmax;
        float h;
        int *pointBucketIndexes;
        T* bucketIndexes;
        float* knnDistances;
        int* knn;
        int *counters;
        int *indexesQuery;
        int *indexes;
        int *continues;
        int *reverseQueryIndexes;
        int *invalidIndexes;
    };
//}

namespace newNeighborSearch{

            __global__
    void getNewData(int *scannedIndexes, int *continues,
                    int *indexes, int *newindexes,
                    int *counters, int *newCounters,
                    int pointsCount) {
        unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i < pointsCount) {
            if (continues[i]) {
                int idx = scannedIndexes[i] - 1;
                newindexes[idx] = indexes[i];
                newCounters[idx] = counters[i];
            }
        }
    }

    __global__
    void invalidateNotFinishedNeighbors(int* indexesQuery, int* indexesContinue, int* invalidIndexes, int pointsLeft){
        unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i < pointsLeft){
            int realIdxQuery = indexesQuery[i];
            if (indexesContinue[i]){
                invalidIndexes[realIdxQuery] = 1;
            }
        }
    }

    __global__
    void negateNotValidKnnIndexes(int* knn, int k, int* invalidIndexes, int pointCount, int offset){
        unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i < pointCount){
            int idxInvalidIndexes = i + offset;
            if (invalidIndexes[idxInvalidIndexes]){
                for (int j = 0; j < k; j++){
                    knn[i * k + j] = -1;
                }
            }
        }
    }

    __device__
    bool appendDistanceGrid(float *knnDistances, int *knn, float distQuery, int idxNeighbor, int indexReference, int k,
                            int &counter, int offset) {
        indexReference -= offset;
        float maxRadius = knnDistances[k - 1 + k * indexReference];
        if (distQuery < maxRadius) {
            bool found = false;
            for (int i = 0; i < counter; i++) {
                if (knn[i + k * indexReference] == idxNeighbor) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                int pos = FindPosition(knnDistances + k * indexReference, distQuery,
                                       0, counter - 1);
                if (counter == k) counter--;
                for (int i = counter - 1; i >= pos; i--) {
                    knnDistances[i + 1 + k * indexReference] = knnDistances[i + k * indexReference];
                    knn[i + 1 + k * indexReference] = knn[i + k * indexReference];
                }
                knnDistances[pos + k * indexReference] = distQuery;
                knn[pos + k * indexReference] = idxNeighbor;
                counter++;
                return true;
            }
        }
        return false;
    }

    template<typename T>
    __global__
    void
    construct_neighbors(PointCloud referencePC, PointCloud queryPC, NeighborHelper<T> helper,
                        int offset, int ii, int jj, int kk, int pointsLeftCount, int advance,
                        float thres, int max_points_in_bucket = 1e5, bool invalidateOutOfCube = false) {
        int MAX_PNTS_IN_CUBE = max_points_in_bucket;
        unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i < pointsLeftCount) {
            int realIdxQuery = helper.indexesQuery[i];
            int counter = helper.counters[i];
            int xi = (int) ((queryPC.pnts_x[realIdxQuery] - helper.xmin) / (helper.xmax - helper.xmin) * helper.bucketCount);
            int yi = (int) ((queryPC.pnts_y[realIdxQuery] - helper.xmin) / (helper.xmax - helper.xmin) * helper.bucketCount);
            int zi = (int) ((queryPC.pnts_z[realIdxQuery] - helper.xmin) / (helper.xmax - helper.xmin) * helper.bucketCount);
            int xni = xi + ii;
            int yni = yi + jj;
            int zni = zi + kk;
            if (inbounds(xni, 0, helper.bucketCount) && inbounds(yni, 0, helper.bucketCount) && inbounds(zni, 0, helper.bucketCount)) {
                T bucketIndex = xni + yni * helper.bucketCount + zni * helper.bucketCount * helper.bucketCount;
                int bucketPosition = binarySearchInt<T>(helper.bucketIndexes, bucketIndex, 0, helper.pointBucketCount - 1);
                if (bucketPosition >= 0 && !helper.invalidIndexes[realIdxQuery]) {
                    int startPointIndex = helper.pointBucketIndexes[bucketPosition];
                    int endPointIndex = bucketPosition < helper.pointBucketCount - 1 ? helper.pointBucketIndexes[bucketPosition + 1]
                                                                              : helper.pointsReference;
                    int numbPnts = endPointIndex - startPointIndex;
                    if (numbPnts < MAX_PNTS_IN_CUBE) {
                        for (int idx = startPointIndex; idx < endPointIndex; idx++) {
                            int realIdxNeighbor = helper.indexes[idx];

                            float dist = EucledianDistance(realIdxQuery, realIdxNeighbor, queryPC.pnts_x, queryPC.pnts_y, queryPC.pnts_z,
                                                           referencePC.pnts_x, referencePC.pnts_y, referencePC.pnts_z);

                            if (dist < thres)
                                appendDistanceGrid(helper.knnDistances, helper.knn, dist, realIdxNeighbor,
                                                   helper.reverseQueryIndexes[realIdxQuery],
                                                   helper.k, counter,
                                                   advance);

                        }
                    } else helper.invalidIndexes[realIdxQuery] = 1;
                }
                helper.counters[i] = counter;
            }
            else{
                if (offset == 0 && invalidateOutOfCube){
                    helper.invalidIndexes[realIdxQuery] = 1;
                }
            }
            bool lastCube = (ii == offset) && (jj == offset) && (kk == offset) && (!helper.invalidIndexes[realIdxQuery]);
            if (lastCube) {
                if (counter < helper.k) {
                    float x = queryPC.pnts_x[realIdxQuery];
                    float y = queryPC.pnts_y[realIdxQuery];
                    float z = queryPC.pnts_z[realIdxQuery];
                    bool searchMore = false;
                    if (xi - offset >= 0 && sqrt(thres) >= x - helper.xmin - (xi - offset) * helper.h)
                        searchMore = true;
                    else if (xi + offset + 1 <= helper.bucketCount &&
                             sqrt(thres) >= helper.xmin - x + (xi + offset + 1) * helper.h)
                        searchMore = true;
                    else if (yi - offset >= 0 && sqrt(thres) >= y - helper.xmin - (yi - offset) * helper.h)
                        searchMore = true;
                    else if (yi + offset + 1 <= helper.bucketCount &&
                             sqrt(thres) >= helper.xmin - y + (yi + offset + 1) * helper.h)
                        searchMore = true;
                    else if (zi - offset >= 0 && sqrt(thres) >= z - helper.xmin - (zi - offset) * helper.h)
                        searchMore = true;
                    else if (zi + offset + 1 <= helper.bucketCount &&
                             sqrt(thres) >= helper.xmin - z + (zi + offset + 1) * helper.h)
                        searchMore = true;
                    helper.continues[i] = searchMore ? 1 : 0;
                } else {
                    float x = queryPC.pnts_x[realIdxQuery];
                    float y = queryPC.pnts_y[realIdxQuery];
                    float z = queryPC.pnts_z[realIdxQuery];
                    realIdxQuery -= advance;
                    float maximumRadius = helper.knnDistances[realIdxQuery * helper.k + helper.k - 1];
                    bool searchMore = false;
                    if (xi - offset >= 0 && sqrt(maximumRadius) >= x - helper.xmin - (xi - offset) * helper.h)
                        searchMore = true;
                    if (xi + offset + 1 <= helper.bucketCount &&
                        sqrt(maximumRadius) >= helper.xmin - x + (xi + offset + 1) * helper.h)
                        searchMore = true;
                    if (yi - offset >= 0 && sqrt(maximumRadius) >= y - helper.xmin - (yi - offset) * helper.h)
                        searchMore = true;
                    if (yi + offset + 1 <= helper.bucketCount &&
                        sqrt(maximumRadius) >= helper.xmin - y + (yi + offset + 1) * helper.h)
                        searchMore = true;
                    if (zi - offset >= 0 && sqrt(maximumRadius) >= z - helper.xmin - (zi - offset) * helper.h)
                        searchMore = true;
                    if (zi + offset + 1 <= helper.bucketCount &&
                        sqrt(maximumRadius) >= helper.xmin - z + (zi + offset + 1) * helper.h)
                        searchMore = true;
                    helper.continues[i] = searchMore ? 1 : 0;
                }
            }
        }
    }

    template<typename T>
    __global__
    void
    construct_neighbors(float *pnts_x, float *pnts_y, float *pnts_z, float *pntsq_x, float *pntsq_y, float *pntsq_z,
                        int *pointBucketIndexes, int pointBucketCount,
                        T *bucketIndexes, int pointsReference, float *knnDistances, int *knn, int bucketCount, int k,
                        int offset, int *counters, int *indexesQuery, int *indexes,
                        int *continues, int ii, int jj, int kk,
                        int pointsLeftCount, float xmin, float xmax, float h, int pointCountQuery, float thres,
                        int *reverseQueryIndexes,
                        int advance, int *invalidIndexes, int max_points_in_bucket = 1e5, bool invalidateOutOfCube = false) {
        int MAX_PNTS_IN_CUBE = max_points_in_bucket;
        unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i < pointsLeftCount) {
            int realIdxQuery = indexesQuery[i];
            int counter = counters[i];
            int xi = (int) ((pntsq_x[realIdxQuery] - xmin) / (xmax - xmin) * bucketCount);
            int yi = (int) ((pntsq_y[realIdxQuery] - xmin) / (xmax - xmin) * bucketCount);
            int zi = (int) ((pntsq_z[realIdxQuery] - xmin) / (xmax - xmin) * bucketCount);
            int xni = xi + ii;
            int yni = yi + jj;
            int zni = zi + kk;
            if (inbounds(xni, 0, bucketCount) && inbounds(yni, 0, bucketCount) && inbounds(zni, 0, bucketCount)) {
                T bucketIndex = xni + yni * bucketCount + zni * bucketCount * bucketCount;
                int bucketPosition = binarySearchInt<T>(bucketIndexes, bucketIndex, 0, pointBucketCount - 1);
                if (bucketPosition >= 0 && !invalidIndexes[realIdxQuery]) {
                    int startPointIndex = pointBucketIndexes[bucketPosition];
                    int endPointIndex = bucketPosition < pointBucketCount - 1 ? pointBucketIndexes[bucketPosition + 1]
                                                                              : pointsReference;
                    int numbPnts = endPointIndex - startPointIndex;
                    if (numbPnts < MAX_PNTS_IN_CUBE) {
                        for (int idx = startPointIndex; idx < endPointIndex; idx++) {
                            int realIdxNeighbor = indexes[idx];

                            float dist = EucledianDistance(realIdxQuery, realIdxNeighbor, pntsq_x, pntsq_y, pntsq_z,
                                                           pnts_x, pnts_y, pnts_z);

                            if (dist < thres)
                                appendDistanceGrid(knnDistances, knn, dist, realIdxNeighbor,
                                                   reverseQueryIndexes[realIdxQuery],
                                                   k, counter,
                                                   advance);

                        }
                    } else invalidIndexes[realIdxQuery] = 1;
                }
                counters[i] = counter;
            }
            else{
                if (offset == 0 && invalidateOutOfCube){
                    invalidIndexes[realIdxQuery] = 1;
                }
            }
            bool lastCube = (ii == offset) && (jj == offset) && (kk == offset) && (!invalidIndexes[realIdxQuery]);
            if (lastCube) {
                if (counter < k) {
                    float x = pntsq_x[realIdxQuery];
                    float y = pntsq_y[realIdxQuery];
                    float z = pntsq_z[realIdxQuery];
                    bool searchMore = false;
                    if (xi - offset >= 0 && sqrt(thres) >= x - xmin - (xi - offset) * h)
                        searchMore = true;
                    else if (xi + offset + 1 <= bucketCount &&
                             sqrt(thres) >= xmin - x + (xi + offset + 1) * h)
                        searchMore = true;
                    else if (yi - offset >= 0 && sqrt(thres) >= y - xmin - (yi - offset) * h)
                        searchMore = true;
                    else if (yi + offset + 1 <= bucketCount &&
                             sqrt(thres) >= xmin - y + (yi + offset + 1) * h)
                        searchMore = true;
                    else if (zi - offset >= 0 && sqrt(thres) >= z - xmin - (zi - offset) * h)
                        searchMore = true;
                    else if (zi + offset + 1 <= bucketCount &&
                             sqrt(thres) >= xmin - z + (zi + offset + 1) * h)
                        searchMore = true;
                    continues[i] = searchMore ? 1 : 0;
                } else {
                    float x = pntsq_x[realIdxQuery];
                    float y = pntsq_y[realIdxQuery];
                    float z = pntsq_z[realIdxQuery];
                    realIdxQuery -= advance;
                    float maximumRadius = knnDistances[realIdxQuery * k + k - 1];
                    bool searchMore = false;
                    if (xi - offset >= 0 && sqrt(maximumRadius) >= x - xmin - (xi - offset) * h)
                        searchMore = true;
                    if (xi + offset + 1 <= bucketCount &&
                        sqrt(maximumRadius) >= xmin - x + (xi + offset + 1) * h)
                        searchMore = true;
                    if (yi - offset >= 0 && sqrt(maximumRadius) >= y - xmin - (yi - offset) * h)
                        searchMore = true;
                    if (yi + offset + 1 <= bucketCount &&
                        sqrt(maximumRadius) >= xmin - y + (yi + offset + 1) * h)
                        searchMore = true;
                    if (zi - offset >= 0 && sqrt(maximumRadius) >= z - xmin - (zi - offset) * h)
                        searchMore = true;
                    if (zi + offset + 1 <= bucketCount &&
                        sqrt(maximumRadius) >= xmin - z + (zi + offset + 1) * h)
                        searchMore = true;
                    continues[i] = searchMore ? 1 : 0;
                }
            }
        }
    }

    template<typename T>
    __global__
    void
    construct_neighbors_self(float *pnts_x, float *pnts_y, float *pnts_z,
                        int *pointBucketIndexes, int pointBucketCount,
                        T *bucketIndexes, int pointsReference, float *knnDistances, int *knn, int bucketCount, int k,
                        int offset, int *counters, int* indexesQuery, int *indexes,
                        int *continues, int ii, int jj, int kk,
                        int pointsLeftCount, float xmin, float xmax, float h, float thres,
                        int *reverseIndexesQuery,
                        int advance, int *invalidIndexes, int max_points_in_bucket = 1e5, bool invalidateOutOfCube = false) {
        int MAX_PNTS_IN_CUBE = max_points_in_bucket;
        unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i < pointsLeftCount) {
            int realIdxQuery = indexesQuery[i];
            int counter = counters[i];
            int xi = (int) ((pnts_x[realIdxQuery] - xmin) / (xmax - xmin) * bucketCount);
            int yi = (int) ((pnts_y[realIdxQuery] - xmin) / (xmax - xmin) * bucketCount);
            int zi = (int) ((pnts_z[realIdxQuery] - xmin) / (xmax - xmin) * bucketCount);
            int xni = xi + ii;
            int yni = yi + jj;
            int zni = zi + kk;
            if (inbounds(xni, 0, bucketCount) && inbounds(yni, 0, bucketCount) && inbounds(zni, 0, bucketCount)) {
                T bucketIndex = xni + yni * bucketCount + zni * bucketCount * bucketCount;
                int bucketPosition = binarySearchInt<T>(bucketIndexes, bucketIndex, 0, pointBucketCount - 1);
                if (bucketPosition >= 0 && !invalidIndexes[realIdxQuery]) {
                    int startPointIndex = pointBucketIndexes[bucketPosition];
                    int endPointIndex = bucketPosition < pointBucketCount - 1 ? pointBucketIndexes[bucketPosition + 1]
                                                                              : pointsReference;
                    int numbPnts = endPointIndex - startPointIndex;
                    if (numbPnts < MAX_PNTS_IN_CUBE) {
                        for (int idx = startPointIndex; idx < endPointIndex; idx++) {
                            int realIdxNeighbor = indexes[idx];
                            if (realIdxNeighbor == realIdxQuery) continue;
                            float dist = EucledianDistance(realIdxQuery, realIdxNeighbor, pnts_x, pnts_y, pnts_z);

                            if (dist < thres)
                                appendDistanceGrid(knnDistances, knn, dist, realIdxNeighbor,
                                                   reverseIndexesQuery[realIdxQuery],
                                                   k, counter,
                                                   advance);

                        }
                    } else invalidIndexes[realIdxQuery] = 1;
                }
                counters[i] = counter;
            }
            else{
                if (offset == 0 && invalidateOutOfCube){
                    invalidIndexes[realIdxQuery] = 1;
                }
            }
            bool lastCube = (ii == offset) && (jj == offset) && (kk == offset) && (!invalidIndexes[realIdxQuery]);
            if (lastCube) {
                if (counter < k) {
                    float x = pnts_x[realIdxQuery];
                    float y = pnts_y[realIdxQuery];
                    float z = pnts_z[realIdxQuery];
                    bool searchMore = false;
                    if (xi - offset >= 0 && sqrt(thres) >= x - xmin - (xi - offset) * h)
                        searchMore = true;
                    else if (xi + offset + 1 <= bucketCount &&
                             sqrt(thres) >= xmin - x + (xi + offset + 1) * h)
                        searchMore = true;
                    else if (yi - offset >= 0 && sqrt(thres) >= y - xmin - (yi - offset) * h)
                        searchMore = true;
                    else if (yi + offset + 1 <= bucketCount &&
                             sqrt(thres) >= xmin - y + (yi + offset + 1) * h)
                        searchMore = true;
                    else if (zi - offset >= 0 && sqrt(thres) >= z - xmin - (zi - offset) * h)
                        searchMore = true;
                    else if (zi + offset + 1 <= bucketCount &&
                             sqrt(thres) >= xmin - z + (zi + offset + 1) * h)
                        searchMore = true;
                    continues[i] = searchMore ? 1 : 0;
                } else {
                    float x = pnts_x[realIdxQuery];
                    float y = pnts_y[realIdxQuery];
                    float z = pnts_z[realIdxQuery];
                    realIdxQuery -= advance;
                    float maximumRadius = knnDistances[realIdxQuery * k + k - 1];
                    bool searchMore = false;
                    if (xi - offset >= 0 && sqrt(maximumRadius) >= x - xmin - (xi - offset) * h)
                        searchMore = true;
                    if (xi + offset + 1 <= bucketCount &&
                        sqrt(maximumRadius) >= xmin - x + (xi + offset + 1) * h)
                        searchMore = true;
                    if (yi - offset >= 0 && sqrt(maximumRadius) >= y - xmin - (yi - offset) * h)
                        searchMore = true;
                    if (yi + offset + 1 <= bucketCount &&
                        sqrt(maximumRadius) >= xmin - y + (yi + offset + 1) * h)
                        searchMore = true;
                    if (zi - offset >= 0 && sqrt(maximumRadius) >= z - xmin - (zi - offset) * h)
                        searchMore = true;
                    if (zi + offset + 1 <= bucketCount &&
                        sqrt(maximumRadius) >= xmin - z + (zi + offset + 1) * h)
                        searchMore = true;
                    continues[i] = searchMore ? 1 : 0;
                }
            }
        }
    }

    template<typename T>
    __global__
    void
    construct_neighbors_self_compact(PointCloud referencePC, NeighborHelper<T> helper,
                                     int offset, int ii, int jj, int kk, int pointsLeftCount, int advance, float thres,
                                     int max_points_in_bucket = 1e5, bool invalidateOutOfCube = false) {
        int MAX_PNTS_IN_CUBE = max_points_in_bucket;
        unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i < pointsLeftCount) {
            int realIdxQuery = helper.indexesQuery[i];
            int counter = helper.counters[i];
            int xi = (int) ((referencePC.pnts_x[realIdxQuery] - helper.xmin) / (helper.xmax - helper.xmin) * helper.bucketCount);
            int yi = (int) ((referencePC.pnts_y[realIdxQuery] - helper.xmin) / (helper.xmax - helper.xmin) * helper.bucketCount);
            int zi = (int) ((referencePC.pnts_z[realIdxQuery] - helper.xmin) / (helper.xmax - helper.xmin) * helper.bucketCount);
            int xni = xi + ii;
            int yni = yi + jj;
            int zni = zi + kk;
            if (inbounds(xni, 0, helper.bucketCount) && inbounds(yni, 0, helper.bucketCount) && inbounds(zni, 0, helper.bucketCount)) {
                T bucketIndex = xni + yni * helper.bucketCount + zni * helper.bucketCount * helper.bucketCount;
                int bucketPosition = binarySearchInt<T>(helper.bucketIndexes, bucketIndex, 0, helper.pointBucketCount - 1);
                if (bucketPosition >= 0 && !helper.invalidIndexes[realIdxQuery]) {
                    int startPointIndex = helper.pointBucketIndexes[bucketPosition];
                    int endPointIndex = bucketPosition < helper.pointBucketCount - 1 ? helper.pointBucketIndexes[bucketPosition + 1]
                                                                              : helper.pointsReference;
                    int numbPnts = endPointIndex - startPointIndex;
                    if (numbPnts < MAX_PNTS_IN_CUBE) {
                        for (int idx = startPointIndex; idx < endPointIndex; idx++) {
                            int realIdxNeighbor = helper.indexes[idx];
                            if (realIdxNeighbor == realIdxQuery) continue;
                            float dist = EucledianDistance(realIdxQuery, realIdxNeighbor, referencePC.pnts_x, referencePC.pnts_y, referencePC.pnts_z);

                            if (dist <= thres)
                                appendDistanceGrid(helper.knnDistances, helper.knn, dist, realIdxNeighbor,
                                                   helper.reverseQueryIndexes[realIdxQuery],
                                                   helper.k, counter,
                                                   advance);

                        }
                    } else helper.invalidIndexes[realIdxQuery] = 1;
                }
                helper.counters[i] = counter;
            }
            else{
                if (offset == 0 && invalidateOutOfCube){
                    helper.invalidIndexes[realIdxQuery] = 1;
                }
            }
            bool lastCube = (ii == offset) && (jj == offset) && (kk == offset) && (!helper.invalidIndexes[realIdxQuery]);
            if (lastCube) {
                if (counter < helper.k) {
                    float x = referencePC.pnts_x[realIdxQuery];
                    float y = referencePC.pnts_y[realIdxQuery];
                    float z = referencePC.pnts_z[realIdxQuery];
                    bool searchMore = false;
                    if (xi - offset >= 0 && sqrt(thres) > x - helper.xmin - (xi - offset) * helper.h)
                        searchMore = true;
                    else if (xi + offset + 1 <= helper.bucketCount &&
                             sqrt(thres) > helper.xmin - x + (xi + offset + 1) * helper.h)
                        searchMore = true;
                    else if (yi - offset >= 0 && sqrt(thres) > y - helper.xmin - (yi - offset) * helper.h)
                        searchMore = true;
                    else if (yi + offset + 1 <= helper.bucketCount &&
                             sqrt(thres) > helper.xmin - y + (yi + offset + 1) * helper.h)
                        searchMore = true;
                    else if (zi - offset >= 0 && sqrt(thres) > z - helper.xmin - (zi - offset) * helper.h)
                        searchMore = true;
                    else if (zi + offset + 1 <= helper.bucketCount &&
                             sqrt(thres) > helper.xmin - z + (zi + offset + 1) * helper.h)
                        searchMore = true;
                    helper.continues[i] = searchMore ? 1 : 0;
                } else {
                    float x = referencePC.pnts_x[realIdxQuery];
                    float y = referencePC.pnts_y[realIdxQuery];
                    float z = referencePC.pnts_z[realIdxQuery];
                    realIdxQuery -= advance;
                    float maximumRadius = helper.knnDistances[realIdxQuery * helper.k + helper.k - 1];
                    bool searchMore = false;
                    if (xi - offset >= 0 && sqrt(maximumRadius) >= x - helper.xmin - (xi - offset) * helper.h)
                        searchMore = true;
                    if (xi + offset + 1 <= helper.bucketCount &&
                        sqrt(maximumRadius) >= helper.xmin - x + (xi + offset + 1) * helper.h)
                        searchMore = true;
                    if (yi - offset >= 0 && sqrt(maximumRadius) >= y - helper.xmin - (yi - offset) * helper.h)
                        searchMore = true;
                    if (yi + offset + 1 <= helper.bucketCount &&
                        sqrt(maximumRadius) >= helper.xmin - y + (yi + offset + 1) * helper.h)
                        searchMore = true;
                    if (zi - offset >= 0 && sqrt(maximumRadius) >= z - helper.xmin - (zi - offset) * helper.h)
                        searchMore = true;
                    if (zi + offset + 1 <= helper.bucketCount &&
                        sqrt(maximumRadius) >= helper.xmin - z + (zi + offset + 1) * helper.h)
                        searchMore = true;
                    helper.continues[i] = searchMore ? 1 : 0;
                }
            }
        }
    }
}

namespace {
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
    void nomralize_distances_grid(float *knnDistances, float max, float min, int counter) {
        unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i < counter) {
            knnDistances[i] = (knnDistances[i] - min) / (max - min);
        }
    }

    __global__
    void addMaximum_grid(float *knnDistances, float max, int pointsCount, int k) {
        unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i < pointsCount) {
            float addon = i * max;
            for (int kk = 0; kk < k; kk++) {
                knnDistances[i * k + kk] += addon;
            }
        }
    }


    /*template<typename T>
    int getBuckets(float*& dpoints_x, float*& dpoints_y, float*& dpoints_z, int pointsCount,
                   T *dbucketIndexes, int bucketsCount,
                   int *dindexes, int *&dpointBucketIndexes, float min, float max, int &maxBucketSize, bool sort_points = false) {
        int threads = 1024;
        int blocks = (int) ceil((1.0 * pointsCount) / threads);
        get_buckets_no_binary<T><<<blocks, threads>>>(dpoints_x, dpoints_y, dpoints_z, dbucketIndexes,
                                                      pointsCount, bucketsCount, min, max);
        cudaDeviceSynchronize();

        int *indexes = new int[pointsCount];
        for (int i = 0; i < pointsCount; i++) indexes[i] = i;
        cudaMemcpy(dindexes, indexes, pointsCount * sizeof(int), cudaMemcpyHostToDevice);
        delete[] indexes;

        cudaMalloc((void **) &dpointBucketIndexes, pointsCount * sizeof(int));
        cudaMemcpy(dpointBucketIndexes, dindexes, pointsCount * sizeof(int), cudaMemcpyDeviceToDevice);

        thrust::sort_by_key(thrust::device, dbucketIndexes, dbucketIndexes + pointsCount, dindexes,
                            thrust::less<T>());

        thrust::pair<T *, int *> end;
        end = thrust::unique_by_key(thrust::device, dbucketIndexes, dbucketIndexes + pointsCount,
                                    dpointBucketIndexes, thrust::equal_to<T>());
        T vertexPartitionSize = end.first - dbucketIndexes;
        //std::cout << "vertex partition size:" << vertexPartitionSize << std::endl;

        int *dpointBucketsCount;
        cudaMalloc((void **) &dpointBucketsCount, vertexPartitionSize * sizeof(int));
        threads = 1024;
        blocks = (int) ceil((1.0 * vertexPartitionSize) / threads);
        getBucketSizes<<<blocks, threads>>>(dpointBucketsCount, dpointBucketIndexes, vertexPartitionSize, pointsCount);
        cudaDeviceSynchronize();
        maxBucketSize = thrust::reduce(thrust::device, dpointBucketsCount,
                                       dpointBucketsCount + vertexPartitionSize, -1e8f,
                                       thrust::maximum<int>());
        cudaFree(dpointBucketsCount);
        //std::cout << "bucket size:" << maxBucketSize << std::endl;
        return vertexPartitionSize;
    }*/

    __global__
    void findQueryIndexes(int *indexes, int *queryIndexes, int totalPointCount, int referencePointCount) {
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i < totalPointCount) {
            int idx = indexes[i];
            queryIndexes[i] = idx < referencePointCount ? 0 : 1;
        }
    }

    __global__
    void packQueryIndexes(int *queryIndexesPacked, int *queryIndexesCounter, int *queryIndexes, int *indexes,
                          int totalPointCount) {
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i < totalPointCount) {
            if (queryIndexes[i]) {
                int idx = queryIndexesCounter[i];
                queryIndexesPacked[idx] = indexes[i];
            }
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
    void denormalize_points(float *x, float *y, float *z, float min, float max, int pointCount) {
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i < pointCount) {
            float d = max - min;
            x[i] = x[i] * d + min;
            y[i] = y[i] * d + min;
            z[i] = z[i] * d + min;
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

    __global__
    void rearrangeNeighbors(int *knnDest, int *knnSource,
                            /*float* knnDistDest, float* knnDistSource,*/
                            int* indexesReference, int* indexesQuery, int k, int pointsCount) {
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i < pointsCount) {
            int idxQuery = indexesQuery[i];
            for (int j = 0; j < k; j++){
                int knnIdx  = indexesReference[knnSource[i * k + j]];
                knnDest[idxQuery * k + j] = knnIdx;
                //knnDistDest[idxQuery * k + j] = knnDistSource[i * k + j];
            }
        }
    }

    __global__
    void validNeighbors(int *invalidIndexes, int *knn, int pointsCount) {
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i < pointsCount) {
            if (invalidIndexes[i]) knn[i] = -1;
        }
    }
}

template<typename T>
void GridStructure::getBuckets(int dev_id, int level, int bucketsCount, float min, float max, int &maxBucketSize) {
    int threads = 1024;
    int blocks = (int) ceil((1.0 * pointsCount) / threads);
    get_buckets_no_binary<T><<<blocks, threads>>>(dpoints_x[dev_id], dpoints_y[dev_id], dpoints_z[dev_id],
                                                  hierachyTree[level]->dbucketIndexes[dev_id],
                                                  pointsCount, bucketsCount, min, max);
    cudaDeviceSynchronize();

    int *indexes = new int[pointsCount];
    for (int i = 0; i < pointsCount; i++) indexes[i] = i;
    cudaMemcpy(hierachyTree[level]->dindexes[dev_id], indexes, pointsCount * sizeof(int), cudaMemcpyHostToDevice);
    delete[] indexes;

    cudaMalloc((void **) &hierachyTree[level]->dpointBucketIndexes[dev_id], pointsCount * sizeof(int));
    cudaMemcpy(hierachyTree[level]->dpointBucketIndexes[dev_id], hierachyTree[level]->dindexes[dev_id], pointsCount * sizeof(int), cudaMemcpyDeviceToDevice);

    thrust::sort_by_key(thrust::device, hierachyTree[level]->dbucketIndexes[dev_id], hierachyTree[level]->dbucketIndexes[dev_id] + pointsCount,
                        hierachyTree[level]->dindexes[dev_id], thrust::less<T>());

    thrust::pair<T *, int *> end;
    end = thrust::unique_by_key(thrust::device, hierachyTree[level]->dbucketIndexes[dev_id], hierachyTree[level]->dbucketIndexes[dev_id] + pointsCount,
                                hierachyTree[level]->dpointBucketIndexes[dev_id], thrust::equal_to<T>());
    T vertexPartitionSize = end.first - hierachyTree[level]->dbucketIndexes[dev_id];
    //std::cout << "vertex partition size:" << vertexPartitionSize << std::endl;

    int *dpointBucketsCount;
    cudaMalloc((void **) &dpointBucketsCount, vertexPartitionSize * sizeof(int));
    threads = 1024;
    blocks = (int) ceil((1.0 * vertexPartitionSize) / threads);
    getBucketSizes<<<blocks, threads>>>(dpointBucketsCount, hierachyTree[level]->dpointBucketIndexes[dev_id], vertexPartitionSize, pointsCount);
    cudaDeviceSynchronize();
    maxBucketSize = thrust::reduce(thrust::device, dpointBucketsCount,
                                   dpointBucketsCount + vertexPartitionSize, -1e8f,
                                   thrust::maximum<int>());
    cudaFree(dpointBucketsCount);
    //std::cout << "bucket size:" << maxBucketSize << std::endl;
    hierachyTree[level]->vertexPartitionSize[dev_id] = vertexPartitionSize;
}

bool GridStructure::InitializeLevel(int dev_id, int level, int bucketsCount, float min, float max){
    if (hierachyTree[level]->initialized[dev_id]) return false;
    cudaMalloc((void **)& hierachyTree[level]->dbucketIndexes[dev_id], pointsCount * sizeof(long64));
    cudaMalloc((void **)& hierachyTree[level]->dindexes[dev_id], pointsCount * sizeof(int));
    int dummy;
    getBuckets<long64>(dev_id, level, bucketsCount, min, max, dummy);
    hierachyTree[level]->initialized[dev_id] = true;
    return true;
}

template <typename T>
void GridStructure::UpdateHelperFromInitialize(NeighborHelper<T>& neighborHelper, int level, int dev_id, int bucketCount){
    neighborHelper.pointBucketCount = hierachyTree[level]->vertexPartitionSize[dev_id];
    neighborHelper.bucketIndexes = hierachyTree[level]->dbucketIndexes[dev_id];
    neighborHelper.bucketCount = bucketCount;
    neighborHelper.indexes = hierachyTree[level]->dindexes[dev_id];
    neighborHelper.pointBucketIndexes = hierachyTree[level]->dpointBucketIndexes[dev_id];
}

void GridStructure::Hierarchy::RefreshHierarchy(){
    for (int dev_id = 0; dev_id < num_cards; dev_id++) {
        cudaSetDevice(dev_id);
        cudaFree(dbucketIndexes[dev_id]);
        dbucketIndexes[dev_id] = nullptr;
        cudaFree(dindexes[dev_id]);
        dindexes[dev_id] = nullptr;
        cudaFree(dpointBucketIndexes[dev_id]);
        dpointBucketIndexes[dev_id] = nullptr;
        vertexPartitionSize[dev_id] = 0;
        initialized[dev_id] = false;
    }
}

GridStructure::Hierarchy::~Hierarchy(){
    for (int dev_id = 0; dev_id < num_cards; dev_id++){
        cudaSetDevice(dev_id);
        cudaFree(dbucketIndexes[dev_id]);
        dbucketIndexes[dev_id] = nullptr;
        cudaFree(dindexes[dev_id]);
        dindexes[dev_id] = nullptr;
        cudaFree(dpointBucketIndexes[dev_id]);
        dpointBucketIndexes[dev_id] = nullptr;
    }
}

void GridStructure::RefreshGridKNNStructure(){
    for (int dev_id = 0; dev_id < num_cards; dev_id++){
        cudaSetDevice(dev_id);
        cudaFree(dQuery_x[dev_id]);
        dQuery_x[dev_id] = nullptr;
        cudaFree(dQuery_y[dev_id]);
        dQuery_y[dev_id] = nullptr;
        cudaFree(dQuery_z[dev_id]);
        dQuery_z[dev_id] = nullptr;
        cudaFree(dQueryIndexesInitial[dev_id]);
        dQueryIndexesInitial[dev_id] = nullptr;
        cudaFree(dknn[dev_id]);
        dknn[dev_id] = nullptr;
    }
}

void GridStructure::RefreshHierarchies(){
    for (int i = 0; i < levels; i++) {
        hierachyTree[i]->RefreshHierarchy();
    }
}

GridStructure::~GridStructure(){
    for (int dev_id = 0; dev_id < num_cards; dev_id++){
        cudaSetDevice(dev_id);
        cudaFree(dpoints_x[dev_id]);
        cudaFree(dpoints_y[dev_id]);
        cudaFree(dpoints_z[dev_id]);
        cudaFree(dQuery_x[dev_id]);
        cudaFree(dQuery_y[dev_id]);
        cudaFree(dQuery_z[dev_id]);
        cudaFree(doriginal_indexes[dev_id]);
        cudaFree(dQueryIndexesInitial[dev_id]);
        cudaFree(dknn[dev_id]);
    }
    for (int i = 0; i < levels; i++) {
        delete hierachyTree[i];
    }
}

void GridStructure::InitStructure(float *points_x, float *points_y, float *points_z) {
    std::vector<int> bucketCount(num_cards);
#pragma omp parallel for num_threads(num_cards)
    for (int dev_id = 0; dev_id < num_cards; dev_id++) {
        bucketCount[dev_id] = finalBucketCount;
        cudaSetDevice(dev_id);
        int level = 0;
        cudaMalloc((void **) &dpoints_x[dev_id], pointsCount * sizeof(float));
        cudaMemcpy(dpoints_x[dev_id], points_x, pointsCount * sizeof(float),
                   cudaMemcpyHostToDevice);

        cudaMalloc((void **) &dpoints_y[dev_id], pointsCount * sizeof(float));
        cudaMemcpy(dpoints_y[dev_id], points_y, pointsCount * sizeof(float),
                   cudaMemcpyHostToDevice);

        cudaMalloc((void **) &dpoints_z[dev_id], pointsCount * sizeof(float));
        cudaMemcpy(dpoints_z[dev_id], points_z, pointsCount * sizeof(float),
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
        //ALEX HERE
        //initialized[dev_id] = true;
        ///
        //allow the query points to be a little further...
        //this is very useful and avoids parts that are completely away
        //from the reference points. Since we are talking about ICP
        //the later should be avoided.

        cudaMalloc((void **) &hierachyTree[level]->dbucketIndexes[dev_id],
                   pointsCount * sizeof(long64));
        cudaMalloc((void **) &hierachyTree[level]->dindexes[dev_id], pointsCount * sizeof(int));

        min = -eps;
        max = 1.0 + eps;
        int maxPntBucketSize = 0;
        hierachyTree[level]->dpointBucketIndexes[dev_id] = NULL;
        bucketCount[dev_id] *= 2;
        //while (maxPntBucketSize < 50) {
        while (maxPntBucketSize <= 0) {
            cudaFree(hierachyTree[level]->dpointBucketIndexes[dev_id]);
            hierachyTree[level]->dpointBucketIndexes[dev_id] = NULL;
            bucketCount[dev_id] /= 2;
            getBuckets<long64>(dev_id, level, bucketCount[dev_id], min, max, maxPntBucketSize);
        }
        hierachyTree[level]->initialized[dev_id] = true;
        if (!dev_id && !level){
            max_bucket_elements = maxPntBucketSize;
        }

        float *temp_pnts_x;
        cudaMalloc((void **) &temp_pnts_x, pointsCount * sizeof(float));
        float *temp_pnts_y;
        cudaMalloc((void **) &temp_pnts_y, pointsCount * sizeof(float));
        float *temp_pnts_z;
        cudaMalloc((void **) &temp_pnts_z, pointsCount * sizeof(float));

        thrust::gather(thrust::device, hierachyTree[level]->dindexes[dev_id], hierachyTree[level]->dindexes[dev_id] + pointsCount,
                           dpoints_x[dev_id], temp_pnts_x);
        thrust::gather(thrust::device, hierachyTree[level]->dindexes[dev_id], hierachyTree[level]->dindexes[dev_id] + pointsCount,
                       dpoints_y[dev_id], temp_pnts_y);
        thrust::gather(thrust::device, hierachyTree[level]->dindexes[dev_id], hierachyTree[level]->dindexes[dev_id] + pointsCount,
                       dpoints_z[dev_id], temp_pnts_z);
        std::swap(temp_pnts_x, dpoints_x[dev_id]);
        std::swap(temp_pnts_y, dpoints_y[dev_id]);
        std::swap(temp_pnts_z, dpoints_z[dev_id]);
        cudaFree(temp_pnts_x);
        cudaFree(temp_pnts_y);
        cudaFree(temp_pnts_z);
        cudaMalloc((void**)& doriginal_indexes[dev_id], pointsCount * sizeof(int));
        thrust::sequence(thrust::device, doriginal_indexes[dev_id], doriginal_indexes[dev_id] + pointsCount, 0);
        std::swap(doriginal_indexes[dev_id], hierachyTree[level]->dindexes[dev_id]);

    }
    finalBucketCount = bucketCount[0];
    //std::cout << "Finished Grid Construction!" << std::endl;
}

void GridStructure::recomputeBuffers(int &pointsLeft, int count, int *&dcontinues,
                                     int *&dIndexesQuery, int *&dcounters) {
    int *dcontinueScan;
    cudaMalloc((void **) &dcontinueScan, pointsLeft * sizeof(int));
    thrust::inclusive_scan(thrust::device, dcontinues, dcontinues + pointsLeft,
                           dcontinueScan);
    int *dCountersNew = NULL, *dIndexesNew = NULL;

    cudaMalloc((void **) &dCountersNew, count * sizeof(int));
    cudaMalloc((void **) &dIndexesNew, count * sizeof(int));

    int threads = 1024;
    int blocks = ceil((1.0 * pointsLeft) / threads);
    newNeighborSearch::getNewData<<<blocks, threads>>>(dcontinueScan, dcontinues,
                                    dIndexesQuery, dIndexesNew,
                                    dcounters, dCountersNew, pointsLeft);
    cudaDeviceSynchronize();
    cudaFree(dcontinueScan);

    std::swap(dIndexesQuery, dIndexesNew);
    std::swap(dcounters, dCountersNew);

    cudaFree(dIndexesNew);
    cudaFree(dCountersNew);

    cudaFree(dcontinues);
    pointsLeft = count;
    cudaMalloc((void **) &dcontinues, pointsLeft * sizeof(int));
}

GridStructure::GridStructure(const Mesh& mesh, int num_cards, float eps,
                             int bucketCount, int maxDivision):
        dpoints_x(num_cards, NULL),
        dpoints_y(num_cards, NULL),
        dpoints_z(num_cards, NULL),
        dQuery_x(num_cards, NULL),
        dQuery_y(num_cards, NULL),
        dQuery_z(num_cards, NULL),
        dknn(num_cards, NULL),
        doriginal_indexes(num_cards, NULL),
        dQueryIndexesInitial(num_cards, NULL),
        num_cards(num_cards),
        pointsCard(num_cards),
        pointsCount(mesh.nV()),
        //ALEX HERE
        //initialized(num_cards, false),
        //
        finalBucketCount(bucketCount),
        eps(eps){
    levels = 0;
    hierachyTree.push_back(new Hierarchy(num_cards));
    std::vector<float> points_x(pointsCount);
    std::vector<float> points_y(pointsCount);
    std::vector<float> points_z(pointsCount);
#pragma omp parallel for
    for (int i = 0; i < pointsCount; i++){
        points_x[i] = mesh.mV[i].x;
        points_y[i] = mesh.mV[i].y;
        points_z[i] = mesh.mV[i].z;
    }
    InitStructure(points_x.data(), points_y.data(), points_z.data());
    int temp = finalBucketCount;
    finalBucketCount /= 2;
    levels++;
    while (finalBucketCount >= maxDivision) {
        hierachyTree.push_back(new Hierarchy(num_cards));
        finalBucketCount /= 2;
        levels++;
    }
    finalBucketCount = temp;
}

//device Pointers
void GridStructure::rearrangeQueryPoints(float*& pntsQx, float*& pntsQy, float*& pntsQz, int queryPointCount,
                                         int* queryIndexes){
    float minx = thrust::reduce(thrust::device, pntsQx,
                                pntsQx + queryPointCount, 1e8f,
                                thrust::minimum<float>()) - 1e-6f;
    float maxx = thrust::reduce(thrust::device, pntsQx,
                                pntsQx + queryPointCount, -1e8f,
                                thrust::maximum<float>()) + 1e-6f;
    float miny = thrust::reduce(thrust::device, pntsQy,
                                pntsQy + queryPointCount, 1e8f,
                                thrust::minimum<float>()) - 1e-6f;
    float maxy = thrust::reduce(thrust::device, pntsQy,
                                pntsQy + queryPointCount, -1e8f,
                                thrust::maximum<float>()) + 1e-6f;
    float minz = thrust::reduce(thrust::device, pntsQz,
                                pntsQz + queryPointCount, 1e8f,
                                thrust::minimum<float>()) - 1e-6f;
    float maxz = thrust::reduce(thrust::device, pntsQz,
                                pntsQz + queryPointCount, -1e8f,
                                thrust::maximum<float>()) + 1e-6f;

    float min = std::min(minx, miny);
    min = std::min(min, minz);

    float max = std::max(maxx, maxy);
    max = std::max(max, maxz);

    long64* dbucketIndexes;
    cudaMalloc((void**)&dbucketIndexes, queryPointCount * sizeof(long64));
    int threads = 1024;
    int blocks = (int)ceil((1.0 * queryPointCount) / threads);
    get_buckets_no_binary<long64><<<blocks, threads>>>(pntsQx, pntsQx, pntsQy, dbucketIndexes,
                                                  queryPointCount, 1024, min, max);
    cudaDeviceSynchronize();

    thrust::sequence(thrust::device, queryIndexes, queryIndexes + queryPointCount, 0);

    thrust::sort_by_key(thrust::device, dbucketIndexes, dbucketIndexes + queryPointCount, queryIndexes,
                        thrust::less<long64>());
    cudaFree(dbucketIndexes);
    float* temp_pnts_x;
    cudaMalloc((void**)&temp_pnts_x, queryPointCount * sizeof(float));
    float* temp_pnts_y;
    cudaMalloc((void**)&temp_pnts_y, queryPointCount * sizeof(float));
    float* temp_pnts_z;
    cudaMalloc((void**)&temp_pnts_z, queryPointCount * sizeof(float));

    thrust::gather(thrust::device, queryIndexes, queryIndexes + queryPointCount,
                   pntsQx,temp_pnts_x);
    thrust::gather(thrust::device, queryIndexes, queryIndexes + queryPointCount,
                   pntsQy,temp_pnts_y);
    thrust::gather(thrust::device, queryIndexes, queryIndexes + queryPointCount,
                   pntsQz,temp_pnts_z);
    std::swap(temp_pnts_x, pntsQx);
    std::swap(temp_pnts_y, pntsQy);
    std::swap(temp_pnts_z, pntsQz);
    cudaFree(temp_pnts_x);
    cudaFree(temp_pnts_y);
    cudaFree(temp_pnts_z);
}

void GridStructure::GRIDCUDAKNN(float* pointsQ_x, float* pointsQ_y, float* pointsQ_z, int pointsCountQuery,
                 int k, int* knn, float thres) {

    thres /= (this->maxExtent -  this->minExtent);
    thres *= thres;
    int pointsCount =  this->pointsCount;
    int num_cards = this->num_cards;

    clock_t starttime =  clock();
    std::vector<int*> dcontinues(num_cards, NULL);
    std::vector<int*> dIndexesQuery(num_cards, NULL);
    std::vector<int*> dIndexesQueryAll(num_cards, NULL);
    std::vector<int*> dIndexesQueryReverse(num_cards, NULL);
    std::vector<int*> dcounters(num_cards, NULL);
    std::vector<float*> dknndistances(num_cards, NULL);
    std::vector<int*> dknn(num_cards, NULL);
    std::vector<float*> dQuery_x(num_cards, NULL);
    std::vector<float*> dQuery_y(num_cards, NULL);
    std::vector<float*> dQuery_z(num_cards, NULL);
    std::vector<int*> dvalidIndexes(num_cards, NULL);
    std::vector<int*> dQueryIndexesInitial(num_cards, NULL);
    std::vector<int> pointsLeft_a(num_cards);
    // Here we will have
    int bucketsCount =  this->finalBucketCount;
    std::vector<int> extent;
    int pointPartitionSize = pointsCountQuery / num_cards;
    std::vector<int> pointsCard(num_cards);
    for (int i = 0; i < num_cards - 1; i++){
        pointsCard[i] = pointPartitionSize;
    }
    pointsCard[num_cards - 1] = pointsCountQuery - (num_cards - 1) * pointPartitionSize;
    //int totalPointCount = pointsCount + pointsCountQuery;

#pragma omp parallel for num_threads(num_cards)
    for (int dev_id = 0; dev_id < num_cards; dev_id++) {
        cudaSetDevice(dev_id);
        cudaMalloc((void **)&dvalidIndexes[dev_id], pointsCountQuery * sizeof(int));
        cudaMemset(dvalidIndexes[dev_id], 0, pointsCountQuery * sizeof(int));
        int level = 0;
        float min = -eps;
        float max = 1.0 + eps;
        cudaMalloc((void **) &dIndexesQueryAll[dev_id], pointsCountQuery * sizeof(int));

        cudaMalloc((void **) &dQuery_x[dev_id], pointsCountQuery * sizeof(float));
        cudaMemcpy(dQuery_x[dev_id], pointsQ_x, pointsCountQuery * sizeof(float),
                   cudaMemcpyHostToDevice);

        cudaMalloc((void **) &dQuery_y[dev_id], pointsCountQuery * sizeof(float));
        cudaMemcpy(dQuery_y[dev_id], pointsQ_y, pointsCountQuery * sizeof(float),
                   cudaMemcpyHostToDevice);

        cudaMalloc((void **) &dQuery_z[dev_id], pointsCountQuery * sizeof(float));
        cudaMemcpy(dQuery_z[dev_id], pointsQ_z, pointsCountQuery * sizeof(float),
                   cudaMemcpyHostToDevice);

        int threads = 1024;
        int blocks = (int) ceil((1.0 * pointsCountQuery) / threads);
        normalize_points<<<blocks, threads>>>(dQuery_x[dev_id],
                                              dQuery_y[dev_id],
                                              dQuery_z[dev_id],  this->minExtent,  this->maxExtent, pointsCountQuery);
        cudaDeviceSynchronize();

        cudaMalloc((void**)&dQueryIndexesInitial[dev_id], pointsCountQuery * sizeof(int));
        //thrust::sequence(thrust::device, dQueryIndexesInitial[dev_id], dQueryIndexesInitial[dev_id] + pointsCountQuery, 0);
        rearrangeQueryPoints(dQuery_x[dev_id], dQuery_y[dev_id], dQuery_z[dev_id], pointsCountQuery, dQueryIndexesInitial[dev_id]);

        cudaMalloc((void **) &dIndexesQueryReverse[dev_id], pointsCountQuery * sizeof(int));
        int *indexes = new int[pointsCountQuery];
        for (int i = 0; i < pointsCountQuery; i++) indexes[i] = i;
        cudaMemcpy(dIndexesQueryAll[dev_id], indexes, pointsCountQuery * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dIndexesQueryReverse[dev_id], dIndexesQueryAll[dev_id], pointsCountQuery * sizeof(int), cudaMemcpyDeviceToDevice);
        delete[] indexes;

        cudaMalloc((void **) &dIndexesQuery[dev_id], pointsCard[dev_id] * sizeof(int));
        cudaMemcpy(dIndexesQuery[dev_id], dIndexesQueryAll[dev_id] + dev_id * pointPartitionSize,
                   pointsCard[dev_id] * sizeof(int), cudaMemcpyDeviceToDevice);

        cudaMalloc((void **) &dcounters[dev_id], pointsCard[dev_id] * sizeof(int));
        cudaMemset(dcounters[dev_id], 0, pointsCard[dev_id] * sizeof(int));
        cudaMalloc((void **) &dcontinues[dev_id], pointsCard[dev_id] * sizeof(int));
        cudaMemset(dcontinues[dev_id], 0, pointsCard[dev_id] * sizeof(int));

        //////////////KNEAREST NEIGHBOURS////////////////////////////////////////////////////
        cudaMalloc((void **) &dknndistances[dev_id], pointsCard[dev_id] * k * sizeof(float));
        cudaMalloc((void **) &dknn[dev_id], pointsCard[dev_id] * k * sizeof(int));
        /////////////////////////////////////////////////////////////////////////////////////
        threads = 1024;
        blocks = ceil((1.0 * k * pointsCard[dev_id] / threads));
        setValue<int><<<blocks, threads>>>(dknn[dev_id], -1, k * pointsCard[dev_id]);
        cudaDeviceSynchronize();
        setValue<float><<<blocks, threads>>>(dknndistances[dev_id], 1e10, k * pointsCard[dev_id]);
        cudaDeviceSynchronize();
        int offset = 0;
        threads = 1024;
        blocks = ceil((1.0 * pointsCard[dev_id]) / threads);
        float h = (max - min) / bucketsCount;
        newNeighborSearch::construct_neighbors<long64><<<blocks, threads>>>( this->dpoints_x[dev_id],  this->dpoints_y[dev_id],
                 this->dpoints_z[dev_id], dQuery_x[dev_id], dQuery_y[dev_id], dQuery_z[dev_id],  this->hierachyTree[level]->dpointBucketIndexes[dev_id],
                 this->hierachyTree[level]->vertexPartitionSize[dev_id],
                 this->hierachyTree[level]->dbucketIndexes[dev_id], pointsCount,
                dknndistances[dev_id], dknn[dev_id],
                bucketsCount, k, offset, dcounters[dev_id], dIndexesQuery[dev_id],
                 this->hierachyTree[level]->dindexes[dev_id], dcontinues[dev_id], 0, 0, 0,
                pointsCard[dev_id], min, max, h, pointsCountQuery, thres,
                dIndexesQueryReverse[dev_id], dev_id * pointPartitionSize, dvalidIndexes[dev_id]);
        cudaDeviceSynchronize();
        threads = 1024;
        int pointsLeft = pointsCard[dev_id];
        blocks = ceil((1.0 * pointsLeft) / threads);
        invalidateIndexesNotValid<<<blocks, threads>>>(dIndexesQuery[dev_id], dcontinues[dev_id], dvalidIndexes[dev_id], pointsLeft);
        cudaDeviceSynchronize();

        int count = thrust::reduce(thrust::device, dcontinues[dev_id], dcontinues[dev_id] + pointsLeft, 0);
        if (count > 0)
            recomputeBuffers(pointsLeft, count, dcontinues[dev_id], dIndexesQuery[dev_id], dcounters[dev_id]);
        else
            pointsLeft = count;
        pointsLeft_a[dev_id] = pointsLeft;
        //std::cout << "Do we work?" << std::endl;
    }

    float min = -eps;
    float max = 1.0f + eps;

    int count_global;
    std::vector<int> bucketsCount_a(num_cards);
    std::vector<int> offset_a (num_cards);
    std::vector<int> level_a (num_cards);
    for (int i = 0; i < num_cards; i++) {
        bucketsCount_a[i] =  this->finalBucketCount;
        offset_a[i] = 1;
        level_a[i] = 0;
    }
    std::cout << "Test 4" << std::endl;
#pragma omp parallel for num_threads(num_cards)
    for (int dev_id = 0; dev_id < num_cards; dev_id++) {
        cudaSetDevice(dev_id);
        int &offset = offset_a[dev_id];
        int &pointsLeft = pointsLeft_a[dev_id];
        int &bucketsCount = bucketsCount_a[dev_id];
        int &level = level_a[dev_id];
        float h = (max - min) / bucketsCount;
        int count;
        int iteration = 0;
        do {
            if (offset > 1 && bucketsCount > 8) {
                bucketsCount /= 2;
                level++;
                h = (max - min) / bucketsCount;
                if (InitializeLevel(dev_id, level, bucketsCount, min, max)) {
                }
                offset = 0;
                offset++;
            }
            int threads, blocks;
            threads = 1024;
            blocks = ceil((1.0 * pointsLeft) / threads);
            cudaMemset(dcontinues[dev_id], 0, pointsLeft * sizeof(int));
            int offsets[2] = {-offset, offset};
            for (int ii = -offset + 1; ii <= offset - 1; ii++) {
                for (int jj = -offset + 1; jj <= offset - 1; jj++) {
                    for (int kk = 0; kk < 2; kk++) {
                        newNeighborSearch::construct_neighbors<long64><<<blocks, threads>>>( this->dpoints_x[dev_id],
                                 this->dpoints_y[dev_id],
                                 this->dpoints_z[dev_id], dQuery_x[dev_id], dQuery_y[dev_id], dQuery_z[dev_id],
                                 this->hierachyTree[level]->dpointBucketIndexes[dev_id],
                                 this->hierachyTree[level]->vertexPartitionSize[dev_id],
                                 this->hierachyTree[level]->dbucketIndexes[dev_id],
                                pointsCount, dknndistances[dev_id], dknn[dev_id], bucketsCount,
                                 k, offset, dcounters[dev_id], dIndexesQuery[dev_id],
                                 this->hierachyTree[level]->dindexes[dev_id], dcontinues[dev_id], ii, jj, offsets[kk],
                                pointsLeft, min, max, h,
                                pointsCountQuery, thres, dIndexesQueryReverse[dev_id], dev_id * pointPartitionSize,
                                dvalidIndexes[dev_id]);
                        cudaDeviceSynchronize();

                    }
                }
            }
            for (int ii = -offset + 1; ii <= offset - 1; ii++) {
                for (int jj = 0; jj < 2; jj++) {
                    for (int kk = -offset + 1; kk <= offset - 1; kk++) {
                        newNeighborSearch::construct_neighbors<long64><<<blocks, threads>>>( this->dpoints_x[dev_id],
                                 this->dpoints_y[dev_id],
                                 this->dpoints_z[dev_id], dQuery_x[dev_id], dQuery_y[dev_id], dQuery_z[dev_id],
                                 this->hierachyTree[level]->dpointBucketIndexes[dev_id],
                                 this->hierachyTree[level]->vertexPartitionSize[dev_id],
                                 this->hierachyTree[level]->dbucketIndexes[dev_id],
                                pointsCount, dknndistances[dev_id], dknn[dev_id], bucketsCount,
                                k, offset, dcounters[dev_id], dIndexesQuery[dev_id],
                                 this->hierachyTree[level]->dindexes[dev_id], dcontinues[dev_id], ii, offsets[jj], kk,
                                pointsLeft, min, max, h,
                                pointsCountQuery, thres, dIndexesQueryReverse[dev_id], dev_id * pointPartitionSize,
                                dvalidIndexes[dev_id]);
                        cudaDeviceSynchronize();
                    }
                }
            }
            for (int ii = 0; ii < 2 ; ii++) {
                for (int jj = -offset + 1; jj <= offset - 1; jj++) {
                    for (int kk = -offset + 1; kk <= offset - 1; kk++) {
                        newNeighborSearch::construct_neighbors<long64><<<blocks, threads>>>( this->dpoints_x[dev_id],
                                 this->dpoints_y[dev_id],
                                 this->dpoints_z[dev_id], dQuery_x[dev_id], dQuery_y[dev_id], dQuery_z[dev_id],
                                 this->hierachyTree[level]->dpointBucketIndexes[dev_id],
                                 this->hierachyTree[level]->vertexPartitionSize[dev_id],
                                 this->hierachyTree[level]->dbucketIndexes[dev_id],
                                pointsCount, dknndistances[dev_id], dknn[dev_id], bucketsCount,
                                k, offset, dcounters[dev_id], dIndexesQuery[dev_id],
                                 this->hierachyTree[level]->dindexes[dev_id], dcontinues[dev_id], offsets[ii], jj, kk,
                                pointsLeft, min, max, h,
                                pointsCountQuery, thres, dIndexesQueryReverse[dev_id], dev_id * pointPartitionSize,
                                dvalidIndexes[dev_id]);
                        cudaDeviceSynchronize();
                    }
                }
            }
            for (int ii = 0; ii < 2 ; ii++) {
                for (int jj = 0 ; jj < 2; jj++) {
                    for (int kk = -offset + 1; kk <= offset - 1; kk++) {
                        newNeighborSearch::construct_neighbors<long64><<<blocks, threads>>>( this->dpoints_x[dev_id],
                                 this->dpoints_y[dev_id],
                                 this->dpoints_z[dev_id], dQuery_x[dev_id], dQuery_y[dev_id], dQuery_z[dev_id],
                                 this->hierachyTree[level]->dpointBucketIndexes[dev_id],
                                 this->hierachyTree[level]->vertexPartitionSize[dev_id],
                                 this->hierachyTree[level]->dbucketIndexes[dev_id],
                                pointsCount, dknndistances[dev_id], dknn[dev_id], bucketsCount,
                                k, offset, dcounters[dev_id], dIndexesQuery[dev_id],
                                 this->hierachyTree[level]->dindexes[dev_id], dcontinues[dev_id], offsets[ii], offsets[jj], kk,
                                pointsLeft, min, max, h,
                                pointsCountQuery, thres, dIndexesQueryReverse[dev_id], dev_id * pointPartitionSize,
                                dvalidIndexes[dev_id]);
                        cudaDeviceSynchronize();

                    }
                }
            }
            for (int ii = 0; ii < 2 ; ii++) {
                for (int jj = -offset + 1 ; jj <= offset - 1; jj++) {
                    for (int kk = 0; kk < 2; kk++) {
                        newNeighborSearch::construct_neighbors<long64><<<blocks, threads>>>( this->dpoints_x[dev_id],
                                 this->dpoints_y[dev_id],
                                 this->dpoints_z[dev_id], dQuery_x[dev_id], dQuery_y[dev_id], dQuery_z[dev_id],
                                 this->hierachyTree[level]->dpointBucketIndexes[dev_id],
                                 this->hierachyTree[level]->vertexPartitionSize[dev_id],
                                 this->hierachyTree[level]->dbucketIndexes[dev_id],
                                pointsCount, dknndistances[dev_id], dknn[dev_id], bucketsCount,
                                k, offset, dcounters[dev_id], dIndexesQuery[dev_id],
                                 this->hierachyTree[level]->dindexes[dev_id], dcontinues[dev_id], offsets[ii], jj, offsets[kk],
                                pointsLeft, min, max, h,
                                pointsCountQuery, thres, dIndexesQueryReverse[dev_id], dev_id * pointPartitionSize,
                                dvalidIndexes[dev_id]);
                        cudaDeviceSynchronize();

                    }
                }
            }
            for (int ii = -offset + 1; ii <= offset - 1 ; ii++) {
                for (int jj = 0; jj < 2; jj++) {
                    for (int kk = 0; kk < 2; kk++) {
                        newNeighborSearch::construct_neighbors<long64><<<blocks, threads>>>( this->dpoints_x[dev_id],
                                 this->dpoints_y[dev_id],
                                 this->dpoints_z[dev_id], dQuery_x[dev_id], dQuery_y[dev_id], dQuery_z[dev_id],
                                 this->hierachyTree[level]->dpointBucketIndexes[dev_id],
                                 this->hierachyTree[level]->vertexPartitionSize[dev_id],
                                 this->hierachyTree[level]->dbucketIndexes[dev_id],
                                pointsCount, dknndistances[dev_id], dknn[dev_id], bucketsCount,
                                k, offset, dcounters[dev_id], dIndexesQuery[dev_id],
                                 this->hierachyTree[level]->dindexes[dev_id], dcontinues[dev_id], ii, offsets[jj], offsets[kk],
                                pointsLeft, min, max, h,
                                pointsCountQuery, thres, dIndexesQueryReverse[dev_id], dev_id * pointPartitionSize,
                                dvalidIndexes[dev_id]);
                        cudaDeviceSynchronize();

                    }
                }
            }
            for (int ii = 0; ii < 2 ; ii++) {
                for (int jj = 0; jj < 2; jj++) {
                    for (int kk = 0; kk < 2; kk++) {
                        newNeighborSearch::construct_neighbors<long64><<<blocks, threads>>>( this->dpoints_x[dev_id],
                                 this->dpoints_y[dev_id],
                                 this->dpoints_z[dev_id], dQuery_x[dev_id], dQuery_y[dev_id], dQuery_z[dev_id],
                                 this->hierachyTree[level]->dpointBucketIndexes[dev_id],
                                 this->hierachyTree[level]->vertexPartitionSize[dev_id],
                                 this->hierachyTree[level]->dbucketIndexes[dev_id],
                                pointsCount, dknndistances[dev_id], dknn[dev_id], bucketsCount,
                                 k, offset, dcounters[dev_id], dIndexesQuery[dev_id],
                                 this->hierachyTree[level]->dindexes[dev_id], dcontinues[dev_id], offsets[ii], offsets[jj], offsets[kk],
                                pointsLeft, min, max, h,
                                pointsCountQuery, thres, dIndexesQueryReverse[dev_id], dev_id * pointPartitionSize,
                                dvalidIndexes[dev_id]);
                        cudaDeviceSynchronize();
                    }
                }
            }
            threads = 1024;
            blocks = ceil((1.0 * pointsLeft) / threads);
            invalidateIndexesNotValid<<<blocks, threads>>>(dIndexesQuery[dev_id], dcontinues[dev_id], dvalidIndexes[dev_id], pointsLeft);
            cudaDeviceSynchronize();
            offset++;
            count = thrust::reduce(thrust::device, dcontinues[dev_id], dcontinues[dev_id] + pointsLeft, 0);
#pragma omp critical
            {
                std::cout << "iteration:" << iteration++ << " dev_id:" << dev_id << " counter:" << count << std::endl;
            }
            if (count > 0)
                recomputeBuffers(pointsLeft, count, dcontinues[dev_id], dIndexesQuery[dev_id], dcounters[dev_id]);
            else
                pointsLeft = count;

        } while (count > 0 && bucketsCount > 8);

        if (pointsLeft > 0){
            int threads = 1024;
            int blocks = ceil((1.0 * pointsLeft) / threads);
            newNeighborSearch::invalidateNotFinishedNeighbors<<<blocks, threads>>>(dIndexesQuery[dev_id], dcontinues[dev_id], dvalidIndexes[dev_id],
                                                                pointsLeft);
            cudaDeviceSynchronize();
        }
        int threads = 1024;
        int blocks = ceil((1.0 * pointsCard[dev_id]) / threads);
        newNeighborSearch::negateNotValidKnnIndexes<<<blocks, threads>>>(dknn[dev_id], k, dvalidIndexes[dev_id], pointsCard[dev_id],
                                                                         dev_id * pointPartitionSize);
        cudaDeviceSynchronize();

        cudaFree(dIndexesQuery[dev_id]);
        cudaFree(dcounters[dev_id]);
        cudaFree(dIndexesQueryReverse[dev_id]);
        cudaFree(dQuery_x[dev_id]);
        cudaFree(dQuery_y[dev_id]);
        cudaFree(dQuery_z[dev_id]);
        cudaFree(dcontinues[dev_id]);
        cudaFree(dknndistances[dev_id]);
        cudaFree(dvalidIndexes[dev_id]);
        cudaFree(dIndexesQueryAll[dev_id]);
    }
    cudaSetDevice(0);
    int *dknnGather, *dknnTemp;
    //float* dknnDist;
    cudaMalloc((void**)& dknnGather, k * pointsCountQuery * sizeof(int));
    //cudaMalloc((void**)& dknnDist, k * pointsCountQuery * sizeof(float));
    cudaMalloc((void**)& dknnTemp, k * pointsCountQuery * sizeof(int));
#pragma omp parallel for num_threads(num_cards)
    for (int dev_id = 0; dev_id < num_cards; dev_id++) {
        //cudaSetDevice(dev_id);
        cudaMemcpy(dknnTemp + k * dev_id * pointPartitionSize, dknn[dev_id], k * pointsCard[dev_id] * sizeof(int),
                   cudaMemcpyDeviceToDevice);
    }
    int threads = 1024;
    int blocks = ceil((1.0 * pointsCountQuery) / threads);
    rearrangeNeighbors<<<blocks, threads>>>(dknnGather, dknnTemp, /*dknnDist,dknndistances[0],*/
                                            this->doriginal_indexes[0], dQueryIndexesInitial[0],
                                             k, pointsCountQuery);
    cudaDeviceSynchronize();
    cudaFree(dknnTemp);
    cudaMemcpy(knn, dknnGather, k * pointsCountQuery * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dknnGather);

    for (int dev_id = 0; dev_id < num_cards; dev_id++){
        cudaSetDevice(dev_id);
        cudaFree(dQueryIndexesInitial[dev_id]);
        cudaFree(dknn[dev_id]);
    }

    /*FILE* file = fopen("knnBugFreeQ.bin","wb");
    fwrite(knn, sizeof(int), k * pointsCountQuery, file);
    fclose(file);

    std::vector<float> knnd(k * pointsCountQuery);
    cudaMemcpy(knnd.data(), dknnDist, k * pointsCountQuery * sizeof(float), cudaMemcpyDeviceToHost);
    file = fopen("knnBugFreeQDistancesQ.bin","wb");
    fwrite(knnd.data(), sizeof(float), k * pointsCountQuery, file);
    fclose(file);*/
}

void GridStructure::GRIDCUDAKNNCompact(float* pointsQ_x, float* pointsQ_y, float* pointsQ_z, int pointsCountQuery,
                                       int k, float thres) {
    self = false;
    this->pointsCountQuery = pointsCountQuery;
    thres /= (this->maxExtent -  this->minExtent);
    thres *= thres;

    clock_t starttime =  clock();
    neighborCount = k;
    std::vector<PointCloud> pcReference(num_cards);
    std::vector<PointCloud> pcQuery(num_cards);
    std::vector<int*> dcontinues(num_cards, NULL);
    std::vector<int*> dIndexesQuery(num_cards, NULL);
    std::vector<int*> dIndexesQueryAll(num_cards, NULL);
    std::vector<int*> dIndexesQueryReverse(num_cards, NULL);
    std::vector<int*> dcounters(num_cards, NULL);
    std::vector<float*> dknndistances(num_cards, NULL);
    std::vector<int*> dvalidIndexes(num_cards, NULL);
    std::vector<int> pointsLeft_a(num_cards);
    std::vector<NeighborHelper<long64>> neighborHelpers(num_cards);
    // Here we will have
    int bucketsCount =  finalBucketCount;
    std::vector<int> extent;
    pointPartitionSize = pointsCountQuery / num_cards;
    //pointsCard(num_cards);
    for (int i = 0; i < num_cards - 1; i++){
        pointsCard[i] = pointPartitionSize;
    }
    pointsCard[num_cards - 1] = pointsCountQuery - (num_cards - 1) * pointPartitionSize;

#pragma omp parallel for num_threads(num_cards)
    for (int dev_id = 0; dev_id < num_cards; dev_id++) {
        cudaSetDevice(dev_id);
        neighborHelpers[dev_id].k = k;
        neighborHelpers[dev_id].pointsReference = pointsCount;
        pcReference[dev_id].pnts_x = dpoints_x[dev_id];
        pcReference[dev_id].pnts_y = dpoints_y[dev_id];
        pcReference[dev_id].pnts_z = dpoints_z[dev_id];
        UpdateHelperFromInitialize(neighborHelpers[dev_id], 0, dev_id, bucketsCount);
        cudaMalloc((void **)&dvalidIndexes[dev_id], pointsCountQuery * sizeof(int));
        cudaMemset(dvalidIndexes[dev_id], 0, pointsCountQuery * sizeof(int));
        neighborHelpers[dev_id].invalidIndexes = dvalidIndexes[dev_id];
        int level = 0;
        float min = -eps;
        float max = 1.0 + eps;
        cudaMalloc((void **) &dIndexesQueryAll[dev_id], pointsCountQuery * sizeof(int));

        cudaMalloc((void **) &dQuery_x[dev_id], pointsCountQuery * sizeof(float));
        cudaMemcpy(dQuery_x[dev_id], pointsQ_x, pointsCountQuery * sizeof(float),
                   cudaMemcpyHostToDevice);

        cudaMalloc((void **) &dQuery_y[dev_id], pointsCountQuery * sizeof(float));
        cudaMemcpy(dQuery_y[dev_id], pointsQ_y, pointsCountQuery * sizeof(float),
                   cudaMemcpyHostToDevice);

        cudaMalloc((void **) &dQuery_z[dev_id], pointsCountQuery * sizeof(float));
        cudaMemcpy(dQuery_z[dev_id], pointsQ_z, pointsCountQuery * sizeof(float),
                   cudaMemcpyHostToDevice);

        int threads = 1024;
        int blocks = (int) ceil((1.0 * pointsCountQuery) / threads);
        normalize_points<<<blocks, threads>>>(dQuery_x[dev_id],
                                              dQuery_y[dev_id],
                                              dQuery_z[dev_id],  this->minExtent,  this->maxExtent, pointsCountQuery);
        cudaDeviceSynchronize();

        cudaMalloc((void**)&dQueryIndexesInitial[dev_id], pointsCountQuery * sizeof(int));
        //thrust::sequence(thrust::device, dQueryIndexesInitial[dev_id], dQueryIndexesInitial[dev_id] + pointsCountQuery, 0);
        rearrangeQueryPoints(dQuery_x[dev_id], dQuery_y[dev_id], dQuery_z[dev_id], pointsCountQuery, dQueryIndexesInitial[dev_id]);

        pcQuery[dev_id].pnts_x = dQuery_x[dev_id];
        pcQuery[dev_id].pnts_y = dQuery_y[dev_id];
        pcQuery[dev_id].pnts_z = dQuery_z[dev_id];

        cudaMalloc((void **) &dIndexesQueryReverse[dev_id], pointsCountQuery * sizeof(int));
        int *indexes = new int[pointsCountQuery];
        for (int i = 0; i < pointsCountQuery; i++) indexes[i] = i;
        cudaMemcpy(dIndexesQueryAll[dev_id], indexes, pointsCountQuery * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dIndexesQueryReverse[dev_id], dIndexesQueryAll[dev_id], pointsCountQuery * sizeof(int), cudaMemcpyDeviceToDevice);
        neighborHelpers[dev_id].reverseQueryIndexes = dIndexesQueryReverse[dev_id];
        delete[] indexes;

        cudaMalloc((void **) &dIndexesQuery[dev_id], pointsCard[dev_id] * sizeof(int));
        cudaMemcpy(dIndexesQuery[dev_id], dIndexesQueryAll[dev_id] + dev_id * pointPartitionSize,
                   pointsCard[dev_id] * sizeof(int), cudaMemcpyDeviceToDevice);
        neighborHelpers[dev_id].indexesQuery = dIndexesQuery[dev_id];

        cudaMalloc((void **) &dcounters[dev_id], pointsCard[dev_id] * sizeof(int));
        cudaMemset(dcounters[dev_id], 0, pointsCard[dev_id] * sizeof(int));
        neighborHelpers[dev_id].counters = dcounters[dev_id];
        cudaMalloc((void **) &dcontinues[dev_id], pointsCard[dev_id] * sizeof(int));
        cudaMemset(dcontinues[dev_id], 0, pointsCard[dev_id] * sizeof(int));
        neighborHelpers[dev_id].continues = dcontinues[dev_id];
        //////////////KNEAREST NEIGHBOURS////////////////////////////////////////////////////
        cudaMalloc((void **) &dknndistances[dev_id], pointsCard[dev_id] * k * sizeof(float));
        cudaMalloc((void **) &dknn[dev_id], pointsCard[dev_id] * k * sizeof(int));
        neighborHelpers[dev_id].knn = dknn[dev_id];
        neighborHelpers[dev_id].knnDistances = dknndistances[dev_id];
        /////////////////////////////////////////////////////////////////////////////////////
        threads = 1024;
        blocks = ceil((1.0 * k * pointsCard[dev_id] / threads));
        setValue<int><<<blocks, threads>>>(dknn[dev_id], -1, k * pointsCard[dev_id]);
        cudaDeviceSynchronize();
        setValue<float><<<blocks, threads>>>(dknndistances[dev_id], 1e10, k * pointsCard[dev_id]);
        cudaDeviceSynchronize();
        int offset = 0;
        threads = 1024;
        blocks = ceil((1.0 * pointsCard[dev_id]) / threads);
        float h = (max - min) / bucketsCount;
        neighborHelpers[dev_id].h = h;
        neighborHelpers[dev_id].xmin = min;
        neighborHelpers[dev_id].xmax = max;

        newNeighborSearch::construct_neighbors<long64><<<blocks, threads>>>(pcReference[dev_id], pcQuery[dev_id],
                                                                            neighborHelpers[dev_id], offset,
                0, 0, 0,
                pointsCard[dev_id], dev_id * pointPartitionSize, thres);

        cudaDeviceSynchronize();
        threads = 1024;
        int pointsLeft = pointsCard[dev_id];
        blocks = ceil((1.0 * pointsLeft) / threads);
        invalidateIndexesNotValid<<<blocks, threads>>>(dIndexesQuery[dev_id], dcontinues[dev_id], dvalidIndexes[dev_id], pointsLeft);
        cudaDeviceSynchronize();

        int count = thrust::reduce(thrust::device, dcontinues[dev_id], dcontinues[dev_id] + pointsLeft, 0);
        if (count > 0) {
            recomputeBuffers(pointsLeft, count, dcontinues[dev_id], dIndexesQuery[dev_id], dcounters[dev_id]);
            neighborHelpers[dev_id].continues = dcontinues[dev_id];
            neighborHelpers[dev_id].indexesQuery = dIndexesQuery[dev_id];
            neighborHelpers[dev_id].counters = dcounters[dev_id];
        }
        else
            pointsLeft = count;
        pointsLeft_a[dev_id] = pointsLeft;
        //std::cout << "Do we work?" << std::endl;
    }

    float min = -eps;
    float max = 1.0 + eps;

    int count_global;
    std::vector<int> bucketsCount_a(num_cards);
    std::vector<int> offset_a (num_cards);
    std::vector<int> level_a (num_cards);
    for (int i = 0; i < num_cards; i++) {
        bucketsCount_a[i] =  this->finalBucketCount;
        offset_a[i] = 1;
        level_a[i] = 0;
        neighborHelpers[i].xmax = max;
        neighborHelpers[i].xmin = min;
    }
    std::cout << "Test 4" << std::endl;
#pragma omp parallel for num_threads(num_cards)
    for (int dev_id = 0; dev_id < num_cards; dev_id++) {
        cudaSetDevice(dev_id);
        int &offset = offset_a[dev_id];
        int &pointsLeft = pointsLeft_a[dev_id];
        int &bucketsCount = bucketsCount_a[dev_id];
        int &level = level_a[dev_id];
        float h = (max - min) / bucketsCount;
        int count;
        int iteration = 0;
        do {
            if (offset > 1 && bucketsCount > 8) {
                bucketsCount /= 2;
                level++;
                h = (max - min) / bucketsCount;
                neighborHelpers[dev_id].h = h;
                InitializeLevel(dev_id, level, bucketsCount, min, max);
                UpdateHelperFromInitialize(neighborHelpers[dev_id], level, dev_id, bucketsCount);
                offset = 0;
                offset++;
            }
            int threads, blocks;
            threads = 1024;
            blocks = ceil((1.0 * pointsLeft) / threads);
            cudaMemset(dcontinues[dev_id], 0, pointsLeft * sizeof(int));
            int offsets[2] = {-offset, offset};
            for (int ii = -offset + 1; ii <= offset - 1; ii++) {
                for (int jj = -offset + 1; jj <= offset - 1; jj++) {
                    for (int kk = 0; kk < 2; kk++) {
                        newNeighborSearch::construct_neighbors<long64><<<blocks, threads>>>(pcReference[dev_id], pcQuery[dev_id],
                                                                                            neighborHelpers[dev_id], offset, ii, jj, offsets[kk],
                                                                                    pointsLeft, dev_id * pointPartitionSize, thres);
                        cudaDeviceSynchronize();
                    }
                }
            }
            for (int ii = -offset + 1; ii <= offset - 1; ii++) {
                for (int jj = 0; jj < 2; jj++) {
                    for (int kk = -offset + 1; kk <= offset - 1; kk++) {
                        newNeighborSearch::construct_neighbors<long64><<<blocks, threads>>>(pcReference[dev_id], pcQuery[dev_id], neighborHelpers[dev_id], offset,
                                ii, offsets[jj], kk,
                                pointsLeft, dev_id * pointPartitionSize, thres);
                        cudaDeviceSynchronize();
                    }
                }
            }
            for (int ii = 0; ii < 2 ; ii++) {
                for (int jj = -offset + 1; jj <= offset - 1; jj++) {
                    for (int kk = -offset + 1; kk <= offset - 1; kk++) {
                        newNeighborSearch::construct_neighbors<long64><<<blocks, threads>>>(pcReference[dev_id], pcQuery[dev_id], neighborHelpers[dev_id], offset,
                                offsets[ii], jj, kk,
                                pointsLeft, dev_id * pointPartitionSize, thres);
                        cudaDeviceSynchronize();
                    }
                }
            }
            for (int ii = 0; ii < 2 ; ii++) {
                for (int jj = 0 ; jj < 2; jj++) {
                    for (int kk = -offset + 1; kk <= offset - 1; kk++) {
                        newNeighborSearch::construct_neighbors<long64><<<blocks, threads>>>(pcReference[dev_id], pcQuery[dev_id], neighborHelpers[dev_id], offset,
                                offsets[ii], offsets[jj], kk,
                                pointsLeft, dev_id * pointPartitionSize, thres);
                        cudaDeviceSynchronize();

                    }
                }
            }
            for (int ii = 0; ii < 2 ; ii++) {
                for (int jj = -offset + 1 ; jj <= offset - 1; jj++) {
                    for (int kk = 0; kk < 2; kk++) {
                        newNeighborSearch::construct_neighbors<long64><<<blocks, threads>>>(pcReference[dev_id], pcQuery[dev_id], neighborHelpers[dev_id], offset,
                                offsets[ii], jj, offsets[kk],
                                pointsLeft, dev_id * pointPartitionSize, thres);
                        cudaDeviceSynchronize();

                    }
                }
            }
            for (int ii = -offset + 1; ii <= offset - 1 ; ii++) {
                for (int jj = 0; jj < 2; jj++) {
                    for (int kk = 0; kk < 2; kk++) {
                        newNeighborSearch::construct_neighbors<long64><<<blocks, threads>>>(pcReference[dev_id], pcQuery[dev_id], neighborHelpers[dev_id], offset,
                                ii, offsets[jj], offsets[kk],
                                pointsLeft, dev_id * pointPartitionSize, thres);
                        cudaDeviceSynchronize();

                    }
                }
            }
            for (int ii = 0; ii < 2 ; ii++) {
                for (int jj = 0; jj < 2; jj++) {
                    for (int kk = 0; kk < 2; kk++) {
                        newNeighborSearch::construct_neighbors<long64><<<blocks, threads>>>(pcReference[dev_id], pcQuery[dev_id], neighborHelpers[dev_id], offset,
                                offsets[ii], offsets[jj], offsets[kk],
                                pointsLeft, dev_id * pointPartitionSize, thres);
                        cudaDeviceSynchronize();
                    }
                }
            }
            threads = 1024;
            blocks = ceil((1.0 * pointsLeft) / threads);
            invalidateIndexesNotValid<<<blocks, threads>>>(dIndexesQuery[dev_id], dcontinues[dev_id], dvalidIndexes[dev_id], pointsLeft);
            cudaDeviceSynchronize();
            offset++;
            count = thrust::reduce(thrust::device, dcontinues[dev_id], dcontinues[dev_id] + pointsLeft, 0);
#pragma omp critical
            {
                std::cout << "iteration:" << iteration++ << " dev_id:" << dev_id << " counter:" << count << std::endl;
            }
            if (count > 0) {
                recomputeBuffers(pointsLeft, count, dcontinues[dev_id], dIndexesQuery[dev_id], dcounters[dev_id]);
                neighborHelpers[dev_id].continues = dcontinues[dev_id];
                neighborHelpers[dev_id].indexesQuery = dIndexesQuery[dev_id];
                neighborHelpers[dev_id].counters = dcounters[dev_id];
            }
            else
                pointsLeft = count;

        } while (count > 0 && bucketsCount > 8);

        if (pointsLeft > 0){
            int threads = 1024;
            int blocks = ceil((1.0 * pointsLeft) / threads);
            newNeighborSearch::invalidateNotFinishedNeighbors<<<blocks, threads>>>(dIndexesQuery[dev_id], dcontinues[dev_id], dvalidIndexes[dev_id],
                    pointsLeft);
            cudaDeviceSynchronize();
        }
        int threads = 1024;
        int blocks = ceil((1.0 * pointsCard[dev_id]) / threads);
        newNeighborSearch::negateNotValidKnnIndexes<<<blocks, threads>>>(dknn[dev_id], k, dvalidIndexes[dev_id], pointsCard[dev_id],
                dev_id * pointPartitionSize);
        cudaDeviceSynchronize();

        cudaFree(dIndexesQuery[dev_id]);
        cudaFree(dcounters[dev_id]);
        cudaFree(dIndexesQueryReverse[dev_id]);
        cudaFree(dcontinues[dev_id]);
        cudaFree(dknndistances[dev_id]);
        cudaFree(dvalidIndexes[dev_id]);
        cudaFree(dIndexesQueryAll[dev_id]);
    }
}

void GridStructure::GetKNNIndexes(int* knnIndexes){
    int* knnIndexesTemp = new int[(size_t)neighborCount * (size_t)pointsCountQuery];
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
    std::vector<int> originalIndexes(pointsCountQuery);
    cudaMemcpy(originalIndexes.data(), GetOriginalQuerIndexes()[0], (size_t)pointsCountQuery * sizeof(int), cudaMemcpyDeviceToHost);
#pragma omp parallel for
    for (int i = 0; i < pointsCountQuery; i++){
        int idx = originalIndexes[i];
        memcpy(knnIndexes + neighborCount * (size_t)idx, knnIndexesTemp + neighborCount * (size_t)i,
               neighborCount * sizeof(int));
    }
    delete[] knnIndexesTemp;
}

void GridStructure::GRIDCUDAKNNSELF(int k, float thres) {

    thres /= (this->maxExtent -  this->minExtent);
    thres *= thres;
    //int pointsCount =  this->pointsCount;
    //int num_cards = this->num_cards;
    int pointsCountQuery = pointsCount;
    clock_t starttime =  clock();
    std::vector<int*> dcontinues(num_cards, NULL);
    std::vector<int*> dIndexesQuery(num_cards, NULL);
    std::vector<int*> dIndexesQueryAll(num_cards, NULL);
    std::vector<int*> dIndexesQueryReverse(num_cards, NULL);
    std::vector<int*> dcounters(num_cards, NULL);
    std::vector<float*> dknndistances(num_cards, NULL);
    std::vector<int*> dvalidIndexes(num_cards, NULL);
    std::vector<int> pointsLeft_a(num_cards);
    // Here we will have
    int bucketsCount =  this->finalBucketCount;
    std::vector<int> extent;
    pointPartitionSize = pointsCountQuery / num_cards;
    //pointsCard(num_cards);
    for (int i = 0; i < num_cards - 1; i++){
        pointsCard[i] = pointPartitionSize;
    }
    pointsCard[num_cards - 1] = pointsCountQuery - (num_cards - 1) * pointPartitionSize;
    //int totalPointCount = pointsCount + pointsCountQuery;

#pragma omp parallel for num_threads(num_cards)
    for (int dev_id = 0; dev_id < num_cards; dev_id++) {
        cudaSetDevice(dev_id);
        cudaMalloc((void **)&dvalidIndexes[dev_id], pointsCountQuery * sizeof(int));
        cudaMemset(dvalidIndexes[dev_id], 0, pointsCountQuery * sizeof(int));
        int level = 0;
        float min = -eps;
        float max = 1.0 + eps;
        cudaMalloc((void **) &dIndexesQueryAll[dev_id], pointsCountQuery * sizeof(int));
        cudaMalloc((void **) &dIndexesQueryReverse[dev_id], pointsCountQuery * sizeof(int));
        int *indexes = new int[pointsCountQuery];
        for (int i = 0; i < pointsCountQuery; i++) indexes[i] = i;
        cudaMemcpy(dIndexesQueryAll[dev_id], indexes, pointsCountQuery * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dIndexesQueryReverse[dev_id], dIndexesQueryAll[dev_id], pointsCountQuery * sizeof(int), cudaMemcpyDeviceToDevice);
        delete[] indexes;

        cudaMalloc((void **) &dIndexesQuery[dev_id], pointsCard[dev_id] * sizeof(int));
        cudaMemcpy(dIndexesQuery[dev_id], dIndexesQueryAll[dev_id] + dev_id * pointPartitionSize,
                   pointsCard[dev_id] * sizeof(int), cudaMemcpyDeviceToDevice);

        cudaMalloc((void **) &dcounters[dev_id], pointsCard[dev_id] * sizeof(int));
        cudaMemset(dcounters[dev_id], 0, pointsCard[dev_id] * sizeof(int));
        cudaMalloc((void **) &dcontinues[dev_id], pointsCard[dev_id] * sizeof(int));
        cudaMemset(dcontinues[dev_id], 0, pointsCard[dev_id] * sizeof(int));

        //////////////KNEAREST NEIGHBOURS////////////////////////////////////////////////////
        cudaMalloc((void **) &dknndistances[dev_id], pointsCard[dev_id] * k * sizeof(float));
        cudaMalloc((void **) &dknn[dev_id], pointsCard[dev_id] * k * sizeof(int));
        /////////////////////////////////////////////////////////////////////////////////////
        int threads = 1024;
        int blocks = ceil((1.0 * k * pointsCard[dev_id] / threads));
        setValue<int><<<blocks, threads>>>(dknn[dev_id], -1, k * pointsCard[dev_id]);
        cudaDeviceSynchronize();
        setValue<float><<<blocks, threads>>>(dknndistances[dev_id], 1e10, k * pointsCard[dev_id]);
        cudaDeviceSynchronize();
        int offset = 0;
        threads = 1024;
        blocks = ceil((1.0 * pointsCard[dev_id]) / threads);
        float h = (max - min) / bucketsCount;

        newNeighborSearch::construct_neighbors_self<long64><<<blocks, threads>>>(this->dpoints_x[dev_id], this->dpoints_y[dev_id],
                this->dpoints_z[dev_id], this->hierachyTree[level]->dpointBucketIndexes[dev_id],
                this->hierachyTree[level]->vertexPartitionSize[dev_id],
                this->hierachyTree[level]->dbucketIndexes[dev_id], pointsCount,
                dknndistances[dev_id], dknn[dev_id],
                bucketsCount, k, offset, dcounters[dev_id], dIndexesQuery[dev_id],
                this->hierachyTree[level]->dindexes[dev_id], dcontinues[dev_id], 0, 0, 0,
                pointsCard[dev_id], min, max, h, thres,
                dIndexesQueryReverse[dev_id], dev_id * pointPartitionSize, dvalidIndexes[dev_id]);

        cudaDeviceSynchronize();
        threads = 1024;
        int pointsLeft = pointsCard[dev_id];
        blocks = ceil((1.0 * pointsLeft) / threads);
        invalidateIndexesNotValid<<<blocks, threads>>>(dIndexesQuery[dev_id], dcontinues[dev_id], dvalidIndexes[dev_id], pointsLeft);
        cudaDeviceSynchronize();

        int count = thrust::reduce(thrust::device, dcontinues[dev_id], dcontinues[dev_id] + pointsLeft, 0);
#pragma omp critical
        {
            std::cout << "initial dev_id:" << dev_id << " counter:" << count << std::endl;
        }
        if (count > 0)
            recomputeBuffers(pointsLeft, count, dcontinues[dev_id], dIndexesQuery[dev_id], dcounters[dev_id]);
        else
            pointsLeft = count;
        pointsLeft_a[dev_id] = count;
        //std::cout << "Do we work?" << std::endl;
    }

    float min = -eps;
    float max = 1.0 + eps;

    int count_global;
    std::vector<int> bucketsCount_a(num_cards);
    std::vector<int> offset_a (num_cards);
    std::vector<int> level_a (num_cards);
    for (int i = 0; i < num_cards; i++) {
        bucketsCount_a[i] =  this->finalBucketCount;
        offset_a[i] = 1;
        level_a[i] = 0;
    }
    std::cout << "Test 4" << std::endl;
#pragma omp parallel for num_threads(num_cards)
    for (int dev_id = 0; dev_id < num_cards; dev_id++) {
        cudaSetDevice(dev_id);
        int &offset = offset_a[dev_id];
        int &pointsLeft = pointsLeft_a[dev_id];
        int &bucketsCount = bucketsCount_a[dev_id];
        int &level = level_a[dev_id];
        float h = (max - min) / bucketsCount;
        int count = pointsLeft;
        int iteration = 0;
        while (count > 0 && bucketsCount > 8) {
            if (offset > 1 && bucketsCount > 8) {
                bucketsCount /= 2;
                level++;
                h = (max - min) / bucketsCount;
                if (InitializeLevel(dev_id, level, bucketsCount, min, max)){

                }
                offset = 0;
                offset++;
            }
            int threads, blocks;
            threads = 1024;
            blocks = ceil((1.0 * pointsLeft) / threads);
            cudaMemset(dcontinues[dev_id], 0, pointsLeft * sizeof(int));
            int offsets[2] = {-offset, offset};
            for (int ii = -offset + 1; ii <= offset - 1; ii++) {
                for (int jj = -offset + 1; jj <= offset - 1; jj++) {
                    for (int kk = 0; kk < 2; kk++) {
                        newNeighborSearch::construct_neighbors_self<long64><<<blocks, threads>>>( this->dpoints_x[dev_id],
                                this->dpoints_y[dev_id],
                                this->dpoints_z[dev_id],
                                this->hierachyTree[level]->dpointBucketIndexes[dev_id],
                                this->hierachyTree[level]->vertexPartitionSize[dev_id],
                                this->hierachyTree[level]->dbucketIndexes[dev_id],
                                pointsCount, dknndistances[dev_id], dknn[dev_id], bucketsCount,
                                 k, offset, dcounters[dev_id], dIndexesQuery[dev_id],
                                this->hierachyTree[level]->dindexes[dev_id], dcontinues[dev_id], ii, jj, offsets[kk],
                                pointsLeft, min, max, h,
                                thres, dIndexesQueryReverse[dev_id], dev_id * pointPartitionSize,
                                dvalidIndexes[dev_id]);
                        cudaDeviceSynchronize();
                    }
                }
            }
            for (int ii = -offset + 1; ii <= offset - 1; ii++) {
                for (int jj = 0; jj < 2; jj++) {
                    for (int kk = -offset + 1; kk <= offset - 1; kk++) {
                        newNeighborSearch::construct_neighbors_self<long64><<<blocks, threads>>>( this->dpoints_x[dev_id],
                                this->dpoints_y[dev_id],
                                this->dpoints_z[dev_id],
                                this->hierachyTree[level]->dpointBucketIndexes[dev_id],
                                this->hierachyTree[level]->vertexPartitionSize[dev_id],
                                this->hierachyTree[level]->dbucketIndexes[dev_id],
                                pointsCount, dknndistances[dev_id], dknn[dev_id], bucketsCount,
                                k, offset, dcounters[dev_id], dIndexesQuery[dev_id],
                                this->hierachyTree[level]->dindexes[dev_id], dcontinues[dev_id], ii, offsets[jj], kk,
                                pointsLeft, min, max, h,
                                thres, dIndexesQueryReverse[dev_id], dev_id * pointPartitionSize,
                                dvalidIndexes[dev_id]);
                        cudaDeviceSynchronize();
                    }
                }
            }
            for (int ii = 0; ii < 2 ; ii++) {
                for (int jj = -offset + 1; jj <= offset - 1; jj++) {
                    for (int kk = -offset + 1; kk <= offset - 1; kk++) {
                        newNeighborSearch::construct_neighbors_self<long64><<<blocks, threads>>>( this->dpoints_x[dev_id],
                                this->dpoints_y[dev_id],
                                this->dpoints_z[dev_id],
                                this->hierachyTree[level]->dpointBucketIndexes[dev_id],
                                this->hierachyTree[level]->vertexPartitionSize[dev_id],
                                this->hierachyTree[level]->dbucketIndexes[dev_id],
                                pointsCount, dknndistances[dev_id], dknn[dev_id], bucketsCount,
                                k, offset, dcounters[dev_id], dIndexesQuery[dev_id],
                                this->hierachyTree[level]->dindexes[dev_id], dcontinues[dev_id], offsets[ii], jj, kk,
                                pointsLeft, min, max, h,
                                thres, dIndexesQueryReverse[dev_id], dev_id * pointPartitionSize,
                                dvalidIndexes[dev_id]);
                        cudaDeviceSynchronize();
                    }
                }
            }

            for (int ii = 0; ii < 2 ; ii++) {
                for (int jj = 0 ; jj < 2; jj++) {
                    for (int kk = -offset + 1; kk <= offset - 1; kk++) {
                        newNeighborSearch::construct_neighbors_self<long64><<<blocks, threads>>>( this->dpoints_x[dev_id],
                                this->dpoints_y[dev_id],
                                this->dpoints_z[dev_id],
                                this->hierachyTree[level]->dpointBucketIndexes[dev_id],
                                this->hierachyTree[level]->vertexPartitionSize[dev_id],
                                this->hierachyTree[level]->dbucketIndexes[dev_id],
                                pointsCount, dknndistances[dev_id], dknn[dev_id], bucketsCount,
                                k, offset, dcounters[dev_id], dIndexesQuery[dev_id],
                                this->hierachyTree[level]->dindexes[dev_id], dcontinues[dev_id], offsets[ii], offsets[jj], kk,
                                pointsLeft, min, max, h,
                                thres, dIndexesQueryReverse[dev_id], dev_id * pointPartitionSize,
                                dvalidIndexes[dev_id]);
                        cudaDeviceSynchronize();

                    }
                }
            }

            for (int ii = 0; ii < 2 ; ii++) {
                for (int jj = -offset + 1 ; jj <= offset - 1; jj++) {
                    for (int kk = 0; kk < 2; kk++) {
                        newNeighborSearch::construct_neighbors_self<long64><<<blocks, threads>>>( this->dpoints_x[dev_id],
                                this->dpoints_y[dev_id],
                                this->dpoints_z[dev_id],
                                this->hierachyTree[level]->dpointBucketIndexes[dev_id],
                                this->hierachyTree[level]->vertexPartitionSize[dev_id],
                                this->hierachyTree[level]->dbucketIndexes[dev_id],
                                pointsCount, dknndistances[dev_id], dknn[dev_id], bucketsCount,
                                k, offset, dcounters[dev_id], dIndexesQuery[dev_id],
                                this->hierachyTree[level]->dindexes[dev_id], dcontinues[dev_id], offsets[ii], jj, offsets[kk],
                                pointsLeft, min, max, h,
                                thres, dIndexesQueryReverse[dev_id], dev_id * pointPartitionSize,
                                dvalidIndexes[dev_id]);
                        cudaDeviceSynchronize();

                    }
                }
            }
            for (int ii = -offset + 1; ii <= offset - 1 ; ii++) {
                for (int jj = 0; jj < 2; jj++) {
                    for (int kk = 0; kk < 2; kk++) {
                        newNeighborSearch::construct_neighbors_self<long64><<<blocks, threads>>>( this->dpoints_x[dev_id],
                                this->dpoints_y[dev_id],
                                this->dpoints_z[dev_id],
                                this->hierachyTree[level]->dpointBucketIndexes[dev_id],
                                this->hierachyTree[level]->vertexPartitionSize[dev_id],
                                this->hierachyTree[level]->dbucketIndexes[dev_id],
                                pointsCount, dknndistances[dev_id], dknn[dev_id], bucketsCount,
                                k, offset, dcounters[dev_id], dIndexesQuery[dev_id],
                                this->hierachyTree[level]->dindexes[dev_id], dcontinues[dev_id], ii, offsets[jj], offsets[kk],
                                pointsLeft, min, max, h,
                                thres, dIndexesQueryReverse[dev_id], dev_id * pointPartitionSize,
                                dvalidIndexes[dev_id]);
                        cudaDeviceSynchronize();

                    }
                }
            }
            for (int ii = 0; ii < 2 ; ii++) {
                for (int jj = 0; jj < 2; jj++) {
                    for (int kk = 0; kk < 2; kk++) {
                        newNeighborSearch::construct_neighbors_self<long64><<<blocks, threads>>>( this->dpoints_x[dev_id],
                                this->dpoints_y[dev_id],
                                this->dpoints_z[dev_id],
                                this->hierachyTree[level]->dpointBucketIndexes[dev_id],
                                this->hierachyTree[level]->vertexPartitionSize[dev_id],
                                this->hierachyTree[level]->dbucketIndexes[dev_id],
                                pointsCount, dknndistances[dev_id], dknn[dev_id], bucketsCount,
                                k, offset, dcounters[dev_id], dIndexesQuery[dev_id],
                                this->hierachyTree[level]->dindexes[dev_id], dcontinues[dev_id], offsets[ii], offsets[jj], offsets[kk],
                                pointsLeft, min, max, h,
                                thres, dIndexesQueryReverse[dev_id], dev_id * pointPartitionSize,
                                dvalidIndexes[dev_id]);
                        cudaDeviceSynchronize();
                    }
                }
            }
            threads = 1024;
            blocks = ceil((1.0 * pointsLeft) / threads);
            invalidateIndexesNotValid<<<blocks, threads>>>(dIndexesQuery[dev_id], dcontinues[dev_id], dvalidIndexes[dev_id], pointsLeft);
            cudaDeviceSynchronize();
            offset++;
            count = thrust::reduce(thrust::device, dcontinues[dev_id], dcontinues[dev_id] + pointsLeft, 0);
#pragma omp critical
            {
                std::cout << "iteration:" << iteration++ << " dev_id:" << dev_id << " counter:" << count << std::endl;
            }
            if (count > 0)
                recomputeBuffers(pointsLeft, count, dcontinues[dev_id], dIndexesQuery[dev_id], dcounters[dev_id]);
            else
                pointsLeft = count;

        }

        if (pointsLeft > 0){
            int threads = 1024;
            int blocks = ceil((1.0 * pointsLeft) / threads);
            newNeighborSearch::invalidateNotFinishedNeighbors<<<blocks, threads>>>(dIndexesQuery[dev_id], dcontinues[dev_id], dvalidIndexes[dev_id],
                    pointsLeft);
            cudaDeviceSynchronize();
        }
        int threads = 1024;
        int blocks = ceil((1.0 * pointsCard[dev_id]) / threads);
        newNeighborSearch::negateNotValidKnnIndexes<<<blocks, threads>>>(dknn[dev_id], k, dvalidIndexes[dev_id], pointsCard[dev_id],
                dev_id * pointPartitionSize);
        cudaDeviceSynchronize();

        cudaFree(dIndexesQuery[dev_id]);
        cudaFree(dcounters[dev_id]);
        cudaFree(dIndexesQueryReverse[dev_id]);

        cudaFree(dcontinues[dev_id]);
        cudaFree(dknndistances[dev_id]);
        cudaFree(dvalidIndexes[dev_id]);
        cudaFree(dIndexesQueryAll[dev_id]);
    }
    /*cudaSetDevice(0);
    int *dknnGather, *dknnTemp;
    //float* dknnDist;
    cudaMalloc((void**)& dknnGather, k * pointsCountQuery * sizeof(int));
    //cudaMalloc((void**)& dknnDist, k * pointsCountQuery * sizeof(float));
    cudaMalloc((void**)& dknnTemp, k * pointsCountQuery * sizeof(int));
#pragma omp parallel for num_threads(num_cards)
    for (int dev_id = 0; dev_id < num_cards; dev_id++) {
        //cudaSetDevice(dev_id);
        cudaMemcpy(dknnTemp + k * dev_id * pointPartitionSize, dknn[dev_id], k * pointsCard[dev_id] * sizeof(int),
                   cudaMemcpyDeviceToDevice);
    }
    int threads = 1024;
    int blocks = ceil((1.0 * pointsCountQuery) / threads);*/
    //rearrangeNeighbors<<<blocks, threads>>>(dknnGather, dknnTemp, /*dknnDist,dknndistances[0],*/
    //                                        this->doriginal_indexes[0], dQueryIndexesInitial[0],
    //                                        k, pointsCountQuery);
    /*cudaDeviceSynchronize();
    cudaFree(dknnTemp);
    cudaMemcpy(knn, dknnGather, k * pointsCountQuery * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dknnGather);

    for (int dev_id = 0; dev_id < num_cards; dev_id++){
        cudaSetDevice(dev_id);
        cudaFree(dQueryIndexesInitial[dev_id]);
        cudaFree(dknn[dev_id]);
    }*/

    /*FILE* file = fopen("knnBugFreeQ.bin","wb");
    fwrite(knn, sizeof(int), k * pointsCountQuery, file);
    fclose(file);

    std::vector<float> knnd(k * pointsCountQuery);
    cudaMemcpy(knnd.data(), dknnDist, k * pointsCountQuery * sizeof(float), cudaMemcpyDeviceToHost);
    file = fopen("knnBugFreeQDistancesQ.bin","wb");
    fwrite(knnd.data(), sizeof(float), k * pointsCountQuery, file);
    fclose(file);*/
}

void GridStructure::GRIDCUDAKNNSELF_COMPACT(int k, float thres) {
    std::cout << "Computing" << std::endl;
    thres /= (this->maxExtent -  this->minExtent);
    thres *= thres;

    self = true;
    pointsCountQuery = pointsCount;
    neighborCount = k;
    clock_t starttime =  clock();
    std::vector<PointCloud> pcReference(num_cards);
    std::vector<int*> dcontinues(num_cards, NULL);
    std::vector<int*> dIndexesQuery(num_cards, NULL);
    std::vector<int*> dIndexesQueryAll(num_cards, NULL);
    std::vector<int*> dIndexesQueryReverse(num_cards, NULL);
    std::vector<int*> dcounters(num_cards, NULL);
    std::vector<float*> dknndistances(num_cards, NULL);
    std::vector<int*> dvalidIndexes(num_cards, NULL);
    std::vector<int> pointsLeft_a(num_cards);
    std::vector<NeighborHelper<long64>> neighborHelpers(num_cards);
    // Here we will have
    int bucketsCount =  this->finalBucketCount;
    std::vector<int> extent;
    pointPartitionSize = pointsCountQuery / num_cards;
    for (int i = 0; i < num_cards - 1; i++){
        pointsCard[i] = pointPartitionSize;
    }
    pointsCard[num_cards - 1] = pointsCountQuery - (num_cards - 1) * pointPartitionSize;
    //int totalPointCount = pointsCount + pointsCountQuery;

#pragma omp parallel for num_threads(num_cards)
    for (int dev_id = 0; dev_id < num_cards; dev_id++) {
        cudaSetDevice(dev_id);
        neighborHelpers[dev_id].k = k;
        neighborHelpers[dev_id].pointsReference = pointsCount;
        pcReference[dev_id].pnts_x = dpoints_x[dev_id];
        pcReference[dev_id].pnts_y = dpoints_y[dev_id];
        pcReference[dev_id].pnts_z = dpoints_z[dev_id];
        float min = -eps;
        float max = 1.0f + eps;
        InitializeLevel(dev_id, 0, bucketsCount, min, max);
        UpdateHelperFromInitialize(neighborHelpers[dev_id], 0, dev_id, bucketsCount);
        cudaMalloc((void **)&dvalidIndexes[dev_id], pointsCountQuery * sizeof(int));
        cudaMemset(dvalidIndexes[dev_id], 0, pointsCountQuery * sizeof(int));
        neighborHelpers[dev_id].invalidIndexes = dvalidIndexes[dev_id];

        int level = 0;

        cudaMalloc((void **) &dIndexesQueryAll[dev_id], pointsCountQuery * sizeof(int));
        cudaMalloc((void **) &dIndexesQueryReverse[dev_id], pointsCountQuery * sizeof(int));
        int *indexes = new int[pointsCountQuery];
        for (int i = 0; i < pointsCountQuery; i++) indexes[i] = i;
        cudaMemcpy(dIndexesQueryAll[dev_id], indexes, pointsCountQuery * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dIndexesQueryReverse[dev_id], dIndexesQueryAll[dev_id], pointsCountQuery * sizeof(int), cudaMemcpyDeviceToDevice);
        neighborHelpers[dev_id].reverseQueryIndexes = dIndexesQueryReverse[dev_id];
        delete[] indexes;

        cudaMalloc((void **) &dIndexesQuery[dev_id], pointsCard[dev_id] * sizeof(int));
        cudaMemcpy(dIndexesQuery[dev_id], dIndexesQueryAll[dev_id] + dev_id * pointPartitionSize,
                   pointsCard[dev_id] * sizeof(int), cudaMemcpyDeviceToDevice);
        neighborHelpers[dev_id].indexesQuery = dIndexesQuery[dev_id];

        cudaMalloc((void **) &dcounters[dev_id], pointsCard[dev_id] * sizeof(int));
        cudaMemset(dcounters[dev_id], 0, pointsCard[dev_id] * sizeof(int));
        neighborHelpers[dev_id].counters = dcounters[dev_id];
        cudaMalloc((void **) &dcontinues[dev_id], pointsCard[dev_id] * sizeof(int));
        cudaMemset(dcontinues[dev_id], 0, pointsCard[dev_id] * sizeof(int));
        neighborHelpers[dev_id].continues = dcontinues[dev_id];

        //////////////KNEAREST NEIGHBOURS////////////////////////////////////////////////////
        cudaMalloc((void **) &dknndistances[dev_id], pointsCard[dev_id] * k * sizeof(float));
        cudaMalloc((void **) &dknn[dev_id], pointsCard[dev_id] * k * sizeof(int));
        neighborHelpers[dev_id].knn = dknn[dev_id];
        neighborHelpers[dev_id].knnDistances = dknndistances[dev_id];
        /////////////////////////////////////////////////////////////////////////////////////
        int threads = 1024;
        int blocks = ceil((1.0 * k * pointsCard[dev_id] / threads));
        setValue<int><<<blocks, threads>>>(dknn[dev_id], -1, k * pointsCard[dev_id]);
        cudaDeviceSynchronize();
        setValue<float><<<blocks, threads>>>(dknndistances[dev_id], 1e10, k * pointsCard[dev_id]);
        cudaDeviceSynchronize();
        int offset = 0;
        threads = 1024;
        blocks = ceil((1.0 * pointsCard[dev_id]) / threads);
        float h = (max - min) / bucketsCount;
        neighborHelpers[dev_id].h = h;
        neighborHelpers[dev_id].xmin = min;
        neighborHelpers[dev_id].xmax = max;
        newNeighborSearch::construct_neighbors_self_compact<long64><<<blocks, threads>>>(pcReference[dev_id], neighborHelpers[dev_id], offset,
                                                                                         0, 0, 0,
                                                                                           pointsCard[dev_id], dev_id * pointPartitionSize, thres);

        cudaDeviceSynchronize();

        threads = 1024;
        int pointsLeft = pointsCard[dev_id];
        blocks = ceil((1.0 * pointsLeft) / threads);
        invalidateIndexesNotValid<<<blocks, threads>>>(dIndexesQuery[dev_id], dcontinues[dev_id], dvalidIndexes[dev_id], pointsLeft);
        cudaDeviceSynchronize();

        int count = thrust::reduce(thrust::device, dcontinues[dev_id], dcontinues[dev_id] + pointsLeft, 0);
#pragma omp critical
        {
            std::cout << "initial dev_id:" << dev_id << " counter:" << count << std::endl;
        }
        if (count > 0)
        {
            recomputeBuffers(pointsLeft, count, dcontinues[dev_id], dIndexesQuery[dev_id], dcounters[dev_id]);
            neighborHelpers[dev_id].continues = dcontinues[dev_id];
            neighborHelpers[dev_id].indexesQuery = dIndexesQuery[dev_id];
            neighborHelpers[dev_id].counters = dcounters[dev_id];
        }
        else
            pointsLeft = count;
        pointsLeft_a[dev_id] = pointsLeft;
        //std::cout << "Do we work?" << std::endl;
    }

    float min = -eps;
    float max = 1.0 + eps;
    //exit(0);
    int count_global;
    std::vector<int> bucketsCount_a(num_cards);
    std::vector<int> offset_a (num_cards);
    std::vector<int> level_a (num_cards);
    for (int i = 0; i < num_cards; i++) {
        bucketsCount_a[i] =  this->finalBucketCount;
        offset_a[i] = 1;
        level_a[i] = 0;
        neighborHelpers[i].xmax = max;
        neighborHelpers[i].xmin = min;
    }
    std::cout << "Test 4" << std::endl;
#pragma omp parallel for num_threads(num_cards)
    for (int dev_id = 0; dev_id < num_cards; dev_id++) {
        cudaSetDevice(dev_id);
        int &offset = offset_a[dev_id];
        int &pointsLeft = pointsLeft_a[dev_id];
        int &bucketsCount = bucketsCount_a[dev_id];
        int &level = level_a[dev_id];
        float h = (max - min) / bucketsCount;
        int count;
        int iteration = 0;
        do {
            if (offset > 1 && bucketsCount > 8) {
                bucketsCount /= 2;
                level++;
                h = (max - min) / bucketsCount;
                neighborHelpers[dev_id].h = h;
                InitializeLevel(dev_id, level, bucketsCount, min, max);
                UpdateHelperFromInitialize(neighborHelpers[dev_id], level, dev_id, bucketsCount);
                offset = 0;
                offset++;
            }
            int threads, blocks;
            threads = 1024;
            blocks = ceil((1.0 * pointsLeft) / threads);
            cudaMemset(dcontinues[dev_id], 0, pointsLeft * sizeof(int));
            int offsets[2] = {-offset, offset};
            for (int ii = -offset + 1; ii <= offset - 1; ii++) {
                for (int jj = -offset + 1; jj <= offset - 1; jj++) {
                    for (int kk = 0; kk < 2; kk++) {
                        newNeighborSearch::construct_neighbors_self_compact<long64><<<blocks, threads>>>(pcReference[dev_id], neighborHelpers[dev_id], offset,
                                ii, jj, offsets[kk],
                                pointsLeft, dev_id * pointPartitionSize, thres);
                        cudaDeviceSynchronize();
                    }
                }
            }
            for (int ii = -offset + 1; ii <= offset - 1; ii++) {
                for (int jj = 0; jj < 2; jj++) {
                    for (int kk = -offset + 1; kk <= offset - 1; kk++) {
                        newNeighborSearch::construct_neighbors_self_compact<long64><<<blocks, threads>>>(pcReference[dev_id], neighborHelpers[dev_id], offset,
                                ii, offsets[jj], kk,
                                pointsLeft, dev_id * pointPartitionSize, thres);
                        cudaDeviceSynchronize();
                    }
                }
            }
            for (int ii = 0; ii < 2 ; ii++) {
                for (int jj = -offset + 1; jj <= offset - 1; jj++) {
                    for (int kk = -offset + 1; kk <= offset - 1; kk++) {
                        newNeighborSearch::construct_neighbors_self_compact<long64><<<blocks, threads>>>(pcReference[dev_id], neighborHelpers[dev_id], offset,
                                offsets[ii], jj, kk,
                                pointsLeft, dev_id * pointPartitionSize, thres);
                        cudaDeviceSynchronize();
                    }
                }
            }
            for (int ii = 0; ii < 2 ; ii++) {
                for (int jj = 0 ; jj < 2; jj++) {
                    for (int kk = -offset + 1; kk <= offset - 1; kk++) {
                        newNeighborSearch::construct_neighbors_self_compact<long64><<<blocks, threads>>>(pcReference[dev_id], neighborHelpers[dev_id], offset,
                                offsets[ii], offsets[jj], kk,
                                pointsLeft, dev_id * pointPartitionSize, thres);
                        cudaDeviceSynchronize();
                    }
                }
            }

            for (int ii = 0; ii < 2 ; ii++) {
                for (int jj = -offset + 1 ; jj <= offset - 1; jj++) {
                    for (int kk = 0; kk < 2; kk++) {
                        newNeighborSearch::construct_neighbors_self_compact<long64><<<blocks, threads>>>(pcReference[dev_id], neighborHelpers[dev_id], offset,
                                offsets[ii], jj, offsets[kk],
                                pointsLeft, dev_id * pointPartitionSize, thres);
                        cudaDeviceSynchronize();
                    }
                }
            }
            for (int ii = -offset + 1; ii <= offset - 1 ; ii++) {
                for (int jj = 0; jj < 2; jj++) {
                    for (int kk = 0; kk < 2; kk++) {
                        newNeighborSearch::construct_neighbors_self_compact<long64><<<blocks, threads>>>(pcReference[dev_id], neighborHelpers[dev_id], offset,
                                ii, offsets[jj], offsets[kk],
                                pointsLeft, dev_id * pointPartitionSize, thres);
                        cudaDeviceSynchronize();

                    }
                }
            }
            for (int ii = 0; ii < 2 ; ii++) {
                for (int jj = 0; jj < 2; jj++) {
                    for (int kk = 0; kk < 2; kk++) {
                        newNeighborSearch::construct_neighbors_self_compact<long64><<<blocks, threads>>>(pcReference[dev_id], neighborHelpers[dev_id], offset,
                                offsets[ii], offsets[jj], offsets[kk],
                                pointsLeft, dev_id * pointPartitionSize, thres);
                        cudaDeviceSynchronize();
                    }
                }
            }
            threads = 1024;
            blocks = ceil((1.0 * pointsLeft) / threads);
            invalidateIndexesNotValid<<<blocks, threads>>>(dIndexesQuery[dev_id], dcontinues[dev_id], dvalidIndexes[dev_id], pointsLeft);
            cudaDeviceSynchronize();
            offset++;
            count = thrust::reduce(thrust::device, dcontinues[dev_id], dcontinues[dev_id] + pointsLeft, 0);
#pragma omp critical
            {
                std::cout << "iteration:" << iteration++ << " dev_id:" << dev_id << " counter:" << count << std::endl;
            }
            if (count > 0)
            {
                recomputeBuffers(pointsLeft, count, dcontinues[dev_id], dIndexesQuery[dev_id], dcounters[dev_id]);
                neighborHelpers[dev_id].continues = dcontinues[dev_id];
                neighborHelpers[dev_id].indexesQuery = dIndexesQuery[dev_id];
                neighborHelpers[dev_id].counters = dcounters[dev_id];
            }
            else
                pointsLeft = count;

        } while (count > 0 && bucketsCount > 8);


        if (pointsLeft > 0){
            int threads = 1024;
            int blocks = ceil((1.0 * pointsLeft) / threads);
            newNeighborSearch::invalidateNotFinishedNeighbors<<<blocks, threads>>>(dIndexesQuery[dev_id],
                                                                                   dcontinues[dev_id], dvalidIndexes[dev_id], pointsLeft);
            cudaDeviceSynchronize();
        }
        int threads = 1024;
        int blocks = ceil((1.0 * pointsCard[dev_id]) / threads);
        newNeighborSearch::negateNotValidKnnIndexes<<<blocks, threads>>>(dknn[dev_id], k, dvalidIndexes[dev_id], pointsCard[dev_id],
                dev_id * pointPartitionSize);
        cudaDeviceSynchronize();

        cudaFree(dIndexesQuery[dev_id]);
        cudaFree(dcounters[dev_id]);
        cudaFree(dIndexesQueryReverse[dev_id]);

        cudaFree(dcontinues[dev_id]);
        cudaFree(dknndistances[dev_id]);
        cudaFree(dvalidIndexes[dev_id]);
        cudaFree(dIndexesQueryAll[dev_id]);
    }
    /*cudaSetDevice(0);
    int *dknnGather, *dknnTemp;
    //float* dknnDist;
    cudaMalloc((void**)& dknnGather, k * pointsCountQuery * sizeof(int));
    //cudaMalloc((void**)& dknnDist, k * pointsCountQuery * sizeof(float));
    cudaMalloc((void**)& dknnTemp, k * pointsCountQuery * sizeof(int));
#pragma omp parallel for num_threads(num_cards)
    for (int dev_id = 0; dev_id < num_cards; dev_id++) {
        //cudaSetDevice(dev_id);
        cudaMemcpy(dknnTemp + k * dev_id * pointPartitionSize, dknn[dev_id], k * pointsCard[dev_id] * sizeof(int),
                   cudaMemcpyDeviceToDevice);
    }
    int threads = 1024;
    int blocks = ceil((1.0 * pointsCountQuery) / threads);*/
    //rearrangeNeighbors<<<blocks, threads>>>(dknnGather, dknnTemp, /*dknnDist,dknndistances[0],*/
    //                                        this->doriginal_indexes[0], dQueryIndexesInitial[0],
    //                                        k, pointsCountQuery);
    /*cudaDeviceSynchronize();
    cudaFree(dknnTemp);
    cudaMemcpy(knn, dknnGather, k * pointsCountQuery * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dknnGather);

    for (int dev_id = 0; dev_id < num_cards; dev_id++){
        cudaSetDevice(dev_id);
        cudaFree(dQueryIndexesInitial[dev_id]);
        cudaFree(dknn[dev_id]);
    }*/

    /*FILE* file = fopen("knnBugFreeQ.bin","wb");
    fwrite(knn, sizeof(int), k * pointsCountQuery, file);
    fclose(file);

    std::vector<float> knnd(k * pointsCountQuery);
    cudaMemcpy(knnd.data(), dknnDist, k * pointsCountQuery * sizeof(float), cudaMemcpyDeviceToHost);
    file = fopen("knnBugFreeQDistancesQ.bin","wb");
    fwrite(knnd.data(), sizeof(float), k * pointsCountQuery, file);
    fclose(file);*/
}