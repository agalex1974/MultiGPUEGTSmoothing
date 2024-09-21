#include <chrono>
#include <omp.h>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/system/omp/execution_policy.h>
#include <thrust/system/omp/vector.h>
#include <thrust/unique.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <algorithm>
#include <string>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
//#include "CGALHeaders.h"

using long64 = long long;

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

template <typename T>
__global__
void setValue(T* vector, T value, int count){
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < count){
        vector[i] = value;
    }
}

__device__
int binarySearchFloat(float value, float a, float h, int bucketsCount)
{
	int lower = 0;
	int upper = bucketsCount;
	int middle = (lower + upper) / 2;
	while (middle > lower)
	{
		if (a + middle * h > value) upper = middle;
		else lower = middle;
		middle = (lower + upper) / 2;
	}
	return middle;
}

template <typename T>
__global__
void get_buckets(float* pnts_x, float* pnts_y, float* pnts_z, int* bucket_indexes_x, int* bucket_indexes_y, int* bucket_indexes_z,
                 T* bucket_indexes, int pointsCount, int bucketsCount, float a, float b)
{
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	float h = (b - a) / bucketsCount;
	if (i < pointsCount)
	{
		float x = pnts_x[i];
		float y = pnts_y[i];
		float z = pnts_z[i];
		int xi = binarySearchFloat(x, a, h, bucketsCount);
		int yi = binarySearchFloat(y, a, h, bucketsCount);
		int zi = binarySearchFloat(z, a, h, bucketsCount);
		bucket_indexes_x[i] = xi;
		bucket_indexes_y[i] = yi;
		bucket_indexes_z[i] = zi;
		bucket_indexes[i] = xi + yi * bucketsCount + zi * bucketsCount * bucketsCount;
	}
}

__device__ __host__
float EucledianDistance(int i, int j, float* pnts_x, float* pnts_y, float* pnts_z)
{
	float xi = pnts_x[i];
	float yi = pnts_y[i];
	float zi = pnts_z[i];
	float xj = pnts_x[j];
	float yj = pnts_y[j];
	float zj = pnts_z[j];
	return (xi - xj) * (xi - xj) + (yi - yj) * (yi - yj) + (zi - zj) * (zi - zj);
}

__device__
void GetMaximumGrid(float* knnDistances, float& maximum, int& index, int indexReference, int k)
{
	maximum = knnDistances[k * indexReference];
	index = 0;
	for (int i = 1; i < k; i++)
	{
		float dist = knnDistances[indexReference * k + i];
		if (maximum < dist)
		{
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
bool appendDistanceGrid(float* knnDistances, int* knn, float distQuery, int idxNeighbor, int indexReference, int k, int pointCount, int counter,
	float& maximum, int& indexmaximum, int offset)
{
    indexReference -= offset;
	if (counter < k)
	{
		bool found = false;
		for (int i = 0; i < counter; i++)
		{
			if (knn[i + k * indexReference] == idxNeighbor) {
				found = true;
				break;
			}
		}

		if (!found) {
			int pos = 0;
			if (counter > 0) {
				pos = FindPosition(knnDistances + k * indexReference, distQuery, 0, counter - 1);
				for (int i = counter - 1; i >= pos; i--)
				{
					knnDistances[i + 1 + k * indexReference] = knnDistances[i + k * indexReference];
					knn[i + 1 + k * indexReference] = knn[i + k * indexReference];
				}
			}
			knnDistances[pos + k * indexReference] = distQuery;
			knn[pos + k * indexReference] = idxNeighbor;
			if (counter == k - 1)
			{
				maximum = knnDistances[k - 1 + k * indexReference];
				indexmaximum = knn[k - 1 + k * indexReference];
			}
			return true;
		}
		return false;
	}
	else {
		if (distQuery < knnDistances[k - 1 + k * indexReference]) {
			bool found = false;
			for (int i = 0; i < k; i++)
			{
				if (knn[i + k * indexReference] == idxNeighbor) {
					found = true;
					break;
				}
			}
			if (!found) {
				int pos = FindPosition(knnDistances + k * indexReference, distQuery, 0, k - 1);
				for (int i = k - 2; i >= pos; i--)
				{
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
void add2vector(int* dvector, int* dtoaddvector, int pointsCount)
{
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < pointsCount)
	{
		dvector[i] |= dtoaddvector[i];
	}
}

__global__
void minVector(float* ddestMin, float* dsourceMin, int* ddestIdx, int* dsourceIdx, int pointsCount)
{
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < pointsCount)
    {
        if (ddestMin[i] > dsourceMin[i]) {
            ddestMin[i] = dsourceMin[i];
            ddestIdx[i] = dsourceIdx[i];
        }
    }
}

__global__
void getMaximumRadiusIndex(float* knnDistances, int* knnIndexes, int* indexesQuery, int* continues,
                           int* counters,
                           float* maxRadius, int* maxRadiusIdx, int pointsLeft, int k){
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < pointsLeft) {
        if (continues[i]) {
            int realIdxQuery = indexesQuery[i];
            int n = counters[i];
            n = n >= k ? k - 1 : n - 1;
            maxRadius[i] = knnDistances[realIdxQuery * k + n];
            maxRadiusIdx[i] = knnIndexes[realIdxQuery * k + n];
        }
    }
}

__global__
void checkIfContinue(float* pnts_x, float* pnts_y, float* pnts_z, int* bucketIndexes_x, int* bucketIndexes_y, int* bucketIndexes_z,
                           int* indexesQuery, int* continues, float* maxRadius, int pointsLeft, int offset, float h, float xmin, int bucketCount){
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < pointsLeft) {
        if (!continues[i]) {
            float searchMore = false;
            int realIdxQuery = indexesQuery[i];
            int xi = bucketIndexes_x[realIdxQuery];
            int yi = bucketIndexes_y[realIdxQuery];
            int zi = bucketIndexes_z[realIdxQuery];
            float maximumRadius = maxRadius[i];
            float x = pnts_x[realIdxQuery];
            float y = pnts_y[realIdxQuery];
            float z = pnts_z[realIdxQuery];

            float ext = x - xmin - (xi - offset) * h;
            ext *= ext;
            if (xi - offset >= 0 && maximumRadius >= ext) searchMore = true;
            else {
                ext = xmin - x + (xi + offset + 1) * h;
                ext *= ext;
                if (xi + offset + 1 <= bucketCount && maximumRadius >= ext) searchMore = true;
                else {
                    ext = y - xmin - (yi - offset) * h;
                    ext *= ext;
                    if (yi - offset >= 0 && maximumRadius >= ext) searchMore = true;
                    else {
                        ext = xmin - y + (yi + offset + 1) * h;
                        ext *= ext;
                        if (yi + offset + 1 <= bucketCount && maximumRadius >= ext) searchMore = true;
                        else {
                            ext = z - xmin - (zi - offset) * h;
                            ext *= ext;
                            if (zi - offset >= 0 && maximumRadius >= ext) searchMore = true;
                            else {
                                ext = xmin - z + (zi + offset + 1) * h;
                                ext *= ext;
                                if (zi + offset + 1 <= bucketCount && maximumRadius >= ext) searchMore = true;
                            }
                        }
                    }
                }
            }
            continues[i] = searchMore;
        }
    }
}

template <typename T>
__global__
void construct_neighbors(float* pnts_x, float* pnts_y, float* pnts_z,
                         int* pointBucketIndexes, int pointBucketCount,
                         T* bucketIndexes,
                         int* bucketIndexes_x, int* bucketIndexes_y, int* bucketIndexes_z,
                         int pointsCount, float* knnDistances, int* knn, int bucketCount, int k,
                         int offset, int slice, int* counters, int* indexesQuery, int* indexes,
                         float* currentRadius, int* indexOfMaximumRadius, int* slicecontinue, int ii, int jj, int kk,
                         int pointsLeftCount, float xmin, float h, int start)
{
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < pointsLeftCount)
    {
        int realIdxQuery = indexesQuery[i];
        int counter = counters[i];
        int xi = bucketIndexes_x[realIdxQuery];
        int yi = bucketIndexes_y[realIdxQuery];
        int zi = bucketIndexes_z[realIdxQuery];
        if (offset == 0)
        {
            //start with the bucket that contains
            T bucketIndex = xi + yi * bucketCount + zi * bucketCount * bucketCount;
            int bucketPosition = binarySearchInt<T>(bucketIndexes, bucketIndex, 0, pointBucketCount - 1);
            int startPointIndex = pointBucketIndexes[bucketPosition];
            int endPointIndex = bucketPosition < pointBucketCount - 1 ? pointBucketIndexes[bucketPosition + 1] : pointsCount;
            float maximumRadius = currentRadius[i];
            int radiusIndex = indexOfMaximumRadius[i];
            for (int idx = startPointIndex; idx < endPointIndex; idx++)
            {
                int realIdxNeighbor = indexes[idx];
                if (realIdxQuery != realIdxNeighbor)
                {
                    float dist = EucledianDistance(realIdxQuery, realIdxNeighbor, pnts_x, pnts_y, pnts_z);
                    if(appendDistanceGrid(knnDistances, knn, dist, realIdxNeighbor, realIdxQuery, k, pointsCount,
                                          counter, maximumRadius, radiusIndex, start))
                        counter++;
                }
            }
            counters[i] = counter;
            currentRadius[i] = maximumRadius;
            indexOfMaximumRadius[i] = radiusIndex;
        }
        else
        {
            bool searchMore = false;
            int offsets[2] = { -offset, offset };
            float maximumRadius = currentRadius[i];
            int radiusIndex = indexOfMaximumRadius[i];

            if (slice == 0)
            {
                int xni = xi + ii;
                if (xni >= 0 && xni < bucketCount)
                {
                    int yni = yi + jj;
                    if (yni >= 0 && yni < bucketCount)
                    {
                        int zni = zi + offsets[kk];
                        if (zni >= 0 && zni < bucketCount)
                        {
                            T bucketIndex = xni + yni * bucketCount + zni * bucketCount * bucketCount;
                            int bucketPosition = binarySearchInt<T>(bucketIndexes, bucketIndex, 0, pointBucketCount - 1);
                            if (bucketPosition >= 0) {
                                int startPointIndex = pointBucketIndexes[bucketPosition];
                                int endPointIndex = bucketPosition < pointBucketCount - 1 ? pointBucketIndexes[bucketPosition + 1] : pointsCount;
                                for (int idx = startPointIndex; idx < endPointIndex; idx++)
                                {
                                    int realIdxNeighbor = indexes[idx];
                                    float dist = EucledianDistance(realIdxQuery, realIdxNeighbor, pnts_x, pnts_y, pnts_z);
                                    bool canAdd = appendDistanceGrid(knnDistances, knn, dist, realIdxNeighbor, realIdxQuery,
                                                                     k, pointsCount, counter, maximumRadius, radiusIndex, start);
                                    searchMore |= canAdd;
                                    if (canAdd) counter++;
                                }
                            }
                        }
                    }
                }
            }
            if (slice == 1)
            {
                int xni = xi + ii;
                if (xni >= 0 && xni < bucketCount)
                {
                    int yni = yi + offsets[jj];
                    if (yni >= 0 && yni < bucketCount)
                    {
                        int zni = zi + kk;
                        if (zni >= 0 && zni < bucketCount)
                        {
                            T bucketIndex = xni + yni * bucketCount + zni * bucketCount * bucketCount;
                            int bucketPosition = binarySearchInt<T>(bucketIndexes, bucketIndex, 0, pointBucketCount - 1);
                            if (bucketPosition >= 0) {
                                int startPointIndex = pointBucketIndexes[bucketPosition];
                                int endPointIndex = bucketPosition < pointBucketCount - 1 ? pointBucketIndexes[bucketPosition + 1] : pointsCount;
                                for (int idx = startPointIndex; idx < endPointIndex; idx++)
                                {
                                    int realIdxNeighbor = indexes[idx];
                                    float dist = EucledianDistance(realIdxQuery, realIdxNeighbor, pnts_x, pnts_y, pnts_z);
                                    bool canAdd = appendDistanceGrid(knnDistances, knn, dist, realIdxNeighbor, realIdxQuery, k, pointsCount,
                                                                     counter, maximumRadius, radiusIndex, start);
                                    searchMore |= canAdd;
                                    if (canAdd) counter++;
                                }
                            }
                        }
                    }
                }
            }
            if (slice == 2)
            {
                int xni = xi + offsets[ii];
                if (xni >= 0 && xni < bucketCount)
                {
                    int yni = yi + jj;
                    if (yni >= 0 && yni < bucketCount)
                    {
                        int zni = zi + kk;
                        if (zni >= 0 && zni < bucketCount)
                        {
                            T bucketIndex = xni + yni * bucketCount + zni * bucketCount * bucketCount;
                            int bucketPosition = binarySearchInt<T>(bucketIndexes, bucketIndex, 0, pointBucketCount - 1);
                            if (bucketPosition >= 0) {
                                int startPointIndex = pointBucketIndexes[bucketPosition];
                                int endPointIndex = bucketPosition < pointBucketCount - 1 ? pointBucketIndexes[bucketPosition + 1] : pointsCount;
                                for (int idx = startPointIndex; idx < endPointIndex; idx++)
                                {
                                    int realIdxNeighbor = indexes[idx];
                                    float dist = EucledianDistance(realIdxQuery, realIdxNeighbor, pnts_x, pnts_y, pnts_z);
                                    bool canAdd = appendDistanceGrid(knnDistances, knn, dist, realIdxNeighbor, realIdxQuery, k,
                                                                     pointsCount, counter, maximumRadius, radiusIndex, start);
                                    searchMore |= canAdd;
                                    if (canAdd) counter++;
                                }
                            }
                        }
                    }
                }
            }
            if (slice == 3)
            {
                int xni = xi + offsets[ii];
                if (xni >= 0 && xni < bucketCount)
                {
                    int yni = yi + offsets[jj];
                    if (yni >= 0 && yni < bucketCount)
                    {
                        int zni = zi + kk;
                        if (zni >= 0 && zni < bucketCount)
                        {
                            T bucketIndex = xni + yni * bucketCount + zni * bucketCount * bucketCount;
                            int bucketPosition = binarySearchInt<T>(bucketIndexes, bucketIndex, 0, pointBucketCount - 1);
                            if (bucketPosition >= 0) {
                                int startPointIndex = pointBucketIndexes[bucketPosition];
                                int endPointIndex = bucketPosition < pointBucketCount - 1 ? pointBucketIndexes[bucketPosition + 1] : pointsCount;
                                for (int idx = startPointIndex; idx < endPointIndex; idx++)
                                {
                                    int realIdxNeighbor = indexes[idx];
                                    float dist = EucledianDistance(realIdxQuery, realIdxNeighbor, pnts_x, pnts_y, pnts_z);
                                    bool canAdd = appendDistanceGrid(knnDistances, knn, dist, realIdxNeighbor, realIdxQuery, k,
                                                                     pointsCount, counter, maximumRadius, radiusIndex, start);
                                    searchMore |= canAdd;
                                    if (canAdd) counter++;
                                }
                            }
                        }
                    }
                }
            }
            if (slice == 4)
            {
                int xni = xi + ii;
                if (xni >= 0 && xni < bucketCount)
                {
                    int yni = yi + offsets[jj];
                    if (yni >= 0 && yni < bucketCount)
                    {
                        int zni = zi + offsets[kk];
                        if (zni >= 0 && zni < bucketCount)
                        {
                            T bucketIndex = xni + yni * bucketCount + zni * bucketCount * bucketCount;
                            int bucketPosition = binarySearchInt<T>(bucketIndexes, bucketIndex, 0, pointBucketCount - 1);
                            if (bucketPosition >= 0) {
                                int startPointIndex = pointBucketIndexes[bucketPosition];
                                int endPointIndex = bucketPosition < pointBucketCount - 1 ? pointBucketIndexes[bucketPosition + 1] : pointsCount;
                                for (int idx = startPointIndex; idx < endPointIndex; idx++)
                                {
                                    int realIdxNeighbor = indexes[idx];
                                    float dist = EucledianDistance(realIdxQuery, realIdxNeighbor, pnts_x, pnts_y, pnts_z);
                                    bool canAdd = appendDistanceGrid(knnDistances, knn, dist, realIdxNeighbor, realIdxQuery, k, pointsCount,
                                                                     counter, maximumRadius, radiusIndex, start);
                                    searchMore |= canAdd;
                                    if (canAdd) counter++;
                                }
                            }
                        }
                    }
                }
            }
            if (slice == 5)
            {
                int xni = xi + offsets[ii];
                if (xni >= 0 && xni < bucketCount)
                {
                    int yni = yi + jj;
                    if (yni >= 0 && yni < bucketCount)
                    {
                        int zni = zi + offsets[kk];
                        if (zni >= 0 && zni < bucketCount)
                        {
                            T bucketIndex = xni + yni * bucketCount + zni * bucketCount * bucketCount;
                            int bucketPosition = binarySearchInt<T>(bucketIndexes, bucketIndex, 0, pointBucketCount - 1);
                            if (bucketPosition >= 0) {
                                int startPointIndex = pointBucketIndexes[bucketPosition];
                                int endPointIndex = bucketPosition < pointBucketCount - 1 ? pointBucketIndexes[bucketPosition + 1] : pointsCount;
                                for (int idx = startPointIndex; idx < endPointIndex; idx++)
                                {
                                    int realIdxNeighbor = indexes[idx];
                                    float dist = EucledianDistance(realIdxQuery, realIdxNeighbor, pnts_x, pnts_y, pnts_z);
                                    bool canAdd = appendDistanceGrid(knnDistances, knn, dist, realIdxNeighbor, realIdxQuery, k,
                                                                     pointsCount, counter, maximumRadius, radiusIndex, start);
                                    searchMore |= canAdd;
                                    if (canAdd) counter++;
                                }
                            }
                        }
                    }
                }
                if (ii == 1 && jj == offset - 1 && kk == 1) {
                    if (counter < k) searchMore = true;
                    else {
                        float x = pnts_x[realIdxQuery];
                        float y = pnts_y[realIdxQuery];
                        float z = pnts_z[realIdxQuery];
                        if (!searchMore) {
                            float ext = x - xmin - (xi - offset) * h;
                            ext *= ext;
                            if (xi - offset >= 0 && maximumRadius >= ext)  searchMore = true;
                            else {
                                ext = xmin - x + (xi + offset + 1) * h;
                                ext *= ext;
                                if (xi + offset + 1 <= bucketCount && maximumRadius >= ext) searchMore = true;
                                else {
                                    ext = y - xmin - (yi - offset) * h;
                                    ext *= ext;
                                    if (yi - offset >= 0 && maximumRadius >= ext) searchMore = true;
                                    else {
                                        ext = xmin - y + (yi + offset + 1) * h;
                                        ext *= ext;
                                        if (yi + offset + 1 <= bucketCount && maximumRadius >= ext) searchMore = true;
                                        else {
                                            ext = z - xmin - (zi - offset) * h;
                                            ext *= ext;
                                            if (zi - offset >= 0 && maximumRadius >= ext) searchMore = true;
                                            else {
                                                ext = xmin - z + (zi + offset + 1) * h;
                                                ext *= ext;
                                                if (zi + offset + 1 <= bucketCount && maximumRadius >= ext) searchMore = true;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            slicecontinue[i] = searchMore;
            currentRadius[i] = maximumRadius;
            indexOfMaximumRadius[i] = radiusIndex;
            counters[i] = counter;
        }
    }
}

__global__
void nomralize_distances_grid(float* knnDistances, float max, float min, int counter) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < counter)
	{
		knnDistances[i] = (knnDistances[i] - min) / (max - min);
	}
}

__global__
void addMaximum_grid(float* knnDistances, float max, int pointsCount, int k)
{
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < pointsCount)
	{
		float addon = i * max;
		for (int kk = 0; kk < k; kk++)
		{
			knnDistances[i * k + kk] += addon;
		}
	}
}

// This naive transpose kernel suffers from completely non-coalesced writes.
// It can be up to 10x slower than the kernel above for large matrices.
__global__ void transpose_naive(int* odata, int* idata, int width, int height)
{
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

	if (xIndex < width && yIndex < height)
	{
		unsigned int index_in = xIndex + width * yIndex;
		unsigned int index_out = yIndex + height * xIndex;
		odata[index_out] = idata[index_in];
	}
}

// This naive transpose kernel suffers from completely non-coalesced writes.
// It can be up to 10x slower than the kernel above for large matrices.
__global__ void transpose1(int* odata, int* idata, int pointCount, int k)
{
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < pointCount)
	{
		for (int kk = 0; kk < k; kk++)
			odata[i + kk * pointCount] = idata[i * k + kk];
	}
}

__global__ void getNewData(int* scannedIndexes, int* continues,
	int* indexes, int* newindexes,
	float* currentRadius, float* currentNewRadius,
	int* indexOfMaximumRadius, int* indexOfnewMaximumRadius,
	int* counters, int* newCounters,
	int pointsCount)
{
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < pointsCount)
	{
		if (continues[i])
		{
			int idx = scannedIndexes[i] - 1;
			newindexes[idx] = indexes[i];
			currentNewRadius[idx] = currentRadius[i];
			indexOfnewMaximumRadius[idx] = indexOfMaximumRadius[i];
			newCounters[idx] = counters[i];
		}
	}
}

template <typename T>
int getBuckets(float* dpoints_x, float* dpoints_y, float* dpoints_z, int pointsCount,
               int* dbucketIndexes_x, int* dbucketIndexes_y, int* dbucketIndexes_z, T* dbucketIndexes, int bucketsCount,
			   int* dindexes, int*& dpointBucketIndexes, float min, float max, int dev_id)
{
	cudaSetDevice(dev_id);
	int threads = 1024;
	int blocks = (int)ceil((1.0 * pointsCount) / threads);
	get_buckets<T><<<blocks, threads>>>(dpoints_x, dpoints_y, dpoints_z,
                                        dbucketIndexes_x, dbucketIndexes_y, dbucketIndexes_z, dbucketIndexes,
                                        pointsCount, bucketsCount, min, max);
	cudaDeviceSynchronize();

	int* indexes = new int[pointsCount];
	for (int i = 0; i < pointsCount; i++) indexes[i] = i;
	cudaMemcpy(dindexes, indexes, pointsCount * sizeof(int), cudaMemcpyHostToDevice);
	delete[] indexes;

	cudaMalloc((void**)&dpointBucketIndexes, pointsCount * sizeof(int));
	cudaMemcpy(dpointBucketIndexes, dindexes, pointsCount * sizeof(int), cudaMemcpyDeviceToDevice);

	thrust::sort_by_key(thrust::device, dbucketIndexes, dbucketIndexes + pointsCount, dindexes, 
		thrust::less<T>());

	thrust::pair<T*, int*> end;
	end = thrust::unique_by_key(thrust::device, dbucketIndexes, dbucketIndexes + pointsCount, 
		dpointBucketIndexes, thrust::equal_to<T>());
	T vertexPartitionSize = end.first - dbucketIndexes;
	return vertexPartitionSize;
}


__global__
void normalize_points(float* x, float* y, float* z, float min, float max, int pointCount)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < pointCount) {
		float d = max - min;
		x[i] = (x[i] - min) / d;
		y[i] = (y[i] - min) / d;
		z[i] = (z[i] - min) / d;
	}
}

__global__
void denormalize_points(float* x, float* y, float* z, float min, float max, int pointCount)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < pointCount) {
		float d = max - min;
		x[i] = x[i] * d + min;
		y[i] = y[i] * d + min;
		z[i] = z[i] * d + min;
	}
}

__device__
void mergeArrays(float* arr1Dist, float* arr2Dist,
                 int* arr1Idx, int* arr2Idx,
                 int n1, int n2, int& n3, int k,
                 float* arr3Dist, int* arr3Idx)
{
    int i = 0, j = 0, l = 0;

    while (i < n1 && j < n2 && l < k)
    {
        if (arr1Idx[i] == arr2Idx[j]) {
            arr3Dist[l] = arr2Dist[j];
            arr3Idx[l++] = arr2Idx[j++];
            i++;
        }
        else if (arr1Dist[i] < arr2Dist[j] ) {
            arr3Dist[l] = arr1Dist[i];
            arr3Idx[l++] = arr1Idx[i++];
        }
        else{
            arr3Dist[l] = arr2Dist[j];
            arr3Idx[l++] = arr2Idx[j++];
        }
    }
    // Store remaining elements of first array
    while (i < n1 && l < k) {
        arr3Dist[l] = arr1Dist[i];
        arr3Idx[l++] = arr1Idx[i++];
    }

    // Store remaining elements of second array
    while (j < n2 && l < k) {
        arr3Dist[l] = arr2Dist[j];
        arr3Idx[l++] = arr2Idx[j++];
    }
    n3 = l;
}

__global__
void mergeNeighbors(float* knnDistances1, float* knnDistances2, float* knnDistances3,
                    int* knnIndexes1, int* knnIndexes2, int* knnIndexes3,
                    int* continues,
                    int* counters1, int* counters2, int* counters3,
                    int* indexes, int pointsleft, int k){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < pointsleft) {
        if (continues[i]){
            int idx = indexes[i];
            int n1 = counters1[i];
            int n2 = counters2[i];
            int n3;
            mergeArrays(knnDistances1 + idx * k, knnDistances2 + idx * k,
                        knnIndexes1 + idx * k, knnIndexes2 + idx * k, n1, n2, n3, k,
                        knnDistances3 + idx * k, knnIndexes3 + idx * k);
            counters3[i] = n3;
        }
    }
}

__global__
void copyNeighbors(int* knnDest, int* knnSource, int* indexes, int k, int pointsCount){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < pointsCount) {
        int idx = indexes[i];
        memcpy(knnDest + idx * k, knnSource + idx * k, k * sizeof(int));
    }
}

__global__
void reverseIndexNeighbors(int* knn, int* indexes, int indexesCount){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < indexesCount){
        int neigh = knn[i];
        if (neigh != -1)
            knn[i] = indexes[neigh];
    }
}

__global__
void rearrangeNeighbors(int* knnDest, int* knnSource, int* indexes, int* reverseIndexes, int k, int pointsCount){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < pointsCount) {
        int idx = reverseIndexes[i];
        //memcpy(knnDest + i * k, knnSource + idx * k, k * sizeof(int));
        for (int kk = 0; kk < k; kk++){
            knnDest[i * k + kk] = indexes[knnSource[idx * k + kk]];
        }
    }
}

__global__
void rearrangePointsBuckets(float* pnts_x_dest, float* pnts_y_dest, float* pnts_z_dest,
                            float* pnts_x, float* pnts_y, float* pnts_z,
                            int* buckets_x_dest, int* buckets_y_dest, int* buckets_z_dest,
                            int* buckets_x, int* buckets_y, int* buckets_z,
                            int* indexes, int pointsCount){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < pointsCount) {
        int idx = indexes[i];
        pnts_x_dest[i] = pnts_x[idx];
        pnts_y_dest[i] = pnts_y[idx];
        pnts_z_dest[i] = pnts_z[idx];
        buckets_x_dest[i] = buckets_x[idx];
        buckets_y_dest[i] = buckets_y[idx];
        buckets_z_dest[i] = buckets_z[idx];
    }
}

__global__
void rearrangePoints(float* pnts_x_dest, float* pnts_y_dest, float* pnts_z_dest,
                            float* pnts_x, float* pnts_y, float* pnts_z,
                            int* indexes, int pointsCount){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < pointsCount) {
        int idx = indexes[i];
        pnts_x_dest[i] = pnts_x[idx];
        pnts_y_dest[i] = pnts_y[idx];
        pnts_z_dest[i] = pnts_z[idx];
    }
}

__global__
void get_densities(float* densities, float* ordered_coordinates, int count) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < count){
        densities[i] = abs(ordered_coordinates[i + 1] - ordered_coordinates[i]);
    }
}

void GRIDCUDAKNN(float* points_x, float* points_y, float* points_z, int pointsCount, int k,
                 float**& dpoints_x, float**& dpoints_y, float**& dpoints_z, int**& dknn, int num_cards,
                 int*& reverseIndexes, int*& pointsCard, int& pointPartitionSize, float& minOriginal, float& maxOriginal,
                 float* x, float* y, float* z, int* knn)
{
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	int** dcontinues = (int**)malloc(num_cards * sizeof(int*));
	int** dIndexesQuery = (int**)malloc(num_cards * sizeof(int*));
	int** dIndexesOriginal = (int**)malloc(num_cards * sizeof(int*));
    int** dIndexesReverseOriginal = (int**)malloc(num_cards * sizeof(int*));
    int** dbucketIndexes_x = (int**)malloc(num_cards * sizeof(int*));
	int** dbucketIndexes_y = (int**)malloc(num_cards * sizeof(int*));
	int** dbucketIndexes_z = (int**)malloc(num_cards * sizeof(int*));
	long64** dbucketIndexes = (long64**)malloc(num_cards * sizeof(long64*));
	int** dindexes = (int**)malloc(num_cards * sizeof(int*));

	int** dpointBucketIndexes = (int**)malloc(num_cards * sizeof(int*));
	int** dcounters = (int**)malloc(num_cards * sizeof(int*));
	int** dslicecontinue = (int**)malloc(num_cards * sizeof(int*));

    ////// This is for the sphere of proximity //////////////////////////////
    float** dCurrentRadius = (float**)malloc(num_cards * sizeof(float*));
    dpoints_x = (float**)malloc(num_cards * sizeof(float*));
    dpoints_y = (float**)malloc(num_cards * sizeof(float*));
    dpoints_z = (float**)malloc(num_cards * sizeof(float*));
    int** dindexesMaximum = (int**)malloc(num_cards * sizeof(int*));
    /////////////////////////////////////////////////////////////////////////
    float** dknndistances = (float**)malloc(num_cards * sizeof(float*));

    dknn = (int**)malloc(num_cards * sizeof(int*));

	float** dRadiusNew = (float**)malloc(num_cards * sizeof(float*));
	int** dRadiusIndexNew = (int**)malloc(num_cards * sizeof(int*));
	int** dCountersNew = (int**)malloc(num_cards * sizeof(int*));
	int** dIndexesNew = (int**)malloc(num_cards * sizeof(int*));
	std::vector<int> bucketsCount(num_cards);
	//float maxOriginal;
	//float minOriginal;
	for (int dev_id = 0; dev_id < num_cards; dev_id++)
	{
		dRadiusNew[dev_id] = nullptr;
		dRadiusIndexNew[dev_id] = nullptr;
		dCountersNew[dev_id] = nullptr;
		dIndexesNew[dev_id] = nullptr;
	}
	std::vector<int> extent;
	int* vertexPartitionSize = (int*)malloc(num_cards * sizeof(int));
    pointPartitionSize = pointsCount / num_cards;
    pointsCard = (int*)malloc(num_cards * sizeof(int));
    for (int i = 0; i < num_cards - 1; i++){
        pointsCard[i] = pointPartitionSize;
    }
    pointsCard[num_cards - 1] = pointsCount - (num_cards - 1) * pointPartitionSize;

#pragma omp parallel for num_threads(num_cards)
	for (int dev_id = 0; dev_id < num_cards; dev_id++){
		cudaSetDevice(dev_id);
        bucketsCount[dev_id] = 1024;
		cudaMalloc((void**)&dpoints_x[dev_id], pointsCount * sizeof(float));
        float* temp_pnts_x;
        cudaMalloc((void**)&temp_pnts_x, pointsCount * sizeof(float));
        //cudaMemcpy(temp_pnts_x, points_x, pointsCount * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dpoints_x[dev_id], points_x, pointsCount * sizeof(float), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dpoints_y[dev_id], pointsCount * sizeof(float));
        float* temp_pnts_y;
        cudaMalloc((void**)&temp_pnts_y, pointsCount * sizeof(float));
        //cudaMemcpy(temp_pnts_y, points_y, pointsCount * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dpoints_y[dev_id], points_y, pointsCount * sizeof(float), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dpoints_z[dev_id], pointsCount * sizeof(float));
		float* temp_pnts_z;
        cudaMalloc((void**)&temp_pnts_z, pointsCount * sizeof(float));
        //cudaMemcpy(temp_pnts_z, points_z, pointsCount * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dpoints_z[dev_id], points_z, pointsCount * sizeof(float), cudaMemcpyHostToDevice);

		float minx = thrust::reduce(thrust::device, dpoints_x[dev_id], dpoints_x[dev_id] + pointsCount, 1e8f, thrust::minimum<float>()) - 1e-6f;
		float maxx = thrust::reduce(thrust::device, dpoints_x[dev_id], dpoints_x[dev_id] + pointsCount, -1e8f, thrust::maximum<float>()) + 1e-6f;

		float miny = thrust::reduce(thrust::device, dpoints_y[dev_id], dpoints_y[dev_id] + pointsCount, 1e8f, thrust::minimum<float>()) - 1e-6f;
		float maxy = thrust::reduce(thrust::device, dpoints_y[dev_id], dpoints_y[dev_id] + pointsCount, -1e8f, thrust::maximum<float>()) + 1e-6f;

		float minz = thrust::reduce(thrust::device, dpoints_z[dev_id], dpoints_z[dev_id] + pointsCount, 1e8f, thrust::minimum<float>()) - 1e-6f;
		float maxz = thrust::reduce(thrust::device, dpoints_z[dev_id], dpoints_z[dev_id] + pointsCount, -1e8f, thrust::maximum<float>()) + 1e-6f;

		float min = std::min(minx, miny);
		min = std::min(min, minz);

		float max = std::max(maxx, maxy);
		max = std::max(max, maxz);

		int threads = 1024;
		int blocks = (int)ceil((1.0 * pointsCount) / threads);

        normalize_points<<<blocks, threads>>>(dpoints_x[dev_id], dpoints_y[dev_id], dpoints_z[dev_id], min, max, pointsCount);
		cudaDeviceSynchronize();

        cudaMemcpy(temp_pnts_x, dpoints_x[dev_id],  pointsCount * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(temp_pnts_y, dpoints_y[dev_id],  pointsCount * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(temp_pnts_z, dpoints_z[dev_id],  pointsCount * sizeof(float), cudaMemcpyDeviceToDevice);

		if (dev_id == 0) {
			maxOriginal = max;
			minOriginal = min;
		}
		min = -1e-6f;
		max = 1.0 + 1e-6f;

		cudaMalloc((void**)&dbucketIndexes_x[dev_id], pointsCount * sizeof(int));
        int* dbucketIndexes_x_temp;
        cudaMalloc((void**)&dbucketIndexes_x_temp, pointsCount * sizeof(int));
		cudaMalloc((void**)&dbucketIndexes_y[dev_id], pointsCount * sizeof(int));
        int* dbucketIndexes_y_temp;
        cudaMalloc((void**)&dbucketIndexes_y_temp, pointsCount * sizeof(int));
		cudaMalloc((void**)&dbucketIndexes_z[dev_id], pointsCount * sizeof(int));
        int* dbucketIndexes_z_temp;
        cudaMalloc((void**)&dbucketIndexes_z_temp, pointsCount * sizeof(int));
        cudaMalloc((void**)&dbucketIndexes[dev_id], pointsCount * sizeof(long64));
		cudaMalloc((void**)&dindexes[dev_id], pointsCount * sizeof(int));

		vertexPartitionSize[dev_id] = getBuckets<long64>(dpoints_x[dev_id], dpoints_y[dev_id], dpoints_z[dev_id],
			pointsCount, dbucketIndexes_x[dev_id], dbucketIndexes_y[dev_id], dbucketIndexes_z[dev_id],
			dbucketIndexes[dev_id], bucketsCount[dev_id], dindexes[dev_id], dpointBucketIndexes[dev_id], min, max, dev_id);


        rearrangePointsBuckets<<<blocks,threads>>>(temp_pnts_x, temp_pnts_y, temp_pnts_z,
                                                   dpoints_x[dev_id], dpoints_y[dev_id], dpoints_z[dev_id],
                                                   dbucketIndexes_x_temp, dbucketIndexes_y_temp, dbucketIndexes_z_temp,
                                                   dbucketIndexes_x[dev_id], dbucketIndexes_y[dev_id], dbucketIndexes_z[dev_id],
                                                   dindexes[dev_id], pointsCount);
        cudaDeviceSynchronize();
        cudaMemcpy(dpoints_z[dev_id], temp_pnts_z, pointsCount * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dpoints_y[dev_id], temp_pnts_y, pointsCount * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dpoints_x[dev_id], temp_pnts_x, pointsCount * sizeof(float), cudaMemcpyDeviceToDevice);

        cudaMemcpy(dbucketIndexes_x[dev_id], dbucketIndexes_x_temp, pointsCount * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dbucketIndexes_y[dev_id], dbucketIndexes_y_temp, pointsCount * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dbucketIndexes_z[dev_id], dbucketIndexes_z_temp, pointsCount * sizeof(float), cudaMemcpyDeviceToDevice);

        //cudaMalloc((void**)&dIndexesOriginal[dev_id], pointsCount * sizeof(int));
        //cudaMalloc((void**)&dIndexesReverseOriginal[dev_id], pointsCount * sizeof(int));
        //cudaMemcpy(dIndexesOriginal[dev_id], dindexes[dev_id], pointsCount * sizeof(int), cudaMemcpyDeviceToDevice);
        //cudaMemcpy(dIndexesReverseOriginal[dev_id], dindexes[dev_id], pointsCount * sizeof(int), cudaMemcpyDeviceToDevice);

        int* dIndexesReverseOriginal;
        cudaMalloc((void**)&dIndexesReverseOriginal, pointsCount * sizeof(int));
        cudaMemcpy(dIndexesReverseOriginal, dindexes[dev_id], pointsCount * sizeof(int), cudaMemcpyDeviceToDevice);

        int* indexes = new int[pointsCount];
        for (int i = 0; i < pointsCount; i++) indexes[i] = i;
        cudaMemcpy(dindexes[dev_id], indexes, pointsCount * sizeof(int), cudaMemcpyHostToDevice);
        delete[] indexes;

        thrust::sort_by_key(thrust::device, dIndexesReverseOriginal, dIndexesReverseOriginal + pointsCount,
                            dindexes[dev_id]);
        std::swap(dIndexesReverseOriginal, dindexes[dev_id]);

        if (dev_id == 0){
            cudaMalloc((void**)&reverseIndexes, pointsCount * sizeof(int));
            cudaMemcpy(reverseIndexes, dIndexesReverseOriginal, pointsCount * sizeof(int), cudaMemcpyDeviceToDevice);
        }
        cudaFree(dIndexesReverseOriginal);

        /*if (dev_id == 0){
            std::vector<int> indexes(pointsCount);
            std::vector<int> indexPresent(pointsCount, 0);
            cudaMemcpy(indexes.data(), dIndexesOriginal[dev_id], pointsCount * sizeof(int), cudaMemcpyDeviceToHost);
            std::cout << "index 1000" << indexes[1000] << std::endl;
            for (auto idx : indexes){
                indexPresent[idx] = 1;
            }
            std::cout << "Counter:" <<
                thrust::reduce(thrust::host, indexPresent.data(), indexPresent.data() + pointsCount, 0) << std::endl;
            int threads = 1024;
            int blocks = ceil((1.0 * pointsCount) / threads);
            rearrangePoints<<<blocks, threads>>>(temp_pnts_x, temp_pnts_y, temp_pnts_z,
                                                 dpoints_x[0], dpoints_y[0], dpoints_z[0], dIndexesOriginal[0], pointsCount);
            cudaDeviceSynchronize();
            denormalize_points << < blocks, threads >> > (temp_pnts_x, temp_pnts_y, temp_pnts_z, minOriginal, maxOriginal, pointsCount);
            cudaDeviceSynchronize();
            std::vector<float> pnts_x(pointsCount);
            cudaMemcpy(pnts_x.data(), temp_pnts_x, pointsCount * sizeof(float), cudaMemcpyDeviceToHost);
            std::vector<float> pnts_y(pointsCount);
            cudaMemcpy(pnts_y.data(), temp_pnts_y, pointsCount * sizeof(float), cudaMemcpyDeviceToHost);
            std::vector<float> pnts_z(pointsCount);
            cudaMemcpy(pnts_z.data(), temp_pnts_z, pointsCount * sizeof(float), cudaMemcpyDeviceToHost);
            FILE* file = fopen("points_ordered.txt", "w");
            for (int i = 0; i < pointsCount; i++){
                fprintf(file, "%f %f %f\n", pnts_x[i], pnts_y[i], pnts_z[i]);
            }
            fclose(file);
        }*/
        cudaFree(temp_pnts_x);
        cudaFree(temp_pnts_y);
        cudaFree(temp_pnts_z);
        cudaFree(dbucketIndexes_x_temp);
        cudaFree(dbucketIndexes_y_temp);
        cudaFree(dbucketIndexes_z_temp);

        cudaMalloc((void**)&dCurrentRadius[dev_id], pointsCard[dev_id] * sizeof(float));
        cudaMalloc((void**)&dindexesMaximum[dev_id], pointsCard[dev_id] * sizeof(int));
		cudaMalloc((void**)&dcounters[dev_id], pointsCard[dev_id] * sizeof(int));
		cudaMemset(dcounters[dev_id], 0, pointsCard[dev_id] * sizeof(int));
		cudaMalloc((void**)&dcontinues[dev_id], pointsCard[dev_id] * sizeof(int));
		cudaMemset(dcontinues[dev_id], 0, pointsCard[dev_id] * sizeof(int));
		cudaMalloc((void**)&dslicecontinue[dev_id], pointsCard[dev_id] * sizeof(int));
		cudaMemset(dslicecontinue[dev_id], 0, pointsCard[dev_id] * sizeof(int));
        cudaMalloc((void**)&dIndexesQuery[dev_id], pointsCard[dev_id] * sizeof(int));
        cudaMemcpy(dIndexesQuery[dev_id], dindexes[dev_id] + dev_id * pointPartitionSize, pointsCard[dev_id] * sizeof(int), cudaMemcpyDeviceToDevice);
        //cudaMemcpy(dIndexesQuery[dev_id], shuffledIndexes + dev_id * pointPartitionSize, pointsCard[dev_id] * sizeof(int), cudaMemcpyHostToDevice);
        //////////////KNEAREST NEIGHBOURS////////////////////////////////////////////////////
		cudaMalloc((void**)&dknndistances[dev_id], pointsCard[dev_id] * k * sizeof(float));
		cudaMalloc((void**)&dknn[dev_id], pointsCard[dev_id] * k * sizeof(int));
        /////////////////////////////////////////////////////////////////////////////////////

        threads = 1024;
		blocks = ceil((1.0 * k * pointsCard[dev_id] / threads));
        setValue<int><<<blocks, threads>>>(dknn[dev_id], -1, k * pointsCard[dev_id]);
        cudaDeviceSynchronize();

        threads = 1024;
		blocks = ceil((1.0 * pointsCard[dev_id]) / threads);
		int offset = 0;
		float h = (max - min) / bucketsCount[dev_id];
        construct_neighbors<long64><<<blocks, threads>>>(dpoints_x[dev_id], dpoints_y[dev_id], dpoints_z[dev_id],
			dpointBucketIndexes[dev_id], vertexPartitionSize[dev_id], dbucketIndexes[dev_id],
			dbucketIndexes_x[dev_id], dbucketIndexes_y[dev_id], dbucketIndexes_z[dev_id], pointsCount,
			dknndistances[dev_id], dknn[dev_id],
			bucketsCount[dev_id], k, offset, -1, dcounters[dev_id], dIndexesQuery[dev_id], dindexes[dev_id],
			dCurrentRadius[dev_id], dindexesMaximum[dev_id],
			dslicecontinue[dev_id], -1, -1, -1, pointsCard[dev_id], min, h, dev_id * pointPartitionSize);
		cudaDeviceSynchronize();
	}

    float min = -1e-6f;
    float max = 1.0 + 1e-6f;

    int count_global;
    int* pointsLeft_a = (int*)malloc(num_cards * sizeof(int));
    int* bucketsCount_a = (int*)malloc(num_cards * sizeof(int));
    int* offset_a = (int*)malloc(num_cards * sizeof(int));
    for (int i = 0; i < num_cards; i++) {
        pointsLeft_a[i] = pointsCard[i];
        bucketsCount_a[i] = bucketsCount[i];
        offset_a[i] = 1;
    }
#pragma omp parallel for num_threads(num_cards)
    for (int dev_id = 0; dev_id < num_cards; dev_id++) {
        cudaSetDevice(dev_id);
        int &offset = offset_a[dev_id];
        int &pointsLeft = pointsLeft_a[dev_id];
        int &bucketsCount = bucketsCount_a[dev_id];
        float h;
        int count;
        int iteration = 0;
        do {
            if (offset > 1 && bucketsCount > 32) {
                bucketsCount /= 2;
                h = (max - min) / bucketsCount;
                cudaFree(dpointBucketIndexes[dev_id]);
                vertexPartitionSize[dev_id] = getBuckets<long64>(dpoints_x[dev_id], dpoints_y[dev_id],
                                                                 dpoints_z[dev_id], pointsCount,
                                                                 dbucketIndexes_x[dev_id],
                                                                 dbucketIndexes_y[dev_id], dbucketIndexes_z[dev_id],
                                                                 dbucketIndexes[dev_id], bucketsCount, dindexes[dev_id],
                                                                 dpointBucketIndexes[dev_id], min, max, dev_id);
                offset = 0;
                offset++;
            }
            int threads, blocks;
            threads = 1024;
            blocks = ceil((1.0 * pointsLeft) / threads);
            //std::cout << offset << std::endl;
            cudaMemset(dcontinues[dev_id], 0, pointsLeft * sizeof(int));
            for (int slice = 0; slice < 6; slice++) {
                if (slice == 0) {
                    for (int ii = -offset + 1; ii <= offset - 1; ii++)
                    {
                        for (int jj = -offset + 1; jj <= offset - 1; jj++)
                        {
                            for (int kk = 0; kk < 2; kk++)
                            {
                                cudaMemset(dslicecontinue[dev_id], 0, pointsLeft * sizeof(int));
                                construct_neighbors<long64><<<blocks, threads>>>(dpoints_x[dev_id], dpoints_y[dev_id],
                                        dpoints_z[dev_id], dpointBucketIndexes[dev_id], vertexPartitionSize[dev_id],
                                        dbucketIndexes[dev_id],
                                        dbucketIndexes_x[dev_id], dbucketIndexes_y[dev_id],
                                        dbucketIndexes_z[dev_id], pointsCount, dknndistances[dev_id], dknn[dev_id],
                                        bucketsCount, k, offset, slice, dcounters[dev_id], dIndexesQuery[dev_id],
                                        dindexes[dev_id], dCurrentRadius[dev_id],
                                        dindexesMaximum[dev_id], dslicecontinue[dev_id], ii, jj, kk, pointsLeft, min, h, dev_id * pointPartitionSize);
                                cudaDeviceSynchronize();
                                add2vector<<<blocks, threads>>>(dcontinues[dev_id], dslicecontinue[dev_id], pointsLeft);
                                cudaDeviceSynchronize();
                            }
                        }
                    }
                }
                if (slice == 1) {
                    for (int ii = -offset + 1; ii <= offset - 1; ii++)
                    {
                        for (int kk = -offset + 1; kk <= offset - 1; kk++)
                        {
                            for (int jj = 0; jj < 2; jj++)
                            {
                                cudaMemset(dslicecontinue[dev_id], 0, pointsLeft * sizeof(int));
                                construct_neighbors<long64><<<blocks, threads>>>(dpoints_x[dev_id], dpoints_y[dev_id],
                                        dpoints_z[dev_id], dpointBucketIndexes[dev_id], vertexPartitionSize[dev_id],
                                        dbucketIndexes[dev_id],
                                        dbucketIndexes_x[dev_id], dbucketIndexes_y[dev_id],
                                        dbucketIndexes_z[dev_id], pointsCount, dknndistances[dev_id], dknn[dev_id],
                                        bucketsCount, k, offset, slice, dcounters[dev_id], dIndexesQuery[dev_id],
                                        dindexes[dev_id], dCurrentRadius[dev_id],
                                        dindexesMaximum[dev_id], dslicecontinue[dev_id], ii, jj, kk, pointsLeft, min, h,
                                        dev_id * pointPartitionSize);
                                cudaDeviceSynchronize();
                                add2vector<<<blocks, threads>>>(dcontinues[dev_id], dslicecontinue[dev_id], pointsLeft);
                                cudaDeviceSynchronize();
                            }
                        }
                    }
                }
                if (slice == 3) {
                    for (int kk = -offset; kk <= offset; kk++)
                    {
                        for (int jj = 0; jj < 2; jj++)
                        {
                            for (int ii = 0; ii < 2; ii++)
                            {
                                cudaMemset(dslicecontinue[dev_id], 0, pointsLeft * sizeof(int));
                                construct_neighbors<long64><<<blocks, threads>>>(dpoints_x[dev_id], dpoints_y[dev_id],
                                        dpoints_z[dev_id], dpointBucketIndexes[dev_id], vertexPartitionSize[dev_id],
                                        dbucketIndexes[dev_id],
                                        dbucketIndexes_x[dev_id], dbucketIndexes_y[dev_id],
                                        dbucketIndexes_z[dev_id], pointsCount, dknndistances[dev_id], dknn[dev_id],
                                        bucketsCount, k, offset, slice, dcounters[dev_id], dIndexesQuery[dev_id],
                                        dindexes[dev_id], dCurrentRadius[dev_id],
                                        dindexesMaximum[dev_id], dslicecontinue[dev_id], ii, jj, kk, pointsLeft, min, h,
                                        dev_id * pointPartitionSize);
                                cudaDeviceSynchronize();
                                add2vector<<<blocks, threads>>>(dcontinues[dev_id], dslicecontinue[dev_id], pointsLeft);
                                cudaDeviceSynchronize();
                            }
                        }
                    }
                }
                if (slice == 2) {
                    for (int kk = -offset + 1; kk <= offset - 1; kk++)
                    {
                        for (int jj = -offset + 1; jj <= offset - 1; jj++)
                        {
                            for (int ii = 0; ii < 2; ii++)
                            {
                                cudaMemset(dslicecontinue[dev_id], 0, pointsLeft * sizeof(int));
                                construct_neighbors<long64><<<blocks, threads>>>(dpoints_x[dev_id], dpoints_y[dev_id],
                                        dpoints_z[dev_id], dpointBucketIndexes[dev_id], vertexPartitionSize[dev_id],
                                        dbucketIndexes[dev_id],
                                        dbucketIndexes_x[dev_id], dbucketIndexes_y[dev_id],
                                        dbucketIndexes_z[dev_id], pointsCount, dknndistances[dev_id], dknn[dev_id],
                                        bucketsCount, k, offset, slice, dcounters[dev_id], dIndexesQuery[dev_id],
                                        dindexes[dev_id], dCurrentRadius[dev_id],
                                        dindexesMaximum[dev_id], dslicecontinue[dev_id], ii, jj, kk, pointsLeft, min, h,
                                        dev_id * pointPartitionSize);
                                cudaDeviceSynchronize();
                                add2vector<<<blocks, threads>>>(dcontinues[dev_id], dslicecontinue[dev_id], pointsLeft);
                                cudaDeviceSynchronize();
                            }
                        }
                    }
                }
                if (slice == 4) {
                    for (int kk = 0; kk < 2; kk++)
                    {
                        for (int jj = 0; jj < 2; jj++)
                        {
                            for (int ii = -offset + 1; ii <= offset - 1; ii++)
                            {
                                cudaMemset(dslicecontinue[dev_id], 0, pointsLeft * sizeof(int));
                                construct_neighbors<long64> << <
                                blocks, threads >> > (dpoints_x[dev_id], dpoints_y[dev_id],
                                        dpoints_z[dev_id], dpointBucketIndexes[dev_id], vertexPartitionSize[dev_id],
                                        dbucketIndexes[dev_id],
                                        dbucketIndexes_x[dev_id], dbucketIndexes_y[dev_id],
                                        dbucketIndexes_z[dev_id], pointsCount, dknndistances[dev_id], dknn[dev_id],
                                        bucketsCount, k, offset, slice, dcounters[dev_id], dIndexesQuery[dev_id],
                                        dindexes[dev_id], dCurrentRadius[dev_id],
                                        dindexesMaximum[dev_id], dslicecontinue[dev_id], ii, jj, kk, pointsLeft, min, h,
                                        dev_id * pointPartitionSize);
                                cudaDeviceSynchronize();
                                add2vector << <
                                blocks, threads >> > (dcontinues[dev_id], dslicecontinue[dev_id], pointsLeft);
                                cudaDeviceSynchronize();
                            }
                        }
                    }
                }
                if (slice == 5) {
                    for (int kk = 0; kk < 2; kk++)
                    {
                        for (int jj = -offset + 1; jj <= offset - 1; jj++)
                        {
                            for (int ii = 0; ii < 2; ii++)
                            {
                                cudaMemset(dslicecontinue[dev_id], 0, pointsLeft * sizeof(int));
                                construct_neighbors<long64> << <
                                blocks, threads >> > (dpoints_x[dev_id], dpoints_y[dev_id],
                                        dpoints_z[dev_id], dpointBucketIndexes[dev_id], vertexPartitionSize[dev_id],
                                        dbucketIndexes[dev_id],
                                        dbucketIndexes_x[dev_id], dbucketIndexes_y[dev_id],
                                        dbucketIndexes_z[dev_id], pointsCount, dknndistances[dev_id], dknn[dev_id],
                                        bucketsCount, k, offset, slice, dcounters[dev_id], dIndexesQuery[dev_id],
                                        dindexes[dev_id], dCurrentRadius[dev_id],
                                        dindexesMaximum[dev_id], dslicecontinue[dev_id], ii, jj, kk, pointsLeft, min, h,
                                        dev_id * pointPartitionSize);
                                cudaDeviceSynchronize();
                                add2vector << <
                                blocks, threads >> > (dcontinues[dev_id], dslicecontinue[dev_id], pointsLeft);
                                cudaDeviceSynchronize();
                            }
                        }
                    }
                }
            }

            offset++;

            count = thrust::reduce(thrust::device, dcontinues[dev_id], dcontinues[dev_id] + pointsLeft, 0);
#pragma omp critical
            {
                std::cout << "iteration:" << iteration++ << " dev_id:" << dev_id <<  " counter:" << count << std::endl;
            }
            if (count > 0)
            {
                int *dcontinueScan;
                cudaMalloc((void **) &dcontinueScan, pointsLeft * sizeof(int));
                thrust::inclusive_scan(thrust::device, dcontinues[dev_id], dcontinues[dev_id] + pointsLeft,
                                       dcontinueScan);

                cudaMalloc((void **) &dRadiusNew[dev_id], count * sizeof(float));
                cudaMalloc((void **) &dCountersNew[dev_id], count * sizeof(int));
                cudaMalloc((void **) &dRadiusIndexNew[dev_id], count * sizeof(int));
                cudaMalloc((void **) &dIndexesNew[dev_id], count * sizeof(int));

                getNewData<<<blocks, threads>>>(dcontinueScan, dcontinues[dev_id],
                        dIndexesQuery[dev_id], dIndexesNew[dev_id],
                        dCurrentRadius[dev_id], dRadiusNew[dev_id],
                        dindexesMaximum[dev_id], dRadiusIndexNew[dev_id],
                        dcounters[dev_id], dCountersNew[dev_id], pointsLeft);
                cudaDeviceSynchronize();

                cudaFree(dcontinueScan);

                std::swap(dIndexesQuery[dev_id], dIndexesNew[dev_id]);
                std::swap(dCurrentRadius[dev_id], dRadiusNew[dev_id]);
                std::swap(dcounters[dev_id], dCountersNew[dev_id]);
                std::swap(dindexesMaximum[dev_id], dRadiusIndexNew[dev_id]);

                cudaFree(dIndexesNew[dev_id]);
                cudaFree(dRadiusNew[dev_id]);
                cudaFree(dRadiusIndexNew[dev_id]);
                cudaFree(dCountersNew[dev_id]);

                cudaFree(dcontinues[dev_id]);
                cudaFree(dslicecontinue[dev_id]);
                pointsLeft = count;
                cudaMalloc((void **) &dslicecontinue[dev_id], pointsLeft * sizeof(int));
                cudaMalloc((void **) &dcontinues[dev_id], pointsLeft * sizeof(int));
            }
        } while (count > 0 && bucketsCount > 32);

        cudaFree(dindexes[dev_id]);
        cudaFree(dIndexesQuery[dev_id]);
        cudaFree(dCurrentRadius[dev_id]);
        cudaFree(dindexesMaximum[dev_id]);
        cudaFree(dcounters[dev_id]);

        cudaFree(dpointBucketIndexes[dev_id]);
        cudaFree(dbucketIndexes[dev_id]);
        cudaFree(dbucketIndexes_x[dev_id]);
        cudaFree(dbucketIndexes_y[dev_id]);
        cudaFree(dbucketIndexes_z[dev_id]);

        cudaFree(dcounters[dev_id]);
        cudaFree(dslicecontinue[dev_id]);

        cudaFree(dCurrentRadius[dev_id]);
        cudaFree(dindexesMaximum[dev_id]);
        cudaFree(dcontinues[dev_id]);
        cudaFree(dknndistances[dev_id]);
    }
    //std::chrono::steady_clock::time_point endtime = std::chrono::steady_clock::now();
    //std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(endtime - begin).count() << "[ms]" << std::endl;


    free(pointsLeft_a);
    free(bucketsCount_a);
    free(offset_a);

    /*FILE *file = fopen("OneGPU.bin","rb");
    std::vector<int> indexes(pointsCount * k);
    fread(indexes.data(), sizeof(int), k * pointsCount, file);
    fclose(file);*/
    int counter = 0;
    /*for (int dev_id = 0; dev_id < num_cards; dev_id++) {
        cudaSetDevice(dev_id);
        std::cout << "GPU_ID:" << dev_id << " " << pointsCard[dev_id] << std::endl;
        std::vector<int> indexes_multi(pointsCard[dev_id] * k);
        cudaMemcpy(indexes_multi.data(), dknn[dev_id], pointsCard[dev_id] * k * sizeof(int), cudaMemcpyDeviceToHost);
        std::string fileOut = "KNN_" + std::to_string(dev_id) + ".bin";
        FILE* fileout = fopen(fileOut.c_str(), "wb");
        fwrite(indexes_multi.data(), sizeof(int), k * pointsCard[dev_id], fileout);
        fclose(fileout);
        counter += pointsCard[dev_id];
    }*/
    //std::cout << "passed!" << std::endl;
    //exit(0);
    /*auto start_time_communication = std::chrono::steady_clock::now();
#pragma omp parallel for num_threads(num_cards)
    for (int dev_id = 1; dev_id < num_cards; dev_id++){
        cudaMemcpy(dknn[0] + k * dev_id * pointPartitionSize, dknn[dev_id] + k * dev_id * pointPartitionSize, k * pointsCard[dev_id] * sizeof(int),
                   cudaMemcpyDeviceToDevice);
    }

    auto end_time_communication = std::chrono::steady_clock::now();
    std::cout << "Time difference communication = " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time_communication - start_time_communication).count()
            << "[ms]" << std::endl;

    auto start_time_rearrange = std::chrono::steady_clock::now();
    //int threads = 1024;
    //int blocks = (int)ceil((1.0 * k * pointsCount) / threads);
    //reverseIndexNeighbors<<<blocks, threads>>>(dknn_result[0], dIndexesOriginal[0], k * pointsCount);
    //cudaDeviceSynchronize();

    int threads = 1024;
	int blocks = (int)ceil((1.0 * pointsCount) / threads);
    /*rearrangeNeighbors<<<blocks, threads>>>(dknn[0], dknn_result[0], dIndexesOriginal[0], dIndexesReverseOriginal[0], k, pointsCount);
    cudaDeviceSynchronize();*/
    //auto end_time_rearrange = std::chrono::steady_clock::now();
    //std::cout << "Time difference rearrange = " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time_rearrange - start_time_rearrange).count()
    //          << "[ms]" << std::endl;
    //int* dknnTranspose;
    //cudaMalloc((void**)&dknnt, pointsCount* k * sizeof(int));
	//transpose1<<<blocks, threads>>>(dknnt, dknn[0], pointsCount, k);
	//cudaDeviceSynchronize();

	//cudaMemcpy(knn, dknnTranspose, pointsCount * k * sizeof(int), cudaMemcpyDeviceToHost);

    //denormalize_points << < blocks, threads >> > (dpoints_x[0], dpoints_y[0], dpoints_z[0], minOriginal, maxOriginal, pointsCount);
    //cudaDeviceSynchronize();


    //std::swap(dpnts_x, dpoints_x[0]);
    //std::swap(dpnts_y, dpoints_y[0]);
    //std::swap(dpnts_z, dpoints_z[0]);

    //cudaMemcpy(points_x, dpoints_x[0], pointsCount * sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpy(points_y, dpoints_y[0], pointsCount * sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpy(points_z, dpoints_z[0], pointsCount * sizeof(float), cudaMemcpyDeviceToHost);

    int start = 0;
    for (int dev_id = 0; dev_id < num_cards; dev_id++) {
        cudaSetDevice(dev_id);
        //cudaFree(dknn[dev_id]);
        //cudaFree(dknn_result[dev_id]);
        cudaFree(dIndexesOriginal[dev_id]);
        cudaFree(dIndexesReverseOriginal[dev_id]);

        //cudaFree(dpoints_x[dev_id]);
        //cudaFree(dpoints_y[dev_id]);
        //cudaFree(dpoints_z[dev_id]);
    }

    free(dcontinues);               //deleted
    free(dIndexesQuery);            //deleted
    free(dIndexesOriginal);         //deleted
    free(dIndexesReverseOriginal);  //deleted
    free(dbucketIndexes_x);         //deleted
    free(dbucketIndexes_y);         //deleted
    free(dbucketIndexes_z);         //deleted
    free(dbucketIndexes);           //deleted
    free(dindexes);                 //deleted
    free(dpointBucketIndexes);      //deleted
    free(dcounters);                //deleted
    free(dslicecontinue);           //deleted
    free(dCurrentRadius);           //deleted
    //free(dpoints_x);                //deleted
    //free(dpoints_y);                //deleted
    //free(dpoints_z);                //deleted
    free(dindexesMaximum);          //deleted
    free(dknndistances);            //deleted
    //free(dknn);                     //deleted
    //free(dknn_result);              //deleted
    free(dRadiusNew);               //deleted
    free(dRadiusIndexNew);      //deleted
    free(dCountersNew);         //deleted
    free(dIndexesNew);          //deleted

    auto endtime = std::chrono::steady_clock::now();
    std::cout << "Time Execution for KNN = " << std::chrono::duration_cast<std::chrono::milliseconds>(endtime - begin).count() << "[ms]" << std::endl;

    //cudaSetDevice(0);
    //cudaMemcpy(x, dpoints_x[0], pointsCount * sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpy(y, dpoints_y[0], pointsCount * sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpy(z, dpoints_z[0], pointsCount * sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpy(knn, dknn[0], k * pointsCount * sizeof(int), cudaMemcpyDeviceToHost);
}