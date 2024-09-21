#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <vector>
__host__
float EucledianDistanceHost(int i, int j, float* pnts_x, float* pnts_y, float* pnts_z)
{
	float xi = pnts_x[i];
	float yi = pnts_y[i];
	float zi = pnts_z[i];
	float xj = pnts_x[j];
	float yj = pnts_y[j];
	float zj = pnts_z[j];
	return (xi - xj) * (xi - xj) + (yi - yj) * (yi - yj) + (zi - zj) * (zi - zj);
}

void cudaWarmup(float* points_x, float* points_y, float* points_z, int pointsCount, int k, int* kNN, int num_cards)
{
    for (int dev_id = 0; dev_id < num_cards; dev_id++) {
        cudaSetDevice(dev_id);
        std::vector<float> distances(pointsCount);
        for (int i = 0; i < pointsCount; i++) {
            distances[i] = EucledianDistanceHost(k, i, points_x, points_y, points_z);
        }

        std::vector<int> distancesIdx(pointsCount);
        for (int i = 0; i < pointsCount; i++) {
            distancesIdx[i] = i;
        }

        float *ddistances;
        cudaMalloc((void **) &ddistances, pointsCount * sizeof(float));
        cudaMemcpy(ddistances, distances.data(), pointsCount * sizeof(float), cudaMemcpyHostToDevice);

        int *ddistancesIdx;
        cudaMalloc((void **) &ddistancesIdx, pointsCount * sizeof(int));
        cudaMemcpy(ddistancesIdx, distancesIdx.data(), pointsCount * sizeof(int), cudaMemcpyHostToDevice);

        thrust::sort_by_key(thrust::device, ddistances, ddistances + pointsCount, ddistancesIdx, thrust::less<float>());
        cudaMemcpy(distancesIdx.data(), ddistancesIdx, pointsCount * sizeof(int), cudaMemcpyDeviceToHost);

        for (int i = 1; i < 41; i++) {
            kNN[(i - 1) + k * 40] = distancesIdx[i];
        }

        cudaFree(ddistances);
        cudaFree(ddistancesIdx);
    }
}