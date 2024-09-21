//
// Created by agalex on 6/19/24.
//

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <algorithm>

#include "EGT_CUDA.cuh"

#define DEBUG 1

namespace{
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
    float norm2_GKNN(const float& x1, const float& y1, const float& z1, const float& x2, const float& y2, const float& z2){
        return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2);
    }

    struct PointCloud{
        float *pnts_x;
        float *pnts_y;
        float *pnts_z;
    };

    struct TaubinHelper{
        int* haloElements;
        float* haloBuffer;
        int* haloIndexes;
        int* gpu_ids;
        int* neighbors;
        int max_neighbors;
        int numb_gpus;
    };

    __device__
    void normalize_GKNN(float& x, float& y, float& z){
        float norm = sqrtf(x * x + y * y + z * z);
        x /= norm; y /= norm; z /= norm;
    }

    __device__
    void GetNormalizedPerpendicularVectorToVector_GKNN(const float& x1, const float& y1, const float& z1,
                                                       float& x2, float& y2, float& z2){
        float max = fabs(x1);

        int cordIndex = 0;
        if (max < fabs(y1))
        {
            cordIndex = 1;
            max = fabs(y1);
        }

        if (max < fabs(z1))
        {
            cordIndex = 2;
        }

        x2 = 1.0;
        y2 = 1.0;
        z2 = 1.0;

        switch (cordIndex)
        {
            case 0:
                x2 = (-y1 * y2 - z1 * z2) / x1;
                break;
            case 1:
                y2 = (-x1 * x2 - z1 * z2) / y1;
                break;
            case 2:
                z2 = (-x1 * x2 - y1 * y2) / z1;
                break;
        }
        normalize_GKNN(x2, y2, z2);
    }

    __device__
    void cross_GKNN(const float& u1, const float& u2, const float& u3,
                    const float& v1, const float& v2, const float& v3,
                    float& x, float&y, float& z){
        x = u2 * v3 - v2 * u3;
        y = v1 * u3 - u1 * v3;
        z = u1 * v2 - v1 * u2;
    }

    __device__
    void matrix_multiplication_GKNN(float& a11, float& a12, float& a13, float& a14,
                                    float& a21, float& a22, float& a23, float& a24,
                                    float& a31, float& a32, float& a33, float& a34,
                                    float& x, float& y, float& z){
        float x1 = x; float y1 = y; float z1 = z;
        x = a11 * x1 + a12 * y1 + a13 * z1 + a14;
        y = a21 * x1 + a22 * y1 + a23 * z1 + a24;
        z = a31 * x1 + a32 * y1 + a33 * z1 + a34;
    }

    __device__
    void CreateLocalCoordinateSystem_GKNN(float& xo, float& yo, float& zo,
                                          const float& xd, const float& yd, const float& zd,
                                          float& a11, float& a12, float& a13, float& a14,
                                          float& a21, float& a22, float& a23, float& a24,
                                          float& a31, float& a32, float& a33, float& a34)
    {
        GetNormalizedPerpendicularVectorToVector_GKNN(xd, yd, zd, a21, a22, a23);
        cross_GKNN(xd, yd, zd, a21, a22, a23, a31, a32, a33);
        normalize_GKNN(a31, a32, a33);
        a14 = 0.0;
        a24 = 0.0;
        a34 = 0.0;
        matrix_multiplication_GKNN(a11, a12, a13, a14, a21, a22, a23, a24, a31, a32, a33, a34, xo, yo, zo);
        a14 = -xo;
        a24 = -yo;
        a34 = -zo;
    }

    __device__
    bool isEllipicGabrielNeighbor_GKNN(int pnt_idx, int i, float* pnts_x, float* pnts_y, float* pnts_z,
                                       const float& x, const float& y, const float& z, int* NNs,
                                       int k, float a)
    {
        int neigh = NNs[pnt_idx * k + i];

        float xi = pnts_x[neigh]; float yi = pnts_y[neigh]; float zi = pnts_z[neigh];
        float xo = 0.5f * (xi + x); float yo = 0.5f * (yi + y); float zo = 0.5f * (zi + z);

        float d = sqrtf(norm2_GKNN(xi, yi, zi, x, y, z)) / 2.0f;
        if (d < 1e-6) return true;
        float xaxis_x = xi - x; float xaxis_y = yi - y; float xaxis_z = zi - z;
        if (sqrt(xaxis_x * xaxis_x + xaxis_y * xaxis_y + xaxis_z * xaxis_z) < 1e-6) return true;
        normalize_GKNN(xaxis_x, xaxis_y, xaxis_z);
        float a11, a12, a13, a14;
        float a21, a22, a23, a24;
        float a31, a32, a33, a34;
        a11 = xaxis_x; a12 = xaxis_y; a13 = xaxis_z;
        CreateLocalCoordinateSystem_GKNN(xo, yo, zo, xaxis_x, xaxis_y, xaxis_z, a11, a12, a13, a14,
                                         a21, a22, a23, a24, a31, a32, a33, a34);
        for (int j = 0; j < i; j++)
        {
            neigh = NNs[pnt_idx * k + j];
            if (neigh >= 0) {
                xi = pnts_x[neigh];
                yi = pnts_y[neigh];
                zi = pnts_z[neigh];
                matrix_multiplication_GKNN(a11, a12, a13, a14, a21, a22, a23, a24, a31, a32, a33, a34, xi, yi, zi);
                float ellipsoidValue = xi * xi + yi * yi / (a * a) + zi * zi / (a * a);
                if (ellipsoidValue < d * d) return false;
            }
        }
        return true;
    }

    __global__
    void calculateEGG_GKNN(PointCloud pc, int* NNs, float ratio, int neighborsCount, int startBatch,
                           int startDevice, int batchCount)
    {
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        int k = neighborsCount;
        if (t < batchCount){
            int i = t + startBatch;
            int realIdx = i + startDevice;
            float x = pc.pnts_x[realIdx], y = pc.pnts_y[realIdx], z = pc.pnts_z[realIdx];
            for (int j = neighborsCount - 1; j >= 0; j--)
            {
                if (NNs[i * k + j] != -1 && !isEllipicGabrielNeighbor_GKNN(i, j, pc.pnts_x, pc.pnts_y, pc.pnts_z, x, y, z, NNs, k, ratio)){
                    NNs[i * k + j] = -1;
                }
            }
        }
    }

    __host__
    __device__
    int dev_membership(int idx, int pointPartitionSize, int numb_gpus){
        int dev_id = 0;
        for (; dev_id < numb_gpus - 1; dev_id++){
            if (idx < (dev_id + 1) * pointPartitionSize){
                return dev_id;
            }
        }
        return dev_id;
    }

    __global__
    void create_halo_indexes(int dev_id, int* halo_indexes, int pointCount, int* dKNN, int k, int pointPartitionSize,
                             int numb_gpus){
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < pointCount){
            for (int j = 0; j < k; j++){
                int neigh = dKNN[i * k + j];
                if (neigh >= 0){
                    if (dev_id == dev_membership(neigh, pointPartitionSize, numb_gpus))
                        halo_indexes[neigh] = 1;
                }
            }
        }
    }

    __global__
    void gather_halo_elements(int* halo_indexes, int* sum_halo_indexes, int pointCount, int* halo_elements){
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < pointCount){
            if (halo_indexes[i]){
                int idx = sum_halo_indexes[i];
                halo_elements[idx] = i;
            }
        }
    }

    __global__
    void taubin_step_GKNN(PointCloud pcIn, PointCloud pcOut, TaubinHelper helper, float scale, int isRegularized,
                          int startBatch, int startDevice, int batchCount,
                          int partitionCount, int dev_id){
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        if (t < batchCount){
            int idx = t + startBatch;
            int i = idx + startDevice;
            float cog_x = 0.0f;
            float cog_y = 0.0f;
            float cog_z = 0.0f;
            float sum = 0.0f;
            float x1 = pcIn.pnts_x[i]; float y1 = pcIn.pnts_y[i]; float z1 = pcIn.pnts_z[i];
            for (int n = 0; n < helper.max_neighbors; n++){
                int neigh = helper.neighbors[idx * helper.max_neighbors + n];
                if (neigh != -1) {
                    int reverseNeigh = neigh;
                    int dev_neigh = dev_membership(reverseNeigh, partitionCount, helper.numb_gpus);
                    float x2, y2, z2;
                    if (dev_neigh == dev_id) {
                        x2 = pcIn.pnts_x[neigh];
                        y2 = pcIn.pnts_y[neigh];
                        z2 = pcIn.pnts_z[neigh];
                    }
                    else {
                        int startIdx = helper.gpu_ids[dev_neigh];
                        int idx = binarySearchInt<int>(helper.haloElements + helper.haloIndexes[startIdx], reverseNeigh, 0,
                                                       helper.haloIndexes[startIdx + 1] - helper.haloIndexes[startIdx] - 1);
                        x2 = helper.haloBuffer[3 * helper.haloIndexes[startIdx] + 3 * idx];
                        y2 = helper.haloBuffer[3 * helper.haloIndexes[startIdx] + 3 * idx + 1];
                        z2 = helper.haloBuffer[3 * helper.haloIndexes[startIdx] + 3 * idx + 2];
                    }
                    float distance = norm2_GKNN(x1, y1, z1, x2, y2, z2);
                    float w;
                    if (isRegularized) w = scale < 0.0f ? 1.0f / (distance + 1e-8f) : 1.0f;
                    else w = expf(-distance);
                    cog_x += w * (x2 - x1);
                    cog_y += w * (y2 - y1);
                    cog_z += w * (z2 - z1);
                    sum += w;
                }
            }
            if (sum == 0.0) sum = 1e-8;
            pcOut.pnts_x[i] = x1 + scale * cog_x / sum;
            pcOut.pnts_y[i] = y1 + scale * cog_y / sum;
            pcOut.pnts_z[i] = z1 + scale * cog_z / sum;
        }
    }
    __global__
    void gather_halo_elements(PointCloud pc,
                              int* halo_indexes, float* halo_buffer, int elementCount){
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < elementCount){
            int idx = halo_indexes[i];
            float x = pc.pnts_x[idx];
            float y = pc.pnts_y[idx];
            float z = pc.pnts_z[idx];
            halo_buffer[3 * i] = x;
            halo_buffer[3 * i + 1] = y;
            halo_buffer[3 * i + 2] = z;
        }
    }
}

void EGT_CUDA::create_halo_elements(){
    std::vector<std::vector<int *>> send_halo_elements(numb_gpus, std::vector<int *>(numb_gpus, nullptr));
#pragma omp parallel for num_threads(numb_gpus)
    for (int dev_id = 0; dev_id < numb_gpus; dev_id++) {
        cudaSetDevice(dev_id);
        int N = points_count_per_device[dev_id];
        int counter = 0;
        for (int i = 0; i < numb_gpus; i++){
            if (i != dev_id)
                host_gpu_ids[dev_id][i] = counter++;
            else host_gpu_ids[dev_id][i] = -1;
        }
        cudaMalloc((void **)&dev_gpu_ids[dev_id], numb_gpus * sizeof(int));
        cudaMemcpy(dev_gpu_ids[dev_id], host_gpu_ids[dev_id].data(), numb_gpus * sizeof(int), cudaMemcpyHostToDevice);
        counter = 0;
        std::vector<int*> haloElements(numb_gpus - 1, nullptr);
        std::vector<int> haloElementsCount(numb_gpus);
        int sum_count = 0;
        for (int i = 0; i < numb_gpus; i++) {
            if (i == dev_id) continue;
            int *halo_indexes;
            int *sum_halo_indexes;
            cudaMalloc((void **)&halo_indexes, pointsCount * sizeof(int));
            cudaMalloc((void **)&sum_halo_indexes, pointsCount * sizeof(int));
            cudaMemset(halo_indexes, 0, pointsCount * sizeof(int));

            int threads = 1024;
            int blocks = ceil((1.0 * N) / threads);
            create_halo_indexes<<<blocks, threads>>>(i, halo_indexes, N,
                                                     dKNN[dev_id], k, pointPartitionSize,
                                                     numb_gpus);
            cudaDeviceSynchronize();
            int count = thrust::reduce(thrust::device, halo_indexes, halo_indexes + pointsCount, 0);
            if (count > 0) {
                thrust::exclusive_scan(thrust::device, halo_indexes, halo_indexes + pointsCount,
                                       sum_halo_indexes);
                cudaMalloc((void **)&haloElements[counter], count * sizeof(int));
                send_halo_elements[i][dev_id] = haloElements[counter]; //the pointer is dev_id
                cudaMalloc((void**)& receive_halo_buffer[dev_id][i], 3 * count * sizeof(float));
                blocks = ceil((1.0 * pointsCount) / threads);
                gather_halo_elements<<<blocks, threads>>>(halo_indexes, sum_halo_indexes,
                                                          pointsCount, haloElements[counter]);
                cudaDeviceSynchronize();
                thrust::sort(thrust::device, haloElements[counter], haloElements[counter] + count);
            }
            send_halo_count[i][dev_id] = count;
            receive_halo_count[dev_id][i] = count;
            haloElementsCount[counter++] = sum_count;
            sum_count += count;
            cudaFree(halo_indexes);
            cudaFree(sum_halo_indexes);
        }
        haloElementsCount[counter] = sum_count;
        host_haloElementsCount[dev_id] = haloElementsCount;
        receive_halo_elements_count[dev_id] = sum_count;
        cudaMalloc((void**)& dev_haloElementsCount[dev_id], numb_gpus * sizeof(int));
        cudaMemcpy(dev_haloElementsCount[dev_id], haloElementsCount.data(), numb_gpus * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMalloc((void**)&dev_haloElements[dev_id], sum_count * sizeof(int));
        cudaMalloc((void**)&receive_halo_buffer_bulk[dev_id], 3 * sum_count * sizeof(float));
        sum_count = 0;
        // halo elements that a gpu needs
        for (int i = 0; i < numb_gpus - 1; i++)
        {
            int count = haloElementsCount[i + 1] - haloElementsCount[i];
            if (count > 0) {
                cudaMemcpy(dev_haloElements[dev_id] + haloElementsCount[i], haloElements[i],
                           count * sizeof(int), cudaMemcpyDeviceToDevice);
            }
        }
    }

#pragma omp parallel for num_threads(numb_gpus)
    for (int dev_id = 0; dev_id < numb_gpus; dev_id++){
        cudaSetDevice(dev_id);
        int counter = 0;
        for (int i = 0; i < numb_gpus; i++) {
            if (send_halo_count[dev_id][i] > 0)
                cudaMalloc((void**)&send_halo_buffer[dev_id][i], 3 * send_halo_count[dev_id][i] * sizeof(float));
            counter += send_halo_count[dev_id][i];
        }
        cudaMalloc((void**)&send_halo_to_gpu[dev_id], counter * sizeof(int));
        cudaMalloc((void**)&send_halo_buffer_bulk[dev_id], 3 * counter * sizeof(float));
        counter = 0;
        for (int i = 0; i < numb_gpus; i++) {
            if (send_halo_count[dev_id][i] > 0) {
                cudaMemcpy(send_halo_to_gpu[dev_id] + counter, send_halo_elements[dev_id][i],
                           send_halo_count[dev_id][i] * sizeof(int),
                           cudaMemcpyDeviceToDevice);
                cudaSetDevice(i);
                cudaFree(send_halo_elements[dev_id][i]);
                cudaSetDevice(dev_id);
                counter += send_halo_count[dev_id][i];
            }
        }
        send_halo_elements_count[dev_id] = counter;
    }
}

EGT_CUDA::EGT_CUDA(KNNInterface& knnInterface):
    din_x(knnInterface.GetRefPointsX()),
    din_y(knnInterface.GetRefPointsY()),
    din_z(knnInterface.GetRefPointsZ()),
    dKNN(knnInterface.GetKNNIndexesInPartitions()),
    points_count_per_device(knnInterface.GetPointCountInCards()),
    k(knnInterface.NeighborCount()),
    pointsCount(knnInterface.pointsRefCount()),
    numb_gpus(knnInterface.GetNumberOfCards()),
    pointPartitionSize(knnInterface.GetPartitionSize())
{
    dev_haloElements.resize(numb_gpus, nullptr);
    dev_gpu_ids.resize(numb_gpus, nullptr);
    host_gpu_ids.resize(numb_gpus, std::vector<int>(numb_gpus));
    dev_haloElementsCount.resize(numb_gpus, nullptr);
    host_haloElementsCount.resize(numb_gpus);
    send_halo_to_gpu.resize(numb_gpus, nullptr);
    send_halo_count.resize(numb_gpus, std::vector<int>(numb_gpus, 0));
    receive_halo_count.resize(numb_gpus, std::vector<int>(numb_gpus, 0));
    receive_halo_buffer.resize(numb_gpus, std::vector<float*>(numb_gpus, nullptr));
    send_halo_buffer.resize(numb_gpus, std::vector<float*>(numb_gpus, nullptr));
    receive_halo_buffer_bulk.resize(numb_gpus, nullptr);
    send_halo_buffer_bulk.resize(numb_gpus, nullptr);
    send_halo_elements_count.resize(numb_gpus, 0);
    receive_halo_elements_count.resize(numb_gpus, 0);
}

void EGT_CUDA::reset_halo_elements() {
    for (int dev_id = 0; dev_id < numb_gpus; dev_id++){
        cudaSetDevice(dev_id);
        cudaFree(dev_haloElements[dev_id]); dev_haloElements[dev_id] = nullptr;
        cudaFree(dev_gpu_ids[dev_id]); dev_gpu_ids[dev_id] = nullptr;
        host_gpu_ids[dev_id] = std::vector<int>(numb_gpus);
        cudaFree(dev_haloElementsCount[dev_id]); dev_haloElementsCount[dev_id] = nullptr;
        host_haloElementsCount[dev_id] = std::vector<int>();
        cudaFree(send_halo_to_gpu[dev_id]); send_halo_to_gpu[dev_id] = nullptr;
        send_halo_count[dev_id] = std::vector<int>(numb_gpus, 0);
        receive_halo_count[dev_id] = std::vector<int>(numb_gpus, 0);
        std::for_each(send_halo_buffer[dev_id].begin(), send_halo_buffer[dev_id].end(),
                      [](float*& buffer){ cudaFree(buffer); buffer = nullptr;});
        std::for_each(receive_halo_buffer[dev_id].begin(), receive_halo_buffer[dev_id].end(),
                      [](float*& buffer){ cudaFree(buffer); buffer = nullptr;});
        cudaFree(receive_halo_buffer_bulk[dev_id]); receive_halo_buffer_bulk[dev_id] = nullptr;
        cudaFree(send_halo_buffer_bulk[dev_id]); send_halo_buffer_bulk[dev_id] = nullptr;
        send_halo_elements_count[dev_id] = 0;
        receive_halo_elements_count[dev_id] = 0;
    }
}

void EGT_CUDA::sender_halo_break(int dev_id){
    int counter = 0;
    for (int i = 0; i < numb_gpus; i++) {
        if (send_halo_count[dev_id][i] > 0) {
            cudaMemcpy(send_halo_buffer[dev_id][i], send_halo_buffer_bulk[dev_id] + 3 * counter, 3 * send_halo_count[dev_id][i] * sizeof(float),
                       cudaMemcpyDeviceToDevice);
            counter += send_halo_count[dev_id][i];
        }
    }
}

void EGT_CUDA::communication_sender_to_receiver(int dev_id, std::vector<int*>& receive_halo_flag){
    for (int i = 0; i < numb_gpus; i++){
        if (send_halo_count[dev_id][i] > 0){
            cudaMemcpy(receive_halo_buffer[i][dev_id], send_halo_buffer[dev_id][i],
                       3 * send_halo_count[dev_id][i] * sizeof(float), cudaMemcpyDeviceToDevice);
        }
        int flag = 1;
        cudaMemcpy(receive_halo_flag[i] + dev_id, &flag,sizeof(int), cudaMemcpyHostToDevice);
    }
#pragma omp barrier
}

void EGT_CUDA::receiver_halo_synthesize(int dev_id, int* receive_halo_flag){
    int counter = 0;
    for (int i = 0; i < numb_gpus; i++) {
        if (receive_halo_count[dev_id][i] > 0){
            cudaMemcpy(receive_halo_buffer_bulk[dev_id] + 3 * counter, receive_halo_buffer[dev_id][i], 3 * receive_halo_count[dev_id][i] * sizeof(float),
                       cudaMemcpyDeviceToDevice);
            counter += receive_halo_count[dev_id][i];
        }
    }
    cudaMemset(receive_halo_flag, 0, numb_gpus * sizeof(int));
}

void EGT_CUDA::PerformSmoothing(int iterationCount, float lambda, float mu, float alpha){
    size_t size = pointsCount * sizeof(int);

    std::vector<float*> dvout_x(numb_gpus); float** dout_x = dvout_x.data();
    std::vector<float*> dvout_y(numb_gpus); float** dout_y = dvout_y.data();
    std::vector<float*> dvout_z(numb_gpus); float** dout_z = dvout_z.data();
    std::vector<int*> receive_halo_flags_lambda(numb_gpus);
    std::vector<int*> receive_halo_flags_mu(numb_gpus);
    std::vector<TaubinHelper> taubinHelpers(numb_gpus);
    std::vector<PointCloud> pcIn(numb_gpus);
    std::vector<PointCloud> pcOut(numb_gpus);
#pragma omp parallel for num_threads(numb_gpus)
    for (int dev_id = 0; dev_id < numb_gpus; dev_id++) {
        cudaSetDevice(dev_id);
        cudaMalloc((void**)&dout_x[dev_id], size);
        cudaMalloc((void**)&dout_y[dev_id], size);
        cudaMalloc((void**)&dout_z[dev_id], size);
        cudaMalloc((void**)&receive_halo_flags_lambda[dev_id], numb_gpus * sizeof(int));
        cudaMemset(receive_halo_flags_lambda[dev_id], 0, numb_gpus * sizeof(int));
        cudaMalloc((void**)&receive_halo_flags_mu[dev_id], numb_gpus * sizeof(int));
        cudaMemset(receive_halo_flags_mu[dev_id], 0, numb_gpus * sizeof(int));
        pcIn[dev_id].pnts_x = din_x[dev_id];
        pcIn[dev_id].pnts_y = din_y[dev_id];
        pcIn[dev_id].pnts_z = din_z[dev_id];
        pcOut[dev_id].pnts_x = dout_x[dev_id];
        pcOut[dev_id].pnts_y = dout_y[dev_id];
        pcOut[dev_id].pnts_z = dout_z[dev_id];
    }

    int max_threads = 2000000;

#pragma omp parallel for num_threads(numb_gpus)
    for (int dev_id = 0; dev_id < numb_gpus; dev_id++) {
        cudaSetDevice(dev_id);
        int times = points_count_per_device[dev_id] / max_threads;
        int start = 0;
        for (int i = 0; i < times; i++) {
            int threads = 1024;
            int blocks = (int) ceil((1.0 * max_threads) / threads);
            float r = alpha;
            calculateEGG_GKNN<<<blocks, threads>>>(pcIn[dev_id], dKNN[dev_id],
                                                   r, k, start, dev_id * pointPartitionSize,
                                                   max_threads);
            cudaDeviceSynchronize();
            start += max_threads;
        }
        if (start < points_count_per_device[dev_id]) {
            int threads = 1024;
            int blocks = (int) ceil((1.0 * (points_count_per_device[dev_id] - start)) / threads);
            calculateEGG_GKNN<<<blocks, threads>>>(pcIn[dev_id], dKNN[dev_id],
                                                   alpha, k, start, dev_id * pointPartitionSize,
                                                   points_count_per_device[dev_id] - start);
            cudaDeviceSynchronize();
        }
    }

    create_halo_elements();

#if DEBUG
    for (int i = 0; i < numb_gpus; i++){
        std::cout << "Statistics for GPU:" << i << std::endl;
        for (int j = 0; j < numb_gpus; j++){
            std::cout << "Send halo elements (" << i << "->" << j << "):" << send_halo_count[i][j] << std::endl;
            std::cout << "Receive halo elements (" << j << "->" << i << "):" << receive_halo_count[i][j] << std::endl;
        }
    }
#endif

    int isRegularized = 0;
#pragma omp parallel for num_threads(numb_gpus)
    for (int dev_id = 0; dev_id < numb_gpus; dev_id++) {
        int start;
        cudaSetDevice(dev_id);
        int times = points_count_per_device[dev_id] / max_threads;

        taubinHelpers[dev_id].haloIndexes = dev_haloElementsCount[dev_id];
        taubinHelpers[dev_id].numb_gpus = numb_gpus;
        taubinHelpers[dev_id].neighbors = dKNN[dev_id];
        taubinHelpers[dev_id].gpu_ids = dev_gpu_ids[dev_id];
        taubinHelpers[dev_id].haloBuffer = receive_halo_buffer_bulk[dev_id];
        taubinHelpers[dev_id].haloElements = dev_haloElements[dev_id];
        taubinHelpers[dev_id].max_neighbors = k;

        for (int k = 0; k < iterationCount; k++) {
            if (numb_gpus > 1) {
                int threads = 1024;
                int blocks = (int) ceil((1.0 * (send_halo_elements_count[dev_id])) / threads);
                gather_halo_elements<<<blocks, threads>>>(pcIn[dev_id], send_halo_to_gpu[dev_id],
                                                          send_halo_buffer_bulk[dev_id], send_halo_elements_count[dev_id]);
                cudaDeviceSynchronize();
                sender_halo_break(dev_id);
                communication_sender_to_receiver(dev_id, receive_halo_flags_lambda);
                while (thrust::reduce(thrust::device, receive_halo_flags_lambda[dev_id], receive_halo_flags_lambda[dev_id] + numb_gpus, 0) < numb_gpus - 1);
                receiver_halo_synthesize(dev_id, receive_halo_flags_lambda[dev_id]);
            }
            start = 0;
            for (int i = 0; i < times; i++) {
                int threads = 1024;
                int blocks = (int) ceil((1.0 * (max_threads)) / threads);

                taubin_step_GKNN<<<blocks, threads>>>(pcIn[dev_id], pcOut[dev_id], taubinHelpers[dev_id], lambda,
                                                      isRegularized, start, dev_id * pointPartitionSize, max_threads,
                                                      pointPartitionSize, dev_id);
                cudaDeviceSynchronize();
                start += max_threads;
            }
            if (start < points_count_per_device[dev_id]) {
                int threads = 1024;
                int blocks = (int) ceil((1.0 * (points_count_per_device[dev_id] - start)) / threads);
                taubin_step_GKNN<<<blocks, threads>>>(pcIn[dev_id], pcOut[dev_id], taubinHelpers[dev_id], lambda,
                                                      isRegularized, start, dev_id * pointPartitionSize,
                                                      points_count_per_device[dev_id] - start,
                                                      pointPartitionSize, dev_id);
                cudaDeviceSynchronize();
            }
            std::swap(din_x[dev_id], dout_x[dev_id]);
            std::swap(pcIn[dev_id].pnts_x, pcOut[dev_id].pnts_x);
            std::swap(din_y[dev_id], dout_y[dev_id]);
            std::swap(pcIn[dev_id].pnts_y, pcOut[dev_id].pnts_y);
            std::swap(din_z[dev_id], dout_z[dev_id]);
            std::swap(pcIn[dev_id].pnts_z, pcOut[dev_id].pnts_z);
            if (numb_gpus > 1) {
                int threads = 1024;
                int blocks = (int) ceil((1.0 * (send_halo_elements_count[dev_id])) / threads);
                gather_halo_elements<<<blocks, threads>>>(pcIn[dev_id],
                                                          send_halo_to_gpu[dev_id], send_halo_buffer_bulk[dev_id], send_halo_elements_count[dev_id]);
                cudaDeviceSynchronize();
                sender_halo_break(dev_id);
                communication_sender_to_receiver(dev_id, receive_halo_flags_mu);
                while (thrust::reduce(thrust::device, receive_halo_flags_mu[dev_id], receive_halo_flags_mu[dev_id] + numb_gpus, 0) < numb_gpus - 1);
                receiver_halo_synthesize(dev_id,receive_halo_flags_mu[dev_id]);
            }
            start = 0;
            for (int i = 0; i < times; i++) {
                int threads = 1024;
                int blocks = (int) ceil((1.0 * (max_threads)) / threads);
                taubin_step_GKNN<<<blocks, threads>>>(pcIn[dev_id],
                                                      pcOut[dev_id], taubinHelpers[dev_id], mu, isRegularized, start, dev_id * pointPartitionSize,
                                                      max_threads, pointPartitionSize, dev_id);
                cudaDeviceSynchronize();
                start += max_threads;
            }
            if (start < points_count_per_device[dev_id]) {
                int threads = 1024;
                int blocks = (int) ceil((1.0 * (points_count_per_device[dev_id] - start)) / threads);
                taubin_step_GKNN<<<blocks, threads>>>(pcIn[dev_id], pcOut[dev_id], taubinHelpers[dev_id], mu, isRegularized, start,
                                                      dev_id * pointPartitionSize,
                                                      points_count_per_device[dev_id] - start,
                                                      pointPartitionSize, dev_id);
                cudaDeviceSynchronize();
            }
            std::swap(din_x[dev_id], dout_x[dev_id]);
            std::swap(pcIn[dev_id].pnts_x, pcOut[dev_id].pnts_x);
            std::swap(din_y[dev_id], dout_y[dev_id]);
            std::swap(pcIn[dev_id].pnts_y, pcOut[dev_id].pnts_y);
            std::swap(din_z[dev_id], dout_z[dev_id]);
            std::swap(pcIn[dev_id].pnts_z, pcOut[dev_id].pnts_z);
        }
    }

#pragma omp parallel for num_threads(numb_gpus)
    for (int dev_id_r = 0; dev_id_r < numb_gpus; dev_id_r++){
        cudaSetDevice(dev_id_r);
        for (int dev_id = 0; dev_id < numb_gpus; dev_id++){
            if (dev_id != dev_id_r){
                cudaMemcpy(din_x[dev_id_r] + dev_id * pointPartitionSize, din_x[dev_id] + dev_id * pointPartitionSize,  points_count_per_device[dev_id] * sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpy(din_y[dev_id_r] + dev_id * pointPartitionSize, din_y[dev_id] + dev_id * pointPartitionSize,  points_count_per_device[dev_id] * sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpy(din_z[dev_id_r] + dev_id * pointPartitionSize, din_z[dev_id] + dev_id * pointPartitionSize,  points_count_per_device[dev_id] * sizeof(float), cudaMemcpyDeviceToDevice);
            }
        }
    }
    reset_halo_elements();
}
