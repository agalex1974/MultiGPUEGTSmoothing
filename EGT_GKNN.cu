#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <algorithm>

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
float norm2_GKNN(const float& x1, const float& y1, const float& z1, const float& x2, const float& y2, const float& z2){
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2);
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
void calculateEGG_GKNN(float* pnts_x, float* pnts_y, float* pnts_z, int* NNs, float ratio, int neighborsCount, int startBatch,
                       int startDevice, int batchCount)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int k = neighborsCount;
    if (t < batchCount){
        int i = t + startBatch;
        int realIdx = i + startDevice;
        float x = pnts_x[realIdx], y = pnts_y[realIdx], z = pnts_z[realIdx];
        for (int j = neighborsCount - 1; j >= 0; j--)
        {
            if (NNs[i * k + j] != -1 && !isEllipicGabrielNeighbor_GKNN(i, j, pnts_x, pnts_y, pnts_z, x, y, z, NNs, k, ratio)){
                NNs[i * k + j] = -1;
            }
        }
    }
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
void taubin_step_GKNN(float* in_x, float* in_y, float* in_z,
                      float* out_x, float* out_y, float* out_z, float scale, int* neighbors,
                      int max_neighbors, int isRegularized, int startBatch, int startDevice, int batchCount,
                      int* haloElements, float* haloBuffer, int* haloIndexes, int numb_gpus, int partitionCount,
                      int dev_id, int* gpu_ids){
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < batchCount){
        int idx = t + startBatch;
        int i = idx + startDevice;
        float cog_x = 0.0f;
        float cog_y = 0.0f;
        float cog_z = 0.0f;
        float sum = 0.0f;
        float x1 = in_x[i]; float y1 = in_y[i]; float z1 = in_z[i];
        for (int n = 0; n < max_neighbors; n++){
            int neigh = neighbors[idx * max_neighbors + n];
            if (neigh != -1) {
                int reverseNeigh = neigh;
                int dev_neigh = dev_membership(reverseNeigh, partitionCount, numb_gpus);
                float x2, y2, z2;
                if (dev_neigh == dev_id) {
                    x2 = in_x[neigh];
                    y2 = in_y[neigh];
                    z2 = in_z[neigh];
                }
                else {
                    int startIdx = gpu_ids[dev_neigh];
                    int idx = binarySearchInt<int>(haloElements + haloIndexes[startIdx], reverseNeigh, 0,
                                                       haloIndexes[startIdx + 1] - haloIndexes[startIdx] - 1);
                    x2 = haloBuffer[3 * haloIndexes[startIdx] + 3 * idx];
                    y2 = haloBuffer[3 * haloIndexes[startIdx] + 3 * idx + 1];
                    z2 = haloBuffer[3 * haloIndexes[startIdx] + 3 * idx + 2];
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
        out_x[i] = x1 + scale * cog_x / sum;
        out_y[i] = y1 + scale * cog_y / sum;
        out_z[i] = z1 + scale * cog_z / sum;
    }
}

__global__
void scale_points_to_unity_GKNN(float* x, float* y, float* z, float min, float max, int pointCount)
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
void produce_output_GKNN(float* x, float* y, float* z, float* xout, float* yout, float* zout, float min, float max, int pointCount)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < pointCount) {
        float d = max - min;
        xout[i] = x[i] * d + min;
        yout[i] = y[i] * d + min;
        zout[i] = z[i] * d + min;
    }
}

int get_device_by_ptr (void *ptr)
{
    cudaPointerAttributes pointer_attributes;
    cudaPointerGetAttributes (&pointer_attributes, ptr);
    return pointer_attributes.device;
}

__global__
void create_halo_indexes(int dev_id, int* halo_indexes, int pointCount, int* dKNN, int k, int pointPartitionSize,
                         int numb_gpus){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < pointCount){
        for (int j = 0; j < k; j++){
            int neigh = dKNN[i * k + j];
            //if (i == 0) printf("neigh:%d\n", neigh);
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

void communication_sender_to_receiver(int dev_id, std::vector<std::vector<float*>>& receive_halo_buffer, std::vector<float*>& send_halo_buffer,
                                      std::vector<std::vector<int>>& send_halo_count, int numb_gpus,
                                      std::vector<int*>& receive_halo_flag){
    for (int i = 0; i < numb_gpus; i++){
        if (send_halo_count[dev_id][i] > 0){
            cudaMemcpy(receive_halo_buffer[i][dev_id], send_halo_buffer[i],
                           3 * send_halo_count[dev_id][i] * sizeof(float), cudaMemcpyDeviceToDevice);
        }
        int flag = 1;
        cudaMemcpy(receive_halo_flag[i] + dev_id, &flag,sizeof(int), cudaMemcpyHostToDevice);
    }
#pragma omp barrier
}

void receiver_halo_synthesize(std::vector<float*>& receive_halo_buffer, float* receive_halo_buffer_bulk,
                              std::vector<int>& receive_halo_count,
                              int numb_gpus, int* receive_halo_flag){
    int counter = 0;
    for (int i = 0; i < numb_gpus; i++) {
        if (receive_halo_count[i] > 0){
            cudaMemcpy(receive_halo_buffer_bulk + 3 * counter, receive_halo_buffer[i], 3 * receive_halo_count[i] * sizeof(float),
                           cudaMemcpyDeviceToDevice);
            counter += receive_halo_count[i];
        }
    }
    cudaMemset(receive_halo_flag, 0, numb_gpus * sizeof(int));
}

void sender_halo_break(std::vector<float*>& send_halo_buffer,
                       float* send_halo_buffer_bulk,
                       std::vector<int>& send_halo_count, int numb_gpus){
    int counter = 0;
    for (int i = 0; i < numb_gpus; i++) {
        if (send_halo_count[i] > 0) {
            cudaMemcpy(send_halo_buffer[i], send_halo_buffer_bulk + 3 * counter, 3 * send_halo_count[i] * sizeof(float),
                       cudaMemcpyDeviceToDevice);
            counter += send_halo_count[i];
        }
    }
}

// dev_haloElements   --> The haloElements needed by each gpu
// send_halo_to_gpu   --> The haloElements that each gpu needs to send
// send_halo_count    --> The number pf haloElements that each gpu need to send to other gpus
// receive_halo_count --> The number of haloElements that eqch gpu needs to receive from other gpus
void halo_elements(std::vector<int*>& dev_haloElements, int** dKNN, std::vector<int*>& dev_gpu_ids, int pointsCount,
                   std::vector<std::vector<int>>& host_gpu_ids, std::vector<int*>& dev_haloElementsCount,
                   std::vector<int*>& send_halo_to_gpu, std::vector<std::vector<int>>& send_halo_count,
                   std::vector<std::vector<int>>& receive_halo_count,
                   std::vector<std::vector<float*>>& receive_halo_buffer, std::vector<std::vector<float*>>& send_halo_buffer,
                   std::vector<float*>& receive_halo_buffer_bulk, std::vector<float*>& send_halo_buffer_bulk,
                   int k, int* pointsInEachGPU, int pointPartitionSize, int numb_gpus,
                   std::vector<int>& send_halo_elements_count, std::vector<int>& receive_halo_elements_count,
                   std::vector<std::vector<int>>& host_haloElementsCount){

    send_halo_elements_count.resize(numb_gpus);
    receive_halo_elements_count.resize(numb_gpus);
    std::vector<std::vector<int *>> send_halo_elements(numb_gpus, std::vector<int *>(numb_gpus, nullptr));
    send_halo_count.resize(numb_gpus, std::vector<int>(numb_gpus, 0));
    receive_halo_count.resize(numb_gpus, std::vector<int>(numb_gpus, 0));
    receive_halo_buffer.resize(numb_gpus, std::vector<float*>(numb_gpus, nullptr));
    send_halo_buffer.resize(numb_gpus, std::vector<float*>(numb_gpus, nullptr));
#pragma omp parallel for num_threads(numb_gpus)
    for (int dev_id = 0; dev_id < numb_gpus; dev_id++) {
        cudaSetDevice(dev_id);
        //std::cout << "Device id:" << dev_id << std::endl;
        int N = pointsInEachGPU[dev_id];
        //std::cout << "Points In Each GPU..." << N << std::endl;
        int counter = 0;
        for (int i = 0; i < numb_gpus; i++){
            if (i != dev_id)
                host_gpu_ids[dev_id][i] = counter++;
            else host_gpu_ids[dev_id][i] = -1;
        }
        //std::cout << "First stage" << std::endl;
        cudaMalloc((void **)&dev_gpu_ids[dev_id], numb_gpus * sizeof(int));
        cudaMemcpy(dev_gpu_ids[dev_id], host_gpu_ids[dev_id].data(), numb_gpus * sizeof(int), cudaMemcpyHostToDevice);
        counter = 0;
        std::vector<int*> haloElements(numb_gpus - 1, nullptr);
        std::vector<int> haloElementsCount(numb_gpus);
        int sum_count = 0;
        for (int i = 0; i < numb_gpus; i++) {
            //std::cout << "GPU ID:" << i << std::endl;
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
            //std::cout << pointPartitionSize << std::endl;
            //std::cout << "Thrust..." << std::endl;
            int count = thrust::reduce(thrust::device, halo_indexes, halo_indexes + pointsCount, 0);
            //std::cout << count << std::endl;
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
                //cudaFree(haloElements[i]);
            }
        }
    }
    //std::cout << "Second stage" << std::endl;
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

__global__
void gather_halo_elements(float* pnts_x, float* pnts_y, float* pnts_z,
                          int* halo_indexes, float* halo_buffer, int elementCount){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < elementCount){
        int idx = halo_indexes[i];
        float x = pnts_x[idx];
        float y = pnts_y[idx];
        float z = pnts_z[idx];
        halo_buffer[3 * i] = x;
        halo_buffer[3 * i + 1] = y;
        halo_buffer[3 * i + 2] = z;
    }
}

__global__
void rearrange_output(float* xout, float* yout, float* zout, float* xin, float* yin, float* zin, int count, int* reverseIndexes){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < count){
        int idx = reverseIndexes[i];
        xout[idx] = xin[i];
        yout[idx] = yin[i];
        zout[idx] = zin[i];
    }
}

void EGTsmoothing_GKNN(float** din_x, float** din_y, float** din_z, int* count_per_device, float lambda, float mu,
                       int** dneighbors, int maxNeighbors, float* out_x, float* out_y, float* out_z, int count,
                       int iterationCount, int isRegularized, float ratio, int numb_gpus, int partition_size, int* reverseIndexes,
                       float minOriginal, float maxOriginal){

    size_t size = count * sizeof(float);

    std::cout << "Partition Size" << partition_size << std::endl;
    for (int i = 0; i < numb_gpus; i++){
        std::cout << count_per_device[i] << std::endl;
    }
    size_t sizeNeighbors = count * maxNeighbors * sizeof(int);

    std::vector<float*> dvout_x(numb_gpus); float** dout_x = dvout_x.data();
    std::vector<float*> dvout_y(numb_gpus); float** dout_y = dvout_y.data();
    std::vector<float*> dvout_z(numb_gpus); float** dout_z = dvout_z.data();
    std::vector<int*> receive_halo_flags_lambda(numb_gpus);
    std::vector<int*> receive_halo_flags_mu(numb_gpus);

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
    }

    int max_threads = 2000000;

#pragma omp parallel for num_threads(numb_gpus)
    for (int dev_id = 0; dev_id < numb_gpus; dev_id++) {
        cudaSetDevice(dev_id);
        int times = count_per_device[dev_id] / max_threads;
        int start = 0;
        for (int i = 0; i < times; i++) {
            int threads = 1024;
            int blocks = (int) ceil((1.0 * max_threads) / threads);
            float r = ratio;
            calculateEGG_GKNN<<<blocks, threads>>>(din_x[dev_id], din_y[dev_id], din_z[dev_id], dneighbors[dev_id],
                                                   r, maxNeighbors, start, dev_id * partition_size,
                                                   max_threads);
            cudaDeviceSynchronize();
            start += max_threads;
        }
        //std::cout << start << std::endl;
        if (start < count_per_device[dev_id]) {
            //std::cout << "Count:" << count_per_device[dev_id] << std::endl;
            int threads = 1024;
            int blocks = (int) ceil((1.0 * (count_per_device[dev_id] - start)) / threads);
            //std::cout << "blocks:" << blocks * threads << std::endl;
            calculateEGG_GKNN<<<blocks, threads>>>(din_x[dev_id], din_y[dev_id], din_z[dev_id], dneighbors[dev_id],
                                                   ratio, maxNeighbors, start, dev_id * partition_size,
                                                   count_per_device[dev_id] - start);
            cudaDeviceSynchronize();
        }
    }

    std::vector<int*> dev_haloElements(numb_gpus, nullptr);
    std::vector<int*> dev_gpu_ids(numb_gpus, nullptr);
    std::vector<std::vector<int>> host_gpu_ids(numb_gpus, std::vector<int>(numb_gpus));
    std::vector<int*> dev_haloElementsCount(numb_gpus, nullptr);
    std::vector<std::vector<int>> host_haloElementsCount(numb_gpus);
    std::vector<int*> send_halo_to_gpu(numb_gpus, nullptr);
    std::vector<std::vector<int>> send_halo_count;
    std::vector<std::vector<int>> receive_halo_count;
    std::vector<std::vector<float*>> receive_halo_buffer;
    std::vector<std::vector<float*>> send_halo_buffer;
    std::vector<float*> receive_halo_buffer_bulk(numb_gpus, nullptr);
    std::vector<float*> send_halo_buffer_bulk(numb_gpus, nullptr);
    std::vector<int> send_halo_elements_count;
    std::vector<int> receive_halo_elements_count;

    halo_elements(dev_haloElements, dneighbors, dev_gpu_ids, count,
                  host_gpu_ids, dev_haloElementsCount, send_halo_to_gpu, send_halo_count,
                  receive_halo_count, receive_halo_buffer, send_halo_buffer, receive_halo_buffer_bulk,
                  send_halo_buffer_bulk, maxNeighbors, count_per_device, partition_size, numb_gpus,
                  send_halo_elements_count, receive_halo_elements_count, host_haloElementsCount);

    for (int i = 0; i < numb_gpus; i++){
        std::cout << "Statistics for GPU:" << i << std::endl;
        for (int j = 0; j < numb_gpus; j++){
            std::cout << "Send halo elements (" << i << "->" << j << "):" << send_halo_count[i][j] << std::endl;
            std::cout << "Receive halo elements (" << j << "->" << i << "):" << receive_halo_count[i][j] << std::endl;
        }
    }

#pragma omp parallel for num_threads(numb_gpus)
    for (int dev_id = 0; dev_id < numb_gpus; dev_id++) {
        int start;
        cudaSetDevice(dev_id);
        int times = count_per_device[dev_id] / max_threads;
        for (int k = 0; k < iterationCount; k++) {
        //for (int k = 0; k < 1; k++) {
            //std::cout << k << std::endl;
            int threads = 1024;
            //std::cout << "number of elements:" << send_halo_elements_count[dev_id] << std::endl;
            int blocks = (int) ceil((1.0 * (send_halo_elements_count[dev_id])) / threads);
            gather_halo_elements<<<blocks, threads>>>(din_x[dev_id], din_y[dev_id], din_z[dev_id],
                                                      send_halo_to_gpu[dev_id], send_halo_buffer_bulk[dev_id], send_halo_elements_count[dev_id]);
            cudaDeviceSynchronize();
            //std::cout << "Finished..." << std::endl;
            sender_halo_break(send_halo_buffer[dev_id], send_halo_buffer_bulk[dev_id], send_halo_count[dev_id], numb_gpus);
            //std::cout << "Started communication..." << std::endl;
            communication_sender_to_receiver(dev_id, receive_halo_buffer, send_halo_buffer[dev_id], send_halo_count, numb_gpus, receive_halo_flags_lambda);
            //std::cout << "Finished communication..." << std::endl;
            while (thrust::reduce(thrust::device, receive_halo_flags_lambda[dev_id], receive_halo_flags_lambda[dev_id] + numb_gpus, 0) < numb_gpus - 1);
            //std::cout << "Finished waiting communication..." << std::endl;
            receiver_halo_synthesize(receive_halo_buffer[dev_id], receive_halo_buffer_bulk[dev_id], receive_halo_count[dev_id], numb_gpus,
                                     receive_halo_flags_lambda[dev_id]);
            start = 0;
            for (int i = 0; i < times; i++) {
                int threads = 1024;
                int blocks = (int) ceil((1.0 * (max_threads)) / threads);

                taubin_step_GKNN<<<blocks, threads>>>(din_x[dev_id], din_y[dev_id], din_z[dev_id],
                                                      dout_x[dev_id], dout_y[dev_id], dout_z[dev_id], lambda, dneighbors[dev_id],
                                                      maxNeighbors, isRegularized, start, dev_id * partition_size, max_threads,
                                                      dev_haloElements[dev_id], receive_halo_buffer_bulk[dev_id], dev_haloElementsCount[dev_id] ,
                                                      numb_gpus, partition_size, dev_id, dev_gpu_ids[dev_id]);
                cudaDeviceSynchronize();
                start += max_threads;
            }
            if (start < count_per_device[dev_id]) {
                int threads = 1024;
                int blocks = (int) ceil((1.0 * (count_per_device[dev_id] - start)) / threads);
                taubin_step_GKNN<<<blocks, threads>>>(din_x[dev_id], din_y[dev_id], din_z[dev_id],
                                                      dout_x[dev_id], dout_y[dev_id], dout_z[dev_id], lambda, dneighbors[dev_id],
                                                      maxNeighbors, isRegularized, start, dev_id * partition_size,
                                                      count_per_device[dev_id] - start,
                                                      dev_haloElements[dev_id], receive_halo_buffer_bulk[dev_id], dev_haloElementsCount[dev_id] ,
                                                      numb_gpus, partition_size, dev_id, dev_gpu_ids[dev_id]);
                cudaDeviceSynchronize();
            }
            std::swap(din_x[dev_id], dout_x[dev_id]);
            std::swap(din_y[dev_id], dout_y[dev_id]);
            std::swap(din_z[dev_id], dout_z[dev_id]);

            threads = 1024;
            blocks = (int) ceil((1.0 * (send_halo_elements_count[dev_id])) / threads);
            gather_halo_elements<<<blocks, threads>>>(din_x[dev_id], din_y[dev_id], din_z[dev_id],
                                                      send_halo_to_gpu[dev_id], send_halo_buffer_bulk[dev_id], send_halo_elements_count[dev_id]);
            cudaDeviceSynchronize();
            sender_halo_break(send_halo_buffer[dev_id], send_halo_buffer_bulk[dev_id], send_halo_count[dev_id], numb_gpus);
            communication_sender_to_receiver(dev_id, receive_halo_buffer, send_halo_buffer[dev_id], send_halo_count, numb_gpus, receive_halo_flags_mu);
            //std::cout << "Communication started..." << std::endl;
            while (thrust::reduce(thrust::device, receive_halo_flags_mu[dev_id], receive_halo_flags_mu[dev_id] + numb_gpus, 0) < numb_gpus - 1);
            //std::cout << "Communication finished..." << std::endl;
            receiver_halo_synthesize(receive_halo_buffer[dev_id], receive_halo_buffer_bulk[dev_id], receive_halo_count[dev_id], numb_gpus,
                                     receive_halo_flags_mu[dev_id]);
            start = 0;
            for (int i = 0; i < times; i++) {
                int threads = 1024;
                int blocks = (int) ceil((1.0 * (max_threads)) / threads);
                taubin_step_GKNN<<<blocks, threads>>>(din_x[dev_id], din_y[dev_id], din_z[dev_id],
                                                      dout_x[dev_id], dout_y[dev_id], dout_z[dev_id], mu, dneighbors[dev_id],
                                                      maxNeighbors, isRegularized, start, dev_id * partition_size, max_threads,
                                                      dev_haloElements[dev_id], receive_halo_buffer_bulk[dev_id], dev_haloElementsCount[dev_id] ,
                                                      numb_gpus, partition_size, dev_id, dev_gpu_ids[dev_id]);
                cudaDeviceSynchronize();
                start += max_threads;
            }
            if (start < count_per_device[dev_id]) {
                int threads = 1024;
                int blocks = (int) ceil((1.0 * (count_per_device[dev_id] - start)) / threads);
                taubin_step_GKNN<<<blocks, threads>>>(din_x[dev_id], din_y[dev_id], din_z[dev_id],
                                                      dout_x[dev_id], dout_y[dev_id], dout_z[dev_id], mu, dneighbors[dev_id],
                                                      maxNeighbors, isRegularized, start, dev_id * partition_size,
                                                      count_per_device[dev_id] - start,
                                                      dev_haloElements[dev_id], receive_halo_buffer_bulk[dev_id], dev_haloElementsCount[dev_id] ,
                                                      numb_gpus, partition_size, dev_id, dev_gpu_ids[dev_id]);
                cudaDeviceSynchronize();
            }
            std::swap(din_x[dev_id], dout_x[dev_id]);
            std::swap(din_y[dev_id], dout_y[dev_id]);
            std::swap(din_z[dev_id], dout_z[dev_id]);
        }
    }



#pragma omp parallel for num_threads(numb_gpus)
    for (int dev_id_r = 0; dev_id_r < numb_gpus; dev_id_r++){
        cudaSetDevice(dev_id_r);
        int counter = count_per_device[0];
        for (int dev_id = 0; dev_id < numb_gpus; dev_id++){
            if (dev_id != dev_id_r){
                cudaMemcpy(din_x[dev_id_r] + dev_id * partition_size, din_x[dev_id] + dev_id * partition_size,  count_per_device[dev_id] * sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpy(din_y[dev_id_r] + dev_id * partition_size, din_y[dev_id] + dev_id * partition_size,  count_per_device[dev_id] * sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpy(din_z[dev_id_r] + dev_id * partition_size, din_z[dev_id] + dev_id * partition_size,  count_per_device[dev_id] * sizeof(float), cudaMemcpyDeviceToDevice);
            }
        }
    }

    int threads = 1024;
	int blocks = (int)ceil((1.0 * count) / threads);
    produce_output_GKNN<<<blocks, threads>>>(din_x[0], din_y[0], din_z[0], dout_x[0], dout_y[0], dout_z[0], minOriginal, maxOriginal, count);
    cudaDeviceSynchronize();

    /*int threads = 1024;
	int blocks = (int)ceil((1.0 * count) / threads);
    produce_output_GKNN<<<blocks, threads>>>(dout_x[0], dout_y[0], dout_z[0], din_x[0], din_y[0], din_z[0], minOriginal, maxOriginal, count);
    cudaDeviceSynchronize();*/

    /*cudaMemcpy(din_x[0], dout_x[0], count * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(din_y[0], dout_y[0], count * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(din_z[0], dout_z[0], count * sizeof(float), cudaMemcpyDeviceToDevice);*/

    rearrange_output<<<blocks, threads>>>(din_x[0], din_y[0], din_z[0],
                                          dout_x[0], dout_y[0], dout_z[0], count, reverseIndexes);
    cudaDeviceSynchronize();

    cudaMemcpy(out_x, din_x[0], size, cudaMemcpyDeviceToHost);
    cudaMemcpy(out_y, din_y[0], size, cudaMemcpyDeviceToHost);
    cudaMemcpy(out_z, din_z[0], size, cudaMemcpyDeviceToHost);

    /*cudaFree(din_x); cudaFree(din_y); cudaFree(din_z);
    cudaFree(dout_x); cudaFree(dout_y); cudaFree(dout_z);
    cudaFree(dneighbors);*/
}
