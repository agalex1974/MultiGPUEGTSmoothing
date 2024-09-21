//
// Created by agalex on 6/20/24.
//

#include "getOutputPoints.cuh"

namespace {

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

    __global__
    void rearrange_output(float* xout, float* yout, float* zout, float* xin, float* yin, float* zin, int count, int* originalIndexes){
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < count){
            int idx = originalIndexes[i];
            xout[idx] = xin[i];
            yout[idx] = yin[i];
            zout[idx] = zin[i];
        }
    }

}

void getOutputPoints(float* pnts_x, float* pnts_y, float* pnts_z, KNNInterface* knnInterface){
    int count = knnInterface->pointsRefCount();
    float minOriginal = knnInterface->GetMinExtent();
    float maxOriginal = knnInterface->GetMaxExtent();
    cudaSetDevice(0);
    float* din_x = knnInterface->GetRefPointsX()[0];
    float* din_y = knnInterface->GetRefPointsY()[0];
    float* din_z = knnInterface->GetRefPointsZ()[0];
    float* temp_x, *temp_y, *temp_z;
    cudaMalloc((void**)&temp_x, count * sizeof(float));
    cudaMalloc((void**)&temp_y, count * sizeof(float));
    cudaMalloc((void**)&temp_z, count * sizeof(float));

    int threads = 1024;
    int blocks = (int)ceil((1.0 * count) / threads);
    produce_output_GKNN<<<blocks, threads>>>(din_x, din_y, din_z, temp_x, temp_y, temp_z, minOriginal, maxOriginal, count);
    cudaDeviceSynchronize();

    rearrange_output<<<blocks, threads>>>(din_x, din_y, din_z,
                                          temp_x, temp_y, temp_z, count, knnInterface->GetOriginalRefIndexes()[0]);
    cudaDeviceSynchronize();

    cudaMemcpy(pnts_x, din_x, count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(pnts_y, din_y, count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(pnts_z, din_z, count * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(temp_x);
    cudaFree(temp_y);
    cudaFree(temp_z);
}