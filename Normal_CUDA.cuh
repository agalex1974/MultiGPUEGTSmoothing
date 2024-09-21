//
// Created by agalex on 6/26/24.
//

#ifndef KNNCUDA_NORMAL_CUDA_CUH
#define KNNCUDA_NORMAL_CUDA_CUH

#include "KNNInterface.cuh"

class Normal_CUDA {
public:
    explicit Normal_CUDA(KNNInterface& knnInterface);
    void GetNormals(float* normal_x, float* normal_y, float* normal_z);
private:
    float** din_x;
    float** din_y;
    float** din_z;
    int* point_count_per_device;
    int* originalIndexes;
    int** dneighbors;
    int k;
    int pointCount;
    int numb_gpus;
    int partition_size;
};


#endif //KNNCUDA_NORMAL_CUDA_CUH
