//
// Created by agalex on 7/17/23.
//

#ifndef KNNCUDA_GPU_NORMAL_CUH
#define KNNCUDA_GPU_NORMAL_CUH

void GPU_NORMAL(float** din_x, float** din_y, float** din_z, int* count_per_device,
                int** dneighbors, int maxNeighbors, int k_use, float* out_x, float* out_y, float* out_z, int count,
                float* normal_x, float* normal_y, float* normal_z,
                int numb_gpus, int partition_size, int* reverseIndexes,
                float minOriginal, float maxOriginal);

#endif //KNNCUDA_GPU_NORMAL_CUH
