//
// Created by agalex on 11/10/21.
//

#ifndef SMOOTHINGOFPOINTCLOUD_EGT_CUH
#define SMOOTHINGOFPOINTCLOUD_EGT_CUH

void EGTsmoothing_GKNN(float** din_x, float** din_y, float** din_z, int* count_per_device, float lambda, float mu,
                       int** dneighbors, int maxNeighbors, float* out_x, float* out_y, float* out_z, int count,
                       int iterationCount, int isRegularized, float ratio, int numb_gpus, int partition_size, int* reverseIndexes,
                       float minOriginal, float maxOriginal);

#endif //SMOOTHINGOFPOINTCLOUD_EGT_CUH
