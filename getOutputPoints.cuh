//
// Created by agalex on 6/20/24.
//

#ifndef KNNCUDA_GETOUTPUTPOINTS_CUH
#define KNNCUDA_GETOUTPUTPOINTS_CUH

#include "KNNInterface.cuh"

void getOutputPoints(float* pnts_x, float* pnts_y, float* pnts_z, KNNInterface* knnInterface);
#endif //KNNCUDA_GETOUTPUTPOINTS_CUH
