//
// Created by agalex on 7/20/23.
//

#ifndef KNNCUDA_MST_H
#define KNNCUDA_MST_H

void orient(int* knn, float* nx, float* ny, float* nz, float* x, float* y, float* z,
            int N, int k, int k_use);

void orient_kruskal(int* knn, float* nx, float* ny, float* nz, float* x, float* y, float* z,
                    int N, int k, int k_use);

#endif //KNNCUDA_MST_H
