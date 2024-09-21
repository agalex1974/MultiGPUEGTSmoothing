//
// Created by agalex on 7/27/23.
//

#ifndef KNNCUDA_BFS_CUH
#define KNNCUDA_BFS_CUH

void BFS(int source, int* edges_d, int* dest_d, int N, float* nx, float* ny, float* nz,
         int* visitedVertices);

#endif //KNNCUDA_BFS_CUH
