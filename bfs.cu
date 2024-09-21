//
// Created by agalex on 7/27/23.
//
#include <iostream>
#include <vector>
#define BLOCK_QUEUE_SIZE 512

__global__
void BFS_Bqueue_kernel (int *p_frontier, int *p_frontier_tail, int *c_frontier,
                        int *c_frontier_tail, int *edges, int *dest, int *label, int* visited,
                        float* nx, float* ny, float* nz, int* visitedVertices) {
    __shared__ int c_frontier_s[BLOCK_QUEUE_SIZE];
    __shared__ int c_frontier_tail_s, our_c_frontier_tail;
    if (threadIdx.x == 0) c_frontier_tail_s = 0;
    __syncthreads();
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < *p_frontier_tail) {
        const int my_vertex = p_frontier[tid];
        for (int i = edges[my_vertex]; i < edges[my_vertex + 1]; ++i) {
            int dest_vertex = dest[i];
            const int was_visited = atomicExch(&(visited[dest_vertex]), 1);
            if (!was_visited) {
                float dot = nx[my_vertex] * nx[dest_vertex] + ny[my_vertex] * ny[dest_vertex] +
                        nz[my_vertex] * nz[dest_vertex];
                visitedVertices[my_vertex] = 1;
                visitedVertices[dest_vertex] = 1;
                if (dot < 0){
                    nx[dest_vertex] = -nx[dest_vertex];
                    ny[dest_vertex] = -ny[dest_vertex];
                    nz[dest_vertex] = -nz[dest_vertex];
                }
                label[dest_vertex] = label[my_vertex] + 1;
                const int my_tail = atomicAdd(&c_frontier_tail_s, 1);
                if (my_tail < BLOCK_QUEUE_SIZE) {
                    c_frontier_s[my_tail] = dest[i];
                } else { // If full, add it to the global queue directly
                    c_frontier_tail_s = BLOCK_QUEUE_SIZE;
                    const int my_global_tail = atomicAdd(c_frontier_tail, 1);
                    c_frontier[my_global_tail] = dest_vertex;
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        our_c_frontier_tail = atomicAdd(c_frontier_tail, c_frontier_tail_s);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < c_frontier_tail_s; i += blockDim.x) {
        c_frontier[our_c_frontier_tail + i] = c_frontier_s[i];
    }
}

template <typename T>
__global__
void setValue_bfs(T* vector, T value, int count){
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < count){
        vector[i] = value;
    }
}

void BFS(int source, int* edges_d, int* dest_d, int N, float* nx, float* ny, float* nz,
         int* visitedVertices){
    int MAX_FRONTIER_SIZE = N;
    int* frontier_d;
    cudaMalloc((void**)&frontier_d, 2 * N * sizeof(int));
    int *c_frontier_tail_d, *p_frontier_tail_d;
    cudaMalloc((void**)&c_frontier_tail_d, sizeof(int));
    cudaMalloc((void**)&p_frontier_tail_d, sizeof(int));
    int* c_frontier_d = &frontier_d[0];
    int* p_frontier_d = &frontier_d[MAX_FRONTIER_SIZE];
    int c_frontier_tail = 0;
    int p_frontier_tail = 1;
    cudaMemcpy(p_frontier_d, &source, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(c_frontier_tail_d, &c_frontier_tail, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(p_frontier_tail_d, &p_frontier_tail, sizeof(int), cudaMemcpyHostToDevice);
    int* label_d;
    cudaMalloc((void**)&label_d, N * sizeof(int));
    int threads = 1024;
    int blocks = ceil((1.0 * N) / threads);
    setValue_bfs<int><<<blocks, threads>>>(label_d, -1, N);
    cudaDeviceSynchronize();
    cudaMemcpy(label_d + source, &c_frontier_tail, sizeof(int), cudaMemcpyHostToDevice);
    int* visited_d;
    cudaMalloc((void**)&visited_d, N * sizeof(int));
    cudaMemset(visited_d, 0, N * sizeof(int));
    cudaMemcpy(visited_d + source, &p_frontier_tail, sizeof(int), cudaMemcpyHostToDevice);
    int iter = 0;
    while (p_frontier_tail > 0){
        //std::vector<int> parents(p_frontier_tail);
        //cudaMemcpy(parents.data(), )
        int threads = 256;
        int blocks = ceil((1.0 * p_frontier_tail) / threads);
        BFS_Bqueue_kernel<<<blocks, threads>>>(p_frontier_d, p_frontier_tail_d,
                                               c_frontier_d, c_frontier_tail_d,
                                               edges_d, dest_d, label_d, visited_d, nx, ny, nz,
                                               visitedVertices);
        cudaDeviceSynchronize();
        cudaMemcpy(&p_frontier_tail, c_frontier_tail_d, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(p_frontier_tail_d, c_frontier_tail_d, sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemset(c_frontier_tail_d, 0, sizeof(int));
        std::swap(c_frontier_d, p_frontier_d);
        //std::cout << p_frontier_tail << std::endl;
    }
    cudaFree(frontier_d);
    cudaFree(p_frontier_tail_d);
    cudaFree(c_frontier_tail_d);
    cudaFree(label_d);
    cudaFree(visited_d);
}