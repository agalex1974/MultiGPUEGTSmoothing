//
// Created by agalex on 6/26/24.
//
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <set>
#include <queue>
#include "definitions.h"
#include "EigenSolver.cuh"
#include "CUDA_MST.cuh"
#include "bfs.cuh"
#include "Normal_CUDA.cuh"

typedef unsigned long long int int_64;

namespace{

    struct ComponentGraphEnd{
        int endIndex;
        int orient;
        bool operator<(const ComponentGraphEnd& other) const {
            return endIndex < other.endIndex;
        }
    };

    struct ComponentGraphParent{
        int parentIndex;
        int orient;
        /*bool operator<(const ComponentGraphParent& other) const {
            return parentIndex < other.parentIndex;
        }*/
    };

    __device__
    int2 order_pair(int i, int j){
        int2 a;
        if (i < j) {
            a.x = i;
            a.y = j;
            return a;
        }
        a.x = j;
        a.y = i;
        return a;
    }

    __device__
    bool isHaloPoint(uint32_t idx, uint32_t advance, uint32_t pointsInCard){
        if (idx < advance) return true;
        uint32_t indexInCard = idx - advance;
        if (indexInCard < pointsInCard) return false;
        return true;
    }

    __global__
    void getEdgeNumber(int* knn, int k, int* edgeNumbers, int pointsInCard, uint32_t advance){
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < pointsInCard){
            bool valid_index = true;
            for (int j = 0; j < k; j++){
                int n = knn[i * k + j];
                if (n < 0) continue;
                if (isHaloPoint(n, advance, pointsInCard)){
                    valid_index = false;
                    break;
                }
            }
            if (valid_index) {
                for (int j = 0; j < k; j++) {
                    int n = knn[i * k + j];
                    if (n >= 0) {
                        uint32_t idxInCard = n - advance;
                        int2 tuple = order_pair(i, idxInCard);
                        atomicAdd(&edgeNumbers[tuple.x], 1);
                    }
                }
            }
        }
    }

    __global__
    void calculateNormal(int* knn, int k, float* x, float* y, float* z,
                         float* nx, float* ny, float* nz, int count, int advance){
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i < count) {
            int realIdx = i + advance;
            float sumX = x[realIdx], sumY = y[realIdx], sumZ = z[realIdx];
            int counter = 1;
            for (int j = 0; j < k; j++) {
                int n = knn[i * k + j];
                if (n >= 0) {
                    sumX += x[n];
                    sumY += y[n];
                    sumZ += z[n];
                    counter++;
                }
            }
            if (counter){
                float cx = sumX / counter;
                float cy = sumY / counter;
                float cz = sumZ / counter;
                float covarianceMatrix[9];
                memset(covarianceMatrix, 0, 9 * sizeof(float));
                for (int j = 0; j < k; j++) {
                    int n = knn[i * k + j];
                    if (n >= 0) {
                        float dx = x[n] - cx;
                        float dy = y[n] - cy;
                        float dz = z[n] - cz;
                        covarianceMatrix[0] += dx * dx;
                        covarianceMatrix[1] += dx * dy;
                        covarianceMatrix[2] += dx * dz;
                        //covarianceMatrix[3] += dy * dx;
                        covarianceMatrix[4] += dy * dy;
                        covarianceMatrix[5] += dy * dz;
                        //covarianceMatrix[6] += dz * dx;
                        //covarianceMatrix[7] += dz * dy;
                        covarianceMatrix[8] += dz * dz;
                    }
                }

                float eigvalues[3];
                float eigvectors[3][3];

                SymmetricEigensolver3x3<float> sv;
                sv(covarianceMatrix[0], covarianceMatrix[1], covarianceMatrix[2], covarianceMatrix[4], covarianceMatrix[5], covarianceMatrix[8],
                   false, 1, eigvalues, eigvectors);

                float nxx = eigvectors[0][0];
                float nyy = eigvectors[0][1];
                float nzz = eigvectors[0][2];
                float norm = sqrt(nxx * nxx + nyy * nyy + nzz * nzz);
                nx[i] = nxx / norm;
                ny[i] = nyy / norm;
                nz[i] = nzz / norm;
            }
        }
    }

    __global__
    void fillEdges(int* knn, int k, int* edgeNumbers, int* offsets, wghEdge<int>* edges, int pointsInCard,
                   float* nx, float* ny, float* nz, int_64* hash, int advance){
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < pointsInCard){
            bool valid_index = true;
            for (int j = 0; j < k && valid_index; j++){
                int n = knn[i * k + j];
                if (n >= 0 && isHaloPoint(n, advance, pointsInCard)){
                    valid_index = false;
                    break;
                }
            }
            if (valid_index) {
                for (int j = 0; j < k; j++) {
                    int n = knn[i * k + j];
                    if (n >= 0) {
                        uint32_t idxInCard = n - advance;
                        int2 tuple = order_pair(i, idxInCard);
                        int position = atomicAdd(&edgeNumbers[tuple.x], 1);
                        position += offsets[tuple.x];
                        float dot_product = nx[i] * nx[idxInCard] + ny[i] * ny[idxInCard] + nz[i] * nz[idxInCard];
                        float weight = fmaxf(0.0f, 1.0f - fabsf(dot_product));
                        edges[position] = wghEdge<int>(tuple.x, tuple.y, weight);
                        hash[position] = (int_64) tuple.x * (int_64) pointsInCard + (int_64) tuple.y;
                    }
                }
            }
        }
    }

    __global__
    void getWeights(wghEdge<int>* edges, int count, float* weights){
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < count){
            weights[i] = edges[i].weight;
        }
    }

    __global__
    void createMSTGraph(int* knn, int k, wghEdge<int>* edges, int* indexes, int count, int* vertexCounter,
                        int* touchedVertices){
        int t = threadIdx.x + blockIdx.x * blockDim.x;
        if (t < count){
            int i = indexes[t];
            wghEdge<int> edge = edges[i];
            int u = edge.u;
            int v = edge.v;
            int positionv = atomicAdd(&vertexCounter[u], 1);
            int positionu = atomicAdd(&vertexCounter[v], 1);
            knn[u * k + positionv] = v;
            knn[v * k + positionu] = u;
            touchedVertices[v] = 1;
            touchedVertices[u] = 1;
        }
    }

    __global__
    void createCSRDEST(int* knn, int k, int count, int* csr_edges, int* csr_dest){
        int t = threadIdx.x + blockIdx.x * blockDim.x;
        if (t < count){
            int i = t;
            int offset = csr_edges[i];
            int counter = 0;
            for (int j = 0; j < k; j++){
                int dest = knn[i * k + j];
                if (dest >= 0){
                    csr_dest[offset + counter++] = dest;
                }
                else break;
            }
        }
    }

    __global__
    void rearrange_output(float* nxout, float* nyout, float* nzout, float* nxin, float* nyin, float* nzin,
                          int count, int* originalIndexes){
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < count){
            int idx = originalIndexes[i];
            nxout[idx] = nxin[i];
            nyout[idx] = nyin[i];
            nzout[idx] = nzin[i];
        }
    }

    __global__
    void findNextRoot(int* visited, int* touched, int count, int* root){
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < count) {
            if (touched[i] && !visited[i]){
                *root = i;
            }
        }
    }

    __global__
    void getConnectedComponent(int* flags, int* positions, int* componentIndexes,
                               int advance, int count){
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < count){
            //if (!i) printf("I have run\n");
            if (flags[i]){
                int idx = positions[i];
                componentIndexes[idx] = i + advance;
            }
        }
    }

    __global__
    void addToVisited(int* visited, int* flags, int count){
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < count){
            visited[i] |= flags[i];
        }
    }

    __global__
    void fillVertexToComponentBuffer(int* component, int* vertToCompBuffer, int componentIdx, int count){
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < count){
            vertToCompBuffer[component[i]] = componentIdx;
        }
    }

    __global__
    void componentCounter(int* component, int* dknn, int advance, int* compCounter, int* vertToCompBuffer,
                          int componentIdx, int k, int count, float* nx, float* ny, float* nz, int compCount){
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < count){
            int test[10] = {0};
            int idx = component[i] - advance;
            for (int n = 0; n < k; n++){
                int neighIdx = dknn[k * idx + n];
                if (neighIdx < 0) continue;
                int neighComp = vertToCompBuffer[neighIdx];
                if (neighComp < 0 || neighComp == componentIdx) continue;
                int trueIndex = idx + advance;
                float dot_product = nx[trueIndex] * nx[neighIdx] + ny[trueIndex] * ny[neighIdx]
                            + nz[trueIndex] * nz[neighIdx];
                int count1 = dot_product > 0 ? 1 : -1;
                atomicAdd(&compCounter[neighComp], count1);
            }
        }
    }

    __global__
    void negateNormals(int* component, float* nx, float* ny, float* nz, int advance, int count){
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < count){
            int idx = component[i] - advance;
            nx[idx] = -nx[idx];
            ny[idx] = -ny[idx];
            nz[idx] = -nz[idx];
        }
    }
}

void Normal_CUDA::GetNormals(float* normal_x, float* normal_y, float* normal_z) {
    std::vector<float *> dnormalv_x(numb_gpus);
    float **dnormal_x = dnormalv_x.data();
    std::vector<float *> dnormalv_y(numb_gpus);
    float **dnormal_y = dnormalv_y.data();
    std::vector<float *> dnormalv_z(numb_gpus);
    float **dnormal_z = dnormalv_z.data();
    std::vector<std::vector<int *>> connected_components(numb_gpus);
    std::vector<std::vector<int>> connected_components_count(numb_gpus);
    std::vector<std::vector<int>> connected_components_global_number(numb_gpus);
#pragma omp parallel for num_threads(numb_gpus)
    for (int dev_id = 0; dev_id < numb_gpus; dev_id++) {
        cudaSetDevice(dev_id);

        cudaMalloc((void **) &dnormal_x[dev_id], point_count_per_device[dev_id] * sizeof(float));
        cudaMalloc((void **) &dnormal_y[dev_id], point_count_per_device[dev_id] * sizeof(float));
        cudaMalloc((void **) &dnormal_z[dev_id], point_count_per_device[dev_id] * sizeof(float));

        int threads = 1024;
        int blocks = (int) ceil((1.0 * (point_count_per_device[dev_id])) / threads);
        calculateNormal<<<blocks, threads>>>(dneighbors[dev_id], k, din_x[dev_id], din_y[dev_id], din_z[dev_id],
                                             dnormal_x[dev_id], dnormal_y[dev_id], dnormal_z[dev_id],
                                             point_count_per_device[dev_id], dev_id * partition_size);
        cudaDeviceSynchronize();

        int *edgeCounter;
        cudaMalloc((void **) &edgeCounter, point_count_per_device[dev_id] * sizeof(int));
        cudaMemset(edgeCounter, 0, point_count_per_device[dev_id] * sizeof(int));
        threads = 1024;
        blocks = (int) ceil((1.0 * point_count_per_device[dev_id]) / threads);
        getEdgeNumber<<<blocks, threads>>>(dneighbors[dev_id], k, edgeCounter, point_count_per_device[dev_id],
                                           dev_id * partition_size);
        cudaDeviceSynchronize();
        int edgeCount = thrust::reduce(thrust::device, edgeCounter, edgeCounter + point_count_per_device[dev_id],
                                       0, thrust::plus<int>());
        std::cout << "PASSED1:" << edgeCount << std::endl;

        wghEdge<int> *edges;
        cudaMalloc((void **) &edges, edgeCount * sizeof(wghEdge<int>));

        int *edgeOffset;
        cudaMalloc((void **) &edgeOffset, point_count_per_device[dev_id] * sizeof(int));

        thrust::exclusive_scan(thrust::device, edgeCounter, edgeCounter + point_count_per_device[dev_id],
                               edgeOffset, 0, thrust::plus<int>());
        cudaMemset(edgeCounter, 0, point_count_per_device[dev_id] * sizeof(int));

        int_64 *hash;
        cudaMalloc((void **) &hash, edgeCount * sizeof(int_64));
        std::cout << "PASSED2:" << std::endl;

        threads = 1024;
        blocks = (int) ceil((1.0 * point_count_per_device[dev_id]) / threads);
        fillEdges<<<blocks, threads>>>(dneighbors[dev_id], k, edgeCounter, edgeOffset, edges,
                                       point_count_per_device[dev_id],
                                       dnormal_x[dev_id], dnormal_y[dev_id], dnormal_z[dev_id], hash,
                                       dev_id * partition_size);
        cudaDeviceSynchronize();
        std::cout << "PASSED3:" << std::endl;
        thrust::sort_by_key(thrust::device, hash, hash + edgeCount, edges,
                            thrust::less<int_64>());

        thrust::pair<int_64 *, wghEdge<int> *> end;
        end = thrust::unique_by_key(thrust::device, hash, hash + edgeCount,
                                    edges, thrust::equal_to<int_64>());

        int_64 edgeCountUnique = end.first - hash;
#pragma omp critical
        std::cout << "Unique edges:" << edgeCountUnique << std::endl;

        wghEdge<int> *edgesUnique;
        cudaMalloc((void **) &edgesUnique, edgeCountUnique * sizeof(wghEdge<int>));
        cudaMemcpy(edgesUnique, edges, edgeCountUnique * sizeof(wghEdge<int>), cudaMemcpyDeviceToDevice);
        float *weights;
        cudaMalloc((void **) &weights, edgeCountUnique * sizeof(float));
        threads = 1024;
        blocks = (int) ceil((1.0 * edgeCountUnique) / threads);
        getWeights<<<blocks, threads>>>(edgesUnique, edgeCountUnique, weights);
        cudaDeviceSynchronize();

        thrust::sort_by_key(thrust::device, weights, weights + edgeCountUnique, edgesUnique,
                            thrust::less<float>());

        cudaFree(edgeCounter);
        cudaFree(edgeOffset);
        cudaFree(edges);
        cudaFree(weights);
        cudaFree(hash);
        wghEdgeArray<int> G(edgesUnique, point_count_per_device[dev_id], edgeCountUnique);
        auto pr = mst(G);
        std::cout << "MST edges:" << pr.second << std::endl;

        int *dmstIndexes;
        cudaMalloc((void **) &dmstIndexes, pr.second * sizeof(int));
        cudaMemcpy(dmstIndexes, pr.first, pr.second * sizeof(int), cudaMemcpyHostToDevice);

        int *mstGraph;
        cudaMalloc((void **) &mstGraph, k * point_count_per_device[dev_id] * sizeof(int));
        thrust::fill(thrust::device, mstGraph, mstGraph + k * point_count_per_device[dev_id], -1);

        int *csr_edges;
        cudaMalloc((void **) &csr_edges, (point_count_per_device[dev_id] + 1) * sizeof(int));
        cudaMemset(csr_edges, 0, (point_count_per_device[dev_id] + 1) * sizeof(int));

        int *touched_vertices;
        cudaMalloc((void **) &touched_vertices, point_count_per_device[dev_id] * sizeof(int));
        cudaMemset(touched_vertices, 0, point_count_per_device[dev_id] * sizeof(int));

        threads = 1024;
        blocks = ceil((1.0 * pr.second / threads));
        createMSTGraph<<<blocks, threads>>>(mstGraph, k, edgesUnique, dmstIndexes, pr.second, csr_edges,
                                            touched_vertices);
        cudaDeviceSynchronize();

        int touchedCount = thrust::reduce(thrust::device, touched_vertices,
                                          touched_vertices + point_count_per_device[dev_id], 0);

#pragma omp critical
        std::cout << "GPU:" << dev_id << ". Number of touched vertices:" << touchedCount << std::endl;

        thrust::exclusive_scan(thrust::device, csr_edges, csr_edges + point_count_per_device[dev_id] + 1,
                               csr_edges);
        int numb_edges = 0;
        cudaMemcpy(&numb_edges, csr_edges + point_count_per_device[dev_id], sizeof(int), cudaMemcpyDeviceToHost);
        std::cout << "Number of edges:" << numb_edges << std::endl;

        int *csr_dest;
        cudaMalloc((void **) &csr_dest, numb_edges * sizeof(int));
        threads = 1024;
        blocks = ceil((1.0 * point_count_per_device[dev_id]) / threads);
        createCSRDEST<<<blocks, threads>>>(mstGraph, k, point_count_per_device[dev_id], csr_edges, csr_dest);
        cudaDeviceSynchronize();
        std::cout << "Finished!";

        wghEdge<int> e_root;
        cudaMemcpy(&e_root, &edgesUnique[0], sizeof(wghEdge<int>), cudaMemcpyDeviceToHost);
        int root = e_root.u;
        std::cout << "orienting...";
        //exit(0);
        int *visited_vertices;
        int *component_flags;
        int *component_indexes;
        cudaMalloc((void **) &visited_vertices, point_count_per_device[dev_id] * sizeof(int));
        cudaMalloc((void **) &component_flags, point_count_per_device[dev_id] * sizeof(int));
        cudaMalloc((void **) &component_indexes, point_count_per_device[dev_id] * sizeof(int));
        cudaMemset(visited_vertices, 0, point_count_per_device[dev_id] * sizeof(int));

        int *droot;
        cudaMalloc((void **) &droot, sizeof(int));
        int visitedCount = 0;
        while (true) {
            std::cout << "INTO " << point_count_per_device[dev_id] << std::endl;
            cudaMemset(component_flags, 0, point_count_per_device[dev_id] * sizeof(int));
            BFS(root, csr_edges, csr_dest, point_count_per_device[dev_id],
                dnormal_x[dev_id], dnormal_y[dev_id], dnormal_z[dev_id], component_flags);
            int connectedComponentCount = thrust::reduce(thrust::device, component_flags,
                                                         component_flags + point_count_per_device[dev_id], 0);
            int *connectedComponent;
            cudaMalloc((void **) &connectedComponent, connectedComponentCount * sizeof(int));
            thrust::exclusive_scan(thrust::device, component_flags,
                                   component_flags + point_count_per_device[dev_id], component_indexes);
            int threads = 1024;
            int blocks = (int) ceil((1.0 * (point_count_per_device[dev_id])) / threads);
            getConnectedComponent<<<blocks, threads>>>(component_flags, component_indexes,
                                                       connectedComponent, dev_id * partition_size,
                                                       point_count_per_device[dev_id]);
            cudaDeviceSynchronize();



            connected_components[dev_id].push_back(connectedComponent);
            connected_components_count[dev_id].push_back(connectedComponentCount);
            int i = connected_components_count[dev_id].size() - 1;
            std::vector<int> host_connected_component(connected_components_count[dev_id][i]);
            std::cout << "Number of elements" << connected_components_count[dev_id][i] << std::endl;
            cudaMemcpy(host_connected_component.data(), connected_components[dev_id][i], connected_components_count[dev_id][i] * sizeof(int), cudaMemcpyDeviceToHost);
            for (int j = 0; j < host_connected_component.size(); j++){
                if (host_connected_component[j] < dev_id * partition_size || host_connected_component[j] - dev_id * partition_size >= point_count_per_device[dev_id]){
                    std::cout << "Oh God you have a bug mate! : " << host_connected_component[j] - dev_id * partition_size << " " << point_count_per_device[dev_id] << std::endl;
                    exit(0);
                }
            }
            addToVisited<<<blocks, threads>>>(visited_vertices, component_flags, point_count_per_device[dev_id]);
            cudaDeviceSynchronize();
            visitedCount = thrust::reduce(thrust::device, visited_vertices,
                                          visited_vertices + point_count_per_device[dev_id], 0);
            if (visitedCount < touchedCount) {
                int threads = 1024;
                int blocks = (int) ceil((1.0 * (point_count_per_device[dev_id])) / threads);
                findNextRoot<<<blocks, threads>>>(visited_vertices, touched_vertices,
                                                  point_count_per_device[dev_id], droot);
                cudaDeviceSynchronize();
                cudaMemcpy(&root, droot, sizeof(int), cudaMemcpyDeviceToHost);
            } else break;
        }

#pragma omp critical
        std::cout << "GPU:" << dev_id << ". Number of visited vertices:" << visitedCount << std::endl;

        cudaFree(csr_edges);
        cudaFree(csr_dest);
        cudaFree(mstGraph);
        cudaFree(dmstIndexes);
        cudaFree(component_indexes);
        cudaFree(component_flags);
        cudaFree(visited_vertices);
        cudaFree(touched_vertices);
        cudaFree(droot);
    }

    std::cout << "Stage1" << std::endl;
    std::cout << connected_components_count[0].size() << std::endl;
    std::cout << connected_components[0].size() << std::endl;
    int componentCount = 0;
    std::vector<int> componentMembership;
    std::vector<int *> vertexToComponent(numb_gpus);
    std::vector<int *> componentToComponentBuffer;
    std::vector<int> componentToComponentCount;
    for (int dev_id = 0; dev_id < numb_gpus; dev_id++) {
        cudaSetDevice(dev_id);
        cudaMalloc((void **) &vertexToComponent[dev_id], pointCount * sizeof(int));
        if (!dev_id) {
            thrust::fill(thrust::device, vertexToComponent[dev_id], vertexToComponent[dev_id] + pointCount, -1);
        }
        if (dev_id > 0) {
            cudaMemcpy(vertexToComponent[dev_id], vertexToComponent[dev_id - 1], pointCount * sizeof(int),
                       cudaMemcpyDeviceToDevice);
        }
        for (int i = 0; i < connected_components[dev_id].size(); i++) {
            int threads = 1024;
            int blocks = (int) ceil((1.0 * (connected_components_count[dev_id][i])) / threads);
            fillVertexToComponentBuffer<<<blocks, threads>>>(connected_components[dev_id][i],
                                                             vertexToComponent[dev_id], componentCount,
                                                             connected_components_count[dev_id][i]);
            cudaDeviceSynchronize();
            componentMembership.push_back(dev_id);
            componentToComponentBuffer.push_back(connected_components[dev_id][i]);
            componentToComponentCount.push_back(connected_components_count[dev_id][i]);
            connected_components_global_number[dev_id].push_back(componentCount);
            componentCount++;
        }
    }

    for (int dev_id = 0; dev_id < numb_gpus - 1; dev_id++) {
        cudaSetDevice(dev_id);
        cudaMemcpy(vertexToComponent[dev_id], vertexToComponent[numb_gpus - 1], pointCount * sizeof(int),
                   cudaMemcpyDeviceToDevice);
        std::cout << dev_id << componentCount << std::endl;
        std::vector<int> host_temp(pointCount);
        cudaMemcpy(host_temp.data(), vertexToComponent[dev_id], pointCount * sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < componentCount; i++){
            if (host_temp[i] >= 0 && host_temp[i] >= componentCount){
                std::cout << "Alert!" << host_temp[i] << std::endl;
            }
        }
    }


    std::cout << "Stage2" << std::endl;

    std::vector<float *> dnorm_x(numb_gpus);
    float **dpnorm_x = dnorm_x.data();
    std::vector<float *> dnorm_y(numb_gpus);
    float **dpnorm_y = dnorm_y.data();
    std::vector<float *> dnorm_z(numb_gpus);
    float **dpnorm_z = dnorm_z.data();
    std::vector<std::set<ComponentGraphEnd>> componentGraph(componentCount);

#pragma omp parallel for num_threads(numb_gpus)
    for (int dev_id = 0; dev_id < numb_gpus; dev_id++) {
        cudaSetDevice(dev_id);
        std::vector<std::set<ComponentGraphEnd>> componentGPUGraph(componentCount);
        cudaMalloc((void **) &dpnorm_x[dev_id], pointCount * sizeof(float));
        cudaMalloc((void **) &dpnorm_y[dev_id], pointCount * sizeof(float));
        cudaMalloc((void **) &dpnorm_z[dev_id], pointCount * sizeof(float));
        for (int dev_id1 = 0; dev_id1 < numb_gpus; dev_id1++) {
            cudaMemcpy(dpnorm_x[dev_id] + dev_id1 * partition_size, dnormal_x[dev_id1],
                       point_count_per_device[dev_id1] * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(dpnorm_y[dev_id] + dev_id1 * partition_size, dnormal_y[dev_id1],
                       point_count_per_device[dev_id1] * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(dpnorm_z[dev_id] + dev_id1 * partition_size, dnormal_z[dev_id1],
                       point_count_per_device[dev_id1] * sizeof(float), cudaMemcpyDeviceToDevice);
        }
        for (int i = 0; i < connected_components[dev_id].size(); i++) {
#pragma omp critical
            {
                std::cout << dev_id << " " << i << std::endl;
            }
            int component_idx = connected_components_global_number[dev_id][i];
            std::cout << "component index:" << component_idx << std::endl;
            int* components_counter;
            std::cout << componentCount << std::endl;
            cudaMalloc((void **)& components_counter, componentCount * sizeof(int));
            cudaMemset(components_counter, 0, componentCount * sizeof(int));
            int threads = 1024;
            int blocks = (int) ceil((1.0 * (connected_components_count[dev_id][i])) / threads);
            componentCounter<<<blocks, threads>>>(connected_components[dev_id][i], dneighbors[dev_id],
                                                  dev_id * partition_size, components_counter,
                                                  vertexToComponent[dev_id], component_idx, k,
                                                  connected_components_count[dev_id][i],
                                                  dpnorm_x[dev_id], dpnorm_y[dev_id], dpnorm_z[dev_id], componentCount);

            std::vector<int> host_connected_component(connected_components_count[dev_id][i]);
            cudaMemcpy(host_connected_component.data(), connected_components[dev_id][i], connected_components_count[dev_id][i] * sizeof(int), cudaMemcpyDeviceToHost);
            for (int j = 0; j < host_connected_component.size(); j++){
                if (host_connected_component[j] < dev_id * partition_size || host_connected_component[j] - dev_id * partition_size >= point_count_per_device[dev_id]){
                    std::cout << dev_id << " " << i << std::endl;
                    std::cout << "Oh God you have a bug mate! : " << host_connected_component[j] - dev_id * partition_size << " " << point_count_per_device[dev_id] << std::endl;
                    exit(0);
                }
            }

            cudaDeviceSynchronize();
            std::vector<int> host_component_counter(componentCount);
            cudaMemcpy(host_component_counter.data(), components_counter, componentCount * sizeof(int),
                       cudaMemcpyDeviceToHost);
#pragma omp critical
            {
                std::cout << dev_id << " " << i << std::endl;
                for (int comp = 0; comp < componentCount; comp++) {
                    std::cout << host_component_counter[comp] << " ";
                }
                std::cout << std::endl << std::endl;
            }

            for (int j = 0; j < componentCount; j++) {
                if (host_component_counter[j] == 0) continue;
                int orientation = host_component_counter[j] > 0 ? 1 : -1;
                componentGPUGraph[component_idx].insert({j, orientation});
                componentGPUGraph[j].insert({component_idx, orientation});
            }
        }
#pragma omp critical
        {
            for (int i = 0; i < componentCount; i++) {
                for (const auto &endGrapgh: componentGPUGraph[i]) {
                    componentGraph[i].insert(endGrapgh);
                }
            }
        }
    }

    std::cout << "The Graph:" << std::endl;
    for (int i = 0; i < componentCount; i++){
        for (const auto& neigh : componentGraph[i]){
            std::cout << i << "->" << neigh.endIndex << std::endl;
        }
    }


    std::cout << "Stage3" << std::endl;
    std::set<int> visitedSet;
    std::cout << componentCount << std::endl;
    while (visitedSet.size() < componentCount) {
        int rootIdx = 0;
        for (;rootIdx < componentCount; rootIdx++){
            if (!visitedSet.count(rootIdx))
                break;
        }
        std::queue<ComponentGraphParent> visitedQueue;
        visitedQueue.push({rootIdx, 1});
        visitedSet.insert(rootIdx);
        while (!visitedQueue.empty()) {
            auto currentComponent = visitedQueue.front();
            visitedQueue.pop();
            int currentIdx = currentComponent.parentIndex;
            int currentOrientation = currentComponent.orient;
            std::cout << "CURRENT INDEX:" << currentIdx << std::endl;
            for (const auto &neighbor: componentGraph[currentIdx]) {
                int neighborIdx = neighbor.endIndex;
                int neighborOrientation = neighbor.orient;
                if (!visitedSet.count(neighborIdx)) {
                    int combinedOrientation = currentOrientation * neighborOrientation;
                    //int combinedOrientation = -1;//currentOrientation * neighborOrientation;
                    if (combinedOrientation < 0) {
                        int dev_id = componentMembership[neighborIdx];
                        std::cout << neighborIdx << ":" << dev_id << std::endl;
                        cudaSetDevice(dev_id);
                        int componentCounter = componentToComponentCount[neighborIdx];
                        int threads = 1024;
                        int blocks = (int) ceil((1.0 * (componentCounter)) / threads);
                        cudaPointerAttributes attributes;
                        cudaError_t err = cudaPointerGetAttributes(&attributes, componentToComponentBuffer[neighborIdx]);

                        if(err != cudaSuccess) {
                            // handle error
                        }

                        int deviceId = attributes.device;
                        std::cout << deviceId << std::endl;
                        std::vector<int> host_component(componentCounter);
                        cudaMemcpy(host_component.data(), componentToComponentBuffer[neighborIdx],
                                   componentCounter * sizeof(int), cudaMemcpyDeviceToHost);
                        for (int i = 0; i < 100; i++){
                            std::cout << host_component[i] << " ";
                        }
                        std::cout << std::endl;
                        negateNormals<<<blocks, threads>>>(componentToComponentBuffer[neighborIdx],
                                                           dnormal_x[dev_id], dnormal_y[dev_id], dnormal_z[dev_id],
                                                           dev_id * partition_size, componentCounter);
                        cudaDeviceSynchronize();
                    }
                    visitedQueue.push({neighborIdx, currentOrientation * neighborOrientation});
                    visitedSet.insert(neighborIdx);
                }
            }
        }
    }

     /*for (int dev_id = 0; dev_id < numb_gpus; dev_id++) {
        cudaSetDevice(dev_id);
        cudaMemcpy(dpnorm_x[0] + dev_id * partition_size, dnormal_x[dev_id],
                   point_count_per_device[dev_id] * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dpnorm_y[0] + dev_id * partition_size, dnormal_y[dev_id],
                   point_count_per_device[dev_id] * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dpnorm_z[0] + dev_id * partition_size, dnormal_z[dev_id],
                   point_count_per_device[dev_id] * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaFree(dnormal_x[dev_id]);
        cudaFree(dnormal_y[dev_id]);
        cudaFree(dnormal_z[dev_id]);
    }*/

    cudaSetDevice(0);
    float *ddnorm_x, *ddnorm_y, *ddnorm_z;
    cudaMalloc((void **) &ddnorm_x, pointCount * sizeof(float));
    cudaMalloc((void **) &ddnorm_y, pointCount * sizeof(float));
    cudaMalloc((void **) &ddnorm_z, pointCount * sizeof(float));

    for (int dev_id = 0; dev_id < numb_gpus; dev_id++) {
        cudaSetDevice(dev_id);
        cudaMemcpy(ddnorm_x + dev_id * partition_size, dnormal_x[dev_id],
                   point_count_per_device[dev_id] * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(ddnorm_y + dev_id * partition_size, dnormal_y[dev_id],
                   point_count_per_device[dev_id] * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(ddnorm_z + dev_id * partition_size, dnormal_z[dev_id],
                   point_count_per_device[dev_id] * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaFree(dnormal_x[dev_id]);
        cudaFree(dnormal_y[dev_id]);
        cudaFree(dnormal_z[dev_id]);
    }

    cudaSetDevice(0);
    float *dnorm_x1, *dnorm_y1, *dnorm_z1;
    cudaMalloc((void **) &dnorm_x1, pointCount * sizeof(float));
    cudaMalloc((void **) &dnorm_y1, pointCount * sizeof(float));
    cudaMalloc((void **) &dnorm_z1, pointCount * sizeof(float));

    int threads = 1024;
    int blocks = (int) ceil((1.0 * (pointCount)) / threads);
    rearrange_output<<<blocks, threads>>>(dnorm_x1, dnorm_y1, dnorm_z1, ddnorm_x, ddnorm_y, ddnorm_z,
                                          pointCount, originalIndexes);
    cudaDeviceSynchronize();

    cudaMemcpy(normal_x, dnorm_x1, pointCount * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(normal_y, dnorm_y1, pointCount * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(normal_z, dnorm_z1, pointCount * sizeof(float), cudaMemcpyDeviceToHost);

    /*for (int dev_id = 0; dev_id < numb_gpus; dev_id++){
        cudaSetDevice(dev_id);
        cudaFree(dnorm_x[dev_id]);
        cudaFree(dnorm_y[dev_id]);
        cudaFree(dnorm_z[dev_id]);
    }*/

    cudaFree(ddnorm_x);
    cudaFree(ddnorm_y);
    cudaFree(ddnorm_z);

    cudaFree(dnorm_x1);
    cudaFree(dnorm_y1);
    cudaFree(dnorm_z1);
}

Normal_CUDA::Normal_CUDA(KNNInterface& knnInterface):
    din_x(knnInterface.GetRefPointsX()),
    din_y(knnInterface.GetRefPointsY()),
    din_z(knnInterface.GetRefPointsZ()),
    point_count_per_device(knnInterface.GetPointCountInCards()),
    dneighbors(knnInterface.GetKNNIndexesInPartitions()),
    k(knnInterface.NeighborCount()),
    pointCount(knnInterface.pointsRefCount()),
    numb_gpus(knnInterface.GetNumberOfCards()),
    partition_size(knnInterface.GetPartitionSize()),
    originalIndexes(knnInterface.GetOriginalRefIndexes()[0])
{

}