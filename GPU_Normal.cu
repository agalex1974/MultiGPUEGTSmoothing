//
// Created by agalex on 7/17/23.
//

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <stdio.h>
#include "EigenSolver.cuh"
#include "MST.h"
#include <chrono>
#include <iostream>
#include "CUDA_MST.cuh"
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <set>
#include "bfs.cuh"
#include <chrono>
#include <thread>

//#include <armadillo>
typedef unsigned long long int int_64;
typedef int int64;

/* convert a symmetric matrix to tridiagonal form */

#define SQR(a) ((a)*(a))
#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

__device__
float pythag(float a, float b)
{
    float absa, absb;
    absa = fabs(a);
    absb = fabs(b);
    if (absa > absb) return absa * sqrt(1.0 + SQR(absb / absa));
    else return (absb == 0.0 ? 0.0 : absb * sqrt(1.0 + SQR(absa / absb)));
}

__device__
void tred2(float **a, int n, float *d, float *e)
{
    int             l, k, j, i;
    float          scale, hh, h, g, f;

    for (i = n - 1; i > 0; i--) {
        l = i - 1;
        h = scale = 0.0;
        if (l > 0) {
            for (k = 0; k < l + 1; k++)
                scale += fabs(a[i][k]);
            if (scale == 0.0)
                e[i] = a[i][l];
            else {
                for (k = 0; k < l + 1; k++) {
                    a[i][k] /= scale;
                    h += a[i][k] * a[i][k];
                }
                f = a[i][l];
                g = (f >= 0.0 ? -sqrt(h) : sqrt(h));
                e[i] = scale * g;
                h -= f * g;
                a[i][l] = f - g;
                f = 0.0;
                for (j = 0; j < l + 1; j++) {
                    /* Next statement can be omitted if eigenvectors not wanted */
                    a[j][i] = a[i][j] / h;
                    g = 0.0;
                    for (k = 0; k < j + 1; k++)
                        g += a[j][k] * a[i][k];
                    for (k = j + 1; k < l + 1; k++)
                        g += a[k][j] * a[i][k];
                    e[j] = g / h;
                    f += e[j] * a[i][j];
                }
                hh = f / (h + h);
                for (j = 0; j < l + 1; j++) {
                    f = a[i][j];
                    e[j] = g = e[j] - hh * f;
                    for (k = 0; k < j + 1; k++)
                        a[j][k] -= (f * e[k] + g * a[i][k]);
                }
            }
        } else
            e[i] = a[i][l];
        d[i] = h;
    }
    /* Next statement can be omitted if eigenvectors not wanted */
    d[0] = 0.0;
    e[0] = 0.0;
    /* Contents of this loop can be omitted if eigenvectors not wanted except for statement d[i]=a[i][i]; */
    for (i = 0; i < n; i++) {
        l = i;
        if (d[i] != 0.0) {
            for (j = 0; j < l; j++) {
                g = 0.0;
                for (k = 0; k < l; k++)
                    g += a[i][k] * a[k][j];
                for (k = 0; k < l; k++)
                    a[k][j] -= g * a[k][i];
            }
        }
        d[i] = a[i][i];
        a[i][i] = 1.0;
        for (j = 0; j < l; j++)
            a[j][i] = a[i][j] = 0.0;
    }
}

/* calculate the eigenvalues and eigenvectors of a symmetric tridiagonal matrix */
__device__
void tqli(float *d, float *e, int n, float **z)
{
    int             m, l, iter, i, k;
    float          s, r, p, g, f, dd, c, b;

    for (i = 1; i < n; i++)
        e[i - 1] = e[i];
    e[n - 1] = 0.0;
    for (l = 0; l < n; l++) {
        iter = 0;
        do {
            for (m = l; m < n - 1; m++) {
                dd = fabs(d[m]) + fabs(d[m + 1]);
                if (fabs(e[m]) + dd == dd)
                    break;
            }
            if (m != l) {
                if (iter++ == 30) {
                    //fprintf(stderr, "[tqli] Too many iterations in tqli.\n");
                    break;
                }
                g = (d[l + 1] - d[l]) / (2.0 * e[l]);
                r = pythag(g, 1.0);
                g = d[m] - d[l] + e[l] / (g + SIGN(r, g));
                s = c = 1.0;
                p = 0.0;
                for (i = m - 1; i >= l; i--) {
                    f = s * e[i];
                    b = c * e[i];
                    e[i + 1] = (r = pythag(f, g));
                    if (r == 0.0) {
                        d[i + 1] -= p;
                        e[m] = 0.0;
                        break;
                    }
                    s = f / r;
                    c = g / r;
                    g = d[i + 1] - p;
                    r = (d[i] - g) * s + 2.0 * c * b;
                    d[i + 1] = g + (p = s * r);
                    g = c * r - b;
                    /* Next loop can be omitted if eigenvectors not wanted */
                    for (k = 0; k < n; k++) {
                        f = z[k][i + 1];
                        z[k][i + 1] = s * z[k][i] + c * f;
                        z[k][i] = c * z[k][i] - s * f;
                    }
                }
                if (r == 0.0 && i >= l)
                    continue;
                d[l] -= p;
                e[l] = g;
                e[m] = 0.0;
            }
        } while (m != l);
    }
}

__device__
int n_eigen_symm(float *_a, int n, float *eval)
{
    float **a, *e;
    int i;
    a = (float**)malloc(n * sizeof(float*));
    e = (float*)malloc(n * sizeof(float));
    for (i = 0; i < n; ++i) a[i] = _a + i * n;
    tred2(a, n, eval, e);
    tqli(eval, e, n, a);
    free(a); free(e);
    return 0;
}




typedef unsigned dimension;
typedef unsigned iterations;

#define ROTATE(S,i,j,k,l) g=S[i*n+j];h=S[k*n+l];S[i*n+j]=g-s*(h+g*tau); \
                          S[k*n+l]=h+s*(g-h*tau)

/* Maximum number of iterations allowed in jacobi() */

__device__
int jacobi2(float *S, dimension n, float *w, float *V) {
    int jacobi_max_iterations = 10;
    iterations i,j,k,iq,ip;
    float tresh,theta,tau,t,sm,s,h,g,c;
    float p;
    float b[3];
    float z[3];
    int nrot = 0;

    for(ip = 0; ip < n; ip++) {
        for(iq = 0; iq < n; iq++) V[ip*n+iq] = 0.0;
        V[ip*n+ip] = 1.0;
    }
    for(ip = 0; ip < n; ip++) {
        b[ip] = w[ip] = S[ip*n+ip];
        z[ip] = 0.0;
    }
    for(i = 1; i <= jacobi_max_iterations; i++) {
        sm = 0.0;
        for(ip = 0; ip < n-1; ip++) {
            for(iq = ip+1; iq < n; iq++) sm += fabs(S[ip*n+iq]);
        }
        if(sm == 0.0) {
            // Eigenvalues & eigenvectors sorting
            for(i = 0; i < n-1; i++) {
                p = w[k = i];
                for(j = i+1; j < n; j++) {
                    if(w[j] >= p) p = w[k = j];
                    if(k != i) {
                        w[k] = w[i];
                        w[i] = p;
                        for(j = 0; j < n; j++) {
                            p = V[j*n+i];
                            V[j*n+i] = V[j*n+k];
                            V[j*n+k] = p;
                        }
                    }
                }
            }
            // Restore symmetric matrix S
            for(i = 1; i < n; i++) {
                for(j = 0; j < i; j++) S[j*n+i] = S[i*n+j];
            }
            return nrot;
        }
        if(i < 4) tresh = 0.2*sm/(n*n); else tresh = 0.0;
        for(ip = 0; ip < n-1; ip++) {
            for(iq = ip+1; iq < n; iq++) {
                g = 100.0*fabs(S[ip*n+iq]);
                if(i > 4 && (fabs(w[ip])+g == fabs(w[ip])) && (fabs(w[iq])+g == fabs(w[iq]))) {
                    S[ip*n+iq] = 0.0;
                }
                else if(fabs(S[ip*n+iq]) > tresh) {
                    h = w[iq] - w[ip];
                    if(fabs(h) + g == fabs(h)) {
                        t = (S[ip*n+iq]) / h;
                    }
                    else {
                        theta = 0.5 * h / (S[ip*n+iq]);
                        t = 1.0 / (fabs(theta) + sqrt(1.0 + theta * theta));
                        if(theta < 0.0) t = -t;
                    }
                    c = 1.0 / sqrt(1 + t*t);
                    s = t * c;
                    tau = s / (1.0 + c);
                    h = t * S[ip*n+iq];
                    z[ip] -= h;
                    z[iq] += h;
                    w[ip] -= h;
                    w[iq] += h;
                    S[ip*n+iq] = 0.0;
                    for(j = 0; j < ip; j++) {
                        ROTATE(S,j,ip,j,iq);
                    }
                    for(j = ip+1; j < iq; j++) {
                        ROTATE(S,ip,j,j,iq);
                    }
                    for(j = iq+1; j < n; j++) {
                        ROTATE(S,ip,j,iq,j);
                    }
                    for(j = 0; j < n; j++) {
                        ROTATE(V,j,ip,j,iq);
                    }
                    ++nrot;
                }
            }
        }
        for(ip = 0; ip < n; ip++) {
            b[ip] += z[ip];
            w[ip] = b[ip];
            z[ip] = 0.0;
        }
    }
    return -1; // Too many iterations in jacobi()
}


#define N 3 // Size of your matrix
#define IDX(i, j) ((i)*N + (j))

// Function to find the maximum off-diagonal element
__device__
void maxOffDiagonal(float A[N*N], int& p, int& q) {
    float max = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = i+1; j < N; j++) {
            if (abs(A[i*N+j]) > max) {
                max = abs(A[i*N+j]);
                p = i;
                q = j;
            }
        }
    }
}

// Function to perform the rotation to make A[p][q] = 0
__device__
void rotate(float A[N*N], float E[N*N], int p, int q) {
    float theta = (A[q*N+q] - A[p*N+p]) / (2.0 * A[p*N+q]);
    float t = 1.0 / (abs(theta) + sqrt(pow(theta, 2) + 1));
    if (theta < 0) {
        t = -t;
    }
    float c = 1.0 / sqrt(pow(t, 2) + 1);
    float s = t * c;
    float tau = s / (1 + c);
    float temp = A[p*N+q];
    A[p*N+q] = 0;
    A[p*N+p] -= t * temp;
    A[q*N+q] += t * temp;
    for (int i = 0; i < N; i++) {
        if (i != p && i != q) {
            float Aip = A[i*N+p];
            float Aiq = A[i*N+q];
            A[i*N+p] -= s * (Aiq + tau * Aip);
            A[i*N+q] += s * (Aip - tau * Aiq);
            A[p*N+i] = A[i*N+p];
            A[q*N+i] = A[i*N+q];
        }
        float Eip = E[i*N+p];
        float Eiq = E[i*N+q];
        E[i*N+p] -= s * (Eiq + tau * Eip);
        E[i*N+q] += s * (Eip - tau * Eiq);
    }
}

// Function to compute the eigenvalues and eigenvectors using the Jacobi method
__device__
void jacobi1(float A[N*N], float eps, float E[N * N]) {
    //float E[N*N] = {0};
    for (int i = 0; i < N; i++) {
        E[i*N+i] = 1.0;
    }
    int p, q;
    maxOffDiagonal(A, p, q);
    while (abs(A[p*N+q]) > eps) {
        rotate(A, E, p, q);
        maxOffDiagonal(A, p, q);
    }

}

__global__
void calculateNormal(int* knn, int k, int k_use, float* x, float* y, float* z, float* nx, float* ny, float* nz, int count, int advance,
                     float* debugCov){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < count) {
        int realIdx = i + advance;
        float sumX = 0.0, sumY = 0.0, sumZ = 0.0;
        int counter = 0;
        for (int j = 0; j < k_use; j++) {
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
            for (int j = 0; j < k_use; j++) {
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

            //memcpy(debugCov + realIdx * 9, covarianceMatrix, 9 * sizeof(float));
            float eigvalues[3];
            float eigvectors[3][3];

            SymmetricEigensolver3x3<float> sv;
            sv(covarianceMatrix[0], covarianceMatrix[1], covarianceMatrix[2], covarianceMatrix[4], covarianceMatrix[5], covarianceMatrix[8],
                                           false, 1, eigvalues, eigvectors);

            float nxx = eigvectors[0][0];
            float nyy = eigvectors[0][1];
            float nzz = eigvectors[0][2];
            float norm = sqrt(nxx * nxx + nyy * nyy + nzz * nzz);
            nx[realIdx] = nxx / norm;
            ny[realIdx] = nyy / norm;
            nz[realIdx] = nzz / norm;
        }
    }
}

__global__
void rearrange_output_reverse(float* xout, float* yout, float* zout, float* xin, float* yin, float* zin,
                      float* nxout, float* nyout, float* nzout, float* nxin, float* nyin, float* nzin,
                      int count, int* reverseIndexes){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < count){
        int idx = reverseIndexes[i];
        xout[i] = xin[idx];
        yout[i] = yin[idx];
        zout[i] = zin[idx];
        nxout[i] = nxin[idx];
        nyout[i] = nyin[idx];
        nzout[i] = nzin[idx];
    }
}

__global__
void rearrange_output(float* xout, float* yout, float* zout, float* xin, float* yin, float* zin,
                              float* nxout, float* nyout, float* nzout, float* nxin, float* nyin, float* nzin,
                              int count, int* originalIndexes){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < count){
        int idx = originalIndexes[i];
        xout[idx] = xin[i];
        yout[idx] = yin[i];
        zout[idx] = zin[i];
        nxout[idx] = nxin[i];
        nyout[idx] = nyin[i];
        nzout[idx] = nzin[i];
    }
}

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

__global__
void getEdgeNumber(int* knn, int k, int k_use, int64* edgeNumbers, int count){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < count){
        for (int j = 0; j < k_use; j++){
            int n = knn[i * k + j];
            if (n >= 0){
                int2 tuple = order_pair(i, n);
                atomicAdd(&edgeNumbers[tuple.x], 1);
            }
        }
    }
}

__global__
void fillEdges(int* knn, int k, int k_use, int64* edgeNumbers, int64* offsets, wghEdge<int>* edges, int count,
               float* nx, float* ny, float* nz, int_64* hash){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < count){
        for (int j = 0; j < k_use; j++){
            int n = knn[i * k + j];
            if (n >= 0){
                int2 tuple = order_pair(i, n);
                int64 position = atomicAdd(&edgeNumbers[tuple.x], 1);
                position += offsets[tuple.x];
                float norm = nx[i] * nx[n] + ny[i] * ny[n] + nz[i] * nz[n];
                float weight = fmaxf(0.0f, 1.0f - fabsf(norm));
                edges[position] = wghEdge<int>(tuple.x, tuple.y, weight);
                hash[position] = (int_64)tuple.x * (int_64)count + (int_64)tuple.y;
            }
        }
    }
}

__global__
void getWeights(wghEdge<int>* edges, int64 count, float* weights){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < count){
        weights[i] = edges[i].weight;
    }
}

__global__
void createMSTGraph(int* knn, int k, wghEdge<int>* edges, int* indexes, int count, int* vertexCounter){
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
void BFS_STEP(int* knn, int k, int* front, int* children, int count, int* processed, int* finished,
              float* nx, float* ny, float* nz){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < count && front[i] > 0){
        bool inserted = false;
        processed[i] = 1;
        for (int j = 0; j < k; j++){
            int n = knn[i * k + j];
            if (n >= 0){
                if (!processed[n]) {
                    children[n] = 1;
                    inserted = true;
                    float dot = nx[i] * nx[n] + ny[i] * ny[n] + nz[i] * nz[n];
                    if (dot < 0) {
                        nx[n] = -nx[n];
                        ny[n] = -ny[n];
                        nz[n] = -nz[n];
                    }
                }
            }
            else break;
        }
        if (inserted){
            int isFinished = *finished;
            if (isFinished) *finished = 0;
        }
    }
}

/*__global__
void createMST(int(knn, )){

}*/

// Creating shortcut for an integer pair
typedef std::pair<int, int> iPair;

// Structure to represent a graph
struct Graph {
    int V, E;
    std::vector <std::pair<double, iPair>> edges;

    // Constructor
    Graph(int V) {
        this->V = V;
        E = 0;
    }

    // Utility function to add an edge
    void addEdge(int u, int v, double w) {
        edges.push_back({w, {u, v}});
        E++;
    }
};

template <typename T>
__global__
void setValueKNN(T* vector, T value, int count){
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < count){
        vector[i] = value;
    }
}

std::pair<int, int> order1(int i, int j){
    if (i < j) return {i, j};
    else return {j, i};
}

__global__
void populateKNN(int* knnout, int* knnin, int k_out, int k_in, int count){
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < count){
        for (int j = 0; j < k_out; j++){
            knnout[i * k_out + j] = knnin[i * k_in + j];
        }
    }
}

__global__
void produce_output_final(float* x, float* y, float* z, float* xout, float* yout, float* zout, float min, float max, int pointCount)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < pointCount) {
        float d = max - min;
        xout[i] = x[i] * d + min;
        yout[i] = y[i] * d + min;
        zout[i] = z[i] * d + min;
    }
}


typedef std::pair<int, int> iPair;

void GPU_NORMAL(float** din_x, float** din_y, float** din_z, int* count_per_device,
                int** dneighbors, int maxNeighbors, int k, float* out_x, float* out_y, float* out_z, int count,
                float* normal_x, float* normal_y, float* normal_z,
                int numb_gpus, int partition_size, int* reverseIndexes,
                float minOriginal, float maxOriginal){

    std::chrono::steady_clock::time_point begin_normal_computation = std::chrono::steady_clock::now();

    std::vector<float*> dnormalv_x(numb_gpus); float** dnormal_x = dnormalv_x.data();
    std::vector<float*> dnormalv_y(numb_gpus); float** dnormal_y = dnormalv_y.data();
    std::vector<float*> dnormalv_z(numb_gpus); float** dnormal_z = dnormalv_z.data();
    std::vector<float*> dcovv(numb_gpus); float** dcov = dcovv.data();
    size_t size = count * sizeof(float);

#pragma omp parallel for num_threads(numb_gpus)
    for (int dev_id = 0; dev_id < numb_gpus; dev_id++){
        cudaSetDevice(dev_id);
        cudaMalloc((void**)&dnormal_x[dev_id], size);
        cudaMalloc((void**)&dnormal_y[dev_id], size);
        cudaMalloc((void**)&dnormal_z[dev_id], size);
        //cudaMalloc((void**)&dcov[dev_id], 9 * size);
        int threads = 1024;
        int blocks = (int) ceil((1.0 * (count_per_device[dev_id])) / threads);
        calculateNormal<<<blocks, threads>>>(dneighbors[dev_id], maxNeighbors, k, din_x[dev_id], din_y[dev_id], din_z[dev_id],
                                             dnormal_x[dev_id], dnormal_y[dev_id], dnormal_z[dev_id], count_per_device[dev_id], dev_id * partition_size, dcov[dev_id]);
        cudaDeviceSynchronize();
    }

    cudaSetDevice(0);

#pragma omp parallel for num_threads(numb_gpus - 1)
    for (int dev_id = 1; dev_id < numb_gpus; dev_id++){
        cudaMemcpy(dnormal_x[0] + dev_id * partition_size, dnormal_x[dev_id] + dev_id * partition_size, count_per_device[dev_id] * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dnormal_y[0] + dev_id * partition_size, dnormal_y[dev_id] + dev_id * partition_size, count_per_device[dev_id] * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dnormal_z[0] + dev_id * partition_size, dnormal_z[dev_id] + dev_id * partition_size, count_per_device[dev_id] * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    std::chrono::steady_clock::time_point endtime_normal_computation = std::chrono::steady_clock::now();
    std::cout << "Execution time Normal Computation = " << std::chrono::duration_cast<std::chrono::milliseconds>(endtime_normal_computation - begin_normal_computation).count() << "[ms]" << std::endl;

    int k_use = 20;
    std::vector<int*> dknn_reduced(numb_gpus);
#pragma omp parallel for num_threads(numb_gpus)
    for (int dev_id = 0; dev_id < numb_gpus; dev_id++){
        cudaSetDevice(dev_id);
        int threads = 1024;
        int blocks = (int) ceil((1.0 * count_per_device[dev_id]) / threads);
        cudaMalloc((void**)&dknn_reduced[dev_id], k_use * count_per_device[dev_id] * sizeof(int));
        populateKNN<<<blocks, threads>>>(dknn_reduced[dev_id], dneighbors[dev_id], k_use, maxNeighbors, count_per_device[dev_id]);
        cudaDeviceSynchronize();
        cudaFree(dneighbors[dev_id]);
    }

    cudaSetDevice(0);
    //std::cout << "Before" << std::endl;
    //std::this_thread::sleep_for(std::chrono::seconds(20));
    int* dknn_global_reduced;
    cudaMalloc((void**)&dknn_global_reduced, k_use * count * sizeof(int));
    //std::cout << "After" << std::endl;
    //std::this_thread::sleep_for(std::chrono::seconds(20));
#pragma omp parallel for num_threads(numb_gpus)
    for (int dev_id = 0; dev_id < numb_gpus; dev_id++) {
        cudaMemcpy(dknn_global_reduced + dev_id * k_use * partition_size, dknn_reduced[dev_id], k_use * count_per_device[dev_id] * sizeof(int), cudaMemcpyDeviceToDevice);
    }

#pragma omp parallel for num_threads(numb_gpus)
    for (int dev_id = 0; dev_id < numb_gpus; dev_id++) {
        cudaSetDevice(dev_id);
        cudaFree(dknn_reduced[dev_id]);
    }
    cudaSetDevice(0);
    int64* edgeCounter;
    cudaMalloc((void**)&edgeCounter, count * sizeof(int64));
    cudaMemset(edgeCounter, 0, count * sizeof(int64));
    int threads = 1024;
    int blocks = (int) ceil((1.0 * count) / threads);
    getEdgeNumber<<<blocks, threads>>>(dknn_global_reduced, k_use, k_use, edgeCounter, count);
    cudaDeviceSynchronize();

    int64 edgeCount = thrust::reduce(thrust::device, edgeCounter, edgeCounter + count, 0, thrust::plus<int64>());
    std::cout << "total edges:" << edgeCount << std::endl;


    wghEdge<int>* edges;
    cudaMalloc((void**)&edges, edgeCount * sizeof(wghEdge<int>));

    int64* edgeOffset;
    cudaMalloc((void**)&edgeOffset, count * sizeof(int64));

    thrust::exclusive_scan(thrust::device, edgeCounter, edgeCounter + count, edgeOffset, 0, thrust::plus<int64>());
    std::cout << "PASSED 1" << std::endl;
    cudaMemset(edgeCounter, 0, count * sizeof(int64));
    int_64* hash;
    cudaMalloc((void**)&hash, edgeCount * sizeof(int_64));

    threads = 1024;
    blocks = (int) ceil((1.0 * count) / threads);
    fillEdges<<<blocks, threads>>>(dknn_global_reduced, k_use, k_use, edgeCounter, edgeOffset, edges, count,
                                   dnormal_x[0], dnormal_y[0], dnormal_z[0], hash);
    cudaDeviceSynchronize();

    thrust::sort_by_key(thrust::device, hash, hash + edgeCount, edges,
                        thrust::less<int_64>());

    std::cout << "PASSED 2" << std::endl;

    thrust::pair<int_64*, wghEdge<int>*> end;
    end = thrust::unique_by_key(thrust::device, hash, hash + edgeCount,
                                edges, thrust::equal_to<int_64>());
    std::cout << "PASSED 3" << std::endl;
    int_64 edgeCountUnique = end.first - hash;
    std::cout << "Unique edges:" << edgeCountUnique << std::endl;

    std::cout << std::endl;
    wghEdge<int>* edgesUnique;
    cudaMalloc((void**)&edgesUnique, edgeCountUnique * sizeof(wghEdge<int>));
    cudaMemcpy(edgesUnique, edges, edgeCountUnique * sizeof(wghEdge<int>), cudaMemcpyDeviceToDevice);
    float* weights;
    cudaMalloc((void**)& weights, edgeCountUnique * sizeof(float));
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

    wghEdgeArray<int> G(edgesUnique, count, edgeCountUnique);
    auto pr = mst(G);
    std::cout << "MST edges:" << pr.second << std::endl;

    //exit(0);

    int* dmstIndexes;
    cudaMalloc((void**)&dmstIndexes, pr.second * sizeof(int));
    cudaMemcpy(dmstIndexes, pr.first, pr.second * sizeof(int), cudaMemcpyHostToDevice);

    threads = 1024;
    blocks = ceil((1.0 * k_use * count) / threads);
    setValueKNN<int><<<blocks, threads>>>(dknn_global_reduced, -1, k_use * count);
    cudaDeviceSynchronize();


    int* csr_edges;
    cudaMalloc((void**)& csr_edges, (count + 1) * sizeof(int));
    cudaMemset(csr_edges, 0, (count + 1) * sizeof(int));

    threads = 1024;
    blocks = ceil((1.0 * pr.second / threads));
    createMSTGraph<<<blocks, threads>>>(dknn_global_reduced, k_use, edgesUnique, dmstIndexes, pr.second, csr_edges);
    cudaDeviceSynchronize();

    thrust::exclusive_scan(thrust::device, csr_edges, csr_edges + count + 1, csr_edges);
    int numb_edges = 0;
    cudaMemcpy(&numb_edges, csr_edges + count, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Number of edges:" << numb_edges << std::endl;
    int* csr_dest;
    cudaMalloc((void**)& csr_dest, numb_edges * sizeof(int));
    threads = 1024;
    blocks = ceil((1.0 * count) / threads);
    createCSRDEST<<<blocks, threads>>>(dknn_global_reduced, k_use, count, csr_edges, csr_dest);
    cudaDeviceSynchronize();
    std::cout << "Finished!";

    wghEdge<int> e_root;
    cudaMemcpy(&e_root, &edgesUnique[0], sizeof(wghEdge<int>), cudaMemcpyDeviceToHost);
    int root = e_root.u;
    std::cout << "orienting...";
    //exit(0);
    //BFS(root, csr_edges, csr_dest, count, dnormal_x[0], dnormal_y[0], dnormal_z[0]);
    std::cout << "Finished orienting!" << std::endl;

    /*std::vector<int> frontier(count, 0);
    frontier[root] = 1;
    int* dfrontier, *dchildren, *dprocessed;
    cudaMalloc((void**)& dfrontier, count * sizeof(int));
    cudaMemcpy(dfrontier, frontier.data(), count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)& dchildren, count * sizeof(int));
    cudaMemset(dchildren, 0, count * sizeof(int));
    cudaMalloc((void**)& dprocessed, count * sizeof(int));
    cudaMemset(dprocessed, 0, count * sizeof(int));
    int finished = 0;
    int* dfinished;
    cudaMalloc((void**)& dfinished, sizeof(int));
    threads = 1024;
    blocks = ceil((1.0 * count / threads));
    std::cout << "Traversing MST" << std::endl;
    while (!finished){
        finished = 1;
        cudaMemcpy(dfinished, &finished, sizeof(int), cudaMemcpyHostToDevice);
        BFS_STEP<<<blocks, threads>>>(dknn, maxNeighbors, dfrontier, dchildren, count, dprocessed, dfinished,
                                      dnormal_x[0], dnormal_y[0], dnormal_z[0]);
        cudaDeviceSynchronize();
        std::swap(dfrontier, dchildren);
        cudaMemset(dchildren, 0, count * sizeof(int));
        cudaMemcpy(&finished, dfinished, sizeof(int), cudaMemcpyDeviceToHost);
    }*/



    /*std::vector<int> knn(maxNeighbors * count);
    cudaMemcpy(normal_x, dnormal_x[0], size, cudaMemcpyDeviceToHost);
    cudaMemcpy(normal_y, dnormal_y[0], size, cudaMemcpyDeviceToHost);
    cudaMemcpy(normal_z, dnormal_z[0], size, cudaMemcpyDeviceToHost);
    std::vector<float> x(count);
    std::vector<float> y(count);
    std::vector<float> z(count);

    cudaMemcpy(x.data(), din_x[0], size, cudaMemcpyDeviceToHost);
    cudaMemcpy(y.data(), din_y[0], size, cudaMemcpyDeviceToHost);
    cudaMemcpy(z.data(), din_z[0], size, cudaMemcpyDeviceToHost);

#pragma omp parallel for num_threads(numb_gpus)
    for (int dev_id = 0; dev_id < numb_gpus; dev_id++) {
        cudaMemcpy(knn.data() + dev_id * maxNeighbors * partition_size, dneighbors[dev_id], maxNeighbors * count_per_device[dev_id] * sizeof(int), cudaMemcpyDeviceToHost);
    }

    Graph g(count);
    std::unordered_map<long long int, std::pair<int, int>> uordmap;
    std::set<std::pair<int, int>> mySet;
    //uordmap.reserve(N * k_use);
    std::cout << "orienting" << std::endl;
    for (int i = 0; i < count; i++){
        for (int j = 0; j < k_use; j++){
            int n = knn[i * k + j];
            if (n >= 0){
                //if (i < n){
                auto pair = order1(i, n);
                //long long int key = (int64)N * (int64)pair.first + (int64)pair.second;
                //auto success = uordmap.insert({key, pair});
                if(mySet.count(pair) == 0){
                //if (success.second){
                    double norm = normal_x[i] * normal_x[n] + normal_y[i] * normal_y[n] + normal_z[i] * normal_z[n];
                    double weight = std::max(0.0, 1.0 - abs(norm));
                    g.addEdge(pair.first, pair.second, weight);
                    mySet.insert(pair);
                    //mySet.insert(order(i, n));
                }
            }
        }
    }
    std::cout << "CPU Edges:" << g.E << std::endl;

    //std::chrono::steady_clock::time_point begin_normal_orientation = std::chrono::steady_clock::now();
    //orient_kruskal(knn.data(), normal_x, normal_y, normal_z, x.data(), y.data(), z.data(), count, maxNeighbors, 12);
    //std::chrono::steady_clock::time_point endtime_normal_orientation = std::chrono::steady_clock::now();
    //std::cout << "Execution time Orientation = " << std::chrono::duration_cast<std::chrono::milliseconds>(endtime_normal_orientation - begin_normal_orientation).count() << "[ms]" << std::endl;

    cudaMemcpy(dnormal_x[0], normal_x,  size, cudaMemcpyHostToDevice);
    cudaMemcpy(dnormal_y[0],normal_y,  size, cudaMemcpyHostToDevice);
    cudaMemcpy(dnormal_z[0], normal_z,  size, cudaMemcpyHostToDevice);*/
    //cudaMemcpy(cov, dcov[0], 9 * count * sizeof(float), cudaMemcpyDeviceToHost);

    float* dx, *dy, *dz;
    float* dnx, *dny, *dnz;
    cudaMalloc((void**)&dx, size);
    cudaMalloc((void**)&dy, size);
    cudaMalloc((void**)&dz, size);
    cudaMalloc((void**)&dnx, size);
    cudaMalloc((void**)&dny, size);
    cudaMalloc((void**)&dnz, size);
    threads = 1024;
    blocks = (int) ceil((1.0 * (count)) / threads);
    rearrange_output<<<blocks, threads>>>(dx, dy, dz, din_x[0], din_y[0], din_z[0],
                                          dnx, dny, dnz, dnormal_x[0], dnormal_y[0], dnormal_z[0],
                                          count, reverseIndexes);
    cudaDeviceSynchronize();

    threads = 1024;
	blocks = (int)ceil((1.0 * count) / threads);
    produce_output_final<<<blocks, threads>>>(dx, dy, dz, din_x[0], din_y[0], din_z[0], minOriginal,
                                              maxOriginal, count);
    cudaDeviceSynchronize();

    cudaMemcpy(out_x, din_x[0], size, cudaMemcpyDeviceToHost);
    cudaMemcpy(out_y, din_y[0], size, cudaMemcpyDeviceToHost);
    cudaMemcpy(out_z, din_z[0], size, cudaMemcpyDeviceToHost);
    cudaMemcpy(normal_x, dnx, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(normal_y, dny, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(normal_z, dnz, size, cudaMemcpyDeviceToHost);
}