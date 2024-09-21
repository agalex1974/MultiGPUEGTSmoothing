#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <set>
#include <cmath>
//#include <armadillo>
//In Vast AI you need the docker image: nvidia/cuda:11.7.1-devel-ubuntu22.04
#include "EGT_GKNN.cuh"
#include "GPU_Normal.cuh"
#include "CGALHeaders.h"
#include "MeshPLY.h"
#include "GridKNNCuda.cuh"
#include "KNNInterface.cuh"
#include "GridMeshCuda1.cuh"
#include "EGT_CUDA.cuh"
#include "Normal_CUDA.cuh"

#include "getOutputPoints.cuh"

#define NUM_CARDS 2
void cudaWarmup(float* points_x, float* points_y, float* points_z, int pointsCount, int k, int* kNN, int num_cards);

void GRIDCUDAKNN(float* pnts_x, float* pnts_y, float* points_z, int pointsCount, int k,
                 float**& dpnts_x, float**& dpnts_y, float**& dpnts_z, int**& dknn, int num_cards,
                 int*& reverseIndexes, int*& pointsCard, int& partitionSize, float& minOriginal, float& maxOriginal,
                 float* x, float* y, float* z, int* knn);

void readPointCloud(const std::string& fileName, std::vector<float>& points_x, 
    std::vector<float>& points_y, std::vector<float>& points_z, std::vector<float>& normals_x,
                    std::vector<float>& normals_y, std::vector<float>& normals_z, bool hasNormals)
{
    int number_of_lines = 0;
    std::string line;
    std::ifstream myfile(fileName, std::ios::in);
    while (std::getline(myfile, line))
        ++number_of_lines;
    std::cout << "reading " << number_of_lines << " points." << std::endl;
    myfile.clear();
    myfile.seekg(0);
    float x, y, z, nx, ny, nz;
    if (hasNormals)
        while (myfile >> x >> y >> z >> nx >> ny >> nz) {
            points_x.emplace_back(x);
            points_y.emplace_back(y);
            points_z.emplace_back(z);
            normals_x.emplace_back(nx);
            normals_y.emplace_back(ny);
            normals_z.emplace_back(nz);
        }
    else
        while (myfile >> x >> y >> z) {
            points_x.emplace_back(x);
            points_y.emplace_back(y);
            points_z.emplace_back(z);
        }
}

void savePointCloud(const char* fileName, float* pnts_x, float* pnts_y, float* pnts_z, int point_count,
                    float* nx, float* ny, float* nz, bool hasNormals)
{
    FILE* file = fopen(fileName, "w");
    for (int i = 0; i < point_count; i++)
    {
        if (hasNormals)
            fprintf(file, "%lf %lf %lf %lf %lf %lf\n",
                pnts_x[i], pnts_y[i], pnts_z[i], nx[i], ny[i], nz[i]);
        else
            fprintf(file, "%lf %lf %lf\n",
                    pnts_x[i], pnts_y[i], pnts_z[i]);
    }
    fclose(file);
}

int main(int argc, char** argv)
{
    std::string meshPathOrigin = argv[1];

    std::vector<float> normals_x;
    std::vector<float> normals_y;
    std::vector<float> normals_z;
    int k = atoi(argv[2]);
    int numcards = atoi(argv[3]);
    Mesh mesh;
    ReadPLY(meshPathOrigin, &mesh);
    normals_x.resize(mesh.mV.size());
    normals_y.resize(mesh.mV.size());
    normals_z.resize(mesh.mV.size());


    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    std::vector<float> out_x(mesh.mV.size());
    std::vector<float> out_y(mesh.mV.size());
    std::vector<float> out_z(mesh.mV.size());
    std::unique_ptr<GridStructure> knnInterface = std::make_unique<GridStructure>(mesh, numcards, 1e-6, 1024, 4);
    knnInterface->GRIDCUDAKNNSELF_COMPACT(k, 1e8);
    //GridMeshCuda* knnInterface = new GridMeshCuda(mesh, numcards, 1e-6,1024);
    //knnInterface->GetNeighborsFromTriangles(k);
    std::chrono::steady_clock::time_point begin_smooth = std::chrono::steady_clock::now();
    EGT_CUDA smoother(*knnInterface);
    smoother.PerformSmoothing(150,0.63,-0.64,0.65);
    Normal_CUDA normalCuda(*knnInterface);
    normalCuda.GetNormals(normals_x.data(), normals_y.data(), normals_z.data());
    getOutputPoints(out_x.data(), out_y.data(), out_z.data(), knnInterface.get());
    mesh.mN.resize(knnInterface->pointsRefCount());
    for (int i = 0; i < mesh.mV.size(); i++) {
        mesh.mV[i].x = out_x[i];
        mesh.mV[i].y = out_y[i];
        mesh.mV[i].z = out_z[i];

        mesh.mN[i].x = normals_x[i];
        mesh.mN[i].y = normals_y[i];
        mesh.mN[i].z = normals_z[i];
    }
    WritePLY("output.ply", &mesh);
	printf("Succeeded!\n");
}