//
// Created by agalex on 6/14/24.
//

#ifndef KNNCUDA_GRIDMESHCUDA_CUH
#define KNNCUDA_GRIDMESHCUDA_CUH

#include "KNNInterface.cuh"
#include "Mesh.h"

typedef long long long64;

class GridMeshCuda: public KNNInterface {
public:
    //k maximum allowable neighbors
    GridMeshCuda(const Mesh& mesh, int num_cards, float eps = 1e-6, int bucketCount = 1024);
    void GetNeighborsFromTriangles(int k);
    ~GridMeshCuda();
    float** GetRefPointsX() override {return dpoints_x.data();}
    float** GetRefPointsY() override {return dpoints_y.data();}
    float** GetRefPointsZ() override {return dpoints_z.data();}

    int** GetKNNIndexesInPartitions() override {return dknn.data();}
    [[nodiscard]] int GetPartitionSize() const override {return pointPartitionSize;}
    int* GetPointCountInCards() override {return pointsCard.data();}
    int** GetOriginalRefIndexes() override {return doriginal_indexes.data();}
    void GetKNNIndexes(int* knnIndexes) override;
    [[nodiscard]] float GetMaxExtent() const override {return maxExtent;}
    [[nodiscard]] float GetMinExtent() const override {return minExtent;}
    [[nodiscard]] bool isReady() const override{return dknn[0] != nullptr;};
    [[nodiscard]] int pointsRefCount() const override {return pointsCount;};
    [[nodiscard]] int NeighborCount() const override {return neighborCount;};
    [[nodiscard]] int GetNumberOfCards() const override {return num_cards;};
private:
    void InitStructure(const float *points_x, const float *points_y, const float *points_z, const I3* indexes);
    template<typename T>
    void getBuckets(int dev_id, int bucketsCount, float min, float max, int &maxBucketSize);

    float eps;
    int pointPartitionSize;
    std::vector<float*> dpoints_x;
    std::vector<float*> dpoints_y;
    std::vector<float*> dpoints_z;
    std::vector<uint32_t*> dtriangleIndexes;

    std::vector<int*> doriginal_indexes;
    std::vector<int*> dreverse_indexes;
    std::vector<int*> dknn;
    const I3* triangleIndexes;
    std::vector<int> pointsCard;
    int neighborCount;
    int num_cards;
    float maxExtent;
    float minExtent;
    int pointsCount;
    int trianglesCount;
    int finalBucketCount;
};

#endif //KNNCUDA_GRIDMESHCUDA_CUH
