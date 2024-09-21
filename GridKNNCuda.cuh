#ifndef GRIDKNNCUDA_H
#define GRIDKNNCUDA_H
#include <vector>
#include <iostream>
#include "KNNInterface.cuh"
#include "Mesh.h"

typedef long long long64;

template <typename T>
class NeighborHelper;

class GridStructure:public KNNInterface{
public:
    GridStructure(const Mesh& Mesh, int num_cards, float eps = 1e-6,
                  int bucketCount = 1024, int maxDivision = 32);
    void GRIDCUDAKNN(float* pointsQ_x, float* pointsQ_y, float* pointsQ_z, int pointsCountQuery,
                     int k, int* knn, float thres);
    void GRIDCUDAKNNCompact(float* pointsQ_x, float* pointsQ_y, float* pointsQ_z, int pointsCountQuery,
                                           int k, float thres);
    void GRIDCUDAKNNSELF(int k, float thres);
    void GRIDCUDAKNNSELF_COMPACT(int k, float thres);
    ~GridStructure();

    float** GetRefPointsX() override {return dpoints_x.data();}
    float** GetRefPointsY() override {return dpoints_y.data();}
    float** GetRefPointsZ() override {return dpoints_z.data();}

    float** GetQuerPointsX() override {return self ? dpoints_x.data() : dQuery_x.data();};
    float** GetQuerPointsY() override {return self ? dpoints_y.data() : dQuery_y.data();};
    float** GetQuerPointsZ() override {return self ? dpoints_z.data() : dQuery_z.data();};

    int** GetKNNIndexesInPartitions() override {return dknn.data();}
    [[nodiscard]] int GetPartitionSize() const override {return pointPartitionSize;}
    int* GetPointCountInCards() override {return pointsCard.data();}
    int** GetOriginalRefIndexes() override {return doriginal_indexes.data();}
    int** GetOriginalQuerIndexes() override {return self ? doriginal_indexes.data() : dQueryIndexesInitial.data();}
    void GetKNNIndexes(int* knnIndexes) override;
    [[nodiscard]] float GetMaxExtent() const override {return maxExtent;}
    [[nodiscard]] float GetMinExtent() const override {return minExtent;}
    [[nodiscard]] bool isReady() const override{return dknn[0] != nullptr;};
    [[nodiscard]] int pointsRefCount() const override {return pointsCount;};
    [[nodiscard]] int pointsQuerCount() const override {return self ? pointsRefCount() : pointsCountQuery;};
    [[nodiscard]] int NeighborCount() const override {return neighborCount;};
    [[nodiscard]] int GetNumberOfCards() const{return num_cards;};
    void RefreshGridKNNStructure();
private:
    bool self;
    struct Hierarchy{
        Hierarchy(int num_cards):
                num_cards(num_cards),
                dindexes(num_cards, NULL),
                dbucketIndexes(num_cards, NULL),
                dpointBucketIndexes(num_cards, NULL),
                vertexPartitionSize(num_cards, 0),
                initialized(num_cards, false)
        {}
        void RefreshHierarchy();
        int num_cards;
        std::vector<bool> initialized;
        std::vector<long64*> dbucketIndexes;
        std::vector<int*> dindexes;
        std::vector<int*> dpointBucketIndexes;
        std::vector<int> vertexPartitionSize;
        ~Hierarchy();
    };
    float eps;
    int pointPartitionSize;
    std::vector<float*> dpoints_x;
    std::vector<float*> dpoints_y;
    std::vector<float*> dpoints_z;
    std::vector<float*> dQuery_x;
    std::vector<float*> dQuery_y;
    std::vector<float*> dQuery_z;
    std::vector<int*> doriginal_indexes;
    std::vector<int*> dQueryIndexesInitial;
    // Alex Here
    //std::vector<bool> initialized;
    //
    std::vector<int*> dknn;
    std::vector<Hierarchy*> hierachyTree;
    std::vector<int> pointsCard;
    int neighborCount;
    int max_bucket_elements;
    int levels;
    int num_cards;
    float maxExtent;
    float minExtent;
    int pointsCount;
    int pointsCountQuery;
    int finalBucketCount;
    //Prohibit copy, assignment operations
    GridStructure & operator = (const GridStructure&);
    GridStructure(const GridStructure&);
    void InitStructure(float* points_x, float* points_y, float* points_z);
    void recomputeBuffers(int &pointsLeft, int count, int *&dcontinues, int *&dIndexesQuery, int *&dcounters);
    void rearrangeQueryPoints(float*& pntsQx, float*& pntsQy, float*& pntsQz, int queryPointCount,
                              int* queryIndexes);
    bool InitializeLevel(int dev_id, int level, int bucketsCount, float min, float max);
    template <typename T>
    void getBuckets(int dev_id, int level, int bucketsCount, float min, float max, int &maxBucketSize);
    template <typename T>
    void UpdateHelperFromInitialize(NeighborHelper<T>& neighborHelper, int level, int dev_id, int bucketCount);
    void RefreshHierarchies();
};

#endif