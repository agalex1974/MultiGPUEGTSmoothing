//
// Created by agalex on 6/14/24.
//

#ifndef KNNCUDA_KNNINTERFACE_CUH
#define KNNCUDA_KNNINTERFACE_CUH

class KNNInterface{
public:
    [[nodiscard]] virtual bool isReady() const = 0;
    [[nodiscard]] virtual int GetNumberOfCards() const = 0;
    virtual float** GetRefPointsX() = 0;
    virtual float** GetRefPointsY() = 0;
    virtual float** GetRefPointsZ() = 0;

    virtual float** GetQuerPointsX(){return GetRefPointsX();};
    virtual float** GetQuerPointsY(){return GetRefPointsY();};
    virtual float** GetQuerPointsZ(){return GetRefPointsZ();};

    virtual int** GetKNNIndexesInPartitions() = 0;
    [[nodiscard]] virtual int GetPartitionSize() const = 0;
    virtual int* GetPointCountInCards() = 0;
    virtual int** GetOriginalRefIndexes() = 0;
    virtual int** GetOriginalQuerIndexes(){return GetOriginalRefIndexes();}
    virtual void GetKNNIndexes(int* knnIndexes) = 0;
    [[nodiscard]] virtual float GetMaxExtent() const = 0;
    [[nodiscard]] virtual float GetMinExtent() const = 0;
    [[nodiscard]] virtual int pointsRefCount() const = 0;
    [[nodiscard]] virtual int pointsQuerCount() const{return pointsRefCount();};
    [[nodiscard]] virtual int NeighborCount() const = 0;
};

#endif //KNNCUDA_KNNINTERFACE_CUH
