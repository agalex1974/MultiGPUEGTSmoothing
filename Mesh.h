//
// Created by agalex on 11/16/23.
//

#ifndef KNNCUDA_MESH_H
#define KNNCUDA_MESH_H
#include <vector>

struct V3 {
    float x;
    float y;
    float z;
};

struct I3 {
    uint32_t v0;
    uint32_t v1;
    uint32_t v2;
};

struct C3 {
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

struct C4 {
    uint8_t a;
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

struct Mesh {
    V3* VData() { return mV.data(); }
    I3* FData() { return mF.data(); }
    V3* NData() { return mN.data(); }
    C4* CData() { return mC.data(); }

    [[nodiscard]] size_t nV() const { return mV.size(); }
    [[nodiscard]] size_t nF() const { return mF.size(); }
    [[nodiscard]] size_t nN() const { return mN.size(); }
    [[nodiscard]] size_t nC() const { return mC.size(); }
    void clear() { mV.clear(); mF.clear(); mN.clear(); mC.clear(); }

    std::vector<V3> mV; // Vertices
    std::vector<I3> mF; // Faces
    std::vector<V3> mN; // Normals
    std::vector<C4> mC; // Colors
};

#endif //KNNCUDA_MESH_H
