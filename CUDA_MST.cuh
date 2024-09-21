//
// Created by agalex on 7/25/23.
//

#ifndef KNNCUDA_CUDA_MST_CUH
#define KNNCUDA_CUDA_MST_CUH

typedef int intT;

template <class intT>
struct wghEdge {
    intT u, v;
    float weight;
    __host__ __device__
    wghEdge() {}
    __host__ __device__
    wghEdge(intT _u, intT _v, float w) : u(_u), v(_v), weight(w) {}
};

template <class intT>
struct wghEdgeArray {
    wghEdge<intT> *E;
    intT n; intT m;
    wghEdgeArray(wghEdge<intT>* EE, intT nn, intT mm) : E(EE), n(nn), m(mm) {}
    void del() { cudaFree(E);}
};

std::pair<intT*,intT> mst(wghEdgeArray<intT>& G);
#endif //KNNCUDA_CUDA_MST_CUH
