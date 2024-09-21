//
// Created by agalex on 6/19/24.
//

#ifndef KNNCUDA_EGT_CUDA_CUH
#define KNNCUDA_EGT_CUDA_CUH

#include "KNNInterface.cuh"
#include <vector>

class EGT_CUDA {
public:
    explicit EGT_CUDA(KNNInterface& knnInterface);
    void PerformSmoothing(int iterationCount, float lambda = 0.63f, float mu = -0.64f, float alpha = 1e-8f);
    virtual ~EGT_CUDA() = default;
private:
    void create_halo_elements();
    void sender_halo_break(int dev_id);
    void communication_sender_to_receiver(int dev_id, std::vector<int*>& receive_halo_flag);
    void receiver_halo_synthesize(int dev_id, int* receive_halo_flag);
    void reset_halo_elements();
    float** din_x;
    float** din_y;
    float** din_z;
    int** dKNN;
    int* points_count_per_device;
    int k;
    int pointsCount;
    int numb_gpus;
    int pointPartitionSize;

    std::vector<int*> dev_haloElements;
    std::vector<int*> dev_gpu_ids;
    std::vector<std::vector<int>> host_gpu_ids;
    std::vector<int*> dev_haloElementsCount;
    std::vector<std::vector<int>> host_haloElementsCount;
    std::vector<int*> send_halo_to_gpu;
    std::vector<std::vector<int>> send_halo_count;
    std::vector<std::vector<int>> receive_halo_count;
    std::vector<std::vector<float*>> receive_halo_buffer;
    std::vector<std::vector<float*>> send_halo_buffer;
    std::vector<float*> receive_halo_buffer_bulk;
    std::vector<float*> send_halo_buffer_bulk;
    std::vector<int> send_halo_elements_count;
    std::vector<int> receive_halo_elements_count;
};

#endif //KNNCUDA_EGT_CUDA_CUH
