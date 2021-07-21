#ifndef _KMEANS_CUDA_KERNEL
#define _KMEANS_CUDA_KERNEL 

#include "cta_config.h"


__global__ void kMeans(float* data_points, float* cluster_centers,
        int* membership, int num_points) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int numThreads = gridDim.x * blockDim.x;

    __shared__ float shared_cluster_centers[NUM_CLUSTERS * NUM_FEATURES];
    for (int i = threadIdx.x; i < (NUM_CLUSTERS * NUM_FEATURES); i += blockDim.x) {
        shared_cluster_centers[i] = cluster_centers[i];
    }
    __syncthreads(); 

    for (int point_id = index; point_id < num_points; point_id += numThreads) {
        int min_center_id = 0;
        float min_dist = 0.0f;

        for (int center_id = 0; center_id < NUM_CLUSTERS; ++center_id) {
            float dist = 0.0f;
            for (int feature_id = 0; feature_id < NUM_FEATURES; ++feature_id) {
                float dp_feature = data_points[feature_id * num_points + point_id];
                float center_feature = shared_cluster_centers[
                    feature_id * NUM_CLUSTERS + center_id];
                float diff = (dp_feature - center_feature);
                dist += diff * diff;
            }

            if (center_id == 0) {
                min_dist = dist;
            }

            if (dist < min_dist) {
                min_dist = dist;
                min_center_id = center_id;
            }
        }

        membership[point_id] = min_center_id; 
    }
}

#endif 
