#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <math.h>
#include <assert.h>

#include <cuda.h>
#include "cta_config.h"
#include "../common/cuda_check.h" 

extern __global__ void kMeans(float* data_points, float* cluster_centers,
        int* membership, int num_points);

void RandFloatArray(float* ptr, int length) {
    for (int i = 0; i < length; ++i) {
        float val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        ptr[i] = val; 
    }
    return; 
}

void AssertIntArrayAlmostEqual(int* ptr1, int* ptr2, int length) {
    int diff_count = 0;
    for (int i = 0; i < length; ++i) {
        if (ptr1[i] != ptr2[i]) { diff_count++; }
    }
    printf("Number of different elements: %d\n", diff_count);
    assert(diff_count <= 1);
    return;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: ./kmeans <number of data points>");
        return -1;
    }

    int num_points = atoi(argv[1]);
    printf("Running the kmeans for %d points, %d clusters, and %d features\n",
            num_points, NUM_CLUSTERS, NUM_FEATURES);

    float* host_points = (float*) malloc(
            num_points * NUM_FEATURES * sizeof(float));
    float* host_centers = (float*) malloc(
            NUM_CLUSTERS * NUM_FEATURES * sizeof(float));
    int* host_membership = (int*) malloc(num_points * sizeof(int));

    RandFloatArray(host_points, num_points * NUM_FEATURES);
    RandFloatArray(host_centers, NUM_CLUSTERS * NUM_FEATURES);

    for (int i = 0; i < num_points; ++i) {
        int index = 0;
        float min_dist = 0.0f;
        for (int j = 0; j < NUM_CLUSTERS; ++j) {
            float dist = 0.0f;
            for (int k = 0; k < NUM_FEATURES; ++k) {
                float diff = (host_points[k * num_points + i] 
                        - host_centers[k * NUM_CLUSTERS + j]);
                dist += diff * diff;
            }

            if (j == 0) {
                min_dist = dist;
            }

            if (dist < min_dist) {
                index = j;
                min_dist = dist;
            }
        }
        host_membership[i] = index; 
    }
    printf("Complted ground truth computation!\n");

    float* device_points;
    float* device_centers;
    int* device_membership;

    CUDA_CHECK(cudaMalloc((void**) &device_points, 
                NUM_FEATURES * num_points * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**) &device_centers,
                NUM_FEATURES * NUM_CLUSTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**) &device_membership,
                num_points * sizeof(int)));
    int* results = (int*)malloc(num_points * sizeof(int));

    CUDA_CHECK(cudaMemcpy(device_points, host_points, 
        NUM_FEATURES * num_points * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_centers, host_centers,
        NUM_FEATURES * NUM_CLUSTERS * sizeof(float), cudaMemcpyHostToDevice));

#ifdef MEASURE_POWER 
    while (true) {
#endif 

    kMeans<<<NUM_BLOCKS, NUM_THREADS>>>(
        device_points, device_centers, device_membership, num_points);
    cudaDeviceSynchronize();

#ifdef MEASURE_POWER 
    }
#endif 

    printf("Completed GPU computation!\n");

    CUDA_CHECK(cudaMemcpy(results, device_membership, 
                num_points * sizeof(int), cudaMemcpyDeviceToHost));

    AssertIntArrayAlmostEqual(host_membership, results, num_points);
    printf("Correctness Check: Accepted!\n");

    free(host_points);
    free(host_centers);
    free(host_membership);
    free(results);

    CUDA_CHECK(cudaFree(device_points));
    CUDA_CHECK(cudaFree(device_centers));
    CUDA_CHECK(cudaFree(device_membership));

    return 0;
}
