#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <math.h>
#include <assert.h>

#include <cuda.h>
#include "cta_config.h"
#include "../common/cuda_check.h" 


extern __global__ void kNN(float* x, float* y, float* dist, int N,
        float point_x, float point_y);


void RandFloatArray(float* ptr, int length) {
    for (int i = 0; i < length; ++i) {
        float val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        ptr[i] = val; 
    }
    return; 
}


void AssertArrayEqual(float* ptr1, float* ptr2, int length, float precision = 1e-5) {
    for (int i = 0; i < length; ++i) {
        assert(fabs(ptr1[i] - ptr2[i]) < 
                precision * max(fabs(ptr1[i]), fabs(ptr2[i]))
        );
    }
    return;
}


int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage ./knn <number of data points>");
        return -1;
    }

    int num_points = atoi(argv[1]);
    printf("Running the knn for %d points\n", num_points); 
    
    float* host_x = (float*) malloc(num_points * sizeof(float));
    float* host_y = (float*) malloc(num_points * sizeof(float));
    float* host_dist = (float*) malloc(num_points * sizeof(float));
    
    RandFloatArray(host_x, num_points);
    RandFloatArray(host_y, num_points);

    float point_x = 0.5;
    float point_y = 0.5;

    for (int i = 0; i < num_points; ++i) {
        float diff_x = host_x[i] - point_x;
        float diff_y = host_y[i] - point_y;
        host_dist[i] = (diff_x * diff_x + diff_y * diff_y);
    }
    printf("Completed ground truth computation!\n");

    float* device_x;
    float* device_y;
    float* device_dist;

    CUDA_CHECK(cudaMalloc((void**) &device_x, num_points * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**) &device_y, num_points * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**) &device_dist, num_points * sizeof(float)));
    float* results = (float*) malloc(num_points * sizeof(float));

    CUDA_CHECK(cudaMemcpy(device_x, host_x, num_points * sizeof(float),
                cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_y, host_y, num_points * sizeof(float),
                cudaMemcpyHostToDevice));

#ifdef MEASURE_POWER 
    while (true) {
#endif 

    kNN<<<NUM_BLOCKS, NUM_THREADS>>>(
        device_x, device_y, device_dist, num_points, point_x, point_y);
    cudaDeviceSynchronize();

#ifdef MEASURE_POWER
    }
#endif 

    printf("Completed GPU computation!\n");

    CUDA_CHECK(cudaMemcpy(results, device_dist, num_points * sizeof(float),
                cudaMemcpyDeviceToHost));

    AssertArrayEqual(results, host_dist, num_points);
    printf("Correctness Check: Accepted!\n");

    free(host_x);
    free(host_y);
    free(host_dist);
    free(results);

    CUDA_CHECK(cudaFree(device_x));
    CUDA_CHECK(cudaFree(device_y));
    CUDA_CHECK(cudaFree(device_dist));

    return 0;
}
