#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <math.h>
#include <assert.h>

#include <cuda.h>
#include "cta_config.h"
#include "../common/cuda_check.h" 

extern __global__ void vectorSum(float* input, float* output, int N);


void RandFloatArray(float* ptr, int length) {
    for (int i = 0; i < length; ++i) {
        float val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        ptr[i] = val;
    }
    return; 
}


int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: ./vectorsum_cuda <length of vector>\n");
        return -1;
    }

    int N = atoi(argv[1]);
    printf("Running vector sum for length: %d\n", N);

    float* host_a = (float*) malloc(N * sizeof(float));

    RandFloatArray(host_a, N);

    float host_sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        host_sum += host_a[i];
    }
    printf("Completed ground truth computation\n");

    
    float* device_a;
    float* device_tmp;
    float* device_result;

    CUDA_CHECK(cudaMalloc((void**)&device_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&device_tmp, NUM_BLOCKS * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&device_result, 1 * sizeof(float)));
    float* result_c = (float*) malloc (1 * sizeof(float));

    CUDA_CHECK(cudaMemcpy(device_a, host_a, N * sizeof(float), cudaMemcpyHostToDevice));

#ifdef MEASURE_POWER
    while (true) {
#endif 

    vectorSum<<<NUM_BLOCKS, NUM_THREADS>>>(device_a, device_tmp, N);
    cudaDeviceSynchronize();

    vectorSum<<<1, NUM_THREADS>>>(device_tmp, device_result, NUM_BLOCKS);
    cudaDeviceSynchronize();

#ifdef MEASURE_POWER 
    } 
#endif 

    printf("Completed GPU computation!\n");

    CUDA_CHECK(cudaMemcpy(result_c, device_result, 1 * sizeof(float), cudaMemcpyDeviceToHost));

    // printf("Host result: %5f\n", host_sum);
    // printf("Device result: %5f\n", result_c[0]);

    float precision = 1e-4;
    assert((fabs(host_sum - result_c[0])) / max(fabs(host_sum), fabs(precision)) < precision);
    printf("Correctness Check: Accepted!\n");

    free(host_a);

    CUDA_CHECK(cudaFree(device_a));
    CUDA_CHECK(cudaFree(device_tmp));
    CUDA_CHECK(cudaFree(device_result));

    return 0;
}
