#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <math.h>
#include <assert.h>

#include <cuda.h>
#include "../common/cuda_check.h" 

#define NUM_BLOCKS 512
#define NUM_THREADS 128

extern __global__ void vectorAdd(float* a, float* b, float* c, int N);


void RandFloatArray(float* ptr, int length) {
    for (int i = 0; i < length; ++i) {
        float val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        ptr[i] = val; 
    }
    return; 
}


void AssertArrayEqual(float* ptr1, float* ptr2, int length, float precision = 1e-5) {
    for (int i = 0; i < length; ++i) {
        assert(fabs(ptr1[i] - ptr2[i]) < precision);
    }
    return;
}


int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: ./vectoradd_cuda <length of vector>\n");
        return -1;
    }

    int N = atoi(argv[1]);
    printf("Running vector add for length: %d\n", N);

    float* host_a = (float*) malloc(N * sizeof(float));
    float* host_b = (float*) malloc(N * sizeof(float));
    float* host_c = (float*) malloc(N * sizeof(float));

    RandFloatArray(host_a, N);
    RandFloatArray(host_b, N);

    for (int i = 0; i < N; ++i) {
        host_c[i] = host_a[i] + host_b[i];
    }
    printf("Completed ground truth computation\n");


    float* device_a;
    float* device_b;
    float* device_c;

    CUDA_CHECK(cudaMalloc((void**) &device_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**) &device_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**) &device_c, N * sizeof(float)));
    float* result_c = (float*) malloc(N * sizeof(float));

    CUDA_CHECK(cudaMemcpy(device_a, host_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_b, host_b, N * sizeof(float), cudaMemcpyHostToDevice));

#ifdef MEASURE_POWER
    while (true) {
#endif 

    vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(device_a, device_b, device_c, N);
    cudaDeviceSynchronize();

#ifdef MEASURE_POWER 
    }
#endif 

    printf("Completed GPU computation!\n");

    CUDA_CHECK(cudaMemcpy(result_c, device_c, N * sizeof(float), cudaMemcpyDeviceToHost));

    AssertArrayEqual(host_c, result_c, N);
    printf("Correctness Check: Accepted!\n");

    free(host_a);
    free(host_b);
    free(host_c);
    free(result_c);

    CUDA_CHECK(cudaFree(device_a));
    CUDA_CHECK(cudaFree(device_b));
    CUDA_CHECK(cudaFree(device_c));

    return 0;
}
