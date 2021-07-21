#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <math.h>
#include <assert.h>

#include <cuda.h>
#include "cta_config.h"
#include "../common/cuda_check.h" 

extern __global__ void Histogram(float* input, int* output, int length);

extern __global__ void reduceAll(int* input, int* output, int num_parts);

void RandFloatArray(float* ptr, int length) {
    for (int i = 0; i < length; ++i) {
        float val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        ptr[i] = val; 
    }
    return; 
}

void AssertArrayEqual(int* ptr1, int* ptr2, int length) {
    int num_diff = 0;
    for (int i = 0; i < length; ++i) {
        if (ptr1[i] != ptr2[i]) { 
            printf("Index %d: %d vs. %d\n", i, ptr1[i], ptr2[i]);
            num_diff++;
        }
    }
    assert(num_diff < 10);
    return;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: ./histogram <num of elements>");
        return -1;
    }

    int length = atoi(argv[1]);
    printf("Running the histogram on %d elements\n", length);

    float* host_input = (float*) malloc(length * sizeof(float));
    int* host_output = (int*) malloc(NUM_BINS * sizeof(int));

    RandFloatArray(host_input, length);

    for (int i = 0; i < NUM_BINS; ++i) {
        host_output[i] = 0;
    }

    for (int i = 0; i < length; ++i) {
        float val = host_input[i] * 255.0;
        int bin_id = (int)(val);
        CLAMP(bin_id, 0, NUM_BINS);
        host_output[bin_id] += 1;
    }
    printf("Completed ground truth computation!\n");

    float* device_input;
    int* device_part_output;
    int* device_output; 

    CUDA_CHECK(cudaMalloc((void**) &device_input, length * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**) &device_part_output,
                NUM_BLOCKS * NUM_BINS * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**) &device_output, NUM_BINS * sizeof(int)));
    int* results = (int*) malloc(NUM_BINS * sizeof(int));

    CUDA_CHECK(cudaMemcpy(device_input, host_input, length * sizeof(float),
                cudaMemcpyHostToDevice));

#ifdef MEASURE_POWER
    while (true) {
#endif 

    Histogram<<<NUM_BLOCKS, NUM_THREADS>>>(
            device_input, device_part_output, length);
    cudaDeviceSynchronize();

    reduceAll<<<NUM_BLOCKS, NUM_THREADS>>>(
            device_part_output, device_output, NUM_BLOCKS);
    cudaDeviceSynchronize();

#ifdef MEASURE_POWER 
    }
#endif 

    printf("Completed GPU computation!\n");

    CUDA_CHECK(cudaMemcpy(results, device_output, 
                NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost));

    AssertArrayEqual(host_output, results, NUM_BINS);
    printf("Correctness Check: Accepted!\n");

    free(host_input);
    free(host_output);
    free(results);

    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_part_output));
    CUDA_CHECK(cudaFree(device_output));

    return 0; 
}
