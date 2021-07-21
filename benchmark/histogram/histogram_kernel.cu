#ifndef _HISTOGRAM_CUDA_KERNEL
#define _HISTOGRAM_CUDA_KERNEL 

#include "cta_config.h"

__global__ void Histogram(float* input, int* output, int length) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int index = bid * blockDim.x + tid;
    int numThreads = gridDim.x * blockDim.x;
    
    __shared__ int tmp_buffer[NUM_BINS];
    for (int i = tid; i < NUM_BINS; i += blockDim.x) {
        tmp_buffer[i] = 0;
    }
    __syncthreads();

    for (int i = index; i < length; i += numThreads) {
        float val = 255.0 * input[i];
        int bin_id = (int)(val);
        CLAMP(bin_id, 0, NUM_BINS);
        atomicAdd(&tmp_buffer[bin_id], 1);
    }
    __syncthreads();

    for (int i = tid; i < NUM_BINS; i += blockDim.x) {
        output[bid * NUM_BINS + i] = tmp_buffer[i];
    }
    return;
}


__global__ void reduceAll(int* input, int* output, int num_parts) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int index = bid * blockDim.x + tid;

    if (index < NUM_BINS) {
        int freq = 0;
        for (int i = 0; i < num_parts; ++i) {
            freq += input[i * NUM_BINS + index];
        }
        output[index] = freq;
    }

    return;
}

#endif 
