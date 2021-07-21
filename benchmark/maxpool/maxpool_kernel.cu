#ifndef _MAXPOOL_CUDA_KERNEL
#define _MAXPOOL_CUDA_KERNEL 

#include "cta_config.h"

__global__ void maxPool(float* input, float* output, 
        int num_output_rows, int num_output_cols) {
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int numThreads = gridDim.x * blockDim.x;
    int numElements = num_output_rows * num_output_cols; 

    __shared__ float tmp_buffer[2][NUM_THREADS * 2];

    for (int i = index; i < numElements; i += numThreads) {
        // Load data into shared memory 
        for (int kx = 0; kx < 2; ++kx) {
            for (int ky = 0; ky < 2; ++ky) {
                tmp_buffer[kx][ky * blockDim.x + tid] = input[
                    (kx * 2 + ky) * numElements + i];
            }
        }
        __syncthreads();

        // Compute the max value over the 2x2 sampling window
        float max_value = 0.0f;
        for (int kx = 0; kx < 2; ++kx) {
            for (int ky = 0; ky < 2; ++ky) {
                float curr_value = tmp_buffer[kx][tid * 2 + ky];
                if (curr_value > max_value) {
                    max_value = curr_value;
                }
            }
        }
        __syncthreads();

        // Write back the computation results 
        output[i] = max_value;
    }
    return;
}

#endif 
