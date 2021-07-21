#ifndef _UPSAMPLE_CUDA_KERNEL
#define _UPSAMPLE_CUDA_KERNEL 

#include "cta_config.h"

__global__ void upSample(float* input, float* output, 
        int num_input_rows, int num_input_cols) {
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int numThreads = gridDim.x * blockDim.x;
    int numElements = num_input_rows * num_input_cols;

    __shared__ float tmp_buffer[NUM_THREADS];

    for (int i = index; i < numElements; i += numThreads) {
        // Load data into shared memory 
        tmp_buffer[tid] = input[i];
        __syncthreads(); 

        for (int kx = 0; kx < 2; ++kx) {
            for (int ky = 0; ky < 2; ++ky) {
                int local_index = (ky * blockDim.x + tid) / 2;
                float curr_val = tmp_buffer[local_index];
                output[(kx * 2 + ky) * numElements + i] = curr_val;
            }
        }
        __syncthreads(); 
    }
    return;
}

#endif 
