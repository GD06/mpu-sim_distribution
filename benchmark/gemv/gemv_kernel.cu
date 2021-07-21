#ifndef _GEMV_CUDA_KERNEL
#define _GEMV_CUDA_KERNEL

#include "cta_config.h"


__global__ void GEMV(float* input_matrix, float* input_vector, 
        float* output_vector, int num_rows, int num_cols) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int num_blocks = gridDim.x;

    __shared__ float psum[NUM_THREADS];

    for (int row_index = bid; row_index < num_rows; row_index += num_blocks) {
        float sum = 0.0f;

        int col_index = tid;
        for (int global_index = (row_index * NUM_THREADS + tid); 
                global_index < (num_rows * num_cols); 
                global_index += (num_rows * NUM_THREADS)) {
            sum += input_matrix[global_index] * input_vector[col_index];
            col_index += NUM_THREADS;
        }

        psum[tid] = sum;
        __syncthreads();

        for (int i = NUM_THREADS / 2; i >= 1; i = (i >> 1)) {
            if (tid < i) {
                psum[tid] += psum[tid + i];
            }
            __syncthreads();
        }

        if (tid == 0) {
            output_vector[row_index] = psum[0];
        }
    }
    return; 
}

#endif 
