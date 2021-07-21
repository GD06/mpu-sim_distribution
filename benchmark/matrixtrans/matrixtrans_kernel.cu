#ifndef _MATRIX_TRANS_CUDA_KERNEL
#define _MATRIX_TRANS_CUDA_KERNEL 

#include "cta_config.h"

__global__ void matrixTranspose(
        float* input, float* output, int num_rows, int num_cols) {
    int bidx = blockIdx.x;
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;

    int num_matrix_blocks = (num_rows * num_cols) / (BLOCK_SIZE * BLOCK_SIZE); 

    __shared__ float trans_buffer[BLOCK_SIZE][BLOCK_SIZE + 1];

    for (int block_id = bidx; block_id < num_matrix_blocks; block_id += gridDim.x) {
        for (int y = 0; y < BLOCK_SIZE; y += NUM_THREADS_Y) {
            trans_buffer[y + tidy][tidx] = input[
                (y / NUM_THREADS_Y) * NUM_THREADS_Y * NUM_THREADS_X * num_matrix_blocks 
                + block_id * NUM_THREADS_Y * NUM_THREADS_X 
                + tidy * NUM_THREADS_X + tidx];
        }
        __syncthreads();

        for (int y = 0; y < BLOCK_SIZE; y += NUM_THREADS_Y) {
            output[(y / NUM_THREADS_Y) * NUM_THREADS_X * NUM_THREADS_Y * num_matrix_blocks
                + block_id * NUM_THREADS_Y * NUM_THREADS_X
                + tidy * NUM_THREADS_X + tidx] = trans_buffer[tidx][y + tidy];
        }
        __syncthreads();
    }
    return; 
}

#endif 
