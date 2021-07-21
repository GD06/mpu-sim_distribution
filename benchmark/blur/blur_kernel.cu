#ifndef _BLUR_CUDA_KERNEL
#define _BLUR_CUDA_KERNEL 

#include "cta_config.h"

__global__ void Blur3x3(float* input, float* output, 
        int num_rows, int num_cols, float alpha) {
    int bidx = blockIdx.x;
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;

    int num_matrix_blocks = (num_rows * num_cols) / (BLOCK_SIZE * BLOCK_SIZE);

    __shared__ float tmp_buffer[BLOCK_SIZE][BLOCK_SIZE];

    for (int block_id = bidx; block_id < num_matrix_blocks; block_id += gridDim.x) {
        for (int y = 0; y < BLOCK_SIZE; y += NUM_THREADS_Y) {
            tmp_buffer[y + tidy][tidx] = input[
                (y / NUM_THREADS_Y) * NUM_THREADS_Y * NUM_THREADS_X * num_matrix_blocks
                + block_id * NUM_THREADS_Y * NUM_THREADS_X
                + tidy * NUM_THREADS_X + tidx];
        }
        __syncthreads();

        for (int y = 0; y < BLOCK_SIZE; y += NUM_THREADS_Y) {
            float sum_val = 0.0f;
            for (int ky = 0; ky < 3; ++ky) {
                for (int kx = 0; kx < 3; ++kx) {
                    int row_index = y + tidy + ky - 1;
                    int col_index = tidx + kx - 1;
                    CLAMP(row_index, 0, BLOCK_SIZE);
                    CLAMP(col_index, 0, BLOCK_SIZE);
                    sum_val += tmp_buffer[row_index][col_index];
                }
            }

            output[
                (y / NUM_THREADS_Y) * NUM_THREADS_Y * NUM_THREADS_X * num_matrix_blocks
                + block_id * NUM_THREADS_Y * NUM_THREADS_X 
                + tidy * NUM_THREADS_X + tidx] = sum_val * alpha;
        }
        __syncthreads(); 
    }

    return;
}

#endif 
