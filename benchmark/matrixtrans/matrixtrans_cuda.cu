#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <math.h>
#include <assert.h>

#include <cuda.h>
#include "cta_config.h"
#include "../common/cuda_check.h" 

extern __global__ void matrixTranspose(
        float* input, float* output, int num_rows, int num_cols);

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
    if (argc < 3) {
        printf("Usag ./matrixtrans_cuda <num of rows> <num of columns>");
        return -1;
    }

    int num_rows = atoi(argv[1]);
    int num_cols = atoi(argv[2]);
    assert(num_rows % BLOCK_SIZE == 0);
    assert(num_cols % BLOCK_SIZE == 0);
    printf("Running the matrix transposition for a matrix %d x %d\n",
            num_rows, num_cols);

    float* host_input = (float*) malloc(num_rows * num_cols * sizeof(float));
    float* host_output = (float*) malloc(num_rows * num_cols * sizeof(float));

    RandFloatArray(host_input, num_rows * num_cols);

    int num_matrix_blocks = (num_rows * num_cols) / (BLOCK_SIZE * BLOCK_SIZE);
    for (int bid = 0; bid < num_matrix_blocks; ++bid) {
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            for (int j = 0; j < BLOCK_SIZE; ++j) {
                int src_slice_id = (i / NUM_THREADS_Y);
                int src_row_id = (i % NUM_THREADS_Y);
                float val = host_input[
                    src_slice_id * NUM_THREADS_X * NUM_THREADS_Y * num_matrix_blocks
                    + bid * NUM_THREADS_X * NUM_THREADS_Y 
                    + src_row_id * NUM_THREADS_X + j];

                int dst_slice_id = (j / NUM_THREADS_Y);
                int dst_row_id = (j % NUM_THREADS_Y);
                host_output[
                    dst_slice_id * NUM_THREADS_X * NUM_THREADS_Y * num_matrix_blocks
                    + bid * NUM_THREADS_X * NUM_THREADS_Y 
                    + dst_row_id * NUM_THREADS_X + i] = val;
            }
        }
    }
    printf("Completed ground truth computation!\n");

    float* device_input;
    float* device_output;

    CUDA_CHECK(cudaMalloc((void**) &device_input, 
                num_rows * num_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**) &device_output, 
                num_rows * num_cols * sizeof(float)));
    float* results = (float*) malloc(num_rows * num_cols * sizeof(float));

    CUDA_CHECK(cudaMemcpy(device_input, host_input, 
                num_rows * num_cols * sizeof(float), cudaMemcpyHostToDevice));

    dim3 dimBlock(NUM_THREADS_X, NUM_THREADS_Y);

#ifdef MEASURE_POWER
    while (true) {
#endif 

    matrixTranspose<<<NUM_BLOCKS, dimBlock>>>(device_input, device_output, num_rows, num_cols);
    cudaDeviceSynchronize();

#ifdef MEASURE_POWER
    }
#endif 

    printf("Completed GPU computation!\n");

    CUDA_CHECK(cudaMemcpy(results, device_output,
                num_rows * num_cols * sizeof(float), cudaMemcpyDeviceToHost));

    AssertArrayEqual(host_output, results, num_rows * num_cols);
    printf("Correctness Check: Accepted!\n");

    free(host_input);
    free(host_output);
    free(results);

    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_output));

    return 0;  
} 
