#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <math.h>
#include <assert.h>

#include <cuda.h>
#include "cta_config.h"
#include "../common/cuda_check.h" 

extern __global__ void Blur3x3(
        float* input, float* output, int num_rows, int num_cols, float alpha);

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
        printf("Usage: ./blur <num of rows> <num of columns>");
        return -1;
    }

    int num_rows = atoi(argv[1]);
    int num_cols = atoi(argv[2]);
    printf("Running the image blur for an image %d x %d\n",
            num_rows, num_cols);

    float* host_input = (float*) malloc(num_rows * num_cols * sizeof(float));
    float* host_output = (float*) malloc(num_rows * num_cols * sizeof(float));

    RandFloatArray(host_input, num_rows * num_cols);

    int num_matrix_blocks = (num_rows * num_cols) / (BLOCK_SIZE * BLOCK_SIZE);
    for (int bid = 0; bid < num_matrix_blocks; ++bid) {
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            for (int j = 0; j < BLOCK_SIZE; ++j) {
                float sum_val = 0.0f;
                for (int ki = 0; ki < 3; ++ki) {
                    for (int kj = 0; kj < 3; ++kj) {
                        int row_index = i + ki - 1;
                        int col_index = j + kj - 1;
                        CLAMP(row_index, 0, BLOCK_SIZE);
                        CLAMP(col_index, 0, BLOCK_SIZE);

                        int slice_id = (row_index / NUM_THREADS_Y);
                        int row_id = (row_index % NUM_THREADS_Y);
                        sum_val += host_input[
                            slice_id * NUM_THREADS_X * NUM_THREADS_Y * num_matrix_blocks
                            + bid * NUM_THREADS_X * NUM_THREADS_Y
                            + row_id * NUM_THREADS_X + col_index];
                    }
                }

                int dst_slice_id = (i / NUM_THREADS_Y);
                int dst_row_id = (i % NUM_THREADS_Y);
                host_output[
                    dst_slice_id * NUM_THREADS_X * NUM_THREADS_Y * num_matrix_blocks
                    + bid * NUM_THREADS_X * NUM_THREADS_Y
                    + dst_row_id * NUM_THREADS_X + j] = (sum_val / 9.0f);
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

    Blur3x3<<<NUM_BLOCKS, dimBlock>>>(device_input, device_output, 
            num_rows, num_cols, (1.0f / 9.0f));
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
