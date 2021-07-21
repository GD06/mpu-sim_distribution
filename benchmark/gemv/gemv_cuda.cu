#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <math.h>
#include <assert.h>

#include <cuda.h>
#include "cta_config.h"
#include "../common/cuda_check.h" 


extern __global__ void GEMV(float* input_matrix, float* input_vector,
        float* output_vector, int num_rows, int num_cols); 


void RandFloatArray(float* ptr, int length) {
    for (int i = 0; i < length; ++i) {
        float val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        ptr[i] = val; 
    }
    return; 
}


void AssertArrayEqual(float* ptr1, float* ptr2, int length, float precision = 1e-5) {
    for (int i = 0; i < length; ++i) {
        assert(fabs(ptr1[i] - ptr2[i]) < 
                precision * max(fabs(ptr1[i]), fabs(ptr2[i]))
        );
    }
    return;
}


int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage ./gemv <num of rows> <num of cols>");
        return -1;
    }

    int num_rows = atoi(argv[1]);
    int num_cols = atoi(argv[2]);
    printf("Running the matrix-vector multiplication for a matrix %d x %d\n",
            num_rows, num_cols);

    float* host_matrix = (float*) malloc(num_rows * num_cols * sizeof(float));
    float* tmp_input = (float*) malloc(num_rows * num_cols * sizeof(float));
    float* host_invec = (float*) malloc(num_cols * sizeof(float));
    float* host_outvec = (float*) malloc(num_rows * sizeof(float));

    RandFloatArray(host_matrix, num_rows * num_cols);
    RandFloatArray(host_invec, num_cols);

    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; j += NUM_THREADS) {
            for (int k = 0; k < NUM_THREADS; ++k) {
                tmp_input[i * num_cols + j + k] = host_matrix[
                    (j / NUM_THREADS) * NUM_THREADS * num_rows +
                    i * NUM_THREADS + k];
            }
        }
    }

    for (int i = 0; i < num_rows; ++i) {
        host_outvec[i] = 0.0f;
        for (int j = 0; j < num_cols; ++j) {
            host_outvec[i] += tmp_input[i * num_cols + j] * host_invec[j];
        }
    }
    printf("Completed ground truth computation!\n");

    float* device_matrix;
    float* device_invec;
    float* device_outvec;

    CUDA_CHECK(cudaMalloc((void**) &device_matrix, 
                num_rows * num_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**) &device_invec, num_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**) &device_outvec, num_rows * sizeof(float)));
    float* results = (float*) malloc(num_rows * sizeof(float));

    CUDA_CHECK(cudaMemcpy(device_matrix, host_matrix, 
                num_rows * num_cols * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_invec, host_invec, 
                num_cols * sizeof(float), cudaMemcpyHostToDevice));

#ifdef MEASURE_POWER 
    while (true) {
#endif 

    GEMV<<<NUM_BLOCKS, NUM_THREADS>>>(device_matrix, device_invec,
            device_outvec, num_rows, num_cols);
    cudaDeviceSynchronize();

#ifdef MEASURE_POWER 
    }
#endif 

    printf("Completed GPU computation!\n");

    CUDA_CHECK(cudaMemcpy(results, device_outvec, 
            num_rows * sizeof(float), cudaMemcpyDeviceToHost));

    AssertArrayEqual(host_outvec, results, num_rows, 1e-3);
    printf("Correctness Check: Accepted!\n");

    free(host_matrix);
    free(tmp_input);
    free(host_invec);
    free(host_outvec);
    free(results);

    CUDA_CHECK(cudaFree(device_matrix));
    CUDA_CHECK(cudaFree(device_invec));
    CUDA_CHECK(cudaFree(device_outvec));

    return 0; 
}
