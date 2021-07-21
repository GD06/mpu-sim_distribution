#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <math.h>
#include <assert.h>

#include <cuda.h>
#include "cta_config.h"
#include "../common/cuda_check.h" 

extern __global__ void upSample(
        float* input, float* output, int num_input_rows, int num_input_cols);

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
        printf("Usag ./upsample <num of input rows> <num of input columns>");
        return -1;
    }

    int num_input_rows = atoi(argv[1]);
    int num_input_cols = atoi(argv[2]);
    printf("Running the up sampling for an input matrix %d x %d\n",
            num_input_rows, num_input_cols);

    int num_output_rows = num_input_rows * 2;
    int num_output_cols = num_input_cols * 2;
    float* host_input = (float*) malloc(
            num_input_rows * num_input_cols * sizeof(float));
    float* host_output = (float*) malloc(
            num_output_rows * num_output_cols * sizeof(float));

    RandFloatArray(host_input, num_input_rows * num_input_cols);

    for (int i = 0; i < num_input_rows; ++i) {
        for (int j = 0; j < num_input_cols; j += NUM_THREADS) {
            for (int kx = 0; kx < 2; ++kx) {
                for (int ky = 0; ky < 2; ++ky) {
                    for (int k = 0; k < NUM_THREADS; ++k) {
                        host_output[(kx * 2 + ky) * num_input_rows * num_input_cols 
                            + i * num_input_cols + (j + k)] = host_input[
                                i * num_input_cols + j + (ky * NUM_THREADS + k) / 2];
                    }
                }
            }
        }
    }
    printf("Completed ground truth computation!\n");

    float* device_input;
    float* device_output;

    CUDA_CHECK(cudaMalloc((void**) &device_input, 
                num_input_rows * num_input_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**) &device_output,
                num_output_rows * num_output_cols * sizeof(float)));
    float* results = (float*) malloc(
            num_output_rows * num_output_cols * sizeof(float));

    CUDA_CHECK(cudaMemcpy(device_input, host_input,
            num_input_rows * num_input_cols * sizeof(float),
            cudaMemcpyHostToDevice));

#ifdef MEASURE_POWER 
    while (true) {
#endif  

    upSample<<<NUM_BLOCKS, NUM_THREADS>>>(device_input, device_output,
            num_input_rows, num_input_cols);
    cudaDeviceSynchronize();

#ifdef MEASURE_POWER
    }
#endif 

    printf("Completed GPU computation!\n");
    
    CUDA_CHECK(cudaMemcpy(results, device_output,
            num_output_rows * num_output_cols * sizeof(float),
            cudaMemcpyDeviceToHost));

    AssertArrayEqual(host_output, results, num_output_rows * num_output_cols);
    printf("Correctness Check: Acceptedd!\n");

    free(host_input);
    free(host_output);
    free(results);

    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_output));

    return 0;
}
