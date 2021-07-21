#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <math.h>
#include <assert.h>

#include <cuda.h>
#include "cta_config.h"
#include "../common/cuda_check.h" 

extern __global__ void maxPool(
        float* input, float* output, int num_output_rows, int num_output_cols);

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
        printf("Usag ./maxpool <num of output rows> <num of output columns>");
        return -1;
    }

    int num_output_rows = atoi(argv[1]);
    int num_output_cols = atoi(argv[2]);
    printf("Running the max pooling for an output matrix %d x %d\n",
            num_output_rows, num_output_cols);

    int num_input_rows = num_output_rows * 2;
    int num_input_cols = num_output_cols * 2;
    float* host_input = (float*) malloc(
            num_input_rows * num_input_cols * sizeof(float));
    float* tmp_input = (float*) malloc(
            num_input_rows * num_input_cols * sizeof(float));
    float* host_output = (float*) malloc(
            num_output_rows * num_output_cols * sizeof(float));

    RandFloatArray(tmp_input, num_input_rows * num_input_cols);

    for (int i = 0; i < num_input_rows; i += 2) {
        for (int j = 0; j < num_input_cols; j += (NUM_THREADS * 2)) {
            for (int kx = 0; kx < 2; ++kx) {
                for (int ky = 0; ky < 2; ++ky) {
                    for (int k = 0; k < NUM_THREADS; ++k) {
                        host_input[(kx * 2 + ky) * num_output_rows * num_output_cols 
                            + (i / 2) * num_output_cols + (j / 2) + k] = tmp_input[
                            (i + kx) * num_input_cols + j + ky * NUM_THREADS + k];
                    }
                }
            }
        }
    }

    for (int i = 0; i < num_output_rows; ++i) {
        for (int j = 0; j < num_output_cols; ++j) {
            float max_value = 0.0f;
            for (int kx = 0; kx < 2; kx++) {
                for (int ky = 0; ky < 2; ky++) {
                    float curr_value = tmp_input[
                        (i * 2 + kx) * num_input_cols + (j * 2 + ky)];
                    if (curr_value > max_value) {
                        max_value = curr_value;
                    }
                }
            }
            host_output[i * num_output_cols + j] = max_value;
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

    maxPool<<<NUM_BLOCKS, NUM_THREADS>>>(device_input, device_output, 
            num_output_rows, num_output_cols);
    cudaDeviceSynchronize();

#ifdef MEASURE_POWER 
    }
#endif 

    printf("Completed GPU computation!\n");

    CUDA_CHECK(cudaMemcpy(results, device_output,
            num_output_rows * num_output_cols * sizeof(float),
            cudaMemcpyDeviceToHost));

    AssertArrayEqual(host_output, results, num_output_rows * num_output_cols);
    printf("Correctness Check: Accepted!\n");

    free(host_input);
    free(tmp_input);
    free(host_output);
    free(results);

    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_output));

    return 0;
}
