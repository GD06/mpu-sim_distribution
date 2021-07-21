#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <math.h>
#include <assert.h>

#include <cuda.h>
#include "cta_config.h"
#include "../common/cuda_check.h" 

extern __global__ void seqAlign(int* input_seq_1, int* input_seq_2, 
        int* input_ref, int* output_matrix, int num_pairs, int penalty);

void RandIntArray(int* ptr, int length) {
    for (int i = 0; i < length; ++i) {
        float val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        float fp_val = (val - 0.5) * 10.0;
        ptr[i] = int(fp_val); 
    }
    return; 
}

void AssertArrayEqual(int* ptr1, int* ptr2, int length) {
    int num_diff = 0;
    for (int i = 0; i < length; ++i) {
        if (ptr1[i] != ptr2[i]) { 
            printf("Index %d: %d vs. %d\n", i, ptr1[i], ptr2[i]);
            num_diff++;
        }
    }
    assert(num_diff == 0);
    return;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: ./nw <num of pairs>");
        return -1;
    }

    int num_pairs = atoi(argv[1]);
    printf("Running the NW on %d sequence pairs\n", num_pairs);

    int* host_seq_1 = (int*) malloc(num_pairs * SEQ_LEN * sizeof(int));
    int* host_seq_2 = (int*) malloc(num_pairs * SEQ_LEN * sizeof(int));
    int* host_ref = (int*) malloc(num_pairs * SEQ_LEN * SEQ_LEN * sizeof(int));
    int* host_output = (int* ) malloc(num_pairs * SEQ_LEN * SEQ_LEN * sizeof(int));
    int penalty = 1;

    RandIntArray(host_seq_1, num_pairs * SEQ_LEN);
    RandIntArray(host_seq_2, num_pairs * SEQ_LEN);
    RandIntArray(host_ref, num_pairs * SEQ_LEN * SEQ_LEN);

    for (int seq_id = 0; seq_id < num_pairs; seq_id++) {
        int tmp[SEQ_LEN + 1][SEQ_LEN + 1];
        int ref[SEQ_LEN][SEQ_LEN];

        tmp[0][0] = 0;
        for (int x = 0; x < SEQ_LEN; x++) {
            tmp[0][x + 1] = host_seq_1[seq_id * SEQ_LEN + x];
        }
        for (int y = 0; y < SEQ_LEN; y++) {
            tmp[y + 1][0] = host_seq_2[seq_id * SEQ_LEN + y];
        }
        for (int y = 0; y < SEQ_LEN; y++) {
            for (int x = 0; x < SEQ_LEN; x++) {
                ref[y][x] = host_ref[y * num_pairs * SEQ_LEN + seq_id * SEQ_LEN + x];
            }
        }

        for (int y = 1; y < (SEQ_LEN + 1); y++) {
            for (int x = 1; x < (SEQ_LEN + 1); x++) {
                int max_val = tmp[y - 1][x - 1] + ref[y - 1][x - 1];
                int tmp_val;

                tmp_val = tmp[y][x - 1] - penalty;
                if (tmp_val > max_val) { max_val = tmp_val; }

                tmp_val = tmp[y - 1][x] - penalty;
                if (tmp_val > max_val) { max_val = tmp_val; }

                tmp[y][x] = max_val;
            }
        }

        for (int y = 0; y < SEQ_LEN; ++y) {
            for (int x = 0; x < SEQ_LEN; ++x) {
                host_output[y * num_pairs * SEQ_LEN + seq_id * SEQ_LEN + x] = (
                        tmp[y + 1][x + 1]);
            }
        }
    }
    printf("Completed ground truth computation!\n");

    int* device_seq_1;
    int* device_seq_2;
    int* device_ref;
    int* device_output;

    CUDA_CHECK(cudaMalloc((void**) &device_seq_1, num_pairs * SEQ_LEN * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**) &device_seq_2, num_pairs * SEQ_LEN * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**) &device_ref, 
                num_pairs * SEQ_LEN * SEQ_LEN * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**) &device_output, 
                num_pairs * SEQ_LEN * SEQ_LEN * sizeof(int)));
    int* results = (int*) malloc(num_pairs * SEQ_LEN * SEQ_LEN * sizeof(int));

    CUDA_CHECK(cudaMemcpy(device_seq_1, host_seq_1, 
                num_pairs * SEQ_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_seq_2, host_seq_2,
                num_pairs * SEQ_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_ref, host_ref,
                num_pairs * SEQ_LEN * SEQ_LEN * sizeof(int),
                cudaMemcpyHostToDevice));

#ifdef MEASURE_POWER 
    while (true) {
#endif 

    seqAlign<<<NUM_BLOCKS, NUM_THREADS>>>(device_seq_1, device_seq_2,
            device_ref, device_output, num_pairs, penalty);
    cudaDeviceSynchronize();

#ifdef MEASURE_POWER 
    }
#endif 

    printf("Completed GPU computation!\n");

    CUDA_CHECK(cudaMemcpy(results, device_output, 
                num_pairs * SEQ_LEN * SEQ_LEN * sizeof(int),
                cudaMemcpyDeviceToHost));

    AssertArrayEqual(host_output, results, num_pairs * SEQ_LEN * SEQ_LEN);
    printf("Correctness Check: Accepted!\n");

    free(host_seq_1);
    free(host_seq_2);
    free(host_ref);
    free(host_output);
    free(results);

    CUDA_CHECK(cudaFree(device_seq_1));
    CUDA_CHECK(cudaFree(device_seq_2));
    CUDA_CHECK(cudaFree(device_ref));
    CUDA_CHECK(cudaFree(device_output));

    return 0;
}
