#ifndef _NW_CUDA_KERNEL 
#define _NW_CUDA_KERNEL 

#include "cta_config.h"

__global__ void seqAlign(int* input_seq_1, int* input_seq_2, int* input_ref, 
        int* output_matrix, int num_pairs, int penalty) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ int ref[SEQ_LEN][SEQ_LEN];
    __shared__ int tmp[SEQ_LEN + 1][SEQ_LEN + 1];

    for (int seq_id = bid; seq_id < num_pairs; seq_id += gridDim.x) {
        if (tid == 0) {
            tmp[0][0] = 0;
        }
        tmp[0][tid + 1] = input_seq_1[seq_id * SEQ_LEN + tid];
        tmp[tid + 1][0] = input_seq_2[seq_id * SEQ_LEN + tid];
        __syncthreads();

        for (int ty = 0; ty < SEQ_LEN; ty++) {
            ref[ty][tid] = input_ref[ty * num_pairs * SEQ_LEN + seq_id * SEQ_LEN + tid];
        }
        __syncthreads();

        for (int m = 0; m < SEQ_LEN; ++m) {

            if (tid <= m) {
                int t_index_x = tid + 1;
                int t_index_y = m - tid + 1;
                int max_val = (tmp[t_index_y - 1][t_index_x -1] 
                        + ref[t_index_y - 1][t_index_x - 1]);
                int tmp_val;
                
                tmp_val = tmp[t_index_y][t_index_x - 1] - penalty;
                if (tmp_val > max_val) { max_val = tmp_val; }

                tmp_val = tmp[t_index_y - 1][t_index_x] - penalty;
                if (tmp_val > max_val) { max_val = tmp_val; }

                tmp[t_index_y][t_index_x] = max_val;
            }
            __syncthreads();

        }

        for (int m = SEQ_LEN - 2; m >= 0; m--) {

            if (tid <= m) {
                int t_index_x = tid + SEQ_LEN - m;
                int t_index_y = SEQ_LEN - tid;
                int max_val = (tmp[t_index_y - 1][t_index_x -1] 
                        + ref[t_index_y - 1][t_index_x - 1]);
                int tmp_val;
                
                tmp_val = tmp[t_index_y][t_index_x - 1] - penalty;
                if (tmp_val > max_val) { max_val = tmp_val; }

                tmp_val = tmp[t_index_y - 1][t_index_x] - penalty;
                if (tmp_val > max_val) { max_val = tmp_val; }

                tmp[t_index_y][t_index_x] = max_val;
            }
            __syncthreads();

        }

        for (int ty = 0; ty < SEQ_LEN; ++ty) {
            output_matrix[ty * num_pairs * SEQ_LEN + seq_id * SEQ_LEN + tid] = (
                    tmp[ty + 1][tid + 1]);
        }
        __syncthreads();
    }

    return; 
}

#endif 
