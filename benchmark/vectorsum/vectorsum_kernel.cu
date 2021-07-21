#ifndef _VECTOR_SUM_CUDA_KERNEL
#define _VECTOR_SUM_CUDA_KERNEL 

#include "cta_config.h"


__global__ void vectorSum(float* a, float* tmp, int N) {
    int bid = blockIdx.x;
    int tid = threadIdx.x; 
    int index = bid * blockDim.x + tid;
    int numThreads = gridDim.x * blockDim.x;

    __shared__ float psum[NUM_THREADS];

    float sum = 0.0f;
    for (int i = index; i < N; i += numThreads) {
        sum += a[i];
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
        tmp[bid] = psum[0];
    } 

    return; 
}

#endif 
