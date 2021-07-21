#ifndef _VECTOR_ADD_CUDA_KERNEL
#define _VECTOR_ADD_CUDA_KERNEL 

__global__ void vectorAdd(float* a, float* b, float* c, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int numThreads = gridDim.x * blockDim.x;
    for (int i = index; i < N; i += numThreads) {
        c[i] = a[i] + b[i];
    }
    return;
}

#endif 
