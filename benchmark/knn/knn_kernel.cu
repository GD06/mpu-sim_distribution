#ifndef _KNN_CUDA_KERNEL
#define _KNN_CUDA_KERNEL 

__global__ void kNN(float* x, float* y, float* dist, int N, 
        float point_x, float point_y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int numThreads = gridDim.x * blockDim.x;
    for (int i = index; i < N; i += numThreads) {
       float diff_x = x[i] - point_x;
       float diff_y = y[i] - point_y;
       dist[i] = (diff_x * diff_x + diff_y * diff_y);
    }
    return; 
}

#endif 
