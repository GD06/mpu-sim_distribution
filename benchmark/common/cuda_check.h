#ifndef CUDA_CHECK 
#define CUDA_CHECK(x) { \
    cudaError_t result = x; \
    if (result != cudaSuccess) { \
        printf("CUDA Failure %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(result)); \
        exit(0); \
    } \
}
#endif 
