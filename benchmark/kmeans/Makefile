include ../common/make.config 

# CUDA Compiler 
NVCC = $(CUDA_DIR)/bin/nvcc 
NVCC_FLAGS = -I$(CUDA_DIR)/include 

kmeans: kmeans_cuda.o kmeans_kernel.o
	$(NVCC) $(NVCC_FLAGS) kmeans_cuda.o kmeans_kernel.o -o kmeans -L$(CUDA_LIB_DIR) -lcuda -lcudart 

%.o: %.cu 
	$(NVCC) $(NVCC_FLAGS) $(CFLAGS) $< -c 

clean:
	rm *.o kmeans 
