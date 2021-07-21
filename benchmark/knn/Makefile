include ../common/make.config 

# CUDA Compiler 
NVCC = $(CUDA_DIR)/bin/nvcc 
NVCC_FLAGS = -I$(CUDA_DIR)/include 

knn: knn_cuda.o knn_kernel.o
	$(NVCC) $(NVCC_FLAGS) knn_cuda.o knn_kernel.o -o knn -L$(CUDA_LIB_DIR) -lcuda -lcudart 

%.o: %.cu 
	$(NVCC) $(NVCC_FLAGS) $(CFLAGS) $< -c 

clean:
	rm *.o knn 
