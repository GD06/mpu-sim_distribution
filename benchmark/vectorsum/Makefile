include ../common/make.config 

# CUDA Compiler 
NVCC = $(CUDA_DIR)/bin/nvcc 
NVCC_FLAGS = -I$(CUDA_DIR)/include 

vectorsum: vectorsum_cuda.o vectorsum_kernel.o
	$(NVCC) $(NVCC_FLAGS) vectorsum_cuda.o vectorsum_kernel.o -o vectorsum -L$(CUDA_LIB_DIR) -lcuda -lcudart 

%.o: %.cu 
	$(NVCC) $(NVCC_FLAGS) $(CFLAGS) $< -c 

clean:
	rm *.o vectorsum 
