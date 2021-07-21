include ../common/make.config 

# CUDA compiler 
NVCC = $(CUDA_DIR)/bin/nvcc 
NVCC_FLAGS = -I$(CUDA_DIR)/include 

vectoradd: vectoradd_cuda.o vectoradd_kernel.o  
	$(NVCC) $(NVCC_FLAGS) vectoradd_cuda.o vectoradd_kernel.o -o vectoradd -L$(CUDA_LIB_DIR) -lcuda -lcudart 

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(CFLAGS) $< -c 

clean:
	rm *.o vectoradd 
