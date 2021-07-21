include ../common/make.config 

# CUDA compiler 
NVCC = $(CUDA_DIR)/bin/nvcc 
NVCC_FLAGS = -I$(CUDA_DIR)/include 

maxpool: maxpool_cuda.o maxpool_kernel.o  
	$(NVCC) $(NVCC_FLAGS) maxpool_cuda.o maxpool_kernel.o -o maxpool -L$(CUDA_LIB_DIR) -lcuda -lcudart 

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(CFLAGS) $< -c 

clean:
	rm *.o maxpool 
