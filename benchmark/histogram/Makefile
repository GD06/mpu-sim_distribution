include ../common/make.config 

# CUDA compiler 
NVCC = $(CUDA_DIR)/bin/nvcc 
NVCC_FLAGS = -I$(CUDA_DIR)/include 

histogram: histogram_cuda.o histogram_kernel.o  
	$(NVCC) $(NVCC_FLAGS) histogram_cuda.o histogram_kernel.o -o histogram -L$(CUDA_LIB_DIR) -lcuda -lcudart 

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(CFLAGS) $< -c 

clean:
	rm *.o histogram 
