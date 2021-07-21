include ../common/make.config 

# CUDA compiler 
NVCC = $(CUDA_DIR)/bin/nvcc 
NVCC_FLAGS = -I$(CUDA_DIR)/include 

upsample: upsample_cuda.o upsample_kernel.o  
	$(NVCC) $(NVCC_FLAGS) upsample_cuda.o upsample_kernel.o -o upsample -L$(CUDA_LIB_DIR) -lcuda -lcudart 

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(CFLAGS) $< -c 

clean:
	rm *.o upsample 
