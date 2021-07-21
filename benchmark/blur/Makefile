include ../common/make.config 

# CUDA compiler 
NVCC = $(CUDA_DIR)/bin/nvcc 
NVCC_FLAGS = -I$(CUDA_DIR)/include 

blur: blur_cuda.o blur_kernel.o  
	$(NVCC) $(NVCC_FLAGS) blur_cuda.o blur_kernel.o -o blur -L$(CUDA_LIB_DIR) -lcuda -lcudart 

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(CFLAGS) $< -c 

clean:
	rm *.o blur 
