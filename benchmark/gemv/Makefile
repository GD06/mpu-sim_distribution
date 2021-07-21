include ../common/make.config 

# CUDA compiler 
NVCC = $(CUDA_DIR)/bin/nvcc 
NVCC_FLAGS = -I$(CUDA_DIR)/include 

gemv: gemv_cuda.o gemv_kernel.o  
	$(NVCC) $(NVCC_FLAGS) gemv_cuda.o gemv_kernel.o -o gemv -L$(CUDA_LIB_DIR) -lcuda -lcudart 

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(CFLAGS) $< -c 

clean:
	rm *.o gemv  
