include ../common/make.config 

# CUDA compiler 
NVCC = $(CUDA_DIR)/bin/nvcc 
NVCC_FLAGS = -I$(CUDA_DIR)/include 

matrixtrans: matrixtrans_cuda.o matrixtrans_kernel.o  
	$(NVCC) $(NVCC_FLAGS) matrixtrans_cuda.o matrixtrans_kernel.o -o matrixtrans -L$(CUDA_LIB_DIR) -lcuda -lcudart 

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(CFLAGS) $< -c 

clean:
	rm *.o matrixtrans  
