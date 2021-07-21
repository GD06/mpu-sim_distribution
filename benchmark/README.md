# Benchmark

## Workload

This directory contains workloads we used to evaluate and compare the performnce of GPU and MPU. 
The details of each benchmark including the steps to run GPU kernels and MPU simulation can be found as follows:

- [blur](./blur/) (BLUR): performing a 3x3 gaussian blur on the input image.
- [conv](./conv/) (CONV): performing a 3x3 convolution on the input image. 
- [gemv](./gemv/) (GEMV): computing the matrix-vector multiplication to producee an output vector. 
- [histogram](./histogram/) (HIST): performing a histogram on a number of input elements. 
- [kmeans](./kmeans/) (KMEANS): computing which cluster each data point belongs to in the k-means algorithm.
- [knn](./knn/) (KNN): computing the distance between all data points the query point for the computing of knn (k-Nearest-Neighbor).
- [matrixtrans](./matrixtrans/) (TTRANS): transposing an input matrix to generate an output matrix. 
- [maxpool](./maxpool/) (MAXP): performing a 2x2 max-pooling on the input 2D tensor.
- [nw](./nw/) (NW): running the sequence alignment algorithm on a number of sequence pairs.
- [upsample](./upsample/) (UPSAMP): performing a 2x2 up-sampling on the input 2D tensor.
- [vectoradd](./vectoradd/) (AXPY): adding two input vectors in an element-wise manner to generate the output vector. 
- [vectorsum](./vectorsum/) (PR): summing up all elements from the input vector to produce an output scalar. 

## Profiling on GPU

### Performance Profiling

Inside the folder of each workload, there is a ```Makefile``` and a ```run.sh``` file. 
Please follow the steps presented in the README.md of each workload to first run the script ```run.sh``` successfully, then ```nvprof``` can be used to profil and collect perfomance metrics of CUDA kernels.

For example, running the following command can get the profiling executin time summary of CUDA kernels and CUDA APIs for the vectoradd workload:

```
cd vectoradd && nvprof --profile-child-processes ./run.sh
```

Another example is the command to collect some performance metrics for the vectoradd workload, including DRAM throughput and the number of executed floating-point and intger instructions:

```
cd vectoradd && nvprof --profile-child-processes \
    --metrics dram_read_throughput,dram_write_throughput,inst_fp_32,inst_integer ./run.sh
```

### Power Measurement

The power measurement is based on the tool ```nvidia-smi``` which is usually installed with CUDA driver. 

First, compile the workload with the option to enable the power measurement, which will make the CUDA kernel running forever.
For example, running the following commands to compile the vectoradd workload:

```
cd vectoradd/
make CFLAGS=-DMEASURE_POWER
```

Second, launch the execution of CUDA programs by running the script ```run.sh```:

```
./run.sh
```

Third, open another terminal on the same machine, and run the following command to get the measured power number displayed on the STDOUT:

```
nvidia-smi -i ${CUDA_VISIBLE_DEVICES} --loop-ms=50 --format=csv,noheader \
  --query-gpu=power.draw | ./scripts/gpu_power_display.py
```

NOTE: the environment variable ```CUDA_VISIBLE_DEVICES``` stands for the GPU ID your workload is running.
Please change that into the actual GPU ID of the GPU used by your workload in the second step. 

Finally, when the power number displayed in the third step is stablized, you need to manually stop both processes launched in the second and the third step. 
The stable output power number in the third step is the maximal power results during the execution of the kernel. 
