# Max-pooling

This directory contains the implementation of performing a 2x2 max-pooling on the input 2D tensor.
In addition to CUDA kernels, this directory includes command-line interface (CLI) to run the max-pooling on both GPU and CPU.

## Running on GPU

We need to compile GPU kernels and the corresponding CPU interfaces by the following commands:

```
make clean
make
```

Then, the generated binary takes two position arguments to specify then number of rows and the number of columns of the output tensor.
By specifying the size of the output tensor, the size of input tensor is assumed as twice large as the output tensor in both dimensions. 
For example, the binary can be directly excuted by the following command to run the max-pooling with a 1024x1024 output tensor:

```
./maxpool 1024 1024
```

We also have a wrapper script as the following command to run the max-pooling on a large output tensor size (default value is 1024x1024). 

```
./run.sh
```

## Running on MPU

To run max-pooling on MPU, we need to compile CUDA kernels into PTX and then compile PTX files to MPU executable programs. 
Then we have CLI to invoke the simulation. 

#### Step 1: Generate the PTX file

Running the following command with ```nvcc``` can generate the PTX file:

```
nvcc -O2 --ptx -o maxpool_kernel.ptx maxpool_kernel.cu 
```

#### Step 2: Generate MPU executable programs

Under the ```${PROJ_DIR}/program``` directory, we have CLI to compile kernels in the PTX file to MPU executable programs. 
Running the following command with generate the MPU executable programs:

```
../../program/ptx2exec_cli.py maxpool_kernel.ptx
```

#### Step 3: Run MPU simulation

Running the following command can invoke the simulation of the max-pooling on MPU:
```
./maxpool_sim.py maxpool_kernel_0.prog
```

There are several command line options in the script ```maxpool_sim.py``` to specify the size of the output tensor and other parameters.
Running the following command can check the details of the command line options:

```
./maxpool_sim.py --help
```

#### Note: Collect performance metrics and hardware events

During the running of the simulation, our simulator will collect a comprehensive set of peformance metrics and hardware events. 
Runinng the simulation as the following command can collect the this information:

```
./maxpool_sim.py --output_perf_file tmp_perf.json --output_trace_file tmp_trace.json
```

The output information is stored in ```tmp_perf.json``` and ```tmp_trace.json```. 
The file ```tmp_trace.json``` can be imported by Chrome Tracing to visualize the timeline of the simulation.
