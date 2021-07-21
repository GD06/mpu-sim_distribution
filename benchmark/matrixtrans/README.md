# Matrixtrans

Matrixtrans is a kernel transposing a matrix to generate the output matrix.
This directory contains the source code of matrix transposition GPU kernel. 
Moreover, this directory includes command-line interface (CLI) to run matrix transposition on both GPU and CPU.

## Running on GPU

We need to compile GPU kernels and the corresponding CPU interfaces by the following commands:

```
make clean
make
```

Then, the generated binary takes two position arguments to specify the number of rows and the number of columns of the input matrix.
For example, the binary can be directly excuted by the following command to run on the input matrix with the size 1024 x 1024:

```
./matrixtrans 1024 1024
```

We also have a wrapper script as the following command to run the matrix transposition on a large input matrix (1024 x 1024). 

```
./run.sh
```

## Running on MPU

To run the matrix transposition on MPU, we need to compile CUDA kernels into PTX and then compile PTX files to MPU executable programs. 
Then we have CLI to invoke the simulation. 

#### Step 1: Generate the PTX file

Running the following command with ```nvcc``` can generate the PTX file:

```
nvcc -O2 --ptx -o matrixtrans_kernel.ptx matrixtrans_kernel.cu 
```

#### Step 2: Generate MPU executable programs

Under the ```${PROJ_DIR}/program``` directory, we have CLI to compile kernels in the PTX file to MPU executable programs. 
Running the following command with generate the MPU executable programs:

```
../../program/ptx2exec_cli.py matrixtrans_kernel.ptx
```

#### Step 3: Run MPU simulation

Running the following command can invoke the simulation of matrix transposition on MPU:
```
./matrixtrans_sim.py matrixtrans_kernel_0.prog
```

There are several command line options in the script ```matrixtrans_sim.py``` to specify the matrix size and other parameters.
Running the following command can check the details of the command line options:

```
./matrixtrans_sim.py --help
```

#### Note: Collect performance metrics and hardware events

During the running of the simulation, our simulator will collect a comprehensive set of peformance metrics and hardware events. 
Runinng the simulation as the following command can collect the this information:

```
./matrixtrans_sim.py --output_perf_file tmp_perf.json --output_trace_file tmp_trace.json
```

The output information is stored in ```tmp_perf.json``` and ```tmp_trace.json```. 
The file ```tmp_trace.json``` can be imported by Chrome Tracing to visualize the timeline of the simulation.
