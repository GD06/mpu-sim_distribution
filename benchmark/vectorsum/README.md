# Vectorsum

Vectorsum is a kernel summing all elements from the input vector to generate an output scalar.
This directory contains the source code of vectorsum GPU kernel. 
Moreover, this directory includes command-line interface (CLI) to run vectorsum on both GPU and CPU.

## Running on GPU

We need to compile GPU kernels and the corresponding CPU interfaces by the following commands:

```
make clean
make
```

Then, the generated binary takes a position argument to specify the vector length to run. 
For example, the binary can be directly excuted by the following command to run on input vectors with 1024 elements:

```
./vectorsum 1024
```

We also have a wrapper script as the following command to run the vectorsum on a long vector length (default value is 16777216). 

```
./run.sh
```

## Running on MPU

To run vectorsum on MPU, we need to compile CUDA kernels into PTX and then compile PTX files to MPU executable programs. 
Then we have CLI to invoke the simulation. 

#### Step 1: Generate the PTX file

Running the following command with ```nvcc``` can generate the PTX file:

```
nvcc -O2 --ptx -o vectorsum_kernel.ptx vectorsum_kernel.cu 
```

#### Step 2: Generate MPU executable programs

Under the ```${PROJ_DIR}/program``` directory, we have CLI to compile kernels in the PTX file to MPU executable programs. 
Running the following command with generate the MPU executable programs:

```
../../program/ptx2exec_cli.py vectorsum_kernel.ptx
```

#### Step 3: Run MPU simulation

Running the following command can invoke the simulation of vectorsum on MPU:
```
./vectorsum_sim.py vectorsum_kernel_0.prog
```

There are several command line options in the script ```vectorsum_sim.py``` to specify the vector length and other parameters.
Running the following command can check the details of the command line options:

```
./vectorsum_sim.py --help
```

#### Note: Collect performance metrics and hardware events

During the running of the simulation, our simulator will collect a comprehensive set of peformance metrics and hardware events. 
Runinng the simulation as the following command can collect the this information:

```
./vectorsum_sim.py --output_perf_file tmp_perf.json --output_trace_file tmp_trace.json
```

The output information is stored in ```tmp_perf.json``` and ```tmp_trace.json```. 
The file ```tmp_trace.json``` can be imported by Chrome Tracing to visualize the timeline of the simulation.
