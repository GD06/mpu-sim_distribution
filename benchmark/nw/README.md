# Needleman-Wunsch (NW)

This directory contains the implementation of sequence aligment based on the NW algorithm for a number of sequence pairs.
In addition to CUDA kernels, this directory includes command-line interface (CLI) to run the NW on both GPU and CPU.

## Running on GPU

We need to compile GPU kernels and the corresponding CPU interfaces by the following commands:

```
make clean
make
```

Then, the generated binary takes one position argument to specify the number of sequence pairs. 
For example, the binary can be directly excuted by the following command to run the NW with 2048 sequence pairs:

```
./nw 2048
```

We also have a wrapper script as the following command to run the NW on a large number of sequence pairs (default value is 4096). 

```
./run.sh
```

## Running on MPU

To run NW on MPU, we need to compile CUDA kernels into PTX and then compile PTX files to MPU executable programs. 
Then we have CLI to invoke the simulation. 

#### Step 1: Generate the PTX file

Running the following command with ```nvcc``` can generate the PTX file:

```
nvcc -O2 --ptx -o nw_kernel.ptx nw_kernel.cu 
```

#### Step 2: Generate MPU executable programs

Under the ```${PROJ_DIR}/program``` directory, we have CLI to compile kernels in the PTX file to MPU executable programs. 
Running the following command with generate the MPU executable programs:

```
../../program/ptx2exec_cli.py nw_kernel.ptx
```

#### Step 3: Run MPU simulation

Running the following command can invoke the simulation of the NW on MPU:
```
./nw_sim.py nw_kernel_0.prog
```

There are several command line options in the script ```nw_sim.py``` to specify the number of sequence pairs and other parameters.
Running the following command can check the details of the command line options:

```
./nw_sim.py --help
```

#### Note: Collect performance metrics and hardware events

During the running of the simulation, our simulator will collect a comprehensive set of peformance metrics and hardware events. 
Runinng the simulation as the following command can collect the this information:

```
./nw_sim.py --output_perf_file tmp_perf.json --output_trace_file tmp_trace.json
```

The output information is stored in ```tmp_perf.json``` and ```tmp_trace.json```. 
The file ```tmp_trace.json``` can be imported by Chrome Tracing to visualize the timeline of the simulation.
