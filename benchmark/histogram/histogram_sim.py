#!/usr/bin/env python3

import argparse 
import json 
import numpy as np 
import struct 
import pickle 

import config.config_api as config_api 
import simulator.sim_api as sim_api 

NUM_BINS = 256


def _py_histogram(host_input):
    results = [0] * NUM_BINS
    for i in range(len(host_input)):
        val = min(max(host_input[i] * 255.0, 0), NUM_BINS - 1)
        index = int(val)
        results[index] += 1
    return np.array(results).astype(np.float32)


def _return_true(event):
    return True 


def main():

    parser = argparse.ArgumentParser(
        description="Run the simulation for the histogram",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter 
    )

    # Application dependent parameters 
    parser.add_argument("hist_prog_file", help="Specify the path of the "
                        "executable program producing partial results.")
    parser.add_argument("reduce_prog_file", help="Specify the path of the "
                        "executable program reducing all partial results.")
    parser.add_argument("--num_elements", type=int, default=16777216)

    # Hardware dependent parameters 
    parser.add_argument("--hardware_config", "-c", default=None, 
                        help="Specify the path of hardware config")

    # Runtime parameters
    parser.add_argument("--num_threads", "-t", type=int, default=128)
    parser.add_argument("--num_blocks", "-b", type=int, default=128)
    parser.add_argument("--mapping_method", "-m", default=None,
                        help="Specify the config of mapping blocks to "
                        "hardware vaults")

    # Output files 
    parser.add_argument("--output_perf_file", default=None,
                        help="The output file storing performance metrics")
    parser.add_argument("--output_trace_file", default=None,
                        help="The output file storing trace events")

    args = parser.parse_args() 
    length = args.num_elements 

    # Load hardware configuration from file:
    hw_config_dict = config_api.load_hardware_config(
        overwrite_config_file_path=args.hardware_config
    )
    hw_config_dict["display_simulation_progress"] = True 

    # Load block mapping from file 
    if args.mapping_method is None:
        mapping_dict = {}
        print("Total number of cores:", hw_config_dict["total_num_cores"])
        for i in range(args.num_blocks):
            # mapping_dict[i] = 0
            mapping_dict[i] = i % hw_config_dict["total_num_cores"]
    else:
        with open(args.mapping_method, "r") as f:
            raw_dict = json.load(f)
            mapping_dict = {}
            for key in raw_dict:
                mapping_dict[int(key)] = raw_dict[key]

    # Load kernel from executable program file  
    with open(args.hist_prog_file, "rb") as f:
        hist_kernel = pickle.load(f)

    with open(args.reduce_prog_file, "rb") as f:
        reduce_kernel = pickle.load(f)

    # Init MPU hardware and allocate memory
    if args.output_trace_file is None:
        hardware = sim_api.init_hardware(hw_config_dict)
    else:
        hardware = sim_api.init_hardware(
            hw_config_dict, filter_func=_return_true)
    print("Hardware initialization: Success!")

    ptr_input = hardware.mem.allocate(length * 4)
    tmp_result = hardware.mem.allocate(args.num_blocks * NUM_BINS * 4)
    output_result = hardware.mem.allocate(NUM_BINS * 4)
    hardware.mem.finalize() 

    host_input = np.random.rand(length).astype(np.float32)
    hardware.mem.set_value(ptr_input, host_input.tobytes())
    print("Hardware memory set-up: Success!") 
    
    # Start simulation
    print("Starting simulation...")

    hist_cycles, sim_freq = hardware.run_simulation(
        kernel=hist_kernel,
        kernel_args=[ptr_input, tmp_result, length],
        grid_dim=(1, 1, args.num_blocks),
        block_dim=(1, 1, args.num_threads),
        block_schedule=mapping_dict, 
    )
    print(
        "Histogram kernel finished: {} cycles at {} MHz".format(
            hist_cycles, sim_freq)
    )

    reduce_cycles, sim_freq = hardware.run_simulation(
        kernel=reduce_kernel,
        kernel_args=[tmp_result, output_result, args.num_blocks],
        grid_dim=(1, 1, 2),
        block_dim=(1, 1, 128),
        block_schedule={0: 0, 1: 1},
    )
    print(
        "Reduction kernel finished: {} cycles at {} MHz".format(
            reduce_cycles, sim_freq)
    )

    total_cycles = hist_cycles + reduce_cycles 
    print(
        "Simulation finished: {} cycles at {} MHz".format(
            total_cycles, sim_freq)
    )
    print("Total time: {} us".format(
        total_cycles / sim_freq)
    )

    # Compare results after simulation 
    c_in_bytes = hardware.mem.get_value(output_result, NUM_BINS * 4)
    c_in_int = struct.unpack("{}i".format(NUM_BINS), c_in_bytes)
    c = np.array(c_in_int).astype(np.float32)

    ground_truth = _py_histogram(host_input)
    np.testing.assert_allclose(c, ground_truth, atol=1e-5)
    print("Correctness check: Success!")

    # Dump output files 
    if args.output_perf_file is not None:
        hardware.dump_perf_metrics(args.output_perf_file)

    if args.output_trace_file is not None:
        hardware.dump_timeline(args.output_trace_file)


if __name__ == "__main__":
    main()
