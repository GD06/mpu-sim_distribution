#!/usr/bin/env python3

import argparse 
import json 
import numpy as np 
import struct 
import pickle 

import config.config_api as config_api 
import simulator.sim_api as sim_api 

NUM_THREADS = 128


def _return_true(event):
    return True 


def _py_maxpool(host_input, num_output_rows, num_output_cols):
    tmp_input = np.zeros((num_output_rows * 2, num_output_cols * 2))
    for i in range(0, num_output_rows * 2, 2):
        for j in range(0, num_output_cols * 2, NUM_THREADS * 2):
            for kx in range(2):
                for ky in range(2):
                    for k in range(NUM_THREADS): 
                        tmp_input[i + kx, j + ky * NUM_THREADS + k] = \
                            host_input.flat[
                                (kx * 2 + ky) * num_output_rows 
                                * num_output_cols + (i // 2) * num_output_cols
                                + (j // 2) + k]

    results = []
    for i in range(num_output_rows):
        results_row = []
        for j in range(num_output_cols):
            val_array = [
                tmp_input[i * 2, j * 2], tmp_input[i * 2, j * 2 + 1],
                tmp_input[i * 2 + 1, j * 2], tmp_input[i * 2 + 1, j * 2 + 1]
            ]
            results_row.append(np.max(val_array))
        results.append(results_row)
    
    np_results = np.array(results).astype(np.float32).reshape(
        (num_output_rows, num_output_cols))
    return np_results 


def main():

    parser = argparse.ArgumentParser(
        description="Run the simulation for the 2x2 maxpooling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter 
    )

    # Application dependent parameters 
    parser.add_argument("input_prog_file", help="Specify the path of "
                        "executable program")
    parser.add_argument("--num_output_rows", type=int, default=1024)
    parser.add_argument("--num_output_cols", type=int, default=1024)

    # Hardware dependent parameters 
    parser.add_argument("--hardware_config", "-c", default=None, 
                        help="Specify the path of hardware config")

    # Runtime parameters
    parser.add_argument("--num_threads", "-t", type=int, default=128)
    parser.add_argument("--num_blocks", "-b", type=int, default=512)
    parser.add_argument("--mapping_method", "-m", default=None,
                        help="Specify the config of mapping blocks to "
                        "hardware vaults")

    # Output files 
    parser.add_argument("--output_perf_file", default=None,
                        help="The output file storing performance metrics")
    parser.add_argument("--output_trace_file", default=None,
                        help="The output file storing trace events")

    args = parser.parse_args()
    assert args.num_threads == NUM_THREADS, "The current CUDA kernel " \
        "implementation has an assumption on the number of threads per block"

    num_output_rows = args.num_output_rows 
    num_output_cols = args.num_output_cols 
    num_input_rows = num_output_rows * 2
    num_input_cols = num_output_cols * 2

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
    with open(args.input_prog_file, "rb") as f:
        opt_kernel = pickle.load(f)

    # Init MPU hardware and allocate memory
    if args.output_trace_file is None:
        hardware = sim_api.init_hardware(hw_config_dict)
    else:
        hardware = sim_api.init_hardware(
            hw_config_dict, filter_func=_return_true)
    print("Hardware initialization: Success!")

    ptr_input = hardware.mem.allocate(num_input_rows * num_input_cols * 4)
    ptr_output = hardware.mem.allocate(num_output_rows * num_output_cols * 4)
    hardware.mem.finalize() 

    host_input = np.random.rand(
        num_input_rows * num_input_cols).astype(np.float32)
    hardware.mem.set_value(ptr_input, host_input.tobytes())
    print("Hardware memory set-up: Success!") 
    
    # Start simulation
    print("Starting simulation...")
    total_cycles, sim_freq = hardware.run_simulation(
        kernel=opt_kernel, 
        kernel_args=[ptr_input, ptr_output, num_output_rows, num_output_cols],
        grid_dim=(1, 1, args.num_blocks),
        block_dim=(1, 1, args.num_threads),
        block_schedule=mapping_dict, 
    )

    print(
        "Simulation finished: {} cycles at {} MHz".format(
            total_cycles, sim_freq)
    )
    print("Total time: {} us".format(
        total_cycles / sim_freq)
    )

    # Compare results after simulation
    output_buffer = hardware.mem.get_value(
        ptr_output, num_output_rows * num_output_cols * 4) 
    sim_results = np.array(
        struct.unpack("{}f".format(num_output_rows * num_output_cols),
                      output_buffer)
    ).astype(np.float32).reshape((num_output_rows, num_output_cols))

    ground_truth = _py_maxpool(
        host_input.reshape((num_input_rows, num_input_cols)),
        num_output_rows, num_output_cols 
    )

    np.testing.assert_allclose(sim_results, ground_truth, atol=1e-5)
    print("Correctness check: Success!")

    # Dump output files 
    if args.output_perf_file is not None:
        hardware.dump_perf_metrics(args.output_perf_file)

    if args.output_trace_file is not None:
        hardware.dump_timeline(args.output_trace_file)


if __name__ == "__main__":
    main()
