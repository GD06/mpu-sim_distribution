#!/usr/bin/env python3

import argparse 
import json 
import numpy as np 
import struct 
import pickle 

import config.config_api as config_api 
import simulator.sim_api as sim_api 

NUM_THREADS = 128


def _relayout_matrix(input_matrix, num_rows, num_cols):
    output_matrix = np.zeros((num_rows, num_cols))
    for i in range(num_rows):
        for j in range(0, num_cols, NUM_THREADS):
            for k in range(NUM_THREADS):
                output_matrix[i, j + k] = input_matrix.flat[
                    (j // NUM_THREADS) * NUM_THREADS * num_rows 
                    + i * NUM_THREADS + k]

    return output_matrix.astype(np.float32)


def _return_true(event):
    return True 


def main():

    parser = argparse.ArgumentParser(
        description="Run the simulation for matrix-vector multiplication",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter 
    )

    # Application dependent parameters 
    parser.add_argument("input_prog_file", help="Specify the path of "
                        "executable program")
    parser.add_argument("--num_rows", type=int, default=4096)
    parser.add_argument("--num_cols", type=int, default=4096)

    # Hardware dependent parameters 
    parser.add_argument("--hardware_config", "-c", default=None, 
                        help="Specify the path of hardware config")

    # Runtime parameters
    parser.add_argument("--num_threads", type=int, default=128)
    parser.add_argument("--num_blocks", type=int, default=512)
    parser.add_argument("--mapping_method", "-m", default=None,
                        help="Specify the config of mapping blocks to "
                        "hardware vaults")

    # Output files 
    parser.add_argument("--output_perf_file", default=None,
                        help="The output file storing performance metrics")
    parser.add_argument("--output_trace_file", default=None,
                        help="The output file storing trace events")

    args = parser.parse_args()
    assert args.num_threads == NUM_THREADS, "The current kernel " \
        "implementation has an assumption on the number of threads per block"

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

    ptr_matrix = hardware.mem.allocate(args.num_rows * args.num_cols * 4)
    ptr_invec = hardware.mem.allocate(args.num_cols * 4)
    ptr_outvec = hardware.mem.allocate(args.num_rows * 4)
    hardware.mem.finalize() 

    input_matrix = np.random.rand(
        args.num_rows * args.num_cols).astype(np.float32)
    input_vector = np.random.rand(args.num_cols).astype(np.float32)
    hardware.mem.set_value(ptr_matrix, input_matrix.tobytes())
    hardware.mem.set_value(ptr_invec, input_vector.tobytes())
    print("Hardware memory set-up: Success!") 
    
    # Start simulation
    print("Starting simulation...")
    total_cycles, sim_freq = hardware.run_simulation(
        kernel=opt_kernel,
        kernel_args=[ptr_matrix, ptr_invec, ptr_outvec, 
                     args.num_rows, args.num_cols],
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
    output_buffer = hardware.mem.get_value(ptr_outvec, args.num_rows * 4)
    sim_results = np.array(
        struct.unpack(
            "{}f".format(args.num_rows), output_buffer)
    ).astype(np.float32)

    tmp_matrix = _relayout_matrix(input_matrix, args.num_rows, args.num_cols)
    ground_truth = tmp_matrix.dot(input_vector)
    np.testing.assert_allclose(sim_results, ground_truth, atol=1e-3)
    print("Correctness check: Success!")

    # Dump output files 
    if args.output_perf_file is not None:
        hardware.dump_perf_metrics(args.output_perf_file)

    if args.output_trace_file is not None:
        hardware.dump_timeline(args.output_trace_file)


if __name__ == "__main__":
    main()
