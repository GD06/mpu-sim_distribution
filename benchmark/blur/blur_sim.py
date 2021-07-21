#!/usr/bin/env python3

import argparse 
import json 
import numpy as np 
import struct 
import pickle 

import config.config_api as config_api 
import simulator.sim_api as sim_api 

NUM_THREADS_X = 32
NUM_THREADS_Y = 4
BLOCK_SIZE = 32


def _py_blur(host_input, num_rows, num_cols):
    output_matrix = np.zeros((num_rows * num_cols))
    num_blocks = (num_rows * num_cols) // (BLOCK_SIZE * BLOCK_SIZE)
    for bid in range(num_blocks):
        for i in range(BLOCK_SIZE):
            for j in range(BLOCK_SIZE):
                sum_val = 0.0
                for kx in range(3):
                    for ky in range(3):
                        row_id = min(max(i + kx - 1, 0), BLOCK_SIZE - 1)
                        col_id = min(max(j + ky - 1, 0), BLOCK_SIZE - 1)
                        slice_id = row_id // NUM_THREADS_Y 
                        row_id = row_id % NUM_THREADS_Y 
                        sum_val += host_input[
                            slice_id * NUM_THREADS_X * NUM_THREADS_Y 
                            * num_blocks + bid * NUM_THREADS_X * NUM_THREADS_Y
                            + row_id * NUM_THREADS_X + col_id]

                sum_val = sum_val * (1.0 / 9.0)
                slice_id = i // NUM_THREADS_Y 
                row_id = i % NUM_THREADS_Y 
                output_matrix[
                    slice_id * NUM_THREADS_X * NUM_THREADS_Y * num_blocks
                    + bid * NUM_THREADS_X * NUM_THREADS_Y 
                    + row_id * NUM_THREADS_X + j] = sum_val 

    return output_matrix.astype(np.float32)


def _return_true(event):
    return True 


def main():

    parser = argparse.ArgumentParser(
        description="Run the simulation for the gaussian blur",
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
    parser.add_argument("--num_threads_x", type=int, default=32)
    parser.add_argument("--num_threads_y", type=int, default=4)
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
    assert (args.num_threads_x == NUM_THREADS_X
            and args.num_threads_y == NUM_THREADS_Y)

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

    ptr_input = hardware.mem.allocate(args.num_rows * args.num_cols * 4)
    ptr_output = hardware.mem.allocate(args.num_rows * args.num_cols * 4)
    hardware.mem.finalize() 

    input_image = np.random.rand(
        args.num_rows * args.num_cols).astype(np.float32)
    hardware.mem.set_value(ptr_input, input_image.tobytes())
    print("Hardware memory set-up: Success!") 
    
    # Start simulation
    print("Starting simulation...")
    total_cycles, sim_freq = hardware.run_simulation(
        kernel=opt_kernel, 
        kernel_args=[ptr_input, ptr_output, args.num_rows, 
                     args.num_cols, (1.0 / 9.0)],
        grid_dim=(1, 1, args.num_blocks),
        block_dim=(1, args.num_threads_y, args.num_threads_x),
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
        ptr_output, args.num_rows * args.num_cols * 4)
    sim_results = np.array(
        struct.unpack(
            "{}f".format(args.num_rows * args.num_cols), output_buffer)
    ).astype(np.float32)

    ground_truth = _py_blur(input_image, args.num_rows, args.num_cols)
    np.testing.assert_allclose(sim_results, ground_truth, atol=1e-5)
    print("Correctness check: Success!")

    # Dump output files 
    if args.output_perf_file is not None:
        hardware.dump_perf_metrics(args.output_perf_file)

    if args.output_trace_file is not None:
        hardware.dump_timeline(args.output_trace_file)


if __name__ == "__main__":
    main()
