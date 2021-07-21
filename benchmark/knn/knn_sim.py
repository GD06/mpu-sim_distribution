#!/usr/bin/env python3

import argparse 
import json 
import numpy as np 
import struct 
import pickle 

import config.config_api as config_api 
import simulator.sim_api as sim_api 


def _return_true(event):
    return True 


def _py_knn(x_array, y_array, point_x, point_y):
    result = [(x - point_x) * (x - point_x) + (y - point_y) * (y - point_y)
              for x, y in zip(x_array.tolist(), y_array.tolist())]

    return np.array(result).astype(np.float32)


def main():

    parser = argparse.ArgumentParser(
        description="Run the simulation for the distance kernel of kNN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter 
    )

    # Application dependent parameters 
    parser.add_argument("input_prog_file", help="Specify the path of "
                        "executable program")
    parser.add_argument("--num_points", type=int, default=16777216)

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

    ptr_x = hardware.mem.allocate(args.num_points * 4)
    ptr_y = hardware.mem.allocate(args.num_points * 4)
    ptr_dist = hardware.mem.allocate(args.num_points * 4)
    hardware.mem.finalize() 

    host_x = np.random.rand(args.num_points).astype(np.float32)
    host_y = np.random.rand(args.num_points).astype(np.float32)

    hardware.mem.set_value(ptr_x, host_x.tobytes())
    hardware.mem.set_value(ptr_y, host_y.tobytes())
    print("Hardware memory set-up: Success!") 
    
    # Start simulation
    print("Starting simulation...")
    total_cycles, sim_freq = hardware.run_simulation(
        kernel=opt_kernel,
        kernel_args=[ptr_x, ptr_y, ptr_dist, args.num_points, 0.5, 0.5],
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
    c_in_bytes = hardware.mem.get_value(ptr_dist, args.num_points * 4)
    c_in_float = struct.unpack("{}f".format(args.num_points), c_in_bytes)
    c = np.array(c_in_float).astype(np.float32)

    ground_truth = _py_knn(host_x, host_y, 0.5, 0.5)
    np.testing.assert_allclose(c, ground_truth, atol=1e-5)
    print("Correctness check: Success!")

    # Dump output files 
    if args.output_perf_file is not None:
        hardware.dump_perf_metrics(args.output_perf_file)

    if args.output_trace_file is not None:
        hardware.dump_timeline(args.output_trace_file)


if __name__ == "__main__":
    main()
