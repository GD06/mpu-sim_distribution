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


_NUM_FEATURES = 4
_NUM_CLUSTERS = 4


def _py_kmeans(np_points, np_centers):
    result_list = []
    num_points, _ = np_points.shape 
    num_centers, _ = np_centers.shape 
    for i in range(num_points):
        dist_array = [np.linalg.norm(np_points[i, :] - np_centers[j, :]) 
                      for j in range(num_centers)]
        result_list.append(np.argmin(dist_array))
    return np.array(result_list).astype(np.float32)


def main():

    parser = argparse.ArgumentParser(
        description="Run the simulation for kmeans",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter 
    )

    # Application dependent parameters 
    parser.add_argument("input_prog_file", help="Specify the path of "
                        "executable program")
    parser.add_argument("--num_points", type=int, default=262144)

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

    num_points = args.num_points 

    ptr_points = hardware.mem.allocate(num_points * _NUM_FEATURES * 4)
    ptr_centers = hardware.mem.allocate(_NUM_CLUSTERS * _NUM_FEATURES * 4)
    ptr_membership = hardware.mem.allocate(num_points * 4)
    hardware.mem.finalize() 

    np_points = np.random.rand(
        num_points * _NUM_FEATURES).astype(np.float32)
    np_centers = np.random.rand(
        _NUM_CLUSTERS * _NUM_FEATURES).astype(np.float32)

    hardware.mem.set_value(ptr_points, np_points.tobytes())
    hardware.mem.set_value(ptr_centers, np_centers.tobytes())
    print("Hardware memory set-up: Success!") 
    
    # Start simulation
    print("Starting simulation...")
    total_cycles, sim_freq = hardware.run_simulation(
        kernel=opt_kernel, 
        kernel_args=[ptr_points, ptr_centers, ptr_membership, num_points],
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
    output_buffer = hardware.mem.get_value(ptr_membership, num_points * 4)
    sim_results = np.array(
        struct.unpack("{}i".format(num_points), output_buffer)
    ).astype(np.float32)
    
    ground_truth = _py_kmeans(
        np.transpose(np_points.reshape((_NUM_FEATURES, num_points))),
        np.transpose(np_centers.reshape((_NUM_FEATURES, _NUM_CLUSTERS))),
    )

    ground_truth_sum = float(np.sum(ground_truth))
    sim_results_sum = float(np.sum(sim_results))
    np.testing.assert_allclose(
        [ground_truth_sum], [sim_results_sum], rtol=1e-3)
    print("Correctness check: Success!")

    # Dump output files 
    if args.output_perf_file is not None:
        hardware.dump_perf_metrics(args.output_perf_file)

    if args.output_trace_file is not None:
        hardware.dump_timeline(args.output_trace_file)


if __name__ == "__main__":
    main()
