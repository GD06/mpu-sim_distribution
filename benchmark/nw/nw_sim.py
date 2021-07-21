#!/usr/bin/env python3

import argparse 
import json 
import numpy as np 
import struct 
import pickle 

import config.config_api as config_api 
import simulator.sim_api as sim_api 

SEQ_LEN = 32


def _py_nw(host_seq_1, host_seq_2, host_ref, num_pairs, penalty):
    host_seq_1 = host_seq_1.reshape((num_pairs, SEQ_LEN))
    host_seq_2 = host_seq_2.reshape((num_pairs, SEQ_LEN))
    host_ref = host_ref.reshape((SEQ_LEN, num_pairs, SEQ_LEN))
    host_output = [0] * (SEQ_LEN * num_pairs * SEQ_LEN)

    for seq_id in range(num_pairs):
        tmp = [[0] * (SEQ_LEN + 1) for _ in range(SEQ_LEN + 1)]
        for x in range(SEQ_LEN):
            tmp[0][x + 1] = host_seq_1[seq_id, x]
            tmp[x + 1][0] = host_seq_2[seq_id, x]

        for y in range(SEQ_LEN):
            for x in range(SEQ_LEN):
                max_val = tmp[y][x] + host_ref[y, seq_id, x]
                max_val = max(max_val, tmp[y][x + 1] - penalty)
                max_val = max(max_val, tmp[y + 1][x] - penalty)
                tmp[y + 1][x + 1] = max_val 
                host_output[y * num_pairs * SEQ_LEN 
                            + seq_id * SEQ_LEN + x] = max_val

    return np.array(host_output).astype(np.float32)


def _get_random_int_array(length):
    fp_array = np.random.rand(length) * 10.0 
    result_array = np.add(fp_array, np.array([-5.0] * length))
    return result_array.astype(np.int32)


def _return_true(event):
    return True 


def main():

    parser = argparse.ArgumentParser(
        description="Run the simulation for the kernel of NW algorithm",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter 
    )

    # Application dependent parameters 
    parser.add_argument("input_prog_file", help="Specify the path of "
                        "executable program")
    parser.add_argument("--num_pairs", type=int, default=4096)

    # Hardware dependent parameters 
    parser.add_argument("--hardware_config", "-c", default=None, 
                        help="Specify the path of hardware config")

    # Runtime parameters
    parser.add_argument("--num_threads", "-t", type=int, default=32)
    parser.add_argument("--num_blocks", "-b", type=int, default=2048)
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
            mapping_dict[i] = ((i // hw_config_dict["num_subcore"]) 
                               % hw_config_dict["total_num_cores"])
    else:
        with open(args.mapping_method) as f:
            raw_dict = json.load(f)
            mapping_dict = {}
            for k in raw_dict.keys():
                mapping_dict[int(k)] = raw_dict[k]

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

    ptr_seq_1 = hardware.mem.allocate(args.num_pairs * SEQ_LEN * 4)
    ptr_seq_2 = hardware.mem.allocate(args.num_pairs * SEQ_LEN * 4)
    ptr_ref = hardware.mem.allocate(args.num_pairs * SEQ_LEN * SEQ_LEN * 4)
    ptr_output = hardware.mem.allocate(args.num_pairs * SEQ_LEN * SEQ_LEN * 4)
    hardware.mem.finalize() 

    host_seq_1 = _get_random_int_array(args.num_pairs * SEQ_LEN)
    host_seq_2 = _get_random_int_array(args.num_pairs * SEQ_LEN)
    host_ref = _get_random_int_array(args.num_pairs * SEQ_LEN * SEQ_LEN)
    hardware.mem.set_value(ptr_seq_1, host_seq_1.tobytes())
    hardware.mem.set_value(ptr_seq_2, host_seq_2.tobytes())
    hardware.mem.set_value(ptr_ref, host_ref.tobytes())
    print("Hardware memory set-up: Success!") 
    
    penalty = 1

    # Start simulation
    print("Starting simulation...")
    total_cycles, sim_freq = hardware.run_simulation(
        kernel=opt_kernel,
        kernel_args=[ptr_seq_1, ptr_seq_2, ptr_ref, ptr_output,
                     args.num_pairs, penalty],
        grid_dim=(1, 1, args.num_blocks),
        block_dim=(1, 1, args.num_threads),
        block_schedule=mapping_dict, 
    )

    print(
        "Simulation finished: {} cycles at {} MHz".format(
            total_cycles, sim_freq)
    )

    # Compare results after simulation
    output_buffer = hardware.mem.get_value(
        ptr_output, args.num_pairs * SEQ_LEN * SEQ_LEN * 4)
    sim_results = np.array(
        struct.unpack("{}i".format(args.num_pairs * SEQ_LEN * SEQ_LEN),
                      output_buffer)
    ).astype(np.float32)

    ground_truth = _py_nw(host_seq_1, host_seq_2, host_ref,
                          args.num_pairs, penalty)
    np.testing.assert_allclose(sim_results, ground_truth, atol=1e-5)
    print("Correctness check: Success!")

    # Dump output files 
    if args.output_perf_file is not None:
        hardware.dump_perf_metrics(args.output_perf_file)

    if args.output_trace_file is not None:
        hardware.dump_timeline(args.output_trace_file)


if __name__ == "__main__":
    main()
