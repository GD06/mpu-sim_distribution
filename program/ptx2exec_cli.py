#!/usr/bin/env python3

import argparse 
import os 
import pickle 

from program.prog_api import load_kernel, compile_to_exec, optimize_kernel 
import config.config_api as config_api 


def main():

    parser = argparse.ArgumentParser(
        description="Compile kernels in PTX file to MPU executable kernels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter 
    )

    parser.add_argument("input_ptx_file_path", 
                        help="Specify the path of input PTX file")
    parser.add_argument("--output_dir", "-o", default=None,
                        help="Specify the directory path of output files")
    parser.add_argument("--hardware_config", "-c", default=None,
                        help="Specify the config file path of hardware config")
    parser.add_argument("--opt", action="store_true",
                        help="Whether to perform backend optimizations")

    args = parser.parse_args() 

    if args.output_dir is None:
        output_dir = os.path.dirname(os.path.realpath(args.input_ptx_file_path))
    else:
        output_dir = args.output_dir 

    input_file_base_name = os.path.basename(args.input_ptx_file_path) 
    pos = input_file_base_name.find(".ptx")
    assert pos > 0, "The input file path does not end in .ptx"

    output_file_base_name = input_file_base_name[:pos] + "_{}.prog"

    hw_config_dict = config_api.load_hardware_config(
        overwrite_config_file_path=args.hardware_config
    )

    kernel_list = load_kernel(args.input_ptx_file_path) 

    for index in range(len(kernel_list)):
        print("Compiling {}/{} kernels".format(index + 1, len(kernel_list)))
        exec_kernel = compile_to_exec(kernel_list[index], hw_config_dict)

        if args.opt: 
            print(
                "Optimizing {}/{} kernels".format(index + 1, len(kernel_list)))
            opt_kernel = optimize_kernel(exec_kernel)
        else:
            opt_kernel = exec_kernel 

        with open(os.path.join(
                output_dir, output_file_base_name.format(index)), "wb") as f:

            pickle.dump(opt_kernel, f)

    print("Finished all {} kernels, please find output files in: {}".format(
        len(kernel_list), output_dir))


if __name__ == "__main__":
    main() 
