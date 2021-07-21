#!/usr/bin/env python3 

import argparse 
import os 
import json 
import subprocess 
import multiprocessing as mp 


def _runner(q):

    while True:
        task = q.get()
        if task is None:
            break 

        task_name = task[0]
        meta_data = task[1]

        curr_dir = os.path.dirname(os.path.realpath(__file__))
        proj_dir = os.path.dirname(os.path.dirname(curr_dir))
        cuda_file_path = os.path.join(
            proj_dir, "benchmark", task_name, "{}_kernel.cu".format(task_name)
        )

        output_dir = meta_data["output_dir"]
        prog_dir = os.path.join(output_dir, "program", task_name)
        os.makedirs(prog_dir, exist_ok=True)

        ptx_file_path = os.path.join(
            prog_dir, "{}_kernel.ptx".format(task_name)
        )

        try:
            subprocess.run(
                ["nvcc", "-O2", "--ptx", "-o", ptx_file_path, cuda_file_path],
                check=True
            )
        except Exception as err:
            print("Error during the compilation of the workload: {}".format(
                task_name))
            print("Error: {}".format(repr(err)))
            continue

        try:
            subprocess.run(
                [os.path.join(proj_dir, "program", "ptx2exec_cli.py"), 
                 ptx_file_path],
                check=True 
            )
        except Exception as err:
            print("Error during the compilation of the workload: {}".format(
                task_name))
            print("Error: {}".format(repr(err)))
            continue 

        results_dir = os.path.join(output_dir, "performance", task_name)
        os.makedirs(results_dir, exist_ok=True)

        log_file = os.path.join(results_dir, "{}_stderr.log".format(task_name))
        perf_trace_file = os.path.join(
            results_dir, "{}_perf_trace.json".format(task_name)) 

        sim_cmd = [
            os.path.join(proj_dir, "benchmark", task_name, 
                         "{}_sim.py".format(task_name))
        ]

        prog_files = []
        for filepath in os.listdir(prog_dir):
            if filepath.endswith(".prog"):
                prog_files.append(os.path.join(prog_dir, filepath))

        sim_cmd.extend(sorted(prog_files))
        sim_cmd.extend(meta_data["sim_options"])
        sim_cmd.extend(["--output_perf_file", perf_trace_file])

        print("Start the simulation of the workload: {}".format(task_name))

        try:
            with open(log_file, "w") as f:
                subprocess.run(
                    sim_cmd, stdout=f, stderr=subprocess.STDOUT, 
                    timeout=meta_data["timeout"], check=True,
                )
        except Exception as err:
            print("Error during the simulation of the workload: {}".format(
                task_name))
            print("Error: {}".format(repr(err)))
            continue 

        print("Finish the simulation of the workload: {}".format(task_name))

    return 


def main():

    parser = argparse.ArgumentParser(
        description="Run the simulation of all workloads",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter 
    )

    parser.add_argument("output_dir", help="The output directory path to store"
                        "the simulation results")

    parser.add_argument("--benchmarks", default="benchmarks.json", help="The "
                        "input file storing the names of all workloads")
    parser.add_argument("--num_processes", type=int, default=20)

    # Timeout after 24 hours 
    parser.add_argument("--timeout", type=int, default=86400)

    args = parser.parse_args() 

    with open(args.benchmarks, "r") as f:
        task_dict = json.load(f)

    task_queue = mp.Queue() 
    proc_list = []
    for i in range(args.num_processes):
        p = mp.Process(target=_runner, args=(task_queue,))
        proc_list.append(p)

    for i in range(args.num_processes):
        proc_list[i].start() 

    for task_name in task_dict.keys():
        sim_options = task_dict[task_name]
        meta_data = {}
        meta_data["sim_options"] = sim_options 
        meta_data["output_dir"] = args.output_dir 
        meta_data["timeout"] = args.timeout 
        task_queue.put((task_name, meta_data))

    for i in range(args.num_processes):
        task_queue.put(None)

    for i in range(args.num_processes):
        proc_list[i].join() 


if __name__ == "__main__":
    main()
