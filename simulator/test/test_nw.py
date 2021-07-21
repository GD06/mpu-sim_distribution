#!/usr/bin/env python3 

import unittest 
import numpy as np
import struct 
import os
import tempfile 
import subprocess 
import json 

from backend.branch_analysis import reconvergence_analysis 
import program.prog_api as prog_api 
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


class TestNW(unittest.TestCase):

    def setUp(self):
        self.curr_dir = os.path.dirname(os.path.realpath(__file__))
        self.proj_dir = os.path.dirname(os.path.dirname(self.curr_dir))
        _, self.ptx_file = tempfile.mkstemp(suffix=".ptx", dir=self.curr_dir)
        _, self.json_file = tempfile.mkstemp(suffix=".json", dir=self.curr_dir)

        cuda_file_path = os.path.join(
            self.proj_dir, "benchmark", "nw", "nw_kernel.cu"
        )
        self.assertTrue(os.path.isfile(cuda_file_path))
        self.assertTrue(os.path.isfile(self.ptx_file)) 

        subprocess.run(
            ["nvcc", "-O2", "--ptx", "-o", self.ptx_file, cuda_file_path],
            check=True 
        )

        with open(self.json_file, "w") as f:
            json.dump({"subcore_reg_file_size": 65536}, f)
        self.config = config_api.load_hardware_config(
            overwrite_config_file_path=self.json_file)
        
        self.raw_kernel_list = prog_api.load_kernel(self.ptx_file)
        self.kernel_list = []
        for each_kernel in self.raw_kernel_list:
            output_kernel = reconvergence_analysis(each_kernel, mode="instr")
            self.kernel_list.append(output_kernel) 
        return 

    def tearDown(self):
        os.remove(self.ptx_file)
        os.remove(self.json_file)

    def _run_nw(self, num_pairs, grid_dim, block_dim, mapping_dict):
        hardware = sim_api.init_hardware(self.config)
        ptr_seq_1 = hardware.mem.allocate(num_pairs * SEQ_LEN * 4)
        ptr_seq_2 = hardware.mem.allocate(num_pairs * SEQ_LEN * 4)
        ptr_ref = hardware.mem.allocate(num_pairs * SEQ_LEN * SEQ_LEN * 4)
        ptr_output = hardware.mem.allocate(
            num_pairs * SEQ_LEN * SEQ_LEN * 4)
        hardware.mem.finalize() 

        host_seq_1 = _get_random_int_array(num_pairs * SEQ_LEN)
        host_seq_2 = _get_random_int_array(num_pairs * SEQ_LEN)
        host_ref = _get_random_int_array(num_pairs * SEQ_LEN * SEQ_LEN)
        hardware.mem.set_value(ptr_seq_1, host_seq_1.tobytes())
        hardware.mem.set_value(ptr_seq_2, host_seq_2.tobytes())
        hardware.mem.set_value(ptr_ref, host_ref.tobytes())

        penalty = 1

        total_cyles, sim_freq = hardware.run_simulation(
            kernel=self.kernel_list[0],
            kernel_args=[ptr_seq_1, ptr_seq_2, ptr_ref, ptr_output,
                         num_pairs, penalty],
            grid_dim=grid_dim,
            block_dim=block_dim,
            block_schedule=mapping_dict,
        )

        output_buffer = hardware.mem.get_value(
            ptr_output, num_pairs * SEQ_LEN * SEQ_LEN * 4)
        sim_results = np.array(
            struct.unpack("{}i".format(num_pairs * SEQ_LEN * SEQ_LEN), 
                          output_buffer)
        ).astype(np.float32)
    
        ground_truth = _py_nw(host_seq_1, host_seq_2, host_ref, 
                              num_pairs, penalty)
        np.testing.assert_allclose(sim_results, ground_truth, rtol=1e-3)
        return 

    def test_single_block(self):
        self._run_nw(
            num_pairs=1,
            grid_dim=(1, 1, 1),
            block_dim=(1, 1, 32),
            mapping_dict={0: 0},
        )
        return

    def test_multiple_blocks(self):
        self._run_nw(
            num_pairs=6,
            grid_dim=(1, 1, 4),
            block_dim=(1, 1, 32),
            mapping_dict={0: 0, 1: 1, 2: 2, 3: 3}
        )
        return 


if __name__ == "__main__":
    unittest.main() 
