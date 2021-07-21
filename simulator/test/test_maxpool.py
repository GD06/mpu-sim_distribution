#!/usr/bin/env python3 

import unittest 
import numpy as np
import struct
import os
import tempfile
import subprocess 

from backend.branch_analysis import reconvergence_analysis 
import program.prog_api as prog_api 
import config.config_api as config_api 
import simulator.sim_api as sim_api 

NUM_THREADS = 128


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


class TestMaxPool(unittest.TestCase): 

    def setUp(self):
        self.curr_dir = os.path.dirname(os.path.realpath(__file__))
        self.proj_dir = os.path.dirname(os.path.dirname(self.curr_dir))
        _, self.ptx_file = tempfile.mkstemp(suffix=".ptx", dir=self.curr_dir)

        cuda_file_path = os.path.join(
            self.proj_dir, "benchmark", "maxpool", "maxpool_kernel.cu"
        )
        self.assertTrue(os.path.isfile(cuda_file_path))
        self.assertTrue(os.path.isfile(self.ptx_file))

        subprocess.run(
            ["nvcc", "-O2", "--ptx", "-o", self.ptx_file, cuda_file_path], 
            check=True
        )

        self.raw_kernel_list = prog_api.load_kernel(self.ptx_file)
        self.kernel_list = []
        for each_kernel in self.raw_kernel_list:
            output_kernel = reconvergence_analysis(each_kernel, mode="instr")
            self.kernel_list.append(output_kernel)
        self.config = config_api.load_hardware_config() 
        return 

    def tearDown(self):
        os.remove(self.ptx_file)

    def _run_maxpool(self, num_output_rows, num_output_cols, grid_dim,
                     block_dim, mapping_dict):
        self.assertTupleEqual(block_dim, (1, 1, NUM_THREADS))
        num_input_rows = num_output_rows * 2
        num_input_cols = num_output_cols * 2

        hardware = sim_api.init_hardware(self.config)
        ptr_input = hardware.mem.allocate(num_input_rows * num_input_cols * 4)
        ptr_output = hardware.mem.allocate(
            num_output_rows * num_output_cols * 4)
        hardware.mem.finalize()

        host_input = np.random.rand(
            num_input_rows * num_input_cols).astype(np.float32)
        host_output = np.random.rand(
            num_output_rows * num_output_cols).astype(np.float32)
        hardware.mem.set_value(ptr_input, host_input.tobytes())
        hardware.mem.set_value(ptr_output, host_output.tobytes())

        total_cycles, sim_freq = hardware.run_simulation(
            kernel=self.kernel_list[0],
            kernel_args=[
                ptr_input, ptr_output, num_output_rows, num_output_cols],
            grid_dim=grid_dim,
            block_dim=block_dim,
            block_schedule=mapping_dict,
        )

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

        np.testing.assert_allclose(sim_results, ground_truth, atol=1e-6)
        return  

    def test_singl_block(self):
        self._run_maxpool(
            num_output_rows=2,
            num_output_cols=256,
            grid_dim=(1, 1, 1),
            block_dim=(1, 1, 128),
            mapping_dict={0: 0},
        )
        return

    def test_multiple_blocks(self):
        self._run_maxpool(
            num_output_rows=10,
            num_output_cols=128,
            grid_dim=(1, 1, 4),
            block_dim=(1, 1, 128),
            mapping_dict={0: 0, 1: 1, 2: 2, 3: 3},
        )


if __name__ == "__main__":
    unittest.main() 
