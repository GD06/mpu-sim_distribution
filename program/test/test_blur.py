#!/usr/bin/env python3 

import unittest 
import numpy as np
import struct
import os
import tempfile
import subprocess 

import program.prog_api as prog_api 
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


class TestBlur(unittest.TestCase): 

    def setUp(self):
        self.curr_dir = os.path.dirname(os.path.realpath(__file__))
        self.proj_dir = os.path.dirname(os.path.dirname(self.curr_dir))
        _, self.ptx_file = tempfile.mkstemp(suffix=".ptx", dir=self.curr_dir)

        cuda_file_path = os.path.join(
            self.proj_dir, "benchmark", "blur", "blur_kernel.cu"
        )
        self.assertTrue(os.path.isfile(cuda_file_path))
        self.assertTrue(os.path.isfile(self.ptx_file))

        subprocess.run(
            ["nvcc", "-O2", "--ptx", "-o", self.ptx_file, cuda_file_path], 
            check=True
        )

        self.config = config_api.load_hardware_config() 
        
        self.raw_kernel_list = prog_api.load_kernel(self.ptx_file)
        self.kernel_list = []
        for each_kernel in self.raw_kernel_list:
            output_kernel = prog_api.compile_to_exec(each_kernel, self.config)
            self.kernel_list.append(output_kernel)
        return 

    def tearDown(self):
        os.remove(self.ptx_file)

    def _run_blur(self, num_rows, num_cols, grid_dim, block_dim, mapping_dict):
        self.assertTupleEqual(block_dim, (1, NUM_THREADS_Y, NUM_THREADS_X))

        hardware = sim_api.init_hardware(self.config)
        ptr_input = hardware.mem.allocate(num_rows * num_cols * 4)
        ptr_output = hardware.mem.allocate(num_rows * num_cols * 4)
        hardware.mem.finalize() 

        host_input = np.random.rand(num_rows * num_cols).astype(np.float32)
        hardware.mem.set_value(ptr_input, host_input.tobytes())

        total_cycles, sim_freq = hardware.run_simulation(
            kernel=self.kernel_list[0],
            kernel_args=[ptr_input, ptr_output, num_rows, num_cols, 1.0 / 9.0],
            grid_dim=grid_dim,
            block_dim=block_dim,
            block_schedule=mapping_dict,
        )

        output_buffer = hardware.mem.get_value(
            ptr_output, num_rows * num_cols * 4)
        sim_results = np.array(
            struct.unpack("{}f".format(num_rows * num_cols), output_buffer)
        ).astype(np.float32)

        ground_truth = _py_blur(host_input, num_rows, num_cols)
        
        np.testing.assert_allclose(sim_results, ground_truth, atol=1e-6)
        return  

    def test_singl_block(self):
        self._run_blur(
            num_rows=32,
            num_cols=64,
            grid_dim=(1, 1, 1),
            block_dim=(1, 4, 32),
            mapping_dict={0: 0},
        )
        return

    def test_multiple_blocks(self):
        self._run_blur(
            num_rows=64,
            num_cols=64,
            grid_dim=(1, 1, 4),
            block_dim=(1, 4, 32),
            mapping_dict={0: 0, 1: 1, 2: 2, 3: 3},
        )
        return 


if __name__ == "__main__":
    unittest.main() 
