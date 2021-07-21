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

NUM_THREADS = 128


def _py_upsample(host_input, num_output_rows, num_output_cols):
    num_input_rows, num_input_cols = host_input.shape 
    results = [0.0] * num_output_rows * num_output_cols

    for i in range(num_input_rows):
        for j in range(0, num_input_cols, NUM_THREADS):
            tmp_buffer = host_input[i, j: j + NUM_THREADS]
            for kx in range(2):
                for ky in range(2):
                    for k in range(NUM_THREADS):
                        output_index = ((kx * 2 + ky) * num_input_rows 
                                        * num_input_cols + i * num_input_cols 
                                        + j + k)
                        results[output_index] = tmp_buffer[
                            (ky * NUM_THREADS + k) // 2]

    np_results = np.array(results).astype(np.float32).reshape(
        (num_output_rows, num_output_cols))
    return np_results 


class TestUpSample(unittest.TestCase): 

    def setUp(self):
        self.curr_dir = os.path.dirname(os.path.realpath(__file__))
        self.proj_dir = os.path.dirname(os.path.dirname(self.curr_dir))
        _, self.ptx_file = tempfile.mkstemp(suffix=".ptx", dir=self.curr_dir)

        cuda_file_path = os.path.join(
            self.proj_dir, "benchmark", "upsample", "upsample_kernel.cu"
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

    def _run_upsample(self, num_input_rows, num_input_cols, grid_dim,
                      block_dim, mapping_dict):
        self.assertTupleEqual(block_dim, (1, 1, NUM_THREADS))

        num_output_rows = num_input_rows * 2
        num_output_cols = num_input_cols * 2

        hardware = sim_api.init_hardware(self.config)
        ptr_input = hardware.mem.allocate(num_input_rows * num_input_cols * 4)
        ptr_output = hardware.mem.allocate(
            num_output_rows * num_output_cols * 4)
        hardware.mem.finalize()

        host_input = np.random.rand(
            num_input_rows * num_input_cols).astype(np.float32)
        hardware.mem.set_value(ptr_input, host_input.tobytes())

        total_cycles, sim_freq = hardware.run_simulation(
            kernel=self.kernel_list[0],
            kernel_args=[
                ptr_input, ptr_output, num_input_rows, num_input_cols],
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

        ground_truth = _py_upsample(
            host_input.reshape((num_input_rows, num_input_cols)), 
            num_output_rows, num_output_cols
        )

        np.testing.assert_allclose(sim_results, ground_truth, atol=1e-6)
        return  

    def test_singl_block(self):
        self._run_upsample(
            num_input_rows=2,
            num_input_cols=256,
            grid_dim=(1, 1, 1),
            block_dim=(1, 1, 128),
            mapping_dict={0: 0},
        )
        return

    def test_multiple_blocks(self):
        self._run_upsample(
            num_input_rows=10,
            num_input_cols=128,
            grid_dim=(1, 1, 4),
            block_dim=(1, 1, 128),
            mapping_dict={0: 0, 1: 1, 2: 2, 3: 3},
        )
        return 


if __name__ == "__main__":
    unittest.main() 
