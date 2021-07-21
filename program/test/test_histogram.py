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

NUM_BINS = 256


def _py_histogram(host_input):
    results = [0] * NUM_BINS
    for i in range(len(host_input)):
        val = min(max(host_input[i] * 255.0, 0), NUM_BINS - 1)
        index = int(val)
        results[index] += 1
    return np.array(results).astype(np.float32)


class TestHistogram(unittest.TestCase):

    def setUp(self):
        self.curr_dir = os.path.dirname(os.path.realpath(__file__))
        self.proj_dir = os.path.dirname(os.path.dirname(self.curr_dir))
        _, self.ptx_file = tempfile.mkstemp(suffix=".ptx", dir=self.curr_dir)

        cuda_file_oath = os.path.join(
            self.proj_dir, "benchmark", "histogram", "histogram_kernel.cu"
        )
        self.assertTrue(os.path.isfile(cuda_file_oath))
        self.assertTrue(os.path.isfile(self.ptx_file)) 

        subprocess.run(
            ["nvcc", "-O2", "--ptx", "-o", self.ptx_file, cuda_file_oath],
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

    def _run_histogram(self, length, grid_dim, block_dim, mapping_dict):
        num_blocks = int(np.prod(grid_dim))

        hardware = sim_api.init_hardware(self.config)
        ptr_input = hardware.mem.allocate(length * 4)
        tmp_result = hardware.mem.allocate(num_blocks * NUM_BINS * 4)
        output_result = hardware.mem.allocate(NUM_BINS * 4)
        hardware.mem.finalize() 

        host_input = np.random.rand(length).astype(np.float32)
        hardware.mem.set_value(ptr_input, host_input.tobytes())

        total_cycles, sim_freq = hardware.run_simulation(
            kernel=self.kernel_list[0],
            kernel_args=[ptr_input, tmp_result, length],
            grid_dim=grid_dim,
            block_dim=block_dim,
            block_schedule=mapping_dict,
        )

        assert NUM_BINS == 256, "Please change the reduction configs"
        total_cycles, sim_freq = hardware.run_simulation(
            kernel=self.kernel_list[1],
            kernel_args=[tmp_result, output_result, num_blocks],
            grid_dim=(1, 1, 2),
            block_dim=(1, 1, 128),
            block_schedule={0: 0, 1: 1},
        )

        output_buffer = hardware.mem.get_value(output_result, NUM_BINS * 4)
        sim_results = np.array(
            struct.unpack("{}i".format(NUM_BINS), output_buffer)
        ).astype(np.float32)
    
        ground_truth = _py_histogram(host_input)
        np.testing.assert_allclose(sim_results, ground_truth, rtol=1e-3)
        return 

    def test_single_block(self):
        self._run_histogram(
            length=256,
            grid_dim=(1, 1, 1),
            block_dim=(1, 1, 128),
            mapping_dict={0: 0},
        )
        return

    def test_multiple_blocks(self):
        self._run_histogram(
            length=1280,
            grid_dim=(1, 1, 4),
            block_dim=(1, 1, 128),
            mapping_dict={0: 0, 1: 1, 2: 2, 3: 3}
        )
        return 


if __name__ == "__main__":
    unittest.main() 
