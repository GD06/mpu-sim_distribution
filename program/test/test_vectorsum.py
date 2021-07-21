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


class TestVectorSum(unittest.TestCase):

    def setUp(self):
        self.curr_dir = os.path.dirname(os.path.realpath(__file__))
        self.proj_dir = os.path.dirname(os.path.dirname(self.curr_dir))
        _, self.ptx_file = tempfile.mkstemp(suffix=".ptx", dir=self.curr_dir)

        cuda_file_oath = os.path.join(
            self.proj_dir, "benchmark", "vectorsum", "vectorsum_kernel.cu"
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

    def _run_vector_sum(self, vec_len, grid_dim, block_dim, mapping_dict):
        num_blocks = int(np.prod(grid_dim))
        num_threads = int(np.prod(block_dim))
        self.assertEqual(
            num_threads, 128, msg="Please double check cta_config.h under "
            "benchmark/vectorsum for the maximum number of threads."
        )
        
        hardware = sim_api.init_hardware(self.config) 
        input_ptr = hardware.mem.allocate(vec_len * 4)
        tmp_result = hardware.mem.allocate(num_blocks * 4)
        output_result = hardware.mem.allocate(4)
        hardware.mem.finalize() 

        a = np.random.rand(vec_len).astype(np.float32)
        ground_truth = float(np.sum(a))

        hardware.mem.set_value(input_ptr, a.tobytes())

        total_cycles, sim_freq = hardware.run_simulation(
            kernel=self.kernel_list[0],
            kernel_args=[input_ptr, tmp_result, vec_len],
            grid_dim=grid_dim,
            block_dim=block_dim, 
            block_schedule=mapping_dict,
        )

        if num_blocks == 1:
            output_buffer = hardware.mem.get_value(tmp_result, 4) 
        else:
            total_cycles, sim_freq = hardware.run_simulation(
                kernel=self.kernel_list[0],
                kernel_args=[tmp_result, output_result, num_blocks],
                grid_dim=(1, 1, 1),
                block_dim=block_dim, 
                block_schedule={0: 0},
            )
            output_buffer = hardware.mem.get_value(output_result, 4)

        sim_results = np.array(
            struct.unpack("1f", output_buffer) 
        ).astype(np.float32)

        np.testing.assert_almost_equal(sim_results[0], ground_truth, decimal=3)
        return 

    def test_single_block(self):
        self._run_vector_sum(
            vec_len=128,
            grid_dim=(1, 1, 1),
            block_dim=(1, 1, 128),
            mapping_dict={0: 0},
        )
        return

    def test_two_blocks(self):
        num_blocks = 2
        mapping_dict = {}
        for i in range(num_blocks):
            mapping_dict[i] = 0

        self._run_vector_sum(
            vec_len=600,
            grid_dim=(1, 1, num_blocks),
            block_dim=(1, 1, 128),
            mapping_dict=mapping_dict,  
        )
        return 

    def test_multiple_blocks(self):
        num_blocks = 8
        mapping_dict = {}
        for i in range(num_blocks):
            mapping_dict[i] = 0

        self._run_vector_sum(
            vec_len=1600,
            grid_dim=(1, 1, num_blocks),
            block_dim=(1, 1, 128),
            mapping_dict=mapping_dict,  
        )
        return 


if __name__ == "__main__":
    unittest.main() 
