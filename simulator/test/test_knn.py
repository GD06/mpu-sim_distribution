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


def _py_knn(x_array, y_array, point_x, point_y):
    result = [(x - point_x) * (x - point_x) + (y - point_y) * (y - point_y)
              for x, y in zip(x_array.tolist(), y_array.tolist())]

    return np.array(result).astype(np.float32)


class TestKNN(unittest.TestCase): 

    def setUp(self):
        self.curr_dir = os.path.dirname(os.path.realpath(__file__))
        self.proj_dir = os.path.dirname(os.path.dirname(self.curr_dir))
        _, self.ptx_file = tempfile.mkstemp(suffix=".ptx", dir=self.curr_dir)

        cuda_file_path = os.path.join(
            self.proj_dir, "benchmark", "knn", "knn_kernel.cu"
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

    def _run_knn(self, num_points, grid_dim, block_dim, mapping_dict):
        hardware = sim_api.init_hardware(self.config)
        ptr_x = hardware.mem.allocate(num_points * 4)
        ptr_y = hardware.mem.allocate(num_points * 4)
        ptr_dist = hardware.mem.allocate(num_points * 4)
        hardware.mem.finalize() 

        host_x = np.random.rand(num_points).astype(np.float32)
        host_y = np.random.rand(num_points).astype(np.float32)

        hardware.mem.set_value(ptr_x, host_x.tobytes())
        hardware.mem.set_value(ptr_y, host_y.tobytes())

        total_cycles, sim_freq = hardware.run_simulation(
            kernel=self.kernel_list[0],
            kernel_args=[ptr_x, ptr_y, ptr_dist, num_points, 0.5, 0.5],
            grid_dim=grid_dim,
            block_dim=block_dim,
            block_schedule=mapping_dict,
        )

        output_buffer = hardware.mem.get_value(ptr_dist, num_points * 4)
        sim_results = np.array(
            struct.unpack("{}f".format(num_points), output_buffer)
        ).astype(np.float32)

        ground_truth = _py_knn(host_x, host_y, 0.5, 0.5)

        np.testing.assert_allclose(sim_results, ground_truth, atol=1e-6)
        return  

    def test_single_warp_single_block(self):
        self._run_knn(    
            num_points=32,
            grid_dim=(1, 1, 1), 
            block_dim=(1, 1, 32),
            mapping_dict={0: 0},
        )
        return

    def test_single_warp_multiple_blocks(self):
        self._run_knn(
            num_points=96,
            grid_dim=(1, 1, 1),
            block_dim=(1, 1, 32),
            mapping_dict={0: 0}, 
        )
        return
    
    def test_single_warp_irregular_length(self):
        self._run_knn(
            num_points=100,
            grid_dim=(1, 1, 1),
            block_dim=(1, 1, 32),
            mapping_dict={0: 0},
        )
        return

    def test_multiple_blocks_multiple_blocks(self):
        num_blocks = 32
        mapping_dict = {}
        for i in range(num_blocks):
            mapping_dict[i] = 0

        self._run_knn(
            num_points=(num_blocks * 128 * 3 + 1000),
            grid_dim=(1, 1, num_blocks),
            block_dim=(1, 1, 128),
            mapping_dict=mapping_dict, 
        )
        return 


if __name__ == "__main__":
    unittest.main() 
