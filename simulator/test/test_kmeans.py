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

_NUM_FEATURES = 4
_NUM_CLUSTERS = 4


def _py_kmeans(np_points, np_centers):
    result_list = []
    num_points, _ = np_points.shape 
    num_centers, _ = np_centers.shape 
    for i in range(num_points):
        dist_array = [np.linalg.norm(np_points[i, :] - np_centers[j, :]) 
                      for j in range(num_centers)]
        result_list.append(np.argmin(dist_array))
    return np.array(result_list).astype(np.float32)


class TestKMeans(unittest.TestCase): 

    def setUp(self):
        self.curr_dir = os.path.dirname(os.path.realpath(__file__))
        self.proj_dir = os.path.dirname(os.path.dirname(self.curr_dir))
        _, self.ptx_file = tempfile.mkstemp(suffix=".ptx", dir=self.curr_dir)

        cuda_file_path = os.path.join(
            self.proj_dir, "benchmark", "kmeans", "kmeans_kernel.cu"
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

    def _run_kmeans(self, num_points, grid_dim, block_dim, mapping_dict):
        hardware = sim_api.init_hardware(self.config)
        ptr_points = hardware.mem.allocate(num_points * _NUM_FEATURES * 4)
        ptr_centers = hardware.mem.allocate(_NUM_CLUSTERS * _NUM_FEATURES * 4)
        ptr_membership = hardware.mem.allocate(num_points * 4)
        hardware.mem.finalize() 

        np_points = np.random.rand(
            num_points * _NUM_FEATURES).astype(np.float32)
        np_centers = np.random.rand(
            _NUM_CLUSTERS * _NUM_FEATURES).astype(np.float32)
        hardware.mem.set_value(ptr_points, np_points.tobytes())
        hardware.mem.set_value(ptr_centers, np_centers.tobytes())

        total_cycles, sim_freq = hardware.run_simulation(
            kernel=self.kernel_list[0],
            kernel_args=[ptr_points, ptr_centers, ptr_membership, num_points],
            grid_dim=grid_dim,
            block_dim=block_dim,
            block_schedule=mapping_dict,
        )

        output_buffer = hardware.mem.get_value(ptr_membership, num_points * 4)
        sim_results = np.array(
            struct.unpack("{}i".format(num_points), output_buffer)
        ).astype(np.float32)

        ground_truth = _py_kmeans(
            np.transpose(np_points.reshape((_NUM_FEATURES, num_points))),
            np.transpose(np_centers.reshape((_NUM_FEATURES, _NUM_CLUSTERS))),
        )

        ground_truth_sum = float(np.sum(ground_truth))
        sim_results_sum = float(np.sum(sim_results))
        np.testing.assert_allclose(
            [ground_truth_sum], [sim_results_sum], rtol=1e-3)
        return

    def test_single_block(self):
        self._run_kmeans(
            num_points=128,
            grid_dim=(1, 1, 1),
            block_dim=(1, 1, 128),
            mapping_dict={0: 0},
        )
        return 

    def test_multiple_block(self):
        self._run_kmeans(
            num_points=500,
            grid_dim=(1, 1, 2),
            block_dim=(1, 1, 128),
            mapping_dict={0: 0, 1: 1},
        )
        return 


if __name__ == "__main__":
    unittest.main()
