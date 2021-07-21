#!/usr/bin/env python3

import unittest 
import subprocess 
import tempfile 
import os 

import program.prog_api as prog_api 
from backend.branch_analysis import reconvergence_analysis 


class TestReconvergenceAnalysis(unittest.TestCase): 

    def setUp(self):
        self.curr_dir = os.path.dirname(os.path.realpath(__file__))
        self.proj_dir = os.path.dirname(os.path.dirname(self.curr_dir))
        _, self.ptx_file = tempfile.mkstemp(suffix=".ptx", dir=self.curr_dir)

    def tearDown(self):
        os.remove(self.ptx_file)

    def test_ptx_exists(self):
        self.assertTrue(os.path.isfile(self.ptx_file))

    def _load_kernels_and_test(self, cuda_file_path):
        self.assertTrue(os.path.isfile(cuda_file_path))

        subprocess.run(
            ["nvcc", "-O2", "--ptx", "-o", self.ptx_file, cuda_file_path],
            check=True 
        )

        kernel_list = prog_api.load_kernel(self.ptx_file) 
        for each_kernel in kernel_list: 
            output_1 = reconvergence_analysis(each_kernel, mode="instr")
            output_2 = reconvergence_analysis(each_kernel, mode="code_block")
            self.assertEqual(len(output_1.instr_list), len(output_2.instr_list))

            num_instrs = len(output_1.instr_list)
            for i in range(num_instrs):
                if "dst_pc" in output_1.instr_list[i].metadata:
                    self.assertEqual(
                        output_1.instr_list[i].metadata["pdom"],
                        output_2.instr_list[i].metadata["pdom"]
                    )

        return 

    def test_vectoradd_analysis(self):
        cuda_file_path = os.path.join(
            self.proj_dir, "benchmark", "vectoradd", "vectoradd_kernel.cu")
        self._load_kernels_and_test(cuda_file_path)
        return

    def test_vectorsum_analysis(self):
        cuda_file_path = os.path.join(
            self.proj_dir, "benchmark", "vectorsum", "vectorsum_kernel.cu")
        self._load_kernels_and_test(cuda_file_path)
        return 

    def test_matrixtrans_analysis(self):
        cuda_file_path = os.path.join(
            self.proj_dir, "benchmark", "matrixtrans", "matrixtrans_kernel.cu")
        self._load_kernels_and_test(cuda_file_path)
        return

    def test_gemv_analysis(self):
        cuda_file_path = os.path.join(
            self.proj_dir, "benchmark", "gemv", "gemv_kernel.cu")
        self._load_kernels_and_test(cuda_file_path)
        return 

    def test_kmeans_analysis(self):
        cuda_file_path = os.path.join(
            self.proj_dir, "benchmark", "kmeans", "kmeans_kernel.cu")
        self._load_kernels_and_test(cuda_file_path)
        return

    def test_knn_analysis(self):
        cuda_file_path = os.path.join(
            self.proj_dir, "benchmark", "knn", "knn_kernel.cu")
        self._load_kernels_and_test(cuda_file_path)
        return 

    def test_maxpool_analysis(self):
        cuda_file_path = os.path.join(
            self.proj_dir, "benchmark", "maxpool", "maxpool_kernel.cu")
        self._load_kernels_and_test(cuda_file_path)
        return

    def test_upsample_analysis(self):
        cuda_file_path = os.path.join(
            self.proj_dir, "benchmark", "upsample", "upsample_kernel.cu")
        self._load_kernels_and_test(cuda_file_path)
        return 

    def test_blur_analysis(self):
        cuda_file_path = os.path.join(
            self.proj_dir, "benchmark", "blur", "blur_kernel.cu")
        self._load_kernels_and_test(cuda_file_path)
        return 

    def test_conv_analysis(self):
        cuda_file_path = os.path.join(
            self.proj_dir, "benchmark", "conv", "conv_kernel.cu")
        self._load_kernels_and_test(cuda_file_path)
        return

    def test_histogram_analysis(self):
        cuda_file_path = os.path.join(
            self.proj_dir, "benchmark", "histogram", "histogram_kernel.cu")
        self._load_kernels_and_test(cuda_file_path)
        return

    def test_nw_analysis(self):
        cuda_file_path = os.path.join(
            self.proj_dir, "benchmark", "nw", "nw_kernel.cu")
        self._load_kernels_and_test(cuda_file_path)
        return 


if __name__ == "__main__":
    unittest.main() 
