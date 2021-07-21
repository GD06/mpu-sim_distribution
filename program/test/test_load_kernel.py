#!/usr/bin/env python3 

import unittest 
import subprocess
import tempfile 
import os 

import program.prog_api as prog_api 


class TestLoadKernelClass(unittest.TestCase):

    def setUp(self):
        self.curr_dir = os.path.dirname(os.path.realpath(__file__))
        self.proj_dir = os.path.dirname(os.path.dirname(self.curr_dir))
        _, self.ptx_file = tempfile.mkstemp(suffix=".ptx", dir=self.curr_dir)

    def tearDown(self):
        os.remove(self.ptx_file)

    def test_ptx_exists(self):
        self.assertTrue(os.path.isfile(self.ptx_file))

    def _load_kernels(self, cuda_file_path): 
        self.assertTrue(os.path.isfile(cuda_file_path))

        subprocess.run(
            ["nvcc", "-O2", "--ptx", "-o", self.ptx_file, cuda_file_path], 
            check=True
        )

        kernel_list = prog_api.load_kernel(self.ptx_file)
        return kernel_list 

    def test_load_vectoradd(self):
        cuda_file_path = os.path.join(
            self.proj_dir, "benchmark", "vectoradd", "vectoradd_kernel.cu"
        ) 
        kernel_list = self._load_kernels(cuda_file_path)
        self.assertEqual(len(kernel_list), 1)

        vectoradd_kernel = kernel_list[0]

        self.assertEqual(len(vectoradd_kernel.arg_list), 4)
        self.assertEqual(len(vectoradd_kernel.reg_usage), 4)
        self.assertEqual(len(vectoradd_kernel.code_blocks.items()), 3)

    def test_load_vectorsum(self):
        cuda_file_path = os.path.join(
            self.proj_dir, "benchmark", "vectorsum", "vectorsum_kernel.cu"
        )
        kernel_list = self._load_kernels(cuda_file_path) 
        self.assertEqual(len(kernel_list), 1)

        vectorsum_kernel = kernel_list[0]

        self.assertEqual(len(vectorsum_kernel.arg_list), 3)
        self.assertEqual(len(vectorsum_kernel.reg_usage), 4)
        self.assertEqual(len(vectorsum_kernel.shared_memory_usage), 1)
        self.assertEqual(len(vectorsum_kernel.code_blocks.items()), 11)

    def test_load_matrixtrans(self):
        cuda_file_path = os.path.join(
            self.proj_dir, "benchmark", "matrixtrans", "matrixtrans_kernel.cu"
        )
        kernel_list = self._load_kernels(cuda_file_path)
        self.assertEqual(len(kernel_list), 1)

        matrixtrans_kernel = kernel_list[0]

        self.assertEqual(len(matrixtrans_kernel.arg_list), 4)
        self.assertEqual(len(matrixtrans_kernel.reg_usage), 4)
        self.assertEqual(len(matrixtrans_kernel.shared_memory_usage), 1)
        self.assertEqual(len(matrixtrans_kernel.code_blocks.items()), 3)

    def test_load_gemv(self):
        cuda_file_path = os.path.join(
            self.proj_dir, "benchmark", "gemv", "gemv_kernel.cu"
        )
        kernel_list = self._load_kernels(cuda_file_path)
        self.assertEqual(len(kernel_list), 1)

        gemv_kernel = kernel_list[0]

        self.assertEqual(len(gemv_kernel.arg_list), 5)
        self.assertEqual(len(gemv_kernel.reg_usage), 4)
        self.assertEqual(len(gemv_kernel.shared_memory_usage), 1)
        self.assertEqual(len(gemv_kernel.code_blocks.items()), 20)

    def test_load_kmeans(self):
        cuda_file_path = os.path.join(
            self.proj_dir, "benchmark", "kmeans", "kmeans_kernel.cu"
        )
        kernel_list = self._load_kernels(cuda_file_path)
        self.assertEqual(len(kernel_list), 1)

        kmeans_kernel = kernel_list[0]

        self.assertEqual(len(kmeans_kernel.arg_list), 4)
        self.assertEqual(len(kmeans_kernel.reg_usage), 4)
        self.assertEqual(len(kmeans_kernel.shared_memory_usage), 1)
        self.assertEqual(len(kmeans_kernel.code_blocks.items()), 5)

    def test_load_knn(self):
        cuda_file_path = os.path.join(
            self.proj_dir, "benchmark", "knn", "knn_kernel.cu"
        )
        kernel_list = self._load_kernels(cuda_file_path)
        self.assertEqual(len(kernel_list), 1)

        knn_kernel = kernel_list[0]

        self.assertEqual(len(knn_kernel.arg_list), 6)
        self.assertEqual(len(knn_kernel.reg_usage), 4)
        self.assertEqual(len(knn_kernel.code_blocks.items()), 3)

    def test_load_maxpool(self):
        cuda_file_path = os.path.join(
            self.proj_dir, "benchmark", "maxpool", "maxpool_kernel.cu"
        )
        kernel_list = self._load_kernels(cuda_file_path)
        self.assertEqual(len(kernel_list), 1)

        maxpool_kernel = kernel_list[0] 

        self.assertEqual(len(maxpool_kernel.arg_list), 4)
        self.assertEqual(len(maxpool_kernel.reg_usage), 4)
        self.assertEqual(len(maxpool_kernel.shared_memory_usage), 1)
        self.assertEqual(len(maxpool_kernel.code_blocks.items()), 3)

    def test_load_upsample(self):
        cuda_file_path = os.path.join(
            self.proj_dir, "benchmark", "upsample", "upsample_kernel.cu"
        )
        kernel_list = self._load_kernels(cuda_file_path)
        self.assertEqual(len(kernel_list), 1)

        upsample_kernel = kernel_list[0]

        self.assertEqual(len(upsample_kernel.arg_list), 4)
        self.assertEqual(len(upsample_kernel.reg_usage), 4)
        self.assertEqual(len(upsample_kernel.shared_memory_usage), 1)
        self.assertEqual(len(upsample_kernel.code_blocks.items()), 3)

    def test_load_blur(self):
        cuda_file_path = os.path.join(
            self.proj_dir, "benchmark", "blur", "blur_kernel.cu" 
        )
        kernel_list = self._load_kernels(cuda_file_path)
        self.assertEqual(len(kernel_list), 1)

        blur_kernel = kernel_list[0]

        self.assertEqual(len(blur_kernel.arg_list), 5)
        self.assertEqual(len(blur_kernel.reg_usage), 4)
        self.assertEqual(len(blur_kernel.shared_memory_usage), 1)
        self.assertEqual(len(blur_kernel.code_blocks.items()), 4)

    def test_load_conv(self):
        cuda_file_path = os.path.join(
            self.proj_dir, "benchmark", "conv", "conv_kernel.cu" 
        )
        kernel_list = self._load_kernels(cuda_file_path)
        self.assertEqual(len(kernel_list), 1)

        conv_kernel = kernel_list[0]

        self.assertEqual(len(conv_kernel.arg_list), 5)
        self.assertEqual(len(conv_kernel.reg_usage), 4)
        self.assertEqual(len(conv_kernel.shared_memory_usage), 2)
        self.assertEqual(len(conv_kernel.code_blocks.items()), 6)

    def test_load_histogram(self):
        cuda_file_path = os.path.join(
            self.proj_dir, "benchmark", "histogram", "histogram_kernel.cu"
        )
        kernel_list = self._load_kernels(cuda_file_path)
        self.assertEqual(len(kernel_list), 2)

        hist_kernel = kernel_list[0]
        reduce_kernel = kernel_list[1]

        self.assertEqual(len(hist_kernel.arg_list), 3)
        self.assertEqual(len(hist_kernel.reg_usage), 4)
        self.assertEqual(len(hist_kernel.shared_memory_usage), 1)
        self.assertEqual(len(hist_kernel.code_blocks.items()), 8)

        self.assertEqual(len(reduce_kernel.arg_list), 3)
        self.assertEqual(len(reduce_kernel.reg_usage), 3)
        self.assertEqual(len(reduce_kernel.code_blocks.items()), 10)

    def test_load_nw(self):
        cuda_file_path = os.path.join(
            self.proj_dir, "benchmark", "nw", "nw_kernel.cu" 
        )
        kernel_list = self._load_kernels(cuda_file_path) 
        self.assertEqual(len(kernel_list), 1)

        nw_kernel = kernel_list[0]

        self.assertEqual(len(nw_kernel.arg_list), 6)
        self.assertEqual(len(nw_kernel.reg_usage), 3)
        self.assertEqual(len(nw_kernel.shared_memory_usage), 2)
        self.assertEqual(len(nw_kernel.code_blocks.items()), 11)


if __name__ == "__main__":
    unittest.main() 
