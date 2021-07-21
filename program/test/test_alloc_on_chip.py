#!/usr/bin/env python3 

import unittest 
import tempfile 
import os 

import program.prog_api as prog_api 
import config.config_api as config_api 


class TestAllocOnChip(unittest.TestCase):

    def setUp(self):
        self.curr_dir = os.path.dirname(os.path.realpath(__file__))
        self.proj_dir = os.path.dirname(os.path.dirname(self.curr_dir))
        _, self.ptx_file = tempfile.mkstemp(suffix=".ptx", dir=self.curr_dir)
        self.config = config_api.load_hardware_config() 

    def tearDown(self):
        os.remove(self.ptx_file)

    def test_ptx_exists(self):
        self.assertTrue(os.path.isfile(self.ptx_file))

    def test_reg_alloc(self):
        # Print out a simple kernel with registers allocated 
        with open(self.ptx_file, "w") as f: 
            print(".visible .entry _Z9Kernel(", file=f)
            print("\t .param .u32 _Z9_param_0", file=f)
            print(")", file=f)
            print("{", file=f)
            print("\t .reg .pred\t %p<11>;", file=f)
            print("\t .reg .f32\t %f<31>;", file=f)
            print("\t .reg .b32\t %r<14>;", file=f)
            print("\t .reg .b64\t %rd<9>;", file=f)
            print("\t ret;", file=f)
            print("}", file=f)

        kernel_list = prog_api.load_kernel(self.ptx_file) 
        self.assertEqual(len(kernel_list), 1)

        kernel = kernel_list[0]
        self.assertEqual(len(kernel.arg_list), 1)
        
        reg_usage_per_warp, shared_memory_usage_per_block = (
            kernel.compute_resource_usage(
                data_path_unit_size=self.config["data_path_unit_size"],
                num_threads_per_warp=self.config["num_threads_per_warp"],
                num_pe=self.config["num_pe"]
            ) 
        )

        self.assertEqual(
            reg_usage_per_warp, 
            384 + 31 * 4 * 32 + 14 * 4 * 32 + 9 * 8 * 32
        )
        self.assertEqual(shared_memory_usage_per_block, 0) 

        self.assertEqual(kernel.reg_offset["%p"], 0)
        self.assertEqual(kernel.reg_offset["%f"], 384) 
        self.assertEqual(kernel.reg_offset["%r"], 384 + 31 * 4 * 32)
        rd_offset = 384 + 31 * 4 * 32 + 14 * 4 * 32
        self.assertEqual(kernel.reg_offset["%rd"], rd_offset)

        # predicate register aligned to 4 bytes
        self.assertEqual(kernel.reg_size["%p"], 32)
        self.assertEqual(kernel.reg_size["%f"], 128)
        self.assertEqual(kernel.reg_size["%r"], 128)
        self.assertEqual(kernel.reg_size["%rd"], 256)
        return

    def test_smem_alloc(self):
        # Print out a simple kernel with shared memory allocated 
        with open(self.ptx_file, "w") as f: 
            print(".visible .entry _Z9Kernel(", file=f)
            print("\t .param .u32 _Z9_param_0", file=f)
            print(")", file=f)
            print("{", file=f)
            print("\t .reg .pred\t %p<2>;", file=f)
            print(".shared .align 4 .b8 _var1[508];", file=f)
            print(".shared .align 2 .b8 _var2[65];", file=f)
            print(".shared .align 4 .b8 _var3[128];", file=f)
            print("\t ret;", file=f)
            print("}", file=f)

        kernel_list = prog_api.load_kernel(self.ptx_file) 
        self.assertEqual(len(kernel_list), 1)

        kernel = kernel_list[0]
        self.assertEqual(len(kernel.arg_list), 1)
        
        reg_usage_per_warp, shared_memory_usage_per_block = (
            kernel.compute_resource_usage(
                data_path_unit_size=self.config["data_path_unit_size"],
                num_threads_per_warp=self.config["num_threads_per_warp"],
                num_pe=self.config["num_pe"],
            ) 
        )

        self.assertEqual(reg_usage_per_warp, 128) 
        self.assertEqual(shared_memory_usage_per_block, 512 + 128 + 128)

        self.assertEqual(kernel.smem_offset["_var1"], 0)
        self.assertEqual(kernel.smem_offset["_var2"], 512)
        self.assertEqual(kernel.smem_offset["_var3"], 512 + 128) 
        return 
       

if __name__ == "__main__":
    unittest.main() 
