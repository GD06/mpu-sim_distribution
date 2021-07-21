#!/usr/bin/env python3 

import unittest 
import numpy as np
import struct 

from simulator.memory import Memory 


class TestMemoryClass(unittest.TestCase):

    def test_mem_alloc(self):
        mem_obj = Memory(alignment=128)

        ptr_a = mem_obj.allocate(4)
        ptr_b = mem_obj.allocate(251)
        ptr_c = mem_obj.allocate(3)

        self.assertEqual(ptr_a, 0)
        self.assertEqual(ptr_b, 128)
        self.assertEqual(ptr_c, 128 + 256)

        return 

    def test_mem_value(self):
        mem_obj = Memory(alignment=128)

        ptr = mem_obj.allocate(65536)
        mem_obj.finalize()

        a = np.random.rand(1024).astype(np.float32)
        mem_obj.set_value(ptr + 1024, a.tobytes())

        b_in_bytes = mem_obj.get_value(ptr + 512, 1024 * 4)
        b_in_list = struct.unpack("1024f", b_in_bytes)
        b = np.array(b_in_list).astype(np.float32)

        np.testing.assert_array_equal(b[128:], a[:(1024 - 128)])
        

if __name__ == "__main__":
    unittest.main() 
