#!/usr/bin/env python3 

import unittest 
import tempfile 
import os 

import util.mtx_api as mtx_api 


class TestMTXAPI(unittest.TestCase):

    def setUp(self):
        self.curr_dir = os.path.dirname(os.path.realpath(__file__))
        _, self.mtx_file = tempfile.mkstemp(suffix=".mtx", dir=self.curr_dir) 

    def tearDown(self):
        os.remove(self.mtx_file)

    def _generate_graph_and_test(self, num_nodes, num_edges, directed):
        ptr_row, ptr_col, ptr_val = mtx_api.gen_random_graph(
            num_nodes, num_edges, directed)
        symmetric = (directed is False)

        mtx_api.dump_mtx_file(self.mtx_file, num_nodes, num_nodes, ptr_row, 
                              ptr_col, ptr_val, symmetric=symmetric) 

        new_ptr_row, new_ptr_col, new_ptr_val = mtx_api.load_mtx_file(
            self.mtx_file)

        self.assertEqual(ptr_row, new_ptr_row)
        self.assertEqual(ptr_col, new_ptr_col)
        self.assertEqual(ptr_val, new_ptr_val)
        return 

    def test_directed_graph(self):
        self._generate_graph_and_test(10000, 100000, True)
        return 

    def test_undirected_graph(self):
        self._generate_graph_and_test(10000, 100000, False)
        return 


if __name__ == "__main__":
    unittest.main() 
