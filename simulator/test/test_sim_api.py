#!/usr/bin/env python3

import unittest 
import os
import tempfile 
import json 

import simulator.sim_api as sim_api 
import config.config_api as config_api 


class TestSimAPI(unittest.TestCase):

    def setUp(self):
        self.curr_dir = os.path.dirname(os.path.realpath(__file__))
        _, self.json_file = tempfile.mkstemp(suffix=".json", dir=self.curr_dir)
        self.config = config_api.load_hardware_config() 
        return 
    
    def tearDown(self):
        os.remove(self.json_file)

    def test_json_exists(self):
        self.assertTrue(os.path.isfile(self.json_file))

    def test_dump_perf_metrics(self):
        hardware = sim_api.init_hardware(self.config)
        hardware.dump_perf_metrics(self.json_file)

        with open(self.json_file, "r") as f:
            perf_metrics = json.load(f)
            self.assertIn("hardware", perf_metrics)
            self.assertEqual(len(perf_metrics), 1)

    def test_dump_timeline(self):
        hardware = sim_api.init_hardware(self.config) 
        hardware.dump_timeline(self.json_file)

        with open(self.json_file, "r") as f:
            timeline = json.load(f)
            self.assertIn("traceEvents", timeline)
            self.assertEqual(len(timeline["traceEvents"]), 0)


if __name__ == "__main__":
    unittest.main() 
