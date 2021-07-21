#!/usr/bin/env python3 

import unittest 
import os
import json 

import config.config_api as config_api 


class TestLoadConfigClass(unittest.TestCase):

    def setUp(self):
        self.curr_dir = os.path.dirname(os.path.realpath(__file__))
        self.proj_dir = os.path.dirname(os.path.dirname(self.curr_dir))
        self.config_dict = config_api.load_hardware_config() 

    def _load_and_compare_config(self, config_file_path):
        with open(config_file_path, "r") as f:
            local_config_dict = json.load(f)

        for config_name, config_value in local_config_dict.items():
            self.assertEqual(self.config_dict[config_name], config_value)

        return 

    def test_hardware_config(self):
        hardware_config_file = os.path.join(
            self.proj_dir, "config", "hardware_config.json")
        self._load_and_compare_config(hardware_config_file) 

    def test_processor_config(self):
        processor_config_file = os.path.join(
            self.proj_dir, "config", "processor_config.json")
        self._load_and_compare_config(processor_config_file) 

    def test_core_config(self):
        core_config_file = os.path.join(
            self.proj_dir, "config", "core_config.json")
        self._load_and_compare_config(core_config_file) 

    def test_dram_config(self):
        dram_config_file = os.path.join(
            self.proj_dir, "config", "dram_config.json")
        self._load_and_compare_config(dram_config_file)

    def test_execution_unit_config(self):
        execution_unit_config_file = os.path.join(
            self.proj_dir, "config", "execution_unit_config.json")
        self._load_and_compare_config(execution_unit_config_file)

    def test_register_file_config(self):
        register_file_config_file = os.path.join(
            self.proj_dir, "config", "register_file_config.json")
        self._load_and_compare_config(register_file_config_file)

    def test_shared_memory_config(self):
        shared_memory_config_file = os.path.join(
            self.proj_dir, "config", "shared_memory_config.json")
        self._load_and_compare_config(shared_memory_config_file)

    def test_load_store_unit_config(self):
        load_store_unit_config_file = os.path.join(
            self.proj_dir, "config", "load_store_unit_config.json")
        self._load_and_compare_config(load_store_unit_config_file)

    def test_interconnect_network_config(self):
        interconnect_network_config_file = os.path.join(
            self.proj_dir, "config", "interconnect_network_config.json")
        self._load_and_compare_config(interconnect_network_config_file)

    def test_simulation_config(self):
        simulator_config_file = os.path.join(
            self.proj_dir, "config", "simulation_config.json")
        self._load_and_compare_config(simulator_config_file)


if __name__ == "__main__":
    unittest.main() 
