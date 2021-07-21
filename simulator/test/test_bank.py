#!/usr/bin/env python3

import unittest
import numpy as np
import random
import logging

import config.config_api as config_api
import simulator.sim_api as sim_api
from simulator.dram_message import DRAMTransaction


class TestBankClass(unittest.TestCase):

    def setUp(self):
        self.config_dict = config_api.load_hardware_config()
        self.dram_clock_unit = (self.config_dict["sim_clock_freq"] 
                                // self.config_dict["dram_clock_freq"])

    def _init_hardware(self):
        self.element_in_bank = 8  # each element is dram_bank_io_width bytes
        self.hardware = sim_api.init_hardware(self.config_dict)
        self.length = self.hardware.mem.alignment * self.element_in_bank

        # current allocation requires bank address on the front
        self.assertEqual(self.config_dict["dram_addr_map"], "dram_addr_map_1")
        ptr = self.hardware.mem.allocate(self.length * 4)
        self.hardware.mem.finalize() 

        self.data = np.random.rand(self.length).astype(np.float32)
        self.hardware.mem.set_value(ptr, self.data.tobytes()) 

    def _run_each_bank_transaction(self, loc, trans_list):
        # (proc_id_x, proc_id_y)
        proc_id = (loc[1], loc[0]) 
        processor = self.hardware.processor_array[proc_id]

        # (core_id_x, core_id_y)
        core_id = (loc[3], loc[2])
        core = processor.core_array[core_id]

        pg_id = loc[4]
        pe_id = loc[5]
        cur_bank = core.pg_array[pg_id].pe_array[pe_id].bank
        trans_token_queue = cur_bank.mem_trans_token_queue
        trans_queue = cur_bank.mem_trans_queue

        for each_trans in trans_list:
            if each_trans.time * cur_bank.clock_unit > cur_bank.env.now:
                yield cur_bank.env.timeout(
                    each_trans.time * cur_bank.clock_unit - cur_bank.env.now
                )
            trans_queue.append(each_trans)
            yield trans_token_queue.put(1)
        return 

    def _run_memory_transaction(self, trans_list):
        trans_dict = {}
        for each_trans in trans_list:
            loc = each_trans.get_mem_loc()
            if loc not in trans_dict:
                trans_dict[loc] = []
            trans_dict[loc].append(each_trans) 

        for each_loc, each_trans_list in trans_dict.items():
            self.hardware.env.process(
                self._run_each_bank_transaction(each_loc, each_trans_list)
            )

        start_cycle = self.hardware.env.now 
        self.hardware.env.run()
        end_cycle = self.hardware.env.now 

        dur = end_cycle - start_cycle 
        return dur 

    def _gen_mem_loc(self, access_pattern, last_loc):
        """return a generated memory location
        Args:
            access_pattern: memory access pattern 
            last_loc: a location tuple formatted as
                (proc_id_y, proc_id_x, core_id_y, core_id_x, pg_id, pe_id, 
                bank_addr, bank_interface_offset)
        Return:
            loc: a location tuple formatted as
                (proc_id_y, proc_id_x, core_id_y, core_id_x, pg_id, pe_id, 
                bank_addr, bank_interface_offset)
        """
        if access_pattern == "random":
            # random access
            proc_id_y = random.randrange(self.config_dict["num_processor_y"])
            proc_id_x = random.randrange(self.config_dict["num_processor_x"])
            core_id_y = random.randrange(self.config_dict["num_core_y"])
            core_id_x = random.randrange(self.config_dict["num_core_x"])
            pg_id = random.randrange(self.config_dict["num_pg"])
            pe_id = random.randrange(self.config_dict["num_pe"])
            bank_addr = random.randrange(self.element_in_bank)
            bank_interface_offset = random.randrange(
                self.config_dict["dram_bank_io_width"])
            return (proc_id_y, proc_id_x, core_id_y, core_id_x, pg_id, pe_id, 
                    bank_addr, bank_interface_offset)
        elif access_pattern == "sequential":
            # sequential access
            addr = self.hardware.re_addr_hashing(last_loc)
            addr = addr + self.config_dict["dram_bank_io_width"]
            loc = self.hardware.addr_hashing(addr)
            return loc
        elif access_pattern == "samebank_random":
            # random accesses to the same bank
            loc = (last_loc[0], last_loc[1], last_loc[2],
                   last_loc[3], last_loc[4], last_loc[5],
                   random.randrange(self.element_in_bank),
                   last_loc[7])
            return loc
        else:
            raise NotImplementedError(
                "Unrecognized memory access pattern: {}".format(access_pattern)
            )

    def _decode_addr(self, mem_loc):
        row_addr, col_addr = self.hardware.translate_bank_addr(
            mem_loc[6]
        )
        global_mem_addr = self.hardware.re_addr_hashing(
            (
                mem_loc[0], mem_loc[1], mem_loc[2], mem_loc[3], mem_loc[4],
                mem_loc[5], mem_loc[6], 0
            )
        )
        return row_addr, col_addr, global_mem_addr

    def _gen_mem_trans(self, access_pattern, access_type, ratio, num_trans, 
                       trans_interval):
        """generate memory transaction list according to configuration

        Args:
            access_pattern: memory access pattern
            access_type: access type (all_load, all_store, or mix)
            ratio: ratio of loads in total acesses 
                (all_load=1, all_store=0, mix=(0,1))
            num_trans: the number of transactions generated
            trans_interval: the dram cycles between each generated transaction
        
        Return:
            trans_list: the transaction list generated
        """
        trans_list = []
        cycle = 1
        # Dummy data
        payload = bytearray(self.config_dict["dram_bank_io_width"])
        # initialize global address tuple
        last_mem_loc = (0, 0, 0, 0, 0, 0, 0, 0)          
        random.seed(1)  # give a seed for random address generation
        self.assertTrue(ratio <= 1)
        self.assertTrue(ratio >= 0)

        while num_trans > 0:
            mem_loc = self._gen_mem_loc(access_pattern, last_mem_loc)
            row_addr, col_addr, global_mem_addr = self._decode_addr(mem_loc)
            if access_type == "all_load":
                dram_trans = DRAMTransaction(
                    trans_type="load",
                    mem_loc=mem_loc,
                    row_addr=row_addr,
                    col_addr=col_addr,
                    global_mem_addr=global_mem_addr
                )
                dram_trans.time = cycle
                trans_list.append(dram_trans)
            elif access_type == "all_store":
                dram_trans = DRAMTransaction(
                    trans_type="store",
                    mem_loc=mem_loc,
                    row_addr=row_addr,
                    col_addr=col_addr,
                    global_mem_addr=global_mem_addr
                )
                dram_trans.time = cycle
                dram_trans.data = payload
                trans_list.append(dram_trans)
            elif access_type == "mix":
                if random.randrange(1000) <= 1000 * ratio:
                    dram_trans = DRAMTransaction(
                        trans_type="load",
                        mem_loc=mem_loc,
                        row_addr=row_addr,
                        col_addr=col_addr,
                        global_mem_addr=global_mem_addr
                    )
                    dram_trans.time = cycle
                    trans_list.append(dram_trans)
                else:
                    dram_trans = DRAMTransaction(
                        trans_type="store",
                        mem_loc=mem_loc,
                        row_addr=row_addr,
                        col_addr=col_addr,
                        global_mem_addr=global_mem_addr
                    )
                    dram_trans.time = cycle
                    dram_trans.data = payload
                    trans_list.append(dram_trans)
            else:
                raise NotImplementedError(
                    "Unrecognized access type: {}".format(access_type)
                )
            last_mem_loc = mem_loc
            num_trans = num_trans - 1
            cycle = cycle + trans_interval
        return trans_list
    
    def test_single_load_store(self):
        self._init_hardware()
        access_pattern = "sequential"
        access_type = "all_load"
        ratio = 1
        num_trans = 1 
        trans_interval = 2
        logging.info("+ Testing single load ...")
        trans_list = self._gen_mem_trans(access_pattern, access_type, ratio, 
                                         num_trans, trans_interval)
        _ = self._run_memory_transaction(trans_list)
        logging.info("+ Testing single store ...")
        access_type = "all_store"
        ratio = 0
        trans_list = self._gen_mem_trans(access_pattern, access_type, ratio, 
                                         num_trans, trans_interval)
        _ = self._run_memory_transaction(trans_list)
        return

    def test_sequential_load_store(self):
        self._init_hardware()
        access_pattern = "sequential"
        access_type = "all_load"
        ratio = 1
        num_trans = 100
        trans_interval = 2
        logging.info("+ Testing sequential loads ...")
        trans_list = self._gen_mem_trans(access_pattern, access_type, ratio, 
                                         num_trans, trans_interval)
        _ = self._run_memory_transaction(trans_list)
        logging.info("+ Testing sequential stores ...")
        access_type = "all_store"
        ratio = 0
        trans_list = self._gen_mem_trans(access_pattern, access_type, ratio, 
                                         num_trans, trans_interval)
        _ = self._run_memory_transaction(trans_list)
        return

    def test_addr_hash(self):
        self._init_hardware()
        random.seed(100)
        addr = random.randrange(self.config_dict["dram_capacity"])
        loc = self.hardware.addr_hashing(addr)
        addr_2 = self.hardware.re_addr_hashing(loc)
        self.assertEqual(addr, addr_2)
        loc_2 = self.hardware.addr_hashing(addr_2)
        self.assertEqual(loc, loc_2)
        
    def test_random_load_store(self):
        self._init_hardware()
        access_pattern = "random"
        access_type = "all_load"
        ratio = 1
        num_trans = 100
        trans_interval = 2
        logging.info("+ Testing random loads ...")
        trans_list = self._gen_mem_trans(access_pattern, access_type, ratio, 
                                         num_trans, trans_interval)
        _ = self._run_memory_transaction(trans_list)
        logging.info("+ Testing random stores ...")
        access_type = "all_store"
        ratio = 0
        trans_list = self._gen_mem_trans(access_pattern, access_type, ratio, 
                                         num_trans, trans_interval)
        _ = self._run_memory_transaction(trans_list)
        return

    def test_random_mix(self):
        self._init_hardware()
        access_pattern = "random"
        access_type = "mix"
        ratio = 0.5
        num_trans = 100
        trans_interval = 2
        logging.info("+ Testing random mixed workloads ...")
        trans_list = self._gen_mem_trans(access_pattern, access_type, ratio, 
                                         num_trans, trans_interval)
        _ = self._run_memory_transaction(trans_list)
        return

    def test_sequential_mix(self):
        self._init_hardware()
        access_pattern = "sequential"
        access_type = "mix"
        ratio = 0.5
        num_trans = 100
        trans_interval = 2
        logging.info("+ Testing sequential mixed workloads ...")
        trans_list = self._gen_mem_trans(access_pattern, access_type, ratio, 
                                         num_trans, trans_interval)
        _ = self._run_memory_transaction(trans_list)
        return
    
    def test_samebank_random_load_store(self):
        self._init_hardware()
        access_pattern = "samebank_random"
        access_type = "all_load"
        ratio = 1
        num_trans = 1000
        trans_interval = 6
        logging.info("+ Testing random loads for the same bank...")
        trans_list = self._gen_mem_trans(access_pattern, access_type, ratio, 
                                         num_trans, trans_interval)
        dur = self._run_memory_transaction(trans_list)

        if self.config_dict["dram_controller"] == "ideal":
            if self.config_dict["dram_ideal_load_latency"] >= trans_interval:
                self.assertEqual(
                    dur, 
                    (1 + num_trans * self.config_dict["dram_ideal_load_latency"]
                     ) * self.dram_clock_unit
                )
            else:
                self.assertEqual(
                    dur, 
                    (1 + (num_trans - 1) * trans_interval 
                        + self.config_dict["dram_ideal_load_latency"]) 
                    * self.dram_clock_unit
                )

        logging.info("+ Testing random stores for the same bank...")
        self._init_hardware()
        access_type = "all_store"
        ratio = 0
        trans_list = self._gen_mem_trans(access_pattern, access_type, ratio, 
                                         num_trans, trans_interval)
        dur = self._run_memory_transaction(trans_list)
        if self.config_dict["dram_controller"] == "ideal":
            if self.config_dict["dram_ideal_store_latency"] >= trans_interval:
                self.assertEqual(
                    dur, 
                    (1 + num_trans 
                        * self.config_dict["dram_ideal_store_latency"])
                    * self.dram_clock_unit
                )
            else:
                self.assertEqual(
                    dur, 
                    (1 + (num_trans - 1) * trans_interval 
                        + self.config_dict["dram_ideal_store_latency"])
                    * self.dram_clock_unit
                )
        return


if __name__ == "__main__":
    unittest.main() 
