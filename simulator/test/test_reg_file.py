#!/usr/bin/env python3

import unittest
import logging
import simpy
import struct

import config.config_api as config_api
from simulator.register_file import RegisterFile, RegReadReq, \
    RegReadResp, RegWriteReq


class TestRegFileClass(unittest.TestCase):

    def setUp(self):
        self.config = config_api.load_hardware_config()
        # current allocation requires certain register file address mapping
        self.assertEqual(self.config["reg_file_addr_map"], 
                         "reg_file_addr_map_1")
        # setup required environment components
        self.env = simpy.Environment()
        self.core_clock_unit = (self.config["sim_clock_freq"]
                                // self.config["core_clock_freq"])
        # setup logger
        logging_level = logging.ERROR
        logger = logging.getLogger(__name__)
        logger.setLevel(logging_level)
        ch = logging.StreamHandler()
        ch.setLevel(logging_level)
        logger.addHandler(ch)

        # initialize register file
        self.reg_file = RegisterFile(
            env=self.env,
            log=logger,
            config=self.config,
            clock_unit=self.core_clock_unit,
            reg_file_type="far-bank"
        )

        return

    def _consume_read_req(self):
        while True:
            _ = yield self.reg_file.read_resp_queue.get()

    def _compose_write_data(self):
        raw_data = []
        for i in range(self.config["num_threads_per_warp"]):
            raw_data.append(i)
        # pack as integer vector
        bytearray_data = bytearray(
            struct.pack(
                "{}i".format(self.config["num_threads_per_warp"]),
                *raw_data
            )
        )
        return (bytearray_data, raw_data)

    def _process_single_write_read(self):
        # compose a write request
        reg_addr = 0
        bytearray_data, self.raw_data = self._compose_write_data()
        # enqueue write request
        bank_index = self.reg_file.calc_bank_index(reg_addr)
        req = RegWriteReq(
            reg_addr=reg_addr,
            data=bytearray_data
        )
        yield self.reg_file.write_req_queue[bank_index].put(req)
        # wait until the write has been processed to enforce RAW dependency
        # NOTE: the dependency is tracked by dep_table_exe in 
        # actual implementation
        _ = yield self.reg_file.write_resp_queue.get(
            lambda x: x.reg_addr == reg_addr
        )
        # compose a read request
        req = RegReadReq(reg_addr=reg_addr)
        # enqueue read request
        yield self.reg_file.read_req_queue[bank_index].put(req)
        resp = yield self.reg_file.read_resp_queue.get()
        assert isinstance(resp, RegReadResp)
        resp_bytearray = resp.data
        assert isinstance(resp_bytearray, bytearray)
        self.resp_data = struct.unpack(
            "{}i".format(self.config["num_threads_per_warp"]),
            resp_bytearray
        )
        return

    def test_single_read_write(self):
        # This test first write a register location, 
        # then read from that location.
        self.env.process(self._process_single_write_read())
        # verify latency
        start_cycle = self.env.now
        self.env.run()
        end_cycle = self.env.now
        dur = end_cycle - start_cycle
        t_rd = self.config["subcore_reg_file_read_latency"]
        t_wr = self.config["subcore_reg_file_write_latency"]
        est_dur = (t_rd + t_wr) * self.core_clock_unit
        self.assertEqual(dur, est_dur)
        # verify content
        self.assertEqual(len(self.raw_data), len(self.resp_data))
        for i in range(len(self.raw_data)):
            self.assertEqual(self.raw_data[i], self.resp_data[i])

    def _process_same_bank_access(self, num_read_req):
        # compose a write request
        reg_addr = 0
        bytearray_data, self.raw_data = self._compose_write_data()
        # enqueue write request
        bank_index = self.reg_file.calc_bank_index(reg_addr)
        req = RegWriteReq(
            reg_addr=reg_addr,
            data=bytearray_data
        )
        yield self.reg_file.write_req_queue[bank_index].put(req)
        # wait until the write has been processed to enforce RAW dependency
        # NOTE: the dependency is tracked by dep_table_exe in 
        # actual implementation
        _ = yield self.reg_file.write_resp_queue.get(
            lambda x: x.reg_addr == reg_addr
        )
        for i in range(num_read_req):
            # compose a read request
            req = RegReadReq(reg_addr=reg_addr)
            # enqueue read request
            yield self.reg_file.read_req_queue[bank_index].put(req)

    def test_same_bank_access(self):
        # This test first write a register location, 
        # then issues multiple read requests to that location. 
        # This simulates register bank conflicts.
        num_read_req = 5
        self.env.process(self._process_same_bank_access(num_read_req))
        self.env.process(self._consume_read_req())
        # verify latency
        start_cycle = self.env.now
        self.env.run()
        end_cycle = self.env.now
        dur = end_cycle - start_cycle
        t_rd = self.config["subcore_reg_file_read_latency"]
        t_wr = self.config["subcore_reg_file_write_latency"]
        est_dur = (t_wr + t_rd * num_read_req) * self.core_clock_unit
        self.assertEqual(dur, est_dur)

    def _issue_read_request(self, reg_addr, num_read_req_each_bank):
        # wait until the write has been processed to enforce RAW dependency
        # NOTE: the dependency is tracked by dep_table_exe in 
        # actual implementation
        _ = yield self.reg_file.write_resp_queue.get(
            lambda x: x.reg_addr == reg_addr
        )
        # compose read requests
        for i in range(num_read_req_each_bank):
            # compose a read request
            bank_index = self.reg_file.calc_bank_index(reg_addr)
            req = RegReadReq(reg_addr=reg_addr)
            # enqueue read request
            yield self.reg_file.read_req_queue[bank_index].put(req)

    def test_multiple_banks_access(self):
        # This test writes a location in each bank of the register file. 
        # Then it issues equivalent number of read requests 
        # to each bank of the register file.
        self.env.process(self._consume_read_req())
        num_read_req_each_bank = 3
        alignment = self.config["data_path_unit_size"] \
            * self.config["num_threads_per_warp"]
        for i in range(self.config["num_subcore_reg_file_bank"]):
            # compose a write request
            reg_addr = i * alignment
            bytearray_data, raw_data = self._compose_write_data()
            # enqueue write request
            bank_index = self.reg_file.calc_bank_index(reg_addr)
            req = RegWriteReq(
                reg_addr=reg_addr,
                data=bytearray_data
            )
            yield self.reg_file.write_req_queue[bank_index].put(req)
            self.env.process(
                self._issue_read_request(reg_addr, num_read_req_each_bank)
            )
        # verify latency
        start_cycle = self.env.now
        self.env.run()
        end_cycle = self.env.now
        dur = end_cycle - start_cycle
        t_rd = self.config["subcore_reg_file_read_latency"]
        t_wr = self.config["subcore_reg_file_write_latency"]
        est_dur = (t_wr + t_rd * num_read_req_each_bank) * self.core_clock_unit
        self.assertEqual(dur, est_dur)


if __name__ == "__main__":
    unittest.main()
