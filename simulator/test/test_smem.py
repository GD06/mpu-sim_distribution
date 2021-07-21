#!/usr/bin/env python3

import unittest
import logging
import simpy
import struct
import random

import config.config_api as config_api
from simulator.shared_memory import SMEMReadReq, SMEMReadResp, \
    SMEMWriteReq, SMEMWriteResp, SharedMemory


class TestSharedMemoryClass(unittest.TestCase):

    def setUp(self):
        self.config = config_api.load_hardware_config()
        # current allocation requires certain smem address mapping
        self.assertEqual(self.config["smem_addr_map"], 
                         "smem_addr_map_1")
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
        self.smem = SharedMemory(
            env=self.env,
            log=logger,
            config=self.config,
            clock_unit=self.core_clock_unit
        )

        return

    def _compose_data(self, data_width):
        raw_data = []
        bytearray_data = None
        for i in range(self.config["num_threads_per_warp"]):
            raw_data.append(random.randrange(100))
        # pack as vector
        if data_width == 4:
            bytearray_data = bytearray(
                struct.pack(
                    "{}i".format(self.config["num_threads_per_warp"]),
                    *raw_data
                )
            )
        elif data_width == 8:
            bytearray_data = bytearray(
                struct.pack(
                    "{}q".format(self.config["num_threads_per_warp"]),
                    *raw_data
                )
            )
        else:
            assert False, "Wrong data width: {}".format(data_width)
        return (bytearray_data, raw_data)

    def _unpack_data(self, bytearray_data, data_width):
        data = None
        if data_width == 4:
            data = struct.unpack(
                "{}i".format(self.config["num_threads_per_warp"]),
                bytearray_data
            )
        elif data_width == 8:
            data = struct.unpack(
                "{}q".format(self.config["num_threads_per_warp"]),
                bytearray_data
            )
        else:
            assert False, "Wrong data width: {}".format(data_width)
        return data

    def _compose_addr(self, mode, data_width):
        addr_list = []
        if mode == "sequential":
            cur_addr = 0
            for i in range(self.config["num_threads_per_warp"]):
                addr_list.append(cur_addr)
                cur_addr += data_width
        elif mode == "random":
            for i in range(self.config["num_threads_per_warp"]):
                cur_addr = random.randrange(
                    self.smem.size
                )
                cur_addr = cur_addr - cur_addr % data_width
                while cur_addr in addr_list:
                    cur_addr = random.randrange(
                        self.smem.size
                    )
                    cur_addr = cur_addr - cur_addr % data_width
                addr_list.append(cur_addr)
        elif mode == "single_bank":
            # for testing only
            bank_index = 0
            bank_addr_range = self.smem.size // self.smem.alignment \
                // self.smem.num_bank
            for i in range(self.config["num_threads_per_warp"]):
                bank_addr = random.randrange(bank_addr_range)
                cur_addr = (bank_addr * (self.smem.num_bank) + bank_index) \
                    * self.smem.alignment
                while cur_addr in addr_list:
                    bank_addr = random.randrange(bank_addr_range)
                    cur_addr = (bank_addr * (self.smem.num_bank) + bank_index) \
                        * self.smem.alignment
                addr_list.append(cur_addr)
        elif mode == "single_addr":
            # for testing only
            cur_addr = 0
            for i in range(self.config["num_threads_per_warp"]):
                addr_list.append(cur_addr)
        else:
            assert False, "Wrong mode: {}".format(mode)
        return addr_list

    def _compose_simt_mask(self, mode):
        simt_mask = 0
        if mode == "uniform":
            for i in range(self.config["num_threads_per_warp"]):
                simt_mask += 1 << i
        else:
            assert False, "Wrong mode: {}".format(mode)
        return simt_mask

    def _process_single_write_read(self, data_width, addr_mode, 
                                   simt_mode):
        bytearray_data, self.raw_data = self._compose_data(data_width)
        addr_list = self._compose_addr(addr_mode, data_width)
        simt_mask = self._compose_simt_mask(simt_mode)
        # compose a write request
        smem_write_req = SMEMWriteReq(
            smem_addr_list=addr_list,
            data_width=data_width,
            data_buffer=bytearray_data,
            simt_mask=simt_mask
        )
        # issue write request
        yield self.smem.req_queue.put(smem_write_req)
        # get response
        _ = yield self.smem.resp_queue.get(
            lambda x: (
                isinstance(x, SMEMWriteResp)
                and x.smem_addr_list == addr_list
            )
        )
        # compose a read request
        smem_read_req = SMEMReadReq(
            smem_addr_list=addr_list,
            data_width=data_width,
            simt_mask=simt_mask
        )
        # issue read request
        yield self.smem.req_queue.put(smem_read_req)
        # get response
        smem_read_resp = yield self.smem.resp_queue.get(
            lambda x: (
                isinstance(x, SMEMReadResp)
                and x.smem_addr_list == addr_list
            )
        )
        data_buffer = smem_read_resp.data_buffer
        self.recv_data = self._unpack_data(data_buffer, data_width)

    def _process_single_read(self, data_width, addr_mode,
                             simt_mode):
        bytearray_data, self.raw_data = self._compose_data(data_width)
        addr_list = self._compose_addr(addr_mode, data_width)
        simt_mask = self._compose_simt_mask(simt_mode)
        # compose a read request
        smem_read_req = SMEMReadReq(
            smem_addr_list=addr_list,
            data_width=data_width,
            simt_mask=simt_mask
        )
        # issue read request
        yield self.smem.req_queue.put(smem_read_req)
        # get response
        smem_read_resp = yield self.smem.resp_queue.get(
            lambda x: (
                isinstance(x, SMEMReadResp)
                and x.smem_addr_list == addr_list
            )
        )
        data_buffer = smem_read_resp.data_buffer
        self.recv_data = self._unpack_data(data_buffer, data_width)

    def _consume_read(self, addr_list):
        _ = yield self.smem.resp_queue.get(
            lambda x: (
                isinstance(x, SMEMReadResp)
                and x.smem_addr_list == addr_list
            )
        )

    def _consume_write(self, addr_list):
        _ = yield self.smem.resp_queue.get(
            lambda x: (
                isinstance(x, SMEMWriteResp)
                and x.smem_addr_list == addr_list
            )
        )

    def _process_multiple_access(self, num_req):
        simt_mode = "uniform"
        addr_mode = "random"
        is_read = False
        data_width = 4

        for i in range(num_req):
            # randomize read / write
            if random.randrange(2) > 0:
                is_read = True
            else:
                is_read = False
            # randomize data width
            if random.randrange(2) > 0:
                data_width = 4
            else:
                data_width = 8

            bytearray_data, self.raw_data = self._compose_data(data_width)
            addr_list = self._compose_addr(addr_mode, data_width)
            simt_mask = self._compose_simt_mask(simt_mode)

            if is_read:
                # compose a read request
                smem_read_req = SMEMReadReq(
                    smem_addr_list=addr_list,
                    data_width=data_width,
                    simt_mask=simt_mask
                )
                # issue read request
                yield self.smem.req_queue.put(smem_read_req)
                self.env.process(
                    self._consume_read(addr_list)
                )
            else:
                # compose a write request
                smem_write_req = SMEMWriteReq(
                    smem_addr_list=addr_list,
                    data_width=data_width,
                    data_buffer=bytearray_data,
                    simt_mask=simt_mask
                )
                # issue write request
                yield self.smem.req_queue.put(smem_write_req)
                self.env.process(
                    self._consume_write(addr_list)
                )

            yield self.env.timeout(1 * self.core_clock_unit)

    def test_single_write_read_sequential_uniform_int32(self):
        # This test first write to the shared memory using interleaved
        # sequential addresses, then read from these locations.
        data_width = 4
        addr_mode = "sequential"
        simt_mode = "uniform"
        self.env.process(
            self._process_single_write_read(
                data_width=data_width, 
                addr_mode=addr_mode,
                simt_mode=simt_mode
            )
        )
        # run simulation
        self.env.run()
        # verify content
        self.assertEqual(len(self.raw_data), len(self.recv_data))
        for i in range(len(self.raw_data)):
            self.assertEqual(self.raw_data[i], self.recv_data[i])

    def test_single_write_read_random_uniform_int32(self):
        # This test first write to the shared memory using
        # random addresses, then read from these locations.
        data_width = 4
        addr_mode = "random"
        simt_mode = "uniform"
        self.env.process(
            self._process_single_write_read(
                data_width=data_width,
                addr_mode=addr_mode,
                simt_mode=simt_mode
            )
        )
        # run simulation
        self.env.run()
        # verify content
        self.assertEqual(len(self.raw_data), len(self.recv_data))
        for i in range(len(self.raw_data)):
            self.assertEqual(self.raw_data[i], self.recv_data[i])

    def test_single_write_read_sequential_uniform_int64(self):
        # This test first write to the shared memory using interleaved
        # sequential addresses, then read from these locations.
        data_width = 8
        addr_mode = "sequential"
        simt_mode = "uniform"
        self.env.process(
            self._process_single_write_read(
                data_width=data_width,
                addr_mode=addr_mode,
                simt_mode=simt_mode
            )
        )
        # run simulation
        self.env.run()
        # verify content
        self.assertEqual(len(self.raw_data), len(self.recv_data))
        for i in range(len(self.raw_data)):
            self.assertEqual(self.raw_data[i], self.recv_data[i])

    def test_single_write_read_random_uniform_int64(self):
        # This test first write to the shared memory using
        # random addresses, then read from these locations.
        data_width = 8
        addr_mode = "random"
        simt_mode = "uniform"
        self.env.process(
            self._process_single_write_read(
                data_width=data_width,
                addr_mode=addr_mode,
                simt_mode=simt_mode
            )
        )
        # run simulation
        self.env.run()
        # verify content
        self.assertEqual(len(self.raw_data), len(self.recv_data))
        for i in range(len(self.raw_data)):
            self.assertEqual(self.raw_data[i], self.recv_data[i])

    def test_single_access_interleaved_uniform_int32(self):
        # This test first write to the shared memory using interleaved
        # sequential addresses, then read from these locations.
        data_width = 4
        addr_mode = "sequential"
        simt_mode = "uniform"
        self.env.process(
            self._process_single_write_read(
                data_width=data_width,
                addr_mode=addr_mode,
                simt_mode=simt_mode
            )
        )
        # run simulation
        start_time = self.env.now
        self.env.run()
        end_time = self.env.now
        # verify timing
        pipeline_overhead = 2
        dur = end_time - start_time
        est_dur = (
            self.config["smem_read_latency"]
            + self.config["smem_write_latency"]
            + pipeline_overhead
        ) * self.core_clock_unit
        self.assertEqual(dur, est_dur)
        # verify content
        self.assertEqual(len(self.raw_data), len(self.recv_data))
        for i in range(len(self.raw_data)):
            self.assertEqual(self.raw_data[i], self.recv_data[i])

    def test_single_access_conflict_uniform_int32(self):
        # This test first write to the shared memory using a single
        # address, then read from these locations.
        data_width = 4
        addr_mode = "single_bank"
        simt_mode = "uniform"
        self.env.process(
            self._process_single_write_read(
                data_width=data_width,
                addr_mode=addr_mode,
                simt_mode=simt_mode
            )
        )
        # run simulation
        start_time = self.env.now
        self.env.run()
        end_time = self.env.now
        # verify timing
        dur = end_time - start_time
        simt_size = self.config["num_threads_per_warp"]
        est_dur = (
            self.config["smem_read_latency"] * simt_size
            + self.config["smem_write_latency"] * simt_size
        ) * self.core_clock_unit
        self.assertEqual(dur, est_dur)
        # verify content
        self.assertEqual(len(self.raw_data), len(self.recv_data))
        for i in range(len(self.raw_data)):
            self.assertEqual(self.raw_data[i], self.recv_data[i])

    def test_single_broadcast_int32(self):
        # This test first write to the shared memory using a single
        # address, then read from these locations.
        data_width = 4
        addr_mode = "single_addr"
        simt_mode = "uniform"
        self.env.process(
            self._process_single_read(
                data_width=data_width,
                addr_mode=addr_mode,
                simt_mode=simt_mode
            )
        )
        # run simulation
        start_time = self.env.now
        self.env.run()
        end_time = self.env.now
        # verify timing
        pipeline_overhead = 1
        dur = end_time - start_time
        est_dur = (
            self.config["smem_read_latency"] + pipeline_overhead
        ) * self.core_clock_unit
        self.assertEqual(dur, est_dur)
        # verify content
        for i in range(len(self.recv_data) - 1):
            self.assertEqual(
                self.recv_data[i], 
                self.recv_data[i + 1]
            )

    def test_multiple_access_uniform(self):
        num_req = 100
        self.env.process(
            self._process_multiple_access(num_req)
        )
        # run simulation
        self.env.run()


if __name__ == "__main__":
    unittest.main()
