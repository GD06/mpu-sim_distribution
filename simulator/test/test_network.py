#!/usr/bin/env python3

import unittest
import numpy as np
import random
import struct

import config.config_api as config_api
import simulator.sim_api as sim_api
from simulator.network_message import SrcRemoteLoadReq, SrcRemoteLoadResp, \
    SrcRemoteStoreReq, SrcRemoteStoreResp


class TestNetworkClass(unittest.TestCase):

    def setUp(self):
        self.config = config_api.load_hardware_config()

    def _init_hardware(self):
        self.element_in_bank = 8  # each element is dram_bank_io_width bytes
        self.hardware = sim_api.init_hardware(self.config)
        self.length = self.hardware.mem.alignment * self.element_in_bank

        # current allocation requires bank address on the front
        self.assertEqual(self.config["dram_addr_map"], "dram_addr_map_1")
        ptr = self.hardware.mem.allocate(self.length * 4)
        self.hardware.mem.finalize()

        self.data = np.random.rand(self.length).astype(np.float32)
        self.hardware.mem.set_value(ptr, self.data.tobytes())

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

    def _compose_simt_mask(self, mode):
        simt_mask = 0
        if mode == "uniform":
            for i in range(self.config["num_threads_per_warp"]):
                simt_mask += 1 << i
        else:
            assert False, "Wrong mode: {}".format(mode)
        return simt_mask

    def _gen_mem_loc(self):
        proc_id_y = random.randrange(self.config["num_processor_y"])
        proc_id_x = random.randrange(self.config["num_processor_x"])
        core_id_y = random.randrange(self.config["num_core_y"])
        core_id_x = random.randrange(self.config["num_core_x"])
        while (proc_id_y + proc_id_x + core_id_y + core_id_x) == 0:
            # since the source loc is (0,0) (0,0), this can
            # avoid sending msg to itself
            proc_id_y = random.randrange(self.config["num_processor_y"])
            proc_id_x = random.randrange(self.config["num_processor_x"])
            core_id_y = random.randrange(self.config["num_core_y"])
            core_id_x = random.randrange(self.config["num_core_x"])
        # NOTE: for texting only
        pg_id = 0
        pe_id = 0
        bank_addr = random.randrange(self.element_in_bank)
        bank_interface_offset = 0
        loc = (proc_id_y, proc_id_x, core_id_y, core_id_x, pg_id, pe_id,
               bank_addr, bank_interface_offset)
        addr = self.hardware.re_addr_hashing(loc)
        return addr

    def _compose_addr(self, mode, data_width):
        addr_list = []
        if mode == "sequential":
            addr = self._gen_mem_loc()
            for i in range(self.config["num_threads_per_warp"]):
                addr_list.append(addr)
                addr += data_width
        elif mode == "random":
            for i in range(self.config["num_threads_per_warp"]):
                addr = self._gen_mem_loc()
                while addr in addr_list:
                    addr = self._gen_mem_loc()
                addr_list.append(addr)
        else:
            raise NotImplementedError(
                "Unknown address mode: {}".format(mode)
            )
        return addr_list

    def _process_st_ld(self, addr_mode, simt_mode, data_width):
        addr_list = self._compose_addr(addr_mode, data_width)
        simt_mask = self._compose_simt_mask(simt_mode)
        bytearray_data, self.raw_data = self._compose_data(data_width)
        # NOTE: assume the source process_id and core_id is fixed
        proc_id = (0, 0)
        core_id = (0, 0)
        src_core = self.hardware.processor_array[proc_id].core_array[core_id]
        # compose a store request
        st_req = SrcRemoteStoreReq(
            addr_list=addr_list,
            data_width=data_width,
            simt_mask=simt_mask,
            data=bytearray_data
        )
        # send the store request
        yield src_core.niu.req_queue.put(st_req)
        # wait until response
        _ = yield src_core.niu.resp_queue.get(
            lambda x: (
                isinstance(x, SrcRemoteStoreResp)
                and x.addr_list == st_req.addr_list
            )
        )

        # compose a load request
        ld_req = SrcRemoteLoadReq(
            addr_list=addr_list,
            data_width=data_width,
            simt_mask=simt_mask
        )
        # send the load request
        yield src_core.niu.req_queue.put(ld_req)
        # wait until response
        ld_resp = yield src_core.niu.resp_queue.get(
            lambda x: (
                isinstance(x, SrcRemoteLoadResp)
                and x.addr_list == ld_req.addr_list
            )
        )
        
        data_buffer = ld_resp.data
        self.recv_data = self._unpack_data(data_buffer, data_width)

    def _consume_load(self, ld_req, src_core):
        _ = yield src_core.niu.resp_queue.get(
            lambda x: (
                isinstance(x, SrcRemoteLoadResp)
                and x.addr_list, ld_req.addr_list
            )
        )

    def _consume_store(self, st_req, src_core):
        _ = yield src_core.niu.resp_queue.get(
            lambda x: (
                isinstance(x, SrcRemoteStoreResp)
                and x.addr_list, st_req.addr_list
            )
        )

    def _process_trans(self, trans_num, addr_mode, simt_mode, data_width):
        # NOTE: assume the source process_id and core_id is fixed
        proc_id = (0, 0)
        core_id = (0, 0)
        src_core = self.hardware.processor_array[proc_id].core_array[core_id]
        for trans_id in range(trans_num):
            addr_list = self._compose_addr(addr_mode, data_width)
            simt_mask = self._compose_simt_mask(simt_mode)
            bytearray_data, self.raw_data = self._compose_data(data_width)
            req = None
            if random.randrange(2) > 0:
                req = SrcRemoteStoreReq(
                    addr_list=addr_list,
                    data_width=data_width,
                    simt_mask=simt_mask,
                    data=bytearray_data
                )
                # spawn a process to consume the request
                self.hardware.env.process(self._consume_store(req, src_core))
            else:
                req = SrcRemoteLoadReq(
                    addr_list=addr_list,
                    data_width=data_width,
                    simt_mask=simt_mask
                )
                # spawn a process to consume the request
                self.hardware.env.process(self._consume_load(req, src_core))
            # send the request
            yield src_core.niu.req_queue.put(req)

    def test_remote_st_ld_sequential_location_uniform_int32(self):
        # first issue a remote store from source cube to a remote cube
        # then issue a remote load from source cube to a remote cube
        self._init_hardware()
        addr_mode = "sequential"
        simt_mode = "uniform"
        data_width = 4
        self.hardware.env.process(
            self._process_st_ld(
                addr_mode=addr_mode,
                simt_mode=simt_mode,
                data_width=data_width
            )
        )
        # run simulation
        self.hardware.env.run()
        # verify content
        self.assertEqual(len(self.raw_data), len(self.recv_data))
        for i in range(len(self.raw_data)):
            self.assertEqual(self.raw_data[i], self.recv_data[i])

    def test_remote_st_ld_random_location_uniform_int32(self):
        # first issue a remote store from source cube to 
        # a number of remote cubes
        # then issue a remote load from source cube to 
        # a number of remote cubes
        self._init_hardware()
        addr_mode = "random"
        simt_mode = "uniform"
        data_width = 4
        self.hardware.env.process(
            self._process_st_ld(
                addr_mode=addr_mode,
                simt_mode=simt_mode,
                data_width=data_width
            )
        )
        # run simulation
        self.hardware.env.run()
        # verify content
        self.assertEqual(len(self.raw_data), len(self.recv_data))
        for i in range(len(self.raw_data)):
            self.assertEqual(self.raw_data[i], self.recv_data[i])
    
    def test_remote_accesses_random_location_uniform_int32(self):
        self._init_hardware()
        addr_mode = "random"
        simt_mode = "uniform"
        data_width = 4
        trans_num = 100
        self.hardware.env.process(
            self._process_trans(
                trans_num=trans_num,
                addr_mode=addr_mode,
                simt_mode=simt_mode,
                data_width=data_width
            )
        )
        # run simulation
        self.hardware.env.run()


if __name__ == "__main__":
    unittest.main()
