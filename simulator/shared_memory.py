import simpy
import struct 
from copy import deepcopy


class SMEMReadReq:

    def __init__(self, smem_addr_list, data_width, simt_mask):
        self.smem_addr_list = smem_addr_list
        self.data_width = data_width
        self.simt_mask = simt_mask
        return 


class SMEMReadResp:

    def __init__(self, smem_addr_list, data_width, data_buffer, simt_mask):
        self.smem_addr_list = smem_addr_list
        self.data_width = data_width
        self.data_buffer = data_buffer
        self.simt_mask = simt_mask
        return


class SMEMWriteReq:

    def __init__(self, smem_addr_list, data_width, data_buffer, simt_mask):
        self.smem_addr_list = smem_addr_list
        self.data_width = data_width
        self.data_buffer = data_buffer
        self.simt_mask = simt_mask
        return


class SMEMWriteResp:

    def __init__(self, smem_addr_list, data_width, simt_mask):
        self.smem_addr_list = smem_addr_list
        self.data_width = data_width
        self.simt_mask = simt_mask
        return


class SMEMAtomReq:

    def __init__(self, smem_addr_list, val_list, data_width, op, simt_mask):
        self.smem_addr_list = smem_addr_list 
        self.val_list = val_list 
        self.data_width = data_width 
        self.op = op 
        self.simt_mask = simt_mask
        return 


class SMEMAtomResp:

    def __init__(self, smem_addr_list, val_list, data_width, op, simt_mask):
        self.smem_addr_list = smem_addr_list 
        self.val_list = val_list 
        self.data_width = data_width 
        self.op = op 
        self.simt_mask = simt_mask 
        return 


class SMEMBankReadReq:

    def __init__(self, smem_addr):
        self.smem_addr = smem_addr
        return


class SMEMBankReadResp:

    def __init__(self, smem_addr, data):
        self.smem_addr = smem_addr
        self.data = data
        return


class SMEMBankWriteReq:

    def __init__(self, smem_addr, data):
        self.smem_addr = smem_addr
        self.data = data
        return


class SMEMBankWriteResp:

    def __init__(self, smem_addr):
        self.smem_addr = smem_addr
        return


class SMEMBankAtomReq:

    def __init__(self, smem_addr, val, op):
        self.smem_addr = smem_addr 
        self.val = val 
        self.op = op
        return 


class SMEMBankAtomResp:

    def __init__(self, smem_addr, val, op):
        self.smem_addr = smem_addr 
        self.val = val
        self.op = op
        return 


class SharedMemory:
    
    def __init__(self, env, log, config, clock_unit):
        self.env = env
        self.log = log
        self.config = config
        self.clock_unit = clock_unit
        
        self.size = self.config["smem_size"] 
        self.array = bytearray(self.size)
        self.alignment = self.config["smem_io_width"]
        self.read_latency = self.config["smem_read_latency"]
        self.write_latency = self.config["smem_write_latency"]

        self.req_queue_size = self.config["smem_req_queue_size"]
        self.req_queue = simpy.Store(env, capacity=self.req_queue_size)
        self.resp_queue = simpy.FilterStore(env)
        
        self.num_bank = self.config["num_smem_bank"]
        self._bank_req_queue = []
        self._bank_resp_queue = simpy.FilterStore(env)
        self.bank_queue_size = self.config["smem_bank_req_queue_size"]
        for i in range(self.num_bank):
            self._bank_req_queue.append(
                simpy.Store(env, capacity=self.bank_queue_size)
            )

        # spawn a process to handle request
        self.env.process(self._handle_request())
        # spawn processes for each bank to handle request
        for i in range(self.num_bank):
            self.env.process(self._serve_bank_request(i))
        
        # performance counter
        self.num_smem_bank_read = 0
        self.num_smem_bank_write = 0
        
        return 

    def get_perf_metrics(self):
        """Get a dictionary of performance metrics"""
        perf_metrics = {}
        perf_metrics["num_smem_bank_read"] = self.num_smem_bank_read
        perf_metrics["num_smem_bank_write"] = self.num_smem_bank_write
        return {"smem": perf_metrics}

    def reset(self):
        self.array = bytearray(self.size)

    def _delay_bank_response(self, resp, delay):
        yield self.env.timeout(delay * self.clock_unit)
        yield self._bank_resp_queue.put(resp)

    def _serve_bank_request(self, bank_id):
        while True:
            req = yield self._bank_req_queue[bank_id].get()
            if isinstance(req, SMEMBankReadReq):
                # read data from the shared memory
                smem_data = deepcopy(
                    self.array[req.smem_addr: req.smem_addr + self.alignment]
                )
                # compose a read response
                resp = SMEMBankReadResp(
                    smem_addr=req.smem_addr,
                    data=smem_data
                )
                # NOTE: pipelined design, non-blocking
                self.env.process(
                    self._delay_bank_response(
                        resp=resp,
                        delay=self.read_latency
                    )
                )
                # can serve another request next cycle
                yield self.env.timeout(1 * self.clock_unit)
                # update perf counter
                self.num_smem_bank_read += 1
            elif isinstance(req, SMEMBankAtomReq):
                # read data from the sharred memory
                smem_src_data = deepcopy(
                    self.array[req.smem_addr: req.smem_addr + self.alignment]
                )

                if req.op == "add.u32":
                    src_value = struct.unpack("i", smem_src_data)
                    dst_value = src_value[0] + int(req.val)
                    smem_dst_data = struct.pack("i", dst_value) 
                else:
                    raise NotImplementedError(
                        "Unknown shared memory atomic operation: {}".format(
                            req.op)
                    )

                # write data to the shared memory 
                self.array[req.smem_addr: req.smem_addr + self.alignment] = \
                    smem_dst_data 
                # compose an atomic response 
                resp = SMEMBankAtomResp(
                    smem_addr=req.smem_addr,
                    val=req.val,
                    op=req.op  
                )
                # NOTE: pipelined design, non-blocking 
                self.env.process(
                    self._delay_bank_response(
                        resp=resp,
                        delay=(self.read_latency + self.write_latency + 1)
                    )
                )
                # can serve another request next cycle
                yield self.env.timeout(1 * self.clock_unit)
                # update performance counter
                self.num_smem_bank_read += 1
                self.num_smem_bank_write += 1
            elif isinstance(req, SMEMBankWriteReq):
                # write data to the shared memory
                assert isinstance(req.data, bytearray)
                self.array[req.smem_addr: req.smem_addr + self.alignment] = \
                    deepcopy(req.data)
                # compose a write response
                resp = SMEMBankWriteResp(
                    smem_addr=req.smem_addr
                )
                # NOTE: pipelined design, non-blocking
                self.env.process(
                    self._delay_bank_response(
                        resp=resp,
                        delay=self.write_latency
                    )
                )
                # can serve another request next cycle
                yield self.env.timeout(1 * self.clock_unit)
                # update performance counter
                self.num_smem_bank_write += 1
            else:
                raise NotImplementedError(
                    "Unknown request type:{}".format(type(req))
                )

    def _sort_smem_request(self, smem_addr_list, data_width, simt_mask):
        """This function sort valid and unique memory request 
        according to alignment
        """
        sorted_smem_addr_list = {}
        for tid in range(len(smem_addr_list)):
            valid = (simt_mask >> tid) & 1
            if valid:
                # this thread is valid
                # NOTE: assume address is already aligned
                assert smem_addr_list[tid] % self.alignment == 0
                # NOTE: a 8-byte access is split into two accesses
                for offset in range(0, data_width, self.alignment):
                    new_addr = smem_addr_list[tid] + offset
                    if new_addr in sorted_smem_addr_list:
                        # merge request to the same shared memory location
                        sorted_smem_addr_list[new_addr][tid] = offset
                    else:
                        sorted_smem_addr_list[new_addr]\
                            = {tid: offset}
        return sorted_smem_addr_list

    def calc_bank_index(self, smem_addr):
        """Calculate the bank index for the given shared memory address
        """
        if self.config["smem_addr_map"] == "smem_addr_map_1":
            bank_index = (smem_addr // self.alignment) % self.num_bank
            return bank_index
        else:
            raise NotImplementedError(
                "Unknown smem address mapping: {}"
                .format(self.config["smem_addr_map"])
            )

    def _handle_read_response(self, req, sorted_smem_addr_list):
        data_width = req.data_width
        data_buffer = bytearray(
            data_width * self.config["num_threads_per_warp"]
        )

        for _ in range(len(sorted_smem_addr_list)):
            # there are still pending request
            bank_resp = yield self._bank_resp_queue.get(
                lambda x: (
                    isinstance(x, SMEMBankReadResp)
                    and (x.smem_addr in sorted_smem_addr_list)
                )
            )
            # update data buffer
            for tid in sorted_smem_addr_list[bank_resp.smem_addr]:
                # get position to write into data buffer
                offset = sorted_smem_addr_list[bank_resp.smem_addr][tid]
                db_start_addr = tid * data_width + offset
                db_end_addr = db_start_addr + self.alignment
                data_buffer[db_start_addr: db_end_addr] = deepcopy(
                    bank_resp.data
                )
        # compose a response
        resp = SMEMReadResp(
            smem_addr_list=req.smem_addr_list,
            data_width=req.data_width,
            data_buffer=data_buffer,
            simt_mask=req.simt_mask
        )
        yield self.resp_queue.put(resp)

    def _handle_write_response(self, req, sorted_smem_addr_list):
        for _ in range(len(sorted_smem_addr_list)):
            # there are still pending request
            _ = yield self._bank_resp_queue.get(
                lambda x: (
                    isinstance(x, SMEMBankWriteResp)
                    and (x.smem_addr in sorted_smem_addr_list)
                )
            )
        # compose a response
        resp = SMEMWriteResp(
            smem_addr_list=req.smem_addr_list,
            data_width=req.data_width,
            simt_mask=req.simt_mask
        )
        yield self.resp_queue.put(resp)

    def _handle_atomic_response(self, req, addr_val_pairs):
        for _ in range(len(addr_val_pairs)):
            # there are still pending requests 
            _ = yield self._bank_resp_queue.get(
                lambda x: (
                    isinstance(x, SMEMBankAtomResp)
                    and ((x.smem_addr, x.val) in addr_val_pairs)
                )
            )
        # compose a response
        resp = SMEMAtomResp(
            smem_addr_list=req.smem_addr_list,
            val_list=req.val_list,
            data_width=req.data_width, 
            op=req.op, 
            simt_mask=req.simt_mask 
        )
        yield self.resp_queue.put(resp)

    def _handle_request(self):
        """Handle shared memory request.
        Current implementation supports:
        (1) gather / scatter addressing
        (2) multi-bank support
        (3) pipelined design (unblocking)
        """
        while True:
            req = yield self.req_queue.get()
            # NOTE: here we implement a blocking shared memory.
            # A new request cannot be accepted until the previous one
            # completelt finishes.
            if isinstance(req, SMEMReadReq):
                smem_addr_list = req.smem_addr_list
                data_width = req.data_width
                simt_mask = req.simt_mask
                assert len(smem_addr_list) == \
                    self.config["num_threads_per_warp"]
                # NOTE: we assume smem access should be aligned already
                assert data_width % self.alignment == 0
                
                # sort request addresses
                sorted_smem_addr_list = self._sort_smem_request(
                    smem_addr_list=smem_addr_list,
                    data_width=data_width,
                    simt_mask=simt_mask
                )
                for smem_addr in sorted_smem_addr_list:
                    # compose a bank read request
                    bank_req = SMEMBankReadReq(
                        smem_addr=smem_addr
                    )
                    # get the bank index
                    bank_index = self.calc_bank_index(smem_addr)
                    # issue request
                    yield self._bank_req_queue[bank_index].put(bank_req)
                # NOTE consume 1 pipeline cycle
                yield self.env.timeout(1 * self.clock_unit)
                # spawn a process to collect response data and send
                # the final read response
                self.env.process(
                    self._handle_read_response(
                        req=req,
                        sorted_smem_addr_list=sorted_smem_addr_list
                    )
                )
                # we can accept another request for the next cycle
                yield self.env.timeout(1 * self.clock_unit)
            elif isinstance(req, SMEMAtomReq):
                smem_addr_list = req.smem_addr_list 
                val_list = req.val_list 
                op = req.op 
                simt_mask = req.simt_mask 
                assert len(smem_addr_list) == len(val_list)
                assert len(smem_addr_list) == \
                    self.config["num_threads_per_warp"]
                assert data_width == self.alignment 

                addr_val_pairs = []
                for tid in range(len(smem_addr_list)):
                    valid = (simt_mask >> tid) & 1
                    if valid:
                        smem_addr = smem_addr_list[tid]
                        val = val_list[tid]
                        addr_val_pairs.append((smem_addr, val))

                for tid in range(len(addr_val_pairs)):
                    smem_addr, val = addr_val_pairs[tid]
                    # compose a bank atomic request 
                    bank_req = SMEMBankAtomReq(
                        smem_addr=smem_addr,
                        val=val,
                        op=op
                    )
                    # get the bank index 
                    bank_index = self.calc_bank_index(smem_addr)
                    # issue request 
                    yield self._bank_req_queue[bank_index].put(bank_req)
                # NOTE consume 1 pipeline cycle 
                yield self.env.timeout(1 * self.clock_unit)
                # spawn a process to collect the response and send the final
                # atomic operation response 
                self.env.process(
                    self._handle_atomic_response(
                        req=req,
                        addr_val_pairs=set(addr_val_pairs) 
                    )
                )
                # we can accept another reequest for the next cycle 
                yield self.env.timeout(1 * self.clock_unit)
            elif isinstance(req, SMEMWriteReq):
                # NOTE: assuiming all accesses are aligned, there will not
                # be read-modify-write cases
                smem_addr_list = req.smem_addr_list
                data_width = req.data_width
                simt_mask = req.simt_mask
                assert len(smem_addr_list) == \
                    self.config["num_threads_per_warp"]
                # NOTE: we assume smem access should be aligned already
                assert data_width % self.alignment == 0

                # sort request addresses
                sorted_smem_addr_list = self._sort_smem_request(
                    smem_addr_list=smem_addr_list,
                    data_width=data_width,
                    simt_mask=simt_mask
                )
                for smem_addr in sorted_smem_addr_list:
                    # NOTE: write to the same address results in 
                    # undetermined behavior
                    assert len(sorted_smem_addr_list[smem_addr]) == 1, \
                        "SMEM write conflict! Unexpected bahavior"\
                        " will happen!"
                    data = bytearray(self.alignment)
                    for tid in sorted_smem_addr_list[smem_addr]:
                        offset = sorted_smem_addr_list[smem_addr][tid]
                        db_start_addr = tid * data_width + offset
                        db_end_addr = db_start_addr + self.alignment
                        data = deepcopy(
                            req.data_buffer[db_start_addr: db_end_addr]
                        )
                        # compose a bank write request
                        bank_req = SMEMBankWriteReq(
                            smem_addr=smem_addr,
                            data=data
                        )
                        # get the bank index
                        bank_index = self.calc_bank_index(smem_addr)
                        # issue request
                        yield self._bank_req_queue[bank_index].put(bank_req)
                # NOTE consume 1 pipeline cycle
                yield self.env.timeout(1 * self.clock_unit)
                # spawn a process to collect response data and send
                # the final read response
                self.env.process(
                    self._handle_write_response(
                        req=req,
                        sorted_smem_addr_list=sorted_smem_addr_list
                    )
                )
                # we can accept another request for the next cycle
                yield self.env.timeout(1 * self.clock_unit)
            else:
                raise NotImplementedError(
                    "Unknown request type:{}".format(type(req))
                )

