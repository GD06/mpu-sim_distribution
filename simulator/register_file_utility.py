import simpy
from copy import deepcopy

from simulator.register_file import RegReadReq, RegWriteReq, \
    RegReadResp, RegWriteAck


class OperandReadReq:
    def __init__(self, operand_id, 
                 base_reg_addr, total_reg_size):
        self.operand_id = operand_id
        self.base_reg_addr = base_reg_addr
        self.total_reg_size = total_reg_size


class OperandReadResp:
    def __init__(self, operand_id, 
                 base_reg_addr, total_reg_size, data):
        self.operand_id = operand_id
        self.base_reg_addr = base_reg_addr
        self.total_reg_size = total_reg_size
        self.data = data


class OperandWriteReq:
    def __init__(self, base_reg_addr, total_reg_size,
                 simt_mask, data):
        self.base_reg_addr = base_reg_addr
        self.total_reg_size = total_reg_size
        self.simt_mask = simt_mask
        self.data = data


class OperandWriteResp:
    def __init__(self, base_reg_addr, total_reg_size):
        self.base_reg_addr = base_reg_addr
        self.total_reg_size = total_reg_size


class RegFileOperandIOInterface:
    """Operand reading / writing interface for register file
    """
    
    def __init__(self, env, log, config, clock_unit, reg_file, interface_type):
        self.env = env
        self.log = log
        self.config = config
        self.clock_unit = deepcopy(clock_unit)
        self.reg_file = reg_file
        self.interface_type = interface_type
        self.simt_len = self.config["num_threads_per_warp"]
        self.alignment = self.reg_file.alignment
        self.all_1s = int("1" * self.simt_len, 2)
        
        # the following read queues also contains queues for 
        # read-modify-write's read request
        self.read_req_queue = {}
        self.read_resp_queue = {}
        #  write queues
        self.write_req_queue = {}
        if interface_type == "far-bank":
            for i in range(self.config["num_subcore_reg_file_read_port"]):
                self.read_req_queue[i] = simpy.Store(env, capacity=1)
                self.read_resp_queue[i] = simpy.Store(env, capacity=1)
            for i in range(self.config["num_subcore_reg_file_read_port"]):
                self.env.process(self._handle_operand_read_req(i))
            # for read-modify-write read ports only
            self.rmw_base_read_port_id = \
                self.config["base_regfile_read_port_id_fb_rmw"]
            for i in range(self.config["num_subcore_reg_file_write_port"]):
                self.write_req_queue[i] = simpy.Store(env, capacity=1)
            self.write_resp_queue = simpy.FilterStore(env, capacity=1)
            for i in range(self.config["num_subcore_reg_file_write_port"]):
                self.env.process(self._handle_operand_write_req(i))
        elif interface_type == "near-bank":
            for i in range(self.config["num_pg_reg_file_read_port"]):
                self.read_req_queue[i] = simpy.Store(env, capacity=1)
                self.read_resp_queue[i] = simpy.Store(env, capacity=1)
            for i in range(self.config["num_pg_reg_file_read_port"]):
                self.env.process(self._handle_operand_read_req(i))
            # for read-modify-write read ports only
            self.rmw_base_read_port_id = \
                self.config["base_regfile_read_port_id_nb_rmw"]
            for i in range(self.config["num_pg_reg_file_write_port"]):
                self.write_req_queue[i] = simpy.Store(env, capacity=1)
            self.write_resp_queue = simpy.FilterStore(env, capacity=1)
            for i in range(self.config["num_pg_reg_file_write_port"]):
                self.env.process(self._handle_operand_write_req(i))
        else:
            raise NotImplementedError(
                "Unknown register file interface type:{}"
                .format(interface_type)
            )

    def _receive_reg_read_resp(self, regfile_read_port_id, operand_read_req, 
                               aligned_start_addr, num_req):
        cur_start_addr = aligned_start_addr
        temp_buf = bytearray(num_req * self.alignment)
        buf_start_addr = 0
        for i in range(num_req):
            resp = yield self.reg_file.read_resp_queue.get(
                lambda x: x.reg_addr == cur_start_addr
            )
            assert isinstance(resp, RegReadResp), "The incorrect type" \
                " from the response queue"
            temp_buf[buf_start_addr: buf_start_addr + self.alignment] = \
                deepcopy(resp.data)
            cur_start_addr = cur_start_addr + self.alignment
            buf_start_addr = buf_start_addr + self.alignment
        # NOTE: consume 1 pipeline cycle
        yield self.env.timeout(1 * self.clock_unit)
        # calculate correct index into the array
        buf_start_addr = operand_read_req.base_reg_addr \
            % self.alignment
        buf_end_addr = buf_start_addr \
            + operand_read_req.total_reg_size
        assert buf_end_addr <= num_req * self.alignment
        # prepare a read response
        operand_read_resp = OperandReadResp(
            operand_id=operand_read_req.operand_id,
            base_reg_addr=operand_read_req.base_reg_addr,
            total_reg_size=operand_read_req.total_reg_size,
            data=deepcopy(temp_buf[buf_start_addr: buf_end_addr])
        )
        yield self.read_resp_queue[regfile_read_port_id].put(operand_read_resp)

    def _handle_operand_read_req(self, regfile_read_port_id):
        while True:
            operand_read_req = \
                yield self.read_req_queue[regfile_read_port_id].get()
            assert isinstance(operand_read_req, OperandReadReq)
            ori_start_addr = operand_read_req.base_reg_addr
            ori_end_addr = ori_start_addr + operand_read_req.total_reg_size
            # calculate aligned address
            aligned_start_addr = self.reg_file._align_down(ori_start_addr)
            aligned_end_addr = self.reg_file._align_up(ori_end_addr)
            num_req = (aligned_end_addr - aligned_start_addr) \
                // self.alignment
            cur_start_addr = aligned_start_addr
            for i in range(num_req):
                # compose and issue reg file req
                bank_index = self.reg_file.calc_bank_index(cur_start_addr)
                req = RegReadReq(reg_addr=cur_start_addr)
                yield self.reg_file.read_req_queue[bank_index].put(req)
                cur_start_addr = cur_start_addr + self.alignment
            # NOTE: issue request take 1 pipeline cycle
            yield self.env.timeout(1 * self.clock_unit)
            # spawn a process to handle returned read response
            self.env.process(
                self._receive_reg_read_resp(
                    regfile_read_port_id=regfile_read_port_id,
                    operand_read_req=operand_read_req,
                    aligned_start_addr=aligned_start_addr,
                    num_req=num_req
                )
            )

    def _receive_reg_write_resp(self, operand_write_req, aligned_start_addr,
                                num_req):
        cur_start_addr = aligned_start_addr
        for i in range(num_req):
            resp = yield self.reg_file.write_resp_queue.get(
                lambda x: x.reg_addr == cur_start_addr
            )
            assert isinstance(resp, RegWriteAck), "The incorrect type" \
                " from the response queue"
            cur_start_addr = cur_start_addr + self.alignment
        # NOTE: consume 1 pipeline cycle
        yield self.env.timeout(1 * self.clock_unit)
        # prepare a write response
        operand_write_resp = OperandWriteResp(
            base_reg_addr=operand_write_req.base_reg_addr,
            total_reg_size=operand_write_req.total_reg_size
        )
        yield self.write_resp_queue.put(operand_write_resp)

    def _issue_aligned_uniform_write(self, operand_write_req, 
                                     aligned_start_addr, num_req,
                                     aligned_data):
        """This function assumes: (1) write data and address are already 
        aligned to register file bank; (2) simt_mask are all 1s
        """
        assert aligned_start_addr % self.alignment == 0
        cur_start_addr = aligned_start_addr
        buf_start_addr = 0
        # NOTE: consume 1 pipeline cycle
        yield self.env.timeout(1 * self.clock_unit)
        # compose and issue reg file req
        for i in range(num_req):
            bank_index = self.reg_file.calc_bank_index(cur_start_addr)
            req = RegWriteReq(
                reg_addr=cur_start_addr,
                data=deepcopy(
                    aligned_data[buf_start_addr: 
                                 buf_start_addr + self.alignment]
                )
            )
            yield self.reg_file.write_req_queue[bank_index].put(req)
            cur_start_addr = cur_start_addr + self.alignment
            buf_start_addr = buf_start_addr + self.alignment
        # spawn a process to handle returned write response
        self.env.process(
            self._receive_reg_write_resp(
                operand_write_req=operand_write_req,
                aligned_start_addr=aligned_start_addr,
                num_req=num_req
            )
        )

    def _is_write_req_aligned(self, reg_addr, reg_size):
        if reg_addr % self.alignment != 0:
            return False
        if reg_size % self.alignment != 0:
            return False
        return True

    def _compose_preload_req(self, operand_write_req):
        # calculate aligned access addresses
        aligned_start_addr = self.reg_file._align_down(
            operand_write_req.base_reg_addr
        )
        aligned_end_addr = self.reg_file._align_up(
            operand_write_req.base_reg_addr
            + operand_write_req.total_reg_size
        )
        aligned_reg_size = aligned_end_addr\
            - aligned_start_addr
        # compose a read request
        operand_read_req = OperandReadReq(
            operand_id=None,
            base_reg_addr=aligned_start_addr,
            total_reg_size=aligned_reg_size
        )
        return operand_read_req

    def _handle_operand_write_req(self, regfile_write_port_id):
        regfile_rmw_read_port_id = self.rmw_base_read_port_id \
            + regfile_write_port_id
        while True:
            operand_write_req = \
                yield self.write_req_queue[regfile_write_port_id].get()
            assert isinstance(operand_write_req, OperandWriteReq)
            simt_mask = operand_write_req.simt_mask
            if simt_mask == self.all_1s:
                # uniform write
                if self._is_write_req_aligned(
                    reg_addr=operand_write_req.base_reg_addr,
                    reg_size=operand_write_req.total_reg_size
                ) is False:
                    # unaligned access
                    # read then modify: get original data first
                    operand_read_req = self._compose_preload_req(
                        operand_write_req=operand_write_req
                    )
                    # issue this to read interface
                    yield self.read_req_queue[regfile_rmw_read_port_id]\
                        .put(operand_read_req)
                    # get original data
                    operand_read_resp = yield self\
                        .read_resp_queue[regfile_rmw_read_port_id]\
                        .get()
                    # NOTE: consume 1 pipeline cycle
                    yield self.env.timeout(1 * self.clock_unit)
                    # compose write data
                    write_data = deepcopy(operand_read_resp.data)
                    buf_start_addr = operand_write_req.base_reg_addr\
                        % self.alignment
                    buf_end_addr = buf_start_addr \
                        + operand_write_req.total_reg_size
                    assert buf_end_addr <= len(write_data)
                    write_data[buf_start_addr: buf_end_addr] = \
                        deepcopy(operand_write_req.data)
                    # issue write request
                    num_req = len(write_data) // self.alignment
                    self.env.process(
                        self._issue_aligned_uniform_write(
                            operand_write_req=operand_write_req,
                            aligned_start_addr=operand_read_resp.base_reg_addr,
                            num_req=num_req,
                            aligned_data=write_data
                        )
                    )
                else:
                    # aligned access
                    # issue write request
                    num_req = operand_write_req.total_reg_size\
                        // self.alignment
                    # compose write data
                    write_data = deepcopy(operand_write_req.data)
                    self.env.process(
                        self._issue_aligned_uniform_write(
                            operand_write_req=operand_write_req,
                            aligned_start_addr=operand_write_req.base_reg_addr,
                            num_req=num_req,
                            aligned_data=write_data
                        )
                    )
            elif simt_mask > 0:
                # divergent write
                # read then modify: get original data first
                operand_read_req = self._compose_preload_req(
                    operand_write_req=operand_write_req
                )
                # issue this to read interface
                yield self.read_req_queue[regfile_rmw_read_port_id]\
                    .put(operand_read_req)
                # get original data
                operand_read_resp = \
                    yield self.read_resp_queue[regfile_rmw_read_port_id].get()
                # NOTE: consume 1 pipeline cycle
                yield self.env.timeout(1 * self.clock_unit)
                # compose write data
                write_data = deepcopy(operand_read_resp.data)
                assert operand_write_req.total_reg_size % self.simt_len == 0
                data_width = operand_write_req.total_reg_size // self.simt_len
                buf_start_addr = operand_write_req.base_reg_addr\
                    % self.alignment
                for i in range(self.simt_len):
                    valid = (simt_mask >> i) & 1
                    base = data_width * i
                    buf_base = buf_start_addr + base
                    if valid:
                        write_data[buf_base: buf_base + data_width] = deepcopy(
                            operand_write_req.data[base: base + data_width]
                        )
                # issue write request
                num_req = len(write_data) // self.alignment
                self.env.process(
                    self._issue_aligned_uniform_write(
                        operand_write_req=operand_write_req,
                        aligned_start_addr=operand_read_resp.base_reg_addr,
                        num_req=num_req,
                        aligned_data=write_data
                    )
                )
            else:
                # empty request
                assert False, "write simt mask is not allowed to be zero!"

