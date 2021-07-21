import simpy

from simulator.register_file_utility import OperandReadReq, OperandReadResp,\
    OperandWriteReq
from program.instruction import Operand


class RegMoveReq:
    def __init__(self, operand, subcore_id, pg_id, warp_id, is_upstream):
        """This is register move request
        If this moves reg from subcore to PG:
        - is_upstream is True
        - subcore_id is source, pg_id is destination
        - this request read from subcore register file
        If this moves reg from PG to subcore:
        - is_upstream is False
        - pg_id is source, subcore_id is destination
        - this request read from PG register file
        """
        assert isinstance(operand, Operand)
        self.operand = operand
        self.subcore_id = subcore_id
        self.pg_id = pg_id
        self.warp_id = warp_id
        self.is_upstream = is_upstream
        self.data = None
        # total reg size in bytes
        self.reg_size = None
        # this flag indicates whether this is used for
        # load / store register movement
        self.ld_st_flag = False
        self.is_nb = False
        self.prt_id = None


class RegMoveAck:
    def __init__(self, operand):
        self.operand = operand


class RegMoveEngine:

    def __init__(self, env, log, config, clock_unit, backend, 
                 regfile_io_interface, bus_arbiter, engine_type):
        """Register movement engine
        Args:
            env: simpy environment
            log: python log
            config: configuration dictionary
            clock_unit: clock unit
            backend: the parent component
                For far-bank, this is subcore
                For near-bank, this is PG
            regfile_io_interface: IO interface to register file
            bus_arbiter: subcore-PG bus arbiter
            engine_type: nb_reg_move or fb_reg_move
        """
        self.env = env
        self.log = log
        self.config = config
        self.clock_unit = clock_unit
        self.backend = backend
        self.regfile_io_interface = regfile_io_interface
        self.bus_arbiter = bus_arbiter
       
        self.simt_len = self.config["num_threads_per_warp"]
        self.all_1s = int("1" * self.simt_len, 2)
        
        self.is_far_bank = True if engine_type == "fb_reg_move" \
            else False
        self.num_unit = self.config["num_{}_engine".format(engine_type)]
        self.base_regfile_read_port_id = \
            self.config["base_regfile_read_port_id_{}_engine"
                        .format(engine_type)]
        self.base_regfile_write_port_id = \
            self.config["base_regfile_write_port_id_{}_engine"
                        .format(engine_type)]

        self.reg_req_queue = simpy.Store(self.env, capacity=1)
        self.reg_ack_queue = simpy.FilterStore(self.env, capacity=1)
        if self.is_far_bank:
            self.reg_req_bus_queue = simpy.FilterStore(self.env, capacity=1)
        # for local read request handling
        self.reg_read_req_queue = simpy.Store(self.env, capacity=1)
        self.reg_read_ack_queue = simpy.FilterStore(self.env, capacity=1)
        # for local write request handling
        self.reg_write_req_queue = simpy.Store(self.env, capacity=1)
        self.reg_write_ack_queue = simpy.FilterStore(self.env, capacity=1)
        # each register move engine can handle a read request and a write
        # request at the same cycle
        for i in range(self.num_unit):
            if self.is_far_bank:
                self.env.process(
                    self._handle_reg_move_req_far_bank()
                )
            else:
                self.env.process(
                    self._handle_reg_move_req_near_bank()
                )
            regfile_read_port_id = self.base_regfile_read_port_id + i
            regfile_write_port_id = self.base_regfile_write_port_id + i
            self.env.process(
                self._handle_reg_move_read_req(regfile_read_port_id)
            )
            self.env.process(
                self._handle_reg_move_write_req(regfile_write_port_id)
            )

    def _get_reg_addr(self, reg_prefix, reg_index, subcore_id, warp_id):
        if self.is_far_bank:
            return self.backend.get_subcore_reg_addr(
                reg_prefix, reg_index, warp_id
            )
        else:
            return self.backend.get_pg_reg_addr(
                reg_prefix, reg_index, subcore_id, warp_id
            )

    def _check_reg_ready(self, op_str, subcore_id, warp_id):
        if self.is_far_bank:
            return self.backend.reg_track_table\
                .entry[warp_id].check_ready(
                    op_str=op_str,
                    is_near_bank=False
                )
        else:
            return self.backend.core\
                .subcore_array[subcore_id]\
                .reg_track_table.entry[warp_id].check_ready(
                    op_str=op_str,
                    is_near_bank=True
                )

    def _handle_reg_move_req_far_bank(self):
        while True:
            req = yield self.reg_req_queue.get()
            assert isinstance(req, RegMoveReq)
            # this is master data movement engine in subcore
            if req.is_upstream:
                # subcore --> pg
                # read subcore's reg file
                yield self.reg_read_req_queue.put(req)
                req = yield self.reg_read_ack_queue.get(
                    lambda x: (x.operand.op_str == req.operand.op_str
                               and x.subcore_id == req.subcore_id
                               and x.pg_id == req.pg_id
                               and x.warp_id == req.warp_id)
                )

                # issue to pg, upstream traffic
                yield self.bus_arbiter\
                    .upstream_req_queue.put(req)
                req = yield self.reg_req_bus_queue.get(
                    lambda x: (x.operand.op_str == req.operand.op_str
                               and x.subcore_id == req.subcore_id
                               and x.pg_id == req.pg_id
                               and x.warp_id == req.warp_id)
                )

                # respond to instr offload engine
                mov_ack = RegMoveAck(
                    operand=req.operand
                )
                yield self.reg_ack_queue.put(mov_ack)
            else:
                # pg --> subcore
                # issue to pg, upstream traffic
                yield self.bus_arbiter\
                    .upstream_req_queue.put(req)
                req = yield self.reg_req_bus_queue.get(
                    lambda x: (x.operand.op_str == req.operand.op_str
                               and x.subcore_id == req.subcore_id
                               and x.pg_id == req.pg_id
                               and x.warp_id == req.warp_id)
                )
                # write subcore's reg file
                yield self.reg_write_req_queue.put(req)
                req = yield self.reg_write_ack_queue.get(
                    lambda x: (x.operand.op_str == req.operand.op_str
                               and x.subcore_id == req.subcore_id
                               and x.pg_id == req.pg_id
                               and x.warp_id == req.warp_id)
                )
                # respond to instr offload engine
                mov_ack = RegMoveAck(
                    operand=req.operand
                )
                yield self.reg_ack_queue.put(mov_ack)

    def _handle_reg_move_req_near_bank(self):
        while True:
            req = yield self.reg_req_queue.get()
            assert isinstance(req, RegMoveReq)
            # this is slave data movement engine in pg
            if req.is_upstream:
                # subcore --> pg
                # write pg's reg file

                yield self.reg_write_req_queue.put(req)
                req = yield self.reg_write_ack_queue.get(
                    lambda x: (x.operand.op_str == req.operand.op_str
                               and x.subcore_id == req.subcore_id
                               and x.pg_id == req.pg_id
                               and x.warp_id == req.warp_id)
                )

                if req.ld_st_flag:
                    assert False, "For ld.global, please use"\
                        " writeback path"
                else:
                    # issue to subcore, downstream traffic
                    yield self.bus_arbiter\
                        .downstream_req_queue.put(req)
            else:
                # pg --> subcore
                # read pg's reg file
                yield self.reg_read_req_queue.put(req)
                req = yield self.reg_read_ack_queue.get(
                    lambda x: (x.operand.op_str == req.operand.op_str
                               and x.subcore_id == req.subcore_id
                               and x.pg_id == req.pg_id
                               and x.warp_id == req.warp_id)
                )
                if req.ld_st_flag and req.is_nb:
                    # near-bank lsu_entension reg write
                    yield self.backend.lsu_extension\
                        .reg_mov_ack_queue.put(req)
                else:
                    # issue to subcore, downstream traffic
                    yield self.bus_arbiter\
                        .downstream_req_queue.put(req)

    def _handle_reg_move_read_req(self, regfile_read_port_id):
        while True:
            req = yield self.reg_read_req_queue.get()
            # check if the source register is ready
            reg_ready = self._check_reg_ready(
                op_str=req.operand.op_str,
                subcore_id=req.subcore_id,
                warp_id=req.warp_id
            )
            assert reg_ready is True, "source register is not valid! {}"\
                .format(req.operand.op_str)

            # compose a read request
            reg_addr, reg_size = self._get_reg_addr(
                reg_prefix=req.operand.reg_prefix,
                reg_index=req.operand.reg_index,
                subcore_id=req.subcore_id,
                warp_id=req.warp_id
            )
            operand_read_req = OperandReadReq(
                operand_id=None,
                base_reg_addr=reg_addr,
                total_reg_size=reg_size
            )
            # issue this to read interface
            yield self.regfile_io_interface\
                .read_req_queue[regfile_read_port_id].put(operand_read_req)
            # wait until request is received
            resp = yield self.regfile_io_interface\
                .read_resp_queue[regfile_read_port_id].get()
            assert isinstance(resp, OperandReadResp), "The incorrect type" \
                " from the response queue"
            # NOTE: register movement between far-bank and near-bank does not
            # change dependency relationship
            # we do not update dep_table here
            
            # set payload
            req.data = resp.data
            # issue ack
            yield self.reg_read_ack_queue.put(req)

    def _handle_reg_move_write_req(self, regfile_write_port_id):
        while True:
            req = yield self.reg_write_req_queue.get()

            reg_addr, reg_size = self._get_reg_addr(
                reg_prefix=req.operand.reg_prefix,
                reg_index=req.operand.reg_index,
                subcore_id=req.subcore_id,
                warp_id=req.warp_id
            )
            # compose a write request
            # NOTE: we move the whole register across all threads, since
            # register trakcing table does not consider SIMT
            assert req.data is not None
            operand_write_req = OperandWriteReq(
                base_reg_addr=reg_addr,
                total_reg_size=reg_size,
                simt_mask=self.all_1s,
                data=req.data
            )
            # issue this to write interface
            yield self.regfile_io_interface\
                .write_req_queue[regfile_write_port_id]\
                .put(operand_write_req)
            # wait until write ack
            _ = yield self.regfile_io_interface.write_resp_queue.get(
                lambda x: (x.base_reg_addr == reg_addr
                           and x.total_reg_size == reg_size)
            )
            # issue ack
            yield self.reg_write_ack_queue.put(req)

