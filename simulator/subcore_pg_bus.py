from copy import deepcopy

import simpy
import math
from simulator.dram_message import DRAMTransaction, PRTEntryReq
from simulator.instr_instance import InstrEntry
from simulator.reg_move_engine import RegMoveReq


class SubcorePGBusArbiter:

    def __init__(self, env, log, config, core):
        self.env = env
        self.log = log
        self.config = config
        self.core = core

        self.filter_func = core.filter_func
        self.traceEvents = []

        # set clock unit
        assert config["sim_clock_freq"] % \
            config["subcore_pg_bus_clock_freq"] == 0
        self.clock_unit = config["sim_clock_freq"] \
            // config["subcore_pg_bus_clock_freq"]

        # shared bus width in bytes
        self.bus_width = self.config["core_shared_bus_io_width"]

        self.upstream_req_queue = simpy.Store(self.env)
        self.downstream_req_queue = simpy.Store(self.env)
        self.env.process(self._handle_upstream_traffic())
        self.env.process(self._handle_downstream_traffic())

        self.pg_bus_buffer_queue = []
        self.subcore_bus_buffer_queue = []

        self.data_bus_lock = simpy.Resource(env, capacity=2)
        self.cmd_bus_lock = simpy.Resource(env, capacity=2)

        # for tracing
        self.downstream_start_time = 0
        self.upstream_start_time = 0

        # performance metrics
        self.num_bus_cyc = 0

    def get_trace_events(self):
        """Get a list of trace events"""
        _trace_Events = deepcopy(self.traceEvents)
        return _trace_Events

    def _append_trace_event_dur(self, tid, name, ts, dur, cat="", args={}):
        new_event = {}
        new_event["pid"] = "proc_{}_core_{}_bus".format(
            self.core.processor.proc_id, self.core.core_id
        )
        new_event["tid"] = tid
        new_event["ph"] = "X"
        new_event["name"] = name
        new_event["ts"] = ts / 1000
        new_event["dur"] = dur / 1000
        if len(cat) > 0:
            new_event["cat"] = cat
        if len(args) > 0:
            new_event["args"] = args
        if self.filter_func(new_event):
            self.traceEvents.append(new_event)
        return

    def get_perf_metrics(self):
        """Get a dictionary of performance metrics"""
        perf_metrics = {}
        perf_metrics["num_bus_cyc"] = self.num_bus_cyc
        return {"subcore_pg_bus": perf_metrics}

    def _enqueue_pg_req(self, pg_id, req):
        yield self.core.pg_array[pg_id]\
            .bus_receive_buffer.put(req)

    def _enqueue_subcore_req(self, subcore_id, req):
        yield self.core.subcore_array[subcore_id]\
            .bus_receive_buffer.put(req)

    def _enqueue_lsu_remote_req(self, req):
        yield self.core.lsu_remote\
            .in_dram_trans_queue.put(req)
    
    def _cal_delay_cycle(self, req, is_upstream):
        """Calculate the number of cycles to transmit the given
        request packet on the shared data bus
        Args:
            req: the request packet to be transmitted
            is_upstream: whether the transmission is upstream 
                or downstream
        Returns:
            delay_cycle: the number of delayed cycles on the 
                shared data bus
        """
        data_size = 0
        if isinstance(req, RegMoveReq):
            assert req.reg_size is not None
            if is_upstream:
                if req.is_upstream:
                    # this contains payload
                    data_size = req.reg_size
            else:
                if not req.is_upstream:
                    # this contains payload
                    data_size = req.reg_size
        elif isinstance(req, DRAMTransaction):
            if is_upstream:
                if req.type == "store":
                    # this contains payload
                    data_size = self.config["dram_bank_io_width"]
            else:
                if req.type == "load":
                    # this contains payload
                    data_size = self.config["dram_bank_io_width"]
        elif isinstance(req, PRTEntryReq):
            # For offloaded PRTEntry, since memory is perfectly aligned
            # and unform, we assume offset_list and co_addr_list can
            # be easily recovered at near-bank site and introduce no
            # transmission overhead
            # The only overhead is the transmission of instruction
            # associated with it
            data_size = 8
        elif isinstance(req, InstrEntry):
            # simt_mask requires 4 byte
            # the instruction requires 4 byte
            data_size = 8
        else:
            assert False
        delay_cycle = math.ceil(data_size / self.bus_width)
        return delay_cycle

    def _trans_cmd_bus(self):
        # try to acquire cmd bus
        with self.cmd_bus_lock.request() as bus_access:
            # wait for access
            yield bus_access
            # NOTE: consume 1 pipeline cycle
            yield self.env.timeout(1 * self.clock_unit)
    
    def _trans_data_bus(self, req, is_upstream):
        # try to acquire data bus
        with self.data_bus_lock.request() as bus_access:
            # wait for access
            yield bus_access
            if is_upstream:
                self.upstream_start_time = self.env.now
            else:
                self.downstream_start_time = self.env.now
            # occupy the bus for a number of cycles
            delay_cycle = self._cal_delay_cycle(
                req=req,
                is_upstream=is_upstream
            )
            yield self.env.timeout(delay_cycle * self.clock_unit)
            # update performance counter
            self.num_bus_cyc += delay_cycle

    def _get_trace_name_args(self, req):
        args = {}
        if isinstance(req, RegMoveReq):
            name = "reg_mov"
            args["operand"] = req.operand.op_str
            args["subcore_id"] = req.subcore_id
            args["pg_id"] = req.pg_id
            args["warp_id"] = req.warp_id
            args["is_upstream"] = req.is_upstream
            args["ld_st_flag"] = req.ld_st_flag
            if req.ld_st_flag:
                args["prt_id"] = req.prt_id
        elif isinstance(req, DRAMTransaction):
            name = "dram_trans"
            args["type"] = req.type
            args["pg_id"] = req.pg_id
            args["pe_id"] = req.pe_id
            args["is_remote"] = req.is_remote
            if not req.is_remote:
                args["subcore_id"] = req.subcore_id
            args["prt_id"] = req.prt_id
        elif isinstance(req, PRTEntryReq):
            name = "prt"
            args["instr"] = req.instr_entry.instr.trace_str()
            args["subcore_id"] = req.instr_entry.subcore_id
            args["pg_id"] = req.pg_id
            args["warp_id"] = req.instr_entry.warp_id
        elif isinstance(req, InstrEntry):
            name = "instr_entry"
            args["instr"] = req.instr.trace_str()
            args["subcore_id"] = req.subcore_id
            args["pg_id"] = req.pg_id
            args["warp_id"] = req.warp_id
        else:
            assert False
        return name, args

    def _handle_upstream_traffic(self):
        """This function handles traffic from subcore to PG
        """
        while True:
            req = yield self.upstream_req_queue.get()
            # here we parallelize data and cmd transmission
            event_list = []
            event_list.append(
                self.env.process(
                    self._trans_cmd_bus()
                )
            )
            event_list.append(
                self.env.process(
                    self._trans_data_bus(
                        req=req,
                        is_upstream=True
                    )
                )
            )
            yield simpy.events.AllOf(self.env, event_list)
            
            # NOTE: trace event
            name, args = self._get_trace_name_args(req)
            self._append_trace_event_dur(
                tid="upstream",
                name=name,
                ts=self.upstream_start_time,
                dur=self.env.now - self.upstream_start_time,
                args=args
            )

            # get PG id
            pg_id = req.pg_id
            assert pg_id is not None
            # send this to the corresponding PG
            # NOTE: this is unblocking
            self.env.process(
                self._enqueue_pg_req(
                    pg_id=pg_id,
                    req=req
                )
            )

    def _handle_downstream_traffic(self):
        """This function handles traffic from PG to subcore
        """
        while True:
            req = yield self.downstream_req_queue.get()
            # here we parallelize data and cmd transmission
            event_list = []
            event_list.append(
                self.env.process(
                    self._trans_cmd_bus()
                )
            )
            event_list.append(
                self.env.process(
                    self._trans_data_bus(
                        req=req,
                        is_upstream=False
                    )
                )
            )
            yield simpy.events.AllOf(self.env, event_list)

            # NOTE: trace event
            name, args = self._get_trace_name_args(req)
            self._append_trace_event_dur(
                tid="downstream",
                name=name,
                ts=self.downstream_start_time,
                dur=self.env.now - self.downstream_start_time,
                args=args
            )
            
            if (
                isinstance(req, DRAMTransaction) 
                and req.is_remote is True
            ):
                # NOTE: this is unblocking
                self.env.process(
                    self._enqueue_lsu_remote_req(
                        req=req
                    )
                )
            else:
                # get subcore id
                subcore_id = req.subcore_id
                assert subcore_id is not None
                # send this to the corresponding subcore
                # NOTE: this is unblocking
                self.env.process(
                    self._enqueue_subcore_req(
                        subcore_id=subcore_id, 
                        req=req
                    )
                )
