import simpy 
from copy import deepcopy 

from simulator.subcore_table import WarpInfoTable, WarpPipelineTable, StackTable 
from simulator.subcore_table import WarpPipelineTableEntry, DepTable, \
    RegTrackTable
from simulator.instr_cache import InstrLoadReq  
from simulator.register_file import RegisterFile 
from simulator.operand_collector import OperandCollector
from simulator.alu import ArithmeticLogicUnit
from simulator.sfu import SpecialFunctionUnit
from simulator.load_store_unit import LoadStoreUnit
from simulator.dram_message import DRAMTransaction
from simulator.synchronization_unit import SynchronizationUnit
from simulator.control_flow_unit import ControlFlowUnit
from simulator.instr_instance import InstrEntry
from simulator.register_file_utility import RegFileOperandIOInterface, \
    OperandWriteReq
from simulator.instr_offload_engine import InstrOffloadEngine
from simulator.reg_move_engine import RegMoveEngine, RegMoveReq


class Subcore:

    def __init__(self, subcore_id, env, config, log, core):
        self.subcore_id = subcore_id 
        self.env = env 
        self.config = config 
        self.log = log
        self.core = core

        self.filter_func = core.filter_func
        self.traceEvents = []

        self.clock_unit = deepcopy(core.clock_unit) 
        self._loc_str = "Processor ID: {proc_id}, Core ID: {core_id}, " \
            "Subcore ID: {subcore_id}".format(
                proc_id=self.core.processor.proc_id,
                core_id=self.core.core_id,
                subcore_id=self.subcore_id, 
            )

        self.start_exec_cmd = simpy.Store(env, capacity=1)
        self.finish_exec_resp = simpy.Store(env, capacity=1) 

        self.num_active_warps = 0
        self.warp_info_table = WarpInfoTable(config=self.config, log=self.log) 
        self.warp_pipeline_table = \
            WarpPipelineTable(config=self.config, log=self.log)
        self.stack_table = StackTable(config=self.config, log=self.log)
        self.dep_table_exe = DepTable(config=self.config, log=self.log)
        # [block_id][warp_id] as index
        # initialized in core.py
        self.num_issued_not_commit_instr = {}

        self.reg_file = RegisterFile(
            env=env, 
            log=log, 
            config=self.config,
            clock_unit=self.clock_unit,  
            reg_file_type="far-bank"
        )

        self.reg_track_table = RegTrackTable(
            config=self.config,
            log=self.log,
            reg_file=self.reg_file
        )

        self.rf_io_interface = RegFileOperandIOInterface(
            env=self.env,
            log=self.log,
            config=self.config,
            clock_unit=self.clock_unit,
            reg_file=self.reg_file,
            interface_type="far-bank",
        )

        self.reg_move_engine = RegMoveEngine(
            env=self.env,
            log=self.log,
            config=self.config,
            clock_unit=self.clock_unit,
            backend=self,
            regfile_io_interface=self.rf_io_interface,
            bus_arbiter=self.core.subcore_pg_bus_arbiter,
            engine_type="fb_reg_move"
        )

        self.instr_offload_engine = InstrOffloadEngine(
            env=self.env,
            config=self.config,
            log=self.log,
            reg_move_engine=self.reg_move_engine,
            subcore=self
        )

        # For far-bank writeback
        self.base_regfile_write_port_id_commit = \
            self.config["base_regfile_write_port_id_fb_commit"]

        self.decode_buffer = []
        self.num_free_decode_buffer_slots = []
        for i in range(self.config["max_num_warp_per_subcore"]):
            self.decode_buffer.append(simpy.Store(env))
            self.num_free_decode_buffer_slots.append(
                self.config["decode_buffer_size"])

        self.execute_buffer = simpy.Store(
            env, capacity=(self.config["subcore_execute_buffer_size"])
        )

        self.writeback_buffer = simpy.Store(
            env, capacity=self.config["subcore_writeback_buffer_size"]
        )

        self.commit_buffer = simpy.Store(
            env, capacity=self.config["subcore_commit_buffer_size"]
        )

        self.sync_buffer = simpy.Store(
            env, capacity=self.config["max_num_warp_per_subcore"]
        )

        self.bus_receive_buffer = simpy.Store(
            env, capacity=self.config["subcore_bus_receive_buffer_size"]
        )

        self.reg_base_ptr = 0

        # Execution Stage
        # arithemtic logic unit
        self.fb_alu = ArithmeticLogicUnit(
            env=self.env,
            log=self.log,
            config=self.config,
            clock_unit=self.clock_unit,
            backend=self,
            alu_type="far-bank"
        )
        self.opc_fb_alu = OperandCollector(
            env=self.env,
            log=self.log,
            config=self.config,
            clock_unit=self.clock_unit,
            backend=self,
            regfile_io_interface=self.rf_io_interface,
            execution_unit=self.fb_alu,
            opc_type="fb_alu"
        )
        
        # special function unit
        self.sfu = SpecialFunctionUnit(
            env=self.env,
            log=self.log,
            config=self.config,
            clock_unit=self.clock_unit,
            subcore=self
        )
        self.opc_sfu = OperandCollector(
            env=self.env,
            log=self.log,
            config=self.config,
            clock_unit=self.clock_unit,
            backend=self,
            regfile_io_interface=self.rf_io_interface,
            execution_unit=self.sfu,
            opc_type="sfu"
        )

        # load-store unit
        self.lsu = LoadStoreUnit(
            env=self.env,
            log=self.log,
            config=self.config,
            clock_unit=self.clock_unit,
            subcore=self
        )
        self.opc_lsu = OperandCollector(
            env=self.env,
            log=self.log,
            config=self.config,
            clock_unit=self.clock_unit,
            backend=self,
            regfile_io_interface=self.rf_io_interface,
            execution_unit=self.lsu,
            opc_type="lsu"
        )

        # control-flow unit
        self.cfu = ControlFlowUnit(
            env=self.env,
            log=self.log,
            config=self.config,
            clock_unit=self.clock_unit,
            subcore=self
        )
        self.opc_cfu = OperandCollector(
            env=self.env,
            log=self.log,
            config=self.config,
            clock_unit=self.clock_unit,
            backend=self,
            regfile_io_interface=self.rf_io_interface,
            execution_unit=self.cfu,
            opc_type="cfu"
        )

        # synchronization unit
        self.syncu = SynchronizationUnit(
            env=self.env,
            log=self.log,
            config=self.config,
            clock_unit=self.clock_unit,
            subcore=self
        )
        self.opc_syncu = OperandCollector(
            env=self.env,
            log=self.log,
            config=self.config,
            clock_unit=self.clock_unit,
            backend=self,
            regfile_io_interface=self.rf_io_interface,
            execution_unit=self.syncu,
            opc_type="syncu"
        )

        # Spawn a process for the fetch and decode subcore stages
        self.env.process(self._fetch_and_decode())

        self.pending_issue_instr = []
        # Spawn processes for the issue subcore stage
        for i in range(self.config["max_num_warp_per_subcore"]):
            self.pending_issue_instr.append({})
            for j in range(self.config["num_issue_port_per_warp"]):
                self.env.process(
                    self._issue(
                        entry_id=i,
                        iss_port=j
                    )
                )

        # Spawn a process for the execute subcore stage 
        self.env.process(self._execute())

        # Spawn processes for the writeback subcore stage
        for i in range(self.config["num_fb_wb_port"]):
            self.env.process(self._writeback(i))

        # Spawn processes for the commit subcore stage 
        for i in range(self.config["num_fb_commit_port"]):
            self.env.process(self._commit(i))

        # Spawn a process to receive data from the bus
        self.env.process(self._receive_bus_data())
        
        # Trace logging flags
        self.trace_decode_fecth = []
        for i in range(self.config["max_num_warp_per_subcore"]):
            self.trace_decode_fecth.append(set())
        self.warp_instr_slot = []
        for i in range(self.config["max_num_warp_per_subcore"]):
            self.warp_instr_slot.append({})
        self.warp_free_slots = []
        for i in range(self.config["max_num_warp_per_subcore"]):
            self.warp_free_slots.append([])
        self._fetch_start_time = [0] \
            * self.config["max_num_warp_per_subcore"]
        
        # Performance metrics
        self.num_instr_executed = 0
        
        return

    def _get_instr_slot_id(self, new_pc, warp_id):
        return new_pc 

    def _delete_instr_slot(self, pc, warp_id):
        return 

    """
    def _get_instr_slot_id(self, new_pc, warp_id):
        if new_pc in self.warp_instr_slot[warp_id]:
            return self.warp_instr_slot[warp_id][new_pc]
        if (self.warp_free_slots[warp_id]):
            slot_id = deepcopy(self.warp_free_slots[warp_id][0])
            del self.warp_free_slots[warp_id][0]
        else:
            slot_id = len(self.warp_instr_slot[warp_id])
        self.warp_instr_slot[warp_id][new_pc] = slot_id
        return slot_id

    def _delete_instr_slot(self, pc, warp_id):
        slot_id = self.warp_instr_slot[warp_id][pc]
        self.warp_free_slots[warp_id].append(slot_id)
        del self.warp_instr_slot[warp_id][pc]
        return
    """

    def get_perf_metrics(self):
        """Get a dictionary of performance metrics."""
        perf_metrics = {}
        # Collect the performance metrics of this hardware module 
        perf_metrics["num_instr_executed"] = self.num_instr_executed 
        perf_metrics["num_instr_fd"] = self.num_instr_executed
        perf_metrics["num_instr_wbc"] = self.num_instr_executed

        # Collect the performance metrics of all hardware sub-modules
        # scoreboard
        dep_table_metrics = self.dep_table_exe.get_perf_metrics()
        assert len(dep_table_metrics) == 1
        perf_metrics.update(dep_table_metrics)
        # instruction offload engine
        instr_offload_engine_metrics = self.instr_offload_engine\
            .get_perf_metrics()
        assert len(instr_offload_engine_metrics) == 1
        perf_metrics.update(instr_offload_engine_metrics)
        # register track table
        reg_track_table_metrics = self.reg_track_table\
            .get_perf_metrics()
        assert len(reg_track_table_metrics) == 1
        perf_metrics.update(reg_track_table_metrics)
        # register file
        reg_file_metrics = self.reg_file.get_perf_metrics()
        assert len(reg_file_metrics) == 1
        perf_metrics.update(reg_file_metrics)
        # operand collector
        opc_fb_alu_metrics = self.opc_fb_alu.get_perf_metrics()
        assert len(opc_fb_alu_metrics) == 1
        opc_sfu_metrics = self.opc_sfu.get_perf_metrics()
        assert len(opc_sfu_metrics) == 1
        opc_lsu_metrics = self.opc_lsu.get_perf_metrics()
        assert len(opc_lsu_metrics) == 1
        opc_cfu_metrics = self.opc_cfu.get_perf_metrics()
        assert len(opc_cfu_metrics) == 1
        opc_syncu_metrics = self.opc_syncu.get_perf_metrics()
        assert len(opc_syncu_metrics) == 1
        fb_opc_metrics = {}
        fb_opc_metrics["num_read"] = \
            opc_fb_alu_metrics["fb_opc"]["num_read"] \
            + opc_sfu_metrics["fb_opc"]["num_read"] \
            + opc_lsu_metrics["fb_opc"]["num_read"] \
            + opc_cfu_metrics["fb_opc"]["num_read"] \
            + opc_syncu_metrics["fb_opc"]["num_read"]
        fb_opc_metrics["num_write"] = \
            opc_fb_alu_metrics["fb_opc"]["num_write"] \
            + opc_sfu_metrics["fb_opc"]["num_write"] \
            + opc_lsu_metrics["fb_opc"]["num_write"] \
            + opc_cfu_metrics["fb_opc"]["num_write"] \
            + opc_syncu_metrics["fb_opc"]["num_write"]
        perf_metrics.update({"fb_opc": fb_opc_metrics})
        # alu
        alu_metrics = self.fb_alu.get_perf_metrics()
        assert len(alu_metrics) == 1
        perf_metrics.update(alu_metrics)
        # sfu
        sfu_metrics = self.sfu.get_perf_metrics()
        assert len(sfu_metrics) == 1
        perf_metrics.update(sfu_metrics)
        # lsu
        lsu_metrics = self.lsu.get_perf_metrics()
        assert len(lsu_metrics) == 1
        perf_metrics.update(lsu_metrics)

        return {"subcore_{}".format(self.subcore_id): perf_metrics} 

    def get_trace_events(self):
        """Get a list of trace events"""
        _trace_Events = deepcopy(self.traceEvents)
        return _trace_Events

    def _append_trace_event_be(self, tid, name, ph, ts, cat="", args={}):
        new_event = {}
        new_event["pid"] = "proc_{}_core_{}_subcore_{}".format(
            self.core.processor.proc_id, self.core.core_id, self.subcore_id
        )
        new_event["tid"] = tid
        new_event["ph"] = ph
        new_event["name"] = name
        new_event["ts"] = ts / 1000
        if len(cat) > 0:
            new_event["cat"] = cat
        if len(args) > 0:
            new_event["args"] = args
        if self.filter_func(new_event):
            self.traceEvents.append(new_event)
        return

    def _append_trace_event_dur(self, tid, name, ts, dur, cat="", args={}):
        new_event = {}
        new_event["pid"] = "proc_{}_core_{}_subcore_{}".format(
            self.core.processor.proc_id, self.core.core_id, self.subcore_id
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

    def reset_status(self):
        """Reset hardware status of subcores.This function is usually used 
        to reset subcore status between different calls of executing thread 
        blocks. 
        """
        self.num_active_warps = 0
        self.reg_base_ptr = 0

        # Reset the tables in the subcore  
        self.warp_info_table.reset()
        self.warp_pipeline_table.reset() 
        self.stack_table.reset() 
        self.dep_table_exe.reset()
        self.reg_track_table.reset()
        self.pending_issue_instr = []
        for i in range(self.config["max_num_warp_per_subcore"]):
            self.pending_issue_instr.append({})
        self.num_issued_not_commit_instr = {}
        self.trace_decode_fecth = []
        for i in range(self.config["max_num_warp_per_subcore"]):
            self.trace_decode_fecth.append(set())
        self.warp_instr_slot = []
        for i in range(self.config["max_num_warp_per_subcore"]):
            self.warp_instr_slot.append({})
        self._free_slots = []
        return 

    def check_warp_usage(self, warp_usage):
        """Return whether the available slots in the warp table is sufficient 
        to accomodate a new thread block 
        """
        new_active_warps = self.num_active_warps + warp_usage 
        if new_active_warps > self.config["max_num_warp_per_subcore"]:
            return False 
        return True 

    def check_reg_usage(self, reg_usage_in_bytes):
        """Returen whether the available amount of register file is sufficient
        to accomodate a new thread block 
        """
        new_base_ptr = self.reg_base_ptr + reg_usage_in_bytes 
        if new_base_ptr > self.config["subcore_reg_file_size"]:
            return False 
        return True 

    def _load_from_icache(self, entry_id, pc):
        """This function loads the instruction from the instruction cache and 
        fill in the corresponding entry of the fetch table. 

        Args:
            entry_id: the position of the entry in the fetch table
            pc: the program counter of the instruction 
        """
        req = InstrLoadReq(
            subcore_id=self.subcore_id, entry_id=entry_id, pc=pc
        )
        yield self.core.icache.load_req_queque.put(req)
        self.log.debug(
            "{loc} at {time_stamp} cycle: Issue load for the "
            "instr with PC={pc} from the entry {entry_id}".format(
                loc=self._loc_str, time_stamp=self.env.now, pc=pc, 
                entry_id=entry_id
            )
        )

        resp = yield self.core.icache.load_resp_queue.get(
            lambda x: ((x.subcore_id == self.subcore_id) 
                       and (x.entry_id == entry_id) and (x.pc == pc)) 
        )
        self.log.debug(
            "{loc} at {time_stamp} cycle: Get the instruction "
            "with PC={pc} for the entry {entry_id}".format(
                loc=self._loc_str, time_stamp=self.env.now,
                pc=pc, entry_id=entry_id,
            )
        )

        self.warp_pipeline_table.entry[entry_id].instr = resp.instr 
        self.warp_pipeline_table.entry[entry_id].valid = True 

        return 

    def _need_fetch_stall(self, instr):
        """Check if the fetch stage needs to be stalled waiting for the execution
        of the current instruction. We will stall the fetch stage on any 
        control flow instructions or synchronization instructions. 

        Args:
            instr: the instruction passed to be checked. 

        Returns:
            A boolean value, True or False, to indicate whether the fetch stage
                needs to be stalled. 
        """
        # Check whether it is a control flow instruction 
        if "dst_pc" in instr.metadata:
            return True 
        
        # Check whether it is a barrier instruction 
        if "bar_id" in instr.metadata: 
            return True 

        return False 

    def _fetch_and_decode(self):
        """This function executes the fetch and decode stages of the subcore 
        """
        while True:
            start_exec = yield self.start_exec_cmd.get() 
            assert start_exec == "start", "Unrecognized " \
                "commands: {}".format(start_exec) 

            all_finish = self.warp_pipeline_table\
                .check_all_finished(self.num_active_warps)
            if all_finish:
                self.finish_exec_resp.put("success")
                continue 

            curr_ptr = 0
            instr_fetch_id = 0
            while True:
                # Step 1: check the exit condition
                all_finish = self.warp_pipeline_table\
                    .check_all_finished(self.num_active_warps)
                if all_finish is True:
                    break 

                curr_entry = deepcopy(self.warp_pipeline_table.entry[curr_ptr])

                # Step 2: skip the current warp if it is finished 
                if curr_entry.warp_finished:
                    curr_ptr = (curr_ptr + 1) % self.num_active_warps 
                    yield self.env.timeout(1 * self.clock_unit)
                    continue 

                # Step 3: check whether the current instruction is valid
                if not curr_entry.valid: 
                    if not curr_entry.issued_to_icache: 
                        self.warp_pipeline_table.entry[curr_ptr]\
                            .issued_to_icache = True
                        self.env.process(
                            self._load_from_icache(curr_ptr, curr_entry.pc)
                        )
                        # for trace event
                        self._fetch_start_time[curr_ptr] = self.env.now
                        # change to next warp
                        curr_ptr = (curr_ptr + 1) % self.num_active_warps 
                    yield self.env.timeout(1 * self.clock_unit)
                    continue
                # entry is valid now
                if (
                    curr_entry.pc not in self.trace_decode_fecth[curr_ptr]
                    and self.warp_pipeline_table.entry[curr_ptr]
                    .issued_to_decode is False
                ):
                    slot_id = self._get_instr_slot_id(
                        new_pc=curr_entry.pc,
                        warp_id=curr_ptr
                    )
                    fetch_dur = \
                        self.env.now - self._fetch_start_time[curr_ptr]
                    self._append_trace_event_be(
                        tid="warp_{}-slot_{}".format(curr_ptr, slot_id),
                        name="[f/d]-{}"
                        .format(
                            self.core.current_kernel
                            .instr_list[curr_entry.pc]
                            .trace_str()
                        ),
                        ph="B",
                        ts=self.env.now,
                        cat="instruction",
                        args={
                            "fetch_dur": fetch_dur,
                            "pc": curr_entry.pc
                        }
                    )
                    self.trace_decode_fecth[curr_ptr].add(curr_entry.pc)

                # Step 4: push into the next stage if resources are available 
                if not curr_entry.issued_to_decode:
                    if self.num_free_decode_buffer_slots[curr_ptr] > 0:
                        self.num_free_decode_buffer_slots[curr_ptr] = (
                            self.num_free_decode_buffer_slots[curr_ptr] - 1
                        )
                        pc = deepcopy(curr_entry.pc)
                        simt_mask = deepcopy(
                            self.stack_table.entry[curr_ptr].get_simt_mask() 
                        )
                        instr_id = deepcopy(instr_fetch_id)
                        instr_fetch_id += 1

                        yield self.decode_buffer[curr_ptr].put(
                            (pc, simt_mask, curr_entry.instr, instr_id)
                        )

                        self.warp_pipeline_table.entry[
                            curr_ptr].issued_to_decode = True
                    else:
                        if curr_entry.skip_resource_contention: 
                            self.warp_pipeline_table.entry[
                                curr_ptr].skip_resource_contention = False 
                            curr_ptr = (curr_ptr + 1) % self.num_active_warps 
                        yield self.env.timeout(1 * self.clock_unit) 

                    continue 

                # Step 5: stall the fetch stage if needed 
                if self._need_fetch_stall(curr_entry.instr):
                    if not curr_entry.executed:
                        curr_ptr = (curr_ptr + 1) % self.num_active_warps 

                        yield self.env.timeout(1 * self.clock_unit) 
                        continue 

                # Step 6: check the SIMT stack to get the correct next PC
                next_pc = curr_entry.next_pc 
                while self.stack_table.entry[curr_ptr].check_converge(next_pc):
                    self.stack_table.entry[curr_ptr].pop() 
                    top_entry = self.stack_table.entry[curr_ptr].top()
                    next_pc = top_entry[1] 
                    yield self.env.timeout(1 * self.clock_unit)

                if next_pc >= self.warp_info_table.entry[curr_ptr].prog_length:
                    self.warp_pipeline_table\
                        .entry[curr_ptr].warp_finished = True 

                    yield self.env.timeout(1 * self.clock_unit)
                    continue 

                # Step 7: generate a new entry for the next PC
                new_warp_pipeline_table_entry = \
                    WarpPipelineTableEntry(pc=next_pc)
                self.warp_pipeline_table.entry[curr_ptr] = \
                    deepcopy(new_warp_pipeline_table_entry) 

        return

    def _issue(self, entry_id, iss_port):
        """This function executes the issue stage of subcore including 
        checking operand dependencies and send instructions to the next 
        stage 
        """
        while True:
            instr_tuple = yield self.decode_buffer[entry_id].get()
            self.num_instr_executed += 1
            self.num_free_decode_buffer_slots[entry_id] += 1
            current_pc = instr_tuple[0]
            simt_mask = instr_tuple[1]
            instr = instr_tuple[2]
            curr_stamp = instr_tuple[3]
            block_id = self.warp_info_table.entry[entry_id].block_id
            self.log.debug(
                "{loc} Entry {entry_id} start checking the "
                "dependency of {instr} at {time_stamp} cycle".format(
                    loc=self._loc_str, entry_id=entry_id,
                    instr=instr.instr_str, time_stamp=self.env.now
                )
            )

            # update trace event
            slot_id = self._get_instr_slot_id(
                new_pc=current_pc,
                warp_id=entry_id
            )
            self._append_trace_event_be(
                tid="warp_{}-slot_{}".format(entry_id, slot_id),
                name="[f/d]-{}"
                .format(instr.trace_str()),
                ph="E",
                ts=self.env.now,
                cat="instruction"
            )
            self.trace_decode_fecth[entry_id].remove(current_pc)
            last_trace_cyc = self.env.now

            src_ops = []
            dst_ops = []

            if "pred_reg" in instr.metadata:
                src_ops.append(instr.metadata["pred_reg"].op_str)

            for each_op in instr.src_operands:
                if each_op.isreg():
                    src_ops.append(each_op.op_str)
            # self.log.debug("source operands: {}".format(src_ops))

            for each_op in instr.dst_operands:
                if each_op.isreg():
                    dst_ops.append(each_op.op_str) 
            # self.log.debug("destination operands: {}".format(dst_ops))

            # Check whether the current instruction depends on any prior
            # pending instructions
            self.pending_issue_instr[entry_id][iss_port] = (
                curr_stamp, src_ops, dst_ops)

            src_op_set = set(src_ops)
            dst_op_set = set(dst_ops)
            dep_pc_set = set() 

            while True:
                break_cond = True 

                for k, v in self.pending_issue_instr[entry_id].items():
                    stamp = v[0]
                    if stamp >= curr_stamp:
                        continue

                    if stamp in dep_pc_set:
                        break_cond = False
                        break 

                    # WAR dependency 
                    for instr_src_op in v[1]:
                        if instr_src_op in dst_op_set:
                            dep_pc_set.add(stamp)
                            break_cond = False 
                    # RAW dependency 
                    for instr_dst_op in v[2]:
                        if instr_dst_op in src_op_set:
                            dep_pc_set.add(stamp)
                            break_cond = False 
                    # WAW dependency
                    for instr_dst_op in v[2]:
                        if instr_dst_op in dst_op_set:
                            dep_pc_set.add(stamp)
                            break_cond = False 

                if break_cond:
                    break 
                else:
                    yield self.env.timeout(1 * self.clock_unit) 

            # NOTE: this will stall instructions which has 
            # execution dependency
            dependency_stall_start = self.env.now
            # Check the dependency table until all dependencies are cleared
            while True:
                break_cond = True 
                # check RAW dependency
                for each_op in src_ops:
                    if (
                        not self.dep_table_exe.entry[entry_id]
                        .check_read(each_op)
                    ):
                        break_cond = False 
                        break 
                # check WAW and WAR dependency
                for each_op in dst_ops:
                    if (
                        not self.dep_table_exe.entry[entry_id]
                        .check_write(each_op)
                    ):
                        break_cond = False 
                        break 
                if break_cond:
                    # update performance counter
                    self.dep_table_exe.num_read += \
                        len(src_ops) + len(dst_ops)
                    break
                else:
                    # check dependency again in the next cycle
                    yield self.env.timeout(1 * self.clock_unit)
            # If this is the last instruction of the kernel,
            # make sure that all previous instructions are committed
            while True:
                break_cond = True
                prog_length = \
                    self.warp_info_table.entry[entry_id].prog_length
                if (current_pc + 1) == prog_length:
                    # this is the last instruction
                    if (
                        self.num_issued_not_commit_instr[block_id][entry_id]
                            > 0
                    ):
                        # some previous instructions are not committed
                        break_cond = False

                if break_cond:
                    break
                else:
                    # check break condition in the next cycle
                    yield self.env.timeout(1 * self.clock_unit)

            dependency_stall = self.env.now - dependency_stall_start
            if dependency_stall == 0:
                # NOTE: checking dependency table and checking break condition
                # are parallel and consume one pipeline cycle
                yield self.env.timeout(1 * self.clock_unit)
            
            self.log.debug(
                "{loc} Entry {entry_id} finished checking the "
                "dependency of {instr} at {time_stamp} cycle".format(
                    loc=self._loc_str, entry_id=entry_id,
                    instr=instr.instr_str, time_stamp=self.env.now
                )
            )

            # Update the dependency table to pend other instructions 
            for each_op in src_ops:
                self.dep_table_exe.entry[entry_id].increase_read(each_op) 
            for each_op in dst_ops:
                self.dep_table_exe.entry[entry_id].increase_write(each_op)
            # Update issue count
            self.num_issued_not_commit_instr[block_id][entry_id] += 1
            # NOTE: recording register dependencies and recording 
            # instruction issue count are in parallel with
            # instruction issue process, so this does not consume cycle

            # compose an instance of instruction
            instr_entry = InstrEntry(
                log=self.log,
                config=self.config,
                instr=instr,
                simt_mask=simt_mask,
                pc=current_pc,
                subcore_id=self.subcore_id,
                warp_id=entry_id
            )

            # update trace event
            instr_entry.slot_id = slot_id
            self._append_trace_event_dur(
                tid="warp_{}-slot_{}".format(instr_entry.warp_id, slot_id),
                name="[iss]-{}".format(instr.trace_str()),
                ts=last_trace_cyc,
                dur=self.env.now - last_trace_cyc,
                cat="instruction",
                args={
                    "dependency_stall": dependency_stall
                }
            )
            instr_entry.last_trace_cyc = self.env.now

            yield self.execute_buffer.put(instr_entry) 

            # Remove the instruction from the buffer of instructions pended at
            # the issue stage
            del self.pending_issue_instr[entry_id][iss_port]
        return 

    def get_subcore_reg_addr(self, reg_prefix, reg_index, entry_id):
        """This function calculates the absolute addrss of a register in the 
        register file of subcore. It also returns the size of registers with 
        the same name across threads in the whole warp.  
        
        Args:
            reg_prefix: the prefix of register name 
            reg_index: the index of the register 
            entry_id: the warp ID of this register 

        Returns:
            (reg_addr, reg_size): the stating address of this register in the 
                register file and the size of the whole register 
        """
        reg_base_addr = self.warp_info_table\
            .entry[entry_id].subcore_reg_base_addr 
        prefix_reg_base_addr = (
            reg_base_addr 
            + self.warp_info_table.entry[entry_id].prog_reg_offset[reg_prefix]
        )
        reg_size = self.warp_info_table\
            .entry[entry_id].prog_reg_size[reg_prefix]
        reg_addr = prefix_reg_base_addr + reg_index * reg_size

        self.log.debug(
            "{loc} Entry {entry_id} {reg_prefix}{reg_index} " 
            "starting from {reg_addr} with size {reg_size}".format(
                loc=self._loc_str, entry_id=entry_id, 
                reg_prefix=reg_prefix, reg_index=reg_index,
                reg_addr=reg_addr, reg_size=reg_size 
            )
        )
        return (reg_addr, reg_size)

    def get_param_value(self, param_name, entry_id):
        """This function gets the value of parameters including kernel function
        arguments and shared memory parameter. 

        Args:
            param_name: the parameter name
            entry_id: the entry ID the current warp. This ID is used to locate 
                shared memory base address and offset for the dynamically 
                allocated shared memory space 

        Returns:
            param_value: the value of parameter in its corresponding type 
        """
        prog_smem_offset = \
            self.warp_info_table.entry[entry_id].prog_smem_offset
        if param_name in self.core.param_dict:
            return self.core.param_dict[param_name]
        elif param_name in prog_smem_offset:
            base_addr = self.warp_info_table.entry[entry_id].smem_base_addr
            offset = (
                self.warp_info_table
                .entry[entry_id].prog_smem_offset[param_name]
            ) 
            return base_addr + offset 
        else: 
            raise NotImplementedError(
                "Unknown parameter:{}".format(param_name)
            )
        return 

    def get_special_reg_value(self, reg_name, entry_id):
        """This function is used to get the value of special registers storing 
        SIMT thread information, such as block ID and thread ID. 

        Args:
            reg_name: the name of special register 
            entry_id: the entry ID of the current warp so that block ID and 
                thread ID can be accurately located. 

        Returns:
            reg_value: an integer if the requested registr has the same value 
                for all threads in the same warp, or a list of values for 
                different values of threads in the same warp. 
        """
        index_mapper = {"z": 0, "y": 1, "x": 2}
        reg_prefix = reg_name.split(".")[0]
        reg_index = index_mapper[reg_name.split(".")[1]]

        if reg_prefix == "%ntid":
            return self.core.block_dim[reg_index]
        elif reg_prefix == "%ctaid":
            return self.warp_info_table.entry[entry_id].block_id[reg_index] 
        elif reg_prefix == "%tid":
            start_id = self.warp_info_table.entry[entry_id].thread_id[reg_index]
            if reg_index != 2:
                return start_id 
            else: 
                value_list = list(range(
                    start_id, start_id + self.config["num_threads_per_warp"]))
                return value_list 
        elif reg_prefix == "%nctaid":
            return self.core.grid_dim[reg_index] 
        else:
            raise NotImplementedError(
                "Unknown special register: {}".format(reg_name)
            )
        return 

    def _receive_bus_data(self):
        """This function receives data from subcore-pg buses
        """
        while True:
            packet = yield self.bus_receive_buffer.get()
            if isinstance(packet, RegMoveReq):
                if packet.ld_st_flag:
                    yield self.lsu\
                        .reg_req_bus_queue.put(packet)
                else:
                    yield self.reg_move_engine\
                        .reg_req_bus_queue.put(packet)
            elif isinstance(packet, InstrEntry):
                # update trace event
                self._append_trace_event_dur(
                    tid="warp_{}-slot_{}"
                    .format(
                        packet.warp_id,
                        packet.slot_id
                    ),
                    name="[tsv-down]-{}"
                    .format(
                        packet.instr.trace_str()
                    ),
                    ts=packet.last_trace_cyc,
                    dur=self.env.now - packet.last_trace_cyc,
                    cat="instruction, bus",
                    args={
                        "pg_id": packet.pg_id
                    }
                )
                packet.last_trace_cyc = self.env.now
                yield self.commit_buffer.put(packet)
            elif isinstance(packet, DRAMTransaction):
                yield self.lsu.in_dram_trans_queue.put(packet)
            else:
                assert False

    def _execute(self):
        """This function executes the functionality of the execute stage of 
        the subcore including instruction offloading, register movement,
        reading values from registers, compute results, 
        and pass the instruction with results to the write back stage. 
        """
        while True:
            instr_entry = yield self.execute_buffer.get()
            yield self.instr_offload_engine\
                .instr_entry_queue.put(instr_entry)
        return

    def _writeback(self, local_regfile_write_port_id_commit):
        """This stage executes the writeback stage of the subcore. 
        Write results of destination operands to register file.
        """
        regfile_write_port_id = \
            self.base_regfile_write_port_id_commit \
            + local_regfile_write_port_id_commit
        while True:
            instr_entry = yield self.writeback_buffer.get()
            simt_mask = instr_entry.simt_mask
            entry_id = instr_entry.warp_id
            instr = instr_entry.instr
            
            for i in range(len(instr.dst_operands)):
                dst_op = instr.dst_operands[i]
                reg_addr, reg_size = self.get_subcore_reg_addr(
                    dst_op.reg_prefix, dst_op.reg_index, entry_id
                )

                if simt_mask > 0:
                    # Write results back to register file
                    operand_write_req = OperandWriteReq(
                        base_reg_addr=reg_addr,
                        total_reg_size=reg_size,
                        simt_mask=simt_mask,
                        data=instr_entry.dst_values[i]
                    )
                    yield self.rf_io_interface\
                        .write_req_queue[regfile_write_port_id]\
                        .put(operand_write_req)

                    _ = yield self.rf_io_interface.write_resp_queue.get(
                        lambda x: (x.base_reg_addr == reg_addr
                                   and x.total_reg_size == reg_size)
                    )
                # NOTE: clear write dependency here
                self.dep_table_exe.entry[entry_id]\
                    .decrease_write(dst_op.op_str)
            
            if len(instr.dst_operands) == 0:
                # NOTE: consume 1 pipeline cycle
                yield self.env.timeout(1 * self.clock_unit)
            # push to commit buffer
            yield self.commit_buffer.put(instr_entry)
            # update trace event
            self._append_trace_event_dur(
                tid="warp_{}-slot_{}"
                .format(instr_entry.warp_id, instr_entry.slot_id),
                name="[fb-wb]-{}".format(instr.trace_str()),
                ts=instr_entry.last_trace_cyc,
                dur=self.env.now - instr_entry.last_trace_cyc,
                cat="instruction",
                args={
                    "wb_port": local_regfile_write_port_id_commit
                }
            )
            instr_entry.last_trace_cyc = self.env.now

    def _commit(self, commit_port_id):
        """This stage executes the commit stage of the subcore including
        releasing the register dependency, release the instruciton stall 
        on the fetch stage, and check whether the current batch of thread 
        blocks finishes the execution.
        """
        while True:
            instr_entry = yield self.commit_buffer.get()
            current_pc = instr_entry.pc
            entry_id = instr_entry.warp_id
            instr = instr_entry.instr

            # NOTE: consume 1 pipeline cycle
            yield self.env.timeout(1 * self.clock_unit)
            
            # update trace event
            self._append_trace_event_dur(
                tid="warp_{}-slot_{}"
                .format(instr_entry.warp_id, instr_entry.slot_id),
                name="[commit]-{}".format(instr.trace_str()),
                ts=instr_entry.last_trace_cyc,
                dur=self.env.now - instr_entry.last_trace_cyc,
                cat="instruction",
                args={
                    "commit_port": commit_port_id
                }
            )
            self._delete_instr_slot(
                pc=instr_entry.pc,
                warp_id=instr_entry.warp_id
            )

            assert instr_entry.processed is True, "Can not pass an " \
                "unprocessed instruction to the writeback stage"
            
            if current_pc == self.warp_pipeline_table.entry[entry_id].pc:
                self.warp_pipeline_table.entry[entry_id].executed = True 
            
            block_id = self.warp_info_table.entry[entry_id].block_id
            self.num_issued_not_commit_instr[block_id][entry_id] -= 1
            assert self.num_issued_not_commit_instr[block_id][entry_id] >= 0
            
            prog_length = \
                self.warp_info_table.entry[entry_id].prog_length
            if (current_pc + 1) == prog_length:
                self.warp_info_table.entry[entry_id].warp_finished = True
                assert instr.opcode == "ret", "The last instruction is not" \
                    "the return instruction!"

                all_finish = \
                    self.warp_info_table\
                    .check_all_finished(self.num_active_warps)
                if all_finish is True:
                    yield self.finish_exec_resp.put("success") 

        return 
