from copy import deepcopy
import simpy

from simulator.execution_unit import ExecutionUnit
from simulator.instr_instance import InstrEntry
from simulator.shared_memory import SMEMReadReq, SMEMReadResp, \
    SMEMWriteReq, SMEMWriteResp, SMEMAtomReq, SMEMAtomResp 
from simulator.dram_message import DRAMTransaction, PRTEntryReq
from simulator.network_message import SrcRemoteLoadReq, SrcRemoteStoreReq, \
    SrcRemoteLoadResp, SrcRemoteStoreResp
from simulator.reg_move_engine import RegMoveReq


class LoadStoreUnit(ExecutionUnit):
    
    def __init__(self, env, log, config, clock_unit, subcore):
        super().__init__(env, log, config, clock_unit)
        self.subcore = subcore
        self.name = "lsu"
        # reference the address hashing function in sim_api
        self.addr_hashing = self.subcore.core.processor\
            .hardware.addr_hashing
        self.re_addr_hashing = self.subcore.core.processor\
            .hardware.re_addr_hashing
        self.translate_bank_addr = self.subcore.core.processor\
            .hardware.translate_bank_addr
        # reference to local processor and core id
        self.local_proc_id = self.subcore.core.processor.proc_id
        self.local_core_id = self.subcore.core.core_id
        # reference to bus arbiter
        self.bus_arbiter = self.subcore.core.subcore_pg_bus_arbiter
        # reference to subcore's trace event
        self._append_trace_event_dur = subcore._append_trace_event_dur
        # get uniform simt_mask
        self.all_1s = int(
            "1" * self.config["num_threads_per_warp"], 2
        )
        # spawn a process for each lsu
        self.num_unit = self.config["num_lsu"]
        for i in range(self.num_unit):
            self.env.process(self._process_instr_entry())
        # queues for receiving local ld.global/st.global request
        self.local_ld_st_global_entry_queue = simpy.Store(env, capacity=1)
        self.num_fb_local_prt_entry = \
            self.config["max_num_fb_local_prt_entry"]
        # spawn processes to process local ld.global/st.global request
        for local_fb_prt_id in range(self.num_fb_local_prt_entry):
            if self.config["bypass_ld_st_global_backend"]:
                self.env.process(
                    self._process_ld_st_bypass_backend()
                )
            else:
                self.env.process(
                    self._process_local_ld_st_global(local_fb_prt_id)
                )
        # queues for receiving returned dram transactions
        self.in_dram_trans_queue = simpy.FilterStore(env)
        # queues for receiving returned reg move request
        self.reg_req_bus_queue = simpy.FilterStore(env)
        # queues for processing ld/st that bypasses backend
        self.ld_st_global_bypass_entry_queue = simpy.Store(env, capacity=1)
        
        # queues for receiving ld.shared/st.shared/atom.shared request
        self.ld_st_shared_entry_queue = simpy.Store(env)
        # spawn processes to process ld/st/atom shared request
        for i in range(self.config["lsu_shared_issue_port"]):
            self.env.process(self._process_ld_st_shared())

        # performance counter
        self.num_prt_read = 0
        self.num_prt_write = 0

    def get_perf_metrics(self):
        """Get a dictionary of performance metrics."""
        perf_metrics = {}
        perf_metrics["num_prt_read"] = self.num_prt_read
        perf_metrics["num_prt_write"] = self.num_prt_write
        return {"lsu": perf_metrics}

    def _check_remote_access(self, mem_loc):
        """Check if the given memory location is a remote access
        or a local access
        Args:
            mem_loc: memory location tuple formated as
                (proc_id_y, proc_id_x, core_id_y, core_id_x, pg_id, pe_id,
                bank_addr, bank_interface_offset)
        Return:
            is_remote: True if the memory location is a remote access
        """
        dst_proc_id = (mem_loc[1], mem_loc[0])
        dst_core_id = (mem_loc[3], mem_loc[2])
        if (
            dst_proc_id == self.local_proc_id
            and dst_core_id == self.local_core_id
        ):
            return False
        else:
            return True

    def _memory_range_check(self, simt_mask, mem_loc_list):
        """Check memory range
        Args:
            simt_mask: simt mask
            mem_loc_list: a list of location tuples formatted as
                (proc_id_y, proc_id_x, core_id_y, core_id_x, pg_id, pe_id,
                bank_addr, bank_interface_offset)
        Return:
            local_simt_mask: simt_mask for local address list
            remote_simt_mask: simt_mask for remote address list
        """
        local_simt_mask = 0
        remote_simt_mask = 0
        for i in range(self.config["num_threads_per_warp"]):
            valid = (simt_mask >> i) & 1
            if valid:
                mem_loc = mem_loc_list[i]
                assert mem_loc is not None
                is_remote = self._check_remote_access(mem_loc)
                if is_remote:
                    remote_simt_mask += 1 << i
                else:
                    local_simt_mask += 1 << i
        return local_simt_mask, remote_simt_mask

    def _memory_coalesce(
        self, simt_mask, mem_loc_list, is_ld, data_width, instr_entry
    ):
        """Perform memory coalescing
        Args:
            simt_mask: simt mask
            mem_loc_list: a list of location tuples formatted as
                (proc_id_y, proc_id_x, core_id_y, core_id_x, pg_id, pe_id,
                bank_addr, bank_interface_offset)
            is_ld: True if is load
            data_width: data width of a single access
            instr_entry: for tracing DRAMTransaction
        Return:
            offset_list: offset into bank interface
            co_addr_list: coalesced dram transaction
        """
        trans_type = "load" if is_ld else "store"
        offset_list = [None] * self.config["num_threads_per_warp"]
        # coalesced address list
        # global_mem_addr (aligned to bank interface) 
        #   -> dram transaction
        co_addr_list = {}
        for i in range(self.config["num_threads_per_warp"]):
            valid = (simt_mask >> i) & 1
            if valid:
                proc_id_y, proc_id_x, core_id_y, core_id_x, pg_id, pe_id,\
                    bank_addr, bank_interface_offset = mem_loc_list[i]
                # update offset
                offset_list[i] = bank_interface_offset
                assert offset_list[i] + data_width \
                    <= self.config["dram_bank_io_width"], \
                    "Access cannot cross bank interface boundary"
                # get aligned memory address
                global_mem_addr = self.re_addr_hashing(
                    (
                        proc_id_y, proc_id_x, core_id_y, core_id_x, 
                        pg_id, pe_id, bank_addr, 0
                    )
                )
                if global_mem_addr in co_addr_list:
                    # merge into an existing transaction
                    dram_trans = co_addr_list[global_mem_addr]
                    dram_trans.simt_mask += 1 << i
                else:
                    # create a new transaction
                    # get bank internal address
                    row_addr, col_addr = self.translate_bank_addr(
                        bank_addr
                    )
                    # compose a dram transaction
                    dram_trans = DRAMTransaction(
                        trans_type=trans_type,
                        mem_loc=mem_loc_list[i],
                        row_addr=row_addr,
                        col_addr=col_addr,
                        global_mem_addr=global_mem_addr
                    )
                    dram_trans.trace_subcore_id = instr_entry.subcore_id
                    dram_trans.trace_warp_id = instr_entry.warp_id
                    dram_trans.simt_mask += 1 << i
                    # add to list
                    co_addr_list[global_mem_addr] = dram_trans
        return offset_list, co_addr_list

    def _pg_loc_check(self, pg_id, co_addr_list):
        """Check if register's pg_id matches dram transactions' pg_id
        Args:
            pg_id: PG ID where the register locates
            co_addr_list: coalesced memory address list
        Return:
            is_diverge_pg_loc: True if divergence happens
        """
        for mem_addr in co_addr_list:
            dram_trans = co_addr_list[mem_addr]
            if pg_id != dram_trans.pg_id:
                return True
        return False

    def _eliminate_write_conflict(self, instr_entry, is_ld):
        if is_ld: 
            return
        addr_list = instr_entry._decoded_src_values[0]
        addr_set = set()
        for i in range(self.config["num_threads_per_warp"]):
            valid = (instr_entry.simt_mask >> i) & 1
            if valid:
                if addr_list[i] not in addr_set:
                    addr_set.add(addr_list[i])
                else:
                    # NOTE: this is a write conflict, we will
                    # prevent this data to be written
                    instr_entry.simt_mask -= 1 << i

    def _process_local_ld_st_global(self, prt_id):
        while True:
            instr_entry = yield self.local_ld_st_global_entry_queue.get()
            
            # NOTE: currently we only support 32b
            num_bits = int(instr_entry.instr.opcode.split(".")[-1][1:])
            assert num_bits == 32
            data_width = 4
            if instr_entry.instr.opcode.startswith("ld.global"):
                is_ld = True
            elif instr_entry.instr.opcode.startswith("st.global"):
                is_ld = False
            else:
                raise NotImplementedError(
                    "Wrong ld/st operation: {}"
                    .format(instr_entry.instr.opcode)
                )
            # NOTE: perform write conflict elimination
            self._eliminate_write_conflict(instr_entry, is_ld)
            simt_mask = instr_entry.simt_mask
            addr_list = instr_entry._decoded_src_values[0]
            warp_id = instr_entry.warp_id
            pg_id = self.subcore.warp_info_table.entry[warp_id].pg_id
            # each decoded address has the format:
            #   proc_id_y, proc_id_x, core_id_y, core_id_x, pg_id, pe_id,
            #   bank_addr, bank_interface_offset
            mem_loc_list = [None] * self.config["num_threads_per_warp"]
            for i in range(self.config["num_threads_per_warp"]):
                valid = (simt_mask >> i) & 1
                if valid:
                    mem_loc_list[i] = self.addr_hashing(addr_list[i])

            # step 1: (1) simt_mask checking, (2) addr continuity checking
            # simt_mask checking
            is_simt_uniform = False
            if simt_mask == self.all_1s:
                is_simt_uniform = True
            # addr continuity checking
            is_addr_cont = True
            if is_simt_uniform:
                for i in range(self.config["num_threads_per_warp"] - 1):
                    if addr_list[i] != addr_list[i + 1] - data_width:
                        is_addr_cont = False
                        break
            else:
                is_addr_cont = False
            
            # step 2: memory range checking
            local_simt_mask, remote_simt_mask = self._memory_range_check(
                simt_mask=simt_mask,
                mem_loc_list=mem_loc_list
            )
            # prepare remote address list
            remote_addr_list = [None] * self.config["num_threads_per_warp"]
            if remote_simt_mask > 0:
                for i in range(self.config["num_threads_per_warp"]):
                    valid = (remote_simt_mask >> i) & 1
                    if valid:
                        remote_addr_list[i] = addr_list[i]
            # NOTE: consume 1 pipeline cycle
            yield self.env.timeout(1 * self.clock_unit)
            
            # step 3: memory coalescing for local access
            offset_list, co_addr_list = self._memory_coalesce(
                simt_mask=local_simt_mask,
                mem_loc_list=mem_loc_list,
                is_ld=is_ld,
                data_width=data_width,
                instr_entry=instr_entry
            )
            # NOTE: consume 1 pipeline cycle
            yield self.env.timeout(1 * self.clock_unit)

            # step 4: PG location checking
            is_diverge_pg_loc = self._pg_loc_check(
                pg_id=pg_id,
                co_addr_list=co_addr_list
            )

            # step 5: PRT entry offloading decision making
            if (
                is_simt_uniform is True
                and is_addr_cont is True
                and remote_simt_mask == 0
                and is_diverge_pg_loc is False
            ):
                # offload to near-bank PRT
                prt_entry = PRTEntryReq(
                    instr_entry=instr_entry,
                    offset_list=offset_list,
                    co_addr_list=co_addr_list,
                    pg_id=pg_id
                )
                # issue to pg, upstream traffic
                yield self.bus_arbiter\
                    .upstream_req_queue.put(prt_entry)
                # update trace event
                self._append_trace_event_dur(
                    tid="warp_{}-slot_{}"
                    .format(
                        instr_entry.warp_id,
                        instr_entry.slot_id
                    ),
                    name="[lsu]-{}"
                    .format(
                        instr_entry.instr.trace_str()
                    ),
                    ts=instr_entry.last_trace_cyc,
                    dur=self.env.now - instr_entry.last_trace_cyc,
                    cat="instruction"
                )
                instr_entry.last_trace_cyc = self.env.now
                # NOTE: can accept another instruction in the
                # next cycle
                yield self.env.timeout(1 * self.clock_unit)
            else:
                if is_ld:
                    # load instruction data path
                    # set PG ID for final writeback
                    # NOTE: this is blocking
                    instr_entry.pg_id = pg_id
                    yield self.env.process(
                        self._handle_ld_instr_fb(
                            instr_entry=instr_entry,
                            offset_list=offset_list,
                            co_addr_list=co_addr_list,
                            remote_addr_list=remote_addr_list,
                            remote_simt_mask=remote_simt_mask,
                            data_width=data_width,
                            prt_id=prt_id
                        )
                    )
                else:
                    # store instruction data path
                    # NOTE: this is blocking
                    yield self.env.process(
                        self._handle_st_instr_fb(
                            instr_entry=instr_entry,
                            offset_list=offset_list,
                            co_addr_list=co_addr_list,
                            remote_addr_list=remote_addr_list,
                            remote_simt_mask=remote_simt_mask,
                            data_width=data_width,
                            warp_id=warp_id,
                            pg_id=pg_id,
                            prt_id=prt_id
                        )
                    )
            # update performance couter
            self.num_prt_read += 1
            self.num_prt_write += 1

    def _issue_ld_local(
        self, data_buffer, co_addr_list, offset_list, 
        data_width, prt_id, instr_entry
    ):
        start_time = self.env.now
        # issue local dram transactions
        for mem_addr in co_addr_list:
            dram_trans = co_addr_list[mem_addr]
            dram_trans.subcore_id = self.subcore.subcore_id
            dram_trans.is_nb = False
            dram_trans.prt_id = prt_id
            dram_trans.time = self.env.now
            # NOTE pg_id is part of dram_trans initialization
            # so we don't need to set here
            # issue to pg, upstream traffic
            yield self.bus_arbiter\
                .upstream_req_queue.put(dram_trans)
        # wait until request return
        for mem_addr in co_addr_list:
            dram_trans = yield self.in_dram_trans_queue.get(
                lambda x: (
                    x.type == "load"
                    and x.global_mem_addr == mem_addr
                    and x.is_nb is False
                    and x.prt_id == prt_id
                )
            )
            assert dram_trans.data is not None
            assert dram_trans.simt_mask > 0
            for i in range(self.config["num_threads_per_warp"]):
                valid = (dram_trans.simt_mask >> i) & 1
                if valid:
                    trans_start_addr = offset_list[i]
                    trans_end_addr = trans_start_addr + data_width
                    assert trans_end_addr <= self.config["dram_bank_io_width"]
                    db_start_addr = i * data_width
                    db_end_addr = db_start_addr + data_width
                    data_buffer[db_start_addr: db_end_addr] = \
                        dram_trans.data[trans_start_addr: trans_end_addr]
                    """
                    addr = instr_entry._decoded_src_values[0][i]
                    db = self.subcore.core.processor.hardware.mem\
                        .get_value(addr, 4)
                    assert db == data_buffer[db_start_addr: db_end_addr]
                    """
            # NOTE: consume 1 pipeline cycle
            yield self.env.timeout(1 * self.clock_unit)
        instr_entry.local_access_cyc = self.env.now - start_time

    def _issue_ld_remote(
        self, data_buffer, remote_simt_mask, remote_addr_list, 
        data_width, prt_id, instr_entry
    ):
        start_time = self.env.now
        # issue remote req
        if remote_simt_mask > 0:
            # compose a remote load request
            ld_req = SrcRemoteLoadReq(
                addr_list=remote_addr_list,
                data_width=data_width,
                simt_mask=remote_simt_mask
            )
            ld_req.prt_id = prt_id
            # issue remote load request
            yield self.subcore.core.niu.req_queue.put(ld_req)
            # wait for remote load response
            ld_resp = yield self.subcore.core.niu.resp_queue.get(
                lambda x: (
                    isinstance(x, SrcRemoteLoadResp)
                    and x.addr_list == ld_req.addr_list
                    and remote_simt_mask == x.simt_mask
                    and x.prt_id == prt_id
                )
            )
            # update data
            for i in range(self.config["num_threads_per_warp"]):
                valid = (remote_simt_mask >> i) & 1
                if valid:
                    start_addr = i * data_width
                    end_addr = start_addr + data_width
                    data_buffer[start_addr: end_addr] = \
                        ld_resp.data[start_addr: end_addr]
                    """
                    addr = remote_addr_list[i]
                    db = self.subcore.core.processor.hardware.mem\
                        .get_value(addr, 4)
                    assert data_buffer[start_addr: end_addr] == db
                    """
        # NOTE: consume 1 pipeline cycle
        yield self.env.timeout(1 * self.clock_unit)
        instr_entry.remote_access_cyc = self.env.now - start_time

    def _handle_ld_instr_fb(
        self, instr_entry, offset_list, co_addr_list,
        remote_addr_list, remote_simt_mask, data_width, prt_id
    ):
        data_buffer = bytearray(
            data_width * self.config["num_threads_per_warp"]
        )
        # here we parallelize local and remote requests
        event_list = []
        event_list.append(
            self.env.process(
                self._issue_ld_local(
                    data_buffer=data_buffer,
                    co_addr_list=co_addr_list,
                    offset_list=offset_list,
                    data_width=data_width,
                    prt_id=prt_id,
                    instr_entry=instr_entry
                )
            )
        )
        event_list.append(
            self.env.process(
                self._issue_ld_remote(
                    data_buffer=data_buffer,
                    remote_simt_mask=remote_simt_mask,
                    remote_addr_list=remote_addr_list,
                    data_width=data_width,
                    prt_id=prt_id,
                    instr_entry=instr_entry
                )
            )
        )
        yield simpy.events.AllOf(self.env, event_list)
        # set data
        instr_entry.dst_values.append(deepcopy(data_buffer))
        instr_entry._decode_dst_operands()
        # set process flag
        instr_entry.processed = True
        # default near-bank writeback
        yield self.bus_arbiter\
            .upstream_req_queue.put(instr_entry)
        # update trace event
        self._append_trace_event_dur(
            tid="warp_{}-slot_{}"
            .format(
                instr_entry.warp_id,
                instr_entry.slot_id
            ),
            name="[lsu]-{}"
            .format(
                instr_entry.instr.trace_str()
            ),
            ts=instr_entry.last_trace_cyc,
            dur=self.env.now - instr_entry.last_trace_cyc,
            cat="instruction",
            args={
                "local_access_cyc": instr_entry.local_access_cyc,
                "remote_access_cyc": instr_entry.remote_access_cyc
            }
        )
        instr_entry.last_trace_cyc = self.env.now

    def _issue_st_local(
        self, data_buffer, co_addr_list, offset_list, 
        data_width, prt_id, instr_entry
    ):
        start_time = self.env.now
        # issue local store request
        for mem_addr in co_addr_list:
            dram_trans = co_addr_list[mem_addr]
            dram_trans.subcore_id = self.subcore.subcore_id
            dram_trans.is_nb = False
            dram_trans.prt_id = prt_id
            dram_trans.time = self.env.now
            # prepare data
            dram_trans.data = bytearray(self.config["dram_bank_io_width"])
            for i in range(self.config["num_threads_per_warp"]):
                valid = (dram_trans.simt_mask >> i) & 1
                if valid:
                    trans_start_addr = offset_list[i]
                    trans_end_addr = trans_start_addr + data_width
                    assert trans_end_addr <= self.config["dram_bank_io_width"]
                    db_start_addr = i * data_width
                    db_end_addr = db_start_addr + data_width
                    dram_trans.data[trans_start_addr: trans_end_addr] = \
                        data_buffer[db_start_addr: db_end_addr]
                    # update data mask
                    # NOTE this is guaranteed to have no conflict
                    # since we have checked write conflict before
                    assert offset_list[i] % 4 == 0
                    dram_trans.data_mask += 1 << (offset_list[i] // 4)
            # issue to pg, upstream traffic
            yield self.bus_arbiter\
                .upstream_req_queue.put(dram_trans)
        # wait until request return
        for mem_addr in co_addr_list:
            _ = yield self.in_dram_trans_queue.get(
                lambda x: (
                    isinstance(x, DRAMTransaction)
                    and x.type == "store"
                    and x.global_mem_addr == mem_addr
                    and x.is_nb is False
                    and x.prt_id == prt_id
                )
            )
            # NOTE: consume 1 pipeline cycle
            yield self.env.timeout(1 * self.clock_unit)
            """
            dram_trans = co_addr_list[mem_addr]
            for i in range(self.config["num_threads_per_warp"]):
                valid = (dram_trans.simt_mask >> i) & 1
                if valid:
                    trans_start_addr = offset_list[i]
                    trans_end_addr = trans_start_addr + data_width
                    addr = instr_entry._decoded_src_values[0][i]
                    db = self.subcore.core.processor.hardware.mem\
                        .get_value(addr, 4)
                    assert db == \
                        dram_trans.data[trans_start_addr: trans_end_addr]
            """
        instr_entry.local_access_cyc = self.env.now - start_time

    def _issue_st_remote(
        self, data_buffer, remote_simt_mask, remote_addr_list,
        data_width, prt_id, instr_entry
    ):
        start_time = self.env.now
        # issue remote req
        if remote_simt_mask > 0:
            # compose a remote store request
            st_req = SrcRemoteStoreReq(
                addr_list=remote_addr_list,
                data_width=data_width,
                simt_mask=remote_simt_mask,
                data=data_buffer
            )
            st_req.prt_id = prt_id
            # issue remote store request
            yield self.subcore.core.niu.req_queue.put(st_req)
            # wait for remote store response
            _ = yield self.subcore.core.niu.resp_queue.get(
                lambda x: (
                    isinstance(x, SrcRemoteStoreResp)
                    and x.addr_list == st_req.addr_list
                    and x.simt_mask == remote_simt_mask
                    and x.prt_id == prt_id
                )
            )
            # NOTE: consume 1 pipeline cycle
            yield self.env.timeout(1 * self.clock_unit)

        instr_entry.remote_access_cyc = self.env.now - start_time
        """
        for i in range(self.config["num_threads_per_warp"]):
            valid = (remote_simt_mask >> i) & 1
            if valid:
                start_addr = i * data_width
                end_addr = start_addr + data_width
                addr = remote_addr_list[i]
                db = self.subcore.core.processor.hardware.mem\
                    .get_value(addr, 4)
                assert data_buffer[start_addr: end_addr] == db
        """

    def _handle_st_instr_fb(
        self, instr_entry, offset_list, co_addr_list,
        remote_addr_list, remote_simt_mask,
        data_width, warp_id, pg_id, prt_id
    ):
        # get reg data operand
        operand = instr_entry.instr.src_operands[1]
        # issue register movement request
        assert operand.isnormalreg()
        reg_move_req = RegMoveReq(
            operand=operand,
            subcore_id=self.subcore.subcore_id,
            pg_id=pg_id,
            warp_id=warp_id,
            is_upstream=False
        )
        reg_move_req.ld_st_flag = True
        reg_move_req.is_nb = False
        reg_move_req.prt_id = prt_id
        reg_move_req.reg_size = \
            self.config["num_threads_per_warp"] * data_width
        # issue to pg, upstream traffic
        yield self.bus_arbiter\
            .upstream_req_queue.put(reg_move_req)
        # wait until response
        reg_move_resp = yield self.reg_req_bus_queue.get(
            lambda x: (
                isinstance(x, RegMoveReq)
                and x.operand.op_str == reg_move_req.operand.op_str
                and x.warp_id == reg_move_req.warp_id
                and x.ld_st_flag is True
                and x.is_nb is False
                and x.prt_id == prt_id
            )
        )
        data_buffer = deepcopy(reg_move_resp.data)
        assert len(data_buffer) == \
            self.config["num_threads_per_warp"] * data_width
        # NOTE: clear data depedency here
        self.subcore.dep_table_exe.entry[instr_entry.warp_id]\
            .decrease_read(operand.op_str)
        # NOTE: consume 1 pipeline cycle
        yield self.env.timeout(1 * self.clock_unit)
        
        # here we parallelize local and remote requests
        event_list = []
        event_list.append(
            self.env.process(
                self._issue_st_local(
                    data_buffer=data_buffer,
                    co_addr_list=co_addr_list,
                    offset_list=offset_list,
                    data_width=data_width,
                    prt_id=prt_id,
                    instr_entry=instr_entry
                )
            )
        )
        event_list.append(
            self.env.process(
                self._issue_st_remote(
                    data_buffer=data_buffer,
                    remote_simt_mask=remote_simt_mask,
                    remote_addr_list=remote_addr_list,
                    data_width=data_width,
                    prt_id=prt_id,
                    instr_entry=instr_entry
                )
            )
        )
        yield simpy.events.AllOf(self.env, event_list)
        # set instruction process flag
        instr_entry.processed = True
        # final writeback
        yield self.subcore.writeback_buffer.put(instr_entry)
        # update trace event
        self._append_trace_event_dur(
            tid="warp_{}-slot_{}"
            .format(
                instr_entry.warp_id,
                instr_entry.slot_id
            ),
            name="[lsu]-{}"
            .format(
                instr_entry.instr.trace_str()
            ),
            ts=instr_entry.last_trace_cyc,
            dur=self.env.now - instr_entry.last_trace_cyc,
            cat="instruction",
            args={
                "local_access_cyc": instr_entry.local_access_cyc,
                "remote_access_cyc": instr_entry.remote_access_cyc
            }
        )
        instr_entry.last_trace_cyc = self.env.now

    def _process_ld_st_bypass_backend(self):
        while True:
            instr_entry = yield self.ld_st_global_bypass_entry_queue.get()
            if instr_entry.instr.opcode.startswith("ld.global"):
                data_buffer = bytearray(0)
                num_bits = int(instr_entry.instr.opcode.split(".")[-1][1:])
                assert num_bits % 8 == 0, "The number of bits per data must " \
                    "be a multiply of 8"
                num_bytes = num_bits // 8
                for i in range(self.config["num_threads_per_warp"]):
                    mem_addr = instr_entry._decoded_src_values[0][i]
                    tmp_buffer = self.subcore.core.processor.hardware\
                        .mem.get_value(
                            mem_addr, num_bytes
                        )
                    data_buffer.extend(tmp_buffer)
                instr_entry.dst_values.append(data_buffer)
                instr_entry._decode_dst_operands()
                # set process flag
                instr_entry.processed = True
                # default near-bank writeback
                # set PG ID
                warp_id = instr_entry.warp_id
                pg_id = self.subcore.warp_info_table.entry[warp_id].pg_id
                instr_entry.pg_id = pg_id
                yield self.bus_arbiter\
                    .upstream_req_queue.put(instr_entry)
            elif instr_entry.instr.opcode.startswith("st.global"):
                # get near-bank reg data
                pg_id = self.subcore.warp_info_table\
                    .entry[instr_entry.warp_id].pg_id
                src_op = instr_entry.instr.src_operands[1]
                reg_addr, reg_size = self.subcore.core\
                    .pg_array[pg_id].get_pg_reg_addr(
                        reg_prefix=src_op.reg_prefix,
                        reg_index=src_op.reg_index,
                        subcore_id=self.subcore.subcore_id,
                        entry_id=instr_entry.warp_id
                    )
                operand_entry_data = deepcopy(
                    self.subcore.core.pg_array[pg_id].reg_file.array[
                        reg_addr: reg_addr + reg_size
                    ]
                )
                # init data
                instr_entry.src_values[1] = operand_entry_data
                # write to dram
                num_bits = int(instr_entry.instr.opcode.split(".")[-1][1:])
                assert num_bits % 8 == 0, "The number of bits per data must " \
                    "be a multiply of 8"
                num_bytes = num_bits // 8
                for i in range(self.config["num_threads_per_warp"]):
                    valid = ((instr_entry.simt_mask >> i) & 1)
                    if valid:
                        mem_addr = instr_entry._decoded_src_values[0][i]
                        reg_data = instr_entry.src_values[1][
                            (i * num_bytes): ((i + 1) * num_bytes)]
                        self.subcore.core.processor.hardware.mem.set_value(
                            mem_addr, reg_data
                        )
                # set instruction process flag
                instr_entry.processed = True
                # final writeback
                yield self.subcore.writeback_buffer.put(instr_entry)
            else:
                assert False, "Unknown ld/st instruction: {}"\
                    .format(instr_entry.instr.opcode)

    def _process_ld_st_shared(self):
        while True:
            instr_entry = yield self.ld_st_shared_entry_queue.get()
            if self.config["default_smem_loc_is_near_bank"]:
                # should be offloaded for near-bank execution
                # get dst pg_id
                pg_id = self.subcore.warp_info_table\
                    .entry[instr_entry.warp_id].pg_id
                instr_entry.pg_id = pg_id
                # issue to pg, upstream traffic
                yield self.subcore.core.subcore_pg_bus_arbiter\
                    .upstream_req_queue.put(instr_entry)
                # update trace event
                self._append_trace_event_dur(
                    tid="warp_{}-slot_{}"
                    .format(
                        instr_entry.warp_id,
                        instr_entry.slot_id
                    ),
                    name="[lsu]-{}"
                    .format(
                        instr_entry.instr.trace_str()
                    ),
                    ts=instr_entry.last_trace_cyc,
                    dur=self.env.now - instr_entry.last_trace_cyc,
                    cat="instruction"
                )
                instr_entry.last_trace_cyc = self.env.now
                continue
            # far-bank execution
            assert self.config["default_smem_loc_is_near_bank"] is False
            if instr_entry.instr.opcode.startswith("ld.shared"):
                num_bits = int(instr_entry.instr.opcode.split(".")[-1][1:])
                assert num_bits % 8 == 0, "The number of bits per " \
                    "data must be a multiply of 8"
                num_bytes = num_bits // 8

                # compose a read request
                smem_read_req = SMEMReadReq(
                    smem_addr_list=instr_entry._decoded_src_values[0],
                    data_width=num_bytes,
                    simt_mask=instr_entry.simt_mask
                )
                # issue read request
                yield self.subcore.core.smem.req_queue.put(smem_read_req)
                # get response
                smem_read_resp = yield self.subcore.core.smem.resp_queue.get(
                    lambda x: (
                        isinstance(x, SMEMReadResp)
                        and x.smem_addr_list == smem_read_req.smem_addr_list
                    )
                )
                data_buffer = smem_read_resp.data_buffer
                self.log.debug("SIMT mask: {}"
                               .format(instr_entry.simt_mask))
                instr_entry.dst_values.append(data_buffer)
            elif instr_entry.instr.opcode.startswith("st.shared"):
                num_bits = int(instr_entry.instr.opcode.split(".")[-1][1:])
                assert num_bits % 8 == 0, "The numbr of bits per " \
                    "data must be a multiply of 8"
                num_bytes = num_bits // 8

                # compose a write request
                smem_write_req = SMEMWriteReq(
                    smem_addr_list=instr_entry._decoded_src_values[0],
                    data_width=num_bytes,
                    data_buffer=instr_entry.src_values[1],
                    simt_mask=instr_entry.simt_mask
                )
                # issue write request
                yield self.subcore.core.smem.req_queue.put(smem_write_req)
                # get response
                _ = yield self.subcore.core.smem.resp_queue.get(
                    lambda x: (
                        isinstance(x, SMEMWriteResp)
                        and x.smem_addr_list 
                        == smem_write_req.smem_addr_list
                    )
                )
            elif instr_entry.instr.opcode.startswith("atom.shared"):
                num_bits = int(instr_entry.instr.opcode.split(".")[-1][1:])
                assert num_bits == 32, "The atomic ops must operate " \
                    "on the same data width as the shared memory bank" 
                num_bytes = num_bits // 8 
                atomic_op = instr_entry.instr.opcode.split(".")[-2]
                data_type = instr_entry.instr.opcode.split(".")[-1]

                # compose an atomic request 
                smem_atomic_req = SMEMAtomReq(
                    smem_addr_list=instr_entry._decoded_src_values[0],
                    val_list=instr_entry._decoded_src_values[1],
                    data_width=num_bytes,
                    op="{}.{}".format(atomic_op, data_type),
                    simt_mask=instr_entry.simt_mask 
                )
                # issue atomic request 
                yield self.subcore.core.smem.req_queue.put(smem_atomic_req) 
                # get response 
                _ = yield self.subcore.core.smem.resp_queue.get(
                    lambda x: (
                        isinstance(x, SMEMAtomResp)
                        and x.smem_addr_list 
                        == smem_atomic_req.smem_addr_list
                        and x.val_list == smem_atomic_req.val_list 
                    )
                )
            else:
                assert False, "Unknown ld/st instruction: {}"\
                    .format(instr_entry.instr.opcode)
            # set instruction process flag
            instr_entry.processed = True
            # final writeback
            yield self.subcore.writeback_buffer.put(instr_entry)
            # update trace event
            self._append_trace_event_dur(
                tid="warp_{}-slot_{}"
                .format(
                    instr_entry.warp_id,
                    instr_entry.slot_id
                ),
                name="[lsu]-{}"
                .format(
                    instr_entry.instr.trace_str()
                ),
                ts=instr_entry.last_trace_cyc,
                dur=self.env.now - instr_entry.last_trace_cyc,
                cat="instruction"
            )
            instr_entry.last_trace_cyc = self.env.now

    def _process_instr_entry(self):
        while True:
            instr_entry = yield self.instr_entry_queue.get()
            assert isinstance(instr_entry, InstrEntry)
            # perform functional simulation
            instr_entry.process_operands()

            if (
                instr_entry.instr.opcode.startswith("ld.global")
                or instr_entry.instr.opcode.startswith("st.global")
            ):
                # NOTE special handle of load/store global memory
                if self.config["bypass_ld_st_global_backend"]:
                    yield self.ld_st_global_bypass_entry_queue.put(instr_entry)
                else:
                    yield self.local_ld_st_global_entry_queue.put(instr_entry)
                continue
            elif (
                instr_entry.instr.opcode.startswith("ld.shared")
                or instr_entry.instr.opcode.startswith("st.shared")
                or instr_entry.instr.opcode.startswith("atom.shared")
            ):
                # NOTE special handle of load/store shared memory
                yield self.ld_st_shared_entry_queue.put(instr_entry)
                continue
            elif instr_entry.instr.opcode.startswith("ld.param"):
                # this is processed in _comp_result of instr_entry
                # NOTE consume 1 pipeline cycle
                yield self.env.timeout(1 * self.clock_unit)
                pass
            elif instr_entry.instr.opcode.startswith("cvta.to.global"):
                # this is processed in _comp_result of instr_entry
                # NOTE consume 1 pipeline cycle
                yield self.env.timeout(1 * self.clock_unit)
                pass
            else:
                raise NotImplementedError(
                    "LSU: not supported operation: {}"
                    .format(instr_entry.instr.opcode)
                )

            # set instruction process flag
            instr_entry.processed = True

            # final writeback
            yield self.subcore.writeback_buffer.put(instr_entry)
            # update trace event
            self._append_trace_event_dur(
                tid="warp_{}-slot_{}"
                .format(
                    instr_entry.warp_id,
                    instr_entry.slot_id
                ),
                name="[lsu]-{}"
                .format(
                    instr_entry.instr.trace_str()
                ),
                ts=instr_entry.last_trace_cyc,
                dur=self.env.now - instr_entry.last_trace_cyc,
                cat="instruction"
            )
            instr_entry.last_trace_cyc = self.env.now
