from copy import deepcopy
import simpy

from simulator.instr_instance import InstrEntry
from simulator.dram_message import DRAMTransaction, PRTEntryReq
from simulator.reg_move_engine import RegMoveReq
from simulator.shared_memory import SMEMReadReq, SMEMReadResp, \
    SMEMWriteReq, SMEMWriteResp, SMEMAtomReq, SMEMAtomResp


class LoadStoreUnitExtension:
    
    def __init__(self, env, log, config, clock_unit, pg):
        self.env = env
        self.log = log
        self.config = config
        self.clock_unit = deepcopy(clock_unit)
        self.pg = pg
        self.in_dram_trans_queue = simpy.Store(env, capacity=1)
        self.out_dram_trans_queue = simpy.FilterStore(env)
        self.num_unit = self.config["num_lsu_extension"]
        # get reference to bus
        self.bus_arbiter = self.pg.core.subcore_pg_bus_arbiter
        # get reference to bank
        self.bank = []
        for i in range(self.config["num_pe"]):
            self.bank.append(self.pg.pe_array[i].bank)
        for i in range(self.num_unit):
            self.env.process(self._process_dram_trans())
        self.prt_entry_queue = simpy.Store(env, capacity=1)
        self.num_nb_local_prt_entry = self.config["max_num_nb_prt_entry"]
        # spawn processes to process offloaded PRT request
        for local_nb_prt_id in range(self.num_nb_local_prt_entry):
            self.env.process(
                self._process_nb_prt_req(local_nb_prt_id)
            )
        self.reg_mov_ack_queue = simpy.FilterStore(env)
        # the dram transaction queues for each PE
        self.pe_in_dram_trans_queue = []
        self.pe_out_dram_trans_queue = simpy.FilterStore(env)
        for i in range(self.config["num_pe"]):
            self.pe_in_dram_trans_queue.append(simpy.Store(env))
        # spawn processes to process pe's dram transaction
        for i in range(self.config["num_pe"]):
            self.env.process(
                self._pe_process_dram_trans(i)
            )
        
        # queues for receiving ld.shared/st.shared/atom.shared request
        self.instr_entry_queue = simpy.Store(env, capacity=1)
        # spawn processes to process ld/st/atom shared request
        for i in range(self.config["lsu_extensiuon_shared_issue_port"]):
            self.env.process(self._process_ld_st_shared())
            
        # performance counter
        self.num_prt_read = 0
        self.num_prt_write = 0

    def get_perf_metrics(self):
        """Get a dictionary of performance metrics."""
        perf_metrics = {}
        perf_metrics["num_prt_read"] = self.num_prt_read
        perf_metrics["num_prt_write"] = self.num_prt_write
        return {"lsu_extension": perf_metrics}

    def _process_ld_st_shared(self):
        while True:
            instr_entry = yield self.instr_entry_queue.get()
            assert isinstance(instr_entry, InstrEntry)
            # perform functional simulation
            instr_entry.process_operands()

            subcore_id = instr_entry.subcore_id
            # reference to trace event
            _append_trace_event_dur = self.pg.core.subcore_array[subcore_id]\
                ._append_trace_event_dur
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
                yield self.pg.core.smem.req_queue.put(smem_read_req)
                # get response
                smem_read_resp = yield self.pg.core.smem.resp_queue.get(
                    lambda x: (
                        isinstance(x, SMEMReadResp)
                        and x.smem_addr_list
                        == smem_read_req.smem_addr_list
                    )
                )
                data_buffer = smem_read_resp.data_buffer
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
                yield self.pg.core.smem.req_queue.put(smem_write_req)
                # get response
                _ = yield self.pg.core.smem.resp_queue.get(
                    lambda x: (
                        isinstance(x, SMEMWriteResp)
                        and x.smem_addr_list == smem_write_req.smem_addr_list
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
                yield self.pg.core.smem.req_queue.put(smem_atomic_req)
                # get response
                _ = yield self.pg.core.smem.resp_queue.get(
                    lambda x: (
                        isinstance(x, SMEMAtomResp)
                        and x.smem_addr_list == smem_atomic_req.smem_addr_list
                        and x.val_list == smem_atomic_req.val_list
                    )
                )
            else:
                assert False, "Unknown ld/st instruction: {}"\
                    .format(instr_entry.instr.opcode)
            # set instruction process flag
            instr_entry.processed = True
            # near-bank writeback
            yield self.pg.writeback_buffer.put(instr_entry)
            # update trace event
            _append_trace_event_dur(
                tid="warp_{}-slot_{}"
                .format(
                    instr_entry.warp_id,
                    instr_entry.slot_id
                ),
                name="[lsu-ext]-{}"
                .format(
                    instr_entry.instr.trace_str()
                ),
                ts=instr_entry.last_trace_cyc,
                dur=self.env.now - instr_entry.last_trace_cyc,
                cat="instruction, near-bank",
                args={
                    "pg_id": instr_entry.pg_id
                }
            )
            instr_entry.last_trace_cyc = self.env.now

    def _process_nb_prt_req(self, local_nb_prt_id):
        while True:
            prt_entry = yield self.prt_entry_queue.get()
            assert isinstance(prt_entry, PRTEntryReq)
            instr_entry = prt_entry.instr_entry
            offset_list = prt_entry.offset_list
            co_addr_list = prt_entry.co_addr_list
            subcore_id = instr_entry.subcore_id
            pg_id = self.pg.pg_id
            
            # reference to trace event
            _append_trace_event_dur = self.pg.core.subcore_array[subcore_id]\
                ._append_trace_event_dur

            # extract fields from instr_entry
            if instr_entry.instr.opcode.startswith("ld.global"):
                is_ld = True
            elif instr_entry.instr.opcode.startswith("st.global"):
                is_ld = False
            else:
                raise NotImplementedError(
                    "LSU-extension: not supported operation: {}"
                    .format(instr_entry.instr.opcode)
                )
            # lsu in the subcore has already ensured data_width == 4 bytes
            data_width = \
                int(instr_entry.instr.opcode.split(".")[-1][1:]) // 8
            warp_id = instr_entry.warp_id

            if is_ld:
                # process load prt entry
                data_buffer = bytearray(
                    data_width * self.config["num_threads_per_warp"]
                )
                # issue local dram transactions
                for mem_addr in co_addr_list:
                    dram_trans = co_addr_list[mem_addr]
                    dram_trans.is_nb = True
                    dram_trans.time = self.env.now
                    dram_trans.prt_id = local_nb_prt_id
                    yield self.in_dram_trans_queue.put(dram_trans)
                # receive dram transactions
                for mem_addr in co_addr_list:
                    dram_trans = yield self.out_dram_trans_queue.get(
                        lambda x: (
                            x.type == "load"
                            and x.global_mem_addr == mem_addr
                            and x.is_nb is True
                            and x.prt_id == local_nb_prt_id
                        )
                    )
                    assert dram_trans.data is not None
                    assert dram_trans.simt_mask > 0
                    for i in range(self.config["num_threads_per_warp"]):
                        valid = (dram_trans.simt_mask >> i) & 1
                        if valid:
                            trans_start_addr = offset_list[i]
                            trans_end_addr = trans_start_addr + data_width
                            assert trans_end_addr \
                                <= self.config["dram_bank_io_width"]
                            db_start_addr = i * data_width
                            db_end_addr = db_start_addr + data_width
                            data_buffer[db_start_addr: db_end_addr] = \
                                dram_trans.data[trans_start_addr: 
                                                trans_end_addr]
                # write to register
                # set data
                instr_entry.dst_values.append(deepcopy(data_buffer))
                instr_entry._decode_dst_operands()
                # set process flag
                instr_entry.processed = True
                # near-bank writeback
                yield self.pg.writeback_buffer.put(instr_entry)
                # update trace event
                _append_trace_event_dur(
                    tid="warp_{}-slot_{}"
                    .format(
                        instr_entry.warp_id,
                        instr_entry.slot_id
                    ),
                    name="[lsu-ext]-{}"
                    .format(
                        instr_entry.instr.trace_str()
                    ),
                    ts=instr_entry.last_trace_cyc,
                    dur=self.env.now - instr_entry.last_trace_cyc,
                    cat="instruction, near-bank",
                    args={
                        "pg_id": instr_entry.pg_id
                    }
                )
                instr_entry.last_trace_cyc = self.env.now
            else:
                # process store prt entry
                data_buffer = None
                # get the source register data
                operand = instr_entry.instr.src_operands[1]
                # using register movement interface
                assert operand.isnormalreg()
                reg_move_req = RegMoveReq(
                    operand=operand,
                    subcore_id=subcore_id,
                    pg_id=pg_id,
                    warp_id=warp_id,
                    is_upstream=False
                )
                reg_move_req.ld_st_flag = True
                reg_move_req.is_nb = True
                reg_move_req.prt_id = local_nb_prt_id
                # issue to reg move engine
                yield self.pg.reg_move_engine\
                    .reg_req_queue.put(reg_move_req)
                # wait for response
                reg_move_resp = yield self.reg_mov_ack_queue.get(
                    lambda x: (
                        x.operand.op_str == reg_move_req.operand.op_str
                        and x.warp_id == reg_move_req.warp_id
                        and x.ld_st_flag is True
                        and x.is_nb is True
                        and x.prt_id == local_nb_prt_id
                    )
                )
                data_buffer = deepcopy(reg_move_resp.data)
                # NOTE: clear data depedency here
                self.pg.core.subcore_array[instr_entry.subcore_id]\
                    .dep_table_exe.entry[instr_entry.warp_id]\
                    .decrease_read(operand.op_str)
                # NOTE: consume 1 pipeline cycle
                yield self.env.timeout(1 * self.clock_unit)
                # issue local store request
                for mem_addr in co_addr_list:
                    dram_trans = co_addr_list[mem_addr]
                    dram_trans.is_nb = True
                    dram_trans.prt_id = local_nb_prt_id
                    dram_trans.time = self.env.now
                    # prepare data
                    dram_trans.data = \
                        bytearray(self.config["dram_bank_io_width"])
                    for i in range(self.config["num_threads_per_warp"]):
                        valid = (dram_trans.simt_mask >> i) & 1
                        if valid:
                            trans_start_addr = offset_list[i]
                            trans_end_addr = trans_start_addr + data_width
                            assert trans_end_addr \
                                <= self.config["dram_bank_io_width"]
                            db_start_addr = i * data_width
                            db_end_addr = db_start_addr + data_width
                            dram_trans.data[trans_start_addr: 
                                            trans_end_addr] = \
                                data_buffer[db_start_addr: db_end_addr]
                            # update data mask
                            # NOTE this is guaranteed to have no conflict
                            # since we have checked write conflict before
                            assert offset_list[i] % 4 == 0
                            dram_trans.data_mask += 1 << (offset_list[i] // 4)
                    # issue to pg, upstream traffic
                    yield self.in_dram_trans_queue.put(dram_trans)
                # wait until request return
                for mem_addr in co_addr_list:
                    _ = yield self.out_dram_trans_queue.get(
                        lambda x: (
                            x.type == "store"
                            and x.global_mem_addr == mem_addr
                            and x.is_nb is True
                            and x.prt_id == local_nb_prt_id
                        )
                    )
                    # NOTE: consume 1 pipeline cycle
                    yield self.env.timeout(1 * self.clock_unit)
                # set process flag
                instr_entry.processed = True
                # commit instruction
                yield self.pg.core.subcore_pg_bus_arbiter\
                    .downstream_req_queue.put(instr_entry)
                # update trace event
                _append_trace_event_dur(
                    tid="warp_{}-slot_{}"
                    .format(
                        instr_entry.warp_id,
                        instr_entry.slot_id
                    ),
                    name="[lsu-ext]-{}"
                    .format(
                        instr_entry.instr.trace_str()
                    ),
                    ts=instr_entry.last_trace_cyc,
                    dur=self.env.now - instr_entry.last_trace_cyc,
                    cat="instruction, near-bank",
                    args={
                        "pg_id": instr_entry.pg_id
                    }
                )
                instr_entry.last_trace_cyc = self.env.now
            # update performance counter
            self.num_prt_read += 1
            self.num_prt_write += 1

    def _handle_load_trans(self, dram_trans, pe_id):
        dram_trans.ld_for_update = False
        self.bank[pe_id].mem_trans_queue.append(dram_trans)
        yield self.bank[pe_id].mem_trans_token_queue.put(1)

    def _handle_store_trans(self, dram_trans, pe_id):
        # currently we implement in-place read-update-write
        # first read
        ld_dram_trans = deepcopy(dram_trans)
        ld_dram_trans.type = "load"
        ld_dram_trans.ld_for_update = True
        self.bank[pe_id].mem_trans_queue.append(ld_dram_trans)
        yield self.bank[pe_id].mem_trans_token_queue.put(1)
        # wait for read response
        ld_dram_trans_resp = yield self.pe_out_dram_trans_queue.get(
            lambda x: (
                x.type == ld_dram_trans.type
                and x.global_mem_addr == ld_dram_trans.global_mem_addr
                and x.ld_for_update is True
            )
        )
        tmp_read_buffer = deepcopy(ld_dram_trans_resp.data)
        # NOTE: consume 1 pipeline cycle
        yield self.env.timeout(1 * self.clock_unit)
        # then update
        assert dram_trans.data is not None
        assert tmp_read_buffer is not None
        # NOTE currently we assume 4 byte data with and 16 byte bank width
        data_width = self.config["data_path_unit_size"]
        assert self.config["dram_bank_io_width"] % data_width == 0
        vec_len = self.config["dram_bank_io_width"] // data_width
        for i in range(vec_len):
            valid = (dram_trans.data_mask >> i) & 1
            if not valid:
                # for invalid field, use original data point
                start_addr = i * data_width
                end_addr = start_addr + data_width
                dram_trans.data[start_addr: end_addr] = \
                    tmp_read_buffer[start_addr: end_addr]
        # then write
        dram_trans.type = "store"
        dram_trans.ld_for_update = False
        self.bank[pe_id].mem_trans_queue.append(dram_trans)
        yield self.bank[pe_id].mem_trans_token_queue.put(1)

    def _pe_process_dram_trans(self, pe_id):
        while True:
            dram_trans = yield self.pe_in_dram_trans_queue[pe_id].get()
            if dram_trans.type == "load":
                yield self.env.process(
                    self._handle_load_trans(dram_trans, pe_id)
                )
            elif dram_trans.type == "store":
                # NOTE this guarantees update atomicity
                yield self.env.process(
                    self._handle_store_trans(dram_trans, pe_id)
                )
            else:
                raise NotImplementedError(
                    "Unknown dram trans type: {}"
                    .format(dram_trans.type)
                )

    def _process_dram_trans(self):
        while True:
            dram_trans = yield self.in_dram_trans_queue.get()
            assert isinstance(dram_trans, DRAMTransaction)
            # NOTE: we don't set the queue limit so this will
            # not block
            yield self.pe_in_dram_trans_queue[dram_trans.pe_id]\
                .put(dram_trans)
            # spawn a process to handle response
            # NOTE: this is nonblocking
            self.env.process(
                self._handle_dram_trans_resp(dram_trans)
            )

    def _handle_dram_trans_resp(self, dram_trans_req):
        dram_trans_resp = yield self.pe_out_dram_trans_queue.get(
            lambda x: (
                x.type == dram_trans_req.type
                and x.global_mem_addr == dram_trans_req.global_mem_addr
                and x.ld_for_update is False
            )
        )
        if dram_trans_resp.is_nb:
            # reutrn to near-bank
            yield self.out_dram_trans_queue.put(dram_trans_resp)
        else:
            # return to far-bank
            yield self.bus_arbiter\
                .downstream_req_queue.put(dram_trans_resp)
