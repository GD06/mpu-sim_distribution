import simpy
from copy import deepcopy

from simulator.processing_engine import ProcessingEngine 
from simulator.register_file import RegisterFile
from simulator.register_file_utility import RegFileOperandIOInterface, \
    OperandWriteReq
from simulator.alu import ArithmeticLogicUnit
from simulator.operand_collector import OperandCollector
from simulator.load_store_unit_extension import LoadStoreUnitExtension
from simulator.reg_move_engine import RegMoveEngine, RegMoveReq
from simulator.instr_instance import InstrEntry
from simulator.dram_message import DRAMTransaction, PRTEntryReq


class ProcessingGroup:

    def __init__(self, pg_id, env, config, log, core):
        self.pg_id = pg_id 
        self.env = env 
        self.config = config 
        self.log = log 
        self.core = core 

        self.filter_func = core.filter_func
        self.traceEvents = []

        assert config["sim_clock_freq"] % config["dram_clock_freq"] == 0, (
            "Undividable simulation clock frequency")
        self.clock_unit = config["sim_clock_freq"] // config["dram_clock_freq"]
        self.pe_array = []
        for i in range(config["num_pe"]):
            pe = ProcessingEngine(
                pe_id=i,
                env=env,
                config=config, 
                log=log,
                pg=self,
            )
            self.pe_array.append(pe)

        self.reg_base_ptr = 0
        # Near-bank register file
        self.reg_file = RegisterFile(
            env=env,
            log=log,
            config=self.config,
            clock_unit=self.clock_unit,
            reg_file_type="near-bank"
        )
        # Near-bank register file IO interface
        self.rf_io_interface = RegFileOperandIOInterface(
            env=self.env,
            log=self.log,
            config=self.config,
            clock_unit=self.clock_unit,
            reg_file=self.reg_file,
            interface_type="near-bank"
        )
        
        # register movement engine
        self.reg_move_engine = RegMoveEngine(
            env=self.env,
            log=self.log,
            config=self.config,
            clock_unit=self.clock_unit,
            backend=self,
            regfile_io_interface=self.rf_io_interface,
            bus_arbiter=self.core.subcore_pg_bus_arbiter,
            engine_type="nb_reg_move"
        )

        # Near-bank ALU
        self.nb_alu = ArithmeticLogicUnit(
            env=self.env,
            log=self.log,
            config=self.config,
            clock_unit=self.clock_unit,
            backend=self,
            alu_type="near-bank"
        )
        self.opc_nb_alu = OperandCollector(
            env=self.env,
            log=self.log,
            config=self.config,
            clock_unit=self.clock_unit,
            backend=self,
            regfile_io_interface=self.rf_io_interface,
            execution_unit=self.nb_alu,
            opc_type="nb_alu"
        )

        # Near-bank lsu extension
        self.lsu_extension = LoadStoreUnitExtension(
            env=self.env,
            log=self.log,
            config=self.config,
            clock_unit=self.clock_unit,
            pg=self
        )
        self.opc_lsu_extension = OperandCollector(
            env=self.env,
            log=self.log,
            config=self.config,
            clock_unit=self.clock_unit,
            backend=self,
            regfile_io_interface=self.rf_io_interface,
            execution_unit=self.lsu_extension,
            opc_type="lsu_extension"
        )

        self.execute_buffer = simpy.Store(
            env, capacity=self.config["pg_execute_buffer_size"]
        )
        self.writeback_buffer = simpy.Store(
            env, capacity=self.config["pg_writeback_buffer_size"]
        )

        self.bus_receive_buffer = simpy.Store(
            env, capacity=self.config["pg_bus_receive_buffer_size"]
        )

        # Spawn a process for the execute pg stage
        self.env.process(self._execute())

        # Spawn processes for the commit pg stage
        for i in range(self.config["num_nb_wb_port"]):
            self.env.process(self._writeback(i))

        # Spawn a process to receive data from the bus
        self.env.process(self._receive_bus_data())

        # For near-bank writeback
        self.base_regfile_write_port_id_commit = \
            self.config["base_regfile_write_port_id_nb_commit"]

        # Performance metrics
        self.num_instr_executed = 0

        return

    def get_trace_events(self):
        """Get a list of trace events"""
        _trace_Events = deepcopy(self.traceEvents)
        return _trace_Events

    def _append_trace_event_dur(self, tid, name, ts, dur, cat="", args={}):
        new_event = {}
        new_event["pid"] = "proc_{}_core_{}_pg_{}".format(
            self.core.processor.proc_id, self.core.core_id, self.pg_id
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
        # Collect the performance metrics of all hardware sub-module
        perf_metrics["num_instr_executed"] = self.num_instr_executed
        # operand collector
        opc_nb_alu_metrics = self.opc_nb_alu.get_perf_metrics()
        assert len(opc_nb_alu_metrics) == 1
        perf_metrics.update(opc_nb_alu_metrics)
        # alu
        nb_alu_metrics = self.nb_alu.get_perf_metrics()
        assert len(nb_alu_metrics) == 1
        perf_metrics.update(nb_alu_metrics)
        # lsu extension
        lsu_extension_metrics = self.lsu_extension.get_perf_metrics()
        assert len(lsu_extension_metrics) == 1
        perf_metrics.update(lsu_extension_metrics)
        # register file
        reg_file_metrics = self.reg_file.get_perf_metrics()
        assert len(reg_file_metrics) == 1
        perf_metrics.update(reg_file_metrics)
        # dram bank
        for i in range(self.config["num_pe"]):
            dram_bank_metrics = self.pe_array[i].bank.get_perf_metrics()
            assert len(dram_bank_metrics) == 1
            perf_metrics.update(dram_bank_metrics)
        return {"pg_{}".format(self.pg_id): perf_metrics}

    def get_pg_reg_addr(self, reg_prefix, reg_index, 
                        subcore_id, entry_id):
        """This function calculates the absolute addrss of a register in the
        register file of PG. It also returns the size of registers with
        the same name across threads in the whole warp.

        Args:
            reg_prefix: the prefix of register name
            reg_index: the index of the register
            subcore_id: the subcore ID of this warp
            entry_id: the warp ID of this register

        Returns:
            (reg_addr, reg_size): the stating address of this register in the
                register file and the size of the whole register
        """
        # get the warp info entry
        warp_info_table_entry = self.core.subcore_array[subcore_id]\
            .warp_info_table.entry[entry_id]

        reg_base_addr = warp_info_table_entry.pg_reg_base_addr
        prefix_reg_base_addr = (
            reg_base_addr
            + warp_info_table_entry.prog_reg_offset[reg_prefix]
        )
        reg_size = warp_info_table_entry.prog_reg_size[reg_prefix]
        reg_addr = prefix_reg_base_addr + reg_index * reg_size

        return (reg_addr, reg_size)

    def reset_status(self):
        self.reg_base_ptr = 0 
        for i in range(self.config["num_pe"]):
            self.pe_array[i].reset_status() 
        return

    def check_reg_usage(self, reg_usage_in_bytes):
        """Check whether the amount of near-bank registers are sufficient to 
        accomodate a new thread block. 
        """
        reg_base_ptr = self.reg_base_ptr 
        if reg_base_ptr + reg_usage_in_bytes > self.config["pg_reg_file_size"]:
            return False 

        return True 
   
    def _execute(self):
        """Corresponding execution stage of the near-bank backend
        """
        while True:
            buffer_entry = yield self.execute_buffer.get()
            if isinstance(buffer_entry, RegMoveReq):
                yield self.reg_move_engine\
                    .reg_req_queue.put(buffer_entry)
            elif isinstance(buffer_entry, InstrEntry):
                instr_entry = buffer_entry
                # dispatch instruction to operand collector units
                opcode = instr_entry.instr.opcode.split(".")[0]
                if opcode in self.config["alu_instr"]:
                    yield self.opc_nb_alu.instr_entry_queue.put(instr_entry)
                elif opcode in self.config["lsu_instr"]:
                    yield self.opc_lsu_extension.instr_entry_queue\
                        .put(instr_entry)
                else:
                    raise NotImplementedError(
                        "Unsupported opcode: {}".format(opcode)
                    )
            else:
                raise NotImplementedError(
                    "Unsupported class: {}".format(type(buffer_entry))
                )

    def _writeback(self, local_regfile_write_port_id_commit):
        """Corresponding writeback stage of the near-bank backend
        """
        regfile_write_port_id = \
            self.base_regfile_write_port_id_commit \
            + local_regfile_write_port_id_commit
        while True:
            instr_entry = yield self.writeback_buffer.get()
            simt_mask = instr_entry.simt_mask
            subcore_id = instr_entry.subcore_id
            entry_id = instr_entry.warp_id
            instr = instr_entry.instr

            for i in range(len(instr.dst_operands)):
                dst_op = instr.dst_operands[i]
                reg_addr, reg_size = self.get_pg_reg_addr(
                    reg_prefix=dst_op.reg_prefix, 
                    reg_index=dst_op.reg_index, 
                    subcore_id=subcore_id,
                    entry_id=entry_id
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
                self.core.subcore_array[subcore_id]\
                    .dep_table_exe.entry[entry_id]\
                    .decrease_write(dst_op.op_str)
            # send to subcore
            yield self.core.subcore_pg_bus_arbiter\
                .downstream_req_queue.put(instr_entry)
            
            # update trace event
            self.core.subcore_array[instr_entry.subcore_id]\
                ._append_trace_event_dur(
                    tid="warp_{}-slot_{}"
                    .format(
                        instr_entry.warp_id,
                        instr_entry.slot_id
                    ),
                    name="[nb-wb]-{}"
                    .format(
                        instr.trace_str()
                    ),
                    ts=instr_entry.last_trace_cyc,
                    dur=self.env.now - instr_entry.last_trace_cyc,
                    cat="instruction, near-bank",
                    args={
                        "pg_id": instr_entry.pg_id,
                        "wb_port": local_regfile_write_port_id_commit
                    }
            )
            instr_entry.last_trace_cyc = self.env.now

    def _receive_bus_data(self):
        """This function receives data from subcore-pg buses
        """
        while True:
            packet = yield self.bus_receive_buffer.get()
            if isinstance(packet, RegMoveReq):
                yield self.execute_buffer.put(packet)
            elif isinstance(packet, DRAMTransaction):
                yield self.lsu_extension\
                    .in_dram_trans_queue.put(packet)
            elif isinstance(packet, PRTEntryReq):
                yield self.lsu_extension\
                    .prt_entry_queue.put(packet)
                # update trace event
                subcore_id = packet.instr_entry.subcore_id
                self.core.subcore_array[subcore_id]._append_trace_event_dur(
                    tid="warp_{}-slot_{}"
                    .format(
                        packet.instr_entry.warp_id,
                        packet.instr_entry.slot_id
                    ),
                    name="[tsv-up]-{}"
                    .format(
                        packet.instr_entry.instr.trace_str()
                    ),
                    ts=packet.instr_entry.last_trace_cyc,
                    dur=self.env.now - packet.instr_entry.last_trace_cyc,
                    cat="instruction, near-bank",
                    args={
                        "pg_id": packet.instr_entry.pg_id
                    }
                )
                packet.instr_entry.last_trace_cyc = self.env.now
            elif isinstance(packet, InstrEntry):
                instr_entry = packet
                opcode = instr_entry.instr.opcode.split(".")[0]
                if opcode in self.config["alu_instr"]:
                    yield self.execute_buffer.put(instr_entry)
                elif opcode in self.config["lsu_instr"]:
                    if instr_entry.instr.opcode.startswith("ld.global"):
                        yield self.writeback_buffer.put(instr_entry)
                    elif (
                        instr_entry.instr.opcode.startswith("ld.shared")
                        or instr_entry.instr.opcode.startswith("st.shared")
                        or instr_entry.instr.opcode.startswith("atom.shared")
                    ):
                        yield self.execute_buffer.put(instr_entry)
                    else:
                        assert False, "Wrong near-bank instruction: {}"\
                            .format(opcode)
                else:
                    raise NotImplementedError(
                        "Not supported operation: {}"
                        .format(opcode)
                    )
                # update trace event
                subcore_id = packet.subcore_id
                self.core.subcore_array[subcore_id]._append_trace_event_dur(
                    tid="warp_{}-slot_{}"
                    .format(
                        packet.warp_id,
                        packet.slot_id
                    ),
                    name="[tsv-up]-{}"
                    .format(
                        packet.instr.trace_str()
                    ),
                    ts=packet.last_trace_cyc,
                    dur=self.env.now - packet.last_trace_cyc,
                    cat="instruction, near-bank",
                    args={
                        "pg_id": packet.pg_id
                    }
                )
                packet.last_trace_cyc = self.env.now
            else:
                assert False
