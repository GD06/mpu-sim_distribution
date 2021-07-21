import simpy
from copy import deepcopy

from simulator.instr_instance import InstrEntry
from simulator.register_file_utility import OperandReadReq, OperandReadResp


class OperandCollector:

    def __init__(self, env, log, config, clock_unit, backend, 
                 regfile_io_interface, execution_unit, opc_type):
        """Homogeneous operand collectors for several execution units
        Args:
            env: simpy environment
            log: python log
            config: configuration dictionary
            clock_unit: clock unit
            backend: the parent component
                For far-bank, this is subcore
                For near-bank, this is PG
            regfile_io_interface: read interface to register file
            execution_unit: the corresponding execution units of 
                the operand collectors
            opc_type: type of operand collector 
                (sfu, lsu, lsu_extension, cfu, syncu, fb_alu, nb_alu)
        """
        self.env = env
        self.log = log
        self.config = config
        self.clock_unit = deepcopy(clock_unit)
        self.backend = backend
        self.regfile_io_interface = regfile_io_interface
        self.execution_unit = execution_unit
        self.opc_type = opc_type
        self.num_threads_per_warp = self.config["num_threads_per_warp"]
        self.data_path_unit_size = self.config["data_path_unit_size"]
        self.alignment = self.num_threads_per_warp * self.data_path_unit_size
        
        self.base_regfile_read_port_id = \
            self.config["base_regfile_read_port_id_{}".format(opc_type)]
        self.num_opc = self.config["num_opc_{}".format(opc_type)]
        
        if (
            opc_type == "nb_alu"
            or opc_type == "lsu_extension"
        ):
            self.is_far_bank = False
        else:
            self.is_far_bank = True

        self.name = "fb-opc" if self.is_far_bank else "nb-opc"

        self.instr_entry_queue = simpy.Store(env, capacity=1)
        self.output_bus_lock = simpy.Resource(env, capacity=1)

        # For each operand collector, spawn a process to handle 
        # operand requests of the instruction, and update
        # operand entries once an operand returns from reg file
        for i in range(self.num_opc):
            regfile_read_port_id = self.base_regfile_read_port_id + i
            self.env.process(self._process_instr_entry(regfile_read_port_id))
        # performance metrics
        self.num_read = 0
        self.num_write = 0

    def get_perf_metrics(self):
        """Get a dictionary of performance metrics."""
        perf_metrics = {}
        perf_metrics["num_read"] = self.num_read
        perf_metrics["num_write"] = self.num_write
        if self.is_far_bank:
            return {"fb_opc": perf_metrics}
        else:
            return {"nb_opc": perf_metrics}

    def _append_trace_event_dur(self, instr_entry, name, ts, dur):
        if self.is_far_bank:
            self.backend._append_trace_event_dur(
                tid="warp_{}-slot_{}"
                .format(
                    instr_entry.warp_id,
                    instr_entry.slot_id
                ),
                name=name,
                ts=ts,
                dur=dur,
                cat="instruction"
            )
        else:
            subcore = self.backend.core.subcore_array[instr_entry.subcore_id]
            subcore._append_trace_event_dur(
                tid="warp_{}-slot_{}"
                .format(
                    instr_entry.warp_id,
                    instr_entry.slot_id
                ),
                name=name,
                ts=ts,
                dur=dur,
                cat="instruction, near-bank",
                args={
                    "pg_id": instr_entry.pg_id
                }
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

    def _get_special_reg_value(self, op_str, subcore_id, warp_id):
        if self.is_far_bank:
            return self.backend.get_special_reg_value(
                op_str, warp_id
            )
        else:
            return self.backend.core.subcore_array[subcore_id]\
                .get_special_reg_value(
                    op_str, warp_id
            )

    def _get_param_value(self, op_str, subcore_id, warp_id):
        if self.is_far_bank:
            return self.backend.get_param_value(op_str, warp_id)
        else:
            return self.backend.core.subcore_array[subcore_id]\
                .get_param_value(op_str, warp_id)

    def _clear_dependency(self, op_str, subcore_id, warp_id):
        if self.is_far_bank:
            self.backend.dep_table_exe.entry[warp_id]\
                .decrease_read(op_str)
        else:
            self.backend.core.subcore_array[subcore_id]\
                .dep_table_exe.entry[warp_id]\
                .decrease_read(op_str)

    def _process_instr_entry(self, regfile_read_port_id):
        while True:
            instr_entry = yield self.instr_entry_queue.get()
            assert isinstance(instr_entry, InstrEntry)

            # record total operand entries for this collector
            total_operand_entry = 0
            if "pred_reg" in instr_entry.instr.metadata:
                total_operand_entry = \
                    len(instr_entry.instr.src_operands) + 1
            else:
                total_operand_entry = \
                    len(instr_entry.instr.src_operands)

            # operand entry data (read from register file)
            operand_entry_data = [None] * total_operand_entry
            # set physical warp id info
            warp_id = instr_entry.warp_id
            # initially number of outbound operand requests is 0
            num_outbound_req = 0
            # operand mapping: operand id --> operand name string
            operand_map = {}
            # start collecting source operand
            operand_id = 0

            # if there is predicate field of this instruction
            if "pred_reg" in instr_entry.instr.metadata:
                pred_reg = instr_entry.instr.metadata["pred_reg"]
                reg_addr, reg_size = self._get_reg_addr(
                    pred_reg.reg_prefix, pred_reg.reg_index,
                    instr_entry.subcore_id, warp_id
                )
                # compose a read request
                operand_read_req = OperandReadReq(
                    operand_id=operand_id,
                    base_reg_addr=reg_addr,
                    total_reg_size=reg_size
                )
                # issue this to read interface
                yield self.regfile_io_interface\
                    .read_req_queue[regfile_read_port_id].put(operand_read_req)
                # record outbound request number
                num_outbound_req += 1
                operand_map[operand_id] = pred_reg.op_str
                # update next operand info
                operand_id += 1
                # update performance counter
                self.num_write += 1
                self.num_read += 1

            # collect all source operands
            for src_op in instr_entry.instr.src_operands:
                req_reg = False
                values = None
                if src_op.isreg():
                    if src_op.isnormalreg():
                        # NOTE: we will not collect registers not belonging
                        # to this register file
                        if (
                            (
                                self.is_far_bank
                                and instr_entry
                                .src_loc_is_nb[src_op.op_str] is False
                            )
                            or (
                                self.is_far_bank is False
                                and instr_entry.src_loc_is_nb[src_op.op_str]
                            )
                        ):
                            reg_addr, reg_size = self._get_reg_addr(
                                src_op.reg_prefix, src_op.reg_index,
                                instr_entry.subcore_id, warp_id
                            )
                            # compose a read request
                            operand_read_req = OperandReadReq(
                                operand_id=operand_id,
                                base_reg_addr=reg_addr,
                                total_reg_size=reg_size
                            )
                            # issue this to read interface
                            yield self.regfile_io_interface\
                                .read_req_queue[regfile_read_port_id]\
                                .put(operand_read_req)
                            # record outbound request number
                            num_outbound_req += 1
                            operand_map[operand_id] = src_op.op_str
                            # update next operand info
                            operand_id += 1
                            req_reg = True
                        else:
                            # NOTE: insert dummy field
                            values = [0] * self.config["num_threads_per_warp"]
                            operand_entry_data[operand_id] = values
                            operand_map[operand_id] = src_op.op_str
                            operand_id += 1
                    else:
                        # Special register reading
                        values = self._get_special_reg_value(
                            op_str=src_op.op_str,
                            subcore_id=instr_entry.subcore_id,
                            warp_id=warp_id
                        )
                        operand_entry_data[operand_id] = values
                        operand_map[operand_id] = src_op.op_str
                        # update next operand info
                        operand_id += 1
                        yield self.env.timeout(
                            self.config["subcore_special_reg_read_latency"]
                            * self.clock_unit
                        )
                        self._clear_dependency(
                            op_str=src_op.op_str,
                            subcore_id=instr_entry.subcore_id,
                            warp_id=warp_id
                        )
                    # update performance counter
                    self.num_write += 1
                    self.num_read += 1
                elif src_op.isparam():
                    # Paramter value reading
                    param_value = self._get_param_value(
                        op_str=src_op.op_str,
                        subcore_id=instr_entry.subcore_id,
                        warp_id=warp_id
                    )
                    op_value = src_op.eval(param_value)
                    values = [op_value] * self.num_threads_per_warp 
                    operand_entry_data[operand_id] = values
                    operand_map[operand_id] = src_op.op_str
                    # update next operand info
                    operand_id += 1
                    yield self.env.timeout(
                        self.config["subcore_param_reg_read_latency"]
                        * self.clock_unit
                    )
                    # update performance counter
                    self.num_write += 1
                    self.num_read += 1
                elif src_op.isimmvalue():
                    values = [src_op.eval()] * self.num_threads_per_warp
                    operand_entry_data[operand_id] = values
                    operand_map[operand_id] = src_op.op_str
                    # update next operand info
                    operand_id += 1
                    # update performance counter
                    self.num_write += 1
                    self.num_read += 1
                else:
                    raise NotImplementedError(
                        "Unknown type of operand: {}".format(src_op.op_str)
                    )
                # if we haven't sent out regiter request,
                # the value should have been ready
                if req_reg is False:
                    assert values is not None, "Unable to get values of the " \
                        "operand: {}".format(src_op.op_str)

            for i in range(num_outbound_req):
                # we expect to receive a given number of replies
                resp = yield self.regfile_io_interface\
                    .read_resp_queue[regfile_read_port_id].get()
                assert isinstance(resp, OperandReadResp), "The incorrect type" \
                    " from the response queue"
                assert resp.operand_id < total_operand_entry
                # get payload
                operand_entry_data[resp.operand_id] = resp.data
                # clear dependency
                if (
                    instr_entry.instr.opcode.startswith("st.global")
                    or instr_entry.instr.opcode.startswith("ld.global")
                ):
                    assert self.is_far_bank
                    # NOTE: only addr field is collected at the subcore
                    assert resp.operand_id == 0
                elif (
                    instr_entry.instr.opcode.startswith("ld.shared")
                    or instr_entry.instr.opcode.startswith("st.shared")
                    or instr_entry.instr.opcode.startswith("atom.shared")
                ):
                    if self.config["default_smem_loc_is_near_bank"]:
                        # near-bank should receive all operands
                        assert self.is_far_bank is False
                    else:
                        # far-bank should receive all operands
                        assert self.is_far_bank
                self._clear_dependency(
                    op_str=operand_map[resp.operand_id],
                    subcore_id=instr_entry.subcore_id,
                    warp_id=warp_id
                )
                # NOTE: consume 1 pipeline cycle
                yield self.env.timeout(1 * self.clock_unit)
            
            if self.is_far_bank:
                # this is far-bank opc
                if "pred_reg" in instr_entry.instr.metadata:
                    instr_entry.pred_buffer = operand_entry_data[0]
                    instr_entry.src_values = operand_entry_data[1:]
                else:
                    instr_entry.src_values = operand_entry_data
            else:
                # this is near-bank opc
                opcode = instr_entry.instr.opcode.split(".")[0]
                if opcode in self.config["alu_instr"]:
                    if "pred_reg" in instr_entry.instr.metadata:
                        instr_entry.pred_buffer = operand_entry_data[0]
                        instr_entry.src_values = operand_entry_data[1:]
                    else:
                        instr_entry.src_values = operand_entry_data
                elif (
                    instr_entry.instr.opcode.startswith("ld.shared")
                    or instr_entry.instr.opcode.startswith("st.shared")
                    or instr_entry.instr.opcode.startswith("atom.shared")
                ):
                    if self.config["default_smem_loc_is_near_bank"]:
                        instr_entry.src_values = operand_entry_data
                    else:
                        assert False, "wrong offloading datapath"
                elif (
                    instr_entry.instr.opcode.startswith("ld.global")
                    or instr_entry.instr.opcode.startswith("st.global")
                ):
                    assert False, "wrong ld.global and st.global datapath"
                else:
                    assert False

            # try to acquire output bus
            with self.output_bus_lock.request() as request:
                # wait for access
                yield request
                # 1 cycle occupied on bus
                yield self.env.timeout(1 * self.clock_unit)
                # issue to execution unit
                yield self.execution_unit.instr_entry_queue.put(
                    instr_entry
                )
                # update trace event
                self._append_trace_event_dur(
                    instr_entry=instr_entry,
                    name="[{}]-{}"
                    .format(
                        self.name,
                        instr_entry.instr.trace_str()
                    ),
                    ts=instr_entry.last_trace_cyc,
                    dur=self.env.now - instr_entry.last_trace_cyc,
                )
                instr_entry.last_trace_cyc = self.env.now
