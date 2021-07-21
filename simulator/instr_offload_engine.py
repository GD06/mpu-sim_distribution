import simpy
from copy import deepcopy

from simulator.reg_move_engine import RegMoveReq
from simulator.instr_instance import InstrEntry


class InstrOffloadEngine:

    def __init__(self, env, config, log, reg_move_engine, subcore):
        """Instruction offload engine is located on subcore
        """
        self.env = env
        self.config = config
        self.log = log
        self.reg_move_engine = reg_move_engine
        self.subcore = subcore
        self.subcore_id = subcore.subcore_id
        self.reg_track_table = self.subcore.reg_track_table
        self.clock_unit = deepcopy(subcore.clock_unit)

        self.num_unit = self.config["num_offload_engine"]
        self.instr_entry_queue = simpy.Store(env, capacity=1)

        for i in range(self.num_unit):
            self.env.process(
                self._handle_instr_entry()
            )

        self.reg_ack_queue = []
        for i in range(self.config["max_num_warp_per_subcore"]):
            self.reg_ack_queue.append(
                simpy.FilterStore(env, capacity=1)
            )

    def get_perf_metrics(self):
        """Get a dictionary of performance metrics."""
        perf_metrics = {}
        # Collect the performance metrics of this hardware module
        return {"instr_offload_engine": perf_metrics}

    def _are_all_operands_nb(self, instr_entry):
        """Check if all source operands (including predicate) of the 
        instruction come from near-bank register file
        Args:
            instr_entry: the instruction entry to be checked
        Returns:
            all_nb: True if all operands are near-bank; else False
        """
        warp_id = instr_entry.warp_id
        track_entry = self.reg_track_table.entry[warp_id]
        if "pred_reg" in instr_entry.instr.metadata:
            pred_reg = instr_entry.instr.metadata["pred_reg"]
            if not track_entry.check_ready(pred_reg.op_str, True):
                return False
        for src_op in instr_entry.instr.src_operands:
            if src_op.isnormalreg():
                if not track_entry.check_ready(src_op.op_str, True):
                    return False
            if src_op.isspecialreg():
                if not self.config["default_specialreg_loc_is_near_bank"]:
                    return False
            if src_op.isparam():
                if not self.config["default_param_loc_is_near_bank"]:
                    return False
            if src_op.isimmvalue():
                if not self.config["default_imm_loc_is_near_bank"]:
                    return False
        return True

    def _set_instr_location(self, instr_entry):
        opcode = instr_entry.instr.opcode.split(".")[0]
        if opcode in self.config["far_bank_op_set"]:
            # NOTE: all LSU instructions are default to far-bank
            instr_entry.instr_location = "fb"
        elif opcode in self.config["alu_instr"]:
            # check if there are compiler hints
            if "location" in instr_entry.instr.metadata:
                # NOTE: compiler hint here
                location = instr_entry.instr.metadata["location"]
                assert location in {"nb", "fb"}, \
                    "Unsupported compiler hint: {}".format(location)
                instr_entry.instr_location = location
            else:
                # use hardware policy
                if self._are_all_operands_nb(instr_entry):
                    instr_entry.instr_location = "nb"
                else:
                    instr_entry.instr_location = "fb"
        else:
            raise NotImplementedError(
                "Unsupported opcode: {}".format(opcode)
            )
        return

    def _set_operand_location(self, instr_entry):
        location = instr_entry.instr_location
        
        # NOTE: handle special instructions first
        # special instructions: source operands and destination operands
        # may be in different locations
        if instr_entry.instr.opcode.startswith("ld.global"):
            assert "pred_reg" not in instr_entry.instr.metadata
            assert len(instr_entry.instr.src_operands) == 1
            assert len(instr_entry.instr.dst_operands) == 1
            # memory address fields of ld.global should be on subcore
            src_op = instr_entry.instr.src_operands[0]
            assert src_op.isnormalreg()
            instr_entry.src_loc_is_nb[src_op.op_str] = False
            # dst register should be on pg
            dst_op = instr_entry.instr.dst_operands[0]
            assert dst_op.isnormalreg()
            instr_entry.dst_loc_is_nb[dst_op.op_str] = True
        elif instr_entry.instr.opcode.startswith("st.global"):
            assert "pred_reg" not in instr_entry.instr.metadata
            assert len(instr_entry.instr.src_operands) == 2
            assert len(instr_entry.instr.dst_operands) == 0
            # memory address fields of st.global should be on subcore
            src_op = instr_entry.instr.src_operands[0]
            assert src_op.isnormalreg()
            instr_entry.src_loc_is_nb[src_op.op_str] = False
            # NOTE: data reg location should be pg
            src_op = instr_entry.instr.src_operands[1]
            assert src_op.isnormalreg()
            instr_entry.src_loc_is_nb[src_op.op_str] = True
        elif instr_entry.instr.opcode.startswith("ld.shared"):
            assert "pred_reg" not in instr_entry.instr.metadata
            assert len(instr_entry.instr.src_operands) == 1
            assert len(instr_entry.instr.dst_operands) == 1
            # memory address fields of ld.shared should be on subcore
            src_op = instr_entry.instr.src_operands[0]
            # NOTE: this could be a param value
            if src_op.isnormalreg():
                if self.config["default_smem_loc_is_near_bank"]:
                    # near-bank
                    instr_entry.src_loc_is_nb[src_op.op_str] = True
                else:
                    # far-bank
                    instr_entry.src_loc_is_nb[src_op.op_str] = False
            # dst register
            dst_op = instr_entry.instr.dst_operands[0]
            assert dst_op.isnormalreg()
            """
            if dst_op.op_str == src_op.op_str:
                instr_entry.dst_loc_is_nb[dst_op.op_str] = \
                    instr_entry.src_loc_is_nb[src_op.op_str]
            """
            if self.config["default_smem_loc_is_near_bank"]:
                # near-bank
                instr_entry.dst_loc_is_nb[dst_op.op_str] = True
            else:
                # far-bank
                instr_entry.dst_loc_is_nb[dst_op.op_str] = False
        elif instr_entry.instr.opcode.startswith("st.shared"):
            assert "pred_reg" not in instr_entry.instr.metadata
            assert len(instr_entry.instr.src_operands) == 2
            assert len(instr_entry.instr.dst_operands) == 0
            # memory address fields of st.shared should be on subcore
            src_op = instr_entry.instr.src_operands[0]
            # NOTE: this could be a param value
            if src_op.isnormalreg():
                if self.config["default_smem_loc_is_near_bank"]:
                    # near-bank
                    instr_entry.src_loc_is_nb[src_op.op_str] = True
                else:
                    # far-bank
                    instr_entry.src_loc_is_nb[src_op.op_str] = False
            # data reg location
            src_op = instr_entry.instr.src_operands[1]
            assert src_op.isnormalreg()
            if self.config["default_smem_loc_is_near_bank"]:
                # near-bank
                instr_entry.src_loc_is_nb[src_op.op_str] = True
            else:
                # far-bank
                instr_entry.src_loc_is_nb[src_op.op_str] = False
        elif instr_entry.instr.opcode.startswith("atom.shared"):
            assert "pred_reg" not in instr_entry.instr.metadata
            assert len(instr_entry.instr.src_operands) == 2
            assert len(instr_entry.instr.dst_operands) == 0
            # memory address fields of st.shared should be on subcore
            src_op = instr_entry.instr.src_operands[0]
            assert src_op.isnormalreg()
            if self.config["default_smem_loc_is_near_bank"]:
                # near-bank
                instr_entry.src_loc_is_nb[src_op.op_str] = True
            else:
                # far-bank
                instr_entry.src_loc_is_nb[src_op.op_str] = False
            # data reg location
            src_op = instr_entry.instr.src_operands[1]
            if src_op.isnormalreg():
                if self.config["default_smem_loc_is_near_bank"]:
                    # near-bank
                    instr_entry.src_loc_is_nb[src_op.op_str] = True
                else:
                    # far-bank
                    instr_entry.src_loc_is_nb[src_op.op_str] = False
            elif src_op.isimmvalue():
                pass
            else:
                "Unsupported atom.shared src operand type: {}"\
                    .format(src_op._type)
            # NOTE: dst register should be null
        elif location in {"fb", "nb"}:
            is_nb = True if location == "nb" else False
            if "pred_reg" in instr_entry.instr.metadata:
                pred_reg = instr_entry.instr.metadata["pred_reg"]
                instr_entry.src_loc_is_nb[pred_reg.op_str] = is_nb
            for src_op in instr_entry.instr.src_operands:
                if src_op.isnormalreg():
                    instr_entry.src_loc_is_nb[src_op.op_str] = is_nb
            for dst_op in instr_entry.instr.dst_operands:
                if dst_op.isnormalreg():
                    instr_entry.dst_loc_is_nb[dst_op.op_str] = is_nb
        else:
            raise NotImplementedError(
                "Unsupported instr location: {}"
                .format(location)
            )

    def _check_reg_ready(self, instr_entry):
        warp_id = instr_entry.warp_id
        track_entry = self.reg_track_table.entry[warp_id]
        if "pred_reg" in instr_entry.instr.metadata:
            pred_reg = instr_entry.instr.metadata["pred_reg"]
            is_nb = instr_entry.src_loc_is_nb[pred_reg.op_str]
            assert track_entry.check_ready(
                pred_reg.op_str,
                is_nb
            )
        for src_op in instr_entry.instr.src_operands:
            if src_op.isnormalreg():
                is_nb = instr_entry.src_loc_is_nb[src_op.op_str]
                assert track_entry.check_ready(src_op.op_str, is_nb)

    def _handle_instr_entry(self):
        """Process an instruction and dispatch it to the corresponding
        far-bank or near-bank engine depending on the instruction type,
        compiler hints, or register location information.
        """
        while True:
            instr_entry = yield self.instr_entry_queue.get()
            assert isinstance(instr_entry, InstrEntry)
           
            # Step 1.1: determine location for instruction offloading
            self._set_instr_location(instr_entry)
            # Step 1.2: determine location for src / dst operands
            self._set_operand_location(instr_entry)
            # NOTE: step 1 consumes 1 pipeline cycle
            yield self.env.timeout(1 * self.clock_unit)
            
            # Step 2: move all source registers to target location
            # update reg track table for both src / dst reg
            # NOTE: this is blocking
            offld_start = self.env.now
            yield self.env.process(
                self._move_instr_entry_reg(instr_entry)
            )
            offld_dur = self.env.now - offld_start
            # make sure all operands are in correct locations
            # before sending instr to opc
            # NOTE: we move this checking from opc to here
            # to handle to special case for st.global
            # since far-bank reg file may not contain valid data reg
            # for st.global but opc-lsu still helps collect it
            self._check_reg_ready(instr_entry)

            # Step 3: issue instruction to the corresponding engine
            opcode = instr_entry.instr.opcode.split(".")[0]
            if opcode in self.config["sfu_instr"]:
                yield self.subcore.opc_sfu\
                    .instr_entry_queue.put(instr_entry)
            elif opcode in self.config["lsu_instr"]:
                yield self.subcore.opc_lsu\
                    .instr_entry_queue.put(instr_entry)
            elif opcode in self.config["cfu_instr"]:
                yield self.subcore.opc_cfu\
                    .instr_entry_queue.put(instr_entry)
            elif opcode in self.config["syncu_instr"]:
                yield self.subcore.opc_syncu\
                    .instr_entry_queue.put(instr_entry)
            elif opcode in self.config["alu_instr"]:
                if instr_entry.instr_location == "nb":
                    # near-bank execution
                    # get dst pg_id
                    pg_id = self.subcore.warp_info_table\
                        .entry[instr_entry.warp_id].pg_id
                    instr_entry.pg_id = pg_id
                    # issue to pg, upstream traffic
                    yield self.subcore.core.subcore_pg_bus_arbiter\
                        .upstream_req_queue.put(instr_entry)
                elif instr_entry.instr_location == "fb":
                    # far-bank execution
                    yield self.subcore.opc_fb_alu\
                        .instr_entry_queue.put(instr_entry)
                else:
                    raise NotImplementedError(
                        "Unsupported location: {}"
                        .format(instr_entry.instr_location)
                    )
            else:
                raise NotImplementedError(
                    "Unknown instruction: {}"
                    .format(opcode)
                )
            self.subcore._append_trace_event_dur(
                tid="warp_{}-slot_{}"
                .format(
                    instr_entry.warp_id,
                    instr_entry.slot_id
                ),
                name="[offld]-{}"
                .format(
                    instr_entry.instr.trace_str()
                ),
                ts=instr_entry.last_trace_cyc,
                dur=self.env.now - instr_entry.last_trace_cyc,
                cat="instruction",
                args={
                    "original_instr": instr_entry.instr.instr_str,
                    "offld_dur": offld_dur
                }
            )
            instr_entry.last_trace_cyc = self.env.now

    def _move_instr_entry_reg(self, instr_entry):
        """This functions moves all source operand registers of the
        instruction entry to the given location (PG or subcore)
        """
        def issue_reg_mov(operand, warp_id, is_nb, op_str_set):
            track_entry = self.reg_track_table.entry[warp_id]
            reg_ready = track_entry.check_ready(
                operand.op_str,
                is_nb
            )
            if reg_ready is False:
                if operand.op_str not in op_str_set:
                    # issue reg move request
                    self.env.process(
                        self._move_reg_req(
                            operand=operand,
                            warp_id=warp_id,
                            is_upstream=is_nb
                        )
                    )
                    op_str_set.append(operand.op_str)
            return op_str_set

        op_str_set = []
        
        if "pred_reg" in instr_entry.instr.metadata:
            pred_reg = instr_entry.instr.metadata["pred_reg"]
            is_nb = instr_entry.src_loc_is_nb[pred_reg.op_str]
            op_str_set = issue_reg_mov(
                operand=pred_reg,
                warp_id=instr_entry.warp_id,
                is_nb=is_nb,
                op_str_set=op_str_set
            )
        for src_op in instr_entry.instr.src_operands:
            if src_op.isnormalreg():
                is_nb = instr_entry.src_loc_is_nb[src_op.op_str]
                op_str_set = issue_reg_mov(
                    operand=src_op,
                    warp_id=instr_entry.warp_id,
                    is_nb=is_nb,
                    op_str_set=op_str_set
                )
        
        # NOTE: issue register movement consumes 1 pipeline cycle
        if len(op_str_set) > 0:
            yield self.env.timeout(1 * self.clock_unit)

        # wait until all reg movements are done
        for i in range(len(op_str_set)):
            operand = yield self.reg_ack_queue[instr_entry.warp_id].get(
                lambda x: (x.op_str in op_str_set)
            )
            # update reg track table for src reg
            is_nb = instr_entry.src_loc_is_nb[operand.op_str]
            reg_file_type = "near-bank" if is_nb else "far-bank"
            self.reg_track_table.entry[instr_entry.warp_id]\
                .mov_update(
                    op_str=operand.op_str,
                    reg_file_type=reg_file_type
            )
        
        # update reg track table for dst reg
        for i in range(len(instr_entry.instr.dst_operands)):
            dst_op = instr_entry.instr.dst_operands[i]
            reg_file_type = "near-bank" \
                if instr_entry.dst_loc_is_nb[dst_op.op_str] \
                else "far-bank"
            self.subcore.reg_track_table.entry[instr_entry.warp_id]\
                .write_update(
                    op_str=dst_op.op_str,
                    reg_file_type=reg_file_type
            )
        
        return

    def _move_reg_req(self, operand, warp_id, is_upstream):
        assert isinstance(is_upstream, bool)
        # get pg id
        pg_id = self.subcore.warp_info_table.entry[warp_id].pg_id
        # compose request
        reg_move_req = RegMoveReq(
            operand=operand,
            subcore_id=self.subcore_id,
            pg_id=pg_id,
            warp_id=warp_id,
            is_upstream=is_upstream
        )
        reg_move_req.reg_size = self.subcore.warp_info_table\
            .entry[warp_id].prog_reg_size[operand.reg_prefix]
        # issue movement request
        yield self.reg_move_engine\
            .reg_req_queue.put(reg_move_req)
        self.env.process(
            self._move_reg_ack(
                operand=operand,
                warp_id=warp_id
            )
        )

    def _move_reg_ack(self, operand, warp_id):
        # wait until the movement is complete
        _ = yield self.reg_move_engine\
            .reg_ack_queue.get(
                lambda x: (x.operand.op_str == operand.op_str)
            )
        # ack
        yield self.reg_ack_queue[warp_id].put(operand)

