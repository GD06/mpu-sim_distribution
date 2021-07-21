from simulator.execution_unit import ExecutionUnit
from simulator.instr_instance import InstrEntry


class ControlFlowUnit(ExecutionUnit):

    def __init__(self, env, log, config, clock_unit, subcore): 
        super().__init__(env, log, config, clock_unit)
        self.subcore = subcore
        self.num_unit = self.config["num_cfu"]
        self.name = "cfu"
        for i in range(self.num_unit):
            self.env.process(self._process_instr_entry())

    def _process_instr_entry(self):
        while True:
            instr_entry = yield self.instr_entry_queue.get()
            assert isinstance(instr_entry, InstrEntry)

            # perform functional simulation
            instr_entry.process_operands()

            # Special handle of control instructions
            if instr_entry.instr.opcode.startswith("bra"):
                assert "dst_pc" in instr_entry.instr.metadata
                stack_top = self.subcore.stack_table\
                    .entry[instr_entry.warp_id].get_simt_mask()
                assert stack_top != 0, "The top of stack should not include " \
                    "an empty mask"
                # NOTE: consume 1 pipeline cycle
                yield self.env.timeout(1 * self.clock_unit)
                if instr_entry.simt_mask == 0:
                    pass
                elif instr_entry.simt_mask == stack_top:
                    block_name = instr_entry.instr.metadata["dst_pc"]
                    dst_pc = self.subcore.core.current_kernel\
                        .code_blocks[block_name]
                    self.subcore.warp_pipeline_table\
                        .entry[instr_entry.warp_id].next_pc = dst_pc
                else:
                    block_name = instr_entry.instr.metadata["dst_pc"]
                    dst_pc = self.subcore.core.current_kernel\
                        .code_blocks[block_name]
                    assert dst_pc != instr_entry.pc + 1,\
                        "The branch will not be executed at all!"
    
                    # Update the next PC in the top entry
                    top_entry = self.subcore.stack_table\
                        .entry[instr_entry.warp_id].top()
                    self.subcore.stack_table\
                        .entry[instr_entry.warp_id].pop()
                    new_entry = (
                        top_entry[0],  # Reconvergence point
                        instr_entry.instr.metadata["pdom"],  # Next PC
                        top_entry[2]  # SIMT mask
                    )
                    self.subcore.stack_table\
                        .entry[instr_entry.warp_id].push(new_entry)
                    # NOTE: consume 1 pipeline cycle
                    yield self.env.timeout(1 * self.clock_unit)
    
                    # The branch continues to PC + 1
                    not_pred_mask = instr_entry.pred_mask \
                        ^ int("1" * self.config["num_threads_per_warp"], 2)
                    new_entry = (
                        instr_entry.instr.metadata["pdom"],
                        instr_entry.pc + 1,
                        not_pred_mask & stack_top
                    )
                    self.subcore.stack_table\
                        .entry[instr_entry.warp_id].push(new_entry)
                    # NOTE: consume 1 pipeline cycle
                    yield self.env.timeout(1 * self.clock_unit)
    
                    # The branch jumps to the destination PC
                    if instr_entry.instr.metadata["pdom"] != dst_pc:
                        new_entry = (
                            instr_entry.instr.metadata["pdom"],
                            dst_pc,
                            instr_entry.pred_mask & stack_top
                        )
                        self.subcore.stack_table\
                            .entry[instr_entry.warp_id].push(new_entry)
                        self.subcore.warp_pipeline_table\
                            .entry[instr_entry.warp_id].next_pc = dst_pc
            elif instr_entry.instr.opcode.startswith("ret"):        
                # NOTE: consume 1 pipeline cycle
                yield self.env.timeout(1 * self.clock_unit)
                pass
            else:
                raise NotImplementedError("CFU not supported opcode: {}"
                                          .format(instr_entry.instr.opcode))
            
            instr_entry.processed = True

            yield self.subcore.writeback_buffer.put(instr_entry)

            # update trace event
            self.subcore._append_trace_event_dur(
                tid="warp_{}-slot_{}"
                .format(
                    instr_entry.warp_id,
                    instr_entry.slot_id
                ),
                name="[cfu]-{}".format(
                    instr_entry.instr.trace_str()
                ),
                ts=instr_entry.last_trace_cyc,
                dur=self.env.now - instr_entry.last_trace_cyc,
                cat="instruction"
            )
            instr_entry.last_trace_cyc = self.env.now
