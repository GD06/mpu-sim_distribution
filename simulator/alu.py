from simulator.execution_unit import ExecutionUnit
from simulator.instr_instance import InstrEntry


class ArithmeticLogicUnit(ExecutionUnit):
    
    def __init__(self, env, log, config, clock_unit, backend, alu_type):
        super().__init__(env, log, config, clock_unit)
        self.backend = backend
        if alu_type == "far-bank":
            self.num_unit = self.config["num_fb_alu"]
            self.is_far_bank = True
            self.name = "fb-alu"
        elif alu_type == "near-bank":
            self.num_unit = self.config["num_nb_alu"]
            self.is_far_bank = False
            self.name = "nb-alu"
        else:
            raise NotImplementedError(
                "Unknown ALU type:{}".format(alu_type)
            )
        for i in range(self.num_unit):
            self.env.process(self._process_instr_entry())
        # performance metrics
        self.num_alu_instr = {}

    def _record_alu_instr(self, opcode):
        if opcode in self.num_alu_instr:
            self.num_alu_instr[opcode] += 1
        else:
            self.num_alu_instr[opcode] = 1

    def get_perf_metrics(self):
        """Get a dictionary of performance metrics."""
        perf_metrics = self.num_alu_instr
        if self.is_far_bank:
            return {"fb_alu": perf_metrics}
        else:
            return {"nb_alu": perf_metrics}

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

    def _process_instr_entry(self):
        while True:
            instr_entry = yield self.instr_entry_queue.get()
            assert isinstance(instr_entry, InstrEntry)

            # perform functional simulation
            instr_entry.process_operands()

            # spawn a process to simulate timing of this ALU
            self.env.process(
                self._delay_and_writeback(instr_entry)
            )
            # NOTE: can accept instruction in the next cycle
            yield self.env.timeout(1 * self.clock_unit)
            # update performance counter
            self._record_alu_instr(instr_entry.instr.opcode)

    def _delay_and_writeback(self, instr_entry):
        # conduct timing simulation
        assert instr_entry.instr.latency > 0
        yield self.env.timeout(instr_entry.instr.latency * self.clock_unit)
        instr_entry.processed = True

        yield self.backend.writeback_buffer.put(instr_entry)
        # update trace event
        self._append_trace_event_dur(
            instr_entry=instr_entry,
            name="[{}]-{}"
            .format(
                self.name,
                instr_entry.instr.trace_str()
            ),
            ts=instr_entry.last_trace_cyc,
            dur=self.env.now - instr_entry.last_trace_cyc
        )
        instr_entry.last_trace_cyc = self.env.now
