from simulator.execution_unit import ExecutionUnit
from simulator.instr_instance import InstrEntry


class SpecialFunctionUnit(ExecutionUnit):
    
    def __init__(self, env, log, config, clock_unit, subcore):
        super().__init__(env, log, config, clock_unit)
        self.subcore = subcore
        self.num_unit = self.config["num_sfu"]
        self.name = "sfu"
        for i in range(self.num_unit):
            self.env.process(self._process_instr_entry())
        # performance metrics
        self.num_sfu_instr = {}

    def _record_sfu_instr(self, opcode):
        if opcode in self.num_sfu_instr:
            self.num_sfu_instr[opcode] += 1
        else:
            self.num_sfu_instr[opcode] = 1

    def get_perf_metrics(self):
        """Get a dictionary of performance metrics."""
        perf_metrics = self.num_sfu_instr
        return {"sfu": perf_metrics}

    def _process_instr_entry(self):
        while True:
            instr_entry = yield self.instr_entry_queue.get()
            assert isinstance(instr_entry, InstrEntry)
            
            # perform functional simulation
            instr_entry.process_operands()

            # spawn a process to simulate timing of this SFU
            self.env.process(
                self._delay_and_writeback(instr_entry)
            )
            # NOTE: can accept instruction in the next cycle
            yield self.env.timeout(1 * self.clock_unit)
            # update performance counter
            self._record_sfu_instr(instr_entry.instr.opcode)

    def _delay_and_writeback(self, instr_entry):
        # conduct timing simulation
        assert instr_entry.instr.latency > 0
        yield self.env.timeout(instr_entry.instr.latency * self.clock_unit)
        instr_entry.processed = True

        yield self.subcore.writeback_buffer.put(instr_entry)
        
        # update trace event
        self.subcore._append_trace_event_dur(
            tid="warp_{}-slot_{}"
            .format(
                instr_entry.warp_id,
                instr_entry.slot_id
            ),
            name="[sfu]-{}".format(
                instr_entry.instr.trace_str()
            ),
            ts=instr_entry.last_trace_cyc,
            dur=self.env.now - instr_entry.last_trace_cyc,
            cat="instruction"
        )
        instr_entry.last_trace_cyc = self.env.now
