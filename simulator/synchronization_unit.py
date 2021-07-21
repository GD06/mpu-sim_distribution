import numpy as np

from simulator.execution_unit import ExecutionUnit
from simulator.instr_instance import InstrEntry


class SynchronizationUnit(ExecutionUnit):

    def __init__(self, env, log, config, clock_unit, subcore): 
        """Each subcore has only one synchronization unit
        """
        super().__init__(env, log, config, clock_unit)
        self.num_unit = self.config["num_syncu"]
        self.subcore = subcore
        self.name = "syncu"
        for i in range(self.config["max_num_warp_per_subcore"]):
            self.env.process(self._process_instr_entry())

    def _process_instr_entry(self):
        """This function tracks the synchronization board to decide whether
        to release the synchronization barrier
        """
        while True:
            instr_entry = yield self.instr_entry_queue.get()
            assert isinstance(instr_entry, InstrEntry)

            # perform functional simulation
            instr_entry.process_operands()

            thread_count = 0
            for i in range(self.config["num_threads_per_warp"]):
                valid = (instr_entry.simt_mask >> i) & 1
                thread_count = thread_count + valid

            assert thread_count != 0, "The empty sync instruction should " \
                "not reach the execution."

            if instr_entry.instr.metadata["num_threads"] is None:
                threshold = int(np.prod(self.subcore.core.block_dim))
            else:
                raise NotImplementedError

            bar_id = instr_entry.instr.metadata["bar_id"]
            assert bar_id <= self.config["max_bar_id_per_block"], "The bar " \
                "out of range!"

            warp_id = instr_entry.warp_id
            block_id = self.subcore.warp_info_table\
                .entry[warp_id].block_id

            # ensure all instr before barrier is drained
            while True:
                if (
                    self.subcore
                        .num_issued_not_commit_instr[block_id][warp_id] > 1
                ):
                    # NOTE: consume 1 pipeline cycle
                    yield self.env.timeout(1 * self.clock_unit)
                    continue
                # now only this sync instr is not committed
                assert self.subcore\
                    .num_issued_not_commit_instr[block_id][warp_id] == 1
                self.subcore.core.bar_count[block_id][bar_id] += thread_count
                break

            # wait for all warps to reach barrier
            while True:
                break_cond = (
                    self.subcore.core.bar_count[block_id][bar_id] >= threshold
                    or self.subcore.core.bar_release[block_id][bar_id]
                )
                if break_cond:
                    # barrier is in sync
                    self.subcore.core.bar_release[block_id][bar_id] = True
                    break
                else:
                    # NOTE: consume 1 pipeline cycle
                    yield self.env.timeout(1 * self.clock_unit)
                    continue

            # leave barrier
            self.subcore.core.bar_count[block_id][bar_id] -= thread_count
            if self.subcore.core.bar_count[block_id][bar_id] <= 0:
                # this is the last warp to leave this barrier
                self.subcore.core.bar_count[block_id][bar_id] = 0
                self.subcore.core.bar_release[block_id][bar_id] = False

            instr_entry.processed = True

            yield self.subcore.writeback_buffer.put(instr_entry)

            # update trace event
            self.subcore._append_trace_event_dur(
                tid="warp_{}-slot_{}"
                .format(
                    instr_entry.warp_id,
                    instr_entry.slot_id
                ),
                name="[syncu]-{}".format(
                    instr_entry.instr.trace_str()
                ),
                ts=instr_entry.last_trace_cyc,
                dur=self.env.now - instr_entry.last_trace_cyc,
                cat="instruction"
            )
            instr_entry.last_trace_cyc = self.env.now
        return
