import simpy 

from simulator.dram_message import DRAMCommand


class Bank:

    def __init__(self, env, config, log, pe):
        self.env = env 
        self.config = config 
        self.log = log
        self.pe = pe 
        self.hardware = pe.pg.core.processor.hardware 
        # reference to pg's trace event
        self._append_trace_event_dur = self.pe.pg._append_trace_event_dur
        self._trace_tid = "bank_{}".format(self.pe.pe_id)
        self._loc_str = "proc_id={proc_id}, core_id={core_id}, " \
            "pg_id={pg_id}, pe_id={pe_id}:".format(
                proc_id=self.pe.pg.core.processor.proc_id,
                core_id=self.pe.pg.core.core_id,
                pg_id=self.pe.pg.pg_id,
                pe_id=self.pe.pe_id
            )

        assert config["sim_clock_freq"] % config["dram_clock_freq"] == 0, (
            "Undividable simulation clock frequency")
        self.clock_unit = config["sim_clock_freq"] // config["dram_clock_freq"]

        # this queue stores memory transactions
        self.mem_trans_queue = []
        # this queue only stores memory tokens
        self.mem_trans_token_queue = simpy.Store(
            env, capacity=config["dram_trans_queue_size"]
        )

        # data structure for memory controller
        self.dram_cmd_queue = simpy.Store(env, capacity=1)
        self.dram_data_bus_sig = simpy.Store(env, capacity=1)
        self.last_mem_trans = None
        self.last_dram_cmd = None
        self.last_four_act_cyc = [0] * 4
        self.last_refresh_end_cyc = 0
        # used for refresh tracking
        self.dram_refresh_finish_sig = simpy.Store(env, capacity=1)
        self.dram_refresh_pending = False
        self.dram_trans_pending = False

        # spawn processes for memory transaction handling
        if config["dram_controller"] == "ideal":
            self.env.process(self._mem_ideal_trans_handler())
        elif config["dram_controller"] == "simple":
            self.env.process(self._mem_simple_trans_handler())
        else:
            raise NotImplementedError(
                "Unrecognized memory controller: {}".format(
                    config["dram_controller"])
            )
        # spawn process for dram command handling and state transitions
        if config["dram_controller"] == "simple":
            self.env.process(self._dram_cmd_handler())
        
        # NOTE: used for multiple row buffer
        self.num_rb = self.config["dram_bank_num_row_buf"]
        # row buffer addr --> last accessed cycle
        self.open_row_buf_set = {}
        
        # performance counter
        self.num_read = 0
        self.num_write = 0
        self.num_refresh = 0
        self.num_activation = 0
        self.num_precharge = 0

        return

    def reset_status(self):
        # NOTE: since a bank can be accessed by both local requests and 
        # remote requests, this should not disrupt bank state
        pass

    def _is_row_buf_hit(self, row_addr):
        if row_addr in self.open_row_buf_set:
            return True
        else:
            return False

    def _is_row_buf_avail(self):
        if len(self.open_row_buf_set) == self.num_rb:
            return False
        elif len(self.open_row_buf_set) < self.num_rb:
            return True
        else:
            assert False, "row buffer overflow!"

    def _create_row_buf(self, row_addr, time):
        assert row_addr not in self.open_row_buf_set
        self.open_row_buf_set[row_addr] = time

    def _update_row_buf_time(self, row_addr, time):
        assert row_addr in self.open_row_buf_set
        self.open_row_buf_set[row_addr] = time

    def _close_all_row_buf(self):
        self.open_row_buf_set = {}

    def _evict_row_buf(self):
        # NOTE: currently we use least-recently used policy
        assert len(self.open_row_buf_set) == self.num_rb
        row_to_evict = None
        row_to_evict_time = None
        first_item = True
        for row_addr in self.open_row_buf_set.keys():
            if first_item:
                first_item = False
                row_to_evict = row_addr
                row_to_evict_time = self.open_row_buf_set[row_addr]
            else:
                if self.open_row_buf_set[row_addr] < row_to_evict_time:
                    row_to_evict_time = self.open_row_buf_set[row_addr]
                    row_to_evict = row_addr
        del self.open_row_buf_set[row_to_evict]
        return row_to_evict

    def get_perf_metrics(self):
        """Get a dictionary of performance metrics."""
        perf_metrics = {}
        perf_metrics["num_read"] = self.num_read
        perf_metrics["num_write"] = self.num_write
        perf_metrics["num_refresh"] = self.num_refresh
        perf_metrics["num_act"] = self.num_activation
        perf_metrics["num_pre"] = self.num_precharge
        return {"dram_bank_{}".format(self.pe.pe_id): perf_metrics}

    def _update_four_act_cyc(self, cycle):
        """This function updates past four activation clock cycle in dram domain
        Args:
            cycle: the dram clock cycle of the current issued ACT command
        Return:
        """
        self.last_four_act_cyc[3] = self.last_four_act_cyc[2]
        self.last_four_act_cyc[2] = self.last_four_act_cyc[1]
        self.last_four_act_cyc[1] = self.last_four_act_cyc[0]
        self.last_four_act_cyc[0] = cycle

    def _dram_cmd_handler(self):
        """This is the dram state transition logic
        Args:
        Return:
        """
        while True:
            if (
                (self.env.now - self.last_refresh_end_cyc) 
                > self.config["dram_tREFI"] * self.clock_unit
            ):
                self.dram_refresh_pending = True
                if len(self.dram_cmd_queue.items) > 0:
                    # there are still pending commands
                    # we must drain command queue first
                    dram_cmd = yield self.dram_cmd_queue.get()
                else:
                    # command queue is drained
                    # dram refresh should start
                    dram_cmd = DRAMCommand("REF", None, None)
            else:
                dram_cmd = yield self.dram_cmd_queue.get()
            # NOTE: update tracing event
            trace_event_start = self.env.now
            # DRAM state machine
            if dram_cmd.type == "ACT":
                self.num_activation += 1

                if (
                    (self.last_four_act_cyc[3] != 0) 
                    and ((self.env.now - self.last_four_act_cyc[3]) 
                         < self.config["dram_tFAW"] * self.clock_unit)
                ):
                    yield self.env.timeout(
                        self.config["dram_tFAW"] * self.clock_unit 
                        - (self.env.now - self.last_four_act_cyc[3])
                    )

                if self.last_dram_cmd is None:
                    # this is the initial state
                    self.log.debug("dram initialized")

                elif (
                    self.last_dram_cmd.type == "PRE"
                    or self.last_dram_cmd.type == "REF"
                ):
                    if ((self.env.now - self.last_dram_cmd.time) 
                            < self.config["dram_tRP"] * self.clock_unit):
                        yield self.env.timeout(
                            self.config["dram_tRP"] * self.clock_unit 
                            - (self.env.now - self.last_dram_cmd.time)
                        )

                elif self.last_dram_cmd.type == "READ_PRE":
                    if ((self.env.now - self.last_dram_cmd.time) 
                            < (self.config["dram_tRTPL"] 
                               + self.config["dram_tRP"]) * self.clock_unit):
                        yield self.env.timeout(
                            (self.config["dram_tRTPL"] 
                                + self.config["dram_tRP"]) * self.clock_unit 
                            - (self.env.now - self.last_dram_cmd.time)
                        )

                elif self.last_dram_cmd.type == "WRITE_PRE":
                    if ((self.env.now - self.last_dram_cmd.time) 
                            < (self.config["dram_tWTRL"] 
                               + self.config["dram_tRP"]) * self.clock_unit):
                        yield self.env.timeout(
                            (self.config["dram_tWTRL"] 
                                + self.config["dram_tRP"]) * self.clock_unit 
                            - (self.env.now - self.last_dram_cmd.time)
                        )
                elif self.last_dram_cmd.type == "READ":
                    # NOTE: row buffer parallelism
                    pass
                elif self.last_dram_cmd.type == "WRITE":
                    # NOTE: row buffer parallelism
                    pass
                else:
                    assert False, "wrong dram command sequence!"
                
                # update activation window
                self._update_four_act_cyc(self.env.now)
                # This command is ready to issue
                dram_cmd.time = self.env.now
                self.last_dram_cmd = dram_cmd
                if self.config["dram_page_policy"] == "open_page":
                    self._update_row_buf_time(dram_cmd.row_addr, self.env.now)

            elif dram_cmd.type == "PRE":
                self.num_precharge += 1
                assert self.last_dram_cmd is not None
                if self.last_dram_cmd.type == "READ":
                    if ((self.env.now - self.last_dram_cmd.time) 
                            < self.config["dram_tRTPL"] * self.clock_unit):
                        yield self.env.timeout(
                            self.config["dram_tRTPL"] * self.clock_unit 
                            - (self.env.now - self.last_dram_cmd.time)
                        )

                elif self.last_dram_cmd.type == "WRITE":
                    if ((self.env.now - self.last_dram_cmd.time) 
                            < self.config["dram_tWTRL"] * self.clock_unit):
                        yield self.env.timeout(
                            self.config["dram_tWTRL"] * self.clock_unit 
                            - (self.env.now - self.last_dram_cmd.time)
                        )

                elif self.last_dram_cmd.type == "ACT":
                    if ((self.env.now - self.last_dram_cmd.time) 
                            < self.config["dram_tRAS"] * self.clock_unit):
                        yield self.env.timeout(
                            self.config["dram_tRAS"] * self.clock_unit 
                            - (self.env.now - self.last_dram_cmd.time)
                        )
                else:
                    assert False, "wrong dram command sequence!"

                # This command is ready to issue
                dram_cmd.time = self.env.now
                self.last_dram_cmd = dram_cmd

            elif dram_cmd.type == "READ":
                self.num_read += 1
                assert self.last_dram_cmd is not None
                assert self._is_row_buf_hit(dram_cmd.row_addr)
                if (self.last_dram_cmd.type == "READ" 
                        or self.last_dram_cmd.type == "WRITE"):
                    if ((self.env.now - self.last_dram_cmd.time) 
                            < self.config["dram_tCCDL"] * self.clock_unit):
                        yield self.env.timeout(
                            self.config["dram_tCCDL"] * self.clock_unit 
                            - (self.env.now - self.last_dram_cmd.time)
                        )

                elif self.last_dram_cmd.type == "ACT":
                    if (
                        (self.env.now - self.last_dram_cmd.time) 
                            < self.config["dram_tRCDR"] * self.clock_unit
                    ):
                        yield self.env.timeout(
                            self.config["dram_tRCDR"] * self.clock_unit 
                            - (self.env.now - self.last_dram_cmd.time)
                        )
                else:
                    assert False, "wrong dram command sequence: {} -> READ"\
                        .format(self.last_dram_cmd.type)

                self._update_row_buf_time(dram_cmd.row_addr, self.env.now)
                # This command is ready to issue
                dram_cmd.time = self.env.now
                self.last_dram_cmd = dram_cmd
                # prepare data on the data bus
                yield self.env.timeout(self.config["dram_CL"] * self.clock_unit)
                yield self.dram_data_bus_sig.put(1)

            elif dram_cmd.type == "WRITE":
                self.num_write += 1
                assert self.last_dram_cmd is not None
                assert self._is_row_buf_hit(dram_cmd.row_addr)
                if (self.last_dram_cmd.type == "READ" 
                        or self.last_dram_cmd.type == "WRITE"):
                    if ((self.env.now - self.last_dram_cmd.time) 
                            < self.config["dram_tCCDL"] * self.clock_unit):
                        yield self.env.timeout(
                            self.config["dram_tCCDL"] * self.clock_unit 
                            - (self.env.now - self.last_dram_cmd.time)
                        )

                elif self.last_dram_cmd.type == "ACT":
                    if ((self.env.now - self.last_dram_cmd.time) 
                            < self.config["dram_tRCDW"] * self.clock_unit):
                        yield self.env.timeout(
                            self.config["dram_tRCDW"] * self.clock_unit 
                            - (self.env.now - self.last_dram_cmd.time)
                        )
                else:
                    assert False, "wrong dram command sequence!"

                self._update_row_buf_time(dram_cmd.row_addr, self.env.now)
                # This command is ready to issue
                dram_cmd.time = self.env.now
                self.last_dram_cmd = dram_cmd
                # Prepare data on the data bus
                yield self.env.timeout(self.config["dram_CWL"] 
                                       * self.clock_unit)
                yield self.dram_data_bus_sig.put(1)

            elif dram_cmd.type == "READ_PRE":
                self.num_read += 1
                self.num_precharge += 1
                assert self.last_dram_cmd is not None
                if (self.last_dram_cmd.type == "READ" 
                        or self.last_dram_cmd.type == "WRITE"):
                    if ((self.env.now - self.last_dram_cmd.time) 
                            < self.config["dram_tCCDL"] * self.clock_unit):
                        yield self.env.timeout(
                            self.config["dram_tCCDL"] * self.clock_unit 
                            - (self.env.now - self.last_dram_cmd.time)
                        )

                elif self.last_dram_cmd.type == "ACT":
                    if ((self.env.now - self.last_dram_cmd.time) 
                            < self.config["dram_tRCDR"] * self.clock_unit):
                        yield self.env.timeout(
                            self.config["dram_tRCDR"] * self.clock_unit 
                            - (self.env.now - self.last_dram_cmd.time)
                        )
                else:
                    assert False, "wrong dram command sequence!"

                # This command is ready to issue
                dram_cmd.time = self.env.now
                self.last_dram_cmd = dram_cmd
                # prepare data on the data bus
                yield self.env.timeout(self.config["dram_CL"] 
                                       * self.clock_unit)
                yield self.dram_data_bus_sig.put(1)

            elif dram_cmd.type == "WRITE_PRE":
                self.num_write += 1
                self.num_precharge += 1
                assert self.last_dram_cmd is not None
                if (self.last_dram_cmd.type == "READ" 
                        or self.last_dram_cmd.type == "WRITE"):
                    if ((self.env.now - self.last_dram_cmd.time) 
                            < self.config["dram_tCCDL"] * self.clock_unit):
                        yield self.env.timeout(
                            self.config["dram_tCCDL"] * self.clock_unit 
                            - (self.env.now - self.last_dram_cmd.time)
                        )

                elif self.last_dram_cmd.type == "ACT":
                    if ((self.env.now - self.last_dram_cmd.time) 
                            < self.config["dram_tRCDW"] * self.clock_unit):
                        yield self.env.timeout(
                            self.config["dram_tRCDW"] * self.clock_unit 
                            - (self.env.now - self.last_dram_cmd.time)
                        )
                else:
                    assert False, "wrong dram command sequence!"

                # This command is ready to issue
                dram_cmd.time = self.env.now
                self.last_dram_cmd = dram_cmd
                # prepare data on the data bus
                yield self.env.timeout(self.config["dram_CWL"] 
                                       * self.clock_unit)
                yield self.dram_data_bus_sig.put(1)

            elif dram_cmd.type == "REF":
                self.num_refresh += 1
                if self.last_dram_cmd.type == "PRE":
                    if ((self.env.now - self.last_dram_cmd.time) 
                            < self.config["dram_tRP"] * self.clock_unit):
                        # wait until we can start refresh
                        yield self.env.timeout(
                            self.config["dram_tRP"] * self.clock_unit 
                            - (self.env.now - self.last_dram_cmd.time)
                        )

                elif self.last_dram_cmd.type == "READ_PRE":
                    if ((self.env.now - self.last_dram_cmd.time) 
                            < (self.config["dram_tRTPL"] 
                                + self.config["dram_tRP"]) * self.clock_unit):
                        # wait until we can start refresh
                        yield self.env.timeout(
                            (self.config["dram_tRTPL"] 
                                + self.config["dram_tRP"]) * self.clock_unit 
                            - (self.env.now - self.last_dram_cmd.time)
                        )

                elif self.last_dram_cmd.type == "WRITE_PRE":
                    if ((self.env.now - self.last_dram_cmd.time) 
                            < (self.config["dram_tWTRL"] 
                                + self.config["dram_tRP"]) * self.clock_unit):
                        # wait until we can start refresh
                        yield self.env.timeout(
                            (self.config["dram_tWTRL"] 
                                + self.config["dram_tRP"]) * self.clock_unit 
                            - (self.env.now - self.last_dram_cmd.time)
                        )

                elif self.last_dram_cmd.type == "READ":
                    if ((self.env.now - self.last_dram_cmd.time) 
                            < (self.config["dram_tRTPL"] 
                                + self.config["dram_tRP"]) * self.clock_unit):
                        # wait until we can start refresh
                        yield self.env.timeout(
                            (self.config["dram_tRTPL"] 
                                + self.config["dram_tRP"]) * self.clock_unit 
                            - (self.env.now - self.last_dram_cmd.time)
                        )

                elif self.last_dram_cmd.type == "WRITE":
                    if ((self.env.now - self.last_dram_cmd.time) 
                            < (self.config["dram_tWTRL"] 
                                + self.config["dram_tRP"]) * self.clock_unit):
                        # wait until we can start refresh
                        yield self.env.timeout(
                            (self.config["dram_tWTRL"] 
                                + self.config["dram_tRP"]) * self.clock_unit 
                            - (self.env.now - self.last_dram_cmd.time)
                        )

                else:
                    assert False, "wrong dram command"
                
                # start refresh
                yield self.env.timeout(
                    self.config["dram_tRFC"] * self.clock_unit
                )
                # refresh ready
                self.last_refresh_end_cyc = self.env.now
                # update cmd
                self.dram_refresh_pending = False
                self.last_dram_cmd = dram_cmd
                self.last_dram_cmd.time = self.env.now
                # update row buffer
                self._close_all_row_buf()
                # if there is pending trans, inform it
                if self.dram_trans_pending:
                    yield self.dram_refresh_finish_sig.put(1)
            else:
                assert False, "wrong dram command"
            
            # NOTE: update tracing event
            if dram_cmd.type == "REF":
                self._append_trace_event_dur(
                    tid=self._trace_tid,
                    name=dram_cmd.type,
                    ts=trace_event_start,
                    dur=self.env.now - trace_event_start,
                    cat="dram_command",
                    args={
                        "row_addr": dram_cmd.row_addr,
                        "col_addr": dram_cmd.col_addr
                    }
                )
            else:
                args = {}
                if dram_cmd.dram_trans.is_remote:
                    args["remote_access"] = True
                else:
                    args["subcore_id"] = \
                        dram_cmd.dram_trans.trace_subcore_id
                    args["warp_id"] = \
                        dram_cmd.dram_trans.trace_warp_id
                args["row_addr"] = dram_cmd.row_addr
                args["col_addr"] = dram_cmd.col_addr
                self._append_trace_event_dur(
                    tid=self._trace_tid,
                    name=dram_cmd.type,
                    ts=self.env.now - 2 * self.clock_unit,
                    dur=2 * self.clock_unit,
                    cat="dram_command",
                    args=args
                )

    def _print_dram_cmd_queue(self):
        for item in self.dram_cmd_queue.items:
            print(item.type)

    def _translate_enqueue_dram_cmd(self, trans):
        """This function translate dram transaction into dram commands and 
        enqueue. 

        Arg:
            trans: the memory transaction to be translated and added to command
            queue
        Return:
        """
        if (
            self.dram_refresh_pending is True
        ):
            self.dram_trans_pending = True
            _ = yield self.dram_refresh_finish_sig.get()
            self.dram_trans_pending = False
        if trans.type == "load":
            if self.config["dram_page_policy"] == "close_page":
                yield self.dram_cmd_queue.put(
                    DRAMCommand(
                        "ACT", trans.row_addr, trans.col_addr, trans
                    )
                )
                yield self.dram_cmd_queue.put(
                    DRAMCommand(
                        "READ_PRE", trans.row_addr, trans.col_addr, trans
                    )
                )
                self.last_mem_trans = trans
            elif self.config["dram_page_policy"] == "open_page":
                if self._is_row_buf_hit(trans.row_addr):
                    # row buffer hit
                    self._update_row_buf_time(trans.row_addr, self.env.now)
                    # translate command
                    yield self.dram_cmd_queue.put(
                        DRAMCommand(
                            "READ", trans.row_addr, trans.col_addr, trans
                        )
                    )
                    self.last_mem_trans = trans
                else:
                    # row buffer miss
                    if self._is_row_buf_avail():
                        # there are still available row buffers
                        self._create_row_buf(trans.row_addr, self.env.now)
                        # translate command
                        yield self.dram_cmd_queue.put(
                            DRAMCommand(
                                "ACT", trans.row_addr, trans.col_addr, trans
                            )
                        )
                        yield self.dram_cmd_queue.put(
                            DRAMCommand(
                                "READ", trans.row_addr, trans.col_addr, trans
                            )
                        )
                        self.last_mem_trans = trans
                    else:
                        # need eviction
                        row_to_evict = self._evict_row_buf()
                        yield self.dram_cmd_queue.put(
                            DRAMCommand(
                                "PRE", row_to_evict, trans.col_addr, trans
                            )
                        )
                        self._create_row_buf(trans.row_addr, self.env.now)
                        yield self.dram_cmd_queue.put(
                            DRAMCommand(
                                "ACT", trans.row_addr, trans.col_addr, trans
                            )
                        )
                        yield self.dram_cmd_queue.put(
                            DRAMCommand(
                                "READ", trans.row_addr, trans.col_addr, trans
                            )
                        )
                        self.last_mem_trans = trans
            else:
                assert False, "wrong page policy: {}"\
                    .format(self.config["dram_page_policy"])
        elif trans.type == "store":
            if self.config["dram_page_policy"] == "close_page":
                yield self.dram_cmd_queue.put(
                    DRAMCommand(
                        "ACT", trans.row_addr, trans.col_addr, trans
                    )
                )
                yield self.dram_cmd_queue.put(
                    DRAMCommand(
                        "WRITE_PRE", trans.row_addr, trans.col_addr, trans
                    )
                )
                self.last_mem_trans = trans
            elif self.config["dram_page_policy"] == "open_page":
                if self._is_row_buf_hit(trans.row_addr):
                    # row buffer hit
                    self._update_row_buf_time(trans.row_addr, self.env.now)
                    # translate command
                    yield self.dram_cmd_queue.put(
                        DRAMCommand(
                            "WRITE", trans.row_addr, trans.col_addr, trans
                        )
                    )
                    self.last_mem_trans = trans
                else:
                    # row buffer miss
                    if self._is_row_buf_avail():
                        # there are still available row buffers
                        self._create_row_buf(trans.row_addr, self.env.now)
                        # translate command
                        yield self.dram_cmd_queue.put(
                            DRAMCommand(
                                "ACT", trans.row_addr, trans.col_addr, trans
                            )
                        )
                        yield self.dram_cmd_queue.put(
                            DRAMCommand(
                                "WRITE", trans.row_addr, trans.col_addr, trans
                            )
                        )
                        self.last_mem_trans = trans
                    else:
                        # need eviction
                        row_to_evict = self._evict_row_buf()
                        yield self.dram_cmd_queue.put(
                            DRAMCommand(
                                "PRE", row_to_evict, trans.col_addr, trans
                            )
                        )
                        self._create_row_buf(trans.row_addr, self.env.now)
                        yield self.dram_cmd_queue.put(
                            DRAMCommand(
                                "ACT", trans.row_addr, trans.col_addr, trans
                            )
                        )
                        yield self.dram_cmd_queue.put(
                            DRAMCommand(
                                "WRITE", trans.row_addr, trans.col_addr, trans
                            )
                        )
                        self.last_mem_trans = trans
            else:
                assert False, "wrong page policy: {}"\
                    .format(self.config["dram_page_policy"])
        else:
            raise NotImplementedError(
                "Unrecognized memory transaction: {}".format(trans.type)
            )
        return

    def _get_mem_trans(self):
        """Return a memory transaction from transaction queue according to 
        scheduling policy
        
        Args:
        Returns:
            trans: memory transaction to be issued
        """
        trans = None
        assert len(self.mem_trans_queue) > 0
        if self.config["dram_schedule_policy"] == "FCFS":
            trans = self.mem_trans_queue.pop(0)  # get the first request
        elif self.config["dram_schedule_policy"] == "FRFCFS":
            for i in range(len(self.mem_trans_queue)):
                # iterate queue from start till end
                if self._is_row_buf_hit(self.mem_trans_queue[i].row_addr):
                    # this is the earliest transaction that results in a row hit
                    trans = self.mem_trans_queue.pop(i)
                    break
            if trans is None:
                # no row hit transaction found
                trans = self.mem_trans_queue.pop(0)  # get the first request
        else:
            raise NotImplementedError(
                "Unrecognized scheduling policy: {}".format(
                    self.config["dram_schedule_policy"])
            )
        return trans

    def _mem_simple_trans_handler(self):
        """This is the simple memory controller
        Args:
        Return:
        """
        while True:
            yield self.mem_trans_token_queue.get()
            trans = self._get_mem_trans()
            if trans.type == "load":
                # Translate transaction into dram commands and enqueue
                self.env.process(self._translate_enqueue_dram_cmd(trans))
                # Wait until data appears on data bus
                yield self.dram_data_bus_sig.get()
                trans.data = self.hardware.mem.get_value(
                    trans.get_mem_addr(), 
                    self.config["dram_bank_io_width"]
                )
                yield self.pe.pg.lsu_extension\
                    .pe_out_dram_trans_queue.put(trans)
            elif trans.type == "store":
                # Translate transaction into dram commands and enqueue
                self.env.process(self._translate_enqueue_dram_cmd(trans))
                # Wait until data could be placed on data bus
                yield self.dram_data_bus_sig.get()
                addr = trans.get_mem_addr()
                data = trans.data
                assert len(data) == self.config["dram_bank_io_width"]
                self.hardware.mem.set_value(addr, data)
                yield self.pe.pg.lsu_extension\
                    .pe_out_dram_trans_queue.put(trans)
            else:
                raise NotImplementedError(
                    "Unrecognized memory transaction: {}".format(trans.type)
                )

    def _mem_ideal_trans_handler(self):
        """This is the ideal memory controller that assumes fixed load/store 
        latency 
        Args:
        Return:
        """
        while True:
            yield self.mem_trans_token_queue.get()
            trans = self._get_mem_trans()
            if trans.type == "load":
                data = self.hardware.mem.get_value(
                    trans.get_mem_addr(), self.config["dram_bank_io_width"]
                )
                trans.data = data
                yield self.env.timeout(
                    self.config["dram_ideal_load_latency"] * self.clock_unit) 
                yield self.pe.pg.lsu_extension\
                    .pe_out_dram_trans_queue.put(trans) 
            elif trans.type == "store":
                addr = trans.get_mem_addr()
                data = trans.data
                assert len(data) == self.config["dram_bank_io_width"]
                self.hardware.mem.set_value(addr, data)
                yield self.env.timeout(
                    self.config["dram_ideal_store_latency"] * self.clock_unit)
                yield self.pe.pg.lsu_extension\
                    .pe_out_dram_trans_queue.put(trans)
            else:
                raise NotImplementedError(
                    "Unrecognized memory transaction: {}".format(trans.type)
                )

        return
