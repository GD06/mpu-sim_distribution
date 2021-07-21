from copy import deepcopy
import simpy

from simulator.dram_message import DRAMTransaction
from simulator.network_message import DstRemoteLoadReq, DstRemoteLoadResp, \
    DstRemoteStoreReq, DstRemoteStoreResp
from simulator.readonly_dcache import ReadOnlyDataCache


class LoadStoreUnitRemote:

    def __init__(self, env, log, config, clock_unit, core):
        self.env = env
        self.log = log
        self.config = config
        self.clock_unit = clock_unit
        self.core = core
        self.dcache = ReadOnlyDataCache(
            name="lsu_remote",
            env=env,
            log=log,
            config=config,
            clock_unit=clock_unit,
            granularity=self.config["dram_bank_io_width"]
        )
        # reference the address hashing function in sim_api
        self.addr_hashing = self.core.processor\
            .hardware.addr_hashing
        self.re_addr_hashing = self.core.processor\
            .hardware.re_addr_hashing
        self.translate_bank_addr = self.core.processor\
            .hardware.translate_bank_addr
        # reference to bus arbiter
        self.bus_arbiter = self.core.subcore_pg_bus_arbiter
        # queues for receiving remote ld.global/st.global request
        self.remote_req_queue = simpy.Store(env, capacity=1)
        self.remote_resp_queue = simpy.FilterStore(env)
        self.num_fb_remote_prt_entry = \
            self.config["max_num_fb_remote_prt_entry"]
        # spawn processes to process remote ld.global/st.global request
        for remote_fb_prt_id in range(self.num_fb_remote_prt_entry):
            self.env.process(
                self._process_remote_ld_st_global(remote_fb_prt_id)
            )
        # queues for receiving returned dram transactions
        self.in_dram_trans_queue = simpy.FilterStore(env)
        # performance counter
        self.num_prt_read = 0
        self.num_prt_write = 0

    def get_perf_metrics(self):
        """Get a dictionary of performance metrics."""
        perf_metrics = {}
        perf_metrics["num_prt_read"] = self.num_prt_read
        perf_metrics["num_prt_write"] = self.num_prt_write
        return {"lsu_remote": perf_metrics}

    def _memory_coalesce(self, mem_loc_list, is_ld, data_width, num_addr):
        """Perform memory coalescing
        Args:
            mem_loc_list: a list of location tuples formatted as
                (proc_id_y, proc_id_x, core_id_y, core_id_x, pg_id, pe_id,
                bank_addr, bank_interface_offset)
            is_ld: True if is load
            data_width: data width of a single access
            num_addr: number of addresses
        Return:
            offset_list: offset into bank interface
            co_addr_list: coalesced dram transaction
        """
        trans_type = "load" if is_ld else "store"
        offset_list = [None] * num_addr
        # coalesced address list
        # global_mem_addr (aligned to bank interface)
        #   -> dram transaction
        co_addr_list = {}
        for i in range(num_addr):
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
                dram_trans.simt_mask += 1 << i
                # add to list
                co_addr_list[global_mem_addr] = dram_trans
        return offset_list, co_addr_list

    def _process_remote_ld_st_global(self, prt_id):
        while True:
            req = yield self.remote_req_queue.get()
            if isinstance(req, DstRemoteLoadReq):
                is_ld = True
            elif isinstance(req, DstRemoteStoreReq):
                is_ld = False
            else:
                raise NotImplementedError(
                    "LSU-Remote: unsupported request: {}"
                    .format(type(req))
                )
            addr_list = req.addr_list
            data_width = req.data_width
            num_addr = len(addr_list)
            # each decoded address has the format:
            #   proc_id_y, proc_id_x, core_id_y, core_id_x, pg_id, pe_id,
            #   bank_addr, bank_interface_offset
            mem_loc_list = []
            for i in range(num_addr):
                mem_loc_list.append(
                    self.addr_hashing(addr_list[i])
                )
            # perform memory coalescing
            offset_list, co_addr_list = self._memory_coalesce(
                mem_loc_list=mem_loc_list,
                is_ld=is_ld,
                data_width=data_width,
                num_addr=num_addr
            )
            # NOTE: consume 1 pipeline cycle
            yield self.env.timeout(1 * self.clock_unit)
            
            # issue request
            # NOTE: this is blocking
            if is_ld:
                yield self.env.process(
                    self._handle_ld(
                        offset_list=offset_list,
                        co_addr_list=co_addr_list,
                        data_width=data_width,
                        prt_id=prt_id,
                        num_addr=num_addr,
                        req=req
                    )
                )
            else:
                yield self.env.process(
                    self._handle_st(
                        offset_list=offset_list,
                        co_addr_list=co_addr_list,
                        data_width=data_width,
                        prt_id=prt_id,
                        num_addr=num_addr,
                        req=req
                    )
                )
            # update performance couter
            self.num_prt_read += 1
            self.num_prt_write += 1

    def _handle_ld(
        self, offset_list, co_addr_list, data_width, prt_id,
        num_addr, req
    ):
        # allocate a data buffer
        data_buffer = bytearray(data_width * num_addr)
        miss_dram_trans_list = {}
        for mem_addr in co_addr_list:
            dram_trans = co_addr_list[mem_addr]
            dram_trans.subcore_id = None
            dram_trans.is_nb = False
            dram_trans.prt_id = prt_id
            dram_trans.is_remote = True
            dram_trans.time = self.env.now
            # NOTE pg_id is part of dram_trans initialization
            # so we don't need to set here
            # issue to pg, upstream traffic
            if self.config["bypass_lsu_remote_dcache"] is False:
                is_hit, data = yield self.env.process(
                    self.dcache.read(mem_addr)
                )
                if is_hit:
                    assert data is not None
                    for i in range(num_addr):
                        valid = (dram_trans.simt_mask >> i) & 1
                        if valid:
                            trans_start_addr = offset_list[i]
                            trans_end_addr = trans_start_addr + data_width
                            assert trans_end_addr <= \
                                self.config["dram_bank_io_width"]
                            db_start_addr = i * data_width
                            db_end_addr = db_start_addr + data_width
                            data_buffer[db_start_addr: db_end_addr] = \
                                data[trans_start_addr: trans_end_addr]
                    continue
                assert is_hit is False
            miss_dram_trans_list[mem_addr] = dram_trans
            yield self.bus_arbiter\
                .upstream_req_queue.put(dram_trans)
        # wait until request return
        for mem_addr in miss_dram_trans_list:
            dram_trans = yield self.in_dram_trans_queue.get(
                lambda x: (
                    x.type == "load"
                    and x.global_mem_addr == mem_addr
                    and x.is_nb is False
                    and x.is_remote is True
                    and x.prt_id == prt_id
                )
            )
            assert dram_trans.data is not None
            assert dram_trans.simt_mask > 0
            if self.config["bypass_lsu_remote_dcache"] is False:
                # update cache content
                self.dcache.update(mem_addr, dram_trans.data)
            for i in range(num_addr):
                valid = (dram_trans.simt_mask >> i) & 1
                if valid:
                    trans_start_addr = offset_list[i]
                    trans_end_addr = trans_start_addr + data_width
                    assert trans_end_addr <= self.config["dram_bank_io_width"]
                    db_start_addr = i * data_width
                    db_end_addr = db_start_addr + data_width
                    data_buffer[db_start_addr: db_end_addr] = \
                        dram_trans.data[trans_start_addr: trans_end_addr]
            # NOTE: consume 1 pipeline cycle
            yield self.env.timeout(1 * self.clock_unit)
        """
        yield self.env.timeout(1 * self.clock_unit)
        addr_list = req.addr_list
        data_buffer = bytearray(0)
        for i in range(len(addr_list)):
            addr = addr_list[i]
            data = self.core.processor.hardware.mem\
                .get_value(addr, data_width)
            data_buffer.extend(deepcopy(data))
        """
        
        # prepare a response
        resp = DstRemoteLoadResp(
            addr_list=req.addr_list,
            data_width=req.data_width
        )
        # set data
        resp.data = deepcopy(data_buffer)
        
        """
        addr_list = req.addr_list
        ld_data = bytearray(0)
        for i in range(len(addr_list)):
            addr = addr_list[i]
            data = self.core.processor.hardware.mem\
                .get_value(addr, data_width)
            ld_data.extend(deepcopy(data))
        assert resp.data == ld_data
        """
        # send response
        yield self.remote_resp_queue.put(resp)

    def _handle_st(
        self, offset_list, co_addr_list, data_width, prt_id,
        num_addr, req
    ):
        for mem_addr in co_addr_list:
            dram_trans = co_addr_list[mem_addr]
            dram_trans.subcore_id = None
            dram_trans.is_nb = False
            dram_trans.prt_id = prt_id
            dram_trans.is_remote = True
            dram_trans.time = self.env.now
            # prepare data
            dram_trans.data = bytearray(self.config["dram_bank_io_width"])
            for i in range(num_addr):
                valid = (dram_trans.simt_mask >> i) & 1
                if valid:
                    trans_start_addr = offset_list[i]
                    trans_end_addr = trans_start_addr + data_width
                    assert trans_end_addr <= self.config["dram_bank_io_width"]
                    db_start_addr = i * data_width
                    db_end_addr = db_start_addr + data_width
                    dram_trans.data[trans_start_addr: trans_end_addr] = \
                        deepcopy(req.data[db_start_addr: db_end_addr])
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
                    and x.is_remote is True
                    and x.prt_id == prt_id
                )
            )
            # NOTE: consume 1 pipeline cycle
            yield self.env.timeout(1 * self.clock_unit)

        """
        yield self.env.timeout(1 * self.clock_unit)
        addr_list = req.addr_list
        data_buffer = req.data
        for i in range(len(addr_list)):
            db_start_addr = i * data_width
            db_end_addr = db_start_addr + data_width
            st_data = data_buffer[db_start_addr: db_end_addr]
            addr = addr_list[i]
            self.core.processor.hardware.mem\
                .set_value(addr, deepcopy(st_data))
        """
        # prepare a response
        resp = DstRemoteStoreResp(
            addr_list=req.addr_list,
            data_width=req.data_width
        )

        # send response
        yield self.remote_resp_queue.put(resp)
