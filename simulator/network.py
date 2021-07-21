import simpy
from copy import deepcopy

from simulator.network_message import NetworkLocationInfo, NetworkMessage, \
    SrcRemoteLoadReq, SrcRemoteStoreReq, SrcRemoteLoadResp, \
    SrcRemoteStoreResp, DstRemoteLoadReq, DstRemoteLoadResp, \
    DstRemoteStoreReq, DstRemoteStoreResp
from simulator.network_router import NetworkRouter
from simulator.readonly_dcache import ReadOnlyDataCache


class NetworkInterfaceUnit:
    
    def __init__(self, env, log, config, core):
        self.env = env
        self.log = log
        self.config = config
        self.core = core
        assert config["sim_clock_freq"] % \
            config["core_clock_freq"] == 0
        self.clock_unit = config["sim_clock_freq"] \
            // config["core_clock_freq"]

        self.dcache = ReadOnlyDataCache(
            name="niu",
            env=env,
            log=log,
            config=config,
            clock_unit=self.clock_unit,
            granularity=self.config["data_path_unit_size"]
        )

        self.router = NetworkRouter(
            env=self.env,
            log=self.log,
            config=self.config,
            niu=self
        )

        # core <-> niu
        self.req_queue_size = self.config["niu_req_queue_size"]
        self.req_queue = simpy.Store(env, capacity=self.req_queue_size)
        self.resp_queue = simpy.FilterStore(env)
        # niu <-> router
        self.in_req_msg_queue_size = \
            self.config["niu_in_req_msg_queue_size"]
        self.in_resp_msg_queue_size = \
            self.config["niu_in_resp_msg_queue_size"]
        self.in_req_msg_queue = simpy\
            .FilterStore(env, capacity=self.in_req_msg_queue_size)
        self.in_resp_msg_queue = simpy\
            .FilterStore(env, capacity=self.in_resp_msg_queue_size)
        # NOTE: out_msg_queue is not used currently
        # will be used when we implement router in the future
        self.out_msg_queue_size = self.config["niu_out_msg_queue_size"]
        self.out_msg_queue = simpy.Store(env, capacity=self.out_msg_queue_size)
        # reference the address hashing function in sim_api
        self.addr_hashing = self.core.processor.hardware.addr_hashing
        self.re_addr_hashing = self.core.processor.hardware.re_addr_hashing

        # spawn processes to handle request
        self.max_num_entry = \
            self.config["max_num_niu_track_req_table_entry"]
        for req_id in range(self.max_num_entry):
            self.env.process(self._handle_req(req_id))
        # spawn a process to handle request message
        self.env.process(self._handle_req_msg())

    def get_perf_metrics(self):
        """Get a dictionary of performance metrics"""
        perf_metrics = {}

        # Collect the performance metrics of all hardware sub-module
        router_metrics = self.router.get_perf_metrics()
        assert len(router_metrics) == 1
        perf_metrics.update(router_metrics)

        return {"network_interface_unit": perf_metrics}

    def _compose_msg(self, req, resp, is_ld):
        addr_list = req.addr_list
        data_width = req.data_width
        simt_mask = req.simt_mask
        msg_list = {}
        
        # compose messages
        # compute msg type
        msg_type = "ld_req" if is_ld else "st_req"
        assert msg_type in self.config["network_msg_type"]
        for i in range(self.config["num_threads_per_warp"]):
            valid = (simt_mask >> i) & 1
            if valid:
                # get address information
                addr = addr_list[i]
                if self.config["bypass_niu_dcache"] is False:
                    if is_ld:
                        # NOTE: check if this is a local hit
                        is_hit, data = yield self.env.process(
                            self.dcache.read(addr)
                        )
                        if is_hit:
                            assert data is not None
                            data_start_addr = i * data_width
                            data_end_addr = data_start_addr + data_width
                            resp.data[data_start_addr: data_end_addr] = \
                                deepcopy(data)
                            continue
                        # cache miss
                        assert is_hit is False
                proc_id_y, proc_id_x, core_id_y, core_id_x, \
                    pg_id, pe_id, bank_addr, bank_interface_offset = \
                    self.addr_hashing(addr)
                dst_proc_id = (proc_id_x, proc_id_y)
                dst_core_id = (core_id_x, core_id_y)
                dst_loc = NetworkLocationInfo(
                    proc_id=dst_proc_id,
                    core_id=dst_core_id
                )
                src_proc_id = self.core.processor.proc_id
                src_core_id = self.core.core_id
                src_loc = NetworkLocationInfo(
                    proc_id=src_proc_id,
                    core_id=src_core_id
                )
                # compose a network message or merge into an existing one
                msg = None
                if dst_proc_id in msg_list:
                    if dst_core_id in msg_list[dst_proc_id]:
                        # merge into an existing msg
                        msg = msg_list[dst_proc_id][dst_core_id]
                    else:
                        # create a new one
                        msg = NetworkMessage(
                            src_loc=src_loc,
                            dst_loc=dst_loc,
                            msg_type=msg_type,
                            msg_id=deepcopy(i)
                        )
                        msg_list[dst_proc_id][dst_core_id] = msg
                else:
                    msg_list[dst_proc_id] = {}
                    # create a new one
                    msg = NetworkMessage(
                        src_loc=src_loc,
                        dst_loc=dst_loc,
                        msg_type=msg_type,
                        msg_id=deepcopy(i)
                    )
                    msg_list[dst_proc_id][dst_core_id] = msg
                
                # update msg fields
                msg.simt_mask += 1 << i
                # NOTE: since we already know the dst core address
                # we use truncate address
                truncate_addr = self.re_addr_hashing(
                    (0, 0, 0, 0,
                     pg_id, pe_id, bank_addr, bank_interface_offset)
                )
                msg.addr_list.append(truncate_addr)
                msg.data_width = data_width
                if not is_ld:
                    start_addr = i * data_width
                    end_addr = start_addr + data_width
                    msg.data_buffer.extend(
                        deepcopy(
                            req.data[start_addr: end_addr]
                        )
                    )

        return msg_list

    def _update_load_resp(self, resp, msg):
        data_buffer = msg.data_buffer
        simt_mask = msg.simt_mask
        data_width = resp.data_width
        db_start_addr = 0
        for i in range(self.config["num_threads_per_warp"]):
            valid = (simt_mask >> i) & 1
            if valid:
                db_end_addr = db_start_addr + data_width
                data_start_addr = i * data_width
                data_end_addr = data_start_addr + data_width
                resp.data[data_start_addr: data_end_addr] = deepcopy(
                    data_buffer[db_start_addr: db_end_addr]
                )
                if self.config["bypass_niu_dcache"] is False:
                    # update cache content
                    self.dcache.update(
                        resp.addr_list[i], 
                        data_buffer[db_start_addr: db_end_addr]
                    )
                db_start_addr += data_width

    def _handle_resp_msg_send_resp(self, is_ld, req, resp, req_id, msg_list):
        # we still have outbound messages to receive
        event_list = []
        for dst_proc_id in msg_list:
            for dst_core_id in msg_list[dst_proc_id]:
                out_msg = msg_list[dst_proc_id][dst_core_id]
                event_list.append(
                    self.env.process(
                        self._recv_msg_update_resp(
                            is_ld=is_ld,
                            req_id=req_id,
                            out_msg=out_msg,
                            resp=resp
                        )
                    )
                )
        if len(event_list) > 0:
            yield simpy.events.AllOf(self.env, event_list)
        # we receive all msg related to this request
        """
        if is_ld:
            for i in range(32):
                v = (req.simt_mask >> i) & 1
                if v:
                    addr = req.addr_list[i]
                    db = self.core.processor.hardware.mem\
                        .get_value(addr, 4)
                    s = i * 4
                    e = s + 4
                    assert db == resp.data[s:e]
        """
        yield self.resp_queue.put(resp)

    def _recv_msg_update_resp(self, is_ld, req_id, out_msg, resp):
        msg_type = "ld_resp" if is_ld else "st_resp"
        in_msg = yield self.in_resp_msg_queue.get(
            lambda x: (
                isinstance(x, NetworkMessage)
                and x.req_id == req_id
                and x.msg_type == msg_type
                and x.src_loc.is_equal(out_msg.dst_loc)
                and x.dst_loc.is_equal(out_msg.src_loc)
            )
        )
        in_msg.decode_data()
        in_msg.src_rcv_cyc = self.env.now
        """
        if out_msg.tracing:
            print("====RECEIVE MSG: (req_id={}), msg_id={} CLK={}"
                .format(in_msg.req_id, in_msg.msg_id, self.env.now))
            print(in_msg.src_loc_str)
            print(in_msg.dst_loc_str)
            print("src routing cyc: "
                +str(in_msg.dst_rcv_cyc - in_msg.src_issue_cyc))
            print("dst cyc: "
                +str(in_msg.dst_issue_cyc - in_msg.dst_rcv_cyc))
            print("dst routing cyc: "
                +str(in_msg.src_rcv_cyc - in_msg.dst_issue_cyc))
        """
        if is_ld:
            """
            for i in range(len(in_msg.addr_list)):
                addr = in_msg.addr_list[i]
                db = self.core.processor.hardware.mem\
                    .get_value(addr, 4)
                s = i * 4
                e = s + 4
                assert db == in_msg.data_buffer[s:e]
            """
            self._update_load_resp(resp, in_msg)
    
    def _handle_req(self, req_id):
        while True:
            req = yield self.req_queue.get()
            if isinstance(req, SrcRemoteLoadReq):
                is_ld = True
            elif isinstance(req, SrcRemoteStoreReq):
                is_ld = False
            else:
                raise NotImplementedError(
                    "Unknown request type:{}"
                    .format(type(req))
                )
            # prepare a response
            if is_ld:
                # compose a load response
                resp = SrcRemoteLoadResp(
                    addr_list=req.addr_list,
                    data_width=req.data_width,
                    simt_mask=req.simt_mask
                )
                resp.prt_id = req.prt_id
                data_width = req.data_width
                resp.data = bytearray(
                    data_width * self.config["num_threads_per_warp"]
                )
            else:
                # compose a store response
                resp = SrcRemoteStoreResp(
                    addr_list=req.addr_list,
                    data_width=req.data_width,
                    simt_mask=req.simt_mask
                )
                resp.prt_id = req.prt_id
            # compose a number of network messages
            # NOTE: we filter messages that have local hits
            msg_list = yield self.env.process(
                self._compose_msg(req, resp, is_ld)
            )
            # NOTE: consume 1 pipeline cycle
            yield self.env.timeout(1 * self.clock_unit)
            # send the messages
            for prod_id in msg_list:
                for core_id in msg_list[prod_id]:
                    msg = msg_list[prod_id][core_id]
                    msg.req_id = req_id
                    msg.encode_data()
                    if self.config["bypass_router"]:
                        dst_core = self.core.processor.hardware\
                            .processor_array[msg.dst_loc.proc_id]\
                            .core_array[msg.dst_loc.core_id]
                        yield dst_core.niu.in_req_msg_queue.put(msg)
                    else:
                        # decompose into packets
                        packet_list = msg.decompose_to_packet(
                            self.config["network_packet_size"]
                        )
                        for packet in packet_list:
                            yield self.router.out_packet_buffer.put(packet)
                            # NOTE: consume 1 pipeline cycle
                            yield self.env.timeout(1 * self.clock_unit)
                    msg.src_issue_cyc = self.env.now
                    """
                    msg.tracing = req.tracing
                    if msg.tracing:
                        print("======SEND MSG (req_id={}), msg_id={} CLK={}"
                            .format(req_id, msg.msg_id, self.env.now))
                    """
            # spawn a process to handle the response
            # NOTE: this is blocking
            yield self.env.process(
                self._handle_resp_msg_send_resp(
                    is_ld=is_ld,
                    req=req,
                    resp=resp,
                    req_id=req_id,
                    msg_list=msg_list
                )
            )

    def _recover_full_addr(self, msg):
        for i in range(len(msg.addr_list)):
            truncate_addr = msg.addr_list[i]
            _, _, _, _, pg_id, pe_id, bank_addr, bank_interface_offset = \
                self.addr_hashing(truncate_addr)
            proc_id_x = self.core.processor.proc_id[0]
            proc_id_y = self.core.processor.proc_id[1]
            core_id_x = self.core.core_id[0]
            core_id_y = self.core.core_id[1]
            full_addr = self.re_addr_hashing(
                (proc_id_y, proc_id_x, core_id_y, core_id_x,
                 pg_id, pe_id, bank_addr, bank_interface_offset)
            )
            msg.addr_list[i] = full_addr

    def _handle_req_msg(self):
        while True:
            in_msg = yield self.in_req_msg_queue.get(
                lambda x: (
                    x.msg_type == "ld_req"
                    or x.msg_type == "st_req"
                )
            )
            assert isinstance(in_msg, NetworkMessage)
            in_msg.decode_data()
            self._recover_full_addr(in_msg)
            in_msg.dst_rcv_cyc = self.env.now
            # NOTE: currently we only support 4 byte data width
            assert in_msg.data_width == 4
            is_ld = True if in_msg.msg_type == "ld_req" else False
            # request message
            if is_ld:
                # compose a load request
                req = DstRemoteLoadReq(
                    addr_list=in_msg.addr_list,
                    data_width=in_msg.data_width
                )
            else:
                # compose a store request
                assert in_msg.data_buffer is not None
                assert in_msg.data_width * len(in_msg.addr_list) \
                    == len(in_msg.data_buffer)
                req = DstRemoteStoreReq(
                    addr_list=in_msg.addr_list,
                    data_width=in_msg.data_width,
                    data=in_msg.data_buffer
                )
            # send request
            yield self.core.lsu_remote.remote_req_queue.put(req)
            # spawn a process to handle response
            # NOTE: this is non-blocking
            self.env.process(
                self._handle_resp_msg(
                    in_msg=in_msg, 
                    req=req,
                    is_ld=is_ld
                )
            )
            # NOTE: can accept request msg in the next cycle
            yield self.env.timeout(1 * self.clock_unit)

    def _handle_resp_msg(self, in_msg, req, is_ld):
        # get response
        if is_ld:
            resp = yield self.core.lsu_remote.remote_resp_queue.get(
                lambda x: (
                    isinstance(x, DstRemoteLoadResp)
                    and x.addr_list == req.addr_list
                    and x.data_width == req.data_width
                )
            )
        else:
            resp = yield self.core.lsu_remote.remote_resp_queue.get(
                lambda x: (
                    isinstance(x, DstRemoteStoreResp)
                    and x.addr_list == req.addr_list
                    and x.data_width == req.data_width
                )
            )
        # compose a response message
        msg_type = "ld_resp" if is_ld else "st_resp"
        assert msg_type in self.config["network_msg_type"]
        out_msg = NetworkMessage(
            src_loc=deepcopy(in_msg.dst_loc),
            dst_loc=deepcopy(in_msg.src_loc),
            msg_type=msg_type,
            msg_id=in_msg.msg_id
        )
        # update message fields
        out_msg.req_id = in_msg.req_id
        out_msg.data_width = in_msg.data_width
        out_msg.simt_mask = in_msg.simt_mask
        # update tracing info
        out_msg.src_issue_cyc = in_msg.src_issue_cyc
        out_msg.dst_rcv_cyc = in_msg.dst_rcv_cyc
        if is_ld:
            out_msg.data_buffer = deepcopy(resp.data)
            """
            out_msg.addr_list = in_msg.addr_list
            for i in range(len(resp.addr_list)):
                addr = resp.addr_list[i]
                db = self.core.processor.hardware.mem\
                    .get_value(addr, 4)
                s = i * 4
                e = s + 4
                assert db == out_msg.data_buffer[s:e]
            """
        out_msg.encode_data()
        # NOTE: consume 1 pipeline cycle
        yield self.env.timeout(1 * self.clock_unit)
        # send msg
        if self.config["bypass_router"]:
            dst_core = self.core.processor.hardware\
                .processor_array[out_msg.dst_loc.proc_id]\
                .core_array[out_msg.dst_loc.core_id]
            yield dst_core.niu.in_resp_msg_queue.put(out_msg)
        else:
            # decompose into packets
            packet_list = out_msg.decompose_to_packet(
                self.config["network_packet_size"]
            )
            for packet in packet_list:
                yield self.router.out_packet_buffer.put(packet)
        out_msg.dst_issue_cyc = self.env.now
