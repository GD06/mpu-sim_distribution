import simpy

from simulator.network_message import NetworkPacket


class NetworkMessageBufferEntry:
    def __init__(self, msg, num_packet):
        self.msg = msg
        self.num_packet = num_packet
        self.num_recv_packet = 0


class NetworkRouter:

    def __init__(self, env, log, config, niu):
        self.env = env
        self.log = log
        self.config = config
        
        # parent network interface unit
        self.niu = niu

        # router parameters
        self.proc_id = niu.core.processor.proc_id
        self.core_id = niu.core.core_id
        assert config["sim_clock_freq"] % \
            config["network_router_clock_freq"] == 0
        self.clock_unit = config["sim_clock_freq"] \
            // config["network_router_clock_freq"]
        self.num_chan = self.config["network_router_num_channel"]
        self.in_chan_size = self.config["network_router_in_channel_size"]
        self.out_chan_size = self.config["network_router_out_channel_size"]
        self.crossbar_delay = self.config["network_router_crossbar_delay"]
        assert self.config["network_routing"] == "deterministic"
        assert self.config["network_flow_control"] == "buffered"
        assert self.config["network_topology"] == "2D_mesh"
        self.chan_idx_map = self.config["network_channel_index"]
        self._loc_str = self.niu.core._loc_str

        # buffer to collect and assemble packets
        # (src_proc_id, src_core_id, req_id, msg_id) 
        # --> NetworkMessageBufferEntry
        self.msg_buffer = {}
        # queues for input / output packet channels
        self.in_packet_chan = []
        self.out_packet_chan = []
        for i in range(self.num_chan):
            self.in_packet_chan\
                .append(simpy.Store(env, capacity=self.in_chan_size))
            self.out_packet_chan\
                .append(simpy.Store(env, capacity=self.out_chan_size))
        # spawn processes to handle channel packets
        for i in range(self.num_chan):
            self.env.process(
                self._process_in_chan_packet(i)
            )
        # packet queues for network interface unit
        self.in_packet_buffer = simpy.Store(env)
        self.out_packet_buffer = simpy.Store(env)
        # spawn processes to handle outbound / inbound packets
        # from network interface unit
        self.env.process(
            self._process_outbound_packet()
        )
        # spawn a process to assemble input packets into messages
        self.env.process(
            self._assemble_inbound_packet()
        )
        # NOTE: performance counter
        self.access_scaling_factor = self.config["network_packet_size"] \
            / self.config["network_flit_size"]
        self.in_chan_access = [0] * self.num_chan
        self.out_chan_access = [0] * self.num_chan
        self.traffic_cyc = {}
        for in_id in range(self.num_chan):
            for out_id in range(self.num_chan):
                self.traffic_cyc[(in_id, out_id)] = 0
        self.out_busy_cyc = [0] * self.num_chan

    def get_perf_metrics(self):
        """Get a dictionary of performance metrics"""
        perf_metrics = {}
        perf_metrics["in_chan_access"] = {}
        perf_metrics["out_chan_access"] = {}
        for chan_id in range(self.num_chan):
            perf_metrics["in_chan_access"][str(chan_id)] = \
                self.in_chan_access[chan_id]
            perf_metrics["out_chan_access"][str(chan_id)] = \
                self.out_chan_access[chan_id]
        perf_metrics["traffic_cyc"] = {}
        for in_id in range(self.num_chan):
            for out_id in range(self.num_chan):
                perf_metrics["traffic_cyc"][str((in_id, out_id))] = \
                    self.traffic_cyc[(in_id, out_id)]
        perf_metrics["out_busy_cyc"] = {}
        for out_id in range(self.num_chan):
            perf_metrics["out_busy_cyc"][str(out_id)] = \
                self.out_busy_cyc[out_id]

        # Collect the performance metrics of all hardware sub-module
        return {"network_router": perf_metrics}

    def _cal_routing_dst_xy_opt(self, packet):
        """
        Calculate routing destination
        Args:
            packet: input packet
        
        Returns:
            is_local: True if the routing destination is reached
            chan_idx: index of output channel (can be 2 choices)
        """
        def _route_x(dst_proc_id, dst_core_id):
            # route x-dim
            if dst_proc_id[0] == self.proc_id[0]:
                # route core in x-dim
                if dst_core_id[0] < self.core_id[0]:
                    return self.chan_idx_map["W"]
                elif dst_core_id[0] > self.core_id[0]:
                    return self.chan_idx_map["E"]
                else:
                    return None
            elif dst_proc_id[0] < self.proc_id[0]:
                return self.chan_idx_map["W"]
            elif dst_proc_id[0] > self.proc_id[0]:
                return self.chan_idx_map["E"]
        
        def _route_y(dst_proc_id, dst_core_id):
            # route y-dim
            if dst_proc_id[1] == self.proc_id[1]:
                # route core in y-dim
                if dst_core_id[1] < self.core_id[1]:
                    return self.chan_idx_map["S"]
                elif dst_core_id[1] > self.core_id[1]:
                    return self.chan_idx_map["N"]
                else:
                    return None
            elif dst_proc_id[1] < self.proc_id[1]:
                return self.chan_idx_map["S"]
            elif dst_proc_id[1] > self.proc_id[1]:
                return self.chan_idx_map["N"]
        
        is_local = False
        chan_idx = None
        # NOTE id tuple format:
        # prod_id_x = proc_id[0]
        # prod_id_y = proc_id[1]
        # core_id_x = core_id[0]
        # core_id_y = core_id[1]
        dst_proc_id = packet.dst_loc.proc_id
        dst_core_id = packet.dst_loc.core_id
        """
        print("======ROUTING")
        print("FROM:")
        print(packet.msg.src_loc_str)
        print("TO:")
        print(packet.msg.dst_loc_str)
        print("CURRENT:")
        print(self._loc_str)
        """
        if (
            dst_proc_id == self.proc_id
            and dst_core_id == self.core_id
        ):
            # destination reached
            is_local = True
        else:
            # still needs routing
            x = _route_x(dst_proc_id, dst_core_id)
            y = _route_y(dst_proc_id, dst_core_id)
            if None in {x, y}:
                chan_idx = x if y is None else y
                assert chan_idx is not None
            else:
                x_size = len(self.out_packet_chan[x].items)
                y_size = len(self.out_packet_chan[y].items)
                if x_size == y_size:
                    chan_idx = x
                elif x_size > y_size:
                    chan_idx = y
                elif x_size < y_size:
                    chan_idx = x
        return (is_local, chan_idx)

    def _cal_routing_dst_xy(self, packet):
        """
        Calculate routing destination
        Args:
            packet: input packet
        
        Returns:
            is_local: True if the routing destination is reached
            chan_idx: index of output channel
        """
        is_local = False
        chan_idx = None
        # NOTE id tuple format:
        # prod_id_x = proc_id[0]
        # prod_id_y = proc_id[1]
        # core_id_x = core_id[0]
        # core_id_y = core_id[1]
        dst_proc_id = packet.dst_loc.proc_id
        dst_core_id = packet.dst_loc.core_id
        """
        print("======ROUTING")
        print("FROM:")
        print(packet.msg.src_loc_str)
        print("TO:")
        print(packet.msg.dst_loc_str)
        print("CURRENT:")
        print(self._loc_str)
        """
        if (
            dst_proc_id == self.proc_id
            and dst_core_id == self.core_id
        ):
            # destination reached
            is_local = True
        else:
            # still needs routing
            if (
                dst_proc_id[0] == self.proc_id[0]
                and dst_core_id[0] == self.core_id[0]
            ):
                # x-dim is aligned, route y-dim
                if dst_proc_id[1] == self.proc_id[1]:
                    # same processor, route core in y-dim
                    if dst_core_id[1] < self.core_id[1]:
                        chan_idx = self.chan_idx_map["S"]
                    elif dst_core_id[1] > self.core_id[1]:
                        chan_idx = self.chan_idx_map["N"]
                    else:
                        assert False
                elif dst_proc_id[1] < self.proc_id[1]:
                    chan_idx = self.chan_idx_map["S"]
                elif dst_proc_id[1] > self.proc_id[1]:
                    chan_idx = self.chan_idx_map["N"]
            else:
                # route x-dim
                if dst_proc_id[0] == self.proc_id[0]:
                    # route core in x-dim
                    if dst_core_id[0] < self.core_id[0]:
                        chan_idx = self.chan_idx_map["W"]
                    elif dst_core_id[0] > self.core_id[0]:
                        chan_idx = self.chan_idx_map["E"]
                    else:
                        assert False
                elif dst_proc_id[0] < self.proc_id[0]:
                    chan_idx = self.chan_idx_map["W"]
                elif dst_proc_id[0] > self.proc_id[0]:
                    chan_idx = self.chan_idx_map["E"]
        return (is_local, chan_idx)
    
    def _process_in_chan_packet(self, in_chan_idx):
        while True:
            packet = yield self.in_packet_chan[in_chan_idx].get()
            assert isinstance(packet, NetworkPacket)
            # update performance counter
            self.in_chan_access[in_chan_idx] += 1 * self.access_scaling_factor
            # calculate output channel information
            is_local, out_chan_idx = self._cal_routing_dst_xy_opt(packet)
            self.env.process(
                self._send_to_out_chan(
                    is_local, in_chan_idx, out_chan_idx, packet
                )
            )
            # NOTE: can accept packet in the next cycle
            yield self.env.timeout(1 * self.clock_unit)

    def _send_to_out_chan(self, is_local, in_chan_idx, out_chan_idx, packet):
        if is_local:
            # NOTE: consume 1 pipeline cycle
            yield self.env.timeout(1 * self.clock_unit)
            yield self.in_packet_buffer.put(packet)
        else:
            # crossbar traversing
            yield self.env.timeout(self.crossbar_delay * self.clock_unit)
            # put into the dst channel buffer
            yield self.out_packet_chan[out_chan_idx].put(packet)
            # update performance counter
            self.traffic_cyc[(in_chan_idx, out_chan_idx)] += \
                self.crossbar_delay
            self.out_busy_cyc[out_chan_idx] += \
                self.crossbar_delay
            self.out_chan_access[out_chan_idx] += \
                1 * self.access_scaling_factor

    def _assemble_inbound_packet(self):
        while True:
            packet = yield self.in_packet_buffer.get()
            # get a unique id
            if packet.msg_type in {"ld_req", "st_req"}:
                proc_id = packet.src_loc.proc_id
                core_id = packet.src_loc.core_id
            elif packet.msg_type in {"ld_resp", "st_resp"}:
                proc_id = packet.dst_loc.proc_id
                core_id = packet.dst_loc.core_id
            else:
                raise NotImplementedError(
                    "Unknown network message type: {}"
                    .format(packet.msg_type)
                )
            req_id = packet.req_id
            msg_id = packet.msg_id
            msg_buf_id = (proc_id, core_id, req_id, msg_id)
            
            # allocate a new message buffer or merge into an existing one
            if msg_buf_id not in self.msg_buffer:
                msg_buffer_entry = NetworkMessageBufferEntry(
                    msg=packet.msg,
                    num_packet=packet.num_packet
                )
                msg_buffer_entry.num_recv_packet += 1
                self.msg_buffer[msg_buf_id] = msg_buffer_entry
            else:
                msg_buffer_entry = self.msg_buffer[msg_buf_id]
                msg_buffer_entry.num_recv_packet += 1
            
            if (
                msg_buffer_entry.num_recv_packet 
                    == msg_buffer_entry.num_packet
            ):
                # we have collected all packets of this message
                # extract msg
                msg = msg_buffer_entry.msg
                """
                if (
                    self.niu.core.core_id == (0,0)
                    and self.niu.core.processor.proc_id == (0,0)
                ):
                    print(self.niu.core._loc_str)
                    print("+++ASSEMBLE MSG: "
                        +msg.msg_type+" (network_router.py)"
                        +" req_id="+str(packet.req_id)
                        +" seq_id="+str(packet.seq_id)
                        +" msg_id="+str(packet.msg_id))
                    print(self.env.now)
                """
                if msg.msg_type in {"ld_req", "st_req"}:
                    yield self.niu.in_req_msg_queue.put(msg)
                elif msg.msg_type in {"ld_resp", "st_resp"}:
                    yield self.niu.in_resp_msg_queue.put(msg)
                else:
                    raise NotImplementedError(
                        "Unknown network message type: {}"
                        .format(msg.msg_type)
                    )
                # remove entry
                del self.msg_buffer[msg_buf_id]
            else:
                assert msg_buffer_entry.num_recv_packet \
                    < msg_buffer_entry.num_packet
            # NOTE: consume 1 pipeline cycle
            yield self.env.timeout(1 * self.clock_unit)

    def _process_outbound_packet(self):
        while True:
            packet = yield self.out_packet_buffer.get()
            assert isinstance(packet, NetworkPacket)
            # calculate output channel information
            is_local, chan_idx = self._cal_routing_dst_xy_opt(packet)
            assert is_local is False
            self.env.process(
                self._send_outbound_packet(
                    chan_idx, packet
                )
            )
            # NOTE: can accept packet in the next cycle
            yield self.env.timeout(1 * self.clock_unit)

    def _send_outbound_packet(self, chan_idx, packet):
        # NOTE: consume 1 pipeline cycle
        yield self.env.timeout(1 * self.clock_unit)
        # put into the dst channel buffer
        """
        if (
            self.niu.core.core_id == (0,0)
            and self.niu.core.processor.proc_id == (0,0)
        ):
            print(self.niu.core._loc_str)
            print("+++SEND PACKET (network_router.py)"
                +" req_id="+str(packet.req_id)
                +" seq_id="+str(packet.seq_id)
                +" msg_id="+str(packet.msg_id)
                +" ch_id="+str(chan_idx))
            print(self.env.now)
        """
        yield self.out_packet_chan[chan_idx].put(packet)
        # update performance counter
        self.out_chan_access[chan_idx] += 1 * self.access_scaling_factor
