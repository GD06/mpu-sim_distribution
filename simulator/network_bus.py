class NetworkBus:
    
    def __init__(
        self, env, log, config, is_onchip, 
        uplink_router, downlink_router, bus_id
    ):
        """Bidirection link bus
        """
        self.env = env
        self.log = log
        self.config = config
        self.is_onchip = is_onchip
        self.uplink_router = uplink_router
        self.downlink_router = downlink_router
        # bus_id formatted as a tuple:
        # (
        #   (downlink_p_id_x, downlink_p_id_y), 
        #   (downlink_c_id_x, downlink_c_id_y),
        #   (uplink_p_id_x, uplink_p_id_y), 
        #   (uplink_c_id_x, uplink_c_id_y)
        # )
        self.bus_id = bus_id
        if is_onchip:
            assert config["sim_clock_freq"] % \
                config["network_onchip_bus_clock_freq"] == 0
            self.clock_unit = config["sim_clock_freq"] \
                // config["network_onchip_bus_clock_freq"]
            self.packet_delay = \
                self.config["network_onchip_bus_packet_delay"]
        else:
            assert config["sim_clock_freq"] % \
                config["network_offchip_bus_clock_freq"] == 0
            self.clock_unit = config["sim_clock_freq"] \
                // config["network_offchip_bus_clock_freq"]
            self.packet_delay = \
                self.config["network_offchip_bus_packet_delay"]
        # spawn a process to process packet
        self.env.process(self._process_uplink_packet())
        self.env.process(self._process_downlink_packet())
        # NOTE: performance counter
        self.upstream_cyc = 0
        self.downstream_cyc = 0

    def config_queue(
        self, uplink_in_chan, uplink_out_chan,
        downlink_in_chan, downlink_out_chan
    ):
        self.uplink_bus_input_queue = uplink_out_chan
        self.uplink_bus_output_queue = uplink_in_chan
        self.downlink_bus_input_queue = downlink_out_chan
        self.downlink_bus_output_queue = downlink_in_chan

    def get_perf_metrics(self):
        """Get a dictionary of performance metrics"""
        perf_metrics = {}
        perf_metrics["is_onchip"] = self.is_onchip
        perf_metrics["upstream_cyc"] = self.upstream_cyc
        perf_metrics["downstream_cyc"] = self.downstream_cyc
        return {
            "down-px{}-py{}-cx{}-cy{}-up-px{}-py{}-cx{}-cy{}"
            .format(
                self.bus_id[0][0], self.bus_id[0][1],
                self.bus_id[1][0], self.bus_id[1][1],
                self.bus_id[2][0], self.bus_id[2][1],
                self.bus_id[3][0], self.bus_id[3][1],
            ):
            perf_metrics
        }

    def _process_uplink_packet(self):
        while True:
            packet = yield self.uplink_bus_input_queue.get()
            """
            print("======UPLINK PACKET:"
                +" req_id="+str(packet.req_id)
                +" seq_id="+str(packet.seq_id)
                +" msg_id="+str(packet.msg_id))
            print(packet.msg.src_loc_str)
            print(packet.msg.dst_loc_str)
            print("==FROM")
            print(self.uplink_router._loc_str)
            print("==TO")
            print(self.downlink_router._loc_str)
            """
            self.upstream_cyc += self.packet_delay
            yield self.env.timeout(self.packet_delay * self.clock_unit)
            yield self.downlink_bus_output_queue.put(packet)

    def _process_downlink_packet(self):
        while True:
            packet = yield self.downlink_bus_input_queue.get()
            """
            print("======DOWNLINK PACKET:"
                +" req_id="+str(packet.req_id)
                +" seq_id="+str(packet.seq_id)
                +" msg_id="+str(packet.msg_id))
            print(packet.msg.src_loc_str)
            print(packet.msg.dst_loc_str)
            print("==FROM")
            print(self.downlink_router._loc_str)
            print("==TO")
            print(self.uplink_router._loc_str)
            """
            self.downstream_cyc += self.packet_delay
            yield self.env.timeout(self.packet_delay * self.clock_unit)
            yield self.uplink_bus_output_queue.put(packet)
