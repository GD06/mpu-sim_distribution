class NetworkEnergy:

    def __init__(self, config, pt, hw):
        self.config = config
        self.pt = pt
        self.hw = hw
        self.energy_item = {}
        # key: bus_id
        self.energy_item["net_bus_energy"] = {}
        # key: router_id
        self.energy_item["net_buffer_enenrgy"] = {}
        self.energy_item["net_switch_ctrl_energy"] = {}
        self.energy_item["net_switch_xbar_energy"] = {}
        self.energy_item["net_switch_output_energy"] = {}
        self.energy_item["net_switch_output_ctrl_energy"] = {}
        self.energy_item["net_switch_output_clk_energy"] = {}
        self._config_network_param()

    def _config_network_param(self):
        self.router_in_chan = self.config["network_router_num_channel"]
        self.router_out_chan = self.config["network_router_num_channel"]
        # switch channel width in bits
        self.xbar_chan_width = self.config["network_flit_size"] * 8
        # mm
        self.metal_pitch = self.config["network_metal_pitch"]
        self.chan_pitch = self.metal_pitch * 2
        self.xbar_pitch = self.metal_pitch * 2
        # F/mm
        self.cw = self.config["network_cw_cpl"] * 2 \
            + self.config["network_cw_gnd"] * 2
        self.c_i_delay = (1.0 + 2.0) \
            * (self.config["network_cg"] + self.config["network_cgdl"])
        self.c_o_delay = (1.0 + 2.0) \
            * self.config["network_cd"]
        self.c_i = (1.0 + 2.0) * self.config["network_cg_pwr"]
        self.c_o = (1.0 + 2.0) * self.config["network_cd_pwr"]
        # V
        self.net_vdd = self.config["network_vdd"]

    def _update_net_bus_energy(self, net_bus_tr, bus_id):
        bus_bit_energy = \
            self.config["network_onchip_bus_energy_per_bit"] \
            if net_bus_tr["is_onchip"] \
            else self.config["network_offchip_bus_energy_per_bit"]
        bus_width = self.config["network_onchip_bus_width"] * 8\
            if net_bus_tr["is_onchip"] \
            else self.config["network_offchip_bus_width"] * 8
        bus_width_energy = bus_bit_energy * bus_width
        self.energy_item["net_bus_energy"][bus_id] = \
            (net_bus_tr["upstream_cyc"] + net_bus_tr["downstream_cyc"]) \
            * bus_width_energy
        self.hw.energy_item["net_bus_energy"] += \
            (net_bus_tr["upstream_cyc"] + net_bus_tr["downstream_cyc"]) \
            * bus_width_energy

    def _update_net_buffer_enenrgy(self, net_router_tr, router_id):
        tot_access = 0
        for i in range(self.router_in_chan):
            tot_access += net_router_tr["in_chan_access"][str(i)]
        for i in range(self.router_out_chan):
            tot_access += net_router_tr["out_chan_access"][str(i)]
        self.energy_item["net_buffer_enenrgy"][router_id] = tot_access \
            * self.config["network_chan_buf_access_energy"]
        self.hw.energy_item["net_buffer_energy"] += tot_access \
            * self.config["network_chan_buf_access_energy"]

    def _cal_net_wire_dff_power(self, M, W, alpha):
        c_din = 2 * 0.8 * (self.c_i + self.c_o) \
            + 2 * (2.0 / 3.0 * 0.8 * self.c_o)
        c_clk = 2 * 0.8 * (self.c_i + self.c_o) \
            + 2 * (2.0 / 3.0 * 0.8 * self.config["network_cg_pwr"])
        c_int = (alpha * 0.5) * c_din + alpha * c_clk
        # NOTE: The unit for power is W
        power = c_int * M * W * (self.net_vdd ** 2) \
            * self.config["network_router_clock_freq"] * (10 ** 6)
        return power

    def _cal_net_router_xbar_power(
        self, xbar_chan_width, num_in_chan, num_out_chan,
        in_id, out_id
    ):
        # datapath traversal power
        xbar_w = xbar_chan_width * num_out_chan * self.xbar_pitch
        xbar_h = xbar_chan_width * num_in_chan * self.xbar_pitch
        # wires
        c_w_in = xbar_w * self.cw
        c_w_out = xbar_h * self.cw
        # cross-points
        c_x_i = (1.0 / 16.0) * c_w_out
        c_x_o = 4.0 * c_x_i * (self.c_o_delay / self.c_i_delay)
        # drivers
        c_t_i = (1.0 / 16.0) * c_w_in
        c_t_o = 4.0 * c_t_i * (self.c_o_delay / self.c_i_delay)
        c_in_driver = 5.0 / 16.0 * (1 + self.c_o_delay / self.c_i_delay) \
            * (0.5 * self.cw * xbar_w + c_t_i)
        # total switched capacitance
        c_in = c_in_driver + c_w_in + c_t_i + (num_out_chan * c_x_i)
        if out_id < num_out_chan / 2:
            c_in -= (
                0.5 * c_w_in
                + num_out_chan / 2 * c_x_i
            )
        c_out = c_w_out + c_t_o + (num_in_chan * c_x_o)
        if in_id < num_in_chan / 2:
            c_out -= (
                0.5 * c_w_out
                + num_in_chan / 2 * c_x_o
            )

        # NOTE: The unit for power is W
        power = 0.5 * (c_in + c_out) * (self.net_vdd ** 2) \
            * self.config["network_router_clock_freq"] * (10 ** 6)
        return power
    
    def _cal_net_wire_clk_power(self, M, W):
        # number of clock wires running down one repeater bank
        columns = self.config["H_DFQD1"] * self.metal_pitch \
            / self.chan_pitch
        # length of clock wire
        clock_length = W * self.chan_pitch
        c_clk = (1 + 5.0 / 16.0 * (1 + self.c_o_delay / self.c_i_delay)) \
            * (clock_length * self.cw * columns + W * self.c_i_delay)
        
        # NOTE: The unit for power is W
        return M * c_clk * (self.net_vdd ** 2) \
            * self.config["network_router_clock_freq"] * (10 ** 6)

    def _cal_net_output_ctrl_power(self, W):
        w_out_mod = W * self.chan_pitch
        c_en = self.c_i
        c_enbale = (1 + 5.0 / 16.0) * (1.0 + self.c_o / self.c_i) \
            * (w_out_mod * self.cw + W * c_en)
        # NOTE: The unit for power is W
        return c_enbale * (self.net_vdd ** 2) \
            * self.config["network_router_clock_freq"] * (10 ** 6)

    def _cal_net_router_xbar_ctrl_power(
        self, xbar_chan_width, num_in_chan, num_out_chan
    ):
        # datapath traversal power
        xbar_w = xbar_chan_width * num_out_chan * self.xbar_pitch
        xbar_h = xbar_chan_width * num_in_chan * self.xbar_pitch
        # wires
        c_w_in = xbar_w * self.cw
        # drivers
        c_t_i = (5.0 / 16.0) * c_w_in
        c_ctrl = xbar_chan_width * c_t_i + (xbar_w + xbar_h) * self.cw
        c_drive = (5.0 / 16.0) * (1 + self.c_o_delay / self.c_i_delay) \
            * c_ctrl
        
        # NOTE: The unit for power is W
        return (c_ctrl + c_drive) * (self.net_vdd ** 2) \
            * self.config["network_router_clock_freq"] * (10 ** 6)

    def _update_net_switch_energy(self, net_router_tr, router_id):
        self.energy_item["net_switch_xbar_energy"][router_id] = 0
        self.energy_item["net_switch_ctrl_energy"][router_id] = 0
        self.energy_item["net_switch_output_clk_energy"][router_id] = 0
        self.energy_item["net_switch_output_energy"][router_id] = 0
        self.energy_item["net_switch_output_ctrl_energy"][router_id] = 0

        for out_id in range(self.router_out_chan):
            for in_id in range(self.router_in_chan):
                # crossbar energy
                pw = self._cal_net_router_xbar_power(
                    xbar_chan_width=self.xbar_chan_width,
                    num_in_chan=self.router_in_chan,
                    num_out_chan=self.router_out_chan,
                    in_id=in_id,
                    out_id=out_id
                )
                xbar_pw = pw * self.xbar_chan_width
                # NOTE: energy unit: nJ
                tmp_energy = xbar_pw \
                    * net_router_tr["traffic_cyc"][str((in_id, out_id))] \
                    / self.config["sim_clock_freq"] * 1000
                self.energy_item["net_switch_xbar_energy"][router_id] += \
                    tmp_energy
                self.hw.energy_item["net_switch_xbar_energy"] += tmp_energy
                # crossbar control energy
                # NOTE: energy unit: nJ
                tmp_energy = self._cal_net_router_xbar_ctrl_power(
                    xbar_chan_width=self.xbar_chan_width,
                    num_in_chan=self.router_in_chan,
                    num_out_chan=self.router_out_chan) \
                    * net_router_tr["traffic_cyc"][str((in_id, out_id))] \
                    / self.config["sim_clock_freq"] * 1000
                self.energy_item["net_switch_ctrl_energy"][router_id] += \
                    tmp_energy
                self.hw.energy_item["net_switch_ctrl_energy"] += tmp_energy
            # output clock energy
            # NOTE: energy unit: nJ
            tmp_energy = self._cal_net_wire_clk_power(
                M=1, W=self.xbar_chan_width) \
                * net_router_tr["out_busy_cyc"][str(out_id)] \
                / self.config["sim_clock_freq"] * 1000
            self.energy_item["net_switch_output_clk_energy"][router_id] += \
                tmp_energy
            self.hw.energy_item["net_switch_output_clk_energy"] += tmp_energy
            # output wire energy
            # NOTE: energy unit is nJ
            tmp_energy = self._cal_net_wire_dff_power(
                M=1, W=self.xbar_chan_width, alpha=1.0) \
                * net_router_tr["out_busy_cyc"][str(out_id)] \
                / self.config["sim_clock_freq"] * 1000
            self.energy_item["net_switch_output_energy"][router_id] += \
                tmp_energy
            self.hw.energy_item["net_switch_output_energy"] += tmp_energy
            # output control energy
            # NOTE: energy unit is nJ
            tmp_energy = \
                self._cal_net_output_ctrl_power(W=self.xbar_chan_width) \
                * net_router_tr["out_busy_cyc"][str(out_id)] \
                / self.config["sim_clock_freq"] * 1000
            self.energy_item["net_switch_output_ctrl_energy"][router_id] += \
                tmp_energy
            self.hw.energy_item["net_switch_output_ctrl_energy"] += tmp_energy

    def get_energy_metrics(self):
        for bus_id in self.pt["hardware"]["network_bus"]:
            self._update_net_bus_energy(
                self.pt["hardware"]["network_bus"][bus_id],
                bus_id
            )
        for proc_x in range(self.config["num_processor_x"]):
            for proc_y in range(self.config["num_processor_y"]):
                proc_tr = self.pt["hardware"][
                    "proc_{}".format((proc_x, proc_y))
                ]
                for core_x in range(self.config["num_core_x"]):
                    for core_y in range(self.config["num_core_y"]):
                        core_tr = proc_tr[
                            "core_{}".format((core_x, core_y))
                        ]
                        net_router_tr = \
                            core_tr["network_interface_unit"]["network_router"]
                        router_id = "px{}-py{}-cx{}-cy{}"\
                            .format(proc_x, proc_y, core_x, core_y)
                        self._update_net_buffer_enenrgy(
                            net_router_tr,
                            router_id
                        )
                        self._update_net_switch_energy(
                            net_router_tr,
                            router_id
                        )
        return {"network": self.energy_item}
