#!/usr/bin/env python3

import argparse
import json

import config.config_api as config_api
from simulator.energy.network_energy import NetworkEnergy
from simulator.energy.processor_energy import ProcessorEnergy


class HardwareEnergy:

    def __init__(self, config, performance_trace):
        """Evaluate energy consumption for a given performance trace
        Args:
            config: hardware configuration file
            pt: performance trace
        Return:
        """
        self.config = config
        self.pt = performance_trace

        # initialize energy items for this module
        self.energy_item = {}
        # pipeline: fetch and decode energy
        self.energy_item["icache_energy"] = 0
        self.energy_item["fetch_decode_energy"] = 0
        # pipeline: issue energy
        self.energy_item["scoreboard_energy"] = 0
        self.energy_item["reg_track_table_energy"] = 0
        # pipeline: execution energy
        self.energy_item["lsu_energy"] = 0
        self.energy_item["lsu_remote_energy"] = 0
        self.energy_item["lsu_extension_energy"] = 0
        self.energy_item["fb_alu_energy"] = 0
        self.energy_item["nb_alu_energy"] = 0
        self.energy_item["sfu_energy"] = 0
        # operand collector
        self.energy_item["fb_opc_energy"] = 0
        self.energy_item["nb_opc_energy"] = 0
        # register file energy
        self.energy_item["fb_rf_energy"] = 0
        self.energy_item["nb_rf_energy"] = 0
        # shared memory energy
        self.energy_item["smem_energy"] = 0
        # dram energy
        self.energy_item["dram_read_energy"] = 0
        self.energy_item["dram_write_energy"] = 0
        self.energy_item["dram_act_energy"] = 0
        self.energy_item["dram_pre_energy"] = 0
        self.energy_item["dram_refresh_energy"] = 0
        # TSV energy
        self.energy_item["subcore_pg_bus_energy"] = 0
        # network energy
        self.energy_item["net_bus_energy"] = 0
        self.energy_item["net_buffer_energy"] = 0
        self.energy_item["net_switch_ctrl_energy"] = 0
        self.energy_item["net_switch_xbar_energy"] = 0
        self.energy_item["net_switch_output_energy"] = 0
        self.energy_item["net_switch_output_ctrl_energy"] = 0
        self.energy_item["net_switch_output_clk_energy"] = 0
        # pipeline: writeback and commit
        self.energy_item["wb_commit_energy"] = 0

        # sub-modules
        self.network = NetworkEnergy(
            config=config,
            pt=performance_trace,
            hw=self
        )
        self.processor_array = {}
        for i in range(config["num_processor_x"]):
            for j in range(config["num_processor_y"]):
                proc_tr = self.pt["hardware"][
                    "proc_{}".format((i, j))
                ]
                proc = ProcessorEnergy(
                    config=config,
                    pt=proc_tr,
                    hw=self,
                    proc_id=(i, j)
                )
                self.processor_array[(i, j)] = proc

    def _cal_fetch_decode_energy(self):
        pass

    def _cal_issue_energy(self):
        pass

    def _cal_execute_energy(self):
        pass

    def _cal_dram_enenrgy(self):
        pass

    def _cal_reg_file_energy(self):
        pass

    def get_energy_metrics(self):
        """Gets a dictionary of energy metrics. """
        # Collect the energy metrics of this hardware module.
        energy_metrics = self.energy_item
        # evaluate network energy
        net_metrics = self.network.get_energy_metrics()
        assert len(net_metrics) == 1
        energy_metrics.update(net_metrics)
        # evaluate processor energy
        for i in range(self.config["num_processor_x"]):
            for j in range(self.config["num_processor_y"]):
                proc_metrics = self.processor_array[(i, j)]\
                    .get_energy_metrics()
                assert len(proc_metrics) == 1
                energy_metrics.update(proc_metrics)
        
        tot_energy = 0
        for key in energy_metrics.keys():
            if "_energy" in key:
                tot_energy += energy_metrics[key]
        energy_metrics["total_energy"] = tot_energy
        # total simulation time in ns
        tot_time = self.pt["hardware"]["total_num_cycles"] \
            / self.config["sim_clock_freq"] * 1000
        power_item = {}
        for key in self.energy_item.keys():
            if "_energy" in key:
                new_key = key[0:-6] + "power"
                power_item[new_key] = \
                    self.energy_item[key] / float(tot_time)
        energy_metrics.update(power_item)
        return {"hardware": energy_metrics}


def main():

    parser = argparse.ArgumentParser(
        description="Perform energy evaluation for the given traces",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    default_perf_trace_file = "../../util/trace/perf/tmp_perf.json"
    default_energy_trace_file = "../../util/trace/energy/tmp_energy.json"

    # Hardware dependent parameters
    parser.add_argument("--hardware_config", "-c", default=None,
                        help="Specify the path of hardware config")
    parser.add_argument("--performance_trace", "-p", 
                        default=default_perf_trace_file,
                        help="Specify the input performance trace file")
    parser.add_argument("--energy_trace", "-e", 
                        default=default_energy_trace_file,
                        help="Specify the output energy trace file")
    
    args = parser.parse_args()

    with open(args.performance_trace, "r") as f:
        performance_trace = json.load(f)

    # Load hardware configuration from file:
    hw_config_dict = config_api.load_hardware_config(
        overwrite_config_file_path=args.hardware_config
    )

    hw_energy = HardwareEnergy(
        config=hw_config_dict, 
        performance_trace=performance_trace
    )

    energy_metrics = hw_energy.get_energy_metrics()
    with open(args.energy_trace, "w") as f:
        json.dump(energy_metrics, f, indent=2)


if __name__ == "__main__":
    main()
