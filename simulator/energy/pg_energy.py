#!/usr/bin/env python3

from simulator.energy.subcore_energy import get_instr_energy


class ProcessingGroupEnergy:

    def __init__(self, config, pt, hw, pg_id):
        self.config = config
        self.pt = pt
        self.hw = hw
        self.pg_id = pg_id

        # initialize energy items for this module
        self.energy_item = {}
        # load-store unit extension energy
        self.energy_item["lsu_extension_energy"] = 0
        # pipeline: execution energy
        self.energy_item["nb_alu_energy"] = 0
        # operand collector
        self.energy_item["nb_opc_energy"] = 0
        # register file energy
        self.energy_item["nb_rf_energy"] = 0
        # dram energy
        self.energy_item["dram_read_energy"] = 0
        self.energy_item["dram_write_energy"] = 0
        self.energy_item["dram_act_energy"] = 0
        self.energy_item["dram_pre_energy"] = 0
        self.energy_item["dram_refresh_energy"] = 0

    def _update_nb_operand_collector_energy(self):
        tr = self.pt["nb_opc"]
        self.energy_item["nb_opc_energy"] = \
            tr["num_read"] * self.config["opc_read_energy"] \
            + tr["num_write"] * self.config["opc_write_energy"]
        self.hw.energy_item["nb_opc_energy"] += \
            self.energy_item["nb_opc_energy"]

    def _update_nb_alu_energy(self):
        tr = self.pt["nb_alu"]
        for opcode in tr.keys():
            num_op = tr[opcode]
            instr_energy = get_instr_energy(opcode, self.config)
            self.energy_item["nb_alu_energy"] += \
                num_op * instr_energy
            self.hw.energy_item["nb_alu_energy"] += \
                num_op * instr_energy

    def _update_lsu_extension_energy(self):
        tr = self.pt["lsu_extension"]
        self.energy_item["lsu_extension_energy"] = \
            tr["num_prt_read"] \
            * self.config["lsu_extension_prt_read_energy"] \
            + tr["num_prt_write"] \
            * self.config["lsu_extension_prt_write_energy"]
        self.hw.energy_item["lsu_extension_energy"] += \
            self.energy_item["lsu_extension_energy"]

    def _update_nb_reg_file_energy(self):
        tr = self.pt["nb_reg_file"] 
        self.energy_item["nb_rf_energy"] = \
            tr["num_read"] * self.config["pg_reg_file_bank_read_energy"] \
            + tr["num_write"] * self.config["pg_reg_file_bank_write_energy"]
        self.hw.energy_item["nb_rf_energy"] += \
            self.energy_item["nb_rf_energy"]

    def _update_dram_energy(self):
        for i in range(self.config["num_pe"]):
            tr = self.pt["dram_bank_{}".format(i)]
            self.energy_item["dram_read_energy"] += \
                tr["num_read"] * self.config["dram_bank_read_energy"]
            self.energy_item["dram_write_energy"] += \
                tr["num_write"] * self.config["dram_bank_write_energy"]
            self.energy_item["dram_refresh_energy"] += \
                tr["num_refresh"] * self.config["dram_bank_refresh_energy"]
            self.energy_item["dram_act_energy"] += \
                tr["num_act"] * self.config["dram_bank_act_energy"]
            self.energy_item["dram_pre_energy"] += \
                tr["num_pre"] * self.config["dram_bank_pre_energy"]
        self.hw.energy_item["dram_read_energy"] += \
            self.energy_item["dram_read_energy"]
        self.hw.energy_item["dram_write_energy"] += \
            self.energy_item["dram_write_energy"]
        self.hw.energy_item["dram_refresh_energy"] += \
            self.energy_item["dram_refresh_energy"]
        self.hw.energy_item["dram_act_energy"] += \
            self.energy_item["dram_act_energy"]
        self.hw.energy_item["dram_pre_energy"] += \
            self.energy_item["dram_pre_energy"]

    def get_energy_metrics(self):
        """Gets a dictionary of energy metrics. """
        # Collect the energy metrics of this hardware module.
        self._update_nb_operand_collector_energy()
        self._update_nb_alu_energy()
        self._update_lsu_extension_energy()
        self._update_nb_reg_file_energy()
        self._update_dram_energy()
        energy_metrics = self.energy_item
        energy_metrics["total_energy"] = 0
        for key in energy_metrics.keys():
            energy_metrics["total_energy"] += energy_metrics[key]
        
        return {"pg_{}".format(self.pg_id): energy_metrics}
